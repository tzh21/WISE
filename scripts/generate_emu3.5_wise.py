#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate all WISE_Verified benchmark images with Emu3.5 (text-to-image).

Writes ``{prompt_id}.png`` (1–1000) into an output directory chosen by ``--emu-variant``
(unless ``--output-dir`` is set), so Emu3.5 and Emu3.5-Image runs do not overwrite each
other — mirroring ``generate_emu3_wise.py`` pattern.

**Variants (``--emu-variant``)**

- ``image``: **Emu3.5-Image** specialist T2I checkpoint. Same recipe as
  ``configs/example_config_t2i.py``: higher visual CFG (default 5 in that config),
  ``image_area`` / ``max_new_tokens`` tuned for image generation. Default hub:
  ``EMU35_IMAGE_HUB`` / ``EMU3P5_IMAGE_HUB`` or ``.../Emu3.5-Image``.
- ``base``: **Emu3.5** full (interleaved-capable) checkpoint used in **plain T2I**
  mode — still loads ``example_config_t2i.py`` for prompt template / aspect presets, but
  swaps in the base weights and applies upstream-recommended T2I defaults (visual CFG 2,
  broader ``image_top_k``, larger ``image_area`` / token budget as in ``configs/config.py``).

Both share the same inference pipeline as ``Emu3.5/inference.py``. ``--gpus`` lists the
**first physical GPU index** of each worker; each worker owns ``--gpu-per-worker``
**consecutive** GPUs (default 1 = one full model per worker on that GPU). With
``--gpu-per-worker`` > 1 the causal LM uses HuggingFace ``device_map="auto"`` across
that worker's visible devices. Use multiprocessing ``spawn`` whenever there is more than
one worker or more than one GPU per worker. Examples::

    cd /share/project/tzh/Emu3.5
    # Four workers, one GPU each: bases 0,1,2,3
    uv run python .../generate_emu3.5_wise.py --emu-variant image --gpus 0 1 2 3
    # One worker using physical GPU 0 and 1 for a single sharded model
    uv run python .../generate_emu3.5_wise.py --emu-variant base --gpus 0 --gpu-per-worker 2
    # Two workers: GPUs 0-1 and 2-3
    uv run python .../generate_emu3.5_wise.py --emu-variant image --gpus 0 2 --gpu-per-worker 2

Or set ``EMU35_REPO`` if the Emu3.5 tree lives elsewhere. Override weights with
``--model-path``, VQ with ``--vq-path`` / ``EMU35_VQ_HUB``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import random
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType
from typing import Any

_WISE_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_EMU35_IMAGE_HUB = "/share/project/tzh/models/Emu3.5-Image"
_DEFAULT_EMU35_BASE_HUB = "/share/project/tzh/models/Emu3.5"


def _default_emu35_repo() -> Path:
    return _WISE_ROOT.parent / "Emu3.5"


def resolve_model_hub(variant: str, explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    if variant == "image":
        return os.environ.get(
            "EMU35_IMAGE_HUB",
            os.environ.get("EMU3P5_IMAGE_HUB", _DEFAULT_EMU35_IMAGE_HUB),
        )
    return os.environ.get("EMU35_HUB", os.environ.get("EMU3P5_HUB", _DEFAULT_EMU35_BASE_HUB))


def resolve_vq_hub(explicit: str | None) -> str | None:
    if explicit and explicit.strip():
        return explicit.strip()
    env = os.environ.get("EMU35_VQ_HUB", os.environ.get("EMU3P5_VQ_HUB", ""))
    return env.strip() or None


def resolve_output_dir(variant: str, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    sub = "emu3p5_image" if variant == "image" else "emu3p5"
    return _WISE_ROOT / "local" / "generated_images" / sub


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate WISE images with Emu3.5 (Emu3.5/inference-style t2i)"
    )
    p.add_argument(
        "--emu35-repo",
        type=Path,
        default=Path(os.environ.get("EMU35_REPO", str(_default_emu35_repo()))),
        help="Root of the Emu3.5 repo (for imports and config paths)",
    )
    p.add_argument(
        "--emu-variant",
        choices=("image", "base"),
        default="image",
        help=(
            'Checkpoint family: "image" = Emu3.5-Image (specialist T2I); '
            '"base" = Emu3.5 full model in T2I-style decoding (different CFG / sampling). '
            "Default output subdirectory and model hub follow the variant unless overridden."
        ),
    )
    p.add_argument(
        "--cfg",
        type=Path,
        default=None,
        help=(
            "Python config module path (default: <emu35-repo>/configs/example_config_t2i.py "
            "for both variants; base still uses this file but applies Emu3.5 T2I defaults)"
        ),
    )
    p.add_argument(
        "--model-path",
        default=None,
        help=(
            "HF/local causal LM checkpoint (overrides variant default hub). Env: "
            'EMU35_IMAGE_HUB / EMU3P5_IMAGE_HUB for --emu-variant image; '
            "EMU35_HUB / EMU3P5_HUB for base"
        ),
    )
    p.add_argument(
        "--vq-path",
        default=None,
        help="Override cfg vq_path (env: EMU35_VQ_HUB, EMU3P5_VQ_HUB)",
    )
    p.add_argument(
        "--tokenizer-path",
        default=None,
        help="Override cfg tokenizer_path (dir with emu3 IBQ tokenizer)",
    )
    p.add_argument(
        "--merge-json",
        type=Path,
        default=_WISE_ROOT / "data_verified" / "merge.json",
        help="WISE_Verified merged prompts JSON",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory for 1.png … 1000.png. Default depends on --emu-variant "
            "(emu3p5_image vs emu3p5)"
        ),
    )
    p.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0],
        metavar="N",
        help=(
            "Per worker: index of the first physical GPU in a consecutive block of length "
            "--gpu-per-worker (default block length 1)"
        ),
    )
    p.add_argument(
        "--gpu-per-worker",
        type=int,
        default=1,
        metavar="K",
        help=(
            "Each worker uses K consecutive GPUs: first index is from --gpus, block is "
            "base..base+K-1. K>1 loads the causal LM with device_map=auto on those GPUs."
        ),
    )
    p.add_argument(
        "--aspect-ratio",
        default="1:1",
        help=(
            'Aspect ratio key from Emu3.5 t2i config (e.g. "1:1", "16:9", "default", "auto")'
        ),
    )
    p.add_argument(
        "--classifier-free-guidance",
        type=float,
        default=None,
        help="Override cfg classifier_free_guidance (CFG scale for visual tokens)",
    )
    p.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="Per-image: random seed uses seed_base + prompt_id (reproducible)",
    )
    p.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip prompt IDs whose PNG already exists (default: true)",
    )
    return p.parse_args()


def load_cfg_module(path: Path, *, repo_root: Path | None = None) -> ModuleType:
    """Load a config ``.py`` file. Emu3.5 configs import ``src.*``; repo root must be on ``sys.path``."""
    path = path.resolve()
    repo_s = ""
    inserted = False
    if repo_root is not None:
        repo_s = str(repo_root.resolve())
        if repo_s not in sys.path:
            sys.path.insert(0, repo_s)
            inserted = True
    try:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load config from {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if inserted:
            try:
                sys.path.remove(repo_s)
            except ValueError:
                pass


def load_prompts(path: Path) -> dict[int, str]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    out: dict[int, str] = {}
    for item in data:
        out[int(item["prompt_id"])] = item["Prompt"]
    return out


def _split_round_robin(ids: list[int], n: int) -> list[list[int]]:
    buckets: list[list[int]] = [[] for _ in range(n)]
    for i, pid in enumerate(ids):
        buckets[i % n].append(pid)
    return buckets


def _validate_worker_gpu_layout(
    worker_bases: list[int], gpu_per_worker: int, n_cuda: int
) -> None:
    if gpu_per_worker < 1:
        raise SystemExit("--gpu-per-worker must be >= 1.")
    seen: set[int] = set()
    for base in worker_bases:
        if base < 0:
            raise SystemExit("GPU indices in --gpus must be non-negative.")
        if base + gpu_per_worker > n_cuda:
            raise SystemExit(
                f"Worker starting at GPU {base} requires devices "
                f"{base}..{base + gpu_per_worker - 1}, but only {n_cuda} CUDA device(s) exist."
            )
        for g in range(base, base + gpu_per_worker):
            if g in seen:
                raise SystemExit(
                    f"Overlapping GPU assignment: device {g} is used by more than one worker."
                )
            seen.add(g)


def _worker_put_results(
    gpu_base: int,
    prompt_ids: list[int],
    prompts: dict[int, str],
    cfg: dict[str, Any],
    result_queue: mp.Queue,
) -> None:
    for item in _generate_shard(gpu_base, prompt_ids, prompts, cfg):
        result_queue.put(item)


def _resolve_path(repo: Path, p: str | Path) -> str:
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((repo / pp).resolve())


def _target_hw(cfg_mod: ModuleType, aspect_ratio: str) -> tuple[int | None, int | None]:
    get_ts = getattr(cfg_mod, "get_target_size", None)
    if callable(get_ts):
        h, w = get_ts(aspect_ratio)
        return h, w
    ar = getattr(cfg_mod, "aspect_ratios", None)
    if isinstance(ar, dict) and aspect_ratio in ar:
        val = ar[aspect_ratio]
        if val is None:
            return None, None
        h, w = map(int, str(val).split("*"))
        return h, w
    raise ValueError(
        f"Unknown aspect_ratio={aspect_ratio!r}; not in config aspect_ratios and "
        "get_target_size missing"
    )


def _apply_variant_hparams(cfg_mod: ModuleType, variant: str) -> None:
    """Tune loaded config for ``base`` (Emu3.5) vs ``image`` (Emu3.5-Image).

    Both load ``example_config_t2i.py`` for templates / aspect-ratio table; upstream
    ``configs/config.py`` vs ``example_config_t2i.py`` differ in CFG, area, and sampling.
    """
    if variant != "base":
        return
    cfg_mod.classifier_free_guidance = 2.0
    cfg_mod.image_area = 518400
    nt = 32768
    if hasattr(cfg_mod, "max_new_tokens"):
        cfg_mod.max_new_tokens = nt
    if hasattr(cfg_mod, "sampling_params"):
        cfg_mod.sampling_params["image_top_k"] = 10240
        cfg_mod.sampling_params["max_new_tokens"] = nt


def _generate_shard(
    gpu_base: int,
    prompt_ids: list[int],
    prompts: dict[int, str],
    run_cfg: dict[str, Any],
) -> Iterator[tuple[int, str | None]]:
    """Yield ``(prompt_id, error)`` per image; ``error`` is None on success."""
    if not prompt_ids:
        return

    gpu_per_worker = int(run_cfg.get("gpu_per_worker", 1))
    if gpu_per_worker > 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
            str(gpu_base + i) for i in range(gpu_per_worker)
        )

    import torch
    from PIL import Image

    emu35_repo = Path(run_cfg["emu35_repo"]).resolve()
    cfg_path = Path(run_cfg["cfg_path"]).resolve()

    os.chdir(emu35_repo)
    repo_s = str(emu35_repo)
    if repo_s not in sys.path:
        sys.path.insert(0, repo_s)

    from src.utils.generation_utils import generate, multimodal_decode
    from src.utils.model_utils import build_emu3p5

    cfg_mod = load_cfg_module(cfg_path, repo_root=emu35_repo)
    variant = run_cfg.get("variant", "image")
    _apply_variant_hparams(cfg_mod, variant)

    model_path = run_cfg.get("model_path") or cfg_mod.model_path
    vq_path = run_cfg.get("vq_path") or cfg_mod.vq_path
    tokenizer_path = run_cfg.get("tokenizer_path") or cfg_mod.tokenizer_path

    model_path = _resolve_path(emu35_repo, model_path)
    vq_path = _resolve_path(emu35_repo, vq_path)
    tokenizer_path = _resolve_path(emu35_repo, tokenizer_path)

    vq_type = getattr(cfg_mod, "vq_type", "ibq")

    if run_cfg.get("classifier_free_guidance") is not None:
        cfg_mod.classifier_free_guidance = float(run_cfg["classifier_free_guidance"])

    th, tw = _target_hw(cfg_mod, run_cfg["aspect_ratio"])
    cfg_mod.target_height = th
    cfg_mod.target_width = tw

    cfg_mod.streaming = False

    if gpu_per_worker > 1:
        torch.cuda.set_device(0)
        model_device: int | str = "auto"
        vq_device = "cuda:0"
    else:
        torch.cuda.set_device(gpu_base)
        model_device = gpu_base
        vq_device = f"cuda:{gpu_base}"

    model, tokenizer, vq_model = build_emu3p5(
        model_path,
        tokenizer_path,
        vq_path,
        vq_type=vq_type,
        model_device=model_device,
        vq_device=vq_device,
        **getattr(cfg_mod, "diffusion_decoder_kwargs", {}),
    )

    cfg_mod.special_token_ids = {}
    for k, v in cfg_mod.special_tokens.items():
        cfg_mod.special_token_ids[k] = tokenizer.encode(v)[0]

    full_unc_ids = None
    if hasattr(cfg_mod, "img_unc_prompt"):
        full_unc_ids = tokenizer.encode(
            cfg_mod.img_unc_prompt, return_tensors="pt", add_special_tokens=False
        ).to(model.device)

    template = cfg_mod.template
    unc_prompt_base = cfg_mod.unc_prompt
    seed_base = int(run_cfg["seed_base"])
    out_dir = Path(run_cfg["output_dir"])

    def _first_image(question: str, pid: int) -> Image.Image | None:
        random.seed(seed_base + pid)
        torch.manual_seed((seed_base + pid) % (2**32))

        prompt = template.format(question=question)
        unc_prompt = unc_prompt_base

        input_ids = tokenizer.encode(
            prompt, return_tensors="pt", add_special_tokens=False
        ).to(model.device)
        if input_ids[0, 0] != cfg_mod.special_token_ids["BOS"]:
            bos = torch.tensor(
                [[cfg_mod.special_token_ids["BOS"]]],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            input_ids = torch.cat([bos, input_ids], dim=1)

        unconditional_ids = tokenizer.encode(
            unc_prompt, return_tensors="pt", add_special_tokens=False
        ).to(model.device)

        force_same_image_size = True

        for result_tokens in generate(
            cfg_mod,
            model,
            tokenizer,
            input_ids,
            unconditional_ids,
            full_unc_ids,
            force_same_image_size,
        ):
            result = tokenizer.decode(result_tokens, skip_special_tokens=False)
            mm_out = multimodal_decode(result, tokenizer, vq_model)
            for kind, payload in mm_out:
                if kind == "image" and isinstance(payload, Image.Image):
                    return payload
        return None

    for pid in prompt_ids:
        try:
            pil = _first_image(prompts[pid], pid)
            if pil is None:
                yield pid, "generation produced no decoded image"
            else:
                pil.save(out_dir / f"{pid}.png")
                yield pid, None
        except Exception as e:
            yield pid, str(e)
        finally:
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    emu35_repo = args.emu35_repo.resolve()
    cfg_path = args.cfg
    if cfg_path is None:
        cfg_path = emu35_repo / "configs" / "example_config_t2i.py"
    cfg_path = cfg_path.resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")

    variant = args.emu_variant
    resolved_model = resolve_model_hub(variant, args.model_path)
    vq_override = resolve_vq_hub(args.vq_path)
    out_dir = resolve_output_dir(variant, args.output_dir)
    worker_gpu_bases = list(args.gpus)
    gpu_per_worker = int(args.gpu_per_worker)
    if len(worker_gpu_bases) < 1:
        raise SystemExit("At least one GPU index is required (--gpus).")

    try:
        from tqdm import tqdm
    except ImportError as e:
        raise SystemExit(
            "Please install tqdm (e.g. `pip install tqdm` or use WISE's uv env)."
        ) from e

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Emu3.5 generation.")

    _validate_worker_gpu_layout(
        worker_gpu_bases, gpu_per_worker, torch.cuda.device_count()
    )

    merge_path = args.merge_json.resolve()
    if not merge_path.is_file():
        raise SystemExit(f"Missing prompts file: {merge_path}")

    # Validate aspect ratio early (before spawning). Config import runs setup_logger
    # with paths relative to cwd; use emu35_repo as cwd for that side effect.
    prev_cwd = os.getcwd()
    try:
        os.chdir(emu35_repo)
        probe = load_cfg_module(cfg_path, repo_root=emu35_repo)
        _target_hw(probe, args.aspect_ratio)
    except ValueError as e:
        raise SystemExit(str(e)) from e
    finally:
        os.chdir(prev_cwd)

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts_map = load_prompts(merge_path)
    ids_sorted = sorted(prompts_map.keys())
    expected = set(range(1, 1001))
    missing_ids = expected - set(prompts_map.keys())
    if missing_ids:
        print(
            f"[WARN] merge.json missing prompt_ids: {sorted(missing_ids)[:20]}… "
            f"({len(missing_ids)} total)"
        )

    to_run: list[int] = []
    for pid in ids_sorted:
        out_png = out_dir / f"{pid}.png"
        if args.skip_existing and out_png.is_file():
            continue
        to_run.append(pid)

    print(f"[INFO] emu_variant={variant}, model_path={resolved_model}")
    print(f"[INFO] emu35_repo={emu35_repo}")
    print(f"[INFO] cfg={cfg_path}")
    print(f"[INFO] Saving images under {out_dir}")
    print(
        f"[INFO] Total prompts: {len(ids_sorted)}, to generate: {len(to_run)}, "
        f"worker_gpu_bases={worker_gpu_bases}, gpu_per_worker={gpu_per_worker}, "
        f"aspect_ratio={args.aspect_ratio}, skip_existing={args.skip_existing}"
    )
    if not to_run:
        print("[DONE] Nothing to generate.")
        return

    cfg: dict[str, Any] = {
        "emu35_repo": str(emu35_repo),
        "cfg_path": str(cfg_path),
        "variant": variant,
        "model_path": resolved_model,
        "vq_path": vq_override,
        "tokenizer_path": args.tokenizer_path,
        "output_dir": str(out_dir),
        "aspect_ratio": args.aspect_ratio,
        "classifier_free_guidance": args.classifier_free_guidance,
        "seed_base": args.seed_base,
        "gpu_per_worker": gpu_per_worker,
    }

    n_workers = len(worker_gpu_bases)
    shards = _split_round_robin(to_run, n_workers)
    for base, shard in zip(worker_gpu_bases, shards):
        if gpu_per_worker > 1:
            dev_rng = f"{base}..{base + gpu_per_worker - 1}"
        else:
            dev_rng = str(base)
        print(f"[INFO] worker base GPU {base} (devices {dev_rng}): {len(shard)} image(s)")

    failures: list[int] = []
    bar_desc = "Emu3.5-Image WISE" if variant == "image" else "Emu3.5 WISE"

    # Child processes must set CUDA_VISIBLE_DEVICES before CUDA init when gpu_per_worker>1.
    use_spawn = n_workers > 1 or gpu_per_worker > 1

    if not use_spawn:
        with tqdm(total=len(to_run), desc=bar_desc, unit="img") as pbar:
            for pid, err in _generate_shard(
                worker_gpu_bases[0], shards[0], prompts_map, cfg
            ):
                pbar.update(1)
                if err is not None:
                    failures.append(pid)
                    tqdm.write(f"[ERR] prompt_id={pid}: {err}")
    else:
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        processes: list[mp.Process] = []
        for base, shard in zip(worker_gpu_bases, shards):
            if not shard:
                continue
            proc = ctx.Process(
                target=_worker_put_results,
                args=(base, shard, prompts_map, cfg, result_queue),
            )
            proc.start()
            processes.append(proc)

        with tqdm(total=len(to_run), desc=bar_desc, unit="img") as pbar:
            for _ in range(len(to_run)):
                pid, err = result_queue.get()
                pbar.update(1)
                if err is not None:
                    failures.append(pid)
                    tqdm.write(f"[ERR] prompt_id={pid}: {err}")

        for proc in processes:
            proc.join()
            if proc.exitcode != 0:
                print(f"[WARN] Worker pid={proc.pid} exited with code {proc.exitcode}")

    if failures:
        print(f"[WARN] Failed count={len(failures)}; ids (first 50): {failures[:50]}")
    print("[DONE]")


if __name__ == "__main__":
    main()
