#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate all WISE_Verified benchmark images with Qwen-Image (diffusers pipeline).

Writes ``{prompt_id}.png`` (1–1000) under ``WISE/local/generated_images/qwen_image`` by default
so evaluation can point ``IMAGE_DIR`` at that folder.

Use ``--gpus`` for one pipeline per GPU via separate processes (``spawn``). Example::

    uv run python .../generate_qwen_wise.py --gpus 0 1 2 3

Requires diffusers/torch consistent with ``pyproject.toml`` and a GPU with enough VRAM.

Default checkpoint: ``QWEN_IMAGE_HUB`` env or ``/share/project/tzh/models/Qwen-Image``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_WISE_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_QWEN_HUB = "/share/project/tzh/models/Qwen-Image"

# Same aspect presets as upstream Qwen-Image usage examples.
ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (1328, 1328),
    "16:9": (1664, 928),
    "9:16": (928, 1664),
    "4:3": (1472, 1140),
    "3:4": (1140, 1472),
    "3:2": (1584, 1056),
    "2:3": (1056, 1584),
}

_POSITIVE_SUFFIX = {
    "en": ", Ultra HD, 4K, cinematic composition.",
    "zh": ", 超清，4K，电影级构图.",
}

_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _default_positive_lang(prompt_text: str) -> str:
    return "zh" if _CJK_RE.search(prompt_text) else "en"


def resolve_model_hub(explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    return os.environ.get("QWEN_IMAGE_HUB", _DEFAULT_QWEN_HUB)


def resolve_output_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    return _WISE_ROOT / "local" / "generated_images" / "qwen_image"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate WISE images with Qwen-Image (diffusers)")
    p.add_argument(
        "--model-hub",
        default=None,
        help=f"HF or local pipeline path (default: env QWEN_IMAGE_HUB or {_DEFAULT_QWEN_HUB})",
    )
    p.add_argument(
        "--merge-json",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data_verified" / "merge.json",
        help="WISE_Verified merged prompts JSON",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for 1.png … 1000.png. Default: WISE/local/generated_images/qwen_image",
    )
    p.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0],
        metavar="N",
        help="GPU indices; one pipeline per GPU in separate processes (default: 0)",
    )
    p.add_argument(
        "--ratio",
        default="1:1",
        choices=sorted(ASPECT_RATIOS.keys()),
        help='Image aspect preset (matches Qwen-Image presets, default "1:1")',
    )
    p.add_argument(
        "--positive-suffix",
        choices=("auto", "none", "en", "zh"),
        default="auto",
        help='Append quality suffix: auto (CJK→zh else en), fixed en/zh, or none',
    )
    p.add_argument(
        "--negative-prompt",
        default=" ",
        help='Negative prompt (default: single space, as in Qwen-Image examples)',
    )
    p.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Diffusion inference steps",
    )
    p.add_argument(
        "--true-cfg-scale",
        type=float,
        default=4.0,
        help="true_cfg_scale passed to the pipeline (Qwen-Image)",
    )
    p.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="Base seed; per image uses seed_base + prompt_id for reproducibility",
    )
    p.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip prompt IDs whose PNG already exists (default: true)",
    )
    return p.parse_args()


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


def _worker_put_results(
    gpu_id: int,
    prompt_ids: list[int],
    prompts: dict[int, str],
    cfg: dict[str, Any],
    result_queue: mp.Queue,
) -> None:
    for item in _generate_shard(gpu_id, prompt_ids, prompts, cfg):
        result_queue.put(item)


def _positive_suffix_for(cfg_mode: str, prompt_text: str) -> str:
    if cfg_mode == "none":
        return ""
    if cfg_mode in ("en", "zh"):
        return _POSITIVE_SUFFIX[cfg_mode]
    return _POSITIVE_SUFFIX[_default_positive_lang(prompt_text)]


def _generate_shard(
    gpu_id: int,
    prompt_ids: list[int],
    prompts: dict[int, str],
    cfg: dict[str, Any],
) -> Iterator[tuple[int, str | None]]:
    """Yield ``(prompt_id, error)`` per image; ``error`` is None on success."""
    if not prompt_ids:
        return

    import torch
    from diffusers import DiffusionPipeline

    device = f"cuda:{gpu_id}"
    model_hub = cfg["model_hub"]
    ratio = cfg["ratio"]
    width, height = ASPECT_RATIOS[ratio]
    steps = int(cfg["num_inference_steps"])
    true_cfg_scale = float(cfg["true_cfg_scale"])
    seed_base = int(cfg["seed_base"])
    positive_mode = cfg["positive_suffix"]
    negative_prompt = cfg["negative_prompt"]
    out_dir = Path(cfg["output_dir"])

    torch_dtype = torch.bfloat16
    pipe = DiffusionPipeline.from_pretrained(model_hub, torch_dtype=torch_dtype)
    pipe = pipe.to(device)

    for pid in prompt_ids:
        out_png = out_dir / f"{pid}.png"
        prompt_text = prompts[pid]
        try:
            full_prompt = prompt_text + _positive_suffix_for(positive_mode, prompt_text)
            gen = torch.Generator(device=device).manual_seed((seed_base + pid) % (2**32))
            image = pipe(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                true_cfg_scale=true_cfg_scale,
                generator=gen,
            ).images[0]
            image.save(out_png)
            yield pid, None
        except Exception as e:
            yield pid, str(e)


def main() -> None:
    args = parse_args()
    model_hub = resolve_model_hub(args.model_hub)
    out_dir = resolve_output_dir(args.output_dir)
    gpu_ids = list(args.gpus)
    if len(gpu_ids) < 1:
        raise SystemExit("At least one GPU index is required (--gpus).")
    if any(g < 0 for g in gpu_ids):
        raise SystemExit("GPU indices must be non-negative.")

    try:
        from tqdm import tqdm
    except ImportError as e:
        raise SystemExit(
            "Please install tqdm (e.g. `pip install tqdm` or use WISE's uv env)."
        ) from e

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Qwen-Image generation.")

    merge_path = args.merge_json.resolve()
    if not merge_path.is_file():
        raise SystemExit(f"Missing prompts file: {merge_path}")

    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(merge_path)
    ids_sorted = sorted(prompts.keys())
    expected = set(range(1, 1001))
    missing_ids = expected - set(prompts.keys())
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

    print(f"[INFO] model_hub={model_hub}")
    print(f"[INFO] Saving images under {out_dir}")
    print(
        f"[INFO] Total prompts: {len(ids_sorted)}, to generate: {len(to_run)}, "
        f"gpus={gpu_ids}, skip_existing={args.skip_existing}"
    )
    if not to_run:
        print("[DONE] Nothing to generate.")
        return

    cfg: dict[str, Any] = {
        "model_hub": str(model_hub),
        "output_dir": str(out_dir),
        "ratio": args.ratio,
        "num_inference_steps": int(args.num_inference_steps),
        "true_cfg_scale": float(args.true_cfg_scale),
        "seed_base": int(args.seed_base),
        "positive_suffix": args.positive_suffix,
        "negative_prompt": args.negative_prompt,
    }

    n_workers = len(gpu_ids)
    shards = _split_round_robin(to_run, n_workers)
    for gid, shard in zip(gpu_ids, shards):
        print(f"[INFO] GPU {gid}: {len(shard)} image(s) in this process")

    failures: list[int] = []

    bar_desc = "Qwen-Image WISE"
    if n_workers == 1:
        with tqdm(total=len(to_run), desc=bar_desc, unit="img") as pbar:
            for pid, err in _generate_shard(gpu_ids[0], shards[0], prompts, cfg):
                pbar.update(1)
                if err is not None:
                    failures.append(pid)
                    tqdm.write(f"[ERR] prompt_id={pid}: {err}")
    else:
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        processes: list[mp.Process] = []
        for gid, shard in zip(gpu_ids, shards):
            if not shard:
                continue
            proc = ctx.Process(
                target=_worker_put_results,
                args=(gid, shard, prompts, cfg, result_queue),
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
