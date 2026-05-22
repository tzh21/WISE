#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate WISE_Verified images with WMDEC (multi-GPU data parallel).

Compared with RISEBench launcher, this script keeps most runtime config in Python
and supports ``--gpus`` for one process per GPU.

Example:
    python scripts/generate_wmdec_wise.py --gpus 0 1 2 3
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from PIL import Image

_WISE_ROOT = Path(__file__).resolve().parents[1]


def resolve_wmdec_checkpoint(checkpoint: str | None, checkpoint_dir: str) -> Path:
    if checkpoint:
        ckpt = Path(checkpoint).expanduser().resolve()
        if not ckpt.is_file():
            raise FileNotFoundError(f"wmdec checkpoint not found: {ckpt}")
        return ckpt

    ckpt_dir = Path(checkpoint_dir).expanduser().resolve()
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"wmdec checkpoint dir not found: {ckpt_dir}")

    candidates = sorted(ckpt_dir.glob("step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no step_*.pt checkpoint found under {ckpt_dir}")

    def sort_key(path: Path) -> int:
        name = path.stem
        if name.startswith("step_"):
            suffix = name[len("step_") :]
            if suffix.isdigit():
                return int(suffix)
        return -1

    return sorted(candidates, key=sort_key)[-1]


def resolve_base_policy_checkpoint(checkpoint_path: str) -> Path:
    ckpt = Path(checkpoint_path).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"base policy checkpoint path not found: {ckpt}")

    if ckpt.name == "pretrained_model":
        model_file = ckpt / "model.safetensors"
        if not model_file.is_file():
            raise FileNotFoundError(f"base policy checkpoint not ready: {model_file}")
        return ckpt.parent

    model_file = ckpt / "pretrained_model" / "model.safetensors"
    if model_file.is_file():
        return ckpt

    raise FileNotFoundError(
        f"base policy checkpoint not ready: expected either {ckpt / 'model.safetensors'} "
        f"or {ckpt / 'pretrained_model' / 'model.safetensors'}"
    )


def resolve_base_config(base_config: str | None, pipeline_root: Path) -> Path:
    if base_config:
        cfg = Path(base_config).expanduser().resolve()
        if cfg.is_file():
            return cfg
        print(f"[WARN] base config not found, fallback to auto discovery: {cfg}")

    outputs_dir = pipeline_root / "outputs"
    if not outputs_dir.is_dir():
        raise FileNotFoundError(
            f"cannot auto-discover base config: outputs dir not found at {outputs_dir}"
        )

    candidates = [p for p in outputs_dir.glob("*/hydra/.hydra/config.yaml") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"cannot auto-discover base config under {outputs_dir}; "
            "please pass --base-config explicitly"
        )

    preferred_keywords = ["decoder", "prev_next", "sd21", "lora", "r32"]

    def score(path: Path) -> tuple[int, float]:
        name = path.as_posix().lower()
        kw_score = sum(1 for kw in preferred_keywords if kw in name)
        return kw_score, path.stat().st_mtime

    best = sorted(candidates, key=score)[-1]
    print(f"[INFO] auto-selected base config: {best}")
    return best


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate WISE images with WMDEC (one process per GPU via --gpus)"
    )
    p.add_argument("--pipeline-root", type=str, default="/share/project/xsw/FS_WM")
    p.add_argument("--base-config", type=str, default=None)
    p.add_argument(
        "--base-policy-checkpoint",
        type=str,
        default="/share/project/mycao/FS_WM/outputs/0513_vla16_vlm4_24node_full/checkpoints/012000/pretrained_model/",
    )
    p.add_argument(
        "--wmdec-checkpoint-dir",
        type=str,
        default="/share/project/xsw/FS_WM/pretrained_decoders/wmdec_mlp_sd21_lora_r32_step_18378",
    )
    p.add_argument("--wmdec-checkpoint", type=str, default=None)
    p.add_argument(
        "--sd21-model",
        type=str,
        default="/share/project/congsheng/model/hub/models--Manojb--stable-diffusion-2-1-base/snapshots/0094d483a120f3f33dafbd187ea4aa60d10de75c",
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
        help="Output dir for generated PNGs, default: WISE/local/generated_images/wmdec",
    )
    p.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help="GPU indices, one WMDEC inferencer process per GPU (default: all visible GPUs)",
    )
    p.add_argument(
        "--direction",
        type=str,
        default="next",
        choices=["prev", "next"],
    )
    p.add_argument(
        "--event-mode",
        type=str,
        default="subtask",
        choices=["auto", "subtask", "atomic"],
    )
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--seed-base", type=int, default=42)
    p.add_argument("--view-index", type=int, default=0)
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument(
        "--image-transform",
        type=str,
        default="center_crop",
        choices=["resize", "center_crop"],
    )
    p.add_argument("--crop-size", type=int, default=448)
    p.add_argument("--decoder-context-tail-tokens", type=int, default=64)
    p.add_argument("--instruction-max-tokens", type=int, default=None)
    p.add_argument("--raw-prompt", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--category", type=str, default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip IDs with existing PNG; set --no-skip-existing to force re-run",
    )
    p.add_argument(
        "--init-image",
        type=Path,
        default=None,
        help="Optional init image path for all prompts; if unset, use a blank RGB image",
    )
    p.add_argument(
        "--save-metadata",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save per-sample metadata JSON under <output-dir>/metadata",
    )
    return p.parse_args()


def resolve_output_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    return _WISE_ROOT / "local" / "generated_images" / "wmdec"


def load_prompts(path: Path, category: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    out: list[dict[str, Any]] = []
    for item in data:
        if category:
            cat = str(item.get("category", ""))
            if cat != category:
                continue
        out.append(item)

    if limit is not None:
        out = out[:limit]
    return out


def split_round_robin(items: list[dict[str, Any]], n: int) -> list[list[dict[str, Any]]]:
    buckets: list[list[dict[str, Any]]] = [[] for _ in range(n)]
    for i, item in enumerate(items):
        buckets[i % n].append(item)
    return buckets


def _worker_put_results(
    gpu_id: int,
    shard: list[dict[str, Any]],
    worker_cfg: dict[str, Any],
    result_queue: mp.Queue,
) -> None:
    for payload in _generate_shard(gpu_id, shard, worker_cfg):
        result_queue.put(payload)


def _generate_shard(
    gpu_id: int,
    shard: list[dict[str, Any]],
    worker_cfg: dict[str, Any],
) -> Iterator[tuple[int, str | None]]:
    if not shard:
        return

    pipeline_root = Path(worker_cfg["pipeline_root"]).resolve()
    if str(pipeline_root) not in sys.path:
        sys.path.insert(0, str(pipeline_root))
    from tools.infer_wmdec_mlp_decoder import WMDECMLPImageInferencer

    base_config = worker_cfg["base_config"]
    base_policy_ckpt = worker_cfg["base_policy_ckpt"]
    wmdec_ckpt = worker_cfg["wmdec_ckpt"]
    output_dir = Path(worker_cfg["output_dir"])
    metadata_dir = Path(worker_cfg["metadata_dir"])
    save_metadata = bool(worker_cfg["save_metadata"])
    init_image_path = worker_cfg.get("init_image_path")
    use_blank = init_image_path is None
    seed_base = int(worker_cfg["seed_base"])

    device = f"cuda:{gpu_id}"
    infer_workdir = output_dir / f"wmdec_runtime_gpu{gpu_id}"
    infer_workdir.mkdir(parents=True, exist_ok=True)

    inferencer = WMDECMLPImageInferencer(
        base_config=base_config,
        base_policy_checkpoint=base_policy_ckpt,
        wmdec_checkpoint=wmdec_ckpt,
        sd21_model=worker_cfg["sd21_model"],
        output_dir=str(infer_workdir),
        image_size=int(worker_cfg["image_size"]),
        image_transform=worker_cfg["image_transform"],
        crop_size=int(worker_cfg["crop_size"]),
        decoder_context_tail_tokens=int(worker_cfg["decoder_context_tail_tokens"]),
        device=device,
        seed=seed_base,
        use_amp=not bool(worker_cfg["no_amp"]),
        instruction_max_tokens=worker_cfg["instruction_max_tokens"],
    )

    init_image: Image.Image | None = None
    if not use_blank:
        init_image = Image.open(init_image_path)
        if init_image.mode != "RGB":
            init_image = init_image.convert("RGB")

    for item in shard:
        pid = int(item["prompt_id"])
        prompt_text = str(item["Prompt"])
        output_png = output_dir / f"{pid}.png"
        if output_png.exists() and not bool(worker_cfg["overwrite"]):
            yield pid, None
            continue

        try:
            if use_blank:
                image = Image.new(
                    "RGB",
                    (int(worker_cfg["image_size"]), int(worker_cfg["image_size"])),
                    color=(255, 255, 255),
                )
            else:
                image = init_image.copy() if init_image is not None else Image.new(
                    "RGB",
                    (int(worker_cfg["image_size"]), int(worker_cfg["image_size"])),
                    color=(255, 255, 255),
                )

            seed = seed_base + pid
            generated, metadata = inferencer.infer(
                image=image,
                text=prompt_text,
                direction=worker_cfg["direction"],
                event_mode=worker_cfg["event_mode"],
                sample_steps=int(worker_cfg["sample_steps"]),
                seed=seed,
                view_index=int(worker_cfg["view_index"]),
                raw_prompt=bool(worker_cfg["raw_prompt"]),
            )
            generated.save(output_png)

            if save_metadata:
                metadata_dir.mkdir(parents=True, exist_ok=True)
                metadata.update(
                    {
                        "prompt_id": pid,
                        "prompt": prompt_text,
                        "category": item.get("category"),
                        "output_image": str(output_png),
                        "base_config": base_config,
                        "base_policy_checkpoint": base_policy_ckpt,
                        "wmdec_checkpoint": wmdec_ckpt,
                        "device": device,
                        "seed": seed,
                    }
                )
                meta_file = metadata_dir / f"{pid}.json"
                meta_file.write_text(
                    json.dumps(metadata, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            yield pid, None
        except Exception as e:
            yield pid, str(e)


def main() -> None:
    args = parse_args()

    try:
        from tqdm import tqdm
    except ImportError as e:
        raise SystemExit(
            "Please install tqdm (e.g. `pip install tqdm` or run in your uv/conda env)."
        ) from e

    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for WMDEC generation.")
    n_cuda = torch.cuda.device_count()

    if args.gpus is None:
        gpu_ids = list(range(n_cuda))
    else:
        gpu_ids = list(args.gpus)
        if not gpu_ids:
            raise SystemExit("At least one GPU index is required (--gpus).")
        if any(g < 0 for g in gpu_ids):
            raise SystemExit("GPU indices in --gpus must be non-negative.")
        if len(set(gpu_ids)) != len(gpu_ids):
            raise SystemExit("GPU indices in --gpus must be unique.")

    for g in gpu_ids:
        if g >= n_cuda:
            raise SystemExit(
                f"Requested GPU {g}, but only {n_cuda} CUDA device(s) are visible."
            )

    pipeline_root = Path(args.pipeline_root).expanduser().resolve()
    if not pipeline_root.exists():
        raise FileNotFoundError(f"pipeline root not found: {pipeline_root}")

    merge_path = args.merge_json.expanduser().resolve()
    if not merge_path.is_file():
        raise FileNotFoundError(f"merge json not found: {merge_path}")

    output_dir = resolve_output_dir(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    init_image_path: Path | None = None
    if args.init_image is not None:
        init_image_path = args.init_image.expanduser().resolve()
        if not init_image_path.is_file():
            raise FileNotFoundError(f"init image not found: {init_image_path}")

    base_config = resolve_base_config(args.base_config, pipeline_root)
    base_policy_ckpt = resolve_base_policy_checkpoint(args.base_policy_checkpoint)
    wmdec_ckpt = resolve_wmdec_checkpoint(args.wmdec_checkpoint, args.wmdec_checkpoint_dir)

    samples = load_prompts(merge_path, category=args.category, limit=args.limit)
    if args.skip_existing:
        samples = [s for s in samples if not (output_dir / f"{int(s['prompt_id'])}.png").is_file()]

    print(f"[INFO] pipeline_root={pipeline_root}")
    print(f"[INFO] base_config={base_config}")
    print(f"[INFO] base_policy_checkpoint={base_policy_ckpt}")
    print(f"[INFO] wmdec_checkpoint={wmdec_ckpt}")
    print(f"[INFO] merge_json={merge_path}")
    print(f"[INFO] output_dir={output_dir}")
    print(
        f"[INFO] total_prompts={len(samples)}, gpus={gpu_ids}, "
        f"skip_existing={args.skip_existing}, overwrite={args.overwrite}"
    )
    if not samples:
        print("[DONE] Nothing to generate.")
        return

    shards = split_round_robin(samples, len(gpu_ids))
    for gid, shard in zip(gpu_ids, shards):
        print(f"[INFO] GPU {gid}: {len(shard)} prompt(s)")

    worker_cfg: dict[str, Any] = {
        "pipeline_root": str(pipeline_root),
        "base_config": str(base_config),
        "base_policy_ckpt": str(base_policy_ckpt),
        "wmdec_ckpt": str(wmdec_ckpt),
        "sd21_model": args.sd21_model,
        "output_dir": str(output_dir),
        "metadata_dir": str(output_dir / "metadata"),
        "save_metadata": args.save_metadata,
        "init_image_path": str(init_image_path) if init_image_path else None,
        "direction": args.direction,
        "event_mode": args.event_mode,
        "sample_steps": args.sample_steps,
        "seed_base": args.seed_base,
        "view_index": args.view_index,
        "image_size": args.image_size,
        "image_transform": args.image_transform,
        "crop_size": args.crop_size,
        "decoder_context_tail_tokens": args.decoder_context_tail_tokens,
        "instruction_max_tokens": args.instruction_max_tokens,
        "raw_prompt": args.raw_prompt,
        "no_amp": args.no_amp,
        "overwrite": args.overwrite,
    }

    failures: list[int] = []
    bar_desc = "WMDEC WISE"

    if len(gpu_ids) == 1:
        with tqdm(total=len(samples), desc=bar_desc, unit="img") as pbar:
            for pid, err in _generate_shard(gpu_ids[0], shards[0], worker_cfg):
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
                args=(gid, shard, worker_cfg, result_queue),
            )
            proc.start()
            processes.append(proc)

        with tqdm(total=len(samples), desc=bar_desc, unit="img") as pbar:
            for _ in range(len(samples)):
                pid, err = result_queue.get()
                pbar.update(1)
                if err is not None:
                    failures.append(pid)
                    tqdm.write(f"[ERR] prompt_id={pid}: {err}")

        for proc in processes:
            proc.join()
            if proc.exitcode != 0:
                print(f"[WARN] worker pid={proc.pid} exited with code {proc.exitcode}")

    if failures:
        print(f"[WARN] failed count={len(failures)}; ids (first 50): {failures[:50]}")
    print("[DONE]")


if __name__ == "__main__":
    main()
