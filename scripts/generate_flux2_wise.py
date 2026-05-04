#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate all WISE_Verified benchmark images with FLUX.2-klein-4B (text-to-image).

Outputs ``{prompt_id}.png`` (1–1000) into the chosen directory for evaluation.

Default model path: ``/share/project/tzh/models/FLUX.2-klein-4B`` (override with
``--model`` or ``FLUX2_4B_HUB`` / ``FLUX2_HUB``).

Default output directory: ``WISE/local/generated_images/flux2_4b``.

Use ``--gpus`` to run one pipeline copy per GPU (multiprocessing, ``spawn``). Example::

    uv run python .../generate_flux2_wise.py --gpus 0 1 2 3
    uv run python .../generate_flux2_wise.py --gpus 0

Each GPU process batches prompts with ``--batch-size`` (passed as a list to the
pipeline for a single forward). Use ``1`` to disable micro-batching.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

_WISE_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_FLUX2_HUB = "/share/project/tzh/models/FLUX.2-klein-4B"


def resolve_model_hub(explicit: str | None) -> str:
    if explicit and explicit.strip():
        return explicit.strip()
    return os.environ.get(
        "FLUX2_4B_HUB", os.environ.get("FLUX2_HUB", _DEFAULT_FLUX2_HUB)
    )


def resolve_output_dir(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    return _WISE_ROOT / "local" / "generated_images" / "flux2_4b"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate WISE images with FLUX.2-klein-4B (diffusers)"
    )
    p.add_argument(
        "--model",
        default=None,
        help=(
            "HF/local FLUX.2-klein-4B checkpoint. Env: FLUX2_4B_HUB, FLUX2_HUB"
        ),
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
        help=(
            "Directory for 1.png … 1000.png. Default: WISE/local/generated_images/flux2_4b"
        ),
    )
    p.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0],
        metavar="N",
        help="GPU indices; one full pipeline per GPU via separate processes (default: 0)",
    )
    p.add_argument("--height", type=int, default=1024, help="Image height")
    p.add_argument("--width", type=int, default=1024, help="Image width")
    p.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="Guidance scale (Klein often uses 1.0)",
    )
    p.add_argument(
        "--num-inference-steps",
        type=int,
        default=4,
        help="Number of denoising steps",
    )
    p.add_argument(
        "--seed-base",
        type=int,
        default=0,
        help="Added to prompt_id for the per-image generator seed (reproducible ids)",
    )
    p.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip prompt IDs whose PNG already exists (default: true)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="B",
        help=(
            "Micro-batch size per GPU process (diffusers batch of prompts). "
            "Increase for throughput; lower if GPU OOM (default: 1)"
        ),
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
    from diffusers import Flux2KleinPipeline

    device = f"cuda:{gpu_id}"
    model_id = cfg["model"]
    out_dir = Path(cfg["output_dir"])
    height = int(cfg["height"])
    width = int(cfg["width"])
    guidance_scale = float(cfg["guidance_scale"])
    num_inference_steps = int(cfg["num_inference_steps"])
    seed_base = int(cfg["seed_base"])
    batch_size = max(1, int(cfg["batch_size"]))

    dtype = torch.bfloat16
    pipe = Flux2KleinPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    def _save_one(pid: int, image: Any) -> None:
        image.save(out_dir / f"{pid}.png")

    def _generate_one(pid: int) -> tuple[int, str | None]:
        try:
            gen = torch.Generator(device=device).manual_seed(seed_base + pid)
            result = pipe(
                prompt=prompts[pid],
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=gen,
            )
            _save_one(pid, result.images[0])
            return pid, None
        except Exception as e:
            return pid, str(e)

    i = 0
    while i < len(prompt_ids):
        chunk = prompt_ids[i : i + batch_size]
        i += len(chunk)

        if len(chunk) == 1:
            pid, err = _generate_one(chunk[0])
            yield pid, err
            continue

        texts = [prompts[pid] for pid in chunk]
        generators = [
            torch.Generator(device=device).manual_seed(seed_base + pid) for pid in chunk
        ]
        try:
            result = pipe(
                prompt=texts,
                height=height,
                width=width,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generators,
            )
            images = result.images
            if len(images) != len(chunk):
                raise RuntimeError(
                    f"expected {len(chunk)} images, got {len(images)} from pipeline"
                )
            for pid, im in zip(chunk, images):
                _save_one(pid, im)
                yield pid, None
        except Exception:
            for pid in chunk:
                yield _generate_one(pid)


def main() -> None:
    args = parse_args()
    model_hub = resolve_model_hub(args.model)
    out_dir = resolve_output_dir(args.output_dir)
    gpu_ids = list(args.gpus)
    if len(gpu_ids) < 1:
        raise SystemExit("At least one GPU index is required (--gpus).")
    if any(g < 0 for g in gpu_ids):
        raise SystemExit("GPU indices must be non-negative.")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1.")

    try:
        from tqdm import tqdm
    except ImportError as e:
        raise SystemExit(
            "Please install tqdm (e.g. `pip install tqdm` or use WISE's uv env)."
        ) from e

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

    print(f"[INFO] model={model_hub}")
    print(f"[INFO] Saving images under {out_dir}")
    print(
        f"[INFO] Total prompts: {len(ids_sorted)}, to generate: {len(to_run)}, "
        f"gpus={gpu_ids}, per-GPU batch_size={args.batch_size}, "
        f"skip_existing={args.skip_existing}"
    )
    if not to_run:
        print("[DONE] Nothing to generate.")
        return

    cfg: dict[str, Any] = {
        "model": str(model_hub),
        "output_dir": str(out_dir),
        "height": args.height,
        "width": args.width,
        "guidance_scale": args.guidance_scale,
        "num_inference_steps": args.num_inference_steps,
        "seed_base": args.seed_base,
        "batch_size": args.batch_size,
    }

    n_workers = len(gpu_ids)
    shards = _split_round_robin(to_run, n_workers)
    for gid, shard in zip(gpu_ids, shards):
        print(f"[INFO] GPU {gid}: {len(shard)} image(s) in this process")

    failures: list[int] = []

    bar_desc = "FLUX.2-klein-4B WISE"
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
