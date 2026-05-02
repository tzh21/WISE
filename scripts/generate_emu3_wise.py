#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate all WISE_Verified benchmark images with Emu3-Gen.

Outputs `{prompt_id}.png` (1–1000) into the chosen directory so you can run
`eval_qwen.sh` with IMAGE_DIR pointing at that folder.

Use ``--gpus`` to run one model copy per GPU (multiprocessing, ``spawn``). Example::

    uv run python .../generate_emu3_wise.py --gpus 0 1 2 3

Run from an environment that has Emu3 dependencies installed (same as Emu3's
image_generation.py), for example::

    cd /share/project/tzh/Emu3
    uv run python /share/project/tzh/WISE/scripts/generate_emu3_wise.py

Or set EMU3_REPO if the Emu3 source tree lives elsewhere.
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


def _default_emu3_repo() -> Path:
    wise_root = Path(__file__).resolve().parents[1]
    return wise_root.parent / "Emu3"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate WISE images with Emu3-Gen")
    p.add_argument(
        "--emu-hub",
        default=os.environ.get("EMU_HUB", "/share/project/tzh/models/Emu3-Gen"),
        help="Emu3-Gen HF/local checkpoint path",
    )
    p.add_argument(
        "--vq-hub",
        default=os.environ.get("VQ_HUB", "/share/project/shirc/model/Emu3-VisionTokenizer"),
        help="Emu3 vision tokenizer checkpoint path",
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
        default=Path("/share/project/tzh/WISE/local/generated_images/emu3"),
        help="Directory for 1.png … 1000.png",
    )
    p.add_argument(
        "--emu3-repo",
        type=Path,
        default=Path(os.environ.get("EMU3_REPO", str(_default_emu3_repo()))),
        help="Root of the Emu3 repo (for importing emu3.*)",
    )
    p.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0],
        metavar="N",
        help="GPU indices; one full model per GPU via separate processes (default: 0)",
    )
    p.add_argument(
        "--ratio",
        default="1:1",
        help='Aspect ratio string passed to Emu3Processor (e.g. "1:1", "16:9")',
    )
    p.add_argument(
        "--classifier-free-guidance",
        type=float,
        default=3.0,
        help="Classifier-free guidance scale",
    )
    p.add_argument(
        "--attn-implementation",
        default="flash_attention_2",
        help='Attention backend (e.g. "flash_attention_2" or "sdpa")',
    )
    p.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip prompt IDs whose PNG already exists (default: true)",
    )
    return p.parse_args()


POSITIVE_PROMPT = " masterpiece, film grained, best quality."
NEGATIVE_PROMPT = (
    "lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, "
    "fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, "
    "signature, watermark, username, blurry."
)


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
    emu3_repo = Path(cfg["emu3_repo"])
    if str(emu3_repo.resolve()) not in sys.path:
        sys.path.insert(0, str(emu3_repo.resolve()))

    import torch
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel, AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import (
        LogitsProcessorList,
        PrefixConstrainedLogitsProcessor,
        UnbatchedClassifierFreeGuidanceLogitsProcessor,
    )
    from transformers.generation.configuration_utils import GenerationConfig

    from emu3.mllm.processing_emu3 import Emu3Processor

    device = f"cuda:{gpu_id}"
    emu_hub = cfg["emu_hub"]
    vq_hub = cfg["vq_hub"]
    ratio = cfg["ratio"]
    cfg_scale = cfg["classifier_free_guidance"]
    attn_impl = cfg["attn_implementation"]
    out_dir = Path(cfg["output_dir"])

    model = AutoModelForCausalLM.from_pretrained(
        emu_hub,
        device_map=device,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        trust_remote_code=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(emu_hub, trust_remote_code=True, padding_side="left")
    image_processor = AutoImageProcessor.from_pretrained(vq_hub, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(
        vq_hub, device_map=device, trust_remote_code=True
    ).eval()
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer)

    generation_config = GenerationConfig(
        use_cache=True,
        eos_token_id=model.config.eos_token_id,
        pad_token_id=model.config.pad_token_id,
        max_new_tokens=40960,
        do_sample=True,
        top_k=2048,
    )

    def generate_one(prompt_text: str) -> Image.Image | None:
        full_prompt = prompt_text + POSITIVE_PROMPT
        kwargs = dict(
            mode="G",
            ratio=ratio,
            image_area=model.config.image_area,
            return_tensors="pt",
            padding="longest",
        )
        pos_inputs = processor(text=[full_prompt], **kwargs)
        neg_inputs = processor(text=[NEGATIVE_PROMPT], **kwargs)

        h = pos_inputs.image_size[:, 0]
        w = pos_inputs.image_size[:, 1]
        constrained_fn = processor.build_prefix_constrained_fn(h, w)
        logits_processor = LogitsProcessorList(
            [
                UnbatchedClassifierFreeGuidanceLogitsProcessor(
                    cfg_scale,
                    model,
                    unconditional_ids=neg_inputs.input_ids.to(device),
                ),
                PrefixConstrainedLogitsProcessor(constrained_fn, num_beams=1),
            ]
        )

        outputs = model.generate(
            pos_inputs.input_ids.to(device),
            generation_config,
            logits_processor=logits_processor,
            attention_mask=pos_inputs.attention_mask.to(device),
        )
        mm_list = processor.decode(outputs[0])
        for im in mm_list:
            if isinstance(im, Image.Image):
                return im
        return None

    for pid in prompt_ids:
        out_png = out_dir / f"{pid}.png"
        try:
            pil = generate_one(prompts[pid])
            if pil is None:
                yield pid, "decode produced no PIL image"
            else:
                pil.save(out_png)
                yield pid, None
        except Exception as e:
            yield pid, str(e)


def main() -> None:
    args = parse_args()
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

    merge_path = args.merge_json.resolve()
    if not merge_path.is_file():
        raise SystemExit(f"Missing prompts file: {merge_path}")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(merge_path)
    ids_sorted = sorted(prompts.keys())
    expected = set(range(1, 1001))
    missing_ids = expected - set(prompts.keys())
    if missing_ids:
        print(f"[WARN] merge.json missing prompt_ids: {sorted(missing_ids)[:20]}… ({len(missing_ids)} total)")

    to_run: list[int] = []
    for pid in ids_sorted:
        out_png = out_dir / f"{pid}.png"
        if args.skip_existing and out_png.is_file():
            continue
        to_run.append(pid)

    print(f"[INFO] Saving images under {out_dir}")
    print(
        f"[INFO] Total prompts: {len(ids_sorted)}, to generate: {len(to_run)}, "
        f"gpus={gpu_ids}, skip_existing={args.skip_existing}"
    )
    if not to_run:
        print("[DONE] Nothing to generate.")
        return

    emu3_repo = args.emu3_repo.resolve()
    cfg: dict[str, Any] = {
        "emu_hub": str(args.emu_hub),
        "vq_hub": str(args.vq_hub),
        "emu3_repo": str(emu3_repo),
        "output_dir": str(out_dir),
        "ratio": args.ratio,
        "classifier_free_guidance": float(args.classifier_free_guidance),
        "attn_implementation": args.attn_implementation,
    }

    n_workers = len(gpu_ids)
    shards = _split_round_robin(to_run, n_workers)
    for gid, shard in zip(gpu_ids, shards):
        print(f"[INFO] GPU {gid}: {len(shard)} image(s) in this process")

    failures: list[int] = []

    if n_workers == 1:
        with tqdm(total=len(to_run), desc="Emu3-Gen WISE", unit="img") as pbar:
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

        with tqdm(total=len(to_run), desc="Emu3-Gen WISE", unit="img") as pbar:
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
