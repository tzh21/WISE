#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate all WISE_Verified benchmark images with Emu3-Gen.

Outputs `{prompt_id}.png` (1–1000) into the chosen directory so you can run
`eval_qwen.sh` with IMAGE_DIR pointing at that folder.

Run from an environment that has Emu3 dependencies installed (same as Emu3's
image_generation.py), for example::

    cd /share/project/tzh/Emu3
    uv run python /share/project/tzh/WISE/scripts/generate_emu3_wise.py

Or set EMU3_REPO if the Emu3 source tree lives elsewhere.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


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
    p.add_argument("--device", default="cuda:0", help="Torch device for generation")
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


def main() -> None:
    args = parse_args()
    emu3_repo = args.emu3_repo.resolve()
    if str(emu3_repo) not in sys.path:
        sys.path.insert(0, str(emu3_repo))

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

    print(f"[INFO] Loading Emu3-Gen from {args.emu_hub}")
    model = AutoModelForCausalLM.from_pretrained(
        args.emu_hub,
        device_map=args.device,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.emu_hub, trust_remote_code=True, padding_side="left")
    image_processor = AutoImageProcessor.from_pretrained(args.vq_hub, trust_remote_code=True)
    image_tokenizer = AutoModel.from_pretrained(
        args.vq_hub, device_map=args.device, trust_remote_code=True
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

    device = args.device
    cfg_scale = args.classifier_free_guidance

    def generate_one(prompt_text: str) -> Image.Image | None:
        full_prompt = prompt_text + POSITIVE_PROMPT
        kwargs = dict(
            mode="G",
            ratio=args.ratio,
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

    to_run: list[int] = []
    for pid in ids_sorted:
        out_png = out_dir / f"{pid}.png"
        if args.skip_existing and out_png.is_file():
            continue
        to_run.append(pid)

    print(f"[INFO] Saving images under {out_dir}")
    print(f"[INFO] Total prompts: {len(ids_sorted)}, to generate: {len(to_run)}, skip_existing={args.skip_existing}")

    failures: list[int] = []
    for pid in tqdm(to_run, desc="Emu3-Gen WISE", unit="img"):
        out_png = out_dir / f"{pid}.png"
        try:
            pil = generate_one(prompts[pid])
            if pil is None:
                failures.append(pid)
                tqdm.write(f"[ERR] prompt_id={pid}: decode produced no PIL image")
                continue
            pil.save(out_png)
        except Exception as e:
            failures.append(pid)
            tqdm.write(f"[ERR] prompt_id={pid}: {e}")

    if failures:
        print(f"[WARN] Failed count={len(failures)}; ids (first 50): {failures[:50]}")
    print("[DONE]")


if __name__ == "__main__":
    main()
