#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a single image from a text prompt with Qwen-Image (diffusers).

- Without an input image path: loads Qwen-Image (text-to-image, ``DiffusionPipeline``).
- With an image path: loads ``Qwen-Image-Edit-2511`` (same folder family as Qwen-Image) via
  ``QwenImageEditPlusPipeline``.
- Uses single-GPU loading by default on ``cuda:0`` (or ``--gpu N``).

Output::
    WISE/local/outputs/qwen_image/{MMDD-HHMMSS}/demo.png

Example::
    uv run python scripts/generate_qwen_single.py "A red apple" --ratio 16:9
    uv run python scripts/generate_qwen_single.py "Make it grayscale" ./input.jpg

    # Single-GPU (default cuda:0 if --gpu omitted)
    uv run python scripts/generate_qwen_single.py "..."
    uv run python scripts/generate_qwen_single.py "..." --gpu 1

Default checkpoints: env ``QWEN_IMAGE_HUB`` or ``/share/project/tzh/models/Qwen-Image``.
Edit checkpoint: env ``QWEN_IMAGE_EDIT_HUB`` or ``{parent}/Qwen-Image-Edit-2511``.
"""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

_WISE_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_QWEN_IMAGE = "/share/project/tzh/models/Qwen-Image"
_EDIT_SUFFIX = "Qwen-Image-Edit-2511"

# Same aspect presets as ``generate_qwen_wise.py`` / upstream Qwen-Image examples.
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


def resolve_image_hub(explicit: str | None) -> Path:
    if explicit and explicit.strip():
        return Path(explicit.strip())
    raw = os.environ.get("QWEN_IMAGE_HUB", _DEFAULT_QWEN_IMAGE)
    return Path(raw.strip())


def resolve_edit_hub(explicit: str | None, image_hub: Path) -> Path:
    if explicit and explicit.strip():
        return Path(explicit.strip())
    raw = os.environ.get("QWEN_IMAGE_EDIT_HUB")
    if raw and raw.strip():
        return Path(raw.strip())
    return image_hub.parent / _EDIT_SUFFIX


def resolve_output_timestamp_dir() -> Path:
    stem = datetime.now().strftime("%m%d-%H%M%S")
    return _WISE_ROOT / "local" / "outputs" / "qwen_image" / stem


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single Qwen-Image / Qwen-Image-Edit generation")
    p.add_argument("prompt", help="Text prompt")
    p.add_argument(
        "image",
        nargs="?",
        default=None,
        help=f"Optional input image path; enables edit ({_EDIT_SUFFIX})",
    )
    p.add_argument(
        "--model-hub",
        default=None,
        help=f"Qwen-Image path (default: env QWEN_IMAGE_HUB or {_DEFAULT_QWEN_IMAGE})",
    )
    p.add_argument(
        "--edit-model-hub",
        default=None,
        help=f"Edit model path (default: env QWEN_IMAGE_EDIT_HUB or parent/{_EDIT_SUFFIX})",
    )
    p.add_argument(
        "--gpu",
        type=int,
        default=0,
        metavar="N",
        help="Single-GPU device id, used as cuda:N (default: 0).",
    )
    p.add_argument(
        "--ratio",
        default="1:1",
        choices=sorted(ASPECT_RATIOS.keys()),
        help='Output aspect preset (default "1:1")',
    )
    p.add_argument(
        "--positive-suffix",
        choices=("auto", "none", "en", "zh"),
        default="auto",
        help="Append quality suffix to prompt: auto / en / zh / none",
    )
    p.add_argument(
        "--negative-prompt",
        default=" ",
        help="Negative prompt (default: single space, as in upstream examples)",
    )
    p.add_argument(
        "--num-inference-steps",
        type=int,
        default=50,
        help="Inference steps (edit README often uses 40; default here matches WISE bench script)",
    )
    p.add_argument(
        "--true-cfg-scale",
        type=float,
        default=4.0,
        help="true_cfg_scale passed to the pipeline",
    )
    p.add_argument(
        "--guidance-scale",
        type=float,
        default=1.0,
        help="guidance_scale (mainly for edit pipeline; keep 1.0 per upstream README)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed",
    )
    return p.parse_args()


def _positive_suffix_for(mode: str, prompt_text: str) -> str:
    if mode == "none":
        return ""
    if mode in ("en", "zh"):
        return _POSITIVE_SUFFIX[mode]
    return _POSITIVE_SUFFIX[_default_positive_lang(prompt_text)]


def main() -> None:
    args = parse_args()
    image_hub = resolve_image_hub(args.model_hub)
    edit_hub = resolve_edit_hub(args.edit_model_hub, image_hub)

    gpu_id = int(args.gpu)

    import torch
    from diffusers import DiffusionPipeline, QwenImageEditPlusPipeline
    from PIL import Image

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for Qwen-Image generation.")

    device = f"cuda:{gpu_id}"
    torch_dtype = torch.bfloat16
    width, height = ASPECT_RATIOS[args.ratio]
    full_prompt = args.prompt + _positive_suffix_for(args.positive_suffix, args.prompt)

    out_dir = resolve_output_timestamp_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / "demo.png"

    def load_and_make_generator(pipe_cls: Any, hub_s: str) -> tuple[Any, torch.Generator]:
        pipe = pipe_cls.from_pretrained(hub_s, torch_dtype=torch_dtype)
        pipe = pipe.to(device)
        gen = torch.Generator(device=device).manual_seed(args.seed % (2**32))
        return pipe, gen

    if args.image is None:
        hub_s = str(image_hub)
        print(f"[INFO] mode=text-to-image model_hub={hub_s}")
        print(f"[INFO] device={device} ratio={args.ratio} ({width}x{height}) seed={args.seed}")
        pipe, generator = load_and_make_generator(DiffusionPipeline, hub_s)
        image = pipe(
            prompt=full_prompt,
            negative_prompt=args.negative_prompt,
            width=width,
            height=height,
            num_inference_steps=int(args.num_inference_steps),
            true_cfg_scale=float(args.true_cfg_scale),
            generator=generator,
        ).images[0]
    else:
        in_path = Path(args.image).expanduser().resolve()
        if not in_path.is_file():
            raise SystemExit(f"Input image not found: {in_path}")

        hub_s = str(edit_hub)
        print(f"[INFO] mode=image-edit model_hub={hub_s}")
        print(f"[INFO] input={in_path} device={device} ratio={args.ratio} ({width}x{height}) seed={args.seed}")
        pil_in = Image.open(in_path).convert("RGB")

        pipe, generator = load_and_make_generator(QwenImageEditPlusPipeline, hub_s)
        with torch.inference_mode():
            image = pipe(
                image=[pil_in],
                prompt=full_prompt,
                negative_prompt=args.negative_prompt,
                true_cfg_scale=float(args.true_cfg_scale),
                guidance_scale=float(args.guidance_scale),
                width=width,
                height=height,
                num_inference_steps=int(args.num_inference_steps),
                num_images_per_prompt=1,
                generator=generator,
            ).images[0]

    image.save(out_png)
    print(f"[DONE] Saved {out_png}")


if __name__ == "__main__":
    main()
