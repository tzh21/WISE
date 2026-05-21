#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Send one local request to serve_qwen.py.

- Without --image: call /generate (Qwen-Image).
- With --image: call /edit (Qwen-Image-Edit).

Examples::
    uv run python scripts/request_qwen.py "A glass teapot on wooden table"
    uv run python scripts/request_qwen.py "让它变成水彩风格" --image ./input.jpg
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
from pathlib import Path

import requests

_WISE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Send one request to local Qwen service")
    p.add_argument("prompt", help="Text prompt")
    p.add_argument("--image", default=None, help="Optional input image path; enables /edit")
    p.add_argument("--server", default="http://127.0.0.1:8000", help="Service base URL")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output PNG path. Default: WISE/local/outputs/qwen_image_request/{MMDD-HHMMSS}/demo.png",
    )
    p.add_argument("--ratio", default="1:1", help="Aspect ratio, e.g. 1:1 / 16:9 / 9:16")
    p.add_argument(
        "--positive-suffix",
        choices=("auto", "none", "en", "zh"),
        default="auto",
        help="Prompt suffix mode: auto / none / en / zh",
    )
    p.add_argument("--negative-prompt", default=" ", help="Negative prompt")
    p.add_argument("--num-inference-steps", type=int, default=50, help="Inference steps")
    p.add_argument("--true-cfg-scale", type=float, default=4.0, help="true_cfg_scale")
    p.add_argument("--guidance-scale", type=float, default=1.0, help="guidance_scale for /edit")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--timeout", type=float, default=600.0, help="HTTP timeout in seconds")
    return p.parse_args()


def default_output_path() -> Path:
    stem = datetime.now().strftime("%m%d-%H%M%S")
    return _WISE_ROOT / "local" / "outputs" / "qwen_image_request" / stem / "demo.png"


def _image_to_base64(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("ascii")


def _base64_to_bytes(data: str) -> bytes:
    return base64.b64decode(data)


def main() -> None:
    args = parse_args()
    base_url = args.server.rstrip("/")
    out_png = (args.output or default_output_path()).expanduser().resolve()
    out_png.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "prompt": args.prompt,
        "ratio": args.ratio,
        "positive_suffix": args.positive_suffix,
        "negative_prompt": args.negative_prompt,
        "num_inference_steps": int(args.num_inference_steps),
        "true_cfg_scale": float(args.true_cfg_scale),
        "guidance_scale": float(args.guidance_scale),
        "seed": int(args.seed),
    }

    if args.image:
        image_path = Path(args.image).expanduser().resolve()
        if not image_path.is_file():
            raise SystemExit(f"Input image not found: {image_path}")
        payload["image_base64"] = _image_to_base64(image_path)
        endpoint = "/edit"
    else:
        endpoint = "/generate"

    url = f"{base_url}{endpoint}"
    print(f"[INFO] POST {url}")
    resp = requests.post(url, json=payload, timeout=float(args.timeout))
    try:
        data = resp.json()
    except Exception as e:  # noqa: BLE001
        raise SystemExit(f"Service returned non-JSON response (HTTP {resp.status_code}): {resp.text}") from e

    if resp.status_code != 200 or not data.get("ok"):
        raise SystemExit(f"Service error (HTTP {resp.status_code}): {data.get('error', data)}")

    image_b64 = data.get("image_base64")
    if not image_b64:
        raise SystemExit("Service response missing `image_base64`.")
    out_png.write_bytes(_base64_to_bytes(str(image_b64)))
    print(f"[DONE] Saved {out_png}")


if __name__ == "__main__":
    main()
