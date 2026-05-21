#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serve local Qwen Image models with two dedicated GPUs:

- Qwen-Image (text-to-image): cuda:0
- Qwen-Image-Edit-2511 (image-edit): cuda:1

Run example::
    uv run python scripts/serve_qwen.py

API:
    GET  /health
    POST /generate      # text-to-image
    POST /edit          # image-edit
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_DEFAULT_QWEN_IMAGE = "/share/project/tzh/models/Qwen-Image"
_EDIT_SUFFIX = "Qwen-Image-Edit-2511"

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


def _positive_suffix_for(mode: str, prompt_text: str) -> str:
    if mode == "none":
        return ""
    if mode in ("en", "zh"):
        return _POSITIVE_SUFFIX[mode]
    return _POSITIVE_SUFFIX[_default_positive_lang(prompt_text)]


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Serve local Qwen-Image and Qwen-Image-Edit")
    p.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    p.add_argument("--model-hub", default=None, help="Qwen-Image path")
    p.add_argument("--edit-model-hub", default=None, help="Qwen-Image-Edit path")
    p.add_argument("--image-gpu", type=int, default=0, help="GPU id for Qwen-Image (default: 0)")
    p.add_argument("--edit-gpu", type=int, default=1, help="GPU id for Qwen-Image-Edit (default: 1)")
    return p.parse_args()


class QwenService:
    def __init__(self, image_hub: Path, edit_hub: Path, image_gpu: int, edit_gpu: int) -> None:
        import torch
        from diffusers import DiffusionPipeline, QwenImageEditPlusPipeline

        if not torch.cuda.is_available():
            raise SystemExit("CUDA is required for serving Qwen-Image.")

        self._torch = torch
        self._image_device = f"cuda:{image_gpu}"
        self._edit_device = f"cuda:{edit_gpu}"

        print(f"[INFO] Loading Qwen-Image on {self._image_device}: {image_hub}")
        self._image_pipe = DiffusionPipeline.from_pretrained(str(image_hub), torch_dtype=torch.bfloat16).to(
            self._image_device
        )

        print(f"[INFO] Loading Qwen-Image-Edit on {self._edit_device}: {edit_hub}")
        self._edit_pipe = QwenImageEditPlusPipeline.from_pretrained(str(edit_hub), torch_dtype=torch.bfloat16).to(
            self._edit_device
        )

        self._image_lock = threading.Lock()
        self._edit_lock = threading.Lock()

    def generate(self, payload: dict[str, Any]) -> bytes:
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("Field `prompt` is required.")

        ratio = str(payload.get("ratio", "1:1"))
        if ratio not in ASPECT_RATIOS:
            raise ValueError(f"Unsupported ratio: {ratio}. Choices: {sorted(ASPECT_RATIOS)}")
        width, height = ASPECT_RATIOS[ratio]

        positive_mode = str(payload.get("positive_suffix", "auto"))
        if positive_mode not in ("auto", "none", "en", "zh"):
            raise ValueError("`positive_suffix` must be one of: auto, none, en, zh.")
        full_prompt = prompt + _positive_suffix_for(positive_mode, prompt)

        negative_prompt = str(payload.get("negative_prompt", " "))
        steps = int(payload.get("num_inference_steps", 50))
        true_cfg_scale = float(payload.get("true_cfg_scale", 4.0))
        seed = int(payload.get("seed", 42))

        with self._image_lock, self._torch.inference_mode():
            gen = self._torch.Generator(device=self._image_device).manual_seed(seed % (2**32))
            image = self._image_pipe(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=steps,
                true_cfg_scale=true_cfg_scale,
                generator=gen,
            ).images[0]
        return _pil_to_png_bytes(image)

    def edit(self, payload: dict[str, Any]) -> bytes:
        prompt = str(payload.get("prompt", "")).strip()
        if not prompt:
            raise ValueError("Field `prompt` is required.")

        image_base64 = payload.get("image_base64")
        if not image_base64:
            raise ValueError("Field `image_base64` is required for /edit.")

        ratio = str(payload.get("ratio", "1:1"))
        if ratio not in ASPECT_RATIOS:
            raise ValueError(f"Unsupported ratio: {ratio}. Choices: {sorted(ASPECT_RATIOS)}")
        width, height = ASPECT_RATIOS[ratio]

        positive_mode = str(payload.get("positive_suffix", "auto"))
        if positive_mode not in ("auto", "none", "en", "zh"):
            raise ValueError("`positive_suffix` must be one of: auto, none, en, zh.")
        full_prompt = prompt + _positive_suffix_for(positive_mode, prompt)

        negative_prompt = str(payload.get("negative_prompt", " "))
        steps = int(payload.get("num_inference_steps", 50))
        true_cfg_scale = float(payload.get("true_cfg_scale", 4.0))
        guidance_scale = float(payload.get("guidance_scale", 1.0))
        seed = int(payload.get("seed", 42))

        pil_in = _decode_base64_image(str(image_base64))
        with self._edit_lock, self._torch.inference_mode():
            gen = self._torch.Generator(device=self._edit_device).manual_seed(seed % (2**32))
            image = self._edit_pipe(
                image=[pil_in],
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                true_cfg_scale=true_cfg_scale,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                num_inference_steps=steps,
                num_images_per_prompt=1,
                generator=gen,
            ).images[0]
        return _pil_to_png_bytes(image)


def _decode_base64_image(data: str):
    from PIL import Image

    raw = base64.b64decode(data)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _pil_to_png_bytes(image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    cl = int(handler.headers.get("Content-Length", "0"))
    if cl <= 0:
        return {}
    raw = handler.rfile.read(cl)
    if not raw:
        return {}
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError("Request body is not valid JSON.") from e


def make_handler(service: QwenService):
    class _Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                _json_response(
                    self,
                    HTTPStatus.OK,
                    {
                        "ok": True,
                        "image_device": service._image_device,  # noqa: SLF001
                        "edit_device": service._edit_device,  # noqa: SLF001
                    },
                )
                return
            _json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found."})

        def do_POST(self) -> None:  # noqa: N802
            try:
                payload = _read_json_body(self)
                if self.path == "/generate":
                    png_bytes = service.generate(payload)
                elif self.path == "/edit":
                    png_bytes = service.edit(payload)
                else:
                    _json_response(self, HTTPStatus.NOT_FOUND, {"ok": False, "error": "Not found."})
                    return

                _json_response(
                    self,
                    HTTPStatus.OK,
                    {"ok": True, "image_base64": base64.b64encode(png_bytes).decode("ascii")},
                )
            except ValueError as e:
                _json_response(self, HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(e)})
            except Exception as e:  # noqa: BLE001
                _json_response(self, HTTPStatus.INTERNAL_SERVER_ERROR, {"ok": False, "error": str(e)})

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[HTTP] {self.address_string()} - {fmt % args}")

    return _Handler


def main() -> None:
    args = parse_args()
    image_hub = resolve_image_hub(args.model_hub)
    edit_hub = resolve_edit_hub(args.edit_model_hub, image_hub)

    service = QwenService(
        image_hub=image_hub,
        edit_hub=edit_hub,
        image_gpu=int(args.image_gpu),
        edit_gpu=int(args.edit_gpu),
    )
    handler = make_handler(service)
    server = ThreadingHTTPServer((args.host, int(args.port)), handler)
    print(f"[INFO] Serving on http://{args.host}:{args.port}")
    print("[INFO] Endpoints: GET /health, POST /generate, POST /edit")
    server.serve_forever()


if __name__ == "__main__":
    main()
