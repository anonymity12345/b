#!/usr/bin/env python3
"""Launch the vLLM-Omni Wan2.1 Streaming Video service."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--address", required=True)
    parser.add_argument(
        "--model",
        default=str(
            Path(__file__).resolve().parents[1]
            / "configs"
            / "vllm_omni_wan21_model"
        ),
    )
    parser.add_argument("--authkey")
    args = parser.parse_args()

    from ex_omni.vllm_omni.video_service import serve_vllm_omni_video

    return serve_vllm_omni_video(
        args.config,
        model_path=args.model,
        address=args.address,
        authkey=args.authkey,
    )


if __name__ == "__main__":
    raise SystemExit(main())
