"""Command line interface. Argument parsing occurs only inside ``main``."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence


DEFAULT_CONFIG = "configs/inference.full_sequence.example.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ex-omni-2d")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate config without loading weights")
    validate.add_argument("--config", default=DEFAULT_CONFIG)
    validate.add_argument("--check-weights", action="store_true")

    video = subparsers.add_parser("video", help="generate video from prepared conditions")
    video.add_argument("--config", default=DEFAULT_CONFIG)
    text = video.add_mutually_exclusive_group(required=True)
    text.add_argument("--video-prompt")
    text.add_argument("--vtp")
    video.add_argument("--ref-img", "--image", dest="ref_img", required=True)
    video.add_argument("--speech-tokens", required=True)
    video.add_argument("--output", required=True)
    video.add_argument("--mode", choices=("full_sequence", "streaming"))
    video.add_argument("--lenient-vtp", action="store_true")

    chat = subparsers.add_parser("chat-to-video", help="run dialogue model then video generator")
    chat.add_argument("--config", default=DEFAULT_CONFIG)
    chat.add_argument("--text", required=True)
    chat.add_argument("--ref-img", "--image", dest="ref_img", required=True)
    chat.add_argument("--ref-audio", required=True)
    chat.add_argument("--role-card")
    chat.add_argument("--output", required=True)
    chat.add_argument("--session-id", default="default")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        from .config import validate_config

        report = validate_config(args.config, check_weights=args.check_weights)
        payload = asdict(report)
        payload["config_path"] = str(payload["config_path"])
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 0

    from .pipeline import ExOmni2DPipeline

    pipeline = ExOmni2DPipeline(args.config)
    if args.command == "video":
        result = pipeline.generate_video(
            video_prompt=args.video_prompt,
            vtp=args.vtp,
            ref_image=Path(args.ref_img),
            speech_tokens=Path(args.speech_tokens),
            output_path=Path(args.output),
            mode=args.mode,
            strict_vtp=not args.lenient_vtp,
        )
        print(str(result.output_path))
        return 0

    result = pipeline.chat_to_video(
        args.text,
        ref_image=Path(args.ref_img),
        ref_audio=Path(args.ref_audio) if args.ref_audio else None,
        role_card=args.role_card,
        output_path=Path(args.output),
        session_id=args.session_id,
    )
    print(str(result.video.output_path))
    return 0
