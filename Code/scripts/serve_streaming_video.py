"""Launch the independent video worker group."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--address", required=True)
    parser.add_argument(
        "--authkey",
        help="Defaults to EX_OMNI_VIDEO_AUTHKEY (local-only fallback otherwise).",
    )
    return parser


def main() -> int:
    import os

    if os.environ.get("EX_OMNI_DEBUG_STACKS") == "1":
        import faulthandler
        import signal

        faulthandler.enable()
        faulthandler.register(signal.SIGUSR1, all_threads=True)
    args = build_parser().parse_args()
    from ex_omni.video_service import serve_video

    return serve_video(
        args.config,
        address=args.address,
        authkey=args.authkey,
    )


if __name__ == "__main__":
    raise SystemExit(main())
