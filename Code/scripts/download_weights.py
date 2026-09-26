#!/usr/bin/env python3
"""Download model dependencies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


PUBLIC_MODELS = {
    "wan2.1": ("Wan-AI/Wan2.1-T2V-1.3B", "Wan2.1-T2V-1.3B"),
    "qwen3-vl": ("Qwen/Qwen3-VL-2B-Instruct", "Qwen3-VL-2B-Instruct"),
    "qwen3-tts-tokenizer": (
        "Qwen/Qwen3-TTS-Tokenizer-12Hz",
        "Qwen3-TTS-Tokenizer-12Hz",
    ),
    "omniavatar": ("OmniAvatar/OmniAvatar-1.3B", "OmniAvatar-1.3B"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("weights"),
        help="Destination root (default: ./weights).",
    )
    parser.add_argument(
        "--component",
        action="append",
        choices=sorted(PUBLIC_MODELS),
        help="Public component to fetch; repeat as needed. Defaults to all.",
    )
    parser.add_argument(
        "--ex-omni-repo",
        help="Optional Hugging Face repository containing Ex-Omni-2D weights.",
    )
    parser.add_argument(
        "--ex-omni-dir-name",
        default="Ex-Omni-2D",
        help="Local directory name for --ex-omni-repo.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Branch, tag, or commit used for every requested repository.",
    )
    return parser.parse_args()


def download(repo_id: str, destination: Path, revision: str, token: str | None) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    info = HfApi(token=token).model_info(repo_id, revision=revision)
    snapshot_download(
        repo_id=repo_id,
        revision=info.sha,
        local_dir=destination,
        token=token,
    )
    return {
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": info.sha,
        "path": str(destination.resolve()),
    }


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    selected = args.component or list(PUBLIC_MODELS)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"models": {}}
    for component in selected:
        repo_id, directory = PUBLIC_MODELS[component]
        print(f"Downloading {component}: {repo_id}", flush=True)
        manifest["models"][component] = download(
            repo_id, args.output_dir / directory, args.revision, token
        )

    if args.ex_omni_repo:
        print(f"Downloading Ex-Omni-2D: {args.ex_omni_repo}", flush=True)
        manifest["models"]["ex-omni-2d"] = download(
            args.ex_omni_repo,
            args.output_dir / args.ex_omni_dir_name,
            args.revision,
            token,
        )

    manifest_path = args.output_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
