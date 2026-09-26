"""Media executable discovery shared by inference entry points."""

from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=8)
def _has_h264_encoder(path: str) -> bool:
    try:
        result = subprocess.run([path, '-hide_banner', '-encoders'],
                                capture_output=True, text=True, timeout=10)
        return result.returncode == 0 and 'libx264' in result.stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


def ffmpeg_executable(configured_path: str | os.PathLike[str] | None = None) -> str:
    """Return the configured or automatically discovered ffmpeg executable."""
    if configured_path not in (None, ""):
        path = Path(configured_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"configured ffmpeg executable does not exist: {path}"
            )
        if not os.access(path, os.X_OK):
            raise PermissionError(
                f"configured ffmpeg executable is not executable: {path}"
            )
        return str(path)
    override = os.environ.get("EX_OMNI_FFMPEG_BIN") or os.environ.get("FFMPEG_BIN")
    if override:
        return override
    system_ffmpeg = shutil.which("ffmpeg")
    # Some cluster system builds omit libx264; such a binary can decode media
    # but cannot produce the browser-compatible MP4s used by the demo.
    if system_ffmpeg and _has_h264_encoder(system_ffmpeg):
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except (ImportError, RuntimeError) as exc:
        raise FileNotFoundError(
            "ffmpeg was not found; install imageio-ffmpeg or set "
            "EX_OMNI_FFMPEG_BIN"
        ) from exc
