"""Generate and cache a silent, image-conditioned avatar loop before chatting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from ex_omni.media import ffmpeg_executable


IDLE_PROMPT = (
    "The person in the reference image is quietly listening to a conversation. "
    "Mouth closed, relaxed neutral expression, subtle natural breathing, occasional "
    "gentle blinking and very slight head movement. Maintain the same identity, "
    "pose, clothing and background. Static camera. No talking or lip movement."
)


def image_identity(image: str | Path, mode: str) -> str:
    return hashlib.sha256(mode.encode() + Path(image).read_bytes()).hexdigest()


def loop_frames(frames: np.ndarray) -> np.ndarray:
    """Return smoothly to the beginning without repeating endpoint frames."""
    if len(frames) < 3:
        raise ValueError("Idle generation returned fewer than three frames")
    return np.concatenate((frames, frames[-2:0:-1]))


def prepare_idle(runtime, ref_image: str | Path) -> Path:
    image = Path(ref_image).resolve()
    config = runtime._config
    if config is None:
        raise RuntimeError("Load the model before preparing the avatar")
    video_config = config['video']
    signature = {
        'version': 1, 'image': image_identity(image, runtime.mode),
        'prompt': IDLE_PROMPT, 'video': video_config, 'units': 48,
    }
    # Include local checkpoint timestamps so replacing weights invalidates cache.
    signature['weights'] = {
        k: Path(str(video_config[k])).stat().st_mtime_ns
        for k in ('base_checkpoint', 'lora_checkpoint', 'dit_path')
        if video_config.get(k) and Path(str(video_config[k])).is_file()
    }
    key = hashlib.sha256(json.dumps(signature, sort_keys=True, default=str).encode()).hexdigest()[:24]
    cache = runtime.output_dir / 'idle'
    cache.mkdir(parents=True, exist_ok=True)
    output = cache / f'{key}.mp4'
    if output.is_file() and output.stat().st_size > 0:
        return output

    # Encode real digital silence. Codebook index zero is NOT a silence token.
    from ex_omni.model.video_generator.models.speech_tokens import load_qwen_speech_tokenizer
    tokenizer = load_qwen_speech_tokenizer(str(video_config['speech_tokenizer_path']), device='cuda:0')
    encoded = tokenizer.encode(np.zeros(4 * 24000, dtype=np.float32), sr=24000)
    units = encoded.audio_codes[0].detach().cpu().numpy().astype(np.int64)
    del encoded, tokenizer
    if units.ndim != 2 or units.shape[1] != 16 or len(units) < 48:
        raise ValueError(f'Unexpected silence token shape: {units.shape}')
    units = units[:48]
    generator = runtime.pipeline.video_generator
    with tempfile.TemporaryDirectory(dir=cache, prefix='prepare-') as temporary:
        if runtime.mode == 'streaming':
            stream = generator.start_stream(
                ref_image=image, video_prompt=IDLE_PROMPT, seed=42,
                steps=int(video_config['steps']),
            )
            chunks = []
            try:
                for offset in range(0, len(units), 6):
                    chunks.extend(item['frames'] for item in stream.push_units(units[offset:offset + 6]))
                chunks.extend(item['frames'] for item in stream.finish())
            finally:
                stream.abort()
            frames = np.concatenate(chunks)
        else:
            from ex_omni.schemas import SpeechTokens
            import imageio_ffmpeg

            result = generator.generate(
                ref_image=image, speech_tokens=SpeechTokens(units),
                output_path=Path(temporary) / 'generated.mp4',
                video_prompt=IDLE_PROMPT, seed=42,
                steps=int(video_config['steps']),
            )
            reader = imageio_ffmpeg.read_frames(str(result.output_path), pix_fmt='rgb24')
            try:
                metadata = next(reader)
                width, height = metadata['size']
                frames = np.stack([np.frombuffer(frame, np.uint8).reshape(height, width, 3) for frame in reader])
            finally:
                reader.close()
        frames = np.ascontiguousarray(loop_frames(frames), dtype=np.uint8)
        height, width = frames.shape[1:3]
        pending = Path(temporary) / 'idle.mp4'
        subprocess.run([
            ffmpeg_executable(runtime._ffmpeg_path), '-hide_banner', '-loglevel', 'error', '-y',
            '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}',
            '-framerate', str(runtime._fps), '-i', 'pipe:0', '-an',
            '-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
            '-movflags', '+faststart', str(pending),
        ], input=frames.tobytes(), check=True, timeout=120)
        pending.replace(output)
    return output
