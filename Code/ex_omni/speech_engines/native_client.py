"""Lifecycle and streaming client for the isolated Speech Generator."""

from __future__ import annotations

import atexit
import json
from multiprocessing.connection import Client
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

import numpy as np

from ex_omni.llm_engines.tensor_wire import encode_bf16_tensor


def _numpy(value, *, dtype):
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            value = value.float()
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


class NativeSpeechClient:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.process: subprocess.Popen | None = None
        self.connection = None
        self.authkey = secrets.token_hex(16)
        socket_root = Path(
            self.config.get("socket_dir", tempfile.gettempdir())
        ).expanduser()
        socket_root.mkdir(parents=True, exist_ok=True)
        self.socket_path = socket_root / f"ex-omni-speech-{os.getpid()}.sock"
        self.log_path = socket_root / f"ex-omni-speech-{os.getpid()}.log"
        atexit.register(self.close)

    def start(self) -> None:
        if self.connection is not None:
            return
        python = Path(
            str(self.config.get("python") or sys.executable)
        ).expanduser()
        project_root = Path(__file__).resolve().parents[2]
        command = [
            str(python),
            "-m",
            "ex_omni.speech_engines.native_service",
            "--socket",
            str(self.socket_path),
            "--authkey",
            self.authkey,
            "--model",
            str(Path(str(self.config["model_path"])).expanduser()),
            "--attention-backend",
            str(self.config.get("attention_backend", "sdpa")),
            "--compile-options",
            json.dumps(self.config.get("compile", {})),
        ]
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            value
            for value in (str(project_root), existing_pythonpath)
            if value
        )
        env["CUDA_VISIBLE_DEVICES"] = str(int(self.config["gpu_id"]))
        log_handle = self.log_path.open("ab", buffering=0)
        self.socket_path.unlink(missing_ok=True)
        self.process = subprocess.Popen(
            command,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timeout = float(self.config.get("startup_timeout_seconds", 1200))
        deadline = time.monotonic() + timeout
        last_error = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    "Speech service exited with code "
                    f"{self.process.returncode}; see {self.log_path}"
                )
            try:
                self.connection = Client(
                    str(self.socket_path),
                    family="AF_UNIX",
                    authkey=self.authkey.encode("utf-8"),
                )
                self.connection.send({"type": "ping"})
                ready = self.connection.recv()
                if ready.get("type") != "ready":
                    raise RuntimeError(
                        f"unexpected Speech service handshake: {ready}"
                    )
                print(
                    "[speech] isolated service ready on GPU "
                    f"{self.config['gpu_id']} (pid={ready.get('pid')})",
                    flush=True,
                )
                return
            except (FileNotFoundError, ConnectionRefusedError, EOFError) as exc:
                last_error = exc
                time.sleep(1)
        raise TimeoutError(
            f"Speech service did not start in {timeout:.0f}s; "
            f"last error={last_error}; see {self.log_path}"
        )

    def prepare_roleplay(
        self,
        *,
        ref_audio_waveform=None,
        ref_audio_waveform_lengths=None,
        has_ref_audio=None,
    ) -> None:
        self.start()
        self.connection.send(
            {
                "type": "prepare_roleplay",
                "ref_audio_waveform": _numpy(
                    ref_audio_waveform, dtype=np.float32
                ),
                "ref_audio_waveform_lengths": _numpy(
                    ref_audio_waveform_lengths, dtype=np.int64
                ),
                "has_ref_audio": _numpy(has_ref_audio, dtype=np.bool_),
            }
        )
        message = self.connection.recv()
        if message.get("type") == "prepared":
            return
        if message.get("type") == "error":
            raise RuntimeError(
                f"Speech role embedding failed: {message.get('error')}\n"
                f"{message.get('traceback', '')}"
            )
        raise RuntimeError(
            f"unexpected Speech service message: {message}"
        )

    def _drain_prediction_response(self) -> None:
        """Restore IPC alignment after a streaming callback is interrupted."""
        while True:
            message = self.connection.recv()
            kind = message.get("type")
            if kind == "unit_chunk":
                continue
            if kind in {"complete", "error"}:
                return
            raise RuntimeError(
                "unexpected Speech service message while draining an "
                f"interrupted prediction: {message}"
            )

    def predict(
        self,
        hidden,
        text_ids,
        *,
        ref_audio_waveform=None,
        ref_audio_waveform_lengths=None,
        has_ref_audio=None,
        prefix_units=None,
        max_speech_tokens: int,
        do_sample: bool,
        top_k: int,
        top_p: float,
        temperature: float,
        repetition_penalty: float,
        use_kv_cache: bool,
        residual_use_kv_cache: bool,
        unit_chunk_size: int,
        unit_chunk_callback=None,
        seed: int | None,
    ) -> np.ndarray:
        self.start()
        self.connection.send(
            {
                "type": "predict",
                "hidden": encode_bf16_tensor(hidden),
                "text_ids": [int(value) for value in text_ids],
                "ref_audio_waveform": _numpy(
                    ref_audio_waveform, dtype=np.float32
                ),
                "ref_audio_waveform_lengths": _numpy(
                    ref_audio_waveform_lengths, dtype=np.int64
                ),
                "has_ref_audio": _numpy(has_ref_audio, dtype=np.bool_),
                "prefix_units": _numpy(prefix_units, dtype=np.int64),
                "max_speech_tokens": int(max_speech_tokens),
                "do_sample": bool(do_sample),
                "top_k": int(top_k),
                "top_p": float(top_p),
                "temperature": float(temperature),
                "repetition_penalty": float(repetition_penalty),
                "use_kv_cache": bool(use_kv_cache),
                "residual_use_kv_cache": bool(residual_use_kv_cache),
                "unit_chunk_size": int(unit_chunk_size),
                "seed": seed,
            }
        )
        while True:
            message = self.connection.recv()
            kind = message.get("type")
            if kind == "unit_chunk":
                if unit_chunk_callback is not None:
                    try:
                        unit_chunk_callback(
                            np.asarray(message["units"], dtype=np.int64)
                        )
                    except BaseException:
                        self._drain_prediction_response()
                        raise
            elif kind == "complete":
                return np.asarray(message["units"], dtype=np.int64)
            elif kind == "error":
                raise RuntimeError(
                    f"Speech generation failed: {message.get('error')}\n"
                    f"{message.get('traceback', '')}"
                )
            else:
                raise RuntimeError(
                    f"unexpected Speech service message: {message}"
                )

    def close(self) -> None:
        connection, self.connection = self.connection, None
        if connection is not None:
            try:
                connection.send({"type": "shutdown"})
                connection.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
            finally:
                connection.close()
        process, self.process = self.process, None
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        self.socket_path.unlink(missing_ok=True)
