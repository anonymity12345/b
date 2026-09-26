"""Lifecycle and streaming client for the isolated vLLM-Omni service."""

from __future__ import annotations

import atexit
from multiprocessing.connection import Client
import os
from pathlib import Path
import secrets
import signal
import subprocess
import tempfile
import time
from typing import Any, Iterator, Mapping

from .tensor_wire import decode_bf16_tensor, encode_bf16_tensor


class VllmOmniClient:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = dict(config)
        self.process: subprocess.Popen | None = None
        self.connection = None
        self.authkey = secrets.token_hex(16)
        socket_root = Path(
            self.config.get("socket_dir", tempfile.gettempdir())
        ).expanduser()
        socket_root.mkdir(parents=True, exist_ok=True)
        self.socket_path = socket_root / f"ex-omni-vllm-{os.getpid()}.sock"
        self.log_path = socket_root / f"ex-omni-vllm-{os.getpid()}.log"
        atexit.register(self.close)

    def _gpu_ids(self) -> list[int]:
        value = os.environ.get("EX_OMNI_LLM_GPUS")
        if value is None:
            value = self.config.get(
                "gpu_ids",
                [self.config.get("gpu_id", 0)],
            )
        if isinstance(value, str):
            gpu_ids = [
                int(item.strip()) for item in value.split(",") if item.strip()
            ]
        else:
            gpu_ids = [int(item) for item in value]
        if not gpu_ids or len(set(gpu_ids)) != len(gpu_ids):
            raise ValueError("vLLM gpu_ids must be a non-empty unique list")
        return gpu_ids

    def start(self) -> None:
        if self.connection is not None:
            return
        python = Path(str(self.config["python"])).expanduser()
        model = Path(str(self.config["model_path"])).expanduser()
        project_root = Path(__file__).resolve().parents[2]
        omni_path = Path(str(self.config["vllm_omni_path"])).expanduser()
        gpu_ids = self._gpu_ids()
        tensor_parallel_size = int(
            self.config.get("tensor_parallel_size", len(gpu_ids))
        )
        if os.environ.get("EX_OMNI_LLM_GPUS") is not None:
            tensor_parallel_size = len(gpu_ids)
        if tensor_parallel_size != len(gpu_ids):
            raise ValueError(
                "vLLM tensor_parallel_size must equal the number of gpu_ids"
            )
        command = [
            str(python),
            "-m",
            "ex_omni.llm_engines.vllm_omni_service",
            "--socket",
            str(self.socket_path),
            "--authkey",
            self.authkey,
            "--model",
            str(model),
            "--max-model-len",
            str(int(self.config.get("max_model_len", 8192))),
            "--gpu-memory-utilization",
            str(float(self.config.get("gpu_memory_utilization", 0.4))),
            "--tensor-parallel-size",
            str(tensor_parallel_size),
        ]
        if bool(self.config.get("enforce_eager", False)):
            command.append("--enforce-eager")
        env = os.environ.copy()
        include_dirs = self.config.get("include_dirs", [])
        if include_dirs:
            env["CPATH"] = os.pathsep.join(
                [str(Path(path).expanduser()) for path in include_dirs]
                + ([env["CPATH"]] if env.get("CPATH") else []))
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(
            value
            for value in (str(project_root), str(omni_path), existing_pythonpath)
            if value
        )
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(item) for item in gpu_ids)
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
                    f"vLLM-Omni service exited with code {self.process.returncode}; "
                    f"see {self.log_path}"
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
                    raise RuntimeError(f"unexpected service handshake: {ready}")
                print(
                    f"[llm] vLLM-Omni service ready on GPUs "
                    f"{gpu_ids} with TP={tensor_parallel_size} "
                    f"(pid={ready.get('pid')})",
                    flush=True,
                )
                return
            except (FileNotFoundError, ConnectionRefusedError, EOFError) as exc:
                last_error = exc
                time.sleep(1)
        raise TimeoutError(
            f"vLLM-Omni service did not start in {timeout:.0f}s; "
            f"last error={last_error}; see {self.log_path}"
        )

    def generate(
        self,
        prompt_embeds,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
        seed: int | None,
    ) -> Iterator[tuple[int, Any | None]]:
        prompt_length = int(prompt_embeds.shape[-2])
        limit = int(self.config.get("max_model_len", 8192))
        required = prompt_length + int(max_new_tokens) + 5
        if required > limit:
            raise ValueError(
                f"Thinker context budget exceeded: input={prompt_length} tokens "
                f"(including image/audio embeddings), response reserve={int(max_new_tokens) + 5}, "
                f"limit={limit}. Clear removes conversation history but retains the selected "
                "avatar inputs. Shorten the input/role description, use a smaller reference "
                "image, or increase llm.vllm.max_model_len."
            )
        print(f"[thinker] input_tokens={prompt_length} response_reserve={int(max_new_tokens) + 5} "
              f"context_limit={limit}", flush=True)
        self.start()
        self.connection.send(
            {
                "type": "generate",
                "prompt_embeds": encode_bf16_tensor(prompt_embeds),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "max_new_tokens": int(max_new_tokens),
                "seed": seed,
            }
        )
        while True:
            message = self.connection.recv()
            kind = message.get("type")
            if kind == "token":
                hidden = (
                    decode_bf16_tensor(message["hidden"]).unsqueeze(0)
                    if message.get("hidden") is not None
                    else None
                )
                yield int(message["token_id"]), hidden
            elif kind == "complete":
                self.last_stats = dict(message.get("stats", {}))
                return
            elif kind == "error":
                raise RuntimeError(
                    f"vLLM-Omni generation failed: {message.get('error')}\n"
                    f"{message.get('traceback', '')}"
                )
            else:
                raise RuntimeError(f"unexpected vLLM-Omni message: {message}")

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

