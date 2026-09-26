"""Shared split-pipeline deployment planning and Video service lifecycle."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import uuid

import yaml

from .execution import (
    derive_request_topology_config,
    normalize_execution_config,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_AUTHKEY = "ex-omni-local"


def _parse_gpu_ids(value: Any, *, name: str) -> tuple[int, ...]:
    if isinstance(value, str):
        values = [item.strip() for item in value.split(",") if item.strip()]
    elif value is None:
        values = []
    else:
        values = list(value)
    try:
        result = tuple(int(item) for item in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer GPU IDs") from exc
    if any(item < 0 for item in result):
        raise ValueError(f"{name} must contain non-negative GPU IDs")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} contains duplicate GPU IDs")
    return result


def _csv(values: Sequence[int]) -> str:
    return ",".join(str(item) for item in values)


def _read_deployment_config(path: str | Path) -> tuple[Path, dict[str, Any]]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise TypeError("configuration root must be a mapping")
    return config_path, payload


@dataclass(frozen=True)
class ReplicaTopology:
    llm_gpu_ids: tuple[int, ...]
    speech_gpu_ids: tuple[int, ...]
    waveform_gpu_id: int | None
    all_video_gpu_ids: tuple[int, ...]
    video_gpu_ids: tuple[int, ...]
    video_visible_gpu_ids: tuple[int, ...]
    video_parallelism: str
    video_engine: str
    video_world_size: int
    vae_gpu_id: int | None = None
    code2wav_default_device: str = "auto"

    @property
    def main_visible_gpu_ids(self) -> tuple[int, ...]:
        result = self.llm_gpu_ids + self.speech_gpu_ids
        if self.waveform_gpu_id is not None:
            result += (self.waveform_gpu_id,)
        return result

    @property
    def code2wav_device(self) -> str:
        if self.waveform_gpu_id is None:
            return self.code2wav_default_device
        return f"cuda:{len(self.llm_gpu_ids) + len(self.speech_gpu_ids)}"


@dataclass(frozen=True)
class SplitDeploymentPlan:
    config_path: Path | None
    project_root: Path
    python_executable: str
    address: Path
    authkey: str
    connect_timeout_seconds: float
    topology: ReplicaTopology

    def main_process_env(self) -> dict[str, str]:
        topology = self.topology
        return {
            "CUDA_VISIBLE_DEVICES": _csv(topology.main_visible_gpu_ids),
            "EX_OMNI_LLM_GPUS": _csv(topology.llm_gpu_ids),
            "EX_OMNI_SPEECH_GPU_ID": _csv(topology.speech_gpu_ids),
            "EX_OMNI_VIDEO_GPUS": _csv(topology.video_gpu_ids),
            "EX_OMNI_VIDEO_VISIBLE_GPUS": _csv(
                topology.video_visible_gpu_ids
            ),
            "EX_OMNI_CODE2WAV_DEVICE": topology.code2wav_device,
            "EX_OMNI_VIDEO_AUTHKEY": self.authkey,
        }

    def video_process_env(
        self, base: Mapping[str, str] | None = None
    ) -> dict[str, str]:
        env = dict(os.environ if base is None else base)
        env["CUDA_VISIBLE_DEVICES"] = _csv(
            self.topology.video_visible_gpu_ids
        )
        env["EX_OMNI_VIDEO_AUTHKEY"] = self.authkey
        if self.topology.vae_gpu_id is not None:
            env["EX_OMNI_VAE_DEVICE"] = "cuda:" + str(
                self.topology.video_visible_gpu_ids.index(self.topology.vae_gpu_id)
            )
        root = str(self.project_root)
        env["PYTHONPATH"] = root + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        return env

    def video_service_command(self) -> tuple[str, ...]:
        if self.config_path is None:
            raise ValueError("config_path is required to launch the Video service")
        topology = self.topology
        if topology.video_engine == "vllm_omni":
            return (
                self.python_executable,
                str(self.project_root / "scripts/serve_vllm_omni_video.py"),
                "--config",
                str(self.config_path),
                "--address",
                str(self.address),
            )
        return (
            self.python_executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={topology.video_world_size}",
            str(self.project_root / "scripts/serve_streaming_video.py"),
            "--config",
            str(self.config_path),
            "--address",
            str(self.address),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["config_path"] = (
            None if self.config_path is None else str(self.config_path)
        )
        payload["project_root"] = str(self.project_root)
        payload["address"] = str(self.address)
        payload["main_process_env"] = self.main_process_env()
        payload["video_service_command"] = list(self.video_service_command())
        return payload


def _environment_override(
    environment: Mapping[str, str] | None,
    name: str,
    default: Any,
) -> Any:
    if environment is None or name not in environment:
        return default
    return environment[name]


def build_split_deployment_plan(
    config: Mapping[str, Any] | str | Path,
    *,
    config_path: str | Path | None = None,
    address: str | Path | None = None,
    authkey: str | None = None,
    connect_timeout_seconds: float = 600.0,
    python_executable: str | None = None,
    project_root: str | Path | None = None,
    environment: Mapping[str, str] | None = None,
) -> SplitDeploymentPlan:
    """Build a torch-free deployment plan for the configured pipeline."""
    if isinstance(config, (str, Path)):
        resolved_config_path, source = _read_deployment_config(config)
    else:
        source = deepcopy(dict(config))
        resolved_config_path = (
            None
            if config_path is None
            else Path(config_path).expanduser().resolve()
        )
    execution = normalize_execution_config(source)
    if not execution["enabled"]:
        raise ValueError("execution.enabled must be true for the split launcher")
    vae_gpu_id = execution["vae_gpu_id"]

    llm_gpu_ids = _parse_gpu_ids(
        _environment_override(environment, "LLM_GPUS", execution["llm"]["gpu_ids"]),
        name="LLM_GPUS",
    )
    speech_gpu_ids = _parse_gpu_ids(
        _environment_override(environment, "SPEECH_GPUS", execution["speech"]["gpu_ids"]),
        name="SPEECH_GPUS",
    )
    default_waveform = "" if execution["speech"]["waveform_gpu_id"] is None else str(execution["speech"]["waveform_gpu_id"])
    waveform_values = _parse_gpu_ids(
        _environment_override(environment, "WAVEFORM_GPU", default_waveform),
        name="WAVEFORM_GPU",
    )
    if len(waveform_values) > 1:
        raise ValueError("WAVEFORM_GPU supports at most one GPU")
    waveform_gpu_id = waveform_values[0] if waveform_values else None
    all_video_gpu_ids = _parse_gpu_ids(
        _environment_override(
            environment, "ALL_VIDEO_GPUS", execution["video"]["gpu_ids"]
        ),
        name="ALL_VIDEO_GPUS",
    )
    video_parallelism = str(
        _environment_override(
            environment,
            "VIDEO_PARALLELISM",
            execution["video"]["parallelism"],
        )
    ).lower()
    video_engine = str(
        _environment_override(
            environment,
            "VIDEO_ENGINE",
            dict(source.get("video", {})).get("engine", "native"),
        )
    ).lower()

    video_gpu_ids = _parse_gpu_ids(
        _environment_override(
            environment, "VIDEO_GPUS", all_video_gpu_ids
        ),
        name="VIDEO_GPUS",
    )
    default_visible_gpu_ids = video_gpu_ids
    if vae_gpu_id is not None:
        default_visible_gpu_ids += (vae_gpu_id,)
    video_visible_gpu_ids = _parse_gpu_ids(
        _environment_override(
            environment,
            "VIDEO_VISIBLE_GPUS",
            default_visible_gpu_ids,
        ),
        name="VIDEO_VISIBLE_GPUS",
    )

    if not llm_gpu_ids:
        raise ValueError("LLM_GPUS must contain at least one GPU")
    if len(speech_gpu_ids) != 1:
        raise ValueError("the split launcher requires exactly one Speech GPU")
    if not video_gpu_ids or not video_visible_gpu_ids:
        raise ValueError("Video GPU groups cannot be empty")
    if video_parallelism not in {
        "single",
        "pipeline",
        "fsdp",
        "sequence_parallel",
    }:
        raise ValueError(
            f"unsupported VIDEO_PARALLELISM={video_parallelism!r}"
        )
    if video_parallelism == "single" and len(video_gpu_ids) != 1:
        raise ValueError("VIDEO_PARALLELISM=single requires one Video GPU")
    if video_parallelism == "sequence_parallel" and len(video_gpu_ids) < 2:
        raise ValueError(
            "VIDEO_PARALLELISM=sequence_parallel requires at least two Video GPUs"
        )
    if not set(video_gpu_ids).issubset(video_visible_gpu_ids):
        raise ValueError("VIDEO_GPUS must be included in VIDEO_VISIBLE_GPUS")
    if vae_gpu_id is not None:
        if vae_gpu_id not in video_visible_gpu_ids or vae_gpu_id in video_gpu_ids:
            raise ValueError("dedicated VAE GPU must be visible and outside the Video worker group")
        if video_visible_gpu_ids[:len(video_gpu_ids)] != video_gpu_ids:
            raise ValueError("Video worker GPUs must precede the dedicated VAE GPU in visibility order")
    if video_engine not in {"native", "vllm_omni"}:
        raise ValueError(f"unsupported video.engine={video_engine!r}")
    if video_engine == "vllm_omni" and len(video_visible_gpu_ids) < 2:
        raise ValueError(
            "vLLM-Omni Video requires two visible GPUs for DiT and VAE"
        )
    groups_to_check = {
        "llm": set(llm_gpu_ids),
        "speech": set(speech_gpu_ids),
        "waveform": (
            set() if waveform_gpu_id is None else {waveform_gpu_id}
        ),
        "video": set(video_visible_gpu_ids),
    }
    for left, right in (
        ("llm", "speech"),
        ("llm", "waveform"),
        ("llm", "video"),
        ("speech", "waveform"),
        ("speech", "video"),
        ("waveform", "video"),
    ):
        overlap = sorted(groups_to_check[left] & groups_to_check[right])
        if overlap:
            raise ValueError(
                "split deployment GPU groups must be disjoint; "
                f"{left}/{right} overlap on {overlap}"
            )

    topology = ReplicaTopology(
        llm_gpu_ids=llm_gpu_ids,
        speech_gpu_ids=speech_gpu_ids,
        waveform_gpu_id=waveform_gpu_id,
        all_video_gpu_ids=all_video_gpu_ids,
        video_gpu_ids=video_gpu_ids,
        video_visible_gpu_ids=video_visible_gpu_ids,
        video_parallelism=video_parallelism,
        video_engine=video_engine,
        video_world_size=(
            1 if video_engine == "vllm_omni" else len(video_gpu_ids)
        ),
        vae_gpu_id=vae_gpu_id,
        code2wav_default_device=str(dict(source.get("speech", source.get("dialogue_model", {}))).get("code2wav_device", "auto")),
    )
    resolved_authkey = (
        authkey
        or (environment or {}).get("EX_OMNI_VIDEO_AUTHKEY")
        or os.environ.get("EX_OMNI_VIDEO_AUTHKEY")
        or DEFAULT_VIDEO_AUTHKEY
    )
    return SplitDeploymentPlan(
        config_path=resolved_config_path,
        project_root=Path(project_root or PROJECT_ROOT).expanduser().resolve(),
        python_executable=str(python_executable or sys.executable),
        address=Path(
            address
            or f"/tmp/ex-omni-video-{os.getpid()}-{uuid.uuid4().hex[:8]}.sock"
        ),
        authkey=str(resolved_authkey),
        connect_timeout_seconds=float(connect_timeout_seconds),
        topology=topology,
    )


def apply_main_process_env(
    plan: SplitDeploymentPlan,
    *,
    environment: dict[str, str] | None = None,
) -> dict[str, str]:
    """Apply the main-process GPU binding before importing CUDA runtimes."""
    target = os.environ if environment is None else environment
    values = plan.main_process_env()
    target.update(values)
    return values


def inject_remote_video_runtime(
    config: Mapping[str, Any],
    plan: SplitDeploymentPlan,
) -> dict[str, Any]:
    result = deepcopy(dict(config))
    runtime = dict(result.get("runtime", {}))
    runtime["video_service_address"] = str(plan.address)
    runtime["video_service_authkey"] = plan.authkey
    runtime["video_service_connect_timeout_seconds"] = (
        plan.connect_timeout_seconds
    )
    result["runtime"] = runtime
    return result


def derive_split_dialogue_config(
    config: Mapping[str, Any],
    *,
    world_size: int = 1,
) -> dict[str, Any]:
    """Derive Dialogue workers while preserving the remote Video topology."""
    result = derive_request_topology_config(config, world_size=world_size)
    result["video"] = deepcopy(dict(config["video"]))
    return result


def derive_embedded_worker_config(
    config: Mapping[str, Any],
    *,
    world_size: int,
) -> dict[str, Any]:
    """Build a native TP/FSDP topology for one embedded request."""
    result = deepcopy(dict(config))
    execution = dict(result.get("execution", {}))
    execution["enabled"] = False
    result["execution"] = execution
    result["dialogue_model"]["engine"] = "native"
    return derive_request_topology_config(result, world_size=world_size)


def derive_single_gpu_demo_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Share one GPU while respecting the configured model residency policy."""
    return derive_embedded_worker_config(config, world_size=1)


class VideoServiceManager:
    """Own one split Video service subprocess and its Unix socket."""

    def __init__(
        self,
        plan: SplitDeploymentPlan,
        *,
        log_path: str | Path,
        poll_interval_seconds: float = 0.25,
    ) -> None:
        self.plan = plan
        self.log_path = Path(log_path).expanduser().resolve()
        self.poll_interval_seconds = float(poll_interval_seconds)
        self.process: subprocess.Popen | None = None
        self._log_handle = None

    def start(self) -> "VideoServiceManager":
        if self.process is not None:
            return self
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.plan.address.unlink(missing_ok=True)
        self._log_handle = self.log_path.open("a", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                self.plan.video_service_command(),
                cwd=self.plan.project_root,
                env=self.plan.video_process_env(),
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + self.plan.connect_timeout_seconds
            while time.monotonic() < deadline:
                if self.plan.address.exists():
                    return self
                return_code = self.process.poll()
                if return_code is not None:
                    raise RuntimeError(
                        "Video service "
                        f"(gpus={_csv(self.plan.topology.video_visible_gpu_ids)}, "
                        f"parallelism={self.plan.topology.video_parallelism}) "
                        f"exited with code {return_code}; see {self.log_path}"
                    )
                time.sleep(self.poll_interval_seconds)
            raise TimeoutError(
                "Video service did not become ready within "
                f"{self.plan.connect_timeout_seconds:g}s; see {self.log_path}"
            )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        process, self.process = self.process, None
        if process is not None:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
        self.plan.address.unlink(missing_ok=True)

    def __enter__(self) -> "VideoServiceManager":
        return self.start()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def print_deployment_plan(
    plan: SplitDeploymentPlan, output_format: str
) -> None:
    topology = plan.topology
    if output_format == "legacy":
        for values in (
            topology.llm_gpu_ids,
            topology.speech_gpu_ids,
            (() if topology.waveform_gpu_id is None else (topology.waveform_gpu_id,)),
            topology.all_video_gpu_ids,
        ):
            print(_csv(values))
        print(topology.video_parallelism)
        print(topology.video_engine)
        return
    if output_format == "shell":
        for value in (
            _csv(topology.llm_gpu_ids),
            _csv(topology.speech_gpu_ids),
            "" if topology.waveform_gpu_id is None else str(topology.waveform_gpu_id),
            _csv(topology.all_video_gpu_ids),
            _csv(topology.video_gpu_ids),
            _csv(topology.video_visible_gpu_ids),
            topology.video_parallelism,
            topology.video_engine,
            str(topology.video_world_size),
            _csv(topology.main_visible_gpu_ids),
            topology.code2wav_device,
        ):
            print(value)
        return
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ex-Omni split deployment tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "serve-video"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", required=True)
        subparser.add_argument("--address")
        subparser.add_argument("--authkey")
    plan_parser = subparsers.choices["plan"]
    plan_parser.add_argument(
        "--format",
        choices=("legacy", "shell", "json"),
        default="json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_split_deployment_plan(
        args.config,
        address=args.address,
        authkey=args.authkey,
        environment=os.environ,
    )
    if args.command == "plan":
        print_deployment_plan(plan, args.format)
        return 0
    command = plan.video_service_command()
    os.execvpe(command[0], command, plan.video_process_env())
    raise AssertionError("os.execvpe returned unexpectedly")


if __name__ == "__main__":
    raise SystemExit(main())
