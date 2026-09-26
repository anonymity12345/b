"""Isolated native Speech Generator service."""

from __future__ import annotations

import argparse
import json
from multiprocessing.connection import Listener
import os
from pathlib import Path
from types import SimpleNamespace
import traceback

import numpy as np

from ex_omni.llm_engines.tensor_wire import decode_bf16_tensor


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(
            **{key: _namespace(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _load_speech_weights(model, checkpoint: Path) -> None:
    from safetensors import safe_open

    prefix = "model.speech_generator."
    state = {}
    for path in sorted(checkpoint.glob("model-*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    state[key[len(prefix) :]] = handle.get_tensor(key)
    if not state:
        raise RuntimeError(
            f"checkpoint contains no keys with prefix {prefix!r}: {checkpoint}"
        )
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed_missing = [
        key for key in missing if key.startswith("roleplay_speaker_encoder.")
    ]
    if unexpected or len(allowed_missing) != len(missing):
        raise RuntimeError(
            "Speech checkpoint mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )


def _build_model(args):
    import torch

    from ex_omni.model.attention import resolve_attn_implementation
    from ex_omni.model.speech_generator.speech_generator import SpeechGenerator

    # Resolve on the Speech worker's GPU, not the parent LLM device. They may
    # have different architectures or run in different Python environments.
    requested_attention = args.attention_backend
    args.attention_backend = resolve_attn_implementation(requested_attention)
    print(f"[speech] attention requested={requested_attention}, resolved={args.attention_backend}", flush=True)
    checkpoint = Path(args.model)
    payload = json.loads((checkpoint / "config.json").read_text())
    # Resolve model references inside the isolated speech process.
    from ex_omni.hub import materialize_pretrained_reference

    for key in ("pretrain_qwen_tts_weights", "pretrain_speaker_encoder_weights"):
        reference = payload.get(key)
        if reference not in (None, "", "none", "None"):
            payload[key] = materialize_pretrained_reference(str(reference))
    payload["attn_implementation"] = args.attention_backend
    payload["decode_attn_implementation"] = args.attention_backend
    model = SpeechGenerator(_namespace(payload))
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    _load_speech_weights(model, checkpoint)
    for module in model.modules():
        if hasattr(module, "attn_implementation"):
            module.attn_implementation = args.attention_backend
        if hasattr(module, "decode_attn_implementation"):
            module.decode_attn_implementation = args.attention_backend
    options = json.loads(args.compile_options)
    residual_graphs = bool(options.get("residual_cuda_graphs", False))
    if residual_graphs:
        from .residual_graph import ResidualCodebookGraphs

        predictor = model.talker.code_predictor
        predictor.logits_for_prefix = ResidualCodebookGraphs(predictor.logits_for_prefix)
        print("[speech] residual predictor CUDA graphs enabled; sampling remains eager", flush=True)
    if bool(options.get("enabled", False)):
        if not hasattr(torch, "compile"):
            raise RuntimeError("compile: true requires torch.compile")
        torch._dynamo.config.recompile_limit = int(
            options.get("recompile_limit", 16)
        )
        torch._dynamo.config.suppress_errors = bool(
            options.get("fallback_on_error", True)
        )
        mode = str(options.get("mode", "default"))
        if bool(options.get("cuda_graphs", False)):
            mode = "reduce-overhead"
        compile_kwargs = {
            "backend": str(options.get("backend", "inductor")),
            "mode": mode,
            "fullgraph": bool(options.get("fullgraph", False)),
            "dynamic": bool(options.get("dynamic", True)),
        }
        print(
            "Compiling Speech Talker and residual predictor "
            f"(backend={compile_kwargs['backend']}, mode={mode}, "
            f"cuda_graphs={bool(options.get('cuda_graphs', False))}, "
            f"dynamic={compile_kwargs['dynamic']}). First request warms graphs.",
            flush=True,
        )
        model.talker.model = torch.compile(
            model.talker.model, **compile_kwargs
        )
        if not residual_graphs:
            model.talker.code_predictor.model = torch.compile(
                model.talker.code_predictor.model, **compile_kwargs
            )
    return model


def _optional_tensor(value, *, device, dtype=None):
    if value is None:
        return None
    import torch

    tensor = torch.from_numpy(np.asarray(value))
    return tensor.to(device=device, dtype=dtype)


def _prepare_roleplay_embedding(model, command):
    import torch

    parameter = next(model.parameters())
    waveform = _optional_tensor(
        command.get("ref_audio_waveform"),
        device=parameter.device,
        dtype=torch.float32,
    )
    waveform_lengths = _optional_tensor(
        command.get("ref_audio_waveform_lengths"),
        device=parameter.device,
        dtype=torch.long,
    )
    has_ref_audio = _optional_tensor(
        command.get("has_ref_audio"),
        device=parameter.device,
        dtype=torch.bool,
    )
    return model.prepare_roleplay_embedding(
        ref_audio_waveform=waveform,
        ref_audio_waveform_lengths=waveform_lengths,
        has_ref_audio=has_ref_audio,
    )


def _predict(
    model,
    connection,
    command,
    embedding=None,
    sampling_generator=None,
) -> None:
    import torch

    parameter = next(model.parameters())
    hidden = decode_bf16_tensor(command["hidden"]).to(
        device=parameter.device,
        dtype=parameter.dtype,
    )
    text_ids = torch.tensor(
        [command["text_ids"]],
        dtype=torch.long,
        device=parameter.device,
    )
    if embedding is None:
        embedding = _prepare_roleplay_embedding(model, command)
    def emit(chunk) -> None:
        connection.send(
            {
                "type": "unit_chunk",
                "units": np.asarray(chunk, dtype=np.int64),
            }
        )

    predicted = model.predict(
        hidden,
        text_ids,
        embedding=embedding,
        prefix_speech_tokens=command.get("prefix_units"),
        max_speech_tokens=int(command["max_speech_tokens"]),
        do_sample=bool(command["do_sample"]),
        top_k=int(command["top_k"]),
        top_p=float(command["top_p"]),
        temperature=float(command["temperature"]),
        repetition_penalty=float(command["repetition_penalty"]),
        return_hidden_states=False,
        use_kv_cache=bool(command["use_kv_cache"]),
        residual_use_kv_cache=bool(command["residual_use_kv_cache"]),
        sampling_generator=sampling_generator,
        unit_chunk_size=int(command["unit_chunk_size"]),
        unit_chunk_callback=(
            emit if int(command["unit_chunk_size"]) > 0 else None
        ),
    )
    connection.send(
        {
            "type": "complete",
            "units": np.asarray(
                predicted[1] if isinstance(predicted, tuple) else predicted,
                dtype=np.int64,
            ),
        }
    )


def serve(args) -> None:
    import torch

    socket_path = Path(args.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    torch.cuda.set_device(0)
    model = _build_model(args)
    roleplay_embedding = None
    sampling_generator = None
    sampling_seed = None
    listener = Listener(
        str(socket_path),
        family="AF_UNIX",
        authkey=args.authkey.encode("utf-8"),
    )
    try:
        while True:
            connection = listener.accept()
            try:
                while True:
                    command = connection.recv()
                    kind = command.get("type")
                    if kind == "ping":
                        connection.send({"type": "ready", "pid": os.getpid()})
                    elif kind == "prepare_roleplay":
                        try:
                            sampling_generator = None
                            sampling_seed = None
                            with torch.inference_mode():
                                roleplay_embedding = _prepare_roleplay_embedding(
                                    model, command
                                )
                            connection.send({"type": "prepared"})
                        except Exception as exc:
                            connection.send(
                                {
                                    "type": "error",
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc(),
                                }
                            )
                    elif kind == "predict":
                        try:
                            seed = command.get("seed")
                            if seed is not None and (
                                sampling_generator is None
                                or sampling_seed != int(seed)
                            ):
                                parameter = next(model.parameters())
                                sampling_generator = torch.Generator(
                                    device=parameter.device
                                )
                                sampling_generator.manual_seed(int(seed))
                                sampling_seed = int(seed)
                            with torch.inference_mode():
                                _predict(
                                    model,
                                    connection,
                                    command,
                                    roleplay_embedding,
                                    sampling_generator,
                                )
                        except Exception as exc:
                            connection.send(
                                {
                                    "type": "error",
                                    "error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc(),
                                }
                            )
                    elif kind == "shutdown":
                        connection.send({"type": "stopped"})
                        return
                    else:
                        raise ValueError(
                            f"unknown Speech service command: {kind!r}"
                        )
            except EOFError:
                pass
            finally:
                connection.close()
    finally:
        listener.close()
        socket_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--authkey", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--attention-backend", required=True)
    parser.add_argument("--compile-options", default="{}")
    serve(parser.parse_args())


if __name__ == "__main__":
    main()
