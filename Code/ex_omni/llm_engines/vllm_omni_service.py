"""Isolated vLLM-Omni service for Ex-Omni prompt-embedding decoding."""

from __future__ import annotations

import argparse
from multiprocessing.connection import Listener
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import time
import traceback
import uuid

from ex_omni.llm_engines.tensor_wire import (
    decode_bf16_tensor,
    encode_bf16_tensor,
)


def _build_engine(args):
    from importlib.metadata import version
    for package in ("vllm", "vllm-omni"):
        if not version(package).startswith("0.26."):
            raise RuntimeError(f"Ex-Omni thinker adapter requires {package} 0.26.x; "
                               "use the isolated requirements-vllm-omni.txt environment")
    # The service is already process-isolated. Keeping EngineCore in-process
    # preserves the dynamic out-of-tree model registry and removes one IPC hop.
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    # uv may create a venv whose Python symlinks to the base interpreter.
    # Keep the venv helpers (notably ninja) visible to FlashInfer JIT.
    python_bin = str(Path(sys.executable).parent)
    os.environ["PATH"] = os.pathsep.join(
        (python_bin, os.environ.get("PATH", ""))
    )

    from ex_omni.vllm_omni.llm_model import (
        MODEL_ARCH,
        register_vllm_omni_llm_model,
    )

    register_vllm_omni_llm_model()
    from vllm.usage.usage_lib import UsageContext
    from vllm.v1.engine.llm_engine import LLMEngine
    from vllm_omni.engine.arg_utils import OmniEngineArgs
    from vllm_omni.outputs.output_processor import MultimodalOutputProcessor

    engine_args = OmniEngineArgs(
        model=args.model,
        model_stage="thinker",
        model_arch=MODEL_ARCH,
        engine_output_type="latent",
        worker_cls="ex_omni.vllm_omni.thinker_worker.ExOmniGPUARWorker",
        scheduler_cls=(
            "vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler"
        ),
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        max_num_seqs=1,
        max_num_batched_tokens=args.max_model_len,
        async_scheduling=False,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        enable_prompt_embeds=True,
        enable_prefix_caching=False,
        disable_log_stats=True,
        trust_remote_code=False,
        tokenizer_mode="slow",
        logits_processors=[
            "ex_omni.llm_engines.vllm_protocol:"
            "StrictAssistantProtocolV1LogitsProcessor"
        ],
    )
    llm_engine = LLMEngine.from_engine_args(
        engine_args=engine_args,
        usage_context=UsageContext.LLM_CLASS,
    )
    llm_engine.output_processor = MultimodalOutputProcessor(
        tokenizer=llm_engine.tokenizer,
        log_stats=llm_engine.log_stats,
        engine_core_output_type="latent",
    )
    return SimpleNamespace(llm_engine=llm_engine)


def _latent_output(output):
    multimodal = getattr(output, "multimodal_output", None)
    if not multimodal and output.outputs:
        multimodal = getattr(output.outputs[0], "multimodal_output", None)
    if not multimodal or "latent" not in multimodal:
        raise RuntimeError(
            "vLLM-Omni output did not contain latent hidden states"
        )
    return multimodal["latent"]


def _generate(engine, connection, command):
    import torch
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    prompt_embeds = decode_bf16_tensor(command["prompt_embeds"])
    if prompt_embeds.ndim == 3 and prompt_embeds.shape[0] == 1:
        prompt_embeds = prompt_embeds[0]
    # vLLM-Omni hashes prompt embeddings through NumPy, which has no BF16
    # scalar type. BF16 -> FP32 is exact; the model casts inputs back to BF16.
    prompt_embeds = prompt_embeds.float()
    prompt_length = int(prompt_embeds.shape[0])
    sampling = SamplingParams(
        temperature=float(command["temperature"]),
        top_p=float(command["top_p"]),
        max_tokens=int(command["max_new_tokens"]) + 5,
        seed=command.get("seed"),
        output_kind=RequestOutputKind.CUMULATIVE,
    )
    request_id = f"ex-omni-{uuid.uuid4().hex}"
    from vllm.inputs.engine import embeds_input
    # Omni 0.26's prefill collector expects token-ID length even for embeds.
    # Use mixed-input metadata with every position explicitly marked
    # as an embedding, so placeholder IDs are never used by the decoder.
    prompt = embeds_input(prompt_embeds, prompt_token_ids=[0] * prompt_length,
                          is_token_ids=[False] * prompt_length)
    engine.llm_engine.add_request(request_id, prompt, sampling)
    sent = 0
    sent_token_ids = []
    missing_hidden_tokens = 0
    final_output = None
    started = time.perf_counter()
    first_token_at = None
    while engine.llm_engine.has_unfinished_requests():
        outputs = engine.llm_engine.step()
        now = time.perf_counter()
        for output in outputs:
            if output.request_id != request_id:
                continue
            final_output = output
            token_ids = list(output.outputs[0].token_ids)
            if token_ids and first_token_at is None:
                first_token_at = now
            latent = _latent_output(output)
            available = max(0, int(latent.shape[0]) - prompt_length)
            ready = min(len(token_ids), available)
            while sent < ready:
                message = {
                    "type": "token",
                    "token_id": int(token_ids[sent]),
                    "hidden": encode_bf16_tensor(
                        latent[prompt_length + sent : prompt_length + sent + 1]
                    ),
                }
                connection.send(message)
                sent_token_ids.append(int(token_ids[sent]))
                sent += 1
    if final_output is None:
        raise RuntimeError("vLLM-Omni completed without a request output")
    token_ids = list(final_output.outputs[0].token_ids)
    while sent < len(token_ids):
        message = {
            "type": "token",
            "token_id": int(token_ids[sent]),
            "hidden": None,
        }
        connection.send(message)
        sent_token_ids.append(int(token_ids[sent]))
        missing_hidden_tokens += 1
        sent += 1
    finished = time.perf_counter()
    completion = final_output.outputs[0]
    print(
        "[llm-request] "
        f"prompt_tokens={prompt_length} output_tokens={len(token_ids)} "
        f"finish_reason={getattr(completion, 'finish_reason', None)!r} "
        f"stop_reason={getattr(completion, 'stop_reason', None)!r} "
        f"sent_tokens={len(sent_token_ids)} "
        f"sent_matches_final={sent_token_ids == token_ids} "
        f"missing_hidden_tokens={missing_hidden_tokens} "
        f"first_ids={token_ids[:8]} last_ids={token_ids[-8:]}",
        flush=True,
    )
    connection.send(
        {
            "type": "complete",
            "stats": {
                "prompt_tokens": prompt_length,
                "output_tokens": len(token_ids),
                "ttft_seconds": (
                    first_token_at - started if first_token_at is not None else None
                ),
                "total_seconds": finished - started,
            },
        }
    )


def serve(args) -> None:
    import torch

    socket_path = Path(args.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    engine = _build_engine(args)
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
                    elif kind == "generate":
                        try:
                            with torch.inference_mode():
                                _generate(engine, connection, command)
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
                        raise ValueError(f"unknown service command: {kind!r}")
            except EOFError:
                pass
            finally:
                connection.close()
    finally:
        listener.close()
        engine_core = getattr(engine.llm_engine, "engine_core", None)
        shutdown = getattr(engine_core, "shutdown", None)
        if callable(shutdown):
            shutdown()
        socket_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--authkey", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.4)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enforce-eager", action="store_true")
    serve(parser.parse_args())


if __name__ == "__main__":
    main()
