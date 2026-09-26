"""Tensor-parallel setup for the Ex-Omni dialogue_model."""

from __future__ import annotations

from typing import Any, Mapping


class WorldMPU:
    """Minimal DeepSpeed MPU for a single TP replica spanning the world."""

    def __init__(self, model_group, data_group, world_size: int):
        self.model_group = model_group
        self.data_group = data_group
        self.world_size = int(world_size)

    def get_model_parallel_group(self):
        return self.model_group

    def get_model_parallel_world_size(self):
        return self.world_size

    def get_model_parallel_rank(self):
        import torch.distributed as dist

        return dist.get_rank(group=self.model_group)

    get_tensor_model_parallel_group = get_model_parallel_group
    get_tensor_model_parallel_world_size = get_model_parallel_world_size
    get_tensor_model_parallel_rank = get_model_parallel_rank
    get_slice_parallel_group = get_model_parallel_group
    get_slice_parallel_world_size = get_model_parallel_world_size
    get_slice_parallel_rank = get_model_parallel_rank

    def get_data_parallel_group(self):
        return self.data_group

    def get_data_parallel_world_size(self):
        return 1

    def get_data_parallel_rank(self):
        return 0


def _resolve_parent(root, path: str):
    parts = [part for part in path.split(".") if part]
    parent = root
    for part in parts[:-1]:
        if parent is None or not hasattr(parent, part):
            return None, None
        parent = getattr(parent, part)
    return parent, parts[-1] if parts else None


def _inner(model):
    return model.get_model() if hasattr(model, "get_model") else getattr(model, "model", None)


def _explicit_injection_policies():
    """Return separate deterministic TP policies for LLM and TTS layers.

    DeepSpeed 0.17 AutoTP merges repeated custom-layer policies into a set.
    Its per-process iteration order is not stable, and native replacement has
    intermittently aborted with SIGSEGV/heap corruption. More importantly,
    applying two client-module policies to one composite root makes DeepSpeed
    traverse that root again after the first in-place rewrite. Keep the two
    roots independent so each ``init_inference`` call sees one untouched model.
    """
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    from ex_omni.model.speech_generator.qwen3_tts_talker import (
        Qwen3TTSDecoderLayer,
    )

    all_reduce_linears = ("self_attn.o_proj", "mlp.down_proj")
    return (
        {Qwen3DecoderLayer: all_reduce_linears},
        {Qwen3TTSDecoderLayer: all_reduce_linears},
    )


def _validate_tp_linear_shapes(layer, tp_size: int, label: str) -> None:
    columnwise = (
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
    )
    rowwise = ("self_attn.o_proj", "mlp.down_proj")
    for path in columnwise:
        parent, leaf = _resolve_parent(layer, path)
        module = getattr(parent, leaf) if parent is not None and leaf else None
        size = getattr(module, "out_features", 0)
        if not size or size % tp_size:
            raise ValueError(
                f"{label}.{path}.out_features={size} is not divisible by "
                f"tp_size={tp_size}"
            )
    for path in rowwise:
        parent, leaf = _resolve_parent(layer, path)
        module = getattr(parent, leaf) if parent is not None and leaf else None
        size = getattr(module, "in_features", 0)
        if not size or size % tp_size:
            raise ValueError(
                f"{label}.{path}.in_features={size} is not divisible by "
                f"tp_size={tp_size}"
            )


def _parallelize_decoder_layers(layers, mesh, tp_size: int, label: str) -> None:
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        RowwiseParallel,
        parallelize_module,
    )

    for index, layer in enumerate(layers):
        _validate_tp_linear_shapes(layer, tp_size, f"{label}[{index}]")
        parallelize_module(
            layer,
            device_mesh=mesh,
            parallelize_plan={
                "self_attn.q_proj": ColwiseParallel(),
                "self_attn.k_proj": ColwiseParallel(),
                "self_attn.v_proj": ColwiseParallel(),
                "self_attn.o_proj": RowwiseParallel(),
                "mlp.gate_proj": ColwiseParallel(),
                "mlp.up_proj": ColwiseParallel(),
                "mlp.down_proj": RowwiseParallel(),
            },
            src_data_rank=0,
        )


def wrap_dialogue_model_native_tp(engine, config: Mapping[str, Any], context):
    """Shard Qwen and TTS decoder layers with PyTorch DTensor TP.

    This avoids DeepSpeed 0.17's in-place AutoTP rewrite, which can corrupt the
    native heap nondeterministically. Decoder parameters are sharded over the
    same all-world process group; embeddings and the explicitly excluded
    multimodal encoders remain replicated.
    """
    if not context.is_distributed:
        engine.distributed_context = context
        return engine
    tp_size = int(config.get("tp_size", context.world_size))
    if tp_size != context.world_size:
        raise ValueError(
            f"all-shared mode requires dialogue_model.tp_size==world_size "
            f"({tp_size}!={context.world_size})"
        )
    if str(config.get("load_method")) != "full_model":
        raise ValueError("native TP requires dialogue_model.load_method=full_model")
    if str(config.get("device_map", "cuda")) != "cuda":
        raise ValueError("native TP requires dialogue_model.device_map=cuda")

    import gc
    import torch
    from torch.distributed.device_mesh import DeviceMesh

    mesh = DeviceMesh.from_group(
        context.process_group,
        "cuda",
        mesh=list(range(context.world_size)),
        mesh_dim_names=("tp",),
    )
    inner = _inner(engine._model)
    layers = getattr(inner, "layers", None)
    if layers is None:
        raise RuntimeError("dialogue_model backbone has no decoder layers")
    if context.is_rank0:
        print("[dialogue_model] initializing native TP for main LLM", flush=True)
    _parallelize_decoder_layers(layers, mesh, tp_size, "llm.layers")

    speech_generator = getattr(inner, "speech_generator", None)
    talker = getattr(speech_generator, "talker", None)
    talker_layers = getattr(getattr(talker, "model", None), "layers", None)
    predictor_layers = getattr(
        getattr(getattr(talker, "code_predictor", None), "model", None),
        "layers",
        None,
    )
    if talker_layers is None or predictor_layers is None:
        raise RuntimeError("speech generator has no TTS decoder layers")
    if context.is_rank0:
        print("[dialogue_model] initializing native TP for speech generator", flush=True)
    _parallelize_decoder_layers(talker_layers, mesh, tp_size, "tts.layers")
    _parallelize_decoder_layers(
        predictor_layers,
        mesh,
        tp_size,
        "tts.code_predictor.layers",
    )

    gc.collect()
    torch.cuda.empty_cache()
    if context.is_rank0:
        print("[dialogue_model] native tensor parallel initialization complete", flush=True)
    engine.distributed_context = context
    engine.distributed_model_group = context.process_group
    engine.distributed_device_mesh = mesh
    engine.distributed_mp_size = context.world_size
    engine.distributed_src_rank = 0
    speech_generator.distributed_tp_group = context.process_group
    speech_generator.distributed_tp_src_rank = 0
    engine._model.eval()
    return engine


def wrap_dialogue_model_deepspeed(engine, config: Mapping[str, Any], context):
    """Shard the dialogue_model backbone with one all-world DeepSpeed TP group."""
    if not context.is_distributed:
        engine.distributed_context = context
        return engine
    if str(config.get("distributed_backend", "deepspeed")) != "deepspeed":
        raise ValueError("distributed dialogue_model requires distributed_backend=deepspeed")
    tp_size = int(config.get("tp_size", context.world_size))
    if tp_size != context.world_size:
        raise ValueError(
            f"all-shared mode requires dialogue_model.tp_size==world_size "
            f"({tp_size}!={context.world_size})"
        )
    if str(config.get("load_method")) != "full_model":
        raise ValueError("DeepSpeed TP requires dialogue_model.load_method=full_model")
    if str(config.get("device_map", "cuda")) != "cuda":
        raise ValueError("DeepSpeed TP requires dialogue_model.device_map=cuda")

    import torch
    import torch.distributed as dist
    import deepspeed

    if not getattr(deepspeed.comm, "is_initialized", lambda: False)():
        deepspeed.comm.init_distributed(
            dist_backend=dist.get_backend(), auto_mpi_discovery=False
        )
    # Every rank creates the same singleton groups in the same order.
    data_groups = [dist.new_group(ranks=[rank]) for rank in range(context.world_size)]
    mpu = WorldMPU(context.process_group, data_groups[context.rank], context.world_size)

    excluded_names = config.get(
        "deepspeed_exclude_modules",
        ["speech_encoder", "vision_encoder", "speech_generator.fusion"],
    )
    if isinstance(excluded_names, str):
        excluded_names = [item.strip() for item in excluded_names.split(",") if item.strip()]
    # DeepSpeed 0.17 mutates the supplied root in place. Applying the Qwen and
    # custom TTS policies to the same composite root is unstable because the
    # second pass traverses modules already rewritten by the first pass. Shard
    # the two roots independently, but over the same all-world TP group.
    main_excluded_names = [
        name for name in excluded_names if not name.startswith("speech_generator.")
    ]
    if "speech_generator" not in main_excluded_names:
        main_excluded_names.append("speech_generator")
    detached = {}
    inner = _inner(engine._model)
    for name in main_excluded_names:
        parent, leaf = _resolve_parent(inner, name)
        if parent is not None and leaf and hasattr(parent, leaf):
            module = getattr(parent, leaf)
            if module is not None:
                detached[name] = module
                setattr(parent, leaf, None)
    llm_policy, tts_policy = _explicit_injection_policies()
    if context.is_rank0:
        print("[dialogue_model] initializing main LLM tensor parallelism", flush=True)
    try:
        ds_engine = deepspeed.init_inference(
            engine._model,
            tensor_parallel={
                "tp_size": context.world_size,
                "mpu": mpu,
                "tp_group": context.process_group,
            },
            dtype=next(engine._model.parameters()).dtype,
            replace_with_kernel_inject=bool(
                config.get("deepspeed_kernel_inject", False)
            ),
            injection_policy=llm_policy,
        )
        engine._model = getattr(ds_engine, "module", ds_engine)
    finally:
        inner = _inner(engine._model)
        for name, module in detached.items():
            parent, leaf = _resolve_parent(inner, name)
            if parent is not None and leaf:
                setattr(parent, leaf, module)

    inner = _inner(engine._model)
    speech_generator = getattr(inner, "speech_generator", None)
    if speech_generator is None:
        raise RuntimeError("speech_generator was not restored after LLM TP setup")
    speech_excluded = {}
    for name in excluded_names:
        prefix = "speech_generator."
        if not name.startswith(prefix):
            continue
        local_name = name[len(prefix):]
        parent, leaf = _resolve_parent(speech_generator, local_name)
        if parent is not None and leaf and hasattr(parent, leaf):
            module = getattr(parent, leaf)
            if module is not None:
                speech_excluded[local_name] = module
                setattr(parent, leaf, None)
    if context.is_rank0:
        print("[dialogue_model] initializing speech generator tensor parallelism", flush=True)
    try:
        speech_ds_engine = deepspeed.init_inference(
            speech_generator,
            tensor_parallel={
                "tp_size": context.world_size,
                "mpu": mpu,
                "tp_group": context.process_group,
            },
            dtype=next(speech_generator.parameters()).dtype,
            replace_with_kernel_inject=False,
            injection_policy=tts_policy,
        )
        speech_generator = getattr(speech_ds_engine, "module", speech_ds_engine)
        inner.speech_generator = speech_generator
    finally:
        for name, module in speech_excluded.items():
            parent, leaf = _resolve_parent(speech_generator, name)
            if parent is not None and leaf:
                setattr(parent, leaf, module)

    if context.is_rank0:
        print("[dialogue_model] tensor parallel initialization complete", flush=True)
    engine.distributed_context = context
    engine.distributed_model_group = context.process_group
    engine.distributed_mp_size = context.world_size
    engine.distributed_src_rank = 0
    speech_generator = getattr(_inner(engine._model), "speech_generator", None)
    if speech_generator is not None:
        speech_generator.distributed_tp_group = context.process_group
        speech_generator.distributed_tp_src_rank = 0
    engine._model.eval()
    return engine


def wrap_dialogue_model_tensor_parallel(engine, config: Mapping[str, Any], context):
    backend = str(config.get("distributed_backend", "native_tp")).lower()
    if backend in {"native", "native_tp", "torch", "pytorch"}:
        return wrap_dialogue_model_native_tp(engine, config, context)
    if backend == "deepspeed":
        return wrap_dialogue_model_deepspeed(engine, config, context)
    raise ValueError(
        "dialogue_model.distributed_backend must be 'native_tp' or 'deepspeed'"
    )
