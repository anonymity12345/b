from dataclasses import dataclass
from typing import Iterable, Iterator, Optional
import time
from .profiling import profile_video_chunk

import torch

from ..schedulers.contracts import FlowMapSchedule

from .state import DiTStreamingConfig, DiTStreamingState, VAEStreamingState


@dataclass
class StreamingChunkOutput:
    frames: torch.Tensor
    latents: torch.Tensor
    timing_seconds: dict[str, float]


@dataclass
class StreamingChunkInput:
    """One online input chunk; callers may provide chunks as audio arrives."""

    initial_latents: torch.Tensor
    context: torch.Tensor
    image_condition: torch.Tensor
    negative_context: Optional[torch.Tensor] = None
    prepared_audio_condition: Optional[torch.Tensor] = None
    prepared_audio_unconditional: Optional[torch.Tensor] = None


def combine_streaming_cfg_predictions(
    positive_prediction: torch.Tensor,
    text_cfg_scale: float,
    audio_cfg_scale: float,
    audio_unconditional_prediction: Optional[torch.Tensor] = None,
    text_unconditional_prediction: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if text_cfg_scale == 1.0 and audio_cfg_scale == 1.0:
        return positive_prediction
    if text_cfg_scale == 1.0:
        if audio_unconditional_prediction is None:
            raise ValueError("Audio CFG requires an audio-unconditional prediction.")
        return audio_unconditional_prediction + audio_cfg_scale * (
            positive_prediction - audio_unconditional_prediction
        )
    if audio_cfg_scale == text_cfg_scale:
        if text_unconditional_prediction is None:
            raise ValueError("Joint CFG requires a text/audio-unconditional prediction.")
        return text_unconditional_prediction + text_cfg_scale * (
            positive_prediction - text_unconditional_prediction
        )
    if (
        audio_unconditional_prediction is None
        or text_unconditional_prediction is None
    ):
        raise ValueError(
            "Separate text/audio CFG requires both unconditional predictions."
        )
    return (
        text_unconditional_prediction
        + text_cfg_scale
        * (audio_unconditional_prediction - text_unconditional_prediction)
        + audio_cfg_scale
        * (positive_prediction - audio_unconditional_prediction)
    )


class StreamingInferenceSession:
    """Causal streaming runtime with clean-history recaching."""

    def __init__(
        self,
        pipe,
        config: Optional[DiTStreamingConfig] = None,
        num_inference_steps: int = 50,
        sigma_shift: float = 5.0,
        text_cfg_scale: float = 1.0,
        audio_cfg_scale: float = 1.0,
        flowmap_schedule: Optional[FlowMapSchedule] = None,
        decode_output: bool = True,
        use_kv_cache: bool = True,
    ):
        if (
            getattr(pipe, "sp_size", 1) != 1
            and not bool(getattr(pipe, "streaming_sequence_parallel", False))
        ):
            raise ValueError(
                "StreamingInferenceSession requires native streaming sequence "
                "parallel support when sp_size > 1."
            )
        if text_cfg_scale < 0:
            raise ValueError("text_cfg_scale must be non-negative.")
        if audio_cfg_scale < 0:
            raise ValueError("audio_cfg_scale must be non-negative.")
        self.pipe = pipe
        self.config = config or DiTStreamingConfig()
        self.config.validate()
        self.num_inference_steps = int(num_inference_steps)
        self.sigma_shift = float(sigma_shift)
        self.text_cfg_scale = float(text_cfg_scale)
        self.audio_cfg_scale = float(audio_cfg_scale)
        if flowmap_schedule is not None and len(
            flowmap_schedule.source_timesteps
        ) != self.num_inference_steps:
            raise ValueError(
                "Flow-map schedule length must equal num_inference_steps."
            )
        self.flowmap_schedule = flowmap_schedule
        self.decode_output = bool(decode_output)
        self.use_kv_cache = bool(use_kv_cache)
        self.dit_state = DiTStreamingState()
        self.audio_uncond_dit_state = DiTStreamingState()
        self.text_uncond_dit_state = DiTStreamingState()
        self.vae_state: Optional[VAEStreamingState] = None

    def reset(self):
        self.dit_state = DiTStreamingState()
        self.audio_uncond_dit_state = DiTStreamingState()
        self.text_uncond_dit_state = DiTStreamingState()
        self.vae_state = None

    def _stabilize_cudagraph_cache(self, state: DiTStreamingState) -> None:
        if not bool(getattr(self.pipe, "cudagraphs_enabled", False)):
            return
        for cache in state.layer_caches:
            if cache is None:
                continue
            cache.key = cache.key.clone()
            cache.value = cache.value.clone()
            cache.positions = cache.positions.clone()
            if cache.compressed_key is not None:
                cache.compressed_key = cache.compressed_key.clone()
                cache.compressed_value = cache.compressed_value.clone()
                cache.compressed_positions = cache.compressed_positions.clone()
            if cache.sink_key is not None:
                cache.sink_key = cache.sink_key.clone()
                cache.sink_value = cache.sink_value.clone()
                cache.sink_positions = cache.sink_positions.clone()

    def _cache_clean_chunk(
        self,
        latents,
        context,
        image_condition,
        prepared_audio_condition,
        state,
    ) -> None:
        if (self.config.use_cuda_graphs and self.flowmap_schedule is not None
                and latents.is_cuda and state.text_context_cache is not None
                and state.layer_caches and all(c is not None for c in state.layer_caches)):
            from .clean_graph import cache_clean_graph
            cache_clean_graph(self, latents, context, image_condition, prepared_audio_condition, state)
            return
        cache_timestep = torch.zeros(
            (latents.shape[0],),
            device=latents.device,
            dtype=self.pipe.torch_dtype,
        )
        cache_time = (
            {"r_timestep": cache_timestep}
            if self.flowmap_schedule is not None
            else {}
        )
        reference_sink_only = state.global_frame_offset == 0 and latents.shape[2] == 1
        if not reference_sink_only:
            self.pipe.dit(
                latents,
                timestep=cache_timestep,
                context=context,
                y=image_condition,
                prepared_audio_condition=prepared_audio_condition,
                streaming_state=state,
                streaming_config=self.config,
                streaming_cache_resolution="compressed",
                update_streaming_cache=True,
                **cache_time,
            )
        self.pipe.dit(
            latents,
            timestep=cache_timestep,
            context=context,
            y=image_condition,
            prepared_audio_condition=prepared_audio_condition,
            streaming_state=state,
            streaming_config=self.config,
            streaming_cache_resolution="full",
            update_streaming_cache=True,
            **cache_time,
        )
        self._stabilize_cudagraph_cache(state)

    def generate(
        self,
        chunks: Iterable[StreamingChunkInput],
    ) -> Iterator[StreamingChunkOutput]:
        """Consume prepared chunks incrementally and emit decoded video chunks."""
        for chunk in chunks:
            yield self.denoise_chunk(
                chunk.initial_latents,
                chunk.context,
                chunk.image_condition,
                negative_context=chunk.negative_context,
                prepared_audio_condition=chunk.prepared_audio_condition,
                prepared_audio_unconditional=chunk.prepared_audio_unconditional,
            )

    @torch.no_grad()
    def prefill_reference(
        self,
        reference_latents: torch.Tensor,
        context: torch.Tensor,
        image_condition: torch.Tensor,
        *,
        negative_context: Optional[torch.Tensor] = None,
        prepared_audio_condition: Optional[torch.Tensor] = None,
        prepared_audio_unconditional: Optional[torch.Tensor] = None,
    ) -> StreamingChunkOutput:
        expected = self.config.first_chunk_latent_frames
        if self.dit_state.global_frame_offset != 0:
            raise ValueError("Reference sink has already been prefilled.")
        if reference_latents.ndim != 5 or reference_latents.shape[2] != expected:
            raise ValueError(
                "Reference prefill must match first_chunk_latent_frames: "
                f"expected {expected}, got {tuple(reference_latents.shape)}."
            )
        if image_condition.shape[2] != expected:
            raise ValueError("Reference image condition length differs from prefill.")
        use_text_cfg = self.text_cfg_scale != 1.0
        use_audio_cfg = self.audio_cfg_scale != 1.0
        use_any_cfg = use_text_cfg or use_audio_cfg
        use_joint_cfg = use_text_cfg and self.audio_cfg_scale == self.text_cfg_scale
        need_audio_unconditional = use_any_cfg and not use_joint_cfg
        if use_text_cfg and negative_context is None:
            raise ValueError("Text CFG reference prefill requires negative_context.")

        started = time.perf_counter()
        self.pipe.load_models_to_device(["dit"])
        self._cache_clean_chunk(
            reference_latents,
            context,
            image_condition,
            prepared_audio_condition,
            self.dit_state,
        )
        if need_audio_unconditional:
            self._cache_clean_chunk(
                reference_latents,
                context,
                image_condition,
                prepared_audio_unconditional,
                self.audio_uncond_dit_state,
            )
        if use_text_cfg:
            self._cache_clean_chunk(
                reference_latents,
                negative_context,
                image_condition,
                prepared_audio_unconditional,
                self.text_uncond_dit_state,
            )
        recached = time.perf_counter()
        if self.decode_output:
            self.pipe.load_models_to_device(["vae"])
            frames, self.vae_state = self.pipe.vae.decode_streaming(
                reference_latents,
                state=self.vae_state,
                device=self.pipe.device,
            )
        else:
            frames = reference_latents.new_empty((0,))
        finished = time.perf_counter()
        return StreamingChunkOutput(
            frames=frames,
            latents=reference_latents,
            timing_seconds={
                "dit_denoise_seconds": 0.0,
                "dit_kv_recache_seconds": recached - started,
                "vae_decode_seconds": finished - recached,
            },
        )

    @torch.no_grad()
    @profile_video_chunk
    def denoise_chunk(
        self,
        initial_latents: torch.Tensor,
        context: torch.Tensor,
        image_condition: torch.Tensor,
        negative_context: Optional[torch.Tensor] = None,
        prepared_audio_condition: Optional[torch.Tensor] = None,
        prepared_audio_unconditional: Optional[torch.Tensor] = None,
        step_progress_bar=None,
    ) -> StreamingChunkOutput:
        if initial_latents.ndim != 5:
            raise ValueError("initial_latents must have shape [B, C, T, H, W].")
        if initial_latents.shape[2] != self.config.chunk_latent_frames:
            raise ValueError("Each generated chunk must contain three new latent frames.")
        if self.dit_state.global_frame_offset == 0:
            raise ValueError("Reference sink must be prefilled before denoising a chunk.")
        if image_condition.shape[2] != initial_latents.shape[2]:
            raise ValueError("image_condition and latent chunk lengths must match.")
        if prepared_audio_condition is not None:
            expected_audio_frames = initial_latents.shape[2]
            if prepared_audio_condition.shape[2] != expected_audio_frames:
                raise ValueError(
                    "Prepared audio condition must contain one entry per latent frame: "
                    f"expected {expected_audio_frames}, got "
                    f"{prepared_audio_condition.shape[2]}."
                )
        use_text_cfg = self.text_cfg_scale != 1.0
        use_audio_cfg = self.audio_cfg_scale != 1.0
        use_any_cfg = use_text_cfg or use_audio_cfg
        if use_text_cfg and negative_context is None:
            raise ValueError("Text CFG requires a negative text context.")
        if use_any_cfg and prepared_audio_condition is not None:
            if prepared_audio_unconditional is None:
                raise ValueError(
                    "Streaming CFG requires an unconditional condition produced by "
                    "passing zero frame embeddings through AudioPack."
                )
            if prepared_audio_unconditional.shape != prepared_audio_condition.shape:
                raise ValueError(
                    "Conditional and unconditional prepared audio shapes must match: "
                    f"{tuple(prepared_audio_condition.shape)} vs "
                    f"{tuple(prepared_audio_unconditional.shape)}."
                )
        use_joint_cfg = use_text_cfg and self.audio_cfg_scale == self.text_cfg_scale
        need_audio_unconditional = use_any_cfg and not use_joint_cfg
        need_text_unconditional = use_text_cfg
        cuda_timing = bool(initial_latents.is_cuda)
        dit_started_event = (
            torch.cuda.Event(enable_timing=True) if cuda_timing else None
        )
        dit_finished_event = (
            torch.cuda.Event(enable_timing=True) if cuda_timing else None
        )
        recache_finished_event = (
            torch.cuda.Event(enable_timing=True) if cuda_timing else None
        )
        vae_finished_event = (
            torch.cuda.Event(enable_timing=True) if cuda_timing else None
        )
        dit_wall_started = time.perf_counter()
        if dit_started_event is not None:
            dit_started_event.record()

        if self.flowmap_schedule is not None:
            self.pipe.scheduler.set_flowmap_schedule(self.flowmap_schedule)
        else:
            self.pipe.scheduler.set_timesteps(
                self.num_inference_steps,
                shift=self.sigma_shift,
            )
        latents = initial_latents
        self.pipe.load_models_to_device(["dit"])
        timesteps = self.pipe.scheduler.timesteps
        # Upload schedules once. A pageable CPU scalar .to(cuda) in every
        # step otherwise synchronizes the previous graph / scheduler kernels.
        device_timesteps = timesteps.to(device=latents.device, dtype=torch.float32)
        model_timesteps = device_timesteps.to(dtype=self.pipe.torch_dtype)
        device_r_timesteps = model_r_timesteps = None
        if self.flowmap_schedule is not None:
            device_r_timesteps = self.pipe.scheduler.r_timesteps.to(
                device=latents.device, dtype=torch.float32)
            model_r_timesteps = device_r_timesteps.to(dtype=self.pipe.torch_dtype)
        if step_progress_bar is not None:
            timesteps = step_progress_bar(timesteps)
        chunk_graph = None
        use_chunk_graph = (
            self.config.use_cuda_graphs and latents.is_cuda and not use_any_cfg
        )
        if use_chunk_graph:
            # RoPE tables are plain CPU attributes, not registered buffers.
            # Upload once before capture; synchronous H2D copies cannot be captured.
            self.pipe.dit.freqs = tuple(f.to(latents.device) for f in self.pipe.dit.freqs)
        def conditional_forward(x, t, r):
            return self.pipe.dit(
                x, timestep=t, context=context, y=image_condition,
                prepared_audio_condition=prepared_audio_condition,
                streaming_state=self.dit_state, streaming_config=self.config,
                update_streaming_cache=False,
                **({"r_timestep": r} if r is not None else {}),
            )
        all_graph = use_chunk_graph
        if all_graph:
            from .clean_graph import prepare_graph_source
            from .denoise_graph import PersistentDenoiseGraph
            source, _, _ = prepare_graph_source(self.pipe.dit, self.config, self.dit_state,
                                                 latents, context)
            graph_key = ("all", id(self.pipe.dit), tuple(latents.shape), latents.dtype, latents.device,
                         tuple(context.shape), tuple(image_condition.shape),
                         tuple(prepared_audio_condition.shape) if prepared_audio_condition is not None else None,
                         self.flowmap_schedule is not None, tuple(vars(self.config).values()))
            first_r = model_r_timesteps[:1] if model_r_timesteps is not None else None
            if getattr(self.pipe, "_persistent_denoise_graph_key", None) != graph_key:
                self.pipe._persistent_denoise_graph = PersistentDenoiseGraph(
                    self.pipe.dit, self.config, source, latents, model_timesteps[:1], first_r,
                    context, image_condition, prepared_audio_condition)
                self.pipe._persistent_denoise_graph_key = graph_key
            else:
                self.pipe._persistent_denoise_graph.update(source, context, image_condition,
                                                           prepared_audio_condition)
            chunk_graph = self.pipe._persistent_denoise_graph
            del source
        timestep_iterator = iter(timesteps)
        for step_id, timestep in enumerate(timestep_iterator):
            model_timestep = model_timesteps[step_id:step_id + 1]
            model_r_timestep = None
            flowmap_kwargs = {}
            if self.flowmap_schedule is not None:
                r_timestep = device_r_timesteps[step_id]
                model_r_timestep = model_r_timesteps[step_id:step_id + 1]
                flowmap_kwargs["r_timestep"] = model_r_timestep
            conditional_velocity = (
                chunk_graph(latents, model_timestep, model_r_timestep)
                if chunk_graph is not None else
                conditional_forward(latents, model_timestep, model_r_timestep)
            )
            audio_unconditional_velocity = None
            text_unconditional_velocity = None
            if need_audio_unconditional:
                audio_unconditional_velocity = self.pipe.dit(
                    latents,
                    timestep=model_timestep,
                    context=context,
                    y=image_condition,
                    prepared_audio_condition=prepared_audio_unconditional,
                    streaming_state=self.audio_uncond_dit_state,
                    streaming_config=self.config,
                    update_streaming_cache=False,
                    **flowmap_kwargs,
                )
            if need_text_unconditional:
                text_unconditional_velocity = self.pipe.dit(
                    latents,
                    timestep=model_timestep,
                    context=negative_context,
                    y=image_condition,
                    prepared_audio_condition=prepared_audio_unconditional,
                    streaming_state=self.text_uncond_dit_state,
                    streaming_config=self.config,
                    update_streaming_cache=False,
                    **flowmap_kwargs,
                )
            velocity = combine_streaming_cfg_predictions(
                conditional_velocity,
                self.text_cfg_scale,
                self.audio_cfg_scale,
                audio_unconditional_prediction=audio_unconditional_velocity,
                text_unconditional_prediction=text_unconditional_velocity,
            )
            latents = self.pipe.scheduler.step(
                velocity,
                (device_timesteps[step_id] if self.flowmap_schedule is not None
                 else self.pipe.scheduler.timesteps[step_id]),
                latents,
                # The DiT receives its native dtype, but scheduler integration
                # must retain the float32 schedule value (for example 937.5).
                r_timestep=r_timestep if self.flowmap_schedule is not None else None,
            )

        dit_wall_finished = time.perf_counter()
        if dit_finished_event is not None:
            dit_finished_event.record()

        if self.use_kv_cache:
            # Rebuild persistent history from clean x0 with a distinct t=r=0 pass.
            self._cache_clean_chunk(
                latents,
                context,
                image_condition,
                prepared_audio_condition,
                self.dit_state,
            )
            if need_audio_unconditional:
                self._cache_clean_chunk(
                    latents,
                    context,
                    image_condition,
                    prepared_audio_unconditional,
                    self.audio_uncond_dit_state,
                )
            if need_text_unconditional:
                self._cache_clean_chunk(
                    latents,
                    negative_context,
                    image_condition,
                    prepared_audio_unconditional,
                    self.text_uncond_dit_state,
                )

        else:
            for state in (
                self.dit_state,
                self.audio_uncond_dit_state,
                self.text_uncond_dit_state,
            ):
                state.global_frame_offset += int(latents.shape[2])

        recache_wall_finished = time.perf_counter()
        if recache_finished_event is not None:
            recache_finished_event.record()
        emitted_latents = latents
        if self.decode_output:
            self.pipe.load_models_to_device(["vae"])
            frames, self.vae_state = self.pipe.vae.decode_streaming(
                emitted_latents,
                state=self.vae_state,
                device=self.pipe.device,
            )
        else:
            frames = latents.new_empty((0,))
        vae_wall_finished = time.perf_counter()
        if vae_finished_event is not None:
            vae_finished_event.record()
            torch.cuda.synchronize(initial_latents.device)
            timing_seconds = {
                "dit_denoise_seconds": dit_started_event.elapsed_time(
                    dit_finished_event
                ) / 1000.0,
                "dit_kv_recache_seconds": dit_finished_event.elapsed_time(
                    recache_finished_event
                ) / 1000.0,
                "vae_decode_seconds": recache_finished_event.elapsed_time(
                    vae_finished_event
                ) / 1000.0,
            }
        else:
            timing_seconds = {
                "dit_denoise_seconds": dit_wall_finished - dit_wall_started,
                "dit_kv_recache_seconds": (
                    recache_wall_finished - dit_wall_finished
                ),
                "vae_decode_seconds": (
                    vae_wall_finished - recache_wall_finished
                ),
            }
        return StreamingChunkOutput(
            frames=frames,
            latents=emitted_latents,
            timing_seconds=timing_seconds,
        )
