"""Opt-in CUDA graph for repeated forwards within a single denoising chunk.

The graph never spans clean-cache updates or requests. Its input tensors and
history references therefore remain valid for its complete lifetime.
"""
import torch


class ChunkDenoiseGraph:
    def __init__(self, forward, latents, timestep, r_timestep):
        self.forward = forward
        self.latents = latents.clone()
        self.timestep = timestep.clone()
        self.r_timestep = None if r_timestep is None else r_timestep.clone()
        self.graph = torch.cuda.CUDAGraph()
        # One eager diffusion step has already warmed up this exact shape.
        torch.cuda.synchronize()
        with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
            try:
                self.output = self.forward(self.latents, self.timestep, self.r_timestep)
            except Exception:
                import traceback
                traceback.print_exc()
                raise

    def __call__(self, latents, timestep, r_timestep):
        self.latents.copy_(latents)
        self.timestep.copy_(timestep)
        if self.r_timestep is not None:
            self.r_timestep.copy_(r_timestep)
        self.graph.replay()
        return self.output


class PersistentDenoiseGraph:
    """Fixed-capacity FA3 history, with valid lengths updated between chunks.

    Only one shape is kept on the pipeline. Clean recaching remains eager and
    authoritative; each chunk copies its freshly prepared history into these
    private graph buffers. Padded KV entries are excluded by FA3 cache_seqlens.
    """
    def __init__(self, model, config, state, latents, timestep, r_timestep,
                 context, image, audio, *, resolution="full", collect_current=False):
        from .state import DiTStreamingState, LayerKVCache
        from ..models.wan_video_dit import _attention_backend, _runtime_args
        if _attention_backend() != "flash_attention_3":
            raise ValueError("Persistent denoise graphs currently require FA3")
        if state.text_context_cache is None or any(
                c is None or c.prepared_attention is None for c in state.layer_caches):
            raise ValueError("Persistent graphs require text/history caches warmed by an eager diffusion step")
        # A dialogue callback can first create this graph inside inference mode,
        # while the final video chunk updates it after that mode has exited.
        # Allocate reusable buffers as ordinary tensors in either case.
        with torch.inference_mode(False):
            self.model, self.config = model, config
            self.resolution, self.collect_current = resolution, collect_current
            self.latents, self.timestep = latents.clone(), timestep.clone()
            self.r_timestep = r_timestep.clone() if r_timestep is not None else None
            self.context, self.image = context.clone(), image.clone()
            self.audio = audio.clone() if audio is not None else None
            full_tokens = (latents.shape[-2] // model.patch_size[-2]) * (latents.shape[-1] // model.patch_size[-1])
            compressed_tokens = (latents.shape[-2] // config.compressed_patch_size[-2]) * (latents.shape[-1] // config.compressed_patch_size[-1])
            self.current_tokens = latents.shape[2] * (compressed_tokens if resolution == "compressed" else full_tokens)
            self.capacity = (config.full_chunk_limit * config.chunk_latent_frames
                             + config.first_chunk_latent_frames + latents.shape[2]) * full_tokens
            self.capacity += config.max_compressed_latent_frames * compressed_tokens
            self.length = torch.zeros(latents.shape[0], device=latents.device, dtype=torch.int32)
            self.indices = torch.empty(self.current_tokens, device=latents.device, dtype=torch.int64)
            memo_key, metadata = next(iter(state.attention_metadata_cache.items()))
            self.query_freqs = metadata[0].clone()
            self.key_freqs = metadata[1].new_empty(self.capacity, *metadata[1].shape[1:])
            self.state = DiTStreamingState(global_frame_offset=state.global_frame_offset)
            self.state.attention_metadata_cache[memo_key] = (self.query_freqs, self.key_freqs, None, None)
            for source in state.layer_caches:
                key = source.prepared_attention[1]
                # FA3 excludes padding scores, but NaN values in its last partial
                # tile can still propagate through 0 * NaN. Initialize all padding.
                graph_key = key.new_zeros(key.shape[0], self.capacity, key.shape[2])
                graph_value = torch.zeros_like(graph_key)
                self.state.layer_caches.append(LayerKVCache(
                    key=key.new_empty(key.shape[0], 0, key.shape[2]),
                    value=key.new_empty(key.shape[0], 0, key.shape[2]),
                    positions=source.positions.new_empty(0, 3),
                    prepared_attention=(self.key_freqs, graph_key, graph_value, self.length, self.indices),
                    current_key_output=(key.new_empty(key.shape[0], self.current_tokens, key.shape[2])
                                        if collect_current else None)))
            _, embedded, text_kv = state.text_context_cache
            self.state.text_context_cache = (self.context, embedded.clone(),
                                            tuple((k.clone(), v.clone()) for k,v in text_kv))
        self.update(state, context, image, audio)
        # Warm the KV-cache attention entry point before capture.
        self.forward()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, capture_error_mode='thread_local'):
            try:
                self.output = self.forward()
            except Exception:
                import traceback
                traceback.print_exc()
                raise

    def update(self, state, context, image, audio):
        metadata = next(iter(state.attention_metadata_cache.values()))
        length = state.layer_caches[0].prepared_attention[1].shape[1]
        if length > self.capacity:
            raise ValueError(f'Graph KV capacity exceeded: {length} > {self.capacity}')
        self.length.fill_(length)
        self.indices.copy_(torch.arange(length-self.current_tokens, length,
                                        device=self.indices.device, dtype=self.indices.dtype))
        self.query_freqs.copy_(metadata[0])
        self.key_freqs[-self.current_tokens:].copy_(metadata[1][-self.current_tokens:])
        for src, dst in zip(state.layer_caches, self.state.layer_caches):
            for a,b in zip(src.prepared_attention[1:3], dst.prepared_attention[1:3]):
                b[:, :length].copy_(a)
        self.context.copy_(context)
        self.image.copy_(image)
        if self.audio is not None:
            self.audio.copy_(audio)
        self.state.text_context_cache[1].copy_(state.text_context_cache[1])
        for src, dst in zip(state.text_context_cache[2], self.state.text_context_cache[2]):
            for a,b in zip(src,dst):
                b.copy_(a)

    def forward(self):
        return self.model(self.latents, timestep=self.timestep, context=self.context,
                          y=self.image, prepared_audio_condition=self.audio,
                          streaming_state=self.state, streaming_config=self.config,
                          update_streaming_cache=False, streaming_cache_resolution=self.resolution,
                          capture_cache_only=self.collect_current,
                          **({'r_timestep':self.r_timestep} if self.r_timestep is not None else {}))

    def __call__(self, latents, timestep, r_timestep):
        self.latents.copy_(latents)
        self.timestep.copy_(timestep)
        if self.r_timestep is not None:
            self.r_timestep.copy_(r_timestep)
        self.graph.replay()
        return self.output


class PersistentDenoiseTailGraph(PersistentDenoiseGraph):
    """Capture all steps after the eager history-preparation step together."""
    is_tail_graph = True

    def __init__(self, *args, source_timesteps, target_timesteps, scheduler, **kwargs):
        if target_timesteps is None:
            raise ValueError('Tail graph requires a flow-map schedule')
        super().__init__(*args, **kwargs)
        self.source_times = source_timesteps[1:].clone()
        self.target_times = target_timesteps[1:].clone()
        self.model_source_times = self.source_times.to(self.timestep.dtype)
        self.model_target_times = self.target_times.to(self.r_timestep.dtype)
        self.scheduler = scheduler
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, capture_error_mode='thread_local'):
            try:
                for index in range(len(self.source_times)):
                    self.timestep.copy_(self.model_source_times[index:index+1])
                    self.r_timestep.copy_(self.model_target_times[index:index+1])
                    velocity = self.forward()
                    updated = scheduler.step(velocity, self.source_times[index],
                                             self.latents, r_timestep=self.target_times[index])
                    self.latents.copy_(updated)
            except Exception:
                import traceback
                traceback.print_exc()
                raise
        self.graph = graph
        self.output = self.latents

    def update_schedule(self, source, target):
        self.source_times.copy_(source[1:])
        self.target_times.copy_(target[1:])
        self.model_source_times.copy_(self.source_times)
        self.model_target_times.copy_(self.target_times)

    def __call__(self, latents, timestep, r_timestep):
        result = super().__call__(latents, timestep, r_timestep)
        # The returned clean latents outlive graph execution (async VAE / IPC).
        return result.clone()
