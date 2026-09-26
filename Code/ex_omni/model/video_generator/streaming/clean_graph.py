"""Experimental captured clean-history forwards; cache policy stays explicit."""
import os
import torch
from .state import DiTStreamingState, LayerKVCache
from .causal_attention import (make_3d_positions, prepare_streaming_attention_metadata,
                               prepare_cache_update_selection, update_full_cache, update_compressed_cache)
from .denoise_graph import PersistentDenoiseGraph


def prepare_graph_source(model, config, state, latents, context, resolution="full"):
    from ..models.wan_video_dit import rope_apply, _runtime_args
    from ex_omni.distributed.sequence_parallel import get_sp_group
    if getattr(_runtime_args(), 'sequence_parallel_mode', 'gather') != 'gather':
        raise ValueError('Explicit graph history preparation requires gather parallelism')
    model._prepare_inference_text_context(context, state)
    group = get_sp_group()
    world, rank = group.world_size, group.rank_in_group
    compressed = resolution == 'compressed'
    patch = config.compressed_patch_size if compressed else model.patch_size
    f, h, w = [size // stride for size, stride in zip(latents.shape[2:], patch)]
    positions = make_3d_positions(state.global_frame_offset, (f,h,w), latents.device)
    padded = positions
    if positions.shape[0] % world:
        padded = torch.cat((positions, positions.new_zeros(world-positions.shape[0]%world,3)))
    local_positions = padded.chunk(world)[rank]
    metadata = prepare_streaming_attention_metadata(
        state.layer_caches[0], local_positions, positions, model.freqs, config,
        compressed=compressed)
    source = DiTStreamingState(global_frame_offset=state.global_frame_offset)
    source.text_context_cache = state.text_context_cache
    key = (state.global_frame_offset, f,h,w,world,compressed,tuple(vars(config).values()))
    source.attention_metadata_cache[key] = metadata
    _, frequencies, full_indices, compressed_indices = metadata
    for cache in state.layer_caches:
        if os.environ.get('EX_OMNI_FUSED_HISTORY') == '1':
            from .history_kernel import pack_history
            rotated, packed_value = pack_history(cache, full_indices, compressed_indices,
                                                 frequencies, model.blocks[0].num_heads,
                                                 include_sink=config.reference_sink_attention)
            source.layer_caches.append(LayerKVCache(
                key=cache.key, value=cache.value, positions=cache.positions,
                prepared_attention=(frequencies, rotated, packed_value)))
            continue
        keys, values = [], []
        if config.reference_sink_attention:
            keys.append(cache.sink_key)
            values.append(cache.sink_value)
        if compressed_indices is not None and compressed_indices.numel():
            keys.append(cache.compressed_key.index_select(1,compressed_indices))
            values.append(cache.compressed_value.index_select(1,compressed_indices))
        if full_indices is not None and full_indices.numel():
            keys.append(cache.key.index_select(1,full_indices))
            values.append(cache.value.index_select(1,full_indices))
        current = cache.key.new_zeros(latents.shape[0], positions.shape[0], model.dim)
        keys.append(current); values.append(current)
        rotated = rope_apply(torch.cat(keys,dim=1), frequencies, model.blocks[0].num_heads)
        source.layer_caches.append(LayerKVCache(
            key=cache.key,value=cache.value,positions=cache.positions,
            prepared_attention=(frequencies,rotated,torch.cat(values,dim=1))))
    return source, positions, frequencies


def cache_clean_graph(session, latents, context, image, audio, state):
    from ..models.wan_video_dit import rope_apply, _runtime_args
    from ex_omni.distributed.sequence_parallel import get_sp_group
    model, config = session.pipe.dit, session.config
    model._prepare_inference_text_context(context, state)
    args = _runtime_args()
    if getattr(args, 'sequence_parallel_mode', 'gather') != 'gather':
        raise ValueError('Clean graphs currently require gather sequence parallelism')
    group = get_sp_group()
    world, rank = group.world_size, group.rank_in_group
    shape_key = (tuple(latents.shape), latents.dtype, latents.device, tuple(context.shape),
                 tuple(image.shape), tuple(audio.shape) if audio is not None else None,
                 tuple(vars(config).values()))
    if getattr(session.pipe, '_clean_graph_shape', None) != shape_key:
        session.pipe._clean_graphs = {}
        session.pipe._clean_graph_shape = shape_key
    zero_time = latents.new_zeros(latents.shape[0])
    for resolution in ('compressed', 'full'):
        compressed = resolution == 'compressed'
        source, positions, frequencies = prepare_graph_source(
            model, config, state, latents, context, resolution)
        selection = prepare_cache_update_selection(state.layer_caches[0], positions, config,
                                                     compressed=compressed, contiguous_suffix=True)
        graph = session.pipe._clean_graphs.get(resolution)
        if graph is None:
            graph = PersistentDenoiseGraph(model,config,source,latents,zero_time,zero_time,
                                           context,image,audio,resolution=resolution,collect_current=True)
            session.pipe._clean_graphs[resolution] = graph
        else:
            graph.update(source,context,image,audio)
        length = frequencies.shape[0]
        graph(latents,zero_time,zero_time)
        update = update_compressed_cache if compressed else update_full_cache
        for index, (cache, buffers) in enumerate(zip(state.layer_caches,graph.state.layer_caches)):
            current_value = buffers.prepared_attention[2][:,length-positions.shape[0]:length]
            updated = update(cache,buffers.current_key_output,current_value,
                             positions,config,selection=selection)
            previous = cache.compressed_key if compressed else cache.key
            previous_length = previous.shape[1] if previous is not None else 0
            if len(selection) == 3 and selection[2] is not None and selection[2] >= previous_length:
                # A retained-current suffix can alias graph outputs; history must
                # survive the next replay. Concatenated/indexed history already owns storage.
                if compressed:
                    updated.compressed_key = updated.compressed_key.clone()
                    updated.compressed_value = updated.compressed_value.clone()
                else:
                    updated.key = updated.key.clone()
                    updated.value = updated.value.clone()
            state.layer_caches[index] = updated
        state.attention_metadata_cache.clear()
        if not compressed:
            state.global_frame_offset += latents.shape[2] // model.patch_size[0]
