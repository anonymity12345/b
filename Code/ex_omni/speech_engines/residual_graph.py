"""Fixed-shape CUDA graphs for the sequential RVQ predictor.

Only the predictor forward is captured. Sampling, its RNG, repetition policy,
and streaming callbacks stay in the original generator. Graph outputs are
borrowed until the same codebook is called again, just like a scratch buffer.
The native speech service processes requests sequentially.
"""
from __future__ import annotations

import torch


class ResidualCodebookGraphs:
    def __init__(self, forward, *, max_graphs=15):
        self.forward = forward
        self.max_graphs = max_graphs
        self.graphs = {}

    @torch.inference_mode()
    def __call__(self, inputs_embeds, codebook_index, *, past_key_values=None, use_cache=False):
        if not inputs_embeds.is_cuda or not use_cache or inputs_embeds.shape[0] != 1:
            return self.forward(inputs_embeds, codebook_index,
                                past_key_values=past_key_values, use_cache=use_cache)
        cache_shapes = tuple(tuple(t.shape) for pair in (past_key_values or ()) for t in pair)
        key = (codebook_index, tuple(inputs_embeds.shape), inputs_embeds.dtype,
               inputs_embeds.device, cache_shapes)
        if key not in self.graphs:
            if len(self.graphs) >= self.max_graphs:
                return self.forward(inputs_embeds, codebook_index,
                                    past_key_values=past_key_values, use_cache=use_cache)
            static_input = inputs_embeds.clone()
            static_past = (tuple(tuple(t.clone() for t in pair) for pair in past_key_values)
                           if past_key_values is not None else None)
            stream = torch.cuda.Stream(device=inputs_embeds.device)
            stream.wait_stream(torch.cuda.current_stream(inputs_embeds.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.forward(static_input, codebook_index,
                                 past_key_values=static_past, use_cache=True)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                output = self.forward(static_input, codebook_index,
                                      past_key_values=static_past, use_cache=True)
            torch.cuda.current_stream(inputs_embeds.device).wait_stream(stream)
            self.graphs[key] = (graph, static_input, static_past, output)
        graph, static_input, static_past, output = self.graphs[key]
        static_input.copy_(inputs_embeds)
        if static_past is not None:
            for destination, source in zip(static_past, past_key_values):
                for dst, src in zip(destination, source):
                    dst.copy_(src)
        graph.replay()
        return output
