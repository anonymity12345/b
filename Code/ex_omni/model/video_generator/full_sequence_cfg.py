"""Exact two-device execution of full-sequence equal-scale CFG branches."""
from concurrent.futures import ThreadPoolExecutor
import copy
import torch


class FullSequenceCFGParallel:
    def __init__(self, model, device, negative_context, image, audio, extra):
        if torch.is_grad_enabled() or model.training:
            raise ValueError('Parallel full-sequence CFG requires inference mode')
        self.primary = model
        self.primary_device = next(model.parameters()).device
        self.device = torch.device(device)
        if self.device == self.primary_device:
            raise ValueError('CFG branches must use different devices')
        self.replica = copy.deepcopy(model).to(self.device)
        self.replica.freqs = tuple(value.to(self.device) for value in model.freqs)
        values = {**negative_context, **image, **audio, **extra}
        self.negative = {key: value.to(self.device) if isinstance(value, torch.Tensor) else value
                         for key, value in values.items()}
        self.pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='full-sequence-cfg')

    @staticmethod
    def _forward(model, device, latents, options):
        with torch.inference_mode(), torch.cuda.device(device):
            output = model(latents, **options)
            torch.cuda.synchronize(device)
            return output

    def __call__(self, latents, timestep, positive):
        secondary_latents = latents.to(self.device)
        negative = dict(self.negative, timestep=timestep.to(self.device))
        # Copies must finish before the independent worker starts consuming them.
        torch.cuda.synchronize(self.device)
        first = self.pool.submit(self._forward, self.primary, self.primary_device,
                                 latents, dict(positive, timestep=timestep))
        second = self.pool.submit(self._forward, self.replica, self.device,
                                  secondary_latents, negative)
        return first.result(), second.result().to(self.primary_device)

    def close(self):
        self.pool.shutdown(wait=True)
        self.replica = None
        self.negative = None
