#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#

from dataclasses import dataclass
from typing import Optional, Union

import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.utils import BaseOutput, logging

from .contracts import FlowMapSchedule

try:
    from diffusers.schedulers.scheduling_flow_map_euler_discrete import FlowMapEulerDiscreteSchedulerOutput
except ModuleNotFoundError:
    # Diffusers 0.33 does not yet ship the output container used by AnyFlow.
    # This shape-only compatibility adapter leaves the scheduler implementation
    # and all algorithmic behavior unchanged.
    @dataclass
    class FlowMapEulerDiscreteSchedulerOutput(BaseOutput):
        prev_sample: torch.FloatTensor

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class FlowMapDiscreteScheduler(SchedulerMixin, ConfigMixin):

    @register_to_config
    def __init__(
        self,
        num_train_timesteps: int = 1000,
        shift: float = 1.0,
        weight_type: str = 'gaussian',
    ):
        self.set_timesteps(num_train_timesteps, device='cpu')
        self.set_train_weight(weight_type)

    def adaptive_weighting(self, loss, p=1.0, eps=1e-3):
        weight = 1.0 / torch.pow(loss.detach() + eps, p)
        return weight * loss

    def set_train_weight(self, weight_type):
        if self.config.weight_type == 'gaussian':
            x = self.timesteps
            y = torch.exp(-2 * ((x - self.config.num_train_timesteps / 2) / self.config.num_train_timesteps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (self.config.num_train_timesteps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing
        elif self.config.weight_type == 'beta08':
            t = self.timesteps / self.config.num_train_timesteps
            y = (t ** 1.0) * ((1 - t) ** 0.5)
            self.linear_timesteps_weights = y * (self.config.num_train_timesteps / y.sum())
        elif self.config.weight_type == 'uniform':
            self.linear_timesteps_weights = torch.ones_like(self.timesteps)
        else:
            raise ValueError(f'Invalid weight type: {weight_type}')

    @torch.no_grad()
    def get_train_weight(self, timesteps):
        timestep_id = torch.argmin((self.timesteps.unsqueeze(1) - timesteps.flatten().unsqueeze(0).to(self.timesteps.device)).abs(), dim=0).reshape(timesteps.shape)  # noqa: E501
        weights = self.linear_timesteps_weights[timestep_id]
        return weights.to(timesteps.device)

    def scale_noise(
        self,
        sample: torch.FloatTensor,
        timestep: Union[float, torch.FloatTensor],
        noise: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        timestep = timestep.to(device=sample.device, dtype=sample.dtype)

        timestep = timestep / self.config.num_train_timesteps
        timestep = timestep.view(*timestep.shape, *([1] * (noise.ndim - timestep.ndim)))
        sample = timestep * noise + (1.0 - timestep) * sample
        return sample

    def apply_shift(self, sigmas):
        """Apply shift transformation to sigmas/timesteps."""
        if self.config.shift == 1.0:
            return sigmas
        return self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)

    def set_timesteps(
        self,
        num_inference_steps: Optional[int] = None,
        device: Union[str, torch.device] = None,
    ):
        timesteps = torch.linspace(1.0, 0.0, num_inference_steps + 1, dtype=torch.float64, device=device)[:-1]
        timesteps = self.apply_shift(timesteps)

        self.timesteps = timesteps * self.config.num_train_timesteps

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: Optional[Union[float, torch.FloatTensor]] = None,
        sample: Optional[Union[float, torch.FloatTensor]] = None,
        r_timestep: Optional[Union[float, torch.FloatTensor]] = None,
        return_dict: bool = True
    ):
        timestep = timestep / self.config.num_train_timesteps
        r_timestep = r_timestep / self.config.num_train_timesteps
        timestep = timestep.view(*timestep.shape, *([1] * (model_output.ndim - timestep.ndim)))
        r_timestep = r_timestep.view(*r_timestep.shape, *([1] * (model_output.ndim - r_timestep.ndim)))
        prev_sample = sample - (timestep - r_timestep) * model_output

        if not return_dict:
            return (prev_sample.to(model_output.dtype),)

        return FlowMapEulerDiscreteSchedulerOutput(prev_sample=prev_sample.to(model_output.dtype))


class FlowMatchScheduler():

    def __init__(self, num_inference_steps=100, num_train_timesteps=1000, shift=3.0, sigma_max=1.0, sigma_min=0.003/1.002, inverse_timesteps=False, extra_one_step=False, reverse_sigmas=False):
        self.num_train_timesteps = num_train_timesteps
        self.shift = shift
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.inverse_timesteps = inverse_timesteps
        self.extra_one_step = extra_one_step
        self.reverse_sigmas = reverse_sigmas
        self.set_timesteps(num_inference_steps)


    def set_timesteps(self, num_inference_steps=100, denoising_strength=1.0, training=False, shift=None):
        if shift is not None:
            self.shift = shift
        sigma_start = self.sigma_min + (self.sigma_max - self.sigma_min) * denoising_strength
        if self.extra_one_step:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps + 1)[:-1]
        else:
            self.sigmas = torch.linspace(sigma_start, self.sigma_min, num_inference_steps)
        if self.inverse_timesteps:
            self.sigmas = torch.flip(self.sigmas, dims=[0])
        self.sigmas = self.shift * self.sigmas / (1 + (self.shift - 1) * self.sigmas)
        if self.reverse_sigmas:
            self.sigmas = 1 - self.sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps
        if training:
            x = self.timesteps
            y = torch.exp(-2 * ((x - num_inference_steps / 2) / num_inference_steps) ** 2)
            y_shifted = y - y.min()
            bsmntw_weighing = y_shifted * (num_inference_steps / y_shifted.sum())
            self.linear_timesteps_weights = bsmntw_weighing

    def set_flowmap_schedule(self, schedule: FlowMapSchedule):
        if schedule.num_train_timesteps != self.num_train_timesteps:
            raise ValueError(
                "Schedule and scheduler training horizons differ: "
                f"{schedule.num_train_timesteps} vs {self.num_train_timesteps}."
            )
        self.timesteps, self.r_timesteps = schedule.tensors(dtype=torch.float32)
        self.sigmas = self.timesteps / float(self.num_train_timesteps)
        self._flowmap_schedule = schedule
        self._flowmap_runtime = FlowMapDiscreteScheduler(
            num_train_timesteps=self.num_train_timesteps,
            shift=schedule.shift,
        )

    def timestep_index(self, timestep) -> int:
        timestep_tensor = torch.as_tensor(
            timestep,
            dtype=self.timesteps.dtype,
            device=self.timesteps.device,
        )
        if timestep_tensor.numel() != 1:
            raise ValueError("FlowMatchScheduler expects one shared batch timestep.")
        return int(torch.argmin((self.timesteps - timestep_tensor.reshape(())).abs()))

    def sigma_for_timestep(self, timestep, device=None, dtype=None):
        sigma = self.sigmas[self.timestep_index(timestep)]
        if device is not None or dtype is not None:
            sigma = sigma.to(device=device, dtype=dtype)
        return sigma


    def step(
        self,
        model_output,
        timestep,
        sample,
        to_final=False,
        r_timestep=None,
        **kwargs,
    ):
        if r_timestep is not None:
            # Omni runtime wrapper: values such as 937.5 round in bfloat16 near
            # t=1000. Run the official transition in fp32 and restore dtype.
            t = torch.as_tensor(timestep, device=sample.device, dtype=torch.float32)
            r = torch.as_tensor(
                r_timestep, device=sample.device, dtype=torch.float32
            )
            result = self._flowmap_runtime.step(
                model_output.to(dtype=torch.float32),
                timestep=t,
                sample=sample.to(dtype=torch.float32),
                r_timestep=r,
                return_dict=False,
            )[0]
            return result.to(dtype=sample.dtype)
        timestep_id = self.timestep_index(timestep)
        sigma = self.sigmas[timestep_id]
        if to_final or timestep_id + 1 >= len(self.timesteps):
            sigma_ = 1 if (self.inverse_timesteps or self.reverse_sigmas) else 0
        else:
            sigma_ = self.sigmas[timestep_id + 1]
        prev_sample = sample + model_output * (sigma_ - sigma)
        return prev_sample
    

    def return_to_timestep(self, timestep, sample, sample_stablized):
        timestep_id = self.timestep_index(timestep)
        sigma = self.sigmas[timestep_id]
        model_output = (sample - sample_stablized) / sigma
        return model_output
    
    
    def add_noise(self, original_samples, noise, timestep):
        timestep_id = self.timestep_index(timestep)
        sigma = self.sigmas[timestep_id]
        sample = (1 - sigma) * original_samples + sigma * noise
        return sample
    

    def training_target(self, sample, noise, timestep):
        target = noise - sample
        return target
    

    def training_weight(self, timestep):
        timestep_id = self.timestep_index(timestep)
        weights = self.linear_timesteps_weights[timestep_id]
        return weights
