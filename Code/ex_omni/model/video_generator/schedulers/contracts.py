"""Dependency-light AnyFlow schedule contract for configuration code."""

from dataclasses import dataclass
from typing import Tuple


ANYFLOW_INFERENCE_STEPS = (2, 4, 8, 16, 32, 50)


@dataclass(frozen=True)
class FlowMapSchedule:
    source_timesteps: Tuple[float, ...]
    target_timesteps: Tuple[float, ...]
    num_train_timesteps: int = 1000
    shift: float = 5.0

    def __post_init__(self):
        sources = tuple(float(value) for value in self.source_timesteps)
        targets = tuple(float(value) for value in self.target_timesteps)
        if not sources or len(sources) != len(targets):
            raise ValueError(
                "Flow-map source and target schedules must have equal non-zero length."
            )
        if self.num_train_timesteps <= 0:
            raise ValueError("num_train_timesteps must be positive.")
        if self.shift <= 0:
            raise ValueError("Flow-map shift must be positive.")
        if any(left <= right for left, right in zip(sources, sources[1:])):
            raise ValueError("Flow-map source timesteps must be strictly descending.")
        if any(target < 0 for target in targets):
            raise ValueError("Flow-map target timesteps must be non-negative.")
        if any(source <= target for source, target in zip(sources, targets)):
            raise ValueError("Each flow-map transition must move from t to a smaller r.")
        if targets[:-1] != sources[1:]:
            raise ValueError("Flow-map transitions must form one contiguous schedule.")
        if targets[-1] != 0.0:
            raise ValueError("Flow-map schedule must terminate at r=0.")
        if sources[0] > self.num_train_timesteps:
            raise ValueError("Flow-map timestep exceeds the training horizon.")
        object.__setattr__(self, "source_timesteps", sources)
        object.__setattr__(self, "target_timesteps", targets)
        object.__setattr__(self, "shift", float(self.shift))

    @classmethod
    def shifted(
        cls,
        num_inference_steps: int = 8,
        *,
        num_train_timesteps: int = 1000,
        shift: float = 5.0,
    ):
        if num_inference_steps not in ANYFLOW_INFERENCE_STEPS:
            raise ValueError(
                "omniavatar-anyflow-v1 supports 2, 4, 8, 16, 32, or 50 inference steps"
            )
        if num_train_timesteps != 1000 or shift != 5.0:
            raise ValueError(
                "omniavatar-anyflow-v1 requires a shift-5 schedule over "
                "1000 training timesteps"
            )
        sources = tuple(
            num_train_timesteps
            * (shift * (1.0 - step / num_inference_steps))
            / (1.0 + (shift - 1.0) * (1.0 - step / num_inference_steps))
            for step in range(num_inference_steps)
        )
        return cls(
            sources,
            sources[1:] + (0.0,),
            num_train_timesteps=num_train_timesteps,
            shift=shift,
        )

    @property
    def model_timesteps(self) -> Tuple[float, ...]:
        return self.source_timesteps

    def tensors(self, device=None, dtype=None):
        import torch

        dtype = dtype or torch.float32
        return (
            torch.tensor(self.source_timesteps, device=device, dtype=dtype),
            torch.tensor(self.target_timesteps, device=device, dtype=dtype),
        )
