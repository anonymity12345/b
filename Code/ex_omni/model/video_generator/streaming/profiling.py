"""Opt-in one-chunk profiler for performance investigations."""
from functools import wraps
import os
from pathlib import Path

_profile_calls = 0


def profile_video_chunk(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        global _profile_calls
        destination = os.environ.get('EX_OMNI_PROFILE_VIDEO')
        if not destination or int(os.environ.get('RANK', '0')) != 0:
            return function(*args, **kwargs)
        _profile_calls += 1
        if _profile_calls != 4:
            return function(*args, **kwargs)
        import torch
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            output = function(*args, **kwargs)
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix('.txt').write_text(
            profile.key_averages().table(sort_by='self_cuda_time_total', row_limit=50) + '\n' +
            profile.key_averages().table(sort_by='self_cpu_time_total', row_limit=50))
        profile.export_chrome_trace(str(path.with_suffix('.trace.json')))
        return output
    return wrapped
