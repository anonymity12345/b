from .contracts import FlowMapSchedule

__all__ = [
    "FlowMapDiscreteScheduler",
    "FlowMapSchedule",
    "FlowMatchScheduler",
]


def __getattr__(name):
    if name in {"FlowMapDiscreteScheduler", "FlowMatchScheduler"}:
        from . import flow_match

        return getattr(flow_match, name)
    raise AttributeError(name)
