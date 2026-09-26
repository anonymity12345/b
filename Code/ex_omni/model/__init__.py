"""Ex-Omni model registrations."""

__all__ = [
    "LlavaHerQwen3ForCausalLM",
    "LlavaHerQwen3Model",
    "LlavaHerQwenConfig",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    from .language_model import llava_her_qwen

    return getattr(llava_her_qwen, name)
