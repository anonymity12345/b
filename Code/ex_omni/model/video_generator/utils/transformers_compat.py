"""Compatibility helpers for optional Transformers dependencies."""


def disable_optional_sklearn():
    """Avoid importing a broken optional sklearn install through Transformers."""
    try:
        from transformers.utils import import_utils
    except Exception:
        return

    import_utils._sklearn_available = False
