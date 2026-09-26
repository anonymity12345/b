"""Lazy Hugging Face weight resolution."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping


HF_PREFIX = "hf://"


class RemoteWeightUnavailableError(FileNotFoundError):
    """A configured Hugging Face artifact is not available yet."""


def is_hf_reference(value: object) -> bool:
    return isinstance(value, str) and value.startswith(HF_PREFIX)


def parse_hf_reference(value: str) -> tuple[str, str | None]:
    if not is_hf_reference(value):
        raise ValueError(f"not a Hugging Face reference: {value!r}")
    body = value[len(HF_PREFIX) :]
    repo_id, separator, filename = body.partition("::")
    if repo_id.count("/") != 1 or not all(repo_id.split("/")):
        raise ValueError(
            "Hugging Face references must use hf://OWNER/REPO or "
            "hf://OWNER/REPO::path/in/repo"
        )
    if separator and not filename:
        raise ValueError(f"missing file path in Hugging Face reference: {value}")
    return repo_id, filename or None


@lru_cache(maxsize=None)
def materialize_hf_reference(value: str) -> str:
    """Resolve one ``hf://`` reference into the shared Hub cache."""
    repo_id, filename = parse_hf_reference(value)
    try:
        from huggingface_hub import hf_hub_download, snapshot_download

        if filename is None:
            return snapshot_download(repo_id=repo_id)
        if filename.endswith("/"):
            snapshot = snapshot_download(
                repo_id=repo_id, allow_patterns=[f"{filename}*"]
            )
            return str(Path(snapshot) / filename)
        return hf_hub_download(repo_id=repo_id, filename=filename)
    except Exception as exc:
        message = str(exc).lower()
        if filename and (
            "entry not found" in message
            or "404" in message
            or "not found" in message
        ):
            raise RemoteWeightUnavailableError(
                f"{repo_id}/{filename} is not available on Hugging Face"
            ) from exc
        raise


@lru_cache(maxsize=None)
def materialize_pretrained_reference(value: str) -> str:
    """Resolve a local path, ``hf://`` URI, or ordinary Hub repo ID."""
    if is_hf_reference(value):
        return materialize_hf_reference(value)
    expanded = Path(value).expanduser()
    if expanded.exists():
        return str(expanded.resolve())
    if value.count("/") == 1 and not value.startswith(("./", "../", "/")):
        from huggingface_hub import snapshot_download

        return snapshot_download(repo_id=value)
    return value


def materialize_hf_references(value: Any) -> Any:
    """Recursively materialize explicit model references without mutating input."""
    if isinstance(value, Mapping):
        return {
            key: materialize_hf_references(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [materialize_hf_references(item) for item in value]
    if is_hf_reference(value):
        return materialize_hf_reference(value)
    return value
