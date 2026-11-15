"""Small shared file and provider-error helpers for offline-tested batches."""

import os
import tempfile
from pathlib import Path


def atomic_write(path: Path, content: str) -> None:
    """Flush a temporary file beside its target, then atomically replace it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file:
            temp_path = Path(file.name)
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        temp_path.replace(path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def is_transient_error(error: Exception) -> bool:
    status = getattr(error, "status_code", None)
    return (
        isinstance(error, (TimeoutError, ConnectionError))
        or status in (408, 429)
        or (isinstance(status, int) and 500 <= status < 600)
        # LiteLLM's connection error inherits from the provider SDK, not OSError.
        or any(cls.__name__ == "APIConnectionError" for cls in type(error).__mro__)
    )
