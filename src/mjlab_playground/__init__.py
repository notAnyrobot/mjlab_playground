"""Repository-local extensions and tasks built on mjlab."""

import sys
from pathlib import Path

MJLAB_PLAYGROUND_SRC_PATH: Path = Path(__file__).parent

__all__ = ["MJLAB_PLAYGROUND_SRC_PATH"]

if "mjlab" in sys.modules:
    from mjlab_playground import tasks as tasks  # noqa: F401
