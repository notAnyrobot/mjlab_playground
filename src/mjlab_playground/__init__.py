"""A collection of tasks built on mjlab."""

from pathlib import Path

MJLAB_PLAYGROUND_SRC_PATH: Path = Path(__file__).parent

from mjlab_playground.tasks import *  # noqa: E402, F401, F403
