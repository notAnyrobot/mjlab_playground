"""A collection of tasks built on mjlab."""

from pathlib import Path

MJLAB_PLAYGROUND_SRC_PATH: Path = Path(__file__).parent

from mjlab_playground.getup.config.go1 import *  # noqa: F401, F403
from mjlab_playground.getup.config.t1 import *  # noqa: F401, F403
