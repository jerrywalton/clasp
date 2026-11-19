"""Policy as a pure function. Business decisions live here, never in prompts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Policy:
    """A named, total pure function: inputs -> decision label."""
    name: str
    decide: Callable[[dict], str]
