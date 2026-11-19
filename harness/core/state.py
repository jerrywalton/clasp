"""Core entities: State and Transition. Pure data, no behavior beyond
what's needed to describe the graph. Validation lives in app/builder.py."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass(frozen=True)
class StateDef:
    """One node of the agent's control plane.

    Exactly one work source for non-terminal states:
      - prompt_template set  -> model path (needs output_schema)
      - handler set          -> pure-function path (code, not the model)
      - neither set          -> passthrough (emits event "next")
    Terminal states do no work.
    """
    name: str
    terminal: bool = False
    prompt_template: str | None = None          # string.Template ($var) style
    output_schema: dict | None = None
    handler: Callable[[dict], dict] | None = None  # ctx -> envelope
    tools: tuple[str, ...] = ()
    entry_gates: tuple[str, ...] = ()
    exit_gates: tuple[str, ...] = ()
    max_corrections: int = 2
    # Declared context writes (#11): when non-empty, the state's
    # envelope data may only write these context keys. Empty means
    # undeclared (unrestricted) — dataflow discipline is declared
    # per state, like capability.
    writes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Transition:
    """A declared edge. Guards are pure predicates over the context dict."""
    from_state: str
    event: str
    to_state: str
    guard: Callable[[dict], bool] | None = None


@dataclass(frozen=True)
class PausePolicy:
    """A declared human-in-the-loop pause on one state (v1.1, #10).

    The pause is a policy, not a behavior: different deployments
    declare different options, all validated at build time. When the
    runner reaches the state it writes PAUSE_REQUESTED to the log and
    stops; the run resumes only when a PAUSE_DECIDED event from the
    declared approver is in the same log. The channel is transport
    only (a CLI prompt, a Slack message, a console interrupt): the
    decision counts because it is in the ledger, not because of
    where it was typed.
    """
    state: str
    approver: str
    channel: str = "cli"
    decisions: tuple[str, ...] = ("approve", "deny")
    timeout_ms: int | None = None
