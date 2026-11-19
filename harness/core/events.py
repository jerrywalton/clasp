"""Domain events: the append-only audit log. The log is the system of record;
control state is a fold over it, and so is data context at committed
boundaries: events are atomic and self-contained (RUN_STARTED holds the
initial context, each TRANSITION the delta its step committed), so the
fold needs nothing outside the log. Declared redacted keys are the one
exception — their values are stored as hashes, never in clear. Events
are versioned for long-lived replay."""

from __future__ import annotations

from dataclasses import dataclass, field

SCHEMA_VERSION = 1

# Event types
RUN_STARTED = "run_started"
STATE_ENTERED = "state_entered"
MODEL_CALLED = "model_called"
MODEL_FAILED = "model_failed"
GATE_VERDICT = "gate_verdict"
TOOL_INVOKED = "tool_invoked"
TOOL_INTENDED = "tool_intended"
LEDGER_POSTED = "ledger_posted"
BUDGET_EXHAUSTED = "budget_exhausted"
ESCALATED = "escalated"
TRANSITION = "transition"
STEP_COMMITTED = "step_committed"
CONTEXT_CHECKPOINT = "context_checkpoint"
PAUSE_REQUESTED = "pause_requested"
PAUSE_DECIDED = "pause_decided"
COMPLETED = "completed"

# Team-scale events (orchestration, v1.2 — docs/ORCHESTRATION-DESIGN-
# 2026-10-31.md). The Orchestrator is the single writer of the team
# log; member runs live in sub-ledgers (their own execution ids in the
# same store) whose digests the MEMBER_FINISHED events carry. A
# handoff is an effect: HANDOFF_INTENDED records intent before it
# executes, HANDOFF_COMMITTED the outcome after it lands — the
# TOOL_INTENDED pattern at team scale.
TEAM_STARTED = "team_started"
MEMBER_STARTED = "member_started"
MEMBER_FINISHED = "member_finished"
HANDOFF_INTENDED = "handoff_intended"
HANDOFF_COMMITTED = "handoff_committed"
HANDOFF_REFUSED = "handoff_refused"
TEAM_PAUSED = "team_paused"
TEAM_DECIDED = "team_decided"
TEAM_FINISHED = "team_finished"


@dataclass(frozen=True)
class DomainEvent:
    execution_id: str
    seq: int
    type: str
    payload: dict
    ts_ms: int
    schema_version: int = SCHEMA_VERSION


def make_event(execution_id: str, seq: int, ts_ms: int,
               type: str, payload: dict) -> DomainEvent:
    return DomainEvent(execution_id=execution_id, seq=seq, type=type,
                       payload=dict(payload), ts_ms=ts_ms)
