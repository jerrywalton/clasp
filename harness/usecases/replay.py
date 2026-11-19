"""Replay: fold an event log back into a trace for audit, debugging, and
model-upgrade regression (replay old episodes against a new model, diff).

Also the recovery fold: `reconstruct_run` reduces a persisted log to the
control state a run had reached (current state, next sequence number,
accumulated step/token budget, transitions taken) AND the data context
at the last committed boundary. Events are self-contained: RUN_STARTED
carries the initial context, MODEL_CALLED carries prompt + output,
TOOL_INVOKED carries args + result, and each TRANSITION carries the
context delta its state committed. Context is therefore a fold over
the log, not a re-execution. Events after the last committed boundary
(the single in-flight state's suffix) are returned separately; only
that suffix may need deterministic re-driving on resume.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from core.events import (
    BUDGET_EXHAUSTED, COMPLETED, CONTEXT_CHECKPOINT, ESCALATED,
    MODEL_CALLED, MODEL_FAILED, PAUSE_DECIDED, PAUSE_REQUESTED,
    RUN_STARTED, STATE_ENTERED, STEP_COMMITTED, TOOL_INVOKED,
    TRANSITION, DomainEvent,
)


def replay_trace(events: list[DomainEvent]) -> list[dict]:
    """Project an event log to a comparable trace (timestamps excluded:
    two identical executions must produce identical traces)."""
    return [{"seq": e.seq, "type": e.type, "payload": e.payload}
            for e in sorted(events, key=lambda e: e.seq)]


def diff_traces(a: list[dict], b: list[dict]) -> list[str]:
    """Human-readable divergences between two traces."""
    divergences: list[str] = []
    for i, (ea, eb) in enumerate(zip(a, b)):
        if ea["type"] != eb["type"]:
            divergences.append(
                f"step {i}: type {ea['type']} -> {eb['type']}")
        elif ea["payload"] != eb["payload"]:
            divergences.append(f"step {i} ({ea['type']}): payload differs")
    if len(a) != len(b):
        divergences.append(f"length differs: {len(a)} vs {len(b)}")
    return divergences


@dataclass(frozen=True)
class RunReconstruction:
    """Control state folded from a persisted event log.

    `steps_used` mirrors the runner's budget accounting: one step per
    state entry that went through the run loop's budget check. Entries
    logged by the escalation/exhaustion jump itself do not consume a
    step, and neither does an entry whose budget check immediately
    exhausted (its next event is BUDGET_EXHAUSTED for the same state).
    `tokens_used` is the exact sum over MODEL_CALLED events.
    """
    execution_id: str | None
    current_state: str | None   # last entered state, or last transition target
    final_state: str | None     # set only when a COMPLETED event exists
    completed: bool
    next_seq: int
    steps_used: int
    tokens_used: int
    model_calls: int
    entered_states: tuple[str, ...]
    transitions: tuple[tuple[str, str, str], ...]  # (from, event, to)
    last_event_type: str | None
    # Atomic-event fold (Phase B). Defaults keep legacy/hand-built logs
    # and direct construction source-compatible.
    recorded_initial_context: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    boundary_seq: int = 0
    boundary_steps_used: int = 0
    boundary_tokens_used: int = 0
    # Budget scope (#7): tool calls fold exactly (one TOOL_INVOKED
    # per fired call). Corrections fold from model attempts per
    # state entry (attempts - 1); handler-state corrections produce
    # no model events and are not fold-visible, so a restored
    # corrections counter can undercount by those — the limit still
    # binds every live attempt after the restore point.
    tool_calls_used: int = 0
    corrections_used: int = 0
    boundary_tool_calls_used: int = 0
    boundary_corrections_used: int = 0
    suffix_events: tuple[DomainEvent, ...] = ()
    current_state_entered: bool = False
    # HITL pause (v1.1): the latest pause request in the log and the
    # decision recorded against it, if any. A request with no
    # decision is an awaiting run: no decision, no resume.
    pause_requested: DomainEvent | None = None
    pause_decision: DomainEvent | None = None


def _budget_fold(events: list[DomainEvent]) -> tuple[int, int, int, int, int]:
    """(steps_used, tokens_used, model_calls, tool_calls_used,
    corrections_used) for an event prefix, using the runner's
    accounting rules for state entries."""
    ordered = sorted(events, key=lambda e: e.seq)
    steps_used = 0
    tokens_used = 0
    model_calls = 0
    tool_calls_used = 0
    corrections_used = 0
    entry_attempts = 0
    for i, event in enumerate(ordered):
        if event.type == STATE_ENTERED:
            corrections_used += max(0, entry_attempts - 1)
            entry_attempts = 0
            prev_type = ordered[i - 1].type if i else None
            if prev_type in (ESCALATED, BUDGET_EXHAUSTED):
                continue
            nxt = ordered[i + 1] if i + 1 < len(ordered) else None
            if (nxt is not None and nxt.type == BUDGET_EXHAUSTED
                    and nxt.payload.get("state")
                    == event.payload.get("state")):
                continue
            steps_used += 1
        elif event.type == MODEL_CALLED:
            model_calls += 1
            entry_attempts += 1
            tokens_used += (int(event.payload.get("prompt_tokens", 0))
                            + int(event.payload.get("completion_tokens", 0)))
        elif event.type == MODEL_FAILED:
            entry_attempts += 1
        elif event.type == TOOL_INVOKED:
            tool_calls_used += 1
    corrections_used += max(0, entry_attempts - 1)
    return steps_used, tokens_used, model_calls, tool_calls_used, \
        corrections_used


def reconstruct_run(events: list[DomainEvent]) -> RunReconstruction:
    """Fold one run's events (any order; sorted by seq here) into the
    control + data state the run had reached when the log ends.

    The returned ``context`` is the data context at the last committed
    boundary: RUN_STARTED's initial context, updated by each
    TRANSITION's ``context_delta`` (and ``context_removed``), or the
    final context recorded on COMPLETED. Events after that boundary
    are returned as ``suffix_events`` — the single in-flight state.
    Raises ValueError if the events span more than one execution."""
    ordered = sorted(events, key=lambda e: e.seq)
    if not ordered:
        return RunReconstruction(
            execution_id=None, current_state=None, final_state=None,
            completed=False, next_seq=0, steps_used=0, tokens_used=0,
            model_calls=0, entered_states=(), transitions=(),
            last_event_type=None)
    execution_ids = {e.execution_id for e in ordered}
    if len(execution_ids) != 1:
        raise ValueError(
            f"events span multiple executions: {sorted(execution_ids)}")

    steps_used, tokens_used, model_calls, tool_calls_used, \
        corrections_used = _budget_fold(ordered)
    entered: list[str] = []
    transitions: list[tuple[str, str, str]] = []
    final_state: str | None = None
    completed = False
    pause_requested: DomainEvent | None = None
    pause_decision: DomainEvent | None = None
    for event in ordered:
        if event.type == STATE_ENTERED:
            entered.append(event.payload.get("state"))
        elif event.type == TRANSITION:
            transitions.append((event.payload.get("from"),
                                  event.payload.get("event"),
                                  event.payload.get("to")))
        elif event.type == PAUSE_REQUESTED:
            pause_requested = event
            pause_decision = None
        elif event.type == PAUSE_DECIDED:
            if pause_requested is not None and pause_decision is None:
                pause_decision = event
        elif event.type == COMPLETED:
            completed = True
            final_state = event.payload.get("state")

    # --- data/context fold to the last committed boundary
    recorded_initial_context: dict = {}
    context: dict = {}
    boundary_idx = 0
    boundary_state: str | None = None
    # Commit records (v1.1): logs whose RUN_STARTED carries
    # "commit_records" fold a TRANSITION's delta only when a
    # STEP_COMMITTED naming that transition follows it. A transition
    # without its commit record is an uncommitted transaction: its
    # delta never happened, and the transition stays in the suffix to
    # be re-driven. Older logs keep the legacy rule (the TRANSITION
    # itself is the commit point).
    strict_commits = bool(
        ordered and ordered[0].type == RUN_STARTED
        and ordered[0].payload.get("commit_records"))
    if ordered and ordered[0].type == RUN_STARTED:
        recorded_initial_context = copy.deepcopy(
            ordered[0].payload.get("initial_context") or {})
        context = copy.deepcopy(recorded_initial_context)
        boundary_state = ordered[0].payload.get("initial_state")
        boundary_idx = 1
    for i, event in enumerate(ordered):
        if event.type == TRANSITION:
            commit_idx = i
            if strict_commits:
                commit_idx = -1
                for j in range(i + 1, len(ordered)):
                    nxt = ordered[j]
                    if nxt.type in (TRANSITION, COMPLETED, RUN_STARTED):
                        break
                    if nxt.type == STEP_COMMITTED and \
                            nxt.payload.get("transition_seq") == event.seq:
                        commit_idx = j
                        break
                if commit_idx < 0:
                    continue  # uncommitted: never happened
            delta = event.payload.get("context_delta") or {}
            for key in event.payload.get("context_removed") or ():
                context.pop(key, None)
            context.update(copy.deepcopy(delta))
            boundary_state = event.payload.get("to")
            boundary_idx = commit_idx + 1
        elif event.type == CONTEXT_CHECKPOINT:
            # A checkpoint carries the folded context at its commit
            # boundary: recovery starts from it instead of folding
            # every delta since RUN_STARTED. The deltas remain the
            # truth; this is the backup in restore + roll-forward.
            recorded = event.payload.get("context")
            if isinstance(recorded, dict):
                context = copy.deepcopy(recorded)
            boundary_idx = i + 1
        elif event.type == COMPLETED:
            recorded_context = event.payload.get("context")
            if isinstance(recorded_context, dict):
                context = copy.deepcopy(recorded_context)
            boundary_state = event.payload.get("state")
            boundary_idx = i + 1

    suffix = tuple(ordered[boundary_idx:])
    boundary_steps, boundary_tokens, _, boundary_tool_calls, \
        boundary_corrections = _budget_fold(ordered[:boundary_idx])
    boundary_seq = (ordered[boundary_idx - 1].seq + 1
                    if boundary_idx else 0)

    current_state = final_state if completed else boundary_state
    current_state_entered = False
    if not completed:
        for event in suffix:
            if event.type == STATE_ENTERED:
                current_state = event.payload.get("state")
                current_state_entered = True
            elif event.type == TRANSITION and not strict_commits:
                current_state = event.payload.get("to")
                current_state_entered = False

    return RunReconstruction(
        execution_id=ordered[0].execution_id,
        current_state=current_state, final_state=final_state,
        completed=completed, next_seq=ordered[-1].seq + 1,
        steps_used=steps_used, tokens_used=tokens_used,
        model_calls=model_calls, entered_states=tuple(entered),
        transitions=tuple(transitions),
        last_event_type=ordered[-1].type,
        recorded_initial_context=recorded_initial_context,
        context=context, boundary_seq=boundary_seq,
        boundary_steps_used=boundary_steps,
        boundary_tokens_used=boundary_tokens,
        tool_calls_used=tool_calls_used,
        corrections_used=corrections_used,
        boundary_tool_calls_used=boundary_tool_calls,
        boundary_corrections_used=boundary_corrections,
        suffix_events=suffix,
        current_state_entered=current_state_entered,
        pause_requested=pause_requested,
        pause_decision=pause_decision)
