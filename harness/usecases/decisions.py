"""Recording human decisions into a paused run's log.

The decision port, from the approver's side. A pause is a
PAUSE_REQUESTED event in the run's own log; a decision counts when —
and only when — a matching PAUSE_DECIDED event is appended to the
same log. Channels (a CLI prompt, a Slack message, a console
interrupt) are transports that end here: whatever the human clicked
or typed, this function is what makes it a fact.

Validation is read off the request event itself, so the log stays
self-contained: the decision must be one the request declared
allowed, and it must come from the declared approver. Recording a
decision takes the execution's single-writer lease briefly, so a
decision can never race a live runner.
"""

from __future__ import annotations

from core.errors import HarnessError
from core.events import PAUSE_DECIDED, PAUSE_REQUESTED, DomainEvent, make_event
from usecases.ports import ClockPort, EventStorePort


def pending_pause(events: EventStorePort,
                  execution_id: str) -> DomainEvent | None:
    """The run's undecided PAUSE_REQUESTED event, or None."""
    stored = sorted(events.read(execution_id), key=lambda e: e.seq)
    pending: DomainEvent | None = None
    for event in stored:
        if event.type == PAUSE_REQUESTED:
            pending = event
        elif event.type == PAUSE_DECIDED and pending is not None:
            pending = None
    return pending


def record_decision(events: EventStorePort, clock: ClockPort,
                    execution_id: str, *, decision: str,
                    decided_by: str,
                    context_patch: dict | None = None,
                    remember: bool = False) -> DomainEvent:
    """Append the PAUSE_DECIDED event for the run's pending pause.

    Raises HarnessError when there is no pending pause, when the
    pause was already decided, when the decision is not one the
    request allowed, when it does not come from the declared
    approver, or when an amend carries no context patch. Returns
    the appended event."""
    stored = sorted(events.read(execution_id), key=lambda e: e.seq)
    if not stored:
        raise HarnessError(
            f"cannot decide: no persisted events for execution "
            f"'{execution_id}'")
    request = pending_pause(events, execution_id)
    if request is None:
        if any(e.type == PAUSE_REQUESTED for e in stored):
            raise HarnessError(
                f"cannot decide: the pause for execution "
                f"'{execution_id}' was already decided; a decision "
                f"is a ledger event and is not rewritten")
        raise HarnessError(
            f"cannot decide: execution '{execution_id}' has no "
            f"pending pause")
    allowed = request.payload.get("allowed_decisions") or []
    if decision not in allowed:
        raise HarnessError(
            f"cannot decide: '{decision}' is not an allowed decision "
            f"for this pause (allowed: {allowed})")
    approver = request.payload.get("approver")
    if decided_by != approver:
        raise HarnessError(
            f"cannot decide: this pause awaits its approver "
            f"'{approver}', not '{decided_by}'")
    if decision == "amend" and not isinstance(context_patch, dict):
        raise HarnessError(
            "cannot decide: an amend decision must carry a "
            "context_patch object")
    if decision != "amend" and context_patch is not None:
        raise HarnessError(
            "cannot decide: a context_patch belongs to an amend "
            "decision only")

    payload: dict = {
        "state": request.payload.get("state"),
        "decision": decision,
        "decided_by": decided_by,
        "remember": bool(remember),
    }
    if decision == "amend":
        payload["context_patch"] = dict(context_patch)

    owner = f"approver-{decided_by}"
    acquire = getattr(events, "acquire_lease", None)
    if acquire is not None and not acquire(execution_id, owner, 5_000):
        raise HarnessError(
            f"cannot decide: execution '{execution_id}' is leased "
            f"by another writer")
    try:
        event = make_event(execution_id, stored[-1].seq + 1,
                           clock.now_ms(), PAUSE_DECIDED, payload)
        events.append(event)
        events.flush()
    finally:
        release = getattr(events, "release_lease", None)
        if release is not None:
            release(execution_id, owner)
    return event
