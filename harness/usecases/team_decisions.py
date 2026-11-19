"""Recording human decisions into a paused team run's log.

The decision port for handoff pauses, from the approver's side —
the team-scale analogue of usecases/decisions.py. A handoff pause is
a TEAM_PAUSED event in the team log; a decision counts when — and
only when — a matching TEAM_DECIDED event is appended to the same
log. Channels are transports that end here: whatever the human
clicked or typed, this function is what makes it a fact.

Validation is read off the pause event itself, so the log stays
self-contained: the decision must be one the pause declared allowed,
and it must come from the declared approver. Recording a decision
takes the team execution's single-writer lease briefly, so a
decision can never race a live Orchestrator.
"""

from __future__ import annotations

from core.errors import HarnessError
from core.events import TEAM_DECIDED, TEAM_PAUSED, DomainEvent, make_event
from usecases.ports import ClockPort, EventStorePort


def pending_team_pause(events: EventStorePort,
                       execution_id: str) -> DomainEvent | None:
    """The team run's undecided TEAM_PAUSED event, or None."""
    stored = sorted(events.read(execution_id), key=lambda e: e.seq)
    pending: DomainEvent | None = None
    for event in stored:
        if event.type == TEAM_PAUSED:
            pending = event
        elif event.type == TEAM_DECIDED and pending is not None:
            pending = None
    return pending


def record_team_decision(events: EventStorePort, clock: ClockPort,
                         execution_id: str, *, decision: str,
                         decided_by: str) -> DomainEvent:
    """Append the TEAM_DECIDED event for the team run's pending
    handoff pause.

    Raises HarnessError when there is no pending pause, when the
    pause was already decided, when the decision is not one the
    pause allowed, or when it does not come from the declared
    approver. Returns the appended event."""
    stored = sorted(events.read(execution_id), key=lambda e: e.seq)
    if not stored:
        raise HarnessError(
            f"cannot decide: no persisted events for team execution "
            f"'{execution_id}'")
    request = pending_team_pause(events, execution_id)
    if request is None:
        if any(e.type == TEAM_PAUSED for e in stored):
            raise HarnessError(
                f"cannot decide: the handoff pause for team "
                f"execution '{execution_id}' was already decided; "
                f"a decision is a ledger event and is not rewritten")
        raise HarnessError(
            f"cannot decide: team execution '{execution_id}' has "
            f"no pending handoff pause")
    allowed = request.payload.get("allowed_decisions") or []
    if decision not in allowed:
        raise HarnessError(
            f"cannot decide: '{decision}' is not an allowed "
            f"decision for this pause (allowed: {allowed})")
    approver = request.payload.get("approver")
    if decided_by != approver:
        raise HarnessError(
            f"cannot decide: this pause awaits its approver "
            f"'{approver}', not '{decided_by}'")

    payload: dict = {
        "from_member": request.payload.get("from_member"),
        "to_member": request.payload.get("to_member"),
        "decision": decision,
        "decided_by": decided_by,
    }

    owner = f"approver-{decided_by}"
    acquire = getattr(events, "acquire_lease", None)
    if acquire is not None and not acquire(execution_id, owner, 5_000):
        raise HarnessError(
            f"cannot decide: team execution '{execution_id}' is "
            f"leased by another writer")
    try:
        event = make_event(execution_id, stored[-1].seq + 1,
                           clock.now_ms(), TEAM_DECIDED, payload)
        events.append(event)
        events.flush()
    finally:
        release = getattr(events, "release_lease", None)
        if release is not None:
            release(execution_id, owner)
    return event
