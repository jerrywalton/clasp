"""Resume preflight: the operator's read of a persisted run.

Folds a run's log WITHOUT resuming it and renders a plain-text
report: where the run stands, what is committed, what is in flight,
what it has spent, whether its ledger balances, and what the
operator's next move is. No events are appended, no model is called,
no lease is taken — this is the read-only instrument panel for the
recovery drill (restore + roll-forward starts with knowing where
the log ends). Fingerprint validation stays resume()'s job; the
report says so where it matters.
"""

from __future__ import annotations

from core.events import (
    CONTEXT_CHECKPOINT, PAUSE_REQUESTED, RUN_STARTED, STEP_COMMITTED,
)
from core.ledger import fold_ledger
from usecases.ports import EventStorePort
from usecases.replay import reconstruct_run


def resume_preflight(events: EventStorePort, execution_id: str) -> str:
    stored = sorted(events.read(execution_id), key=lambda e: e.seq)
    lines: list[str] = [f"RESUME PREFLIGHT — {execution_id}"]
    discarded = getattr(events, "last_discarded_tail", None)

    if not stored:
        lines.append("status: NO LOG — no persisted events for this "
                     "execution")
        lines.append("verdict: NOTHING TO RESUME")
        return "\n".join(lines)

    first = stored[0]
    plan = reconstruct_run(stored)
    book = fold_ledger(stored)
    committed = [e for e in stored if e.seq < plan.boundary_seq]
    commits = [e for e in stored if e.type == STEP_COMMITTED]
    checkpoints = [e for e in stored if e.type == CONTEXT_CHECKPOINT]

    if first.type == RUN_STARTED:
        lines.append(
            f"definition: {first.payload.get('definition')} "
            f"v{first.payload.get('version')} "
            f"(fingerprint v{first.payload.get('fingerprint_version')}, "
            f"checked at resume)")
    else:
        lines.append("definition: UNKNOWN — log has no RUN_STARTED; "
                     "only self-contained logs are resumable")
    lines.append(f"events: {len(stored)} (seq 0..{stored[-1].seq}), "
                 f"{len(committed)} committed, "
                 f"{len(plan.suffix_events)} in flight")
    lines.append(f"commits: {len(commits)} step(s), "
                 f"{len(checkpoints)} checkpoint(s)")
    lines.append(
        f"budget spent: {plan.steps_used} step(s), "
        f"{plan.tokens_used} token(s), {plan.tool_calls_used} tool "
        f"call(s), {plan.corrections_used} correction(s)")
    if book.transactions:
        lines.append(
            f"ledger: {len(book.transactions)} transaction(s), "
            f"{'balanced' if book.balanced else 'NOT balanced'}"
            + ("" if book.balanced else
               f" — incomplete: "
               f"{', '.join(t.txn_id for t in book.incomplete)}"))
    if discarded:
        lines.append("torn tail: a partial final event was discarded "
                     "on read (uncommitted; it never happened)")

    if first.type != RUN_STARTED:
        lines.append("verdict: NOT RESUMABLE — log is not self-contained")
    elif plan.completed:
        lines.append(f"status: COMPLETED in state '{plan.final_state}'")
        lines.append("verdict: NOTHING TO RESUME — resume returns the "
                     "folded outcome with no new work")
    elif plan.pause_requested is not None \
            and plan.pause_decision is None:
        req = plan.pause_requested.payload
        lines.append(
            f"status: PAUSED in state '{req.get('state')}' — awaiting "
            f"a decision from {req.get('approver')} "
            f"(channel: {req.get('channel')}, allowed: "
            f"{', '.join(req.get('allowed_decisions') or [])})")
        lines.append("verdict: RECORD A DECISION FIRST — no decision, "
                     "no resume")
    else:
        lines.append(
            f"status: INTERRUPTED in state '{plan.current_state}' "
            f"({len(plan.suffix_events)} in-flight event(s) will be "
            f"re-driven deterministically; committed states are "
            f"folded, never re-run)")
        lines.append("verdict: READY TO RESUME")
    return "\n".join(lines)
