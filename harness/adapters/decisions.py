"""Decision channels: rendering a pause for a human.

A channel is transport only. This module renders a PAUSE_REQUESTED
event as plain text a deployment can print in a CLI, post to a chat
channel, or raise in a console — the same text everywhere, because
the fact lives in the log, not in the rendering. Collecting the
human's answer and calling usecases.decisions.record_decision is
the deployment's half of the channel.
"""

from __future__ import annotations

import json

from core.events import DomainEvent


def render_pause_request(event: DomainEvent) -> str:
    """Render one PAUSE_REQUESTED event as a plain-text card."""
    p = event.payload
    lines = [
        "PAUSE REQUESTED — a run is awaiting a decision",
        f"run: {event.execution_id}",
        f"state: {p.get('state')}",
        f"approver: {p.get('approver')}",
        f"channel: {p.get('channel')}",
        f"allowed decisions: {', '.join(p.get('allowed_decisions') or [])}",
    ]
    if p.get("timeout_ms") is not None:
        lines.append(
            f"timeout: {p['timeout_ms']} ms (expiry denies, fail-closed)")
    lines.append(
        "context: "
        + json.dumps(p.get("context") or {}, sort_keys=True, default=str))
    lines.append(
        "Record the decision in the run's log to resume it. "
        "No decision, no resume.")
    return "\n".join(lines)
