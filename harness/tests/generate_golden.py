"""Regenerate the golden-log corpus.

    python3 harness/tests/generate_golden.py

Each scenario runs against the CURRENT code and its event log is
written into harness/tests/golden/ as JSONL, with a manifest entry
recording the fold's verdict (final/current state, budget totals,
context digest, trace digest). test_golden_logs.py replays the
checked-in corpus against the manifest: behavior drift in the fold,
the trace projection, or event generation is a test failure.

Regenerate ONLY when a behavior change is intended, and read the
git diff of the corpus before committing it — the diff is the
record of what changed.
"""

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adapters.events.stores import JsonlStore
from adapters.model.models import ScriptedModel
from app.builder import Harness
from usecases.decisions import record_decision
from usecases.replay import reconstruct_run, replay_trace

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")


class Clock:
    def __init__(self):
        self.t = 0

    def now_ms(self):
        self.t += 1
        return self.t


def envelope(event="go", data=None, tool_calls=None):
    return {"event": event, "data": data or {},
            "tool_calls": tool_calls or []}


def canonical_hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, default=str,
                     separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def two_state_builder(**extra):
    b = Harness.define("golden-two-state").initial("a")
    b.state("a", prompt_template="A $correction",
            output_schema={"type": "object"}, tools=("record",))
    b.state("b", prompt_template="B $correction",
            output_schema={"type": "object"})
    b.state("done", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("a", "go", "b")
    b.transition("b", "go", "done")
    b.escalation_state("esc").exhausted_state("exh")
    b.register_tool(
        "record", lambda args: {"recorded": args["value"]},
        legal_in={"a"},
        args_schema={"type": "object",
                     "properties": {"value": {"type": "integer"}},
                     "required": ["value"]})
    b.budget(step_limit=10, token_limit=100000)
    for key, value in extra.items():
        getattr(b, key)(**value)
    return b


RESPONSES = [
    envelope(data={"x": 1},
             tool_calls=[{"name": "record", "args": {"value": 7}}]),
    envelope(data={"y": 2}),
]


def scenario_completed(path):
    store = JsonlStore(path)
    two_state_builder().build().runner(
        ScriptedModel(list(RESPONSES)), store, Clock()
    ).run({"request": "r1"}, execution_id="golden-completed")
    return store


def scenario_paused_resumed(path):
    store = JsonlStore(path)
    b = two_state_builder()
    b.pause("b", approver="ops-lead", channel="cli",
            decisions=("approve", "deny", "amend"))
    built = b.build()
    outcome = built.runner(
        ScriptedModel(list(RESPONSES)), store, Clock()
    ).run({"request": "r1"}, execution_id="golden-paused")
    assert outcome.awaiting_decision
    record_decision(store, Clock(), "golden-paused",
                    decision="approve", decided_by="ops-lead")
    built.resume(ScriptedModel([RESPONSES[1]]), store, Clock(),
                 execution_id="golden-paused")
    return store


def scenario_escalated(path):
    store = JsonlStore(path)
    bad = envelope(event="bogus")
    two_state_builder().build().runner(
        ScriptedModel([bad, bad, bad]), store, Clock()
    ).run({"request": "r1"}, execution_id="golden-escalated")
    return store


def scenario_checkpointed(path):
    b = Harness.define("golden-ckpt").initial("a")
    b.state("a", prompt_template="A $correction",
            output_schema={"type": "object"})
    b.state("b", prompt_template="B $correction",
            output_schema={"type": "object"})
    b.state("c", handler=lambda ctx: envelope(
        "go", {"z": ctx["x"] + ctx["y"]}))
    b.state("done", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("a", "go", "b")
    b.transition("b", "go", "c")
    b.transition("c", "go", "done")
    b.escalation_state("esc").exhausted_state("exh")
    b.budget(step_limit=10, token_limit=100000)
    b.checkpoint(every_steps=1)
    store = JsonlStore(path)
    b.build().runner(
        ScriptedModel([envelope(data={"x": 1}),
                       envelope(data={"y": 2})]), store, Clock()
    ).run({"request": "r1"}, execution_id="golden-ckpt")
    return store


def scenario_interrupted(path):
    """A completed log truncated into the in-flight state: the
    corpus also pins recovery's starting point, not just endings."""
    tmp = path + ".full"
    full = scenario_completed(tmp)
    events = full.read("golden-completed")
    kept = events[:9]  # through A's commit and B's entry: B in flight
    store = JsonlStore(path)
    for event in kept:
        store.append(event)
    os.remove(tmp)
    if os.path.exists(tmp + ".leases.json"):
        os.remove(tmp + ".leases.json")
    return store


SCENARIOS = {
    "completed": scenario_completed,
    "paused_resumed": scenario_paused_resumed,
    "escalated": scenario_escalated,
    "checkpointed": scenario_checkpointed,
    "interrupted": scenario_interrupted,
}

EXECUTION_IDS = {
    "completed": "golden-completed",
    "paused_resumed": "golden-paused",
    "escalated": "golden-escalated",
    "checkpointed": "golden-ckpt",
    "interrupted": "golden-completed",
}


def main():
    os.makedirs(GOLDEN_DIR, exist_ok=True)
    manifest = {}
    for name, generate in SCENARIOS.items():
        path = os.path.join(GOLDEN_DIR, f"{name}.jsonl")
        if os.path.exists(path):
            os.remove(path)
        leases = path + ".leases.json"
        if os.path.exists(leases):
            os.remove(leases)
        store = generate(path)
        events = store.read(EXECUTION_IDS[name])
        plan = reconstruct_run(events)
        manifest[name] = {
            "file": f"{name}.jsonl",
            "execution_id": EXECUTION_IDS[name],
            "events": len(events),
            "completed": plan.completed,
            "final_state": plan.final_state,
            "current_state": plan.current_state,
            "steps_used": plan.steps_used,
            "tokens_used": plan.tokens_used,
            "tool_calls_used": plan.tool_calls_used,
            "corrections_used": plan.corrections_used,
            "context_sha256": canonical_hash(plan.context),
            "trace_sha256": canonical_hash(replay_trace(events)),
        }
        print(f"{name}: {len(events)} events, "
              f"state={plan.final_state or plan.current_state}")
    with open(os.path.join(GOLDEN_DIR, "manifest.json"), "w",
              encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print("manifest written")


if __name__ == "__main__":
    main()
