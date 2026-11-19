"""Tool intent (#2) and budget counting scope (#7). Plain asserts:
    python3 harness/tests/test_tool_intent_budget.py
Intent: every tool call is preceded in the log by a TOOL_INTENDED
record carrying the same idempotency key — on the plain path and on
the ledger path — and a crash between intent and invocation leaves
the intent visible. Counting: tool calls and correction attempts are
priced actions with their own budget limits, consulted before the
action, and restored from the fold on resume."""

import os
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adapters.events.stores import InMemoryStore, SqliteStore
from adapters.model.models import ScriptedModel
from app.builder import Harness
from core.events import (
    BUDGET_EXHAUSTED, LEDGER_POSTED, TOOL_INTENDED, TOOL_INVOKED,
)
from usecases.replay import reconstruct_run


class Clock:
    def __init__(self):
        self.t = 0

    def now_ms(self):
        self.t += 1
        return self.t


def envelope(event="go", data=None, tool_calls=None):
    return {"event": event, "data": data or {},
            "tool_calls": tool_calls or []}


def build_tool_machine(**budget_kwargs):
    records = {}

    def record(args):
        records[args["value"]] = {"recorded": args["value"]}
        return records[args["value"]]

    b = Harness.define("intent-test").initial("a")
    b.state("a", prompt_template="A $correction",
            output_schema={"type": "object"}, tools=("record",))
    b.state("done", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("a", "go", "done")
    b.escalation_state("esc").exhausted_state("exh")
    b.register_tool("record", record, legal_in={"a"},
                    args_schema={"type": "object",
                                 "properties": {
                                     "value": {"type": "integer"}},
                                 "required": ["value"]})
    b.budget(step_limit=10, token_limit=100000, **budget_kwargs)
    return b.build(), records


def test_intent_precedes_invocation_with_same_key():
    built, _ = build_tool_machine()
    store = InMemoryStore()
    outcome = built.runner(ScriptedModel([
        envelope(tool_calls=[{"name": "record", "args": {"value": 7}}]),
    ]), store, Clock()).run({}, execution_id="run-i")
    assert outcome.success
    events = store.read("run-i")
    intended = next(e for e in events if e.type == TOOL_INTENDED)
    invoked = next(e for e in events if e.type == TOOL_INVOKED)
    assert intended.seq < invoked.seq
    assert intended.payload["key"] == invoked.payload["key"]
    assert intended.payload["args"] == {"value": 7}


def test_intent_on_the_ledger_path():
    b = Harness.define("intent-ledger").initial("a")
    b.state("a", prompt_template="A $correction",
            output_schema={"type": "object"}, tools=("charge",))
    b.state("done", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("a", "go", "done")
    b.escalation_state("esc").exhausted_state("exh")
    b.accounts("cash", "revenue")
    b.register_ledger_tool(
        "charge", lambda args: {"charged": args["amount_cents"]},
        legal_in={"a"}, debit_account="cash", credit_account="revenue",
        args_schema={"type": "object",
                     "properties": {
                         "amount_cents": {"type": "integer"},
                         "order_id": {"type": "string"}},
                     "required": ["amount_cents", "order_id"]})
    b.budget(step_limit=10, token_limit=100000)
    store = InMemoryStore()
    outcome = b.build().runner(ScriptedModel([
        envelope(tool_calls=[{"name": "charge",
                              "args": {"amount_cents": 500,
                                       "order_id": "ORD-1"}}]),
    ]), store, Clock()).run({}, execution_id="run-l")
    assert outcome.success
    events = store.read("run-l")
    intended = next(e for e in events if e.type == TOOL_INTENDED)
    first_leg = next(e for e in events if e.type == LEDGER_POSTED)
    invoked = next(e for e in events if e.type == TOOL_INVOKED)
    assert intended.seq < first_leg.seq < invoked.seq
    assert intended.payload["key"] == invoked.payload["key"]


def test_tool_call_limit_binds_before_the_call():
    built, records = build_tool_machine(tool_call_limit=1)
    store = InMemoryStore()
    outcome = built.runner(ScriptedModel([
        envelope(tool_calls=[
            {"name": "record", "args": {"value": 1}},
            {"name": "record", "args": {"value": 2}}]),
    ]), store, Clock()).run({}, execution_id="run-b")
    assert outcome.success is False and outcome.final_state == "exh"
    assert records == {1: {"recorded": 1}}  # the second call never fired
    exhausted = [e for e in store.read("run-b")
                 if e.type == BUDGET_EXHAUSTED]
    assert "tool-call budget" in exhausted[-1].payload["reason"]
    # And no intent was recorded for the refused call: intent means
    # an approved call about to fire.
    assert len([e for e in store.read("run-b")
                if e.type == TOOL_INTENDED]) == 1


def test_correction_limit_binds_across_attempts():
    b = Harness.define("correction-budget").initial("a")
    b.state("a", prompt_template="A $correction",
            output_schema={"type": "object"}, max_corrections=5)
    b.state("done", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("a", "go", "done")
    b.escalation_state("esc").exhausted_state("exh")
    b.budget(step_limit=10, token_limit=100000, correction_limit=1)
    store = InMemoryStore()
    bad = envelope(event="bogus")
    outcome = b.build().runner(
        ScriptedModel([bad, bad, bad, bad]), store, Clock()
    ).run({}, execution_id="run-c")
    # Attempt 1 fails, correction 1 fails, correction 2 is refused by
    # the budget: the run exhausts instead of escalating.
    assert outcome.success is False and outcome.final_state == "exh"
    exhausted = [e for e in store.read("run-c")
                 if e.type == BUDGET_EXHAUSTED]
    assert "correction budget" in exhausted[-1].payload["reason"]


def test_budget_counters_fold_and_restore():
    built, _ = build_tool_machine(tool_call_limit=10)
    with TemporaryDirectory() as td:
        path = os.path.join(td, "e.db")
        built.runner(ScriptedModel([
            envelope(tool_calls=[{"name": "record",
                                  "args": {"value": 3}}]),
        ]), SqliteStore(path), Clock()).run({}, execution_id="run-f")
        plan = reconstruct_run(SqliteStore(path).read("run-f"))
        assert plan.tool_calls_used == 1, plan
        assert plan.corrections_used == 0, plan


TESTS = [v for k, v in sorted(globals().items())
         if k.startswith("test_") and callable(v)]


def main():
    failed = 0
    for test in TESTS:
        try:
            test()
        except Exception as exc:  # noqa: BLE001 - report, don't hide
            failed += 1
            print(f"FAIL {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {test.__name__}")
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
