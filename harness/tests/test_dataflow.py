"""Frozen context and dataflow checks (#11, v1.1). Plain asserts:
    python3 harness/tests/test_dataflow.py
Every context write flows through an envelope's data, so both rules
are checked before any tool fires and a violation has no effects:
(1) a state with declared writes may write only those keys;
(2) a frozen context key, once set, may not change value. Violations
are correctable failures, visible in the correction reasons and in
the escalation record."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adapters.events.stores import InMemoryStore
from adapters.model.models import ScriptedModel
from app.builder import Harness
from core.errors import DefinitionError
from core.events import ESCALATED, TOOL_INVOKED


class Clock:
    def __init__(self):
        self.t = 0

    def now_ms(self):
        self.t += 1
        return self.t


def envelope(event="go", data=None, tool_calls=None):
    return {"event": event, "data": data or {},
            "tool_calls": tool_calls or []}


def build_machine(*, frozen=(), writes_a=(), writes_b=()):
    b = Harness.define("dataflow-test").initial("a")
    b.state("a", prompt_template="A $correction",
            output_schema={"type": "object"}, writes=writes_a)
    b.state("b", prompt_template="B $correction",
            output_schema={"type": "object"}, writes=writes_b)
    b.state("done", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("a", "go", "b")
    b.transition("b", "go", "done")
    b.escalation_state("esc").exhausted_state("exh")
    b.budget(step_limit=10, token_limit=100000)
    if frozen:
        b.frozen_context_keys(*frozen)
    return b.build()


def test_frozen_key_may_not_change_once_set():
    built = build_machine(frozen=("order_id",))
    store = InMemoryStore()
    outcome = built.runner(ScriptedModel([
        envelope(data={"order_id": "ORD-1"}),
        envelope(data={"order_id": "ORD-2"}),   # tries to change it
        envelope(data={"order_id": "ORD-2"}),
        envelope(data={"order_id": "ORD-2"}),
    ]), store, Clock()).run({}, execution_id="run-f")
    assert outcome.success is False and outcome.final_state == "esc"
    esc = [e for e in store.read("run-f") if e.type == ESCALATED]
    reasons = str(esc[-1].payload["failures"])
    assert "frozen" in reasons and "order_id" in reasons


def test_frozen_key_set_once_and_same_value_rewrite_are_fine():
    built = build_machine(frozen=("order_id",))
    store = InMemoryStore()
    outcome = built.runner(ScriptedModel([
        envelope(data={"order_id": "ORD-1"}),
        envelope(data={"order_id": "ORD-1", "note": "ok"}),
    ]), store, Clock()).run({}, execution_id="run-ok")
    assert outcome.success, outcome
    assert outcome.context["order_id"] == "ORD-1"


def test_frozen_violation_fires_no_tools():
    calls = {"n": 0}

    def probe(args):
        calls["n"] += 1
        return {"ok": True}

    b = Harness.define("dataflow-tools").initial("a")
    b.state("a", prompt_template="A $correction",
            output_schema={"type": "object"})
    b.state("b", prompt_template="B $correction",
            output_schema={"type": "object"}, tools=("probe",))
    b.state("done", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("a", "go", "b")
    b.transition("b", "go", "done")
    b.escalation_state("esc").exhausted_state("exh")
    b.budget(step_limit=10, token_limit=100000)
    b.register_tool("probe", probe, legal_in={"b"},
                    args_schema={"type": "object"})
    b.frozen_context_keys("order_id")
    store = InMemoryStore()
    bad = envelope(data={"order_id": "ORD-2"},
                   tool_calls=[{"name": "probe", "args": {}}])
    outcome = b.build().runner(ScriptedModel([
        envelope(data={"order_id": "ORD-1"}),   # a sets the frozen key
        bad, bad, bad,                          # b tries, with a tool
    ]), store, Clock()).run({}, execution_id="run-t")
    assert outcome.success is False and outcome.final_state == "esc"
    # The dataflow check runs before invocation: the tool never fired.
    assert calls["n"] == 0
    assert not [e for e in store.read("run-t")
                if e.type == TOOL_INVOKED]


def test_declared_writes_restrict_the_state():
    built = build_machine(writes_b=("y",))
    store = InMemoryStore()
    outcome = built.runner(ScriptedModel([
        envelope(data={"x": 1}),
        envelope(data={"z": 9}),   # b may not write z
        envelope(data={"z": 9}),
        envelope(data={"z": 9}),
    ]), store, Clock()).run({}, execution_id="run-w")
    assert outcome.success is False and outcome.final_state == "esc"
    esc = [e for e in store.read("run-w") if e.type == ESCALATED]
    reasons = str(esc[-1].payload["failures"])
    assert "may not write context key 'z'" in reasons


def test_declared_writes_conforming_run_passes():
    built = build_machine(writes_a=("x",), writes_b=("y",))
    store = InMemoryStore()
    outcome = built.runner(ScriptedModel([
        envelope(data={"x": 1}),
        envelope(data={"y": 2}),
    ]), store, Clock()).run({}, execution_id="run-c")
    assert outcome.success
    assert outcome.context["x"] == 1 and outcome.context["y"] == 2


def test_correction_can_fix_a_dataflow_violation():
    built = build_machine(writes_b=("y",))
    store = InMemoryStore()
    outcome = built.runner(ScriptedModel([
        envelope(data={"x": 1}),
        envelope(data={"z": 9}),   # wrong key first
        envelope(data={"y": 2}),   # corrected
    ]), store, Clock()).run({}, execution_id="run-fix")
    assert outcome.success
    assert "z" not in outcome.context


def test_build_validation():
    b = Harness.define("bad-flow").initial("a")
    b.state("a", prompt_template="A $correction",
            output_schema={"type": "object"}, writes=("x", "x"))
    b.state("done", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("a", "go", "done")
    b.escalation_state("esc").exhausted_state("exh")
    b.budget(step_limit=10, token_limit=1000)
    try:
        b.build()
    except DefinitionError:
        pass
    else:
        raise AssertionError("duplicate declared writes built")

    b2 = Harness.define("bad-flow-2").initial("a")
    b2.state("a", prompt_template="A $correction",
             output_schema={"type": "object"})
    b2.state("done", terminal=True)
    b2.state("esc", terminal=True)
    b2.state("exh", terminal=True)
    b2.transition("a", "go", "done")
    b2.escalation_state("esc").exhausted_state("exh")
    b2.budget(step_limit=10, token_limit=1000)
    b2.frozen_context_keys("k", "k")
    try:
        b2.build()
    except DefinitionError:
        pass
    else:
        raise AssertionError("duplicate frozen keys built")


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
