"""Harness v1 tests. Plain asserts, no test framework needed:
    python3 harness/tests/test_harness.py
Covers: build-time validation (the "type checker"), runner behavior,
correction loops, escalation, capability gating, budgets, idempotency,
and replay determinism."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adapters.events.stores import InMemoryStore
from adapters.gates.checks import PredicateGate, SchemaGate
from adapters.model.models import ScriptedModel
from app.builder import Harness
from core.errors import DefinitionError
from core.events import COMPLETED, ESCALATED, TOOL_INVOKED
from usecases.replay import diff_traces, replay_trace


class Clock:
    def __init__(self):
        self.t = 0

    def now_ms(self):
        self.t += 1
        return self.t


def envelope(event="go", data=None, tool_calls=None):
    return {"event": event, "data": data or {},
            "tool_calls": tool_calls or []}


def minimal():
    """Smallest valid harness: a --(go)--> b, plus required terminals."""
    b = Harness.define("t").initial("a")
    b.state("a", handler=lambda ctx: envelope())
    b.state("b", terminal=True)
    b.transition("a", "go", "b")
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.escalation_state("esc").exhausted_state("exh")
    return b


def run_built(built, model=None, ctx=None):
    store = InMemoryStore()
    runner = built.runner(model or ScriptedModel([]), store, Clock())
    return runner.run(ctx or {}), store


# ------------------------------------------------------- build-time checks
def test_unreachable_state_rejected():
    b = minimal()
    b.state("lost", handler=lambda ctx: envelope())
    try:
        b.build()
    except DefinitionError:
        return
    raise AssertionError("unreachable state was accepted")


def test_dead_end_rejected():
    b = minimal()
    # reroute a -> c, leaving b unreachable AND c a dead end
    b2 = Harness.define("t").initial("a")
    b2.state("a", handler=lambda ctx: envelope())
    b2.state("c", handler=lambda ctx: envelope())
    b2.transition("a", "go", "c")
    b2.state("esc", terminal=True)
    b2.state("exh", terminal=True)
    b2.escalation_state("esc").exhausted_state("exh")
    try:
        b2.build()
    except DefinitionError:
        return
    raise AssertionError("dead-end state was accepted")


def test_terminal_with_outbound_rejected():
    b = minimal()
    b.transition("b", "go", "a")
    try:
        b.build()
    except DefinitionError:
        return
    raise AssertionError("terminal with outbound was accepted")


def test_escalation_must_be_terminal():
    b = Harness.define("t").initial("a")
    b.state("a", handler=lambda ctx: envelope())
    b.state("b", terminal=True)
    b.transition("a", "go", "b")
    b.state("esc", handler=lambda ctx: envelope())  # not terminal!
    b.state("exh", terminal=True)
    b.escalation_state("esc").exhausted_state("exh")
    try:
        b.build()
    except DefinitionError:
        return
    raise AssertionError("non-terminal escalation state was accepted")


def test_cycle_requires_step_budget():
    b = Harness.define("t").initial("a")
    b.state("a", handler=lambda ctx: envelope("loop"))
    b.state("b", handler=lambda ctx: envelope("loop"))
    b.transition("a", "loop", "b")
    b.transition("b", "loop", "a")
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.escalation_state("esc").exhausted_state("exh")
    try:
        b.build()
    except DefinitionError:
        pass
    else:
        raise AssertionError("cycle without step budget was accepted")
    b.budget(step_limit=10)  # now it must build
    b.build()


def test_prompt_state_requires_schema():
    b = minimal()
    b2 = Harness.define("t").initial("a")
    b2.state("a", prompt_template="hi")  # no output_schema
    b2.state("b", terminal=True)
    b2.transition("a", "go", "b")
    b2.state("esc", terminal=True)
    b2.state("exh", terminal=True)
    b2.escalation_state("esc").exhausted_state("exh")
    try:
        b2.build()
    except DefinitionError:
        return
    raise AssertionError("prompt state without schema was accepted")


# ------------------------------------------------------------- runner
def test_happy_path():
    outcome, _ = run_built(minimal().build())
    assert outcome.final_state == "b", outcome.final_state
    assert outcome.success is True
    assert any(e.type == COMPLETED for e in outcome.events)


def test_exit_gate_correction_then_success():
    b = Harness.define("t").initial("a")
    b.state("a", prompt_template="x? $correction",
            output_schema={"type": "object",
                           "properties": {"x": {"type": "integer"}},
                           "required": ["x"]},
            exit_gates=("positive",))
    b.state("b", terminal=True)
    b.transition("a", "go", "b")
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.escalation_state("esc").exhausted_state("exh")
    b.register_gate("positive", PredicateGate(
        lambda s: (s["data"]["x"] > 0, "x must be positive")))
    model = ScriptedModel([envelope("go", {"x": -1}),
                           envelope("go", {"x": 2})])
    outcome, _ = run_built(b.build(), model=model)
    assert outcome.final_state == "b", outcome.final_state
    assert outcome.context["x"] == 2
    assert len(model.prompts) == 2  # one correction happened


def test_exit_gate_escalation_after_budget():
    b = Harness.define("t").initial("a")
    b.state("a", prompt_template="x? $correction",
            output_schema={"type": "object"},
            exit_gates=("never",), max_corrections=1)
    b.state("b", terminal=True)
    b.transition("a", "go", "b")
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.escalation_state("esc").exhausted_state("exh")
    b.register_gate("never", PredicateGate(lambda s: (False, "nope")))
    model = ScriptedModel([envelope("go", {}), envelope("go", {})])
    outcome, _ = run_built(b.build(), model=model)
    assert outcome.final_state == "esc", outcome.final_state
    assert outcome.success is False
    assert any(e.type == ESCALATED for e in outcome.events)


def test_entry_gate_failure_escalates():
    b = Harness.define("t").initial("a")
    b.state("a", handler=lambda ctx: envelope(), entry_gates=("g",))
    b.state("b", terminal=True)
    b.transition("a", "go", "b")
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.escalation_state("esc").exhausted_state("exh")
    b.register_gate("g", PredicateGate(lambda s: (False, "bad entry")))
    outcome, _ = run_built(b.build())
    assert outcome.final_state == "esc", outcome.final_state


def test_illegal_tool_corrected():
    calls = []

    def tool(args):
        calls.append(args)
        return {"ok": True}

    b = Harness.define("t").initial("a")
    b.state("a", prompt_template="do it $correction",
            output_schema={"type": "object"},
            tools=("allowed",))
    b.state("b", terminal=True)
    b.transition("a", "go", "b")
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.escalation_state("esc").exhausted_state("exh")
    b.register_tool("allowed", tool, legal_in={"a"},
                    args_schema={"type": "object",
                                 "properties": {"n": {"type": "integer"}}})
    # "sneaky" is deliberately NOT registered and NOT in state's tool set:
    # the runner's capability check must reject it before the registry.
    model = ScriptedModel([
        envelope("go", {}, [{"name": "sneaky", "args": {}}]),   # illegal
        envelope("go", {}, [{"name": "allowed", "args": {"n": 1}}]),
    ])
    outcome, _ = run_built(b.build(), model=model)
    assert outcome.final_state == "b", outcome.final_state
    assert calls == [{"n": 1}], calls  # sneaky never ran


def test_budget_exhaustion():
    b = minimal().budget(step_limit=1)
    outcome, _ = run_built(b.build())
    assert outcome.final_state == "exh", outcome.final_state
    assert outcome.success is False


def test_tool_idempotent_across_correction():
    calls = []

    def tool(args):
        calls.append(args)
        return {"ok": True}

    b = Harness.define("t").initial("a")
    b.state("a", prompt_template="do it $correction",
            output_schema={"type": "object"},
            tools=("t1",), exit_gates=("g",))
    b.state("b", terminal=True)
    b.transition("a", "go", "b")
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.escalation_state("esc").exhausted_state("exh")
    b.register_tool("t1", tool, legal_in={"a"},
                    args_schema={"type": "object",
                                 "properties": {
                                     "fixed": {"type": "integer"}},
                                 "required": ["fixed"]})
    b.register_gate("g", PredicateGate(
        lambda s: (s["context"].get("tried", 0) > 0, "not yet")))
    # Both attempts emit the SAME tool call: same args -> same idempotency key.
    # First attempt fails the exit gate (no 'tried' yet); second passes.
    seen = {"n": 0}

    def responder(prompt, schema):
        seen["n"] += 1
        return envelope("go", {"tried": seen["n"]},
                        [{"name": "t1", "args": {"fixed": 1}}])

    from adapters.model.models import FnModel
    outcome, _ = run_built(b.build(), model=FnModel(responder))
    assert outcome.final_state == "b", outcome.final_state
    assert len(calls) == 1, f"tool ran {len(calls)}x, expected 1 (idempotent)"


def test_replay_deterministic():
    built = minimal().build()
    o1, _ = run_built(built)
    o2, _ = run_built(built)
    d = diff_traces(replay_trace(o1.events), replay_trace(o2.events))
    assert d == [], d


def test_schema_gate_adapter():
    gate = SchemaGate({"type": "object",
                       "properties": {"a": {"type": "string"}},
                       "required": ["a"]})
    assert gate.check({"data": {"a": "x"}}).passed
    assert not gate.check({"data": {}}).passed
    assert not gate.check({"data": {"a": 1}}).passed


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
