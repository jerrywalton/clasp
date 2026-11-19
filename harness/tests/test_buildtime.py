"""Build-time completeness tests (critique C1/C2 regression probes).

    python3 harness/tests/test_buildtime.py

One regression test per probe from docs/CRITIQUE-2026-10-27.md that
the build-time completeness batch closed:

  P4  undeclared envelope event -> correctable failure inside the
      correction loop, before any tool invocation (fix #5)
  P2  capability declared twice (StateDef.tools vs registry
      legal_in) -> cross-checked at build, both directions (fix #4)
  P3  tool-call arguments validated against declared per-tool
      argument schemas, at build and in-path (fix #3)
  P1  unknown single type names fail closed in both validators and
      at build (fix #6a)
  P18 array `items` schemas enforced in validate_data and
      validate_schema; unsupported schema keywords rejected (fix #6b)
  P13 malformed SchemaGate schema -> DefinitionError at build
      (fix #6c)
  P12 .budget() unknown kwarg -> DefinitionError at the call (fix #6d)
  P14 redefining a state -> DefinitionError, not silent overwrite
      (fix #6e)

The thesis, for all of them: a definition that builds must not be
able to fail structurally at runtime.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adapters.events.stores import InMemoryStore
from adapters.gates.checks import SchemaGate
from adapters.model.models import ScriptedModel
from app.builder import Harness
from core.errors import DefinitionError
from core.events import ESCALATED
from core.validate import validate_data, validate_schema


class Clock:
    def __init__(self):
        self.t = 0

    def now_ms(self):
        self.t += 1
        return self.t


def envelope(event="go", data=None, tool_calls=None):
    return {"event": event, "data": data or {},
            "tool_calls": tool_calls or []}


def base(tools=()):
    """a --go--> b with the required terminals; state a is a model
    state declaring `tools`."""
    b = Harness.define("t").initial("a")
    b.state("a", prompt_template="do it $correction",
            output_schema={"type": "object"}, tools=tools)
    b.state("b", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("a", "go", "b")
    b.escalation_state("esc").exhausted_state("exh")
    return b


def expect_definition_error(fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except DefinitionError as exc:
        return str(exc)
    raise AssertionError(f"expected DefinitionError from {fn!r}")


OBJECT = {"type": "object"}


# ------------------------------------------------------------------- P4
def test_p4_hallucinated_event_corrected_before_tools():
    calls = []

    def rec(args):
        calls.append(args)
        return {"ok": True}

    b = base(tools=("rec",))
    b.register_tool("rec", rec, legal_in={"a"},
                    args_schema={"type": "object",
                                 "properties": {"n": {"type": "integer"}}})
    built = b.build()
    model = ScriptedModel([
        # Undeclared event WITH a tool call: the call must not fire.
        envelope("hallucinated", {}, [{"name": "rec", "args": {"n": 1}}]),
        envelope("go", {}, [{"name": "rec", "args": {"n": 1}}]),
    ])
    outcome = built.runner(model, InMemoryStore(), Clock()).run({})
    assert outcome.final_state == "b", outcome.final_state
    assert outcome.success
    assert calls == [{"n": 1}], calls  # exactly one firing: the retry


def test_p4_persistent_hallucination_escalates_with_outcome():
    calls = []

    def rec(args):
        calls.append(args)
        return {"ok": True}

    b = base(tools=("rec",))
    b.register_tool("rec", rec, legal_in={"a"}, args_schema=OBJECT)
    built = b.build()
    bad = envelope("hallucinated", {}, [{"name": "rec", "args": {}}])
    model = ScriptedModel([bad, bad, bad])  # 1 + max_corrections(2)
    outcome = built.runner(model, InMemoryStore(), Clock()).run({})
    # No InvalidTransition escapes run(); the run ends in the declared
    # escalation state with a real Outcome, and no tool ever fired.
    assert outcome.final_state == "esc", outcome.final_state
    assert outcome.success is False
    assert calls == [], calls
    assert any(e.type == ESCALATED for e in outcome.events)


def test_p4_handler_hallucination_escalates():
    b = Harness.define("t").initial("a")
    b.state("a", handler=lambda ctx: envelope("nope"))
    b.state("b", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("a", "go", "b")
    b.escalation_state("esc").exhausted_state("exh")
    outcome = b.build().runner(
        ScriptedModel([]), InMemoryStore(), Clock()).run({})
    assert outcome.final_state == "esc", outcome.final_state


# ------------------------------------------------------------------- P2
def _pay_builder(state_tools, legal_in):
    b = base(tools=state_tools)
    b.register_tool("pay", lambda args: {"ok": True}, legal_in=legal_in,
                    args_schema=OBJECT)
    return b


def test_p2_declared_but_not_legal_rejected():
    # P2a, direction 1: state declares the tool, the registry does
    # not grant it there — used to build, then crash mid-flight on
    # the registry's refusal (or, with legal_in={"b"}, disagree in
    # both directions at once; either way the build must refuse).
    msg = expect_definition_error(_pay_builder(("pay",), set()).build)
    assert "not registered legal" in msg, msg
    expect_definition_error(_pay_builder(("pay",), {"b"}).build)


def test_p2_legal_but_not_declared_rejected():
    # P2a, direction 2: registry grants the tool in a state whose
    # declaration does not include it — a second source of truth.
    msg = expect_definition_error(_pay_builder((), {"a"}).build)
    assert "does not declare it" in msg, msg


def test_p2_legal_in_undefined_state_rejected():
    # P2b: legal_in naming a state that does not exist.
    msg = expect_definition_error(_pay_builder((), {"ghost"}).build)
    assert "undefined state 'ghost'" in msg, msg


def test_p2_matching_declarations_build():
    _pay_builder(("pay",), {"a"}).build()  # must not raise


# ------------------------------------------------------------------- P3
def test_p3_wrong_typed_args_corrected_before_tool():
    received = []

    def pay(args):
        received.append(args)
        return {"ok": True}

    b = base(tools=("pay",))
    b.register_tool(
        "pay", pay, legal_in={"a"},
        args_schema={"type": "object",
                     "properties": {"amount": {"type": "integer"}},
                     "required": ["amount"]})
    built = b.build()
    model = ScriptedModel([
        envelope("go", {}, [{"name": "pay",
                             "args": {"amount": "not-a-number"}}]),
        envelope("go", {}, [{"name": "pay", "args": {"amount": 500}}]),
    ])
    outcome = built.runner(model, InMemoryStore(), Clock()).run({})
    assert outcome.final_state == "b", outcome.final_state
    # The tool only ever saw schema-valid args.
    assert received == [{"amount": 500}], received


def test_p3_missing_argument_schema_rejected_at_build():
    b = base(tools=("pay",))
    b.register_tool("pay", lambda args: {}, legal_in={"a"})
    msg = expect_definition_error(b.build)
    assert "no declared argument schema" in msg, msg


def test_p3_malformed_argument_schema_rejected_at_build():
    b = base(tools=("pay",))
    b.register_tool("pay", lambda args: {}, legal_in={"a"},
                    args_schema={"type": "objct"})
    msg = expect_definition_error(b.build)
    assert "invalid argument schema" in msg, msg


# ------------------------------------------------------------------- P1
def test_p1_unknown_single_type_fails_closed():
    assert validate_schema({"type": "strnig"}) != []
    assert validate_schema(
        {"type": "object",
         "properties": {"x": {"type": "strnig"}}}) != []
    # validate_data produces a problem instead of silently passing.
    assert validate_data(12345, {"type": "strnig"}) != []
    assert validate_data("fine", {"type": "string"}) == []
    b2 = Harness.define("t").initial("a")
    b2.state("a", prompt_template="x $correction",
             output_schema={"type": "object",
                            "properties": {"x": {"type": "strnig"}}})
    b2.state("b", terminal=True)
    b2.state("esc", terminal=True)
    b2.state("exh", terminal=True)
    b2.transition("a", "go", "b")
    b2.escalation_state("esc").exhausted_state("exh")
    expect_definition_error(b2.build)


# ------------------------------------------------------------------ P18
def test_p18_array_items_enforced_both_layers():
    schema = {"type": "array", "items": {"type": "integer"}}
    assert validate_data([1, 2, 3], schema) == []
    assert validate_data([1, "x", None], schema) != []
    assert validate_data([], schema) == []
    # Nested: items inside properties, and items of items.
    nested = {"type": "object",
              "properties": {"rows": {"type": "array",
                                      "items": {"type": "array",
                                                "items": {"type": "string"}}}}}
    assert validate_data({"rows": [["a"], ["b", "c"]]}, nested) == []
    assert validate_data({"rows": [["a"], ["b", 3]]}, nested) != []
    assert validate_schema(schema) == []
    assert validate_schema({"type": "array",
                            "items": {"type": "strnig"}}) != []
    assert validate_schema({"type": "array", "items": "integer"}) != []


def test_p18_unsupported_keyword_rejected():
    assert validate_schema(
        {"type": "object", "additionalProperties": False}) != []
    assert validate_schema(
        {"type": "object",
         "properties": {"x": {"type": "string", "minLength": 2}}}) != []
    # The supported set itself is accepted.
    assert validate_schema(
        {"type": "object", "required": ["x"], "enum": [1],
         "properties": {"x": {"type": "array",
                              "items": {"type": "integer"}}},
         "anyOf": [{"type": "object"}]}) == []


def test_p18_items_schema_validated_at_build():
    b = Harness.define("t").initial("a")
    b.state("a", prompt_template="x $correction",
            output_schema={"type": "array",
                           "items": {"type": "strnig"}})
    b.state("b", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("a", "go", "b")
    b.escalation_state("esc").exhausted_state("exh")
    expect_definition_error(b.build)


# ------------------------------------------------------------------ P13
def test_p13_malformed_gate_schema_rejected_at_build():
    b = Harness.define("t").initial("a")
    b.state("a", prompt_template="do it $correction",
            output_schema={"type": "object"}, exit_gates=("g",))
    b.state("b", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("a", "go", "b")
    b.escalation_state("esc").exhausted_state("exh")
    b.register_gate("g", SchemaGate({"type": ["string", "wat"]}))
    msg = expect_definition_error(b.build)
    assert "gate 'g'" in msg, msg


def test_p13_wellformed_gate_schema_builds():
    b = Harness.define("t").initial("a")
    b.state("a", prompt_template="do it $correction",
            output_schema={"type": "object"}, exit_gates=("g",))
    b.state("b", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("a", "go", "b")
    b.escalation_state("esc").exhausted_state("exh")
    b.register_gate("g", SchemaGate({"type": "object"}))
    b.build()  # must not raise


# ------------------------------------------------------------------ P12
def test_p12_budget_typo_kwarg_rejected_at_the_call():
    b = Harness.define("t")
    msg = expect_definition_error(b.budget, step_lmit=5)
    assert "step_lmit" in msg, msg
    # The supported kwargs are accepted at the call.
    b.budget(step_limit=5, token_limit=100)


# ------------------------------------------------------------------ P14
def test_p14_duplicate_state_rejected():
    b = Harness.define("t").initial("a")
    b.state("a", handler=lambda ctx: envelope())
    msg = expect_definition_error(
        b.state, "a", handler=lambda ctx: envelope())
    assert "already defined" in msg, msg


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
