"""fingerprint_version 2 tests: content identity for everything that
bears execution semantics. Plain asserts:
    python3 harness/tests/test_fingerprint_v2.py
Covers: handler body edited under the same name/module refused on
resume (and named), cosmetic-only handler changes accepted, closure
value changes refused, guard changes refused, gate config changes
refused (and named), model descriptor mismatches refused, and the
fingerprint_version 1 fallback (legacy comparison + explicit note).
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adapters.events.stores import InMemoryStore
from adapters.gates.checks import SchemaGate
from adapters.model.models import ScriptedModel
from app.builder import Harness
from core.errors import HarnessError
from core.events import RUN_STARTED, make_event
from usecases.runner import (
    _callable_digest, _definition_fingerprint,
)


class Clock:
    def __init__(self):
        self.t = 0

    def now_ms(self):
        self.t += 1
        return self.t


class CrashModel:
    """Crashes on the first generate call (the process 'dies' in the
    in-flight prompt state)."""

    def generate(self, prompt, schema):
        raise RuntimeError("simulated process crash")


class ExplodingModel:
    def generate(self, prompt, schema):
        raise AssertionError("model called despite resume refusal")


class NamedModel:
    """Model adapter exposing a stable model identity."""

    def __init__(self, name, inner=None, crash=False):
        self.model_name = name
        self._inner = inner or ScriptedModel([B_RESPONSE])
        self._crash = crash

    def generate(self, prompt, schema):
        if self._crash:
            raise RuntimeError("simulated process crash")
        return self._inner.generate(prompt, schema)


B_RESPONSE = {"event": "b_done", "data": {"y": 2}, "tool_calls": []}

HANDLER_V1 = (
    "def handler(ctx):\n"
    "    return {'event': 'go', 'data': {'v': 1}, 'tool_calls': []}\n")
HANDLER_V2_BODY = (
    "def handler(ctx):\n"
    "    return {'event': 'go', 'data': {'v': 2}, 'tool_calls': []}\n")
HANDLER_V1_COSMETIC = (
    "# a leading comment\n"
    "def handler(ctx):\n"
    "    # the same body, reformatted, with comments\n"
    "    return {'event': 'go', 'data': {'v': 1}, 'tool_calls': []}  # x\n")
HANDLER_RENAMED = (
    "def handler_under_another_name(ctx):\n"
    "    return {'event': 'go', 'data': {'v': 1}, 'tool_calls': []}\n")

GUARD_TRUE = "def guard(ctx):\n    return True\n"
GUARD_TRUE_OTHER_BODY = "def guard(ctx):\n    return 1 == 1\n"

Y_SCHEMA = {"type": "object",
            "properties": {"y": {"type": "integer"}},
            "required": ["y"]}
Y_SCHEMA_CHANGED = {"type": "object",
                    "properties": {"y": {"type": "string"}},
                    "required": ["y"]}


def make_handler(source, module="genmod"):
    namespace: dict = {}
    exec(compile(source, "<generated>", "exec"), namespace)
    fn = next(v for v in namespace.values() if callable(v))
    fn.__module__ = module
    return fn


def make_closure_handler(value):
    def handler(ctx):
        return {"event": "go", "data": {"v": value}, "tool_calls": []}
    return handler


def build_fp(handler=None, guard=None, gate_schema=None):
    b = Harness.define("fp-fixture").initial("a")
    b.state("a", handler=handler or make_handler(HANDLER_V1))
    b.state("b", prompt_template="B $correction",
            output_schema={"type": "object"},
            exit_gates=("ygate",) if gate_schema is not None else ())
    b.state("done", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("a", "go", "b", guard=guard)
    b.transition("b", "b_done", "done")
    b.escalation_state("esc").exhausted_state("exh")
    if gate_schema is not None:
        b.register_gate("ygate", SchemaGate(gate_schema))
    b.budget(step_limit=10, token_limit=1000)
    return b.build()


def crashed_store(built, model=None, execution_id="fp-1"):
    """Run the fixture with a model that dies in state b; return the
    store holding the incomplete (atomic, v2) log."""
    store = InMemoryStore()
    runner = built.runner(model or CrashModel(), store, Clock())
    try:
        runner.run({"request": "r1"}, execution_id=execution_id)
    except RuntimeError:
        pass
    else:
        raise AssertionError("the simulated crash did not happen")
    return store


def expect_refusal(built, store, model=None, execution_id="fp-1"):
    try:
        built.resume(model or ExplodingModel(), store, Clock(),
                     execution_id=execution_id)
    except HarnessError as exc:
        return str(exc)
    raise AssertionError("resume was not refused")


# ------------------------------------------------------------- digest units
def test_callable_digest_properties():
    digest_a, id_only_a = _callable_digest(make_handler(HANDLER_V1))
    assert not id_only_a
    # Reformatting / comments: same digest.
    digest_cosmetic, _ = _callable_digest(make_handler(HANDLER_V1_COSMETIC))
    assert digest_cosmetic == digest_a
    # Renaming the function: same digest.
    digest_renamed, _ = _callable_digest(make_handler(HANDLER_RENAMED))
    assert digest_renamed == digest_a
    # Body edit (a different constant): different digest.
    digest_body, _ = _callable_digest(make_handler(HANDLER_V2_BODY))
    assert digest_body != digest_a
    # Default edit: different digest.
    def with_default(x=1):
        return x
    def with_other_default(x=2):
        return x
    assert _callable_digest(with_default)[0] != \
        _callable_digest(with_other_default)[0]
    # Closure value edit: different digest (bytecode identical).
    assert _callable_digest(make_closure_handler(1))[0] != \
        _callable_digest(make_closure_handler(2))[0]
    # functools.partial: bound args are part of the digest.
    import functools
    def base(a, b):
        return a + b
    assert _callable_digest(functools.partial(base, 1))[0] != \
        _callable_digest(functools.partial(base, 2))[0]
    # C-level callable: identity-only fallback, marked as such.
    _, id_only_len = _callable_digest(len)
    assert id_only_len is True


def test_run_started_records_v2_components():
    built = build_fp(gate_schema=Y_SCHEMA)
    store = InMemoryStore()
    outcome = built.runner(ScriptedModel([B_RESPONSE]), store,
                           Clock()).run({"request": "r1"},
                                        execution_id="fp-ok")
    assert outcome.success
    started = next(e for e in outcome.events if e.type == RUN_STARTED)
    assert started.payload["fingerprint_version"] == 2
    components = started.payload["fingerprint_components"]
    assert {"definition", "handler:a", "gate:ygate", "model"} <= \
        set(components)
    assert components["model"] is None  # ScriptedModel exposes no name
    assert components["handler:a"]["digest"]
    assert "identity_only" not in components["handler:a"]


# ------------------------------------------------------------- resume refusals
def test_handler_body_edit_refused_and_named():
    store = crashed_store(build_fp())
    message = expect_refusal(
        build_fp(handler=make_handler(HANDLER_V2_BODY)), store)
    assert "handler for state 'a' changed since the run started" \
        in message, message


def test_handler_cosmetic_change_resumes():
    store = crashed_store(build_fp())
    resumed = build_fp(handler=make_handler(HANDLER_V1_COSMETIC)).resume(
        ScriptedModel([B_RESPONSE]), store, Clock(), execution_id="fp-1")
    assert resumed.final_state == "done" and resumed.success
    assert resumed.context["v"] == 1


def test_handler_closure_value_change_refused():
    store = crashed_store(build_fp(handler=make_closure_handler(1)))
    message = expect_refusal(
        build_fp(handler=make_closure_handler(2)), store)
    assert "handler for state 'a' changed since the run started" \
        in message, message


def test_guard_change_refused_and_named():
    guard_v1 = _make_guard(GUARD_TRUE)
    store = crashed_store(build_fp(guard=guard_v1))
    guard_v2 = _make_guard(GUARD_TRUE_OTHER_BODY)
    message = expect_refusal(build_fp(guard=guard_v2), store)
    assert "guard for transition 'a'" in message, message
    assert "changed since the run started" in message, message


def _make_guard(source):
    namespace: dict = {}
    exec(compile(source, "<generated-guard>", "exec"), namespace)
    fn = next(v for v in namespace.values() if callable(v))
    fn.__module__ = "genmod"
    return fn


def test_gate_config_change_refused_and_named():
    store = crashed_store(build_fp(gate_schema=Y_SCHEMA))
    message = expect_refusal(build_fp(gate_schema=Y_SCHEMA_CHANGED),
                             store)
    assert "gate 'ygate' changed since the run started" in message, \
        message
    # Control: the same gate config resumes fine.
    store2 = crashed_store(build_fp(gate_schema=Y_SCHEMA),
                           execution_id="fp-2")
    resumed = build_fp(gate_schema=Y_SCHEMA).resume(
        ScriptedModel([B_RESPONSE]), store2, Clock(),
        execution_id="fp-2")
    assert resumed.success and resumed.context["y"] == 2


def test_model_descriptor_mismatch_refused():
    store = crashed_store(build_fp(),
                          model=NamedModel("alpha", crash=True))
    message = expect_refusal(build_fp(), store,
                             model=NamedModel("beta"))
    assert "model changed since the run started" in message, message
    assert "'alpha'" in message and "'beta'" in message, message
    # Control: the same named model resumes fine.
    store2 = crashed_store(build_fp(),
                           model=NamedModel("alpha", crash=True),
                           execution_id="fp-3")
    resumed = build_fp().resume(
        NamedModel("alpha"), store2, Clock(), execution_id="fp-3")
    assert resumed.success


# ------------------------------------------------------------- v1 fallback
def test_v1_log_resumes_under_legacy_comparison_with_note():
    built = build_fp()
    store = crashed_store(built)
    events = store.read("fp-1")
    # Rewrite the log as fingerprint_version 1 wrote it: RUN_STARTED
    # carries only the legacy name-based fingerprint string.
    v1_store = InMemoryStore()
    for event in events:
        payload = dict(event.payload)
        if event.type == RUN_STARTED:
            payload.pop("fingerprint_version", None)
            payload.pop("fingerprint_components", None)
            payload.pop("model_descriptor", None)
            payload["definition_fingerprint"] = \
                _definition_fingerprint(built.definition)
        v1_store.append(make_event(event.execution_id, event.seq,
                                   event.ts_ms, event.type, payload))
    resumed = build_fp().resume(
        ScriptedModel([B_RESPONSE]), v1_store, Clock(),
        execution_id="fp-1")
    assert resumed.final_state == "done" and resumed.success
    notes = " ".join(resumed.notes)
    assert "fingerprint v1" in notes and "legacy name-based" in notes, \
        resumed.notes
    # And a v1 log whose (name-based) fingerprint does not match is
    # still refused under the legacy comparison.
    tampered = InMemoryStore()
    for event in v1_store.read("fp-1"):
        payload = dict(event.payload)
        if event.type == RUN_STARTED:
            payload["version"] = "9.9.9"
        tampered.append(make_event(event.execution_id, event.seq,
                                   event.ts_ms, event.type, payload))
    # version is part of the recorded shape only via the definition
    # the caller presents; present a different-version definition.
    b = Harness.define("fp-fixture", version="2.0.0").initial("a")
    b.state("a", handler=make_handler(HANDLER_V1))
    b.state("b", prompt_template="B $correction",
            output_schema={"type": "object"})
    b.state("done", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("a", "go", "b")
    b.transition("b", "b_done", "done")
    b.escalation_state("esc").exhausted_state("exh")
    b.budget(step_limit=10, token_limit=1000)
    message = expect_refusal(b.build(), tampered)
    assert "definition fingerprint differs" in message, message


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
