"""Durability + schema-union tests. Plain asserts, no framework:
    python3 harness/tests/test_durability_unions.py
Covers: SqliteStore persistence across instances, append-only integrity,
store equivalence (memory / JSONL / SQLite), reconstruct_run folding,
resume-after-crash equivalence with an uninterrupted run, and union
types in core.validate (type lists and anyOf) including build-time
rejection of malformed union specs."""

import os
import sqlite3
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "examples"))

from adapters.events.stores import InMemoryStore, JsonlStore, SqliteStore
from adapters.model.models import FnModel, ScriptedModel
from app.builder import Harness
from core.errors import DefinitionError, HarnessError
from core.events import (
    COMPLETED, GATE_VERDICT, MODEL_CALLED, STATE_ENTERED, TRANSITION,
    make_event,
)
from core.validate import validate_data, validate_schema
from usecases.replay import diff_traces, reconstruct_run, replay_trace

import refund_desk


class Clock:
    def __init__(self):
        self.t = 0

    def now_ms(self):
        self.t += 1
        return self.t


def envelope(event="go", data=None, tool_calls=None):
    return {"event": event, "data": data or {},
            "tool_calls": tool_calls or []}


# ------------------------------------------------------------- stores
def test_sqlite_persists_across_instances():
    with TemporaryDirectory() as td:
        path = os.path.join(td, "events.db")
        e0 = make_event("run-a", 0, 100, STATE_ENTERED, {"state": "a"})
        e1 = make_event("run-a", 1, 101, TRANSITION,
                        {"from": "a", "event": "go", "to": "b"})
        e2 = make_event("run-b", 0, 102, STATE_ENTERED, {"state": "a"})
        first = SqliteStore(path)
        for event in (e0, e1, e2):
            first.append(event)
        second = SqliteStore(path)  # separate instance, same file
        assert second.read("run-a") == [e0, e1]
        assert second.read("run-b") == [e2]
        assert second.read("missing") == []
        assert second.list_runs() == ["run-a", "run-b"]
        try:
            second.append(e0)  # same (run, seq): must not rewrite history
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("duplicate (run, seq) append was accepted")


def test_all_stores_round_trip():
    events = [make_event("r1", 0, 1, STATE_ENTERED, {"state": "a"}),
              make_event("r1", 1, 2, COMPLETED, {"state": "b"}),
              make_event("r2", 0, 3, STATE_ENTERED, {"state": "a"})]
    memory = InMemoryStore()
    with TemporaryDirectory() as td:
        stores = [memory,
                  JsonlStore(os.path.join(td, "log.jsonl")),
                  SqliteStore(os.path.join(td, "log.db"))]
        for store in stores:
            for event in events:
                store.append(event)
            assert store.read("r1") == events[:2], type(store).__name__
            assert store.read("r2") == events[2:], type(store).__name__
            assert store.list_runs() == ["r1", "r2"], type(store).__name__
        # Fresh instances over the same files see the same history.
        assert JsonlStore(os.path.join(td, "log.jsonl")).read("r1") \
            == events[:2]
        assert SqliteStore(os.path.join(td, "log.db")).read("r1") \
            == events[:2]


# ------------------------------------------------------- reconstruction
def test_reconstruct_run_folds_state_and_budget():
    events = [
        make_event("r", 0, 0, STATE_ENTERED, {"state": "a"}),
        make_event("r", 1, 0, MODEL_CALLED,
                   {"state": "a", "prompt_tokens": 10,
                    "completion_tokens": 20}),
        make_event("r", 2, 0, TRANSITION,
                   {"from": "a", "event": "go", "to": "b"}),
        make_event("r", 3, 0, STATE_ENTERED, {"state": "b"}),
    ]
    plan = reconstruct_run(events)
    assert plan.execution_id == "r"
    assert plan.current_state == "b", plan
    assert plan.next_seq == 4
    assert plan.steps_used == 2, plan
    assert plan.tokens_used == 30, plan
    assert plan.model_calls == 1
    assert plan.transitions == (("a", "go", "b"),)
    assert plan.completed is False and plan.final_state is None
    empty = reconstruct_run([])
    assert empty.current_state is None and empty.next_seq == 0
    try:
        reconstruct_run(events + [make_event("other", 9, 0, COMPLETED,
                                             {"state": "b"})])
    except ValueError:
        pass
    else:
        raise AssertionError("multi-execution fold was accepted")


# ------------------------------------------------------------- resume
class CrashOnCall:
    """Model double: delegates to a ScriptedModel, then 'kills the
    process' with a RuntimeError on the Nth call (nothing is returned,
    so no MODEL_CALLED event is written for the interrupted call)."""

    def __init__(self, responses, crash_call):
        self._inner = ScriptedModel(responses)
        self._crash_call = crash_call
        self.calls = 0

    def generate(self, prompt, schema):
        self.calls += 1
        if self.calls == self._crash_call:
            raise RuntimeError("simulated process crash")
        return self._inner.generate(prompt, schema)


def resume_responses():
    return [
        {"event": "a_done", "data": {"x": 1},
         "tool_calls": [{"name": "record", "args": {"value": 1}}]},
        {"event": "b_done", "data": {"y": 2}, "tool_calls": []},
    ]


def build_resume_harness(records):
    def record(args):
        # External system, idempotent by construction: keyed write.
        return records.setdefault(args["value"],
                                  {"recorded": args["value"]})

    b = Harness.define("resume-test").initial("a")
    b.state("a", prompt_template="A $correction",
            output_schema={"type": "object"}, tools=("record",))
    b.state("b", prompt_template="B $correction",
            output_schema={"type": "object"})
    b.state("c", handler=lambda ctx: envelope(
        "c_done", {"z": ctx["x"] + ctx["y"]}))
    b.state("done", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("a", "a_done", "b")
    b.transition("b", "b_done", "c")
    b.transition("c", "c_done", "done")
    b.escalation_state("esc").exhausted_state("exh")
    b.register_tool("record", record, legal_in={"a"},
                    args_schema={"type": "object",
                                 "properties": {
                                     "value": {"type": "integer"}},
                                 "required": ["value"]})
    b.budget(step_limit=10, token_limit=1000)
    return b.build()


def test_resume_matches_uninterrupted():
    # Reference: the same definition + scripted model, never interrupted.
    ref_records: dict = {}
    ref_built = build_resume_harness(ref_records)
    ref_outcome = ref_built.runner(
        ScriptedModel(resume_responses()), InMemoryStore(), Clock()
    ).run({"request": "r1"}, execution_id="run-42")
    assert ref_outcome.final_state == "done" and ref_outcome.success

    records: dict = {}
    built = build_resume_harness(records)
    with TemporaryDirectory() as td:
        path = os.path.join(td, "events.db")
        store1 = SqliteStore(path)
        crashing = built.runner(
            CrashOnCall(resume_responses(), crash_call=2), store1, Clock())
        try:
            crashing.run({"request": "r1"}, execution_id="run-42")
        except RuntimeError:
            pass
        else:
            raise AssertionError("the simulated crash did not happen")

        store2 = SqliteStore(path)  # 'restarted process': fresh instances
        partial = store2.read("run-42")
        plan = reconstruct_run(partial)
        assert plan.current_state == "b", plan
        assert plan.steps_used == 2 and plan.tokens_used == 30, plan
        assert plan.completed is False
        assert store2.list_runs() == ["run-42"]

        rebuilt = build_resume_harness(records)  # same external system
        store3 = SqliteStore(path)
        # Phase B: committed state A is recreated by folding the log,
        # so the resume model only supplies the in-flight state B
        # response — no model call is made for A.
        resumed = rebuilt.resume(
            ScriptedModel(resume_responses()[1:]), store3, Clock(),
            execution_id="run-42", initial_context={"request": "r1"})

        assert resumed.final_state == ref_outcome.final_state
        assert resumed.success == ref_outcome.success
        assert resumed.context == ref_outcome.context, (
            resumed.context, ref_outcome.context)
        final_events = store3.read("run-42")
        assert diff_traces(replay_trace(final_events),
                           replay_trace(ref_outcome.events)) == []
        assert diff_traces(replay_trace(resumed.events),
                           replay_trace(ref_outcome.events)) == []
        # The prefix was not duplicated in the store.
        assert [e.seq for e in final_events] == \
            list(range(len(final_events)))
        # Idempotent tool effect landed exactly once across the restart.
        assert records == {1: {"recorded": 1}}, records


def test_resume_unknown_execution_rejected():
    built = build_resume_harness({})
    runner = built.runner(ScriptedModel([]), InMemoryStore(), Clock())
    try:
        runner.resume({}, execution_id="never-ran")
    except HarnessError:
        return
    raise AssertionError("resume of an unknown execution was accepted")


# ------------------------------------------------------------- unions
EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "order_id": {"type": ["string", "null"]},
        "amount_cents": {"type": ["integer", "null"]},
        "confidence": {"type": "number"},
    },
    "required": ["order_id", "amount_cents", "confidence"],
}


def test_union_type_list_accepts_members_and_null():
    good = {"order_id": "ORD-1", "amount_cents": 100, "confidence": 0.9}
    assert validate_data(good, EXTRACTION_SCHEMA) == []
    nulls = {"order_id": None, "amount_cents": None, "confidence": 0.0}
    assert validate_data(nulls, EXTRACTION_SCHEMA) == []
    assert validate_schema(EXTRACTION_SCHEMA) == []


def test_union_type_list_rejects_non_members():
    base = {"order_id": "ORD-1", "amount_cents": 100, "confidence": 0.9}
    for bad in ({"order_id": 42}, {"amount_cents": "100"},
                {"amount_cents": True}, {"order_id": ["ORD-1"]}):
        data = {**base, **bad}
        assert validate_data(data, EXTRACTION_SCHEMA) != [], bad
    # required still means present: a missing key is not a null value.
    assert validate_data({"amount_cents": 1, "confidence": 0.5},
                         EXTRACTION_SCHEMA) != []


def test_union_anyof_accepts_and_rejects():
    schema = {"anyOf": [{"type": "string"}, {"type": "null"}]}
    assert validate_data("x", schema) == []
    assert validate_data(None, schema) == []
    assert validate_data(7, schema) != []
    assert validate_schema(schema) == []
    nested = {"type": "object",
              "properties": {"note": {"anyOf": [{"type": "integer"},
                                                 {"type": "null"}]}},
              "required": ["note"]}
    assert validate_data({"note": None}, nested) == []
    assert validate_data({"note": 3}, nested) == []
    assert validate_data({"note": "three"}, nested) != []


def builder_with_schema(schema):
    b = Harness.define("union-test").initial("a")
    b.state("a", prompt_template="x $correction", output_schema=schema)
    b.state("b", terminal=True)
    b.transition("a", "go", "b")
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.escalation_state("esc").exhausted_state("exh")
    return b


def test_malformed_unions_rejected_at_build_time():
    malformed = [
        {"type": "object",
         "properties": {"x": {"type": []}}},                      # empty
        {"type": "object",
         "properties": {"x": {"type": ["string", "wat"]}}},       # unknown
        {"type": "object",
         "properties": {"x": {"type": ["string", 3]}}},           # non-str
        {"type": "object",
         "properties": {"x": {"anyOf": []}}},                     # empty
        {"type": "object",
         "properties": {"x": {"anyOf": [{"type": "string"},
                                        {"type": ["null", "wat"]}]}}},
    ]
    for schema in malformed:
        assert validate_schema(schema) != [], schema
        try:
            builder_with_schema(schema).build()
        except DefinitionError:
            continue
        raise AssertionError(f"malformed union schema was accepted: {schema}")
    # The well-formed forms build fine.
    builder_with_schema(EXTRACTION_SCHEMA).build()
    builder_with_schema(
        {"anyOf": [{"type": "string"}, {"type": "null"}]}).build()


def test_refund_desk_nullable_extraction_uses_schema_path():
    issue_refund, flag_for_review, ledger, review_queue = \
        refund_desk.make_tools()
    built = refund_desk.build((issue_refund, flag_for_review))
    # Clean approval: real values flow through the union-typed schema.
    store = InMemoryStore()
    outcome = built.runner(FnModel(refund_desk.model_fn), store,
                           refund_desk.Clock()).run(
        {"message": "Hi, order ORD-1001 arrived broken. "
                    "Please refund $42.50 to my card."})
    assert outcome.final_state == "done" and outcome.success
    assert outcome.context["order_id"] == "ORD-1001"
    assert ledger and ledger[0]["amount_cents"] == 4250
    # No order id / no amount: the model now emits nulls; the *schema*
    # accepts them (union), the semantic exit gate still refuses the
    # extraction, and the run escalates exactly as before.
    outcome2 = built.runner(FnModel(refund_desk.model_fn), InMemoryStore(),
                            refund_desk.Clock()).run(
        {"message": "my package never arrived, I want my money back"})
    assert outcome2.final_state == "escalated" and not outcome2.success
    # Proof the nulls passed the *schema*: exit-gate verdicts are only
    # logged after envelope + schema validation succeed. Had the schema
    # rejected the nulls, the correction loop would have burned its
    # attempts on schema failures and logged no exit verdict at all.
    exit_verdicts = [e for e in outcome2.events
                     if e.type == GATE_VERDICT
                     and e.payload.get("phase") == "exit"
                     and e.payload.get("state") == "classified"]
    assert exit_verdicts, "schema rejected the nullable extraction"
    assert all(not e.payload["passed"] for e in exit_verdicts)


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
