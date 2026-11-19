"""Resume preflight report. Plain asserts:
    python3 harness/tests/test_preflight.py
The report is a read-only fold: it appends nothing, calls no model,
and takes no lease. Cases: a completed run, a paused run, an
interrupted run, an empty log, and a torn tail."""

import os
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adapters.events.stores import InMemoryStore, JsonlStore, SqliteStore
from adapters.model.models import ScriptedModel
from app.builder import Harness
from core.events import PAUSE_REQUESTED
from usecases.preflight import resume_preflight


class Clock:
    def __init__(self):
        self.t = 0

    def now_ms(self):
        self.t += 1
        return self.t


class CrashOnCall(ScriptedModel):
    def __init__(self, responses, crash_call):
        super().__init__(responses)
        self._crash_call = crash_call
        self._calls = 0

    def generate(self, prompt, schema):
        self._calls += 1
        if self._calls == self._crash_call:
            raise RuntimeError("simulated crash")
        return super().generate(prompt, schema)


RESPONSES = [
    {"event": "go", "data": {"x": 1}, "tool_calls": []},
    {"event": "go", "data": {"y": 2}, "tool_calls": []},
]


def build_machine(pause=False):
    b = Harness.define("preflight-test").initial("a")
    b.state("a", prompt_template="A $correction",
            output_schema={"type": "object"})
    b.state("b", prompt_template="B $correction",
            output_schema={"type": "object"})
    b.state("done", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("a", "go", "b")
    b.transition("b", "go", "done")
    b.escalation_state("esc").exhausted_state("exh")
    b.budget(step_limit=10, token_limit=100000)
    if pause:
        b.pause("b", approver="ops-lead", channel="slack",
                decisions=("approve", "deny"))
    return b.build()


def test_completed_run_report():
    store = InMemoryStore()
    build_machine().runner(
        ScriptedModel(list(RESPONSES)), store, Clock()
    ).run({"request": "r1"}, execution_id="run-c")
    before = len(store.read("run-c"))
    report = resume_preflight(store, "run-c")
    assert "COMPLETED in state 'done'" in report
    assert "NOTHING TO RESUME" in report
    assert "2 step(s)" in report  # both commits are counted
    assert len(store.read("run-c")) == before  # read-only


def test_paused_run_report():
    store = InMemoryStore()
    build_machine(pause=True).runner(
        ScriptedModel(list(RESPONSES)), store, Clock()
    ).run({"request": "r1"}, execution_id="run-p")
    report = resume_preflight(store, "run-p")
    assert "PAUSED in state 'b'" in report
    assert "ops-lead" in report and "slack" in report
    assert "RECORD A DECISION FIRST" in report


def test_interrupted_run_report():
    with TemporaryDirectory() as td:
        path = os.path.join(td, "e.db")
        crashing = build_machine().runner(
            CrashOnCall(list(RESPONSES), crash_call=2),
            SqliteStore(path), Clock())
        try:
            crashing.run({"request": "r1"}, execution_id="run-i")
        except RuntimeError:
            pass
        report = resume_preflight(SqliteStore(path), "run-i")
        assert "INTERRUPTED in state 'b'" in report
        assert "READY TO RESUME" in report
        assert "folded, never re-run" in report


def test_empty_log_report():
    report = resume_preflight(InMemoryStore(), "nope")
    assert "NO LOG" in report and "NOTHING TO RESUME" in report


def test_torn_tail_is_reported():
    with TemporaryDirectory() as td:
        path = os.path.join(td, "e.jsonl")
        store = JsonlStore(path)
        build_machine().runner(
            ScriptedModel(list(RESPONSES)), store, Clock()
        ).run({"request": "r1"}, execution_id="run-t")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write('{"execution_id": "run-t", "seq": 99, "type": "sta')
        report = resume_preflight(JsonlStore(path), "run-t")
        assert "torn tail" in report
        assert "COMPLETED" in report


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
