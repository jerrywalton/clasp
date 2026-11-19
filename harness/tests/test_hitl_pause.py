"""HITL pause (#10, v1.1). Plain asserts:
    python3 harness/tests/test_hitl_pause.py
Covers the configurable pause as built: the awaiting state is a log
event and survives process exit (here: separate store instances over
one SQLite file), no decision means no resume, decisions are
validated against the request in the log (allowed set + approver),
approve proceeds, amend applies its logged patch, deny escalates,
timeout expiry is fail-closed, approve-and-remember waives later
pauses on the same state, and the policy itself is validated at
build time."""

import os
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adapters.decisions import render_pause_request
from adapters.events.stores import InMemoryStore, SqliteStore
from adapters.model.models import ScriptedModel
from app.builder import Harness
from core.errors import DefinitionError, HarnessError
from core.events import (
    ESCALATED, MODEL_CALLED, PAUSE_DECIDED, PAUSE_REQUESTED,
)
from usecases.decisions import pending_pause, record_decision


class Clock:
    def __init__(self):
        self.t = 0

    def now_ms(self):
        self.t += 1
        return self.t


RESP_A = {"event": "go", "data": {"x": 1}, "tool_calls": []}
RESP_B = {"event": "go", "data": {"y": 2}, "tool_calls": []}


def build_pause_machine(**pause_kwargs):
    kwargs = {"approver": "ops-lead"}
    kwargs.update(pause_kwargs)
    b = Harness.define("pause-test").initial("a")
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
    b.pause("b", **kwargs)
    return b.build()


def paused_run(path, **pause_kwargs):
    """Run to the pause on a fresh store; return (built, outcome)."""
    built = build_pause_machine(**pause_kwargs)
    outcome = built.runner(
        ScriptedModel([RESP_A, RESP_B]), SqliteStore(path), Clock()
    ).run({"request": "r1"}, execution_id="run-p")
    return built, outcome


def test_pause_stops_run_and_persists():
    with TemporaryDirectory() as td:
        path = os.path.join(td, "e.db")
        _, outcome = paused_run(path)
        assert outcome.awaiting_decision is True
        assert outcome.success is False
        assert outcome.final_state == "b"
        events = SqliteStore(path).read("run-p")
        assert events[-1].type == PAUSE_REQUESTED
        assert events[-1].payload["approver"] == "ops-lead"
        # State B's work never ran: the model was called once (A).
        assert len([e for e in events
                    if e.type == MODEL_CALLED]) == 1
        request = pending_pause(SqliteStore(path), "run-p")
        assert request is not None
        text = render_pause_request(request)
        assert "ops-lead" in text and "approve" in text \
            and "No decision, no resume." in text


def test_resume_without_decision_refuses():
    with TemporaryDirectory() as td:
        path = os.path.join(td, "e.db")
        paused_run(path)
        before = SqliteStore(path).read("run-p")
        try:
            build_pause_machine().resume(
                ScriptedModel([RESP_B]), SqliteStore(path), Clock(),
                execution_id="run-p")
        except HarnessError as exc:
            assert "no decision, no resume" in str(exc)
        else:
            raise AssertionError("resume proceeded without a decision")
        assert SqliteStore(path).read("run-p") == before


def test_approve_and_resume_completes():
    with TemporaryDirectory() as td:
        path = os.path.join(td, "e.db")
        paused_run(path)
        event = record_decision(
            SqliteStore(path), Clock(), "run-p",
            decision="approve", decided_by="ops-lead")
        assert event.type == PAUSE_DECIDED
        outcome = build_pause_machine().resume(
            ScriptedModel([RESP_B]), SqliteStore(path), Clock(),
            execution_id="run-p")
        assert outcome.success and outcome.final_state == "done"
        assert outcome.context["y"] == 2
        types = [e.type for e in SqliteStore(path).read("run-p")]
        assert types.index(PAUSE_DECIDED) > types.index(PAUSE_REQUESTED)


def test_deny_escalates():
    with TemporaryDirectory() as td:
        path = os.path.join(td, "e.db")
        paused_run(path)
        record_decision(
            SqliteStore(path), Clock(), "run-p",
            decision="deny", decided_by="ops-lead")
        outcome = build_pause_machine().resume(
            ScriptedModel([]), SqliteStore(path), Clock(),
            execution_id="run-p")
        assert outcome.success is False
        assert outcome.final_state == "esc"
        esc = [e for e in SqliteStore(path).read("run-p")
               if e.type == ESCALATED]
        assert esc and esc[-1].payload["phase"] == "pause"
        assert esc[-1].payload["failures"][0][0] == "pause_denied"


def test_decision_validation_against_the_log():
    with TemporaryDirectory() as td:
        path = os.path.join(td, "e.db")
        paused_run(path)
        store = SqliteStore(path)
        for kwargs, fragment in (
                ({"decision": "approve", "decided_by": "someone-else"},
                 "approver"),
                ({"decision": "amend", "decided_by": "ops-lead"},
                 "not an allowed decision"),
                ({"decision": "approve", "decided_by": "ops-lead",
                  "context_patch": {"x": 9}},
                 "context_patch belongs")):
            try:
                record_decision(store, Clock(), "run-p", **kwargs)
            except HarnessError as exc:
                assert fragment in str(exc), (fragment, str(exc))
            else:
                raise AssertionError(f"decision accepted: {kwargs}")
        record_decision(store, Clock(), "run-p",
                        decision="approve", decided_by="ops-lead")
        try:
            record_decision(store, Clock(), "run-p",
                            decision="approve", decided_by="ops-lead")
        except HarnessError as exc:
            assert "already decided" in str(exc)
        else:
            raise AssertionError("a second decision was recorded")


def test_amend_applies_its_logged_patch():
    with TemporaryDirectory() as td:
        path = os.path.join(td, "e.db")
        b = Harness.define("amend-test").initial("a")
        b.state("a", prompt_template="A $correction",
                output_schema={"type": "object"})
        b.state("c", handler=lambda ctx: {
            "event": "go", "data": {"seen_limit": ctx["limit"]},
            "tool_calls": []})
        b.state("done", terminal=True)
        b.state("esc", terminal=True)
        b.state("exh", terminal=True)
        b.transition("a", "go", "c")
        b.transition("c", "go", "done")
        b.escalation_state("esc").exhausted_state("exh")
        b.budget(step_limit=10, token_limit=100000)
        b.pause("c", approver="ops-lead",
                decisions=("approve", "deny", "amend"))
        built = b.build()
        outcome = built.runner(
            ScriptedModel([RESP_A]), SqliteStore(path), Clock()
        ).run({"limit": 3}, execution_id="run-a")
        assert outcome.awaiting_decision
        record_decision(
            SqliteStore(path), Clock(), "run-a", decision="amend",
            decided_by="ops-lead", context_patch={"limit": 10})
        resumed = b.build().resume(
            ScriptedModel([]), SqliteStore(path), Clock(),
            execution_id="run-a")
        assert resumed.success
        assert resumed.context["seen_limit"] == 10, resumed.context


def test_timeout_expiry_is_fail_closed():
    with TemporaryDirectory() as td:
        path = os.path.join(td, "e.db")
        clock = Clock()
        built = build_pause_machine(timeout_ms=50)
        outcome = built.runner(
            ScriptedModel([RESP_A, RESP_B]), SqliteStore(path), clock
        ).run({"request": "r1"}, execution_id="run-p")
        assert outcome.awaiting_decision
        clock.t += 10_000  # long past the declared timeout
        resumed = build_pause_machine(timeout_ms=50).resume(
            ScriptedModel([]), SqliteStore(path), clock,
            execution_id="run-p")
        assert resumed.success is False
        assert resumed.final_state == "esc"
        esc = [e for e in SqliteStore(path).read("run-p")
               if e.type == ESCALATED]
        assert esc[-1].payload["failures"][0][0] == "pause_timeout"


def test_remember_waives_later_pauses_on_the_state():
    records = {"entries": 0}

    def b_work(ctx):
        records["entries"] += 1
        return {"event": "again" if records["entries"] == 1
                else "finish",
                "data": {"visits": records["entries"]},
                "tool_calls": []}

    b = Harness.define("remember-test").initial("a")
    b.state("a", prompt_template="A $correction",
            output_schema={"type": "object"})
    b.state("b", handler=b_work)
    b.state("done", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("a", "go", "b")
    b.transition("b", "again", "a")
    b.transition("b", "finish", "done")
    b.escalation_state("esc").exhausted_state("exh")
    b.budget(step_limit=10, token_limit=100000)
    b.pause("b", approver="ops-lead")
    store = InMemoryStore()
    outcome = b.build().runner(
        ScriptedModel([RESP_A, RESP_A]), store, Clock()
    ).run({"request": "r1"}, execution_id="run-m")
    assert outcome.awaiting_decision
    record_decision(store, Clock(), "run-m", decision="approve",
                    decided_by="ops-lead", remember=True)
    resumed = b.build().resume(
        ScriptedModel([RESP_A]), store, Clock(), execution_id="run-m")
    assert resumed.success and resumed.final_state == "done"
    requests = [e for e in store.read("run-m")
                if e.type == PAUSE_REQUESTED]
    assert len(requests) == 1, requests
    assert resumed.context["visits"] == 2


def test_policy_validated_at_build_time():
    def attempt(**pause_kwargs):
        b = Harness.define("bad-pause").initial("a")
        b.state("a", prompt_template="A $correction",
                output_schema={"type": "object"})
        b.state("done", terminal=True)
        b.state("esc", terminal=True)
        b.state("exh", terminal=True)
        b.transition("a", "go", "done")
        b.escalation_state("esc").exhausted_state("exh")
        b.budget(step_limit=10, token_limit=1000)
        b.pause("a", **pause_kwargs)
        return b

    for kwargs in ({"approver": ""},
                   {"approver": "ops", "decisions": ()},
                   {"approver": "ops", "decisions": ("maybe",)},
                   {"approver": "ops",
                    "decisions": ("approve", "approve")},
                   {"approver": "ops", "timeout_ms": -5},
                   {"approver": "ops", "timeout_ms": 0},
                   {"approver": "ops", "channel": ""}):
        try:
            attempt(**kwargs).build()
        except DefinitionError:
            continue
        raise AssertionError(f"pause policy built: {kwargs}")
    try:
        attempt(approver="ops").pause("ghost", approver="ops").build()
    except DefinitionError:
        pass
    else:
        raise AssertionError("pause on an undefined state built")


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
