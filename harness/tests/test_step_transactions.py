"""Step transactions (v1.1 durability pack). Plain asserts:
    python3 harness/tests/test_step_transactions.py
Covers the database model for durable execution: a step is a
transaction whose TRANSITION is committed by a STEP_COMMITTED record
(an uncommitted transition is re-driven on resume, never folded),
single-writer leases on every store, and torn-tail discard in the
JSONL store (a partial final line is an uncommitted fragment and
never happened; corruption mid-log still raises)."""

import json
import os
import sys
import time
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adapters.events.stores import InMemoryStore, JsonlStore, SqliteStore
from adapters.model.models import ScriptedModel
from app.builder import Harness
from core.errors import HarnessError
from core.events import (
    RUN_STARTED, STEP_COMMITTED, TRANSITION, make_event,
)
from usecases.replay import reconstruct_run


class Clock:
    def __init__(self):
        self.t = 0

    def now_ms(self):
        self.t += 1
        return self.t


RESPONSES = [
    {"event": "go", "data": {"x": 1}, "tool_calls": []},
    {"event": "go", "data": {"y": 2}, "tool_calls": []},
]


def build_two_state():
    b = Harness.define("txn-test").initial("a")
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
    return b.build()


# ------------------------------------------------------------ commit records
def test_commit_record_follows_each_transition():
    built = build_two_state()
    store = InMemoryStore()
    outcome = built.runner(
        ScriptedModel(list(RESPONSES)), store, Clock()
    ).run({"request": "r1"}, execution_id="run-c")
    assert outcome.success
    events = store.read("run-c")
    assert events[0].type == RUN_STARTED
    assert events[0].payload["commit_records"] == 2
    transitions = [e for e in events if e.type == TRANSITION]
    commits = [e for e in events if e.type == STEP_COMMITTED]
    assert len(transitions) == len(commits) == 2
    for transition, commit in zip(transitions, commits):
        assert commit.seq == transition.seq + 1
        assert commit.payload["transition_seq"] == transition.seq
        assert commit.payload["state"] == transition.payload["from"]
        assert commit.payload["to"] == transition.payload["to"]


def test_uncommitted_transition_is_redriven_on_resume():
    with TemporaryDirectory() as td:
        path = os.path.join(td, "events.jsonl")
        ref = build_two_state().runner(
            ScriptedModel(list(RESPONSES)), InMemoryStore(), Clock()
        ).run({"request": "r1"}, execution_id="run-u")
        assert ref.success

        built = build_two_state()
        built.runner(
            ScriptedModel(list(RESPONSES)), JsonlStore(path), Clock()
        ).run({"request": "r1"}, execution_id="run-u")

        # Simulate a crash after state A's TRANSITION but before its
        # commit record: keep the log only through that TRANSITION.
        with open(path, encoding="utf-8") as fh:
            lines = fh.readlines()
        cut = next(i for i, line in enumerate(lines)
                   if '"type": "transition"' in line)
        with open(path, "w", encoding="utf-8") as fh:
            fh.writelines(lines[:cut + 1])

        plan = reconstruct_run(JsonlStore(path).read("run-u"))
        assert plan.context == {"request": "r1"}, plan.context
        assert plan.current_state == "a", plan

        resumed = build_two_state().resume(
            ScriptedModel(list(RESPONSES)), JsonlStore(path), Clock(),
            execution_id="run-u")
        assert resumed.success and resumed.final_state == "done"
        assert resumed.context == ref.context, (resumed.context,
                                                ref.context)
        final = JsonlStore(path).read("run-u")
        assert [e.seq for e in final] == list(range(len(final)))
        commits = [e for e in final if e.type == STEP_COMMITTED]
        assert len(commits) == 2


def test_legacy_log_transition_still_commits():
    events = [
        make_event("old", 0, 0, RUN_STARTED,
                   {"initial_context": {"a": 1}, "initial_state": "s1"}),
        make_event("old", 1, 0, "state_entered", {"state": "s1"}),
        make_event("old", 2, 0, TRANSITION,
                   {"from": "s1", "event": "go", "to": "s2",
                    "context_delta": {"b": 2}, "context_removed": []}),
        make_event("old", 3, 0, "state_entered", {"state": "s2"}),
    ]
    plan = reconstruct_run(events)
    assert plan.context == {"a": 1, "b": 2}, plan.context
    assert plan.current_state == "s2", plan


# ------------------------------------------------------------------- leases
def _lease_roundtrip(store):
    assert store.acquire_lease("exec", "owner-1", 60_000) is True
    assert store.acquire_lease("exec", "owner-2", 60_000) is False
    assert store.acquire_lease("exec", "owner-1", 60_000) is True
    store.release_lease("exec", "owner-2")  # not the holder: no-op
    assert store.acquire_lease("exec", "owner-2", 60_000) is False
    store.release_lease("exec", "owner-1")
    assert store.acquire_lease("exec", "owner-2", 60_000) is True
    store.release_lease("exec", "owner-2")
    # Expiry is the dead-writer cleanup: a 1ms lease is free at once.
    assert store.acquire_lease("exec", "owner-1", 1) is True
    time.sleep(0.02)
    assert store.acquire_lease("exec", "owner-2", 60_000) is True
    store.release_lease("exec", "owner-2")


def test_leases_on_all_stores():
    _lease_roundtrip(InMemoryStore())
    with TemporaryDirectory() as td:
        _lease_roundtrip(JsonlStore(os.path.join(td, "e.jsonl")))
        # A second instance over the same file sees the lease.
        path = os.path.join(td, "e2.jsonl")
        first, second = JsonlStore(path), JsonlStore(path)
        assert first.acquire_lease("exec", "owner-1", 60_000) is True
        assert second.acquire_lease("exec", "owner-2", 60_000) is False
        first.release_lease("exec", "owner-1")
    with TemporaryDirectory() as td:
        path = os.path.join(td, "e.db")
        _lease_roundtrip(SqliteStore(path))
        first, second = SqliteStore(path), SqliteStore(path)
        assert first.acquire_lease("exec", "owner-1", 60_000) is True
        assert second.acquire_lease("exec", "owner-2", 60_000) is False
        first.release_lease("exec", "owner-1")


def test_runner_refuses_when_leased():
    built = build_two_state()
    store = InMemoryStore()
    assert store.acquire_lease("run-l", "other-writer", 60_000)
    runner = built.runner(ScriptedModel(list(RESPONSES)), store, Clock())
    try:
        runner.run({"request": "r1"}, execution_id="run-l")
    except HarnessError as exc:
        assert "leased by another writer" in str(exc)
    else:
        raise AssertionError("run proceeded under another writer's lease")
    assert store.read("run-l") == []  # nothing was appended
    store.release_lease("run-l", "other-writer")
    outcome = built.runner(
        ScriptedModel(list(RESPONSES)), store, Clock()
    ).run({"request": "r1"}, execution_id="run-l")
    assert outcome.success


# ---------------------------------------------------------------- torn tail
def test_torn_tail_discarded_and_reported():
    with TemporaryDirectory() as td:
        path = os.path.join(td, "e.jsonl")
        store = JsonlStore(path)
        for seq in range(3):
            store.append(make_event("r", seq, seq, "state_entered",
                                    {"state": "a"}))
        with open(path, "a", encoding="utf-8") as fh:
            fh.write('{"execution_id": "r", "seq": 3, "type": "trans')
        events = store.read("r")
        assert [e.seq for e in events] == [0, 1, 2]
        assert store.last_discarded_tail is not None
        assert "trans" in store.last_discarded_tail
        assert store.list_runs() == ["r"]


def test_mid_log_corruption_raises():
    with TemporaryDirectory() as td:
        path = os.path.join(td, "e.jsonl")
        store = JsonlStore(path)
        store.append(make_event("r", 0, 0, "state_entered",
                                {"state": "a"}))
        with open(path, "a", encoding="utf-8") as fh:
            fh.write("this is not json\n")
        store.append(make_event("r", 1, 1, "state_entered",
                                {"state": "a"}))
        try:
            store.read("r")
        except ValueError as exc:
            assert "line 2" in str(exc)
        else:
            raise AssertionError("mid-log corruption was tolerated")


def test_resume_reports_discarded_tail():
    with TemporaryDirectory() as td:
        path = os.path.join(td, "e.jsonl")
        built = build_two_state()
        built.runner(
            ScriptedModel(list(RESPONSES)), JsonlStore(path), Clock()
        ).run({"request": "r1"}, execution_id="run-t")
        with open(path, "a", encoding="utf-8") as fh:
            fh.write('{"execution_id": "run-t", "seq": 99, "type": "sta')
        outcome = build_two_state().resume(
            ScriptedModel([]), JsonlStore(path), Clock(),
            execution_id="run-t")
        assert outcome.success
        assert any("torn write" in note for note in outcome.notes), \
            outcome.notes


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
