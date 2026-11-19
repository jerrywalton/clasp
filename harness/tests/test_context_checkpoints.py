"""Context checkpoints (v1.1). Plain asserts:
    python3 harness/tests/test_context_checkpoints.py
Covers the backup half of the database model: context deltas ride
TRANSITIONs as events (already the fold), and a CONTEXT_CHECKPOINT
carrying the folded context is written at a commit boundary when a
declared trigger fires (steps, log bytes, or time). Recovery folds
from the latest checkpoint. The policy is definition material:
validated at build time and covered by the fingerprint."""

import os
import sys
from dataclasses import replace
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adapters.events.stores import InMemoryStore, SqliteStore
from adapters.model.models import ScriptedModel
from app.builder import Harness
from core.errors import DefinitionError, HarnessError
from core.events import CONTEXT_CHECKPOINT
from usecases.replay import diff_traces, reconstruct_run, replay_trace


class Clock:
    def __init__(self, step=1):
        self.t = 0
        self.step = step

    def now_ms(self):
        self.t += self.step
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


def envelope(event="go", data=None, tool_calls=None):
    return {"event": event, "data": data or {},
            "tool_calls": tool_calls or []}


def build_chain(checkpoint=None):
    b = Harness.define("ckpt-test").initial("a")
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
    if checkpoint is not None:
        b.checkpoint(**checkpoint)
    return b.build()


def checkpoints_of(events):
    return [e for e in events if e.type == CONTEXT_CHECKPOINT]


def test_steps_trigger_writes_folded_contexts():
    built = build_chain({"every_steps": 1})
    store = InMemoryStore()
    outcome = built.runner(
        ScriptedModel(list(RESPONSES)), store, Clock()
    ).run({"request": "r1"}, execution_id="run-s")
    assert outcome.success
    ckpts = checkpoints_of(store.read("run-s"))
    assert len(ckpts) == 3, ckpts
    assert all(c.payload["trigger"] == "steps" for c in ckpts)
    base = {"request": "r1", "_tool_results": {}}
    assert ckpts[0].payload["context"] == {**base, "x": 1}
    assert ckpts[1].payload["context"] == {**base, "x": 1, "y": 2}
    assert ckpts[2].payload["context"] == {**base, "x": 1, "y": 2,
                                           "z": 3}


def test_no_policy_no_checkpoints():
    built = build_chain()
    store = InMemoryStore()
    built.runner(
        ScriptedModel(list(RESPONSES)), store, Clock()
    ).run({"request": "r1"}, execution_id="run-n")
    assert checkpoints_of(store.read("run-n")) == []


def test_bytes_and_time_triggers():
    built = build_chain({"every_bytes": 10})
    store = InMemoryStore()
    built.runner(
        ScriptedModel(list(RESPONSES)), store, Clock()
    ).run({"request": "r1"}, execution_id="run-b")
    ckpts = checkpoints_of(store.read("run-b"))
    assert ckpts and ckpts[0].payload["trigger"] == "bytes", ckpts

    built = build_chain({"every_ms": 250})
    store = InMemoryStore()
    built.runner(
        ScriptedModel(list(RESPONSES)), store, Clock(step=100)
    ).run({"request": "r1"}, execution_id="run-t")
    ckpts = checkpoints_of(store.read("run-t"))
    assert ckpts and ckpts[0].payload["trigger"] == "time", ckpts


def test_fold_starts_from_checkpoint():
    built = build_chain({"every_steps": 1})
    store = InMemoryStore()
    built.runner(
        ScriptedModel(list(RESPONSES)), store, Clock()
    ).run({"request": "r1"}, execution_id="run-f")
    events = store.read("run-f")
    first_ckpt = next(i for i, e in enumerate(events)
                      if e.type == CONTEXT_CHECKPOINT)
    truncated = events[:first_ckpt + 1]
    # Tamper with the pre-checkpoint delta: recovery must trust the
    # checkpoint's folded context, not the doctored delta.
    tampered = [
        replace(e, payload={**e.payload,
                            "context_delta": {"x": 999}})
        if e.type == "transition" else e
        for e in truncated]
    plan = reconstruct_run(tampered)
    assert plan.context == {"request": "r1", "x": 1,
                            "_tool_results": {}}, plan.context
    assert plan.suffix_events == ()
    assert plan.current_state == "b"


def test_policy_validated_at_build_time():
    for kwargs in ({"every_steps": -1}, {"every_bytes": "big"},
                   {"every_ms": True}):
        b = Harness.define("bad-ckpt").initial("a")
        b.state("a", prompt_template="A $correction",
                output_schema={"type": "object"})
        b.state("done", terminal=True)
        b.state("esc", terminal=True)
        b.state("exh", terminal=True)
        b.transition("a", "go", "done")
        b.escalation_state("esc").exhausted_state("exh")
        b.budget(step_limit=10, token_limit=1000)
        b.checkpoint(**kwargs)
        try:
            b.build()
        except DefinitionError:
            continue
        raise AssertionError(f"checkpoint {kwargs} built without error")


def test_policy_is_fingerprinted():
    with TemporaryDirectory() as td:
        path = os.path.join(td, "e.db")
        build_chain({"every_steps": 1}).runner(
            ScriptedModel(list(RESPONSES)), SqliteStore(path), Clock()
        ).run({"request": "r1"}, execution_id="run-p")
        try:
            build_chain().resume(
                ScriptedModel([]), SqliteStore(path), Clock(),
                execution_id="run-p")
        except HarnessError as exc:
            assert "changed since the run started" in str(exc)
        else:
            raise AssertionError(
                "resume under a different checkpoint policy was accepted")


def test_resume_with_checkpoints_matches_uninterrupted():
    ref = build_chain({"every_steps": 1}).runner(
        ScriptedModel(list(RESPONSES)), InMemoryStore(), Clock()
    ).run({"request": "r1"}, execution_id="run-r")
    assert ref.success
    with TemporaryDirectory() as td:
        path = os.path.join(td, "e.db")
        crashing = build_chain({"every_steps": 1}).runner(
            CrashOnCall(list(RESPONSES), crash_call=2),
            SqliteStore(path), Clock())
        try:
            crashing.run({"request": "r1"}, execution_id="run-r")
        except RuntimeError:
            pass
        else:
            raise AssertionError("the simulated crash did not happen")
        resumed = build_chain({"every_steps": 1}).resume(
            ScriptedModel([RESPONSES[1]]), SqliteStore(path), Clock(),
            execution_id="run-r")
        assert resumed.success and resumed.final_state == "done"
        assert resumed.context == ref.context
        final = SqliteStore(path).read("run-r")
        assert diff_traces(replay_trace(final),
                           replay_trace(ref.events)) == []
        assert checkpoints_of(final), "checkpoints survived the resume"


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
