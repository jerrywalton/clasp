"""Async logging tests: persistence decoupled from run execution.
Plain asserts, no framework:
    python3 harness/tests/test_async_logging.py
Covers: ordered + complete persistence without a caller flush, the
builder's default wrapping of persistent stores, non-blocking appends
over a slow store, failure isolation (a raising store neither crashes
the run mid-flight nor kills the logger; flush surfaces it), logger
lifecycle, flush timeouts, crash coverage (model / handler / tool
exceptions leave every fired event persisted; resume over an async
store), and the atexit flush on interpreter exit."""

import os
import subprocess
import sys
import time
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adapters.events.async_store import AsyncEventStore
from adapters.events.stores import InMemoryStore, SqliteStore
from adapters.model.models import ScriptedModel
from app.builder import Harness
from core.errors import HarnessError
from core.events import (
    GATE_VERDICT, MODEL_CALLED, RUN_STARTED, STATE_ENTERED,
    STEP_COMMITTED, TOOL_INTENDED, TRANSITION, make_event,
)
from usecases.replay import diff_traces, reconstruct_run, replay_trace


class Clock:
    def __init__(self):
        self.t = 0

    def now_ms(self):
        self.t += 1
        return self.t


def envelope(event="go", data=None, tool_calls=None):
    return {"event": event, "data": data or {},
            "tool_calls": tool_calls or []}


def responses():
    return [
        {"event": "a_done", "data": {"x": 1},
         "tool_calls": [{"name": "record", "args": {"value": 1}}]},
        {"event": "b_done", "data": {"y": 2}, "tool_calls": []},
    ]


def build_two_state(records, async_logging=True):
    """a (model + tool) -> b (model) -> c (handler) -> done."""
    def record(args):
        return records.setdefault(args["value"],
                                  {"recorded": args["value"]})

    b = Harness.define("async-test").initial("a")
    if not async_logging:
        b.async_logging(False)
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


class CrashOnCall:
    """Model double: delegates to a ScriptedModel, then 'kills the
    process' with a RuntimeError on the Nth call."""

    def __init__(self, resps, crash_call):
        self._inner = ScriptedModel(resps)
        self._crash_call = crash_call
        self.calls = 0

    def generate(self, prompt, schema):
        self.calls += 1
        if self.calls == self._crash_call:
            raise RuntimeError("simulated process crash")
        return self._inner.generate(prompt, schema)


class SlowStore:
    """EventStorePort whose append sleeps: persistence as pure latency."""

    def __init__(self, delay):
        self.delay = delay
        self.events = []

    def append(self, event):
        time.sleep(self.delay)
        self.events.append(event)

    def read(self, execution_id):
        return [e for e in self.events if e.execution_id == execution_id]

    def list_runs(self):
        return list(dict.fromkeys(e.execution_id for e in self.events))

    def flush(self, timeout=None):
        pass


class FlakyStore:
    """EventStorePort delegating to memory, raising on the Nth append."""

    def __init__(self, fail_on_call):
        self._inner = InMemoryStore()
        self._fail_on_call = fail_on_call
        self.calls = 0

    def append(self, event):
        self.calls += 1
        if self.calls == self._fail_on_call:
            raise RuntimeError("disk on fire")
        self._inner.append(event)

    def read(self, execution_id):
        return self._inner.read(execution_id)

    def list_runs(self):
        return self._inner.list_runs()

    def flush(self, timeout=None):
        pass


# ------------------------------------------------------- ordering/completeness
def test_async_run_persists_full_ordered_log():
    built = build_two_state({})
    with TemporaryDirectory() as td:
        path = os.path.join(td, "events.db")
        store = AsyncEventStore(SqliteStore(path))
        try:
            runner = built.runner(ScriptedModel(responses()), store,
                                  Clock())
            outcome = runner.run({"request": "r1"}, execution_id="run-1")
            assert outcome.success and outcome.final_state == "done"
            # No caller flush: the runner flushed at completion, so a
            # fresh store instance already sees the whole log, in order.
            stored = SqliteStore(path).read("run-1")
            assert stored == outcome.events
            assert [e.seq for e in stored] == list(range(len(stored)))
            assert store.read("run-1") == outcome.events
            assert store.list_runs() == ["run-1"]
        finally:
            store.close()


def test_builder_wraps_persistent_stores_by_default():
    built = build_two_state({})
    with TemporaryDirectory() as td:
        path = os.path.join(td, "events.db")
        raw = SqliteStore(path)
        runner = built.runner(ScriptedModel(responses()), raw, Clock())
        assert isinstance(runner.events, AsyncEventStore)
        assert runner.events.inner is raw
        outcome = runner.run({"request": "r1"}, execution_id="run-1")
        assert SqliteStore(path).read("run-1") == outcome.events
        # A second runner over the same store reuses the one wrapper:
        # one logger thread per store, so ordering is never split.
        runner2 = built.runner(ScriptedModel(responses()), raw, Clock())
        assert runner2.events is runner.events
        built.close()
        assert runner.events.closed
        assert not runner.events.thread.is_alive()
        built.close()  # idempotent

    # Opt-out: the raw synchronous path.
    built_sync = build_two_state({}, async_logging=False)
    with TemporaryDirectory() as td:
        raw = SqliteStore(os.path.join(td, "events.db"))
        runner = built_sync.runner(ScriptedModel(responses()), raw,
                                   Clock())
        assert runner.events is raw

    # In-memory stores are never wrapped.
    runner = built.runner(ScriptedModel(responses()), InMemoryStore(),
                          Clock())
    assert type(runner.events) is InMemoryStore


# ------------------------------------------------------------ non-blocking
def test_append_is_non_blocking():
    inner = SlowStore(delay=0.05)
    store = AsyncEventStore(inner)
    try:
        events = [make_event("r", i, i, STATE_ENTERED, {"state": "a"})
                  for i in range(20)]
        start = time.monotonic()
        for event in events:
            store.append(event)
        append_elapsed = time.monotonic() - start
        # Synchronous appends would take 20 * 0.05 = 1.0s here.
        assert append_elapsed < 0.5, append_elapsed
        start = time.monotonic()
        store.flush()
        flush_elapsed = time.monotonic() - start
        # The work didn't vanish; the flush pays it, all at once.
        assert flush_elapsed >= 0.8, flush_elapsed
        assert inner.events == events  # exact order preserved
    finally:
        store.close()


def test_run_execution_not_serialized_on_slow_store():
    stamps = {}

    def build():
        b = Harness.define("slow-test").initial("a")
        b.state("a", prompt_template="A $correction",
                output_schema={"type": "object"})
        b.state("b", handler=lambda ctx: (
            stamps.__setitem__("b", time.monotonic()),
            envelope("go"))[1])
        b.state("done", terminal=True)
        b.state("esc", terminal=True)
        b.state("exh", terminal=True)
        b.transition("a", "go", "b")
        b.transition("b", "go", "done")
        b.escalation_state("esc").exhausted_state("exh")
        return b.build()

    model_resp = [{"event": "go", "data": {}, "tool_calls": []}]

    # Synchronous: four events fire (and each blocks 60ms) before b's
    # handler runs.
    stamps.clear()
    start = time.monotonic()
    sync_store = SlowStore(delay=0.06)
    build().runner(ScriptedModel(model_resp), sync_store,
                   Clock()).run({})
    sync_stamp = stamps["b"] - start
    assert sync_stamp > 0.18, sync_stamp

    # Async: b's handler runs immediately; the store drains behind it.
    stamps.clear()
    async_store = AsyncEventStore(SlowStore(delay=0.06))
    try:
        start = time.monotonic()
        outcome = build().runner(
            ScriptedModel(model_resp), async_store, Clock()).run({})
        async_stamp = stamps["b"] - start
        assert async_stamp < 0.12, async_stamp
        assert async_stamp < sync_stamp / 2
        # ...and by the time run() returns, the flush has landed all
        # of the run's events in the inner store anyway.
        assert async_store.inner.events == outcome.events
    finally:
        async_store.close()


# --------------------------------------------------------- failure isolation
def test_failure_isolation_and_flush_surfaces():
    built = build_two_state({})
    flaky = FlakyStore(fail_on_call=3)  # 3rd append == seq 2
    seen = []
    store = AsyncEventStore(flaky, on_error=seen.append)
    try:
        runner = built.runner(ScriptedModel(responses()), store, Clock())
        try:
            runner.run({"request": "r1"}, execution_id="run-1")
        except HarnessError as exc:
            # The failure surfaces at the completion flush, as a
            # HarnessError about persistence — not as the store's raw
            # RuntimeError crashing the run mid-flight.
            assert "failed to persist" in str(exc), exc
            assert "1 event(s)" in str(exc), exc
        else:
            raise AssertionError("flush did not surface the failure")
        # Inspectable: exactly the seq-2 event failed...
        assert len(store.errors) == 1
        assert store.errors[0].event.seq == 2
        assert isinstance(store.errors[0].error, RuntimeError)
        assert seen == store.errors  # on_error callback fired too
        # ...and the consumer kept draining behind it: every later
        # event was still written — the stored seqs are exactly the
        # fired range with only seq 2 missing.
        written = [e.seq for e in flaky.read("run-1")]
        assert written == [s for s in range(len(written) + 1)
                           if s != 2], written
        # The failure was reported once; a later flush is clean.
        store.flush()
    finally:
        store.close()


# ---------------------------------------------------------------- lifecycle
def test_logger_lifecycle():
    inner = InMemoryStore()
    store = AsyncEventStore(inner)
    assert store.thread.daemon and store.thread.is_alive()
    event = make_event("r", 0, 0, STATE_ENTERED, {"state": "a"})
    store.append(event)
    store.flush()
    assert inner.read("r") == [event]
    store.close()
    assert store.closed and not store.thread.is_alive()
    store.close()   # double close is harmless
    store.flush()   # flush after close: drained, nothing to report
    try:
        store.append(event)
    except HarnessError:
        pass
    else:
        raise AssertionError("append after close was accepted")

    with AsyncEventStore(InMemoryStore()) as managed:
        managed.append(event)
        managed_thread = managed.thread
    # Context-manager exit flushed and stopped the thread.
    assert managed.inner.read("r") == [event]
    assert not managed_thread.is_alive()


def test_flush_timeout():
    store = AsyncEventStore(SlowStore(delay=0.1))
    try:
        for i in range(5):
            store.append(make_event("r", i, i, STATE_ENTERED,
                                    {"state": "a"}))
        try:
            store.flush(timeout=0.05)
        except HarnessError as exc:
            assert "timed out" in str(exc), exc
        else:
            raise AssertionError("flush timeout was not enforced")
        store.flush()  # no timeout: drains fully
        assert len(store.inner.events) == 5
    finally:
        store.close()


# ------------------------------------------------------------ crash coverage
def test_model_crash_leaves_every_fired_event_persisted():
    ref = build_two_state({}).runner(
        ScriptedModel(responses()), InMemoryStore(), Clock()
    ).run({"request": "r1"}, execution_id="run-42")
    built = build_two_state({})
    with TemporaryDirectory() as td:
        path = os.path.join(td, "events.db")
        runner = built.runner(
            CrashOnCall(responses(), crash_call=2), SqliteStore(path),
            Clock())
        try:
            runner.run({"request": "r1"}, execution_id="run-42")
        except RuntimeError:
            pass
        else:
            raise AssertionError("the simulated crash did not happen")
        # The run's try/finally flushed before the exception reached
        # us: a fresh store instance reads the complete fired prefix.
        stored = SqliteStore(path).read("run-42")
        assert [e.seq for e in stored] == list(range(len(stored)))
        assert diff_traces(replay_trace(stored),
                           replay_trace(ref.events[:len(stored)])) == []
        plan = reconstruct_run(stored)
        assert plan.current_state == "b" and plan.completed is False
        built.close()


def test_handler_crash_leaves_every_fired_event_persisted():
    def boom(ctx):
        raise ValueError("handler blew up")

    b = Harness.define("handler-crash").initial("a")
    b.state("a", prompt_template="A $correction",
            output_schema={"type": "object"})
    b.state("b", handler=boom)
    b.state("done", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("a", "go", "b")
    b.transition("b", "go", "done")
    b.escalation_state("esc").exhausted_state("exh")
    built = b.build()
    resp = [{"event": "go", "data": {}, "tool_calls": []}]
    with TemporaryDirectory() as td:
        path = os.path.join(td, "events.db")
        runner = built.runner(ScriptedModel(resp), SqliteStore(path),
                              Clock())
        try:
            runner.run({}, execution_id="run-h")
        except ValueError:
            pass
        else:
            raise AssertionError("the handler crash did not happen")
        stored = SqliteStore(path).read("run-h")
        assert [e.type for e in stored] == [
            RUN_STARTED, STATE_ENTERED, MODEL_CALLED, GATE_VERDICT,
            TRANSITION, STEP_COMMITTED, STATE_ENTERED]
        assert [e.seq for e in stored] == list(range(7))
        built.close()


def test_tool_crash_leaves_every_fired_event_persisted():
    def explode(args):
        raise RuntimeError("tool exploded")

    b = Harness.define("tool-crash").initial("a")
    b.state("a", prompt_template="A $correction",
            output_schema={"type": "object"}, tools=("explode",))
    b.state("done", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("a", "go", "done")
    b.escalation_state("esc").exhausted_state("exh")
    b.register_tool("explode", explode, legal_in={"a"},
                    args_schema={"type": "object"})
    built = b.build()
    resp = [{"event": "go", "data": {},
             "tool_calls": [{"name": "explode", "args": {}}]}]
    with TemporaryDirectory() as td:
        path = os.path.join(td, "events.db")
        runner = built.runner(ScriptedModel(resp), SqliteStore(path),
                              Clock())
        try:
            runner.run({}, execution_id="run-t")
        except RuntimeError:
            pass
        else:
            raise AssertionError("the tool crash did not happen")
        stored = SqliteStore(path).read("run-t")
        # The tool never returned, so no TOOL_INVOKED was fired; the
        # events that did fire are all persisted, in order — including
        # the TOOL_INTENDED recorded before the call (#2): intent
        # without invocation is exactly what a crash should leave.
        assert [e.type for e in stored] == [
            RUN_STARTED, STATE_ENTERED, MODEL_CALLED, TOOL_INTENDED]
        assert [e.seq for e in stored] == [0, 1, 2, 3]
        built.close()


def test_resume_over_async_store():
    records = {}
    ref = build_two_state({}).runner(
        ScriptedModel(responses()), InMemoryStore(), Clock()
    ).run({"request": "r1"}, execution_id="run-42")
    built = build_two_state(records)
    with TemporaryDirectory() as td:
        path = os.path.join(td, "events.db")
        store = AsyncEventStore(SqliteStore(path))
        crashing = built.runner(
            CrashOnCall(responses(), crash_call=2), store, Clock())
        try:
            crashing.run({"request": "r1"}, execution_id="run-42")
        except RuntimeError:
            pass
        else:
            raise AssertionError("the simulated crash did not happen")
        store.close()  # the 'process' ends; the log is fully written

        store2 = AsyncEventStore(SqliteStore(path))
        try:
            # Phase B: state A is folded from the log, so the resume
            # model starts at the in-flight state B response.
            resumed = built.resume(
                ScriptedModel(responses()[1:]), store2, Clock(),
                execution_id="run-42", initial_context={"request": "r1"})
            assert resumed.final_state == ref.final_state
            assert resumed.success == ref.success
            assert resumed.context == ref.context
            final = store2.read("run-42")
            assert diff_traces(replay_trace(final),
                               replay_trace(ref.events)) == []
            assert [e.seq for e in final] == list(range(len(final)))
            assert records == {1: {"recorded": 1}}, records
        finally:
            store2.close()


# ------------------------------------------------------------------- atexit
def test_atexit_flushes_on_interpreter_exit():
    harness_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), ".."))
    with TemporaryDirectory() as td:
        path = os.path.join(td, "log.jsonl")
        script = os.path.join(td, "child.py")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write(
                "import sys\n"
                "sys.path.insert(0, sys.argv[1])\n"
                "from adapters.events.async_store import AsyncEventStore\n"
                "from adapters.events.stores import JsonlStore\n"
                "from core.events import STATE_ENTERED, make_event\n"
                "store = AsyncEventStore(JsonlStore(sys.argv[2]))\n"
                "for i in range(3):\n"
                "    store.append(make_event('r', i, i, STATE_ENTERED,"
                " {'state': 'a'}))\n"
                "# No flush, no close: exit must flush via atexit.\n")
        result = subprocess.run(
            [sys.executable, script, harness_dir, path],
            capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        with open(path, encoding="utf-8") as fh:
            lines = [line for line in fh if line.strip()]
        assert len(lines) == 3, lines


def test_close_is_idempotent_and_atexit_safe():
    # close() is the atexit hook: calling it any number of extra times
    # (explicitly, then again at interpreter exit) must be a no-op.
    store = AsyncEventStore(InMemoryStore())
    store.append(make_event("r", 0, 0, STATE_ENTERED, {"state": "a"}))
    store.close()
    store.close()
    assert store.closed and not store.thread.is_alive()
    assert store.inner.read("r") != []


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
