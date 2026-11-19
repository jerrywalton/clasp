"""Double-entry ledger + atomic-event tests. Plain asserts:
    python3 harness/tests/test_atomic_ledger.py
Covers: balanced refund postings and trial balance, crash between
ledger legs (incomplete, named by reconciliation, never claimed
complete), the P16 regression (resume completes the missing leg once —
no double refund), resume over a balanced refund re-firing nothing,
build-time account validation, unequal legs failing closed, recreate-
from-log in a fresh process, zero model calls for committed states on
resume, the P22 regression (substituted initial context refused), the
P8 regression (data-only divergence diffs), and redaction."""

import json
import os
import subprocess
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..",
                                "examples"))

from adapters.events.stores import InMemoryStore, SqliteStore
from adapters.model.models import FnModel, ScriptedModel
from app.builder import Harness
from core.errors import DefinitionError, HarnessError
from core.events import (
    LEDGER_POSTED, RUN_STARTED, make_event,
)
from core.ledger import (
    CREDIT, DEBIT, Leg, Transaction, fold_ledger, transaction_balanced,
)
from usecases.reconciliation import trial_balance
from usecases.replay import (
    diff_traces, reconstruct_run, replay_trace,
)

import refund_desk


class Clock:
    def __init__(self):
        self.t = 0

    def now_ms(self):
        self.t += 1
        return self.t


MESSAGE = ("Hi, order ORD-1001 arrived broken. "
           "Please refund $42.50 to my card.")


def run_refund(store, message=MESSAGE, execution_id=None, tools=None):
    tools = tools or refund_desk.make_tools()
    built = refund_desk.build((tools[0], tools[1]))
    runner = built.runner(FnModel(refund_desk.model_fn), store,
                          refund_desk.Clock())
    kwargs = {"execution_id": execution_id} if execution_id else {}
    outcome = runner.run({"message": message}, **kwargs)
    return outcome, tools, built


# ------------------------------------------------------------- Phase A core
def test_balanced_refund_posts_both_legs_and_trial_balance():
    store = InMemoryStore()
    outcome, tools, _ = run_refund(store)
    assert outcome.final_state == "done" and outcome.success
    assert outcome.ledger_incomplete == ()
    events = store.read(outcome.execution_id)
    legs = [e for e in events if e.type == LEDGER_POSTED]
    assert len(legs) == 2, legs
    payloads = [e.payload for e in legs]
    assert {p["side"] for p in payloads} == {DEBIT, CREDIT}
    assert payloads[0]["txn_id"] == payloads[1]["txn_id"]
    assert all(p["amount_cents"] == 4250 for p in payloads)
    assert all(p["reference"] == "ORD-1001" for p in payloads)
    by_side = {p["side"]: p for p in payloads}
    assert by_side[DEBIT]["account"] == "refunds_issued"
    assert by_side[CREDIT]["account"] == "cash"

    book = fold_ledger(events)
    assert book.balanced and len(book.transactions) == 1
    assert transaction_balanced(book.transactions[0])
    trial = trial_balance(events)
    assert trial.balanced and trial.completed_balanced
    assert trial.total_debits_cents == trial.total_credits_cents == 4250
    accounts = {a.account: a for a in trial.accounts}
    assert accounts["refunds_issued"].debit_cents == 4250
    assert accounts["cash"].credit_cents == 4250
    assert trial.incomplete_transactions == ()
    # The real-world effect happened exactly once.
    assert tools[2] == [{"order_id": "ORD-1001", "amount_cents": 4250,
                         "status": "issued"}]


def test_refund_desk_episode_outcomes_unchanged():
    expected = [
        ("done", True, "APPROVE"),
        ("done", True, "REVIEW"),
        ("done", True, "REVIEW"),
        ("escalated", False, None),
    ]
    for (title, message), (final, success, decision) in zip(
            refund_desk.EPISODES, expected):
        outcome, _, _ = run_refund(InMemoryStore(), message)
        assert outcome.final_state == final, (title, outcome.final_state)
        assert outcome.success is success, title
        assert outcome.context.get("decision") == decision, title


def test_unequal_legs_fail_closed():
    # A posted Transaction cannot even be constructed unequal.
    try:
        Transaction("txn-x", (Leg("a", DEBIT, 100), Leg("b", CREDIT, 90)))
    except ValueError:
        pass
    else:
        raise AssertionError("unequal Transaction was constructed")
    # And a log containing unequal legs folds as unbalanced, never as
    # a completed transaction.
    events = [
        make_event("r", 0, 0, LEDGER_POSTED,
                   {"txn_id": "txn-x", "account": "a", "side": DEBIT,
                    "amount_cents": 100, "reference": "ORD-1"}),
        make_event("r", 1, 0, LEDGER_POSTED,
                   {"txn_id": "txn-x", "account": "b", "side": CREDIT,
                    "amount_cents": 90, "reference": "ORD-1"}),
    ]
    trial = trial_balance(events)
    assert not trial.balanced
    assert len(trial.incomplete_transactions) == 1
    assert trial.incomplete_transactions[0].status == "unbalanced"
    assert trial.completed_debits_cents == 0
    assert trial.completed_credits_cents == 0


def test_undeclared_account_and_duplicate_chart_rejected_at_build():
    def base_builder():
        b = Harness.define("ledger-build").initial("a")
        b.state("a", handler=lambda ctx: {"event": "go", "data": {},
                                          "tool_calls": []},
                tools=("pay",))
        b.state("done", terminal=True)
        b.state("esc", terminal=True)
        b.state("exh", terminal=True)
        b.transition("a", "go", "done")
        b.escalation_state("esc").exhausted_state("exh")
        return b

    b = base_builder()
    b.register_ledger_tool("pay", lambda args: {}, legal_in={"a"},
                           debit_account="refunds_issued",
                           credit_account="cash",
                           args_schema={"type": "object"})
    try:
        b.build()
    except DefinitionError as exc:
        assert "undeclared account" in str(exc), exc
    else:
        raise AssertionError("posting to undeclared accounts built")

    b = base_builder()
    b.register_ledger_tool("pay", lambda args: {}, legal_in={"a"},
                           debit_account="refunds_issued",
                           credit_account="cash",
                           args_schema={"type": "object"})
    b.accounts("cash", "cash")
    try:
        b.build()
    except DefinitionError as exc:
        assert "duplicate account" in str(exc), exc
    else:
        raise AssertionError("duplicate account declaration built")

    b = base_builder()
    b.accounts("cash", "refunds_issued")
    b.register_ledger_tool("pay", lambda args: {}, legal_in={"a"},
                           debit_account="refunds_issued",
                           credit_account="cash",
                           args_schema={"type": "object"})
    b.build()  # declared chart + rule builds fine


# ------------------------------------------- crash between legs + P16
class FailCreditLegStore:
    """Store wrapper that 'crashes the process' while the credit leg
    of a ledger posting is being appended."""

    def __init__(self, inner):
        self.inner = inner
        self.fail = True

    def append(self, event):
        if (self.fail and event.type == LEDGER_POSTED
                and event.payload.get("side") == CREDIT):
            raise RuntimeError("simulated crash between ledger legs")
        self.inner.append(event)

    def read(self, execution_id):
        return self.inner.read(execution_id)

    def list_runs(self):
        return self.inner.list_runs()

    def flush(self, timeout=None):
        pass


def test_crash_between_legs_incomplete_then_resume_completes_once():
    inner = InMemoryStore()
    store = FailCreditLegStore(inner)
    tools = refund_desk.make_tools()
    built = refund_desk.build((tools[0], tools[1]))
    runner = built.runner(FnModel(refund_desk.model_fn), store,
                          refund_desk.Clock())
    try:
        runner.run({"message": MESSAGE}, execution_id="refund-crash")
    except RuntimeError:
        pass
    else:
        raise AssertionError("the simulated crash did not happen")

    # The effect fired once in the real world; the log holds ONE leg.
    assert len(tools[2]) == 1
    partial = inner.read("refund-crash")
    legs = [e for e in partial if e.type == LEDGER_POSTED]
    assert len(legs) == 1 and legs[0].payload["side"] == DEBIT
    trial = trial_balance(partial)
    assert not trial.balanced
    assert len(trial.incomplete_transactions) == 1
    incomplete = trial.incomplete_transactions[0]
    assert incomplete.status == "incomplete"
    assert incomplete.debit_cents == 4250 and incomplete.credit_cents == 0
    # No outcome was returned claiming the effect complete (the run
    # unwound), and reconciliation names the hole from the log alone.

    # Resume in a 'fresh process': fresh harness/registry, same store,
    # fault cleared. Committed states fold; the in-flight decided
    # state completes the missing leg exactly once and never re-fires.
    store.fail = False
    rebuilt = refund_desk.build((tools[0], tools[1]))
    prompts = []

    def counting_model(prompt, schema):
        prompts.append(prompt)
        return refund_desk.model_fn(prompt, schema)

    resumed = rebuilt.resume(
        FnModel(counting_model), store, refund_desk.Clock(),
        execution_id="refund-crash")
    assert resumed.final_state == "done" and resumed.success
    assert resumed.ledger_incomplete == ()
    # Zero model calls for committed states: only the responded
    # state's prompt was generated; classify was folded from the log.
    assert len(prompts) == 1 and prompts[0].startswith("RESPOND"), prompts
    # No double refund: the external effect still happened once, and
    # the log now holds exactly one balanced transaction.
    assert len(tools[2]) == 1, tools[2]
    final = inner.read("refund-crash")
    legs = [e for e in final if e.type == LEDGER_POSTED]
    assert len(legs) == 2
    trial = trial_balance(final)
    assert trial.balanced
    assert trial.total_debits_cents == trial.total_credits_cents == 4250
    assert len(trial.incomplete_transactions) == 0


class FailTerminalEntryStore:
    """Crashes while the escalation terminal's STATE_ENTERED is being
    appended, leaving a log that ends at ESCALATED."""

    def __init__(self, inner):
        self.inner = inner
        self.fail = True

    def append(self, event):
        if (self.fail and event.type == "state_entered"
                and event.payload.get("state") == "escalated"):
            raise RuntimeError("simulated crash at terminal entry")
        self.inner.append(event)

    def read(self, execution_id):
        return self.inner.read(execution_id)

    def list_runs(self):
        return self.inner.list_runs()

    def flush(self, timeout=None):
        pass


def test_resume_adopts_recorded_escalation_without_redriving():
    inner = InMemoryStore()
    store = FailTerminalEntryStore(inner)
    tools = refund_desk.make_tools()
    built = refund_desk.build((tools[0], tools[1]))
    runner = built.runner(FnModel(refund_desk.model_fn), store,
                          refund_desk.Clock())
    try:
        runner.run({"message": "my package never arrived, I want my "
                               "money back"},
                   execution_id="esc-crash")
    except RuntimeError:
        pass
    else:
        raise AssertionError("the simulated crash did not happen")
    partial = inner.read("esc-crash")
    assert partial[-1].type == "escalated", partial[-1].type

    store.fail = False

    class ExplodingModel:
        def generate(self, prompt, schema):
            raise AssertionError("model called despite recorded outcome")

    rebuilt = refund_desk.build((tools[0], tools[1]))
    resumed = rebuilt.resume(ExplodingModel(), store, refund_desk.Clock(),
                             execution_id="esc-crash")
    # The recorded escalation is adopted: terminal outcome, no success,
    # zero model calls, and the log gains exactly the terminal entry
    # plus COMPLETED.
    assert resumed.final_state == "escalated" and not resumed.success
    final = inner.read("esc-crash")
    assert len(final) == len(partial) + 2
    assert [e.type for e in final[-2:]] == ["state_entered", "completed"]


def test_ledger_over_async_sqlite_store():
    # Ledger events ride the async logging path: the runner's flush at
    # completion guarantees both legs are persisted before it returns.
    with TemporaryDirectory() as td:
        path = os.path.join(td, "events.db")
        store = SqliteStore(path)
        outcome, tools, built = run_refund(store, execution_id="async-1")
        try:
            assert outcome.success
            stored = SqliteStore(path).read("async-1")
            trial = trial_balance(stored)
            assert trial.balanced
            assert trial.total_debits_cents == 4250
            assert len([e for e in stored
                        if e.type == LEDGER_POSTED]) == 2
        finally:
            built.close()


def test_resume_over_balanced_refund_refires_nothing():
    with TemporaryDirectory() as td:
        path = os.path.join(td, "events.db")
        store = SqliteStore(path)
        tools = refund_desk.make_tools()
        built = refund_desk.build((tools[0], tools[1]))
        outcome = built.runner(
            FnModel(refund_desk.model_fn), store, refund_desk.Clock()
        ).run({"message": MESSAGE}, execution_id="refund-done")
        assert outcome.success and len(tools[2]) == 1
        before = SqliteStore(path).read("refund-done")

        class ExplodingModel:
            def generate(self, prompt, schema):
                raise AssertionError("model called on a completed run")

        rebuilt = refund_desk.build((tools[0], tools[1]))
        folded = rebuilt.resume(
            ExplodingModel(), SqliteStore(path), refund_desk.Clock(),
            execution_id="refund-done")
        assert folded.success and folded.final_state == "done"
        assert folded.context == outcome.context
        assert folded.events == before  # nothing appended
        assert len(tools[2]) == 1       # tool never re-fired
        assert SqliteStore(path).read("refund-done") == before


# ------------------------------------------------------------- Phase B
def test_recreate_from_log_in_fresh_process():
    harness_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), ".."))
    examples_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
    with TemporaryDirectory() as td:
        path = os.path.join(td, "events.db")
        run_script = os.path.join(td, "run_child.py")
        with open(run_script, "w", encoding="utf-8") as fh:
            fh.write(
                "import json, sys\n"
                f"sys.path.insert(0, {harness_dir!r})\n"
                f"sys.path.insert(0, {examples_dir!r})\n"
                "from adapters.events.stores import SqliteStore\n"
                "from adapters.model.models import FnModel\n"
                "import refund_desk\n"
                "tools = refund_desk.make_tools()\n"
                "built = refund_desk.build((tools[0], tools[1]))\n"
                "outcome = built.runner(FnModel(refund_desk.model_fn),"
                " SqliteStore(sys.argv[1]), refund_desk.Clock()).run("
                "{'message': sys.argv[2]}, execution_id='ep-1')\n"
                "print(json.dumps({'context': outcome.context,"
                " 'final_state': outcome.final_state,"
                " 'success': outcome.success}))\n")
        result = subprocess.run(
            [sys.executable, run_script, path, MESSAGE],
            capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        original = json.loads(result.stdout.strip())

        fold_script = os.path.join(td, "fold_child.py")
        with open(fold_script, "w", encoding="utf-8") as fh:
            fh.write(
                "import json, sys\n"
                f"sys.path.insert(0, {harness_dir!r})\n"
                "from adapters.events.stores import SqliteStore\n"
                "from usecases.replay import reconstruct_run\n"
                "events = SqliteStore(sys.argv[1]).read('ep-1')\n"
                "plan = reconstruct_run(events)\n"
                "print(json.dumps({'context': plan.context,"
                " 'final_state': plan.final_state,"
                " 'current_state': plan.current_state,"
                " 'completed': plan.completed}))\n")
        result = subprocess.run(
            [sys.executable, fold_script, path],
            capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        folded = json.loads(result.stdout.strip())
        # A fresh process with ONLY the log recreated the run.
        assert folded["completed"] is True
        assert folded["final_state"] == original["final_state"] == "done"
        assert folded["context"] == original["context"]


def envelope(event="go", data=None, tool_calls=None):
    return {"event": event, "data": data or {},
            "tool_calls": tool_calls or []}


def build_fixture(records):
    def record(args):
        return records.setdefault(args["value"],
                                  {"recorded": args["value"]})

    b = Harness.define("fixture").initial("a")
    b.state("a", prompt_template="A $correction",
            output_schema={"type": "object"}, tools=("record",))
    b.state("b", prompt_template="B $correction",
            output_schema={"type": "object"})
    b.state("done", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("a", "a_done", "b")
    b.transition("b", "b_done", "done")
    b.escalation_state("esc").exhausted_state("exh")
    b.register_tool("record", record, legal_in={"a"},
                    args_schema={"type": "object",
                                 "properties": {
                                     "value": {"type": "integer"}},
                                 "required": ["value"]})
    b.budget(step_limit=10, token_limit=1000)
    return b.build()


FIXTURE_RESPONSES = [
    {"event": "a_done", "data": {"x": 1},
     "tool_calls": [{"name": "record", "args": {"value": 1}}]},
    {"event": "b_done", "data": {"y": 2}, "tool_calls": []},
]


class CrashOnCall:
    def __init__(self, responses, crash_call):
        self._inner = ScriptedModel(responses)
        self._crash_call = crash_call
        self.calls = 0

    def generate(self, prompt, schema):
        self.calls += 1
        if self.calls == self._crash_call:
            raise RuntimeError("simulated process crash")
        return self._inner.generate(prompt, schema)


def crashed_fixture_store(td):
    records = {}
    built = build_fixture(records)
    path = os.path.join(td, "events.db")
    runner = built.runner(
        CrashOnCall(FIXTURE_RESPONSES, crash_call=2), SqliteStore(path),
        Clock())
    try:
        runner.run({"request": "r1"}, execution_id="fix-1")
    except RuntimeError:
        pass
    else:
        raise AssertionError("the simulated crash did not happen")
    return path, records


def test_resume_makes_zero_model_calls_for_committed_states():
    with TemporaryDirectory() as td:
        path, records = crashed_fixture_store(td)
        prompts = []
        inner = ScriptedModel(FIXTURE_RESPONSES[1:])

        class CountingModel:
            def generate(self, prompt, schema):
                prompts.append(prompt)
                return inner.generate(prompt, schema)

        resumed = build_fixture(records).resume(
            CountingModel(), SqliteStore(path), Clock(),
            execution_id="fix-1")
        assert resumed.final_state == "done" and resumed.success
        # Exactly one model call: the in-flight state B. Committed
        # state A was recreated by folding, not re-driven.
        assert len(prompts) == 1 and prompts[0].startswith("B"), prompts
        assert resumed.context["x"] == 1 and resumed.context["y"] == 2
        assert records == {1: {"recorded": 1}}


def test_p22_substituted_initial_context_refused():
    with TemporaryDirectory() as td:
        path, records = crashed_fixture_store(td)

        class ExplodingModel:
            def generate(self, prompt, schema):
                raise AssertionError("model called despite refusal")

        try:
            build_fixture(records).resume(
                ExplodingModel(), SqliteStore(path), Clock(),
                execution_id="fix-1",
                initial_context={"request": "tampered"})
        except HarnessError as exc:
            assert "contradicts" in str(exc), exc
        else:
            raise AssertionError("substituted initial context accepted")
        # The recorded context is accepted, and None (log-authoritative)
        # is accepted.
        resumed = build_fixture(records).resume(
            ScriptedModel(FIXTURE_RESPONSES[1:]), SqliteStore(path),
            Clock(), execution_id="fix-1",
            initial_context={"request": "r1"})
        assert resumed.success


def test_p8_data_only_divergence_diffs():
    def run_with_value(value):
        built = build_fixture({})
        model = ScriptedModel([
            {"event": "a_done", "data": {"v": value}, "tool_calls": []},
            {"event": "b_done", "data": {}, "tool_calls": []}])
        return built.runner(model, InMemoryStore(), Clock()).run(
            {"request": "r"})

    o1 = run_with_value(1)
    o2 = run_with_value(999)
    divergences = diff_traces(replay_trace(o1.events),
                              replay_trace(o2.events))
    assert divergences, "data-only divergence diffed clean"
    assert any("model_called" in d or "transition" in d
               for d in divergences), divergences


def test_redacted_keys_are_hash_only_and_opaque_to_recreation():
    b = Harness.define("redacted").initial("a")
    b.state("a", handler=lambda ctx: envelope("go", {"seen": True}))
    b.state("done", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("a", "go", "done")
    b.escalation_state("esc").exhausted_state("exh")
    b.redacted_context_keys("secret")
    built = b.build()
    store = InMemoryStore()
    outcome = built.runner(ScriptedModel([]), store, Clock()).run(
        {"secret": "s3cr3t-value", "plain": 1}, execution_id="red-1")
    assert outcome.success
    blob = json.dumps([e.payload for e in outcome.events], default=str)
    assert "s3cr3t-value" not in blob
    started = next(e for e in outcome.events if e.type == RUN_STARTED)
    assert started.payload["redacted_context_keys"] == ["secret"]
    recorded = started.payload["initial_context"]["secret"]
    assert set(recorded) == {"__redacted__"}
    assert recorded["__redacted__"].startswith("sha256:")
    plan = reconstruct_run(outcome.events)
    # Recreation prices the redaction: the value comes back opaque.
    assert plan.context["secret"] == recorded
    assert plan.context["plain"] == 1
    # A wrong secret on resume is a contradiction and is refused; the
    # correct one is accepted (its hash matches the recorded hash).
    try:
        built.resume(ScriptedModel([]), store, Clock(),
                     execution_id="red-1",
                     initial_context={"secret": "wrong", "plain": 1})
    except HarnessError:
        pass
    else:
        raise AssertionError("wrong redacted value accepted on resume")
    folded = built.resume(ScriptedModel([]), store, Clock(),
                          execution_id="red-1",
                          initial_context={"secret": "s3cr3t-value",
                                           "plain": 1})
    assert folded.success


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
