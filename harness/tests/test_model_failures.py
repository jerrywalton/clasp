"""Typed model failures (#8) and the correction-model seam (#12).
Plain asserts:
    python3 harness/tests/test_model_failures.py
A model call fails in one of three typed ways. Transient and output
failures are logged as MODEL_FAILED and enter the correction loop;
permanent failures escalate immediately, without burning correction
attempts. When a correction model is declared, retries go to it (and
the log says whose output each attempt was); it is bound by the
fingerprint like the primary model."""

import os
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adapters.events.stores import InMemoryStore, SqliteStore
from adapters.model.models import (
    FnModel, OpenAIAdapter, ScriptedModel, classify_model_error,
)
from app.builder import Harness
from core.errors import (
    MODEL_OUTPUT, MODEL_PERMANENT, MODEL_TRANSIENT, HarnessError,
    ModelCallError,
)
from core.events import ESCALATED, MODEL_CALLED, MODEL_FAILED


class Clock:
    def __init__(self):
        self.t = 0

    def now_ms(self):
        self.t += 1
        return self.t


GOOD = {"event": "go", "data": {"x": 1}, "tool_calls": []}
BAD_EVENT = {"event": "bogus", "data": {}, "tool_calls": []}


def build_machine():
    b = Harness.define("model-failure-test").initial("a")
    b.state("a", prompt_template="A $correction",
            output_schema={"type": "object"}, max_corrections=2)
    b.state("done", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("a", "go", "done")
    b.escalation_state("esc").exhausted_state("exh")
    b.budget(step_limit=10, token_limit=100000)
    return b.build()


class FlakyModel:
    """Fails with a typed error for the first N calls, then answers."""

    def __init__(self, kind, failures, response=GOOD):
        self.kind = kind
        self.failures = failures
        self.response = response
        self.calls = 0

    def generate(self, prompt, schema):
        from usecases.ports import ModelResult
        self.calls += 1
        if self.calls <= self.failures:
            raise ModelCallError(self.kind, f"{self.kind} boom")
        return ModelResult(data=self.response, prompt_tokens=10,
                           completion_tokens=20)


def test_transient_failure_is_logged_and_corrected():
    model = FlakyModel(MODEL_TRANSIENT, failures=1)
    store = InMemoryStore()
    outcome = build_machine().runner(model, store, Clock()).run(
        {}, execution_id="run-t")
    assert outcome.success and model.calls == 2
    failed = [e for e in store.read("run-t") if e.type == MODEL_FAILED]
    assert len(failed) == 1
    assert failed[0].payload["kind"] == MODEL_TRANSIENT
    assert failed[0].payload["role"] == "primary"


def test_output_failure_enters_correction_loop():
    model = FlakyModel(MODEL_OUTPUT, failures=1)
    store = InMemoryStore()
    outcome = build_machine().runner(model, store, Clock()).run(
        {}, execution_id="run-o")
    assert outcome.success
    failed = [e for e in store.read("run-o") if e.type == MODEL_FAILED]
    assert failed[0].payload["kind"] == MODEL_OUTPUT


def test_permanent_failure_escalates_without_retrying():
    model = FlakyModel(MODEL_PERMANENT, failures=99)
    store = InMemoryStore()
    outcome = build_machine().runner(model, store, Clock()).run(
        {}, execution_id="run-p")
    assert outcome.success is False and outcome.final_state == "esc"
    assert model.calls == 1  # no correction attempts were burned
    esc = [e for e in store.read("run-p") if e.type == ESCALATED]
    assert esc[-1].payload["failures"][0][0] == "model_permanent"


def test_transient_failures_exhaust_corrections_then_escalate():
    model = FlakyModel(MODEL_TRANSIENT, failures=99)
    store = InMemoryStore()
    outcome = build_machine().runner(model, store, Clock()).run(
        {}, execution_id="run-x")
    assert outcome.success is False and outcome.final_state == "esc"
    # 1 initial attempt + 2 corrections, each one logged.
    assert model.calls == 3
    assert len([e for e in store.read("run-x")
                if e.type == MODEL_FAILED]) == 3


def test_correction_model_takes_the_retries():
    primary = ScriptedModel([BAD_EVENT])
    correction = ScriptedModel([GOOD])
    store = InMemoryStore()
    built = build_machine()
    outcome = built.runner(primary, store, Clock(),
                           correction_model=correction).run(
        {}, execution_id="run-c")
    assert outcome.success
    roles = [e.payload["role"] for e in store.read("run-c")
             if e.type == MODEL_CALLED]
    assert roles == ["primary", "correction"], roles
    # The correction model saw the failure reason in its prompt.
    assert "not a declared outcome" in correction.prompts[0]


def test_no_correction_model_means_primary_corrects_itself():
    primary = ScriptedModel([BAD_EVENT, GOOD])
    store = InMemoryStore()
    outcome = build_machine().runner(primary, store, Clock()).run(
        {}, execution_id="run-s")
    assert outcome.success
    roles = [e.payload["role"] for e in store.read("run-s")
             if e.type == MODEL_CALLED]
    assert roles == ["primary", "primary"], roles


class NamedModel:
    """A model adapter with a bindable identity."""

    def __init__(self, name, response=GOOD):
        self.model_name = name
        self.response = response

    def generate(self, prompt, schema):
        from usecases.ports import ModelResult
        return ModelResult(data=self.response, prompt_tokens=10,
                           completion_tokens=20)


def test_correction_model_is_fingerprinted():
    with TemporaryDirectory() as td:
        path = os.path.join(td, "e.db")
        built = build_machine()
        built.runner(NamedModel("primary-1"), SqliteStore(path),
                     Clock(),
                     correction_model=NamedModel("corr-1")).run(
            {}, execution_id="run-f")
        try:
            built.resume(NamedModel("primary-1"), SqliteStore(path),
                         Clock(), execution_id="run-f",
                         correction_model=NamedModel("corr-2"))
        except HarnessError as exc:
            assert "correction model changed" in str(exc)
        else:
            raise AssertionError(
                "resume under a different correction model accepted")
        # Same correction model: the completed run folds back fine.
        outcome = built.resume(NamedModel("primary-1"),
                               SqliteStore(path), Clock(),
                               execution_id="run-f",
                               correction_model=NamedModel("corr-1"))
        assert outcome.success


class _StatusError(Exception):
    def __init__(self, status):
        super().__init__(f"http {status}")
        self.status_code = status


def test_error_classification():
    assert classify_model_error(_StatusError(429)).kind == MODEL_TRANSIENT
    assert classify_model_error(_StatusError(503)).kind == MODEL_TRANSIENT
    assert classify_model_error(_StatusError(401)).kind == MODEL_PERMANENT
    assert classify_model_error(_StatusError(400)).kind == MODEL_PERMANENT
    assert classify_model_error(TimeoutError("slow")).kind \
        == MODEL_TRANSIENT


class _FakeCompletions:
    def __init__(self, content):
        self._content = content

    def create(self, **kwargs):
        class Choice:
            pass

        class Message:
            pass

        message = Message()
        message.content = self._content
        choice = Choice()
        choice.message = message

        class Response:
            pass

        response = Response()
        response.choices = [choice]
        response.usage = None
        return response


class _FakeClient:
    def __init__(self, content):
        class Chat:
            pass

        chat = Chat()
        chat.completions = _FakeCompletions(content)
        self.chat = chat


def test_openai_adapter_classifies_unparseable_output():
    adapter = OpenAIAdapter(model="stub")
    adapter._client = _FakeClient("this is not json")
    try:
        adapter.generate("prompt", {})
    except ModelCallError as exc:
        assert exc.kind == MODEL_OUTPUT
    else:
        raise AssertionError("unparseable output was accepted")


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
