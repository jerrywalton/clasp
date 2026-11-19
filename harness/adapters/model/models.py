"""Model adapters."""

from __future__ import annotations

import json

from core.errors import (
    MODEL_OUTPUT, MODEL_PERMANENT, MODEL_TRANSIENT, ModelCallError,
)
from usecases.ports import ModelResult


def classify_model_error(exc: Exception) -> ModelCallError:
    """Map an SDK/transport exception to a typed ModelCallError (#8).

    Duck-typed on purpose: a status_code attribute or a well-known
    exception name decides the kind, so adapters classify without
    importing a vendor SDK. Unknown errors classify as transient:
    they get the bounded correction loop and a MODEL_FAILED record
    instead of an untyped crash."""
    if isinstance(exc, ModelCallError):
        return exc
    if isinstance(exc, json.JSONDecodeError):
        return ModelCallError(MODEL_OUTPUT, f"model output: {exc}")
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    if name in ("APITimeoutError", "APIConnectionError",
                "Timeout", "TimeoutError", "ConnectError",
                "ReadTimeout"):
        kind = MODEL_TRANSIENT
    elif status == 429 or (isinstance(status, int) and status >= 500):
        kind = MODEL_TRANSIENT
    elif isinstance(status, int) and 400 <= status < 500:
        kind = MODEL_PERMANENT
    else:
        kind = MODEL_TRANSIENT
    return ModelCallError(kind, f"model call failed ({name}): {exc}")


class ScriptedModel:
    """Deterministic test double: returns queued responses in order.

    Each queued item is either a dict (the envelope) or a callable
    (prompt, schema) -> dict for prompt-dependent scripting.
    """

    def __init__(self, responses: list) -> None:
        self._responses = list(responses)
        self.prompts: list[str] = []

    def generate(self, prompt: str, schema: dict) -> ModelResult:
        self.prompts.append(prompt)
        if not self._responses:
            raise AssertionError("ScriptedModel: no more queued responses")
        item = self._responses.pop(0)
        data = item(prompt, schema) if callable(item) else item
        return ModelResult(data=data, prompt_tokens=10, completion_tokens=20)


class FnModel:
    """Wraps any (prompt, schema) -> dict function as a ModelPort."""

    def __init__(self, fn, prompt_tokens: int = 10,
                 completion_tokens: int = 20) -> None:
        self._fn = fn
        self._pt = prompt_tokens
        self._ct = completion_tokens

    def generate(self, prompt: str, schema: dict) -> ModelResult:
        return ModelResult(data=self._fn(prompt, schema),
                           prompt_tokens=self._pt,
                           completion_tokens=self._ct)


class OpenAIAdapter:
    """Real model via the OpenAI (or compatible) API. Lazy import so the
    core never depends on the SDK. Expects JSON in the reply."""

    def __init__(self, model: str = "gpt-4o-mini", **client_kwargs) -> None:
        self._model = model
        self._client_kwargs = client_kwargs
        self._client = None

    def _client_or_raise(self):
        if self._client is None:
            import openai  # lazy: optional dependency
            self._client = openai.OpenAI(**self._client_kwargs)
        return self._client

    def generate(self, prompt: str, schema: dict) -> ModelResult:
        client = self._client_or_raise()
        try:
            response = client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
        except Exception as exc:  # noqa: BLE001 - classified, then raised typed
            raise classify_model_error(exc) from exc
        choice = response.choices[0].message.content
        usage = response.usage
        try:
            data = json.loads(choice)
        except (json.JSONDecodeError, TypeError) as exc:
            raise ModelCallError(
                MODEL_OUTPUT,
                f"model returned unparseable output: {exc}") from exc
        return ModelResult(
            data=data,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
