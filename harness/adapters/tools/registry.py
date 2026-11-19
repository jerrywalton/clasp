"""ToolRegistry: capability gating, enforced in the registry.

Enforcement: invoke() refuses out-of-state invocation even if asked.
(The runner rejects illegal tools in the envelope before this, and
the builder derives this registry's legality from the states' own
tool declarations — one source of truth. Presentation — rendering
the legal set into a prompt — is the author's choice; tools_for()
exists for that and the runner does not call it.)

Handlers must be idempotent: results are cached by idempotency key, so the
correction loop can safely re-invoke the same logical call."""

from __future__ import annotations

from core.errors import HarnessError
from usecases.ports import ToolCall


class ToolRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, callable] = {}
        self._legal_in: dict[str, set[str]] = {}
        self._cache: dict[str, dict] = {}
        self._call_counts: dict[str, int] = {}

    def register(self, name: str, handler: callable,
                 legal_in: set[str] | tuple[str, ...]) -> None:
        self._handlers[name] = handler
        self._legal_in[name] = set(legal_in)

    def tools_for(self, state: str) -> tuple[str, ...]:
        return tuple(sorted(n for n, states in self._legal_in.items()
                            if state in states))

    def invoke(self, call: ToolCall, current_state: str) -> dict:
        if call.name not in self._handlers:
            raise HarnessError(f"unknown tool '{call.name}'")
        if current_state not in self._legal_in[call.name]:
            raise HarnessError(
                f"tool '{call.name}' is not legal in state '{current_state}'")
        if call.idempotency_key in self._cache:
            return self._cache[call.idempotency_key]
        self._call_counts[call.name] = self._call_counts.get(call.name, 0) + 1
        result = self._handlers[call.name](dict(call.args))
        if not isinstance(result, dict):
            result = {"result": result}
        self._cache[call.idempotency_key] = result
        return result

    def call_count(self, name: str) -> int:
        """Distinct (non-cached) invocations. Test seam for idempotency."""
        return self._call_counts.get(name, 0)
