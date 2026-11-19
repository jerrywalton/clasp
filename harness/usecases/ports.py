"""Ports: narrow protocols at the boundary of the use-case layer.
Everything volatile (models, tools, storage, time) implements one of these.
Use cases depend only on these abstractions (DIP)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from core.events import DomainEvent


@dataclass(frozen=True)
class ModelResult:
    data: dict                      # parsed envelope
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict
    idempotency_key: str


@dataclass(frozen=True)
class Verdict:
    passed: bool
    reason: str = ""


class ModelPort(Protocol):
    def generate(self, prompt: str, schema: dict) -> ModelResult: ...


class ToolPort(Protocol):
    def invoke(self, call: ToolCall, current_state: str) -> dict: ...


class GatePort(Protocol):
    def check(self, subject: dict) -> Verdict: ...


class EventStorePort(Protocol):
    def append(self, event: DomainEvent) -> None: ...
    def read(self, execution_id: str) -> list[DomainEvent]: ...
    def list_runs(self) -> list[str]: ...

    def flush(self, timeout: float | None = None) -> None:
        """Block until every event appended so far is durably written.

        Synchronous stores write inside append(), so flush() is a no-op
        for them. Asynchronous stores (see adapters.events.async_store)
        enqueue in append() and write from a logger thread; flush()
        waits for the queue to drain and raises HarnessError if any
        queued event failed to persist. The runner calls flush() at
        completion: a returned Outcome guarantees its full event
        sequence is persisted."""
        ...

    def acquire_lease(self, execution_id: str, owner: str,
                      ttl_ms: int) -> bool:
        """Try to become the single writer for one execution.

        Returns True when the caller holds the lease afterwards: it
        was free, already held by this owner, or expired (expiry is
        the dead-writer cleanup, the lock manager's job). Returns
        False when another owner's lease is still live. The runner
        acquires before run()/resume() appends anything and releases
        when it exits; a writer that dies without releasing blocks
        other writers only until its TTL expires."""
        ...

    def release_lease(self, execution_id: str, owner: str) -> None:
        """Release the lease if (and only if) this owner holds it."""
        ...


class ClockPort(Protocol):
    def now_ms(self) -> int: ...
