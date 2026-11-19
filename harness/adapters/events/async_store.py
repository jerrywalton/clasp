"""Async event store: persistence decoupled from run execution.

The owner's directive: "Have the agent fire a log message at every
event, and have the logger running in a separate thread. Decouple
logging from run events."

AsyncEventStore DECORATES any EventStorePort store. append() does no
I/O: it enqueues the event on a thread-safe FIFO queue and returns
immediately, so a slow store can no longer stall the run's control
flow and a raising store can no longer kill it mid-state. A single
dedicated consumer thread (daemon) drains the queue in order and calls
the inner store's append(); a single consumer means per-execution
sequence order is preserved exactly.

The decoupling is bought with a window: between append() returning
and the consumer writing, the event exists only in process memory.
If the process dies inside that window, the queued events are lost —
the same shape as the effect->log window the 2026-10-27 critique
flagged (P16), widened from microseconds to however far the consumer
has fallen behind. The contract that bounds the window:

- flush(timeout=None) blocks until every queued event has been
  written (read() and list_runs() drain the queue first, so they
  observe everything appended so far: read-your-writes).
- The runner flushes at completion, so a returned Outcome still
  guarantees its full event sequence is persisted — durability is
  deferred to a defined point, not abandoned.
- If the inner store raises inside the consumer, the consumer does
  NOT die and the run does NOT crash: the failure is recorded in
  `errors` (and reported to the optional on_error callback) and
  draining continues with the next event. The next flush() raises a
  HarnessError summarizing the unreported failures, exactly once per
  batch; `errors` keeps the full inspectable history.
- At interpreter exit, an atexit hook flushes and closes the store,
  so even a run abandoned without close() is written out on an
  orderly shutdown. The hook is just close(): idempotent, and a
  no-op for an already-closed store. What no in-process mechanism
  can cover is SIGKILL, power loss, or a hard interpreter crash —
  events still queued at that instant are lost.

Lifecycle: close() flushes, then stops the consumer via a sentinel
and joins the thread; it is idempotent. append() after close() raises
HarnessError. Usable as a context manager. All threading lives here,
in the adapters layer; core/ and usecases/ stay thread-free.
"""

from __future__ import annotations

import atexit
import queue
import threading
from dataclasses import dataclass

from core.errors import HarnessError
from core.events import DomainEvent

_STOP = object()  # sentinel: tells the consumer to exit


@dataclass(frozen=True)
class PersistenceFailure:
    """One event the inner store refused, kept for inspection."""
    event: DomainEvent
    error: Exception


class AsyncEventStore:
    """EventStorePort decorator: fire-and-forget appends, ordered
    writes on a dedicated logger thread."""

    def __init__(self, inner, on_error=None) -> None:
        self._inner = inner
        self._on_error = on_error
        self._queue: queue.Queue = queue.Queue()
        # One condition guards _closed and _pending (events enqueued
        # but not yet written). append() counts an event before it
        # becomes visible to the consumer, so flush()/close() can
        # never slip past an append that is in flight.
        self._cond = threading.Condition()
        self._closed = False
        self._pending = 0
        self.errors: list[PersistenceFailure] = []
        self._reported = 0  # errors[:_reported] already raised by flush()
        self.thread = threading.Thread(
            target=self._drain, name="harness-event-logger", daemon=True)
        self.thread.start()
        # Orderly interpreter exit flushes and closes (the daemon
        # thread alone would simply vanish). close() is idempotent,
        # so this is a silent no-op if the store was already closed.
        atexit.register(self.close)

    # ------------------------------------------------------------ identity
    @property
    def inner(self):
        """The decorated store (writes ultimately land here)."""
        return self._inner

    @property
    def closed(self) -> bool:
        return self._closed

    # -------------------------------------------------------------- append
    def append(self, event: DomainEvent) -> None:
        """Enqueue and return. No I/O happens on the caller's thread."""
        with self._cond:
            if self._closed:
                raise HarnessError(
                    "cannot append: this async event store is closed")
            self._pending += 1
        self._queue.put(event)

    # -------------------------------------------------------------- reads
    def read(self, execution_id: str) -> list[DomainEvent]:
        self._wait_drained(None)
        return self._inner.read(execution_id)

    def list_runs(self) -> list[str]:
        self._wait_drained(None)
        return self._inner.list_runs()

    # ------------------------------------------------------------- leases
    @property
    def last_discarded_tail(self):
        return getattr(self._inner, "last_discarded_tail", None)

    def acquire_lease(self, execution_id: str, owner: str,
                      ttl_ms: int) -> bool:
        # The decorator stays transparent for any EventStorePort: an
        # inner store with no lease support simply has no leases to
        # take (the lease is unenforced, exactly as if the runner had
        # been handed the bare inner store through a legacy adapter).
        delegate = getattr(self._inner, "acquire_lease", None)
        if delegate is None:
            return True
        return delegate(execution_id, owner, ttl_ms)

    def release_lease(self, execution_id: str, owner: str) -> None:
        delegate = getattr(self._inner, "release_lease", None)
        if delegate is not None:
            delegate(execution_id, owner)

    # --------------------------------------------------------------- flush
    def flush(self, timeout: float | None = None) -> None:
        """Block until the queue is drained, then surface failures.

        Raises HarnessError if the drain does not finish within
        `timeout` (when given), or if any events failed to persist
        since the last flush() that reported failures. Each failure
        batch is raised once; the full history stays in `errors`."""
        self._wait_drained(timeout)
        with self._cond:
            fresh = self.errors[self._reported:]
            self._reported = len(self.errors)
        if fresh:
            first = fresh[0]
            raise HarnessError(
                f"{len(fresh)} event(s) failed to persist "
                f"(first: execution {first.event.execution_id!r} "
                f"seq {first.event.seq} {first.event.type}: "
                f"{type(first.error).__name__}: {first.error})")

    def _wait_drained(self, timeout: float | None) -> None:
        with self._cond:
            drained = self._cond.wait_for(lambda: self._pending == 0,
                                          timeout)
        if not drained:
            raise HarnessError(
                f"flush timed out after {timeout}s with "
                f"{self._pending} event(s) still unwritten")

    # -------------------------------------------------------------- close
    def close(self) -> None:
        """Flush, stop the logger thread, and join it. Idempotent.

        If the final flush reports persistence failures, the thread is
        still stopped and joined before the HarnessError propagates."""
        with self._cond:
            if self._closed:
                return
            self._closed = True
        try:
            self.flush()
        finally:
            self._queue.put(_STOP)
            self.thread.join()

    def __enter__(self) -> "AsyncEventStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ------------------------------------------------------------ consumer
    def _drain(self) -> None:
        while True:
            item = self._queue.get()
            if item is _STOP:
                return
            try:
                self._inner.append(item)
            except Exception as exc:  # noqa: BLE001 - isolation is the point
                failure = PersistenceFailure(item, exc)
                self.errors.append(failure)
                if self._on_error is not None:
                    try:
                        self._on_error(failure)
                    except Exception:  # noqa: BLE001 - a bad callback
                        pass          # must not kill the consumer either
            finally:
                with self._cond:
                    self._pending -= 1
                    self._cond.notify_all()
