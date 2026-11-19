"""Event store adapters: in-memory (tests), append-only JSONL, and SQLite.

All three implement EventStorePort. The SQLite store is the durable
default for runs that must survive a process restart: events are keyed
by (execution_id, seq), so appends can never overwrite history, and a
run's events always replay in sequence order.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import asdict

from core.events import DomainEvent


class _LeaseBook:
    """Single-writer leases keyed by execution_id, shared shape for
    the file/memory stores: owner -> wall-clock expiry. An expired
    lease is free for the taking (the dead writer's lock is cleaned
    up by the next contender, never by a background thread)."""

    def __init__(self) -> None:
        self._leases: dict[str, tuple[str, float]] = {}

    def acquire(self, execution_id: str, owner: str,
                ttl_ms: int) -> bool:
        now = time.time()
        held = self._leases.get(execution_id)
        if held is not None and held[0] != owner and held[1] > now:
            return False
        self._leases[execution_id] = (owner, now + ttl_ms / 1000.0)
        return True

    def release(self, execution_id: str, owner: str) -> None:
        held = self._leases.get(execution_id)
        if held is not None and held[0] == owner:
            del self._leases[execution_id]


class InMemoryStore:
    def __init__(self) -> None:
        self._events: list[DomainEvent] = []
        self._leases = _LeaseBook()
        self.last_discarded_tail: str | None = None

    def append(self, event: DomainEvent) -> None:
        self._events.append(event)

    def flush(self, timeout: float | None = None) -> None:
        """No-op: appends are already visible; nothing is buffered."""

    def acquire_lease(self, execution_id: str, owner: str,
                      ttl_ms: int) -> bool:
        return self._leases.acquire(execution_id, owner, ttl_ms)

    def release_lease(self, execution_id: str, owner: str) -> None:
        self._leases.release(execution_id, owner)

    def read(self, execution_id: str) -> list[DomainEvent]:
        return [e for e in self._events if e.execution_id == execution_id]

    def list_runs(self) -> list[str]:
        return list(dict.fromkeys(e.execution_id for e in self._events))


class JsonlStore:
    """Append-only JSONL. One file; each line one event. Appends are
    synchronous writes.

    Leases live in a sidecar file (``<path>.leases.json``) rewritten
    atomically (temp file + rename) on each change.

    Torn tail: a crash or power loss mid-append can leave a partial
    final line. That fragment is an uncommitted transaction: read()
    discards it (and reports it on ``last_discarded_tail``) instead of
    failing the whole log. Corruption anywhere before the final line
    is real corruption and raises ValueError."""

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.last_discarded_tail: str | None = None

    def append(self, event: DomainEvent) -> None:
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(event), default=str) + "\n")

    def flush(self, timeout: float | None = None) -> None:
        """No-op: each append is written and closed synchronously."""

    # ------------------------------------------------------------- leases
    def _leases_path(self) -> str:
        return self.path + ".leases.json"

    def _read_leases(self) -> _LeaseBook:
        book = _LeaseBook()
        try:
            with open(self._leases_path(), encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            return book
        now = time.time()
        for execution_id, (owner, expires) in raw.items():
            if expires > now:
                book._leases[execution_id] = (owner, expires)
        return book

    def _write_leases(self, book: _LeaseBook) -> None:
        tmp = self._leases_path() + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump({k: list(v) for k, v in book._leases.items()}, fh)
        os.replace(tmp, self._leases_path())

    def acquire_lease(self, execution_id: str, owner: str,
                      ttl_ms: int) -> bool:
        book = self._read_leases()
        if not book.acquire(execution_id, owner, ttl_ms):
            return False
        self._write_leases(book)
        return True

    def release_lease(self, execution_id: str, owner: str) -> None:
        book = self._read_leases()
        book.release(execution_id, owner)
        self._write_leases(book)

    # -------------------------------------------------------------- reads
    def _iter_raw(self) -> list[dict]:
        """Parse every line. A torn final line is discarded and
        reported; a bad line anywhere else raises ValueError."""
        self.last_discarded_tail = None
        if not os.path.exists(self.path):
            return []
        with open(self.path, encoding="utf-8") as fh:
            lines = [line.strip() for line in fh if line.strip()]
        raw_events: list[dict] = []
        for idx, line in enumerate(lines):
            try:
                raw = json.loads(line)
                if not isinstance(raw, dict) or not {
                        "execution_id", "seq", "type", "payload",
                        "ts_ms"} <= set(raw):
                    raise ValueError("line is not a complete event")
            except ValueError as exc:
                if idx == len(lines) - 1:
                    self.last_discarded_tail = line
                    break
                raise ValueError(
                    f"corrupt event log {self.path}: line {idx + 1} "
                    f"does not parse ({exc})") from exc
            raw_events.append(raw)
        return raw_events

    def read(self, execution_id: str) -> list[DomainEvent]:
        events = []
        for raw in self._iter_raw():
            if raw.get("execution_id") == execution_id:
                try:
                    events.append(DomainEvent(**raw))
                except TypeError as exc:
                    raise ValueError(
                        f"corrupt event log {self.path}: event at seq "
                        f"{raw.get('seq')} has missing fields ({exc})"
                    ) from exc
        return sorted(events, key=lambda e: e.seq)

    def list_runs(self) -> list[str]:
        runs: list[str] = []
        for raw in self._iter_raw():
            execution_id = raw.get("execution_id")
            if execution_id is not None and execution_id not in runs:
                runs.append(execution_id)
        return runs


class SqliteStore:
    """SQLite-backed event store (stdlib sqlite3, adapters layer only).

    Schema: one append-only table,

        events(execution_id TEXT, seq INTEGER, type TEXT,
               payload TEXT /* JSON */, ts_ms INTEGER,
               schema_version INTEGER,
               PRIMARY KEY (execution_id, seq))

    The composite primary key is the integrity rule: a second append for
    the same (run, seq) fails with sqlite3.IntegrityError instead of
    rewriting history. Reads are ordered by seq. Each operation uses a
    short-lived connection and commits before returning, so separate
    store instances (including in separate processes) pointing at the
    same file see each other's events immediately. WAL + FULL sync are
    requested so a committed event survives a process crash.

    Leases live in a second table, leases(execution_id PRIMARY KEY,
    owner, expires_ms): one row per execution, written in the same
    database. A torn write cannot produce a partial row or a partial
    lease: each statement commits or does not happen.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    execution_id   TEXT    NOT NULL,
                    seq            INTEGER NOT NULL,
                    type           TEXT    NOT NULL,
                    payload        TEXT    NOT NULL,
                    ts_ms          INTEGER NOT NULL,
                    schema_version INTEGER NOT NULL,
                    PRIMARY KEY (execution_id, seq)
                )
                """)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS leases (
                    execution_id TEXT PRIMARY KEY,
                    owner        TEXT    NOT NULL,
                    expires_ms   INTEGER NOT NULL
                )
                """)
        self.last_discarded_tail: str | None = None

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=FULL")
        return conn

    def append(self, event: DomainEvent) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO events (execution_id, seq, type, payload,"
                " ts_ms, schema_version) VALUES (?, ?, ?, ?, ?, ?)",
                (event.execution_id, event.seq, event.type,
                 json.dumps(event.payload, sort_keys=True, default=str),
                 event.ts_ms, event.schema_version))

    def flush(self, timeout: float | None = None) -> None:
        """No-op: each append commits before it returns."""

    def acquire_lease(self, execution_id: str, owner: str,
                      ttl_ms: int) -> bool:
        now_ms = int(time.time() * 1000)
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT owner, expires_ms FROM leases"
                " WHERE execution_id = ?", (execution_id,)).fetchone()
            if row is not None and row[0] != owner and row[1] > now_ms:
                conn.execute("ROLLBACK")
                return False
            conn.execute(
                "INSERT OR REPLACE INTO leases"
                " (execution_id, owner, expires_ms) VALUES (?, ?, ?)",
                (execution_id, owner, now_ms + ttl_ms))
            conn.execute("COMMIT")
        return True

    def release_lease(self, execution_id: str, owner: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM leases WHERE execution_id = ? AND owner = ?",
                (execution_id, owner))

    def read(self, execution_id: str) -> list[DomainEvent]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT execution_id, seq, type, payload, ts_ms,"
                " schema_version FROM events WHERE execution_id = ?"
                " ORDER BY seq ASC", (execution_id,)).fetchall()
        return [DomainEvent(execution_id=row[0], seq=row[1], type=row[2],
                            payload=json.loads(row[3]), ts_ms=row[4],
                            schema_version=row[5])
                for row in rows]

    def list_runs(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT execution_id FROM events GROUP BY execution_id"
                " ORDER BY MIN(rowid) ASC").fetchall()
        return [row[0] for row in rows]
