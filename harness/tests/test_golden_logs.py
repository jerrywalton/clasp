"""Golden-log regression corpus. Plain asserts:
    python3 harness/tests/test_golden_logs.py
Replays the checked-in corpus in harness/tests/golden/ against its
manifest: event counts, fold verdicts (state, budgets), and SHA-256
digests of the folded context and the replay trace must match
exactly. Any drift in the fold, the trace projection, or event
generation fails here — regenerate the corpus only for intended
behavior changes (see golden/README.md)."""

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adapters.events.stores import JsonlStore
from core.events import RUN_STARTED
from usecases.replay import reconstruct_run, replay_trace

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "golden")


def canonical_hash(value) -> str:
    raw = json.dumps(value, sort_keys=True, default=str,
                     separators=(",", ":"))
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_manifest():
    with open(os.path.join(GOLDEN_DIR, "manifest.json"),
              encoding="utf-8") as fh:
        return json.load(fh)


def _check_scenario(name, entry):
    path = os.path.join(GOLDEN_DIR, entry["file"])
    events = JsonlStore(path).read(entry["execution_id"])
    assert len(events) == entry["events"], (name, len(events))
    assert events[0].type == RUN_STARTED
    assert [e.seq for e in events] == list(range(len(events))), name
    plan = reconstruct_run(events)
    assert plan.completed == entry["completed"], name
    assert plan.final_state == entry["final_state"], name
    assert plan.current_state == entry["current_state"], name
    assert plan.steps_used == entry["steps_used"], name
    assert plan.tokens_used == entry["tokens_used"], name
    assert plan.tool_calls_used == entry["tool_calls_used"], name
    assert plan.corrections_used == entry["corrections_used"], name
    assert canonical_hash(plan.context) == entry["context_sha256"], name
    assert canonical_hash(replay_trace(events)) == entry["trace_sha256"], \
        name
    # The fold is deterministic: folding twice changes nothing.
    again = reconstruct_run(events)
    assert canonical_hash(again.context) == entry["context_sha256"], name


def test_golden_corpus_matches_manifest():
    manifest = _load_manifest()
    assert set(manifest) == {"completed", "paused_resumed", "escalated",
                             "checkpointed", "interrupted"}, manifest
    for name, entry in sorted(manifest.items()):
        _check_scenario(name, entry)


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
