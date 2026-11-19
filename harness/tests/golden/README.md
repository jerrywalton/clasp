# Golden-log corpus

Checked-in event logs, one per scenario, with `manifest.json`
recording what the fold must find in each: event count, completion,
final/current state, budget totals, and SHA-256 digests of the
folded context and of the replay trace.

`harness/tests/test_golden_logs.py` replays this corpus on every
run: if the fold, the trace projection, or the recorded verdicts
drift from the manifest, the test fails. That is the point — the
corpus is the regression instrument for behavior that every other
test constructs fresh and so cannot see drift in.

Scenarios: `completed` (two states, a tool call), `paused_resumed`
(a pause, a recorded decision, a resume), `escalated` (correction
budget spent, escalation), `checkpointed` (a checkpoint per commit),
`interrupted` (a log cut mid-flight; it pins recovery's starting
point, not an ending).

Regenerate only when a behavior change is intended:

    python3 harness/tests/generate_golden.py

Then read the diff before committing it. The diff is the record of
what changed.
