# Harness v1 — correct-by-construction agent scaffolding

A reusable framework for building agents whose control plane is deterministic
code, not model-decided. The model produces *data within a state*; transitions,
tool capabilities, budgets, and verifier gates are all declared up front and
enforced by the runner.

Spec: `../docs/HARNESS-ARCHITECTURE.md`

## Layout

- `core/` — pure entities (stdlib only): states, transitions, policy,
  budget, events, errors, schema-subset validator
- `usecases/` — `StateMachineRunner`, gate evaluator, replay/diff, and the
  five ports (Model, Tool, Gate, EventStore, Clock)
- `adapters/` — model (scripted/fn/OpenAI), tool registry with capability
  gating + idempotency, gate checks, in-memory/JSONL event stores
- `app/builder.py` — `Harness.define(...)` fluent builder with build-time
  definition validation (unreachable states, dead ends, terminal misuse,
  unregistered tools/gates, prompt-without-schema, cycles without step
  budget)
- `tests/` — 15 assert-based tests, no framework needed, plus `test_architecture.py`, which enforces the layering rules (core stdlib-only, usecases never import adapters/app) so generated code can't quietly break them

## Run

```bash
python3 harness/tests/test_harness.py      # 15/15
python3 examples/refund_desk.py           # reference agent, 4 episodes
```

## The contract

Every non-terminal state produces an envelope
`{"event", "data", "tool_calls"}` — via model call (constrained JSON),
pure handler, or passthrough. The runner validates the envelope, enforces
tool legality per state, runs exit gates with a bounded correction loop,
then follows a *declared* transition. Failed gates escalate; exhausted
budgets terminate; every step is logged as a versioned event.

v1 defaults for the open design questions: event payloads hashed by default
(full opt-in), same-model correction loop, in-memory execution (durable
execution deferred), escalation as a terminal state + event.
