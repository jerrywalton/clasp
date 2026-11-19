# CLASP — Control Layer for Agent Systems and Policy

CLASP is a control layer for agent systems: scaffolding that makes an
agent's control plane deterministic code instead of model discretion.

The model produces *data within a state*. Everything that matters —
which transitions may fire, which tools may be invoked, how much may be
spent, who may hand work to whom, and what actually happened — is
declared up front and enforced by the runner. Some violations are not
discouraged, detected, or retried. They are **unrepresentable**: the
definition that would permit them fails to build, or the effect cannot
be expressed through the ports the runner mediates.

This repository is the open-source core of CLASP v1.1: the enforcement
build. It is pure Python, standard library only, with an assert-based
test suite that needs no framework and no network.

## What it enforces

- **Declared grants.** Every agent definition declares its states,
  transitions, tools, gates, and budget before it runs. The builder
  validates the definition at build time (unreachable states, dead
  ends, terminal misuse, unregistered tools or gates, cycles without
  a step budget) and refuses malformed definitions.
- **Effect gates.** Tools are capability-gated per state: the model
  never calls a tool directly. The harness invokes an effect only in
  a state where that effect is declared legal, after exit gates pass.
- **Budgets as entities.** Steps, tokens, tool calls, and correction
  attempts are hard limits consulted *before* priced actions.
  Exhaustion is a first-class transition, not an exception.
- **The delegation invariant.** A hand-off may carry only a **subset**
  of the sender's authority — across capabilities, budget, time,
  delegation depth, and context — never an amplification. Fan-out
  divides the parent's budget; it never multiplies it. Team
  composition is governed by a **union bound**: a composer may not
  assemble a team whose combined authority exceeds the composer's
  own. Depth decrements at every hop, so received authority can be
  passed onward only finitely many times.
- **Audit.** Every step is recorded as a versioned, append-only event
  in a ledger (in-memory, JSONL, or SQLite). Runs can be replayed and
  reconstructed from the ledger alone; team runs keep one ledger with
  member sub-ledgers, so after the fact someone can say exactly which
  member authorized which effect.

Guarantees are scoped to runner-mediated paths. The authoritative
statement of every claim — and its limits — is
[`docs/CLAIMS.md`](docs/CLAIMS.md). The architecture is specified in
[`docs/HARNESS-ARCHITECTURE.md`](docs/HARNESS-ARCHITECTURE.md).

## Layout

```
harness/
  core/       pure entities (stdlib only): states, transitions, policy,
              budget, events, ledger, team authority, schema validator
  usecases/   the runner, gate evaluator, replay, team orchestrator,
              and the five ports (Model, Tool, Gate, EventStore, Clock)
  adapters/   model adapters (scripted / function / OpenAI-compatible),
              capability-gated tool registry, gate checks, event stores
  app/        Harness.define(...) and Team.define(...) builders with
              build-time definition validation
  tests/      assert-based suite, plus test_architecture.py, which
              enforces the layering rules (core stdlib-only; usecases
              never import adapters or app)
examples/
  refund_desk.py   a complete reference agent on the harness
docs/
  CLAIMS.md                the claims register (authoritative)
  HARNESS-ARCHITECTURE.md  the architecture specification
```

## Quickstart

Requires Python 3.11+. No dependencies.

```bash
# the core test suite (each file runs standalone)
python3 harness/tests/test_harness.py
python3 harness/tests/test_team_orchestration.py
python3 harness/tests/test_golden_logs.py

# the reference agent: four episodes through the refund desk
python3 examples/refund_desk.py
```

## Proof results

The core is exercised continuously against an adversarial proof
apparatus that is *not* part of this repository (see below). Headline
results from the 2026-11-17 proof runs, all synthetic and offline:

- **Adversarial A/B proof:** a scripted adversary attempted violations
  from a 19-class catalog across 99 loops against a three-member team.
  **99 of 99 attempts caught** by the harness — refused at build time,
  refused at handoff, stopped by budget or envelope exhaustion, or
  contained by construction. The honest worker completed **99 of 99**
  loops unharmed.
- **Whisper (collusion) test:** a bad agent with a declared channel to
  a fully persuadable worker holding legitimate authority achieved a
  100% persuasion rate — and **zero unauthorized effects**. Talk is
  not authority; whispers mint no grants.
- **Team golden scenarios:** **11 of 11** pass, covering handoff
  chains, refused undeclared handoffs, single-writer context keys,
  budget sub-allocation exhaustion, approver-gated pauses, the union
  bound, and delegation-depth exhaustion.

## What is not in this repository

CLASP has two parts. This repository is the enforcement build, and it
is complete and usable on its own.

The second part — the **factory**: the adversarial generator that
manufactures task software and attacks against the core, and the
conformance suite that scores them — is **not** open source and is
not included. It remains closed, as does ownership of the project.
The proof numbers above are published so the core's claims can be
weighed; the apparatus that produced them is the part we keep.

## License and trademark

Licensed under the Apache License, Version 2.0. See
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Copyright 2026 Symbolic Languages LLC.

CLASP™ is a trademark of Symbolic Languages LLC. The Apache-2.0
license grants no rights to the CLASP name or marks; you may use the
name to refer to this project truthfully, but not in a way that
implies endorsement by, or affiliation with, Symbolic Languages LLC.
