# Correct-by-Construction Agent Scaffolding — Architecture

A reusable, durable agent harness framework. It generalizes the System B
prototype from the A/B experiment (`src/system_b.py`) into a framework others
can build on — with the philosophy applied recursively: the framework itself
rejects malformed agent definitions at *build* time, not runtime.

---

## 1. Requirements analysis: what "correct by construction" demands

The core claim: an agent system should be *incapable* of its failure modes,
not merely *discouraged* from them and *measured* after the fact. Evals then
cover only the irreducible residual (model judgment on novel inputs), not the
whole system.

### Functional requirements

- **FR1 — Deterministic control plane.** Orchestration is code (a declared
  state machine), never model-decided. The model produces *data within a
  state*; transitions fire on data plus guards.
- **FR2 — Constrained model outputs.** Every model call declares an output
  schema; non-conforming output is rejected by the harness, never parsed
  hopefully.
- **FR3 — Capability-gated tools.** The set of invocable tools is a pure
  function of current state. Through the runner, an illegal action is
  unrepresentable, not forbidden-by-prompt: the envelope is rejected
  before invocation and arguments are validated against declared
  schemas. Definition-author code and direct adapter use are outside
  this guarantee (see `docs/CLAIMS.md`, C2).
- **FR4 — Verifier gates with fail-closed semantics.** Each state may declare
  entry/exit gates. A gate returns a verdict; on rejection the harness runs a
  *bounded* correction loop (reprompt with the reason, N attempts), then
  transitions to an escalation state. It never proceeds silently past a
  failed gate.
- **FR5 — Policy as pure functions.** Business decisions (refund tiers,
  approval thresholds, routing rules) are total pure functions over explicit
  inputs — no policy lives in prompts.
- **FR6 — Audit by construction.** Every state entry, model call, gate
  verdict, tool invocation, and budget event is appended to an event log.
  Observability is not bolted on; the runner cannot act without recording.
- **FR7 — Budgets as hard limits.** Step, tool-call, and correction
  budgets are entities the runner consults *before* each priced
  action; exhaustion is a first-class transition to a terminal
  state, not an exception. Token budgets are accounted after each
  call (a bounded one-call overrun). Latency and cost budgets are
  not implemented (see `docs/CLAIMS.md`, C4).
- **FR8 — Retry-safe effects.** Tool adapters must be idempotent or carry
  idempotency keys; the harness may retry a state, never blindly re-execute
  a side effect.

### Non-functional requirements

- **NFR1 — Deterministic replay.** Given an event log, execution is
  reproducible (model calls recorded/stubbed). Enables regression testing
  across model upgrades: replay old episodes against a new model and diff.
- **NFR2 — Testability without a model.** Every layer testable via test
  doubles; the core has zero I/O and zero model dependencies.
- **NFR3 — Framework-agnostic core.** No dependency on LangChain, Strands,
  or any agent framework in the core; they attach as adapters.
- **NFR4 — Long-lived definitions.** Agent definitions, event schemas, and
  the harness API are versioned; resume compatibility is enforced by
  content fingerprint (see §14). Event-schema migration paths are
  roadmap, not built.
- **NFR5 — Ergonomic strictness.** The definition API makes the right thing
  easy and the wrong thing a build-time error (see §13).

### Explicit non-goals

- Does not make the model deterministic. Judgment on novel inputs remains
  stochastic and remains measured (the bounded residual).
- Does not replace evals. It *shrinks* what evals must cover.
- Does not prescribe a model provider, vector store, or deployment target.

---

## 2. Architectural style

**CLEAN architecture** (concentric layers, dependency rule: dependencies point
inward only) composed with **ports & adapters** at the boundary. Rationale:

- The *thesis* lives in the core as pure logic; if the core is pure, the
  thesis is testable, portable, and durable.
- Everything that can change (model provider, storage, tools, frameworks)
  sits behind a narrow port, so a 5-year lifecycle survives provider churn.
- SOLID falls out naturally when each layer has one reason to change.

---

## 3. Layer map

```
┌─────────────────────────────────────────────────────────┐
│ Frameworks & Drivers                                    │
│  openai_adapter · anthropic_adapter · jsonl_store ·      │
│  http_server · cli · langchain_bridge                   │  ← volatile
├─────────────────────────────────────────────────────────┤
│ Interface Adapters                                      │
│  ModelClientAdapter · ToolRegistryAdapter · GateAdapters│
│  EventStoreAdapter · Replayer · Presenters              │  ← translate
├─────────────────────────────────────────────────────────┤
│ Use Cases (application business rules)                  │
│  StateMachineRunner · GateEvaluator · BudgetEnforcer    │
│  ports: ModelPort · ToolPort · GatePort · EventStorePort │
│         ClockPort                                       │  ← orchestrates
├─────────────────────────────────────────────────────────┤
│ Entities (enterprise business rules)                    │
│  State · Transition · Policy · Budget · DomainEvent     │
│  Verdict · HarnessError                                 │  ← pure, eternal
└─────────────────────────────────────────────────────────┘
```

Dependency rule: `usecases` may depend on `core`; `adapters` may depend on
`usecases` ports and `core`; nothing depends outward. `core` imports nothing
but the standard library.

---

## 4. Core entities (`core/`)

Pure data + pure functions. No I/O, no time, no randomness, no model.

- **`State`** — name, declared entry/exit gates, terminal flag. A value
  object.
- **`Transition`** — `(from_state, event, guard, to_state)`. Guards are pure
  predicates over `(state_data, context)`.
- **`StateGraph`** — the full declared graph. Validated at build time (§13).
- **`Policy`** — a named pure function `(inputs) -> Decision`. Total:
  defined for every input in its domain (cf. `decide()` in the experiment).
- **`Budget`** — `{token_limit, step_limit, latency_limit_ms, cost_limit}` +
  consumed counters. `check(action_cost) -> ok | exhausted`. Pure accounting;
  the runner acts on the answer.
- **`DomainEvent`** — append-only facts: `StateEntered`, `ModelCalled`
  (prompt hash, schema id, response hash — not raw PII by default),
  `GateVerdict`, `ToolInvoked` (idempotency key), `BudgetExhausted`,
  `Escalated`. Each carries `schema_version`.
- **`Verdict`** — `pass | fail(reason)`. Returned by gates.
- **`HarnessError`** — `GateRejected`, `BudgetExceeded`,
  `InvalidTransition`, `DefinitionError` (build-time).

---

## 5. Use cases (`usecases/`)

- **`StateMachineRunner.run(definition, initial_context) -> Outcome`**
  The heart. Loop: enter state → run entry gates → perform state's work
  (model call and/or tool invocations via ports) → run exit gates →
  evaluate transitions → next state. Every step appends events. The runner
  contains *no* policy, *no* prompts, *no* provider code — only the loop.
- **`GateEvaluator.evaluate(gates, subject) -> Verdict`**
  Runs a state's gate pipeline; aggregates reasons; drives the bounded
  correction loop.
- **`BudgetEnforcer`** — consults `Budget` before each priced action;
  on exhaustion emits `BudgetExhausted` and forces transition to the
  declared exhausted-state.
- **`Replayer.replay(event_log) -> ExecutionTrace`** — folds events into a
  trace for audit, debugging, and model-upgrade regression.

---

## 6. Ports — the seams (`usecases/ports.py`)

Narrow protocols. Everything volatile implements one.

```python
class ModelPort(Protocol):
    def generate(self, prompt: ModelPrompt, schema: OutputSchema) -> ModelResult: ...

class ToolPort(Protocol):
    def invoke(self, call: ToolCall) -> ToolResult: ...   # must be idempotent

class GatePort(Protocol):
    def check(self, subject: GateSubject) -> Verdict: ...

class EventStorePort(Protocol):
    def append(self, event: DomainEvent) -> None: ...
    def read(self, execution_id: str) -> list[DomainEvent]: ...

class ClockPort(Protocol):
    def now_ms(self) -> int: ...
```

No god-interfaces: a test double implements one port in ten lines.

---

## 7. Adapters (`adapters/`)

- **model/** — `OpenAIAdapter` (OpenAI-compatible HTTP), `ScriptedModel`
  (scripted, deterministic — the test double), `FnModel` (wraps a plain
  function). Adapters raise typed model failures (`ModelCallError`:
  transient, permanent, output).
- **tools/** — `ToolRegistry`: handlers plus where they are legal. The
  capability declaration's single source of truth is the states' tool
  lists; the builder cross-checks the registry against it at build time
  and derives the registry's legality from it. `tools_for(state)` lists
  a state's tools for authors who want to render them into a prompt —
  the runner does not render them (§9).
- **gates/** — `SchemaGate` (output conformance), `ContentGate`
  (PII/secret patterns), `PredicateGate` (user predicate).
- **events/** — `InMemoryStore`, `JsonlStore` (append-only file;
  a torn final line is discarded on read as uncommitted),
  `SqliteStore` (stdlib sqlite3; events keyed by `(execution_id, seq)`,
  the durable default for runs that must survive a process restart).
  All implement `EventStorePort` including `list_runs()` and the
  single-writer lease. `AsyncEventStore` decorates a persistent store
  so persistence runs on a dedicated logger thread.
- **decisions/** — renders a pause request as plain text for any
  channel (CLI, chat, console); the channel is transport only (§10,
  pause).
- **replay** — `usecases/replay.py`: the fold (`reconstruct_run`),
  trace projection, and trace diff for model-upgrade regression.

---

## 8. The runner — pseudocode

```
run(definition, ctx):
  state = definition.initial
  log(StateEntered(state))
  loop:
    enforce budget for state entry
    for gate in state.entry_gates: verdict = evaluate(gate)
      if fail: return correct_or_escalate(state, verdict)
    result = perform(state, ctx)          # model call and/or tools, via ports
    for gate in state.exit_gates: verdict = evaluate(gate)
      if fail: return correct_or_escalate(state, verdict)
    next = select_transition(definition, state, result, ctx)  # pure
    if none: raise InvalidTransition      # fail-closed, never drifts
    log transition; state = next
    if state.terminal: return Outcome(state, ctx)

correct_or_escalate(state, verdict):
  repeat up to state.max_corrections:
    correction = build_correction_prompt(verdict.reason)
    result = perform(state, ctx, correction)
    if exit gates pass: continue loop
  log(Escalated); transition to definition.escalation_state
```

Notes: the model never appears in `select_transition`. There is no path
from a failed gate back into normal flow except through passing gates.

---

## 9. Capability gating

Capability is declared in exactly one place: each state's `tools`
list. The builder cross-checks the tool registry against that
declaration at build time (both directions) and derives the
registry's legality from it. At runtime the guarantee is
**enforcement, in-path**:

- The runner rejects an envelope naming a tool outside the current
  state's declared set before the registry is consulted, and
  validates every call's arguments against the tool's declared
  schema before invocation.
- The registry independently refuses out-of-state invocation, and
  dedupes by idempotency key.
- A refund tool legal only in `APPROVED` therefore cannot be invoked
  from any other state through the runner.

**Not claimed:** presentation. The runner does not render the legal
tool set into the model's prompt; the model learns tool names from
the prompt author. `tools_for(state)` exists for authors who want
to render the set themselves. The audit log shows exactly which
capabilities existed at each state and what was invoked, with
arguments and results (see `docs/CLAIMS.md`, C2).

---

## 10. Gate pipeline and fail-closed semantics

Gates are ordered; evaluation short-circuits on first failure (fail fast,
diagnose precisely). Correction loop:

1. Gate fails with reason R.
2. Harness re-invokes the state's work with R appended
   ("your output was rejected: R; fix it").
3. Bounded by `max_corrections` (default 2).
4. Still failing → `Escalated` event → transition to escalation state
   (human review queue, safe fallback response, etc.).

The escalation state is *declared* in the definition — escalation is
designed, not improvised.

---

## 11. Event sourcing and replay

- Every action appends a versioned event, and events are atomic and
  self-contained: each carries what recreating its step requires.
  The log is the system of record; control state and — at committed
  boundaries — data context are folds over it.
- **Replay:** `replay_trace` + `diff_traces` (`usecases/replay.py`)
  project a log to a comparable trace and diff two traces — this is
  how you survive model upgrades with evidence instead of hope.
- **Retention:** full payloads — rendered prompts, model outputs,
  tool arguments and results, context deltas — with declared
  redacted keys stored as hashes only, the redaction visible in the
  log and priced by recreation.
- **Steps as transactions:** a step's work commits with an explicit
  `STEP_COMMITTED` record. A transition without its commit record
  is an uncommitted transaction and never happened. A second writer
  is refused by the store's single-writer lease; a torn final line
  in the JSONL log is discarded on recovery.
- **Checkpoints:** a declared policy (every N committed steps,
  N log bytes, or N ms) writes a `CONTEXT_CHECKPOINT` carrying the
  folded context at a commit boundary. Recovery is restore +
  roll-forward: start from the latest checkpoint, fold forward to
  the last committed step.
- **Resume:** `reconstruct_run` folds a persisted log to the last
  committed boundary; `StateMachineRunner.resume` (wired as
  `BuiltHarness.resume`) continues under the same execution id.
  Committed states are never re-run: their model responses stand as
  recorded. Only the single in-flight step re-drives, its
  regenerated events verified against the stored suffix before
  anything new appends, and the rebuilt budget checked against the
  folded budget. Before any of that, resume refuses when the
  definition, a handler, a guard, a gate, or the model differs from
  what the log recorded (fingerprint v2). `resume_preflight` renders
  the same fold as an operator report without resuming.
- **Pause:** a state with a declared pause policy stops before its
  work: the awaiting state is a log event, the decision is a log
  event recorded by the declared approver, and no decision means no
  resume. Expiry of a declared timeout is fail-closed.

---

## 12. Budgets

`Budget` is an entity, not a config comment. Steps, tool calls, and
correction attempts are hard limits, each consulted *before* the
action it prices; on exhaustion the runner transitions to the
declared exhausted terminal state. Tokens are accounted after each
model call, so a token budget can be exceeded by a bounded overrun
of one call. Latency and cost budgets are not implemented. That
scoping — hard where it is hard, bounded where it is bounded — is
the entire difference between a budget and a wish (see
`docs/CLAIMS.md`, C4).

---

## 13. Build-time validation — correctness, recursively

The framework validates agent *definitions* at construction, like a type
checker for agent graphs. `DefinitionError` at build time for:

- unreachable states; states with no inbound transition (except initial)
- non-terminal states with no outbound transition (dead ends)
- terminal states with outbound transitions
- tools registered to no state, or to undefined states; capability
  declared in the states and the registry disagreeing, in either
  direction (one source of truth, cross-checked)
- a tool used by a state with no declared argument schema, or a
  malformed one
- gates referencing undefined checks; malformed gate schemas
- missing escalation or exhausted state
- cycles without a step budget (non-termination risk)
- malformed output schemas, including union specs: a `type` list with
  an unknown member type or an empty union, or an empty/malformed
  `anyOf` (the validator supports both union forms; nullability is a
  union with `"null"`); unknown single type names are rejected, not
  ignored
- unknown or malformed budget arguments; a state defined twice
- malformed checkpoint triggers; malformed pause policies (undefined
  or terminal state, empty approver or channel, empty, duplicated,
  or unknown decision set, non-positive timeout)
- duplicate or malformed account, redaction, frozen-key, and
  declared-writes declarations; ledger posting rules naming
  undeclared accounts

If it builds, the graph is well-formed. This is the philosophy applied to
the framework itself.

---

## 14. Versioning and long-lived lifecycle

- **Content fingerprints (implemented):** every run records a
  per-component fingerprint of its definition — graph, handlers,
  guards, gates, models, policies — digested from compiled content,
  not names. Resume refuses, before appending anything, when any
  component differs from the one the log was written under. Logs
  written under the legacy name-based fingerprint resume under that
  weaker check, and the outcome says so.
- **Definition versions:** a definition carries `version`; it is
  recorded at `RUN_STARTED`. It is not compared — the fingerprint,
  not the label, is the compatibility check.
- **Event schema versions:** events carry `SCHEMA_VERSION`.
  Migrations between event schema versions do not exist yet;
  they are roadmap, and no claim is made for them.
- **Harness API:** semantic versioning; ports are the stability boundary —
  adapters may churn, ports change on major versions only.
- **Golden-log regression suite:** a checked-in corpus of event logs
  lands with v1.1; replaying the corpus against new code is a test,
  and behavior drift is a failure.

---

## 15. SOLID traceability

| Principle | Where it lives |
|---|---|---|
| SRP | Runner transitions; Gates verdict; Policy decides; Budget accounts; each adapter handles one external concern. |
| OCP | New states, tools, gates, policies register via the builder; `StateMachineRunner` is never modified to add behavior. |
| LSP | `StubModel` substitutes for `OpenAIAdapter` anywhere a `ModelPort` is expected — tests, replay, dry runs. Same for event stores. |
| ISP | Five narrow ports instead of one infrastructure interface; a gate author implements only `GatePort`. |
| DIP | Use cases depend on port protocols; concrete adapters are injected at the composition root (`app/builder.py`). |

---

## 16. Module layout

```
harness/
  core/           # entities — stdlib only
    state.py transition.py policy.py budget.py events.py errors.py
  usecases/       # runner, gate evaluator, budget enforcer, replayer
    runner.py gates.py budgets.py replay.py ports.py
  adapters/
    model/        # openai.py anthropic.py stub.py
    tools/        # registry.py
    gates/        # schema.py policy_gate.py budget_gate.py content.py
    events/       # memory.py jsonl.py
  app/            # composition root — the only place that news things up
    builder.py config.py
  docs/           # this file, ADRs, migration notes
```

---

## 17. Worked reference: the refund desk

The A/B experiment's refund task is the reference implementation: states
`RECEIVED → CLASSIFIED → DECIDED → RESPONDED`, terminal states
`DONE / ESCALATED / BUDGET_EXHAUSTED`; `decide()` is the Policy entity;
`issue_refund` legal only in `APPROVE`; outbound validated by `SchemaGate`
+ `ContentGate`; every episode replayable from its event log.

---

## 18. Open decisions

1. **Event payload retention** — *decided:* full payloads by default,
   with declared redacted keys stored as hashes (see §11). PII
   hygiene is a declaration, not a global default.
2. **Correction-loop model** — *built (v1.1):* a run may declare a
   separate correction model; retries after a failure go to it, the
   primary model corrects itself otherwise, and both are bound by
   the fingerprint. Which to use per deployment stays open.
3. **Distribution** — durable execution within a process boundary
   has landed (SQLite/JSONL stores, commit records, single-writer
   lease, torn-tail discard, checkpoints, deterministic resume —
   §11); cross-process/distributed execution is still a later
   adapter, not a core concern.
4. **Human-in-the-loop** — *built (v1.1):* the pause is a declared,
   configurable policy per state — trigger state, approver,
   decisions, channel, fail-closed timeout — with the awaiting
   state and the decision both recorded as log events (§11).
