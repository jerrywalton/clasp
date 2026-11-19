# CLASP — Claims Register (corrected, v1.1)

**Status:** corrected 2026-10-31. This register supersedes every
earlier statement of these claims, in every document. Where another
document disagrees with this register, the register is right and the
document is stale.

**Rule:** a claim covers exactly the seams the runner mediates.
Guarantees are runner-mediated: they hold for work that flows
through the runner (model-proposed envelopes, declared tools,
declared gates). Definition-author code (handlers, guards) and any
direct use of adapters outside the runner are outside the guarantee,
and every claim below is written that way.

The corrections originate in the independent critique of
2026-10-27 (`docs/CRITIQUE-2026-10-27.md`): nine claim corrections,
adopted here with their wording updated for what the v1.1 builds
have since made true (atomic self-contained events, argument
schemas, typed model failures, the correction-model seam, tool
intent records, budget counting, commit records, leases, torn-tail
discard, context checkpoints, the configurable pause, frozen
context and declared writes).

---

## C1 — Build-time validation

**Corrected claim.** The builder validates the definition's
structure at build time: graph shape (unreachable states, dead
ends, terminal states with outbound transitions, cycles without a
step budget), capability declared in exactly one place (the states'
tool lists, cross-checked against the registry in both directions),
a declared argument schema for every tool a state may invoke,
well-formed output schemas and gate schemas (unions included;
unknown type names are rejected, not ignored), budget and
checkpoint and pause-policy arguments, and duplicate or malformed
declarations of every kind.

**Not claimed.** Data edges are not type-checked: a guard reading a
context key no state writes is a runtime error, not a build error.
Declared per-state writes and frozen keys are enforced at runtime
on the envelope, before effects — they are not a build-time proof
that the dataflow is coherent.

## C2 — Capability as a function of declared state

**Corrected claim.** For runner-mediated invocation, the set of
tools a model may invoke is a pure function of the current state:
the runner rejects an envelope naming a tool outside the state's
declared set before the registry is consulted, the registry
independently refuses out-of-state invocation, and every call's
arguments are validated against the tool's declared schema before
invocation.

**Not claimed.** Presentation: the runner does not render the legal
tool set into the prompt. `tools_for(state)` exists for authors who
want to render it; the guarantee is enforcement, not presentation.
Handler and guard code is the author's own and is outside this
claim.

## C3 — Proposer/disposer

**Corrected claim (the critique's defensible version, updated).**
For tasks that decompose into declared decision points, restricting
the model to proposing data and selecting among declared
transitions loses no demonstrated capability (A/B pending);
openness beyond the graph is absorbed by handler states, where the
framework's guarantees do not apply. Model-path failures are typed
(transient, permanent, output) and enter the same correction
machinery as envelope failures; they do not bypass it.

## C4 — Budgets

**Corrected claim.** Steps, tool calls, and correction attempts are
hard limits, each consulted before the action it prices; exhaustion
is a declared terminal transition, not an exception. Tokens are
accounted after each model call, so the token limit can be exceeded
by a bounded overrun of one call. Gate evaluations are not counted.
Latency and cost budgets do not exist. A budget resource binds only
when its limit is declared; a step limit is forced when the graph
has a cycle, the rest are the author's to arm.

## C5 — Audit by construction

**Corrected claim.** Events are atomic and self-contained: the log
carries the rendered prompt and full output of each model call
(successful or failed, with its typed failure kind), each tool
call's intent, arguments, and result, gate subjects and verdicts,
and the context delta of every committed step. Control state and,
at committed boundaries, data context are folds over the log.
Declared redacted keys are stored as hashes, never values; that
redaction is the one place the log deliberately withholds content,
and recreation prices it.

**Not claimed.** Author code that acts outside the runner (a
handler invoking a tool function directly) leaves no events; the
audit covers what the runner mediates. Policy purity is a
discipline the framework supports (policies as pure functions over
explicit inputs), not a property it can prove about author code.

## C6 — Ports and substrates

**Corrected claim (the critique's defensible version, updated).**
The ports sufficed for three model sources and three event stores
in-repo, plus an asynchronous logging decorator and the decision
port for human pauses. Portability to a loop-owning substrate is
untested and would require the substrate to cede the loop: the
control layer replaces the substrate's loop rather than sitting
inside it.

## C7 — The control layer is the layer worth owning

**Corrected claim.** A bet with unusually good internal evidence:
the layer is separable and enforced as separable (layering is an
AST-checked property, not a convention), and its guarantees are now
scoped exactly (this register). Its worth beyond this repo —
portability exercised across external substrates — is unproven.

---

## The nine corrections, as applied

1. **§9 presentation claim — deleted.** The architecture doc and the
   registry docstring no longer claim the model receives only
   `tools_for(current_state)`. Enforcement is the claim (see C2).
2. **§11 / events fold — rewritten to the built truth.** Context is
   a fold over the log at committed boundaries (atomic events carry
   the data); retention is full payloads with declared redactions
   stored as hashes. The old "hashes by default" and
   "control-state-only fold" wordings are gone.
3. **§12 / FR7 — scoped.** "Impossible to exceed" now applies to
   steps, tool calls, and corrections only; tokens carry the
   bounded one-call overrun in writing; latency and cost are named
   as not implemented (see C4).
4. **§13 — reconciled with the builder as built**, including
   gate-schema validation, budget-argument validation, state
   redefinition, capability cross-checks, and pause/checkpoint
   policy validation.
5. **§14 / NFR4 — marked to status.** Resume refuses a run whose
   definition, handler, guard, gate, or model content differs
   (fingerprint v2, checked before any append). `Definition.version`
   is recorded, not compared. Event-schema migrations do not exist
   and are roadmap. The golden-log corpus lands with v1.1.
6. **§7 adapter inventory — corrected to the code:** model —
   `OpenAIAdapter`, `ScriptedModel`, `FnModel`; gates — `SchemaGate`,
   `ContentGate`, `PredicateGate`; events — three stores plus the
   async decorator; decisions — the pause-request renderer. The
   fictional `AnthropicAdapter`, `PolicyGate`, and `BudgetGate` are
   removed.
7. **FR3 — scoped.** "Unrepresentable" now reads: an illegal action
   is unrepresentable *through the runner*; arguments are validated;
   author code and direct adapter use are outside the guarantee.
8. **Durability language — scoped to the built drill.** A persisted
   log resumes deterministically: committed steps fold (never
   re-run), the single in-flight step re-drives and is verified
   against the log, recorded model responses stand, and in-flight
   effects are safe under the tools' idempotency contract. Steps
   commit by explicit record; a second writer is refused by lease;
   a torn tail is discarded as uncommitted.
9. **C3 / C6 / C7 — adopted in their scoped versions** (above).
