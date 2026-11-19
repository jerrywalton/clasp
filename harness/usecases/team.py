"""Orchestrator: the team runner.

The Orchestrator owns the team run and is the single writer of the
team log (docs/ORCHESTRATION-DESIGN-2026-10-31.md, rule 4). It runs
each member through the existing StateMachineRunner — no new runtime
concepts below the team layer — in a sub-ledger: the member's own
execution id in the same event store. The team log carries each
sub-ledger's digest in its MEMBER_FINISHED event; if a handoff is
not in the team log, it did not happen.

Routing is declared, never emergent. After a member finishes, the
Orchestrator selects among the member's declared handoffs: an
explicit request (the member outcome's ``next_member`` key) when one
is present — refused and recorded when it names anything but a
declared successor — otherwise the first handoff whose guard passes
over team context. A handoff is an effect: HANDOFF_INTENDED records
intent (sender, receiver, context digest, grant) before it executes,
HANDOFF_COMMITTED the outcome after it lands.

Control state is a fold over the team log, like a run's: the log
carries the initial context, each successful member's committed
context delta, each committed handoff's carried context, and every
member's spend, so resume() rebuilds the team from the log alone.

Budgets compose. A member runs under its declared sub-allocation
(the member runner's Budget); a member that exhausts it stops, its
uncommitted writes never reach team context, and the team run ends.
The team envelope is a second, outer limit over summed member spend.

The delegation invariant is enforced at this layer as a checked,
recorded contract: the effective grant of every handoff was
validated against the sender's authority at build time and is
re-checked here before the handoff executes; the context the
receiver actually receives is exactly the grant's declared keys; and
the carried delegation depth decrements at every handoff — a
handoff that would carry depth below zero is refused and recorded.
"""

from __future__ import annotations

import copy
import hashlib
import json
import uuid
from dataclasses import dataclass, field, replace

from core.budget import Budget
from core.errors import HarnessError
from core.events import (
    BUDGET_EXHAUSTED, HANDOFF_COMMITTED, HANDOFF_INTENDED,
    HANDOFF_REFUSED, MEMBER_FINISHED, MEMBER_STARTED, MODEL_CALLED,
    STEP_COMMITTED, TEAM_DECIDED, TEAM_FINISHED, TEAM_PAUSED,
    TEAM_STARTED, TOOL_INVOKED, DomainEvent, make_event,
)
from core.team import (
    NEXT_MEMBER_KEY, Authority, HandoffDef, TeamDef,
    authority_violations,
)
from usecases.ports import ClockPort, EventStorePort, ModelPort
from usecases.runner import Definition, Outcome, StateMachineRunner

# Team run statuses (TEAM_FINISHED payloads and TeamOutcome.status).
STATUS_COMPLETED = "completed"
STATUS_BUDGET_EXHAUSTED = "budget_exhausted"
STATUS_MEMBER_FAILED = "member_failed"
STATUS_MEMBER_PAUSED = "member_paused"
STATUS_REFUSED = "refused"
STATUS_DENIED = "denied"
STATUS_NO_ROUTE = "no_route"
STATUS_HANDOFF_LIMIT = "handoff_limit"
STATUS_PAUSED = "paused"


@dataclass
class MemberRuntime:
    """A built member's runtime bundle, assembled by the composition
    root (app/team_builder.py) from a BuiltHarness: the member's
    definition plus the ports its runner needs. The Orchestrator
    depends on this bundle, never on member internals."""
    definition: Definition
    tools: object  # ToolPort (the member's tool registry)
    gates: dict


@dataclass
class TeamOutcome:
    execution_id: str
    status: str
    context: dict
    events: list[DomainEvent]
    success: bool
    awaiting_decision: bool = False
    notes: tuple[str, ...] = ()


def sub_ledger_digest(events: list[DomainEvent]) -> str:
    """Digest of a member sub-ledger: what the team log cites as
    proof of exactly which member run it summarizes."""
    canonical = json.dumps(
        [[e.seq, e.type, e.payload]
         for e in sorted(events, key=lambda e: e.seq)],
        sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _context_digest(context: dict) -> str:
    canonical = json.dumps(context, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _member_spend(events: list[DomainEvent]) -> dict:
    """A member run's spend, folded from its own log: steps
    committed, tokens across model calls, tool calls invoked."""
    tokens = 0
    steps = 0
    tool_calls = 0
    for event in events:
        if event.type == MODEL_CALLED:
            tokens += int(event.payload.get("prompt_tokens") or 0)
            tokens += int(event.payload.get("completion_tokens") or 0)
        elif event.type == STEP_COMMITTED:
            steps += 1
        elif event.type == TOOL_INVOKED:
            tool_calls += 1
    return {"steps": steps, "tokens": tokens, "tool_calls": tool_calls}


def _authority_from_summary(summary: dict) -> Authority:
    return Authority(
        capabilities=frozenset(summary.get("capabilities") or ()),
        budget_tokens=summary.get("budget_tokens"),
        window_ms=summary.get("window_ms"),
        delegation_depth=int(summary.get("delegation_depth") or 0),
        context_keys=frozenset(summary.get("context_keys") or ()))


@dataclass
class _DriveState:
    ctx: dict
    spend: dict = field(
        default_factory=lambda: {"steps": 0, "tokens": 0,
                                 "tool_calls": 0})
    handoffs_done: int = 0
    members_run: list = field(default_factory=list)
    carried_depth: int = 0


@dataclass
class _Folded:
    """Team state reconstructed from the team log alone."""
    initial_context: dict = field(default_factory=dict)
    ctx: dict = field(default_factory=dict)
    spend: dict = field(
        default_factory=lambda: {"steps": 0, "tokens": 0,
                                 "tool_calls": 0})
    handoffs_done: int = 0
    members_run: list = field(default_factory=list)
    activations: dict = field(default_factory=dict)
    carried_depth: int = 0
    last_finished: str | None = None
    last_requested: object = None
    pending_run: tuple | None = None  # (member, initial_ctx, Authority)
    pending_pause: DomainEvent | None = None
    pause_decision: str | None = None
    finished: bool = False


class TeamRunner:
    """The Orchestrator for one TeamDef."""

    def __init__(self, team: TeamDef, members: dict[str, MemberRuntime],
                 *, models: dict[str, ModelPort],
                 events: EventStorePort, clock: ClockPort,
                 default_model: ModelPort | None = None) -> None:
        self.team = team
        self._members = members
        self._models = models
        self._default_model = default_model
        self.events = events
        self.clock = clock
        self._seq = 0
        self._local: list[DomainEvent] = []
        self._activations: dict[str, int] = {}
        self._lease_owner: str | None = None

    # ------------------------------------------------------------ logging
    def _log(self, type: str, payload: dict) -> DomainEvent:
        event = make_event(self._exec, self._seq, self.clock.now_ms(),
                           type, payload)
        self._seq += 1
        self.events.append(event)
        self._local.append(event)
        return event

    def _acquire_lease(self, execution_id: str) -> None:
        acquire = getattr(self.events, "acquire_lease", None)
        if acquire is None:
            return
        owner = f"orchestrator-{uuid.uuid4().hex[:8]}"
        if not acquire(execution_id, owner, 60_000):
            raise HarnessError(
                f"team execution '{execution_id}' is leased by "
                f"another writer")
        self._lease_owner = owner

    def _release_lease(self, execution_id: str) -> None:
        if self._lease_owner is None:
            return
        release = getattr(self.events, "release_lease", None)
        if release is not None:
            try:
                release(execution_id, self._lease_owner)
            except Exception:  # noqa: BLE001 - never mask the outcome
                pass
        self._lease_owner = None

    # ---------------------------------------------------------------- run
    def run(self, initial_context: dict | None = None,
            execution_id: str | None = None) -> TeamOutcome:
        self._exec = execution_id or uuid.uuid4().hex
        self._acquire_lease(self._exec)
        try:
            self._seq = 0
            self._local = []
            self._activations = {}
            ctx = copy.deepcopy(initial_context or {})
            self._log(TEAM_STARTED, {
                "team": self.team.name,
                "version": self.team.version,
                "entry": self.team.entry,
                "initial_context": copy.deepcopy(ctx),
                "envelope": dict(self.team.envelope_kwargs),
                "composer": self.team.composer.summary(),
                "max_handoffs": self.team.max_handoffs,
            })
            entry = self.team.member(self.team.entry)
            carried = {key: ctx[key]
                       for key in entry.authority.context_keys
                       if key in ctx}
            state = _DriveState(
                ctx=ctx, carried_depth=entry.authority.delegation_depth)
            return self._drive(state, ("run", self.team.entry, carried,
                                       entry.authority))
        finally:
            self.events.flush()
            self._release_lease(self._exec)

    def resume(self, execution_id: str) -> TeamOutcome:
        """Continue a team run from its log: after a decision on a
        paused handoff, or from the last committed boundary after an
        interruption. Raises HarnessError when the log holds no
        pending work — including a pause nobody has decided."""
        self._exec = execution_id
        stored = sorted(self.events.read(execution_id),
                        key=lambda e: e.seq)
        if not stored:
            raise HarnessError(
                f"cannot resume: no persisted events for team "
                f"execution '{execution_id}'")
        folded = self._fold(stored)
        if folded.finished:
            raise HarnessError(
                f"cannot resume: team execution '{execution_id}' "
                f"already finished; a finished team log is not "
                f"rewritten")
        self._acquire_lease(execution_id)
        try:
            self._seq = stored[-1].seq + 1
            self._local = []
            self._activations = dict(folded.activations)
            state = _DriveState(
                ctx=folded.ctx, spend=folded.spend,
                handoffs_done=folded.handoffs_done,
                members_run=list(folded.members_run),
                carried_depth=folded.carried_depth)

            if folded.pending_pause is not None:
                pause = folded.pending_pause
                if folded.pause_decision is None:
                    raise HarnessError(
                        f"no decision, no resume: team execution "
                        f"'{execution_id}' is paused at the handoff "
                        f"'{pause.payload.get('from_member')}' -> "
                        f"'{pause.payload.get('to_member')}' awaiting "
                        f"'{pause.payload.get('approver')}'")
                if folded.pause_decision not in ("approve", "deny"):
                    raise HarnessError(
                        f"cannot resume: decision "
                        f"'{folded.pause_decision}' on the paused "
                        f"handoff of team execution '{execution_id}' "
                        f"is not one the team core can act on "
                        f"(approve or deny)")
                if folded.pause_decision == "deny":
                    return self._finish(
                        state, STATUS_DENIED,
                        notes=(f"handoff "
                               f"'{pause.payload.get('from_member')}' "
                               f"-> '{pause.payload.get('to_member')}' "
                               f"denied by "
                               f"'{pause.payload.get('approver')}'; "
                               f"the team does not resume",))
                carried_ctx = copy.deepcopy(
                    pause.payload.get("carried_context") or {})
                carried_auth = _authority_from_summary(
                    pause.payload.get("carried_authority") or {})
                self._log(HANDOFF_COMMITTED, {
                    "from_member": pause.payload.get("from_member"),
                    "to_member": pause.payload.get("to_member"),
                    "reason": pause.payload.get("reason"),
                    "context_digest": _context_digest(carried_ctx),
                    "carried_context": carried_ctx,
                    "carried_authority": carried_auth.summary(),
                    "decided": True,
                })
                state.handoffs_done += 1
                action = ("run", pause.payload.get("to_member"),
                          carried_ctx, carried_auth)
                return self._drive(state, action)

            if folded.pending_run is not None:
                name, initial_ctx, authority = folded.pending_run
                return self._drive(
                    state, ("run", name, initial_ctx, authority))
            if folded.last_finished is not None:
                return self._drive(
                    state, ("route", folded.last_finished,
                            folded.last_requested))
            entry = self.team.member(self.team.entry)
            carried = {key: folded.initial_context[key]
                       for key in entry.authority.context_keys
                       if key in folded.initial_context}
            state.ctx = copy.deepcopy(folded.initial_context)
            state.carried_depth = entry.authority.delegation_depth
            return self._drive(state, ("run", self.team.entry, carried,
                                       entry.authority))
        finally:
            self.events.flush()
            self._release_lease(self._exec)

    # ---------------------------------------------------------------- fold
    def _fold(self, stored: list[DomainEvent]) -> _Folded:
        folded = _Folded()
        for event in stored:
            payload = event.payload or {}
            if event.type == TEAM_STARTED:
                folded.initial_context = copy.deepcopy(
                    payload.get("initial_context") or {})
                folded.ctx = copy.deepcopy(folded.initial_context)
            elif event.type == MEMBER_STARTED:
                member = payload.get("member")
                folded.activations[member] = \
                    folded.activations.get(member, 0) + 1
                folded.carried_depth = int(
                    (payload.get("carried_authority") or {})
                    .get("delegation_depth") or 0)
                folded.pending_run = (
                    member,
                    copy.deepcopy(payload.get("initial_context") or {}),
                    _authority_from_summary(
                        payload.get("carried_authority") or {}))
            elif event.type == MEMBER_FINISHED:
                member = payload.get("member")
                folded.members_run.append(member)
                spend = payload.get("spend") or {}
                for key in folded.spend:
                    folded.spend[key] += int(spend.get(key) or 0)
                if payload.get("outcome") == "success":
                    folded.ctx.update(copy.deepcopy(
                        payload.get("context_delta") or {}))
                    folded.last_finished = member
                    folded.last_requested = payload.get("requested_next")
                folded.pending_run = None
            elif event.type == HANDOFF_COMMITTED:
                folded.handoffs_done += 1
                folded.pending_run = (
                    payload.get("to_member"),
                    copy.deepcopy(payload.get("carried_context") or {}),
                    _authority_from_summary(
                        payload.get("carried_authority") or {}))
            elif event.type == TEAM_PAUSED:
                folded.pending_pause = event
                folded.pending_run = None
            elif event.type == TEAM_DECIDED:
                if folded.pending_pause is not None:
                    folded.pause_decision = payload.get("decision")
            elif event.type == TEAM_FINISHED:
                folded.finished = True
        return folded

    # --------------------------------------------------------------- drive
    def _drive(self, state: _DriveState, action: tuple) -> TeamOutcome:
        while True:
            if action[0] == "run":
                _, name, initial_ctx, effective = action
                state.carried_depth = effective.delegation_depth
                outcome, spend, exhausted = self._run_member(
                    state, name, initial_ctx, effective)
                state.members_run.append(name)
                for key in state.spend:
                    state.spend[key] += spend[key]
                member = self.team.member(name)
                if exhausted:
                    return self._finish(
                        state, STATUS_BUDGET_EXHAUSTED,
                        notes=(f"member '{name}' exhausted its "
                               f"sub-allocation and stopped; its "
                               f"uncommitted writes were not merged",))
                if outcome.awaiting_decision:
                    return self._finish(
                        state, STATUS_MEMBER_PAUSED,
                        notes=(f"member '{name}' paused for a "
                               f"decision inside its own run; the "
                               f"team stops at the boundary",))
                if not outcome.success:
                    return self._finish(
                        state, STATUS_MEMBER_FAILED,
                        notes=(f"member '{name}' ended in state "
                               f"'{outcome.final_state}'",))
                delta = {key: copy.deepcopy(outcome.context[key])
                         for key in member.writes
                         if key in outcome.context}
                state.ctx.update(delta)
                if self._envelope_exceeded(state):
                    return self._finish(
                        state, STATUS_BUDGET_EXHAUSTED,
                        notes=("team envelope exceeded by summed "
                               "member spend",))
                if member.terminal:
                    return self._finish(state, STATUS_COMPLETED)
                action = ("route", name,
                          outcome.context.get(NEXT_MEMBER_KEY))
            else:  # "route"
                _, name, requested = action
                step = self._route(state, name, requested)
                if step[0] == "finish":
                    return self._finish(state, step[1], notes=step[2])
                if step[0] == "pause":
                    return TeamOutcome(
                        execution_id=self._exec, status=STATUS_PAUSED,
                        context=copy.deepcopy(state.ctx),
                        events=list(self._local), success=False,
                        awaiting_decision=True,
                        notes=(f"paused at the handoff "
                               f"'{step[1].from_member}' -> "
                               f"'{step[1].to_member}' awaiting "
                               f"'{step[1].pause.approver}'",))
                _, handoff, effective, carried_ctx = step
                action = ("run", handoff.to_member, carried_ctx,
                          effective)

    def _route(self, state: _DriveState, name: str, requested) -> tuple:
        """Select and execute the next declared handoff out of
        ``name``. Returns ("run", handoff, effective, carried_ctx),
        ("pause", handoff), or ("finish", status, notes)."""
        member = self.team.member(name)
        successors = self.team.successors(name)
        if state.handoffs_done >= self.team.max_handoffs:
            return ("finish", STATUS_HANDOFF_LIMIT,
                    (f"team handoff limit reached "
                     f"({self.team.max_handoffs})",))
        selected: HandoffDef | None = None
        if requested is not None:
            matches = [h for h in successors
                       if h.to_member == requested]
            if not matches:
                declared = sorted({h.to_member for h in successors})
                self._log(HANDOFF_REFUSED, {
                    "from_member": name,
                    "to_member": requested,
                    "reason": "undeclared handoff",
                    "detail": f"member '{name}' requested "
                              f"'{requested}', which is not a "
                              f"declared successor (declared: "
                              f"{declared})",
                })
                return ("finish", STATUS_REFUSED,
                        (f"member '{name}' requested undeclared "
                         f"handoff to '{requested}'; refused and "
                         f"recorded",))
            selected = matches[0]
        else:
            for handoff in successors:
                if handoff.guard is None \
                        or handoff.guard(dict(state.ctx)):
                    selected = handoff
                    break
            if selected is None:
                return ("finish", STATUS_NO_ROUTE,
                        (f"no declared handoff out of '{name}' "
                         f"matched the team context",))

        receiver = self.team.member(selected.to_member)
        effective = selected.grant if selected.grant is not None \
            else receiver.authority
        violations = authority_violations(
            effective, member.authority,
            child_label=f"handoff '{name}' -> "
                        f"'{selected.to_member}' grant",
            parent_label=f"sender '{name}'")
        if violations:
            self._log(HANDOFF_REFUSED, {
                "from_member": name,
                "to_member": selected.to_member,
                "reason": "grant amplifies sender authority",
                "violations": violations,
            })
            return ("finish", STATUS_REFUSED, tuple(violations))

        receiver_depth = min(effective.delegation_depth,
                             state.carried_depth - 1)
        if receiver_depth < 0:
            self._log(HANDOFF_REFUSED, {
                "from_member": name,
                "to_member": selected.to_member,
                "reason": "delegation depth exhausted",
                "detail": f"sender '{name}' carries delegation "
                          f"depth {state.carried_depth}; a handoff "
                          f"decrements it, and depth below zero "
                          f"carries no authority",
            })
            return ("finish", STATUS_REFUSED,
                    (f"handoff '{name}' -> '{selected.to_member}' "
                     f"refused: delegation depth exhausted",))
        effective = replace(effective, delegation_depth=receiver_depth)
        carried_ctx = {key: copy.deepcopy(state.ctx[key])
                       for key in effective.context_keys
                       if key in state.ctx}
        digest = _context_digest(carried_ctx)
        self._log(HANDOFF_INTENDED, {
            "from_member": name,
            "to_member": selected.to_member,
            "reason": selected.reason,
            "context_digest": digest,
            "carried_keys": sorted(carried_ctx),
            "grant": effective.summary(),
        })
        if selected.pause is not None:
            policy = selected.pause
            self._log(TEAM_PAUSED, {
                "from_member": name,
                "to_member": selected.to_member,
                "reason": selected.reason,
                "approver": policy.approver,
                "channel": policy.channel,
                "allowed_decisions": list(policy.decisions),
                "timeout_ms": policy.timeout_ms,
                "carried_context": carried_ctx,
                "carried_authority": effective.summary(),
            })
            return ("pause", selected)
        self._log(HANDOFF_COMMITTED, {
            "from_member": name,
            "to_member": selected.to_member,
            "reason": selected.reason,
            "context_digest": digest,
            "carried_context": carried_ctx,
            "carried_authority": effective.summary(),
        })
        state.handoffs_done += 1
        return ("run", selected, effective, carried_ctx)

    # -------------------------------------------------------------- member
    def _run_member(self, state: _DriveState, name: str,
                    initial_ctx: dict,
                    effective: Authority) -> tuple[Outcome, dict, bool]:
        runtime = self._members[name]
        member = self.team.member(name)
        model = self._models.get(name) or self._default_model
        if model is None:
            raise HarnessError(
                f"no model supplied for team member '{name}'")
        activation = self._activations.get(name, 0) + 1
        self._activations[name] = activation
        member_exec = f"{self._exec}:{name}:{activation}"
        self._log(MEMBER_STARTED, {
            "member": name,
            "execution_id": member_exec,
            "initial_context": copy.deepcopy(initial_ctx),
            "carried_authority": effective.summary(),
        })
        runner = StateMachineRunner(
            runtime.definition, model=model, tools=runtime.tools,
            gates=runtime.gates, events=self.events, clock=self.clock,
            budget=Budget.from_kwargs(**member.budget_kwargs))
        outcome = runner.run(copy.deepcopy(initial_ctx),
                             execution_id=member_exec)
        spend = _member_spend(outcome.events)
        exhausted = any(e.type == BUDGET_EXHAUSTED
                        for e in outcome.events)
        if exhausted:
            result = "budget_exhausted"
        elif outcome.awaiting_decision:
            result = "awaiting_decision"
        elif outcome.success:
            result = "success"
        else:
            result = "failed"
        delta = {}
        if result == "success":
            delta = {key: copy.deepcopy(outcome.context[key])
                     for key in member.writes
                     if key in outcome.context}
        self._log(MEMBER_FINISHED, {
            "member": name,
            "execution_id": member_exec,
            "outcome": result,
            "final_state": outcome.final_state,
            "spend": spend,
            "context_delta": delta,
            "requested_next": outcome.context.get(NEXT_MEMBER_KEY),
            "sub_ledger_digest": sub_ledger_digest(outcome.events),
            "sub_ledger_events": len(outcome.events),
        })
        return outcome, spend, exhausted

    # -------------------------------------------------------------- finish
    def _envelope_exceeded(self, state: _DriveState) -> bool:
        envelope = self.team.envelope_kwargs
        checks = (("step_limit", "steps"), ("token_limit", "tokens"),
                  ("tool_call_limit", "tool_calls"))
        for limit_key, spend_key in checks:
            limit = envelope.get(limit_key)
            if limit is not None and state.spend[spend_key] > limit:
                return True
        return False

    def _finish(self, state: _DriveState, status: str,
                notes: tuple = ()) -> TeamOutcome:
        self._log(TEAM_FINISHED, {
            "status": status,
            "handoffs": state.handoffs_done,
            "members_run": list(state.members_run),
            "spend": dict(state.spend),
        })
        return TeamOutcome(
            execution_id=self._exec, status=status,
            context=copy.deepcopy(state.ctx),
            events=list(self._local),
            success=status == STATUS_COMPLETED,
            awaiting_decision=False, notes=tuple(notes))
