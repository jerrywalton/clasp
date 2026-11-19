"""Composition root and build-time validation for teams.

Team.define(...) builds a validated TeamDef plus the member runtime
bundles the Orchestrator needs. Validation implements
docs/ORCHESTRATION-DESIGN-2026-10-31.md: if it builds, the team is
well-formed. An undeclared handoff, a double-written context key, an
amplifying grant, and a team whose combined authority exceeds its
composer's are DefinitionErrors here, never runtime surprises.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.builder import BuiltHarness
from core.errors import DefinitionError
from core.team import (
    Authority, HandoffDef, HandoffPausePolicy, MemberDef, TeamDef,
    composition_violations, handoff_violations,
)
from usecases.ports import ClockPort, EventStorePort, ModelPort
from usecases.team import MemberRuntime, TeamOutcome, TeamRunner

# The only keyword arguments a budget declaration accepts — the
# Budget entity's limits (same set as app.builder's, kept in step
# with core.budget.Budget).
_BUDGET_KWARGS = frozenset({"step_limit", "token_limit",
                            "tool_call_limit", "correction_limit"})


@dataclass
class BuiltTeam:
    definition: TeamDef
    members: dict[str, BuiltHarness]

    def orchestrator(self, models, events: EventStorePort,
                     clock: ClockPort,
                     default_model: ModelPort | None = None
                     ) -> TeamRunner:
        """Assemble the Orchestrator. ``models`` maps member names
        to their ModelPort; a single ModelPort (or ``default_model``)
        serves every member without its own entry."""
        if isinstance(models, dict):
            model_map = dict(models)
        elif models is not None:
            model_map = {}
            default_model = models
        else:
            model_map = {}
        runtimes = {
            name: MemberRuntime(definition=built.definition,
                                tools=built.tools, gates=built.gates)
            for name, built in self.members.items()}
        return TeamRunner(self.definition, runtimes, models=model_map,
                          events=events, clock=clock,
                          default_model=default_model)

    def run(self, models, events: EventStorePort, clock: ClockPort,
            initial_context: dict | None = None,
            execution_id: str | None = None,
            default_model: ModelPort | None = None) -> TeamOutcome:
        return self.orchestrator(
            models, events, clock, default_model=default_model).run(
                initial_context, execution_id=execution_id)

    def resume(self, models, events: EventStorePort, clock: ClockPort,
               execution_id: str,
               default_model: ModelPort | None = None) -> TeamOutcome:
        return self.orchestrator(
            models, events, clock, default_model=default_model).resume(
                execution_id)


class TeamBuilder:
    def __init__(self, name: str, version: str = "1.2.0") -> None:
        self._name = name
        self._version = version
        self._composer = Authority()
        self._entry: str | None = None
        self._members: dict[str, MemberDef] = {}
        self._built: dict[str, BuiltHarness] = {}
        self._handoffs: list[HandoffDef] = []
        self._envelope_kwargs: dict = {}
        self._max_handoffs = 32

    # ------------------------------------------------------------ fluent API
    def composer(self, authority: Authority) -> "TeamBuilder":
        """Declare the composer's own authority: the ceiling the
        union bound checks the assembled team against."""
        self._composer = authority
        return self

    def entry(self, member_name: str) -> "TeamBuilder":
        self._entry = member_name
        return self

    def member(self, name: str, built: BuiltHarness, *,
               writes: tuple[str, ...] = (),
               authority: Authority | None = None,
               budget: dict | None = None,
               terminal: bool = False) -> "TeamBuilder":
        """Add a built member harness to the roster, with its
        team-scale declaration: the team context keys it solely
        writes, its granted authority, and its budget sub-allocation
        (Budget kwargs: step_limit, token_limit, tool_call_limit,
        correction_limit)."""
        if name in self._members:
            raise DefinitionError(f"member '{name}' is already on "
                                  f"the roster")
        if not isinstance(built, BuiltHarness):
            raise DefinitionError(
                f"member '{name}': expected a BuiltHarness from "
                f"Harness.define(...).build()")
        self._members[name] = MemberDef(
            name=name, writes=tuple(writes),
            authority=authority if authority is not None else Authority(),
            budget_kwargs=dict(budget or {}), terminal=bool(terminal))
        self._built[name] = built
        return self

    def handoff(self, from_member: str, to_member: str, *,
                reason: str = "next", guard=None,
                grant: Authority | None = None,
                pause=None) -> "TeamBuilder":
        """Declare a handoff. ``pause`` is a HandoffPausePolicy or a
        dict of its fields (approver, channel, decisions,
        timeout_ms)."""
        if isinstance(pause, dict):
            pause = HandoffPausePolicy(**pause)
        self._handoffs.append(HandoffDef(
            from_member=from_member, to_member=to_member,
            reason=reason, guard=guard, grant=grant, pause=pause))
        return self

    def envelope(self, **budget_kwargs) -> "TeamBuilder":
        """Declare the team budget envelope (Budget kwargs): the
        outer limit over summed member spend."""
        self._envelope_kwargs.update(budget_kwargs)
        return self

    def max_handoffs(self, n: int) -> "TeamBuilder":
        self._max_handoffs = n
        return self

    # ----------------------------------------------------------------- build
    def build(self) -> BuiltTeam:
        self._validate()
        definition = TeamDef(
            name=self._name, version=self._version,
            members=dict(self._members), handoffs=list(self._handoffs),
            entry=self._entry, composer=self._composer,
            envelope_kwargs=dict(self._envelope_kwargs),
            max_handoffs=self._max_handoffs)
        return BuiltTeam(definition=definition,
                         members=dict(self._built))

    # ------------------------------------------------------------ validation
    def _fail(self, msg: str) -> None:
        raise DefinitionError(msg)

    def _validate(self) -> None:
        if not self._members:
            self._fail("team has no members: the roster is declared "
                       "at build time")
        if not self._entry:
            self._fail("no entry member declared")
        if self._entry not in self._members:
            self._fail(f"entry member '{self._entry}' is not on the "
                       f"roster")
        if self._members[self._entry].terminal:
            self._fail("entry member must not be terminal")

        # Handoffs are declared at build time: both endpoints must
        # be on the roster. Anything else is the undeclared-handoff
        # build error — the free swarm cannot be expressed.
        for h in self._handoffs:
            for endpoint in (h.from_member, h.to_member):
                if endpoint not in self._members:
                    self._fail(
                        f"handoff '{h.from_member}' -> '{h.to_member}' "
                        f"references member '{endpoint}', who is not "
                        f"on the roster: an undeclared handoff is a "
                        f"build error")
            if h.from_member == h.to_member:
                self._fail(f"handoff '{h.from_member}' -> "
                           f"'{h.to_member}' is a self-handoff; a "
                           f"member's internal loops belong to its "
                           f"own definition")

        outbound: dict[str, int] = {name: 0 for name in self._members}
        for h in self._handoffs:
            outbound[h.from_member] += 1
        for name, member in self._members.items():
            if member.terminal and outbound[name]:
                self._fail(f"terminal member '{name}' has outbound "
                           f"handoffs")
            if not member.terminal and not outbound[name]:
                self._fail(f"non-terminal member '{name}' is a dead "
                           f"end: no outbound handoffs, and it is "
                           f"not declared terminal")

        # Reachability from the entry member.
        adjacency: dict[str, list[str]] = {n: [] for n in self._members}
        for h in self._handoffs:
            adjacency[h.from_member].append(h.to_member)
        seen: set[str] = set()
        stack = [self._entry]
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adjacency[node])
        for name in self._members:
            if name not in seen:
                self._fail(f"member '{name}' is unreachable from "
                           f"entry member '{self._entry}'")

        # Exactly one writer per shared context key.
        owners: dict[str, str] = {}
        for name, member in self._members.items():
            if len(set(member.writes)) != len(member.writes):
                self._fail(f"member '{name}': duplicate declared "
                           f"writes")
            for key in member.writes:
                if not isinstance(key, str) or not key:
                    self._fail(f"member '{name}': declared writes "
                               f"must be non-empty strings")
                if key in owners:
                    self._fail(
                        f"context key '{key}' is written by both "
                        f"'{owners[key]}' and '{name}': exactly one "
                        f"writer per shared context key")
                owners[key] = name

        # Budget declarations: known axes, positive integer limits.
        for label, kwargs in (
                [("team envelope", self._envelope_kwargs)]
                + [(f"member '{name}'", m.budget_kwargs)
                   for name, m in self._members.items()]):
            unknown = sorted(set(kwargs) - _BUDGET_KWARGS)
            if unknown:
                self._fail(f"{label}: unknown budget argument(s): "
                           f"{', '.join(unknown)} (supported: "
                           f"{', '.join(sorted(_BUDGET_KWARGS))})")
            for key, value in kwargs.items():
                if isinstance(value, bool) \
                        or not isinstance(value, int) or value <= 0:
                    self._fail(f"{label}: budget '{key}' must be a "
                               f"positive integer, got {value!r}")

        # A sub-allocation divides the member's own declared budget,
        # never multiplies it: where the member's definition declares
        # a limit on an axis, the team's sub-allocation on that axis
        # may not exceed it.
        for name, member in self._members.items():
            own = self._built[name].budget_kwargs
            for axis, own_limit in own.items():
                sub = member.budget_kwargs.get(axis)
                if sub is not None and sub > own_limit:
                    self._fail(
                        f"member '{name}': sub-allocation {axis}="
                        f"{sub} exceeds the member's own declared "
                        f"budget {own_limit}: a sub-allocation "
                        f"divides the member's budget, never "
                        f"multiplies it")

        # The union bound on composition.
        probe = TeamDef(name=self._name, version=self._version,
                        members=self._members, handoffs=self._handoffs,
                        entry=self._entry, composer=self._composer,
                        envelope_kwargs=self._envelope_kwargs,
                        max_handoffs=self._max_handoffs)
        problems = composition_violations(probe)
        if problems:
            self._fail("team composition exceeds the composer's "
                       "authority: " + "; ".join(problems))

        # Every handoff's grant must attenuate, never amplify.
        for h in self._handoffs:
            problems = handoff_violations(
                h, self._members[h.from_member],
                self._members[h.to_member])
            if problems:
                self._fail("handoff amplifies authority: "
                           + "; ".join(problems))

        # Pause policies on handoffs: same discipline as state
        # pauses (builder.pause), except the decision set is
        # approve/deny only — an amend at a handoff boundary would
        # rewrite shared context between members, which the team
        # core does not do; deny and redefine instead.
        for h in self._handoffs:
            policy = h.pause
            if policy is None:
                continue
            label = (f"pause policy on handoff '{h.from_member}' -> "
                     f"'{h.to_member}'")
            if not isinstance(policy.approver, str) \
                    or not policy.approver:
                self._fail(f"{label}: approver must be a non-empty "
                           f"string")
            if not isinstance(policy.channel, str) \
                    or not policy.channel:
                self._fail(f"{label}: channel must be a non-empty "
                           f"string")
            if not policy.decisions:
                self._fail(f"{label}: the decision set is empty; a "
                           f"pause no one can decide is a dead end")
            unknown = sorted(set(policy.decisions)
                             - {"approve", "deny"})
            if unknown:
                self._fail(f"{label}: unknown decision(s): "
                           f"{', '.join(unknown)} (handoff pauses "
                           f"decide approve or deny only)")
            if len(set(policy.decisions)) != len(policy.decisions):
                self._fail(f"{label}: duplicate decisions declared")
            timeout = policy.timeout_ms
            if timeout is not None and (
                    isinstance(timeout, bool)
                    or not isinstance(timeout, int) or timeout <= 0):
                self._fail(f"{label}: timeout_ms must be a positive "
                           f"integer or None, got {timeout!r}")

        if isinstance(self._max_handoffs, bool) \
                or not isinstance(self._max_handoffs, int) \
                or self._max_handoffs <= 0:
            self._fail(f"max_handoffs must be a positive integer, "
                       f"got {self._max_handoffs!r}")


class Team:
    """Entry point: Team.define("name", version="1.2.0")."""

    @staticmethod
    def define(name: str, version: str = "1.2.0") -> TeamBuilder:
        return TeamBuilder(name, version)
