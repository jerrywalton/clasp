"""Team-scale entities: roster, handoffs, authority, budget envelope.

Pure data and pure checks, like the rest of core. The team layer is
CLASP's orchestration construct (docs/ORCHESTRATION-DESIGN-2026-10-31.md):
a Team is a definition — declared roster, declared handoffs, exactly
one writer per shared context key, one budget envelope divided into
hard member sub-allocations — validated at build time in
app/team_builder.py and executed by the Orchestrator in
usecases/team.py. Nothing here knows how a member runs internally;
the team depends on member ports, never on member internals.

Authority is the delegation invariant as data (GLOSSARY.md, adopted
Nov 3, 2026): a hand-off may carry only a subset of the sender's
authority across capabilities, budget, time, delegation depth, and
context, and a composer may not assemble a team whose combined
authority exceeds the composer's own (the union bound).

Declaration convention, fail-closed: on the limit axes (budget,
window), None means no limit was declared at that layer; a limit the
parent declares binds the child, so a child that declares no limit
under a limited parent is amplification by silence and fails the
check. On the set axes (capabilities, context keys), the empty set
means the holder holds none — a plain subset test applies. Delegation
depth is a plain count: a child's depth may not exceed its parent's,
and the Orchestrator decrements the carried depth at every handoff,
so authority a member received can be passed onward only finitely
many times.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# The team-context key a member's outcome may use to request its next
# handoff target. A request is honored only when the named member is
# a declared successor; anything else is refused and recorded. The
# key is control-plane, not data: it is never merged into team
# context as a declared write.
NEXT_MEMBER_KEY = "next_member"


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) \
        and value > 0


def _is_non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) \
        and value >= 0


@dataclass(frozen=True)
class Authority:
    """A declared grant of authority on the five invariant axes."""

    capabilities: frozenset = frozenset()
    budget_tokens: int | None = None
    window_ms: int | None = None
    delegation_depth: int = 0
    context_keys: frozenset = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(self, "capabilities",
                           frozenset(self.capabilities or ()))
        object.__setattr__(self, "context_keys",
                           frozenset(self.context_keys or ()))
        for name in self.capabilities | self.context_keys:
            if not isinstance(name, str) or not name:
                raise ValueError(
                    "authority capabilities and context keys must be "
                    "non-empty strings")
        for label, value in (("budget_tokens", self.budget_tokens),
                             ("window_ms", self.window_ms)):
            if value is not None and not _is_positive_int(value):
                raise ValueError(
                    f"authority {label} must be a positive integer "
                    f"or None, got {value!r}")
        if not _is_non_negative_int(self.delegation_depth):
            raise ValueError(
                f"authority delegation_depth must be a non-negative "
                f"integer, got {self.delegation_depth!r}")

    def summary(self) -> dict:
        """The authority as a log payload: sorted, JSON-shaped."""
        return {
            "capabilities": sorted(self.capabilities),
            "budget_tokens": self.budget_tokens,
            "window_ms": self.window_ms,
            "delegation_depth": self.delegation_depth,
            "context_keys": sorted(self.context_keys),
        }


def authority_violations(child: Authority, parent: Authority, *,
                         child_label: str = "child",
                         parent_label: str = "parent") -> list[str]:
    """Why ``child`` is not a subset of ``parent`` (empty = subset).

    The delegation invariant, axis by axis: capabilities and context
    keys are subset tests; budget and window may not exceed the
    parent's declared limit, and declaring no limit under a limited
    parent is itself a violation; delegation depth may not exceed
    the parent's."""
    problems: list[str] = []
    extra_caps = sorted(child.capabilities - parent.capabilities)
    if extra_caps:
        problems.append(
            f"{child_label} holds capabilities the {parent_label} "
            f"does not: {', '.join(extra_caps)}")
    extra_keys = sorted(child.context_keys - parent.context_keys)
    if extra_keys:
        problems.append(
            f"{child_label} may receive context keys the "
            f"{parent_label} may not carry: {', '.join(extra_keys)}")
    for label, child_value, parent_value in (
            ("budget_tokens", child.budget_tokens, parent.budget_tokens),
            ("window_ms", child.window_ms, parent.window_ms)):
        if parent_value is None:
            continue
        if child_value is None:
            problems.append(
                f"{child_label} declares no {label} limit under a "
                f"{parent_label} limited to {parent_value}: an "
                f"undeclared limit is amplification by silence")
        elif child_value > parent_value:
            problems.append(
                f"{child_label} {label} {child_value} exceeds the "
                f"{parent_label} limit {parent_value}")
    if child.delegation_depth > parent.delegation_depth:
        problems.append(
            f"{child_label} delegation depth {child.delegation_depth} "
            f"exceeds the {parent_label} depth "
            f"{parent.delegation_depth}")
    return problems


@dataclass(frozen=True)
class HandoffPausePolicy:
    """A declared human-in-the-loop pause on one handoff.

    The team-scale analogue of core.state.PausePolicy: the trigger
    is the handoff, the approver a named role, the allowed decisions
    a subset of approve/deny/amend, the channel transport only. The
    invariants do not vary: the awaiting handoff is a log event
    (TEAM_PAUSED), the decision is a log event (TEAM_DECIDED), and
    no decision means no resume."""
    approver: str
    channel: str = "cli"
    decisions: tuple[str, ...] = ("approve", "deny")
    timeout_ms: int | None = None


@dataclass
class MemberDef:
    """One roster entry: a member's team-scale declaration.

    ``writes`` are the team context keys this member solely writes;
    ``budget_kwargs`` is its hard sub-allocation of the team envelope
    (the Budget entity's kwargs, core.budget); ``authority`` is the
    authority the composer grants it. The member's internal
    definition is not part of this declaration."""
    name: str
    writes: tuple[str, ...] = ()
    authority: Authority = field(default_factory=Authority)
    budget_kwargs: dict = field(default_factory=dict)
    terminal: bool = False


@dataclass
class HandoffDef:
    """A declared transition from one member to another.

    ``guard`` is a pure predicate over team context, like a
    transition guard inside a run. ``grant`` is the authority the
    handoff carries to the receiver; None means the receiver's own
    composed authority, checked against the sender's either way.
    ``pause`` attaches a HandoffPausePolicy to this boundary."""
    from_member: str
    to_member: str
    reason: str = "next"
    guard: Callable[[dict], bool] | None = None
    grant: Authority | None = None
    pause: HandoffPausePolicy | None = None


@dataclass
class TeamDef:
    """A validated team definition (built by app/team_builder.py)."""
    name: str
    version: str
    members: dict[str, MemberDef]
    handoffs: list[HandoffDef]
    entry: str
    composer: Authority = field(default_factory=Authority)
    envelope_kwargs: dict = field(default_factory=dict)
    max_handoffs: int = 32

    def member(self, name: str) -> MemberDef:
        return self.members[name]

    def successors(self, name: str) -> list[HandoffDef]:
        """The declared handoffs out of ``name``, in declaration order."""
        return [h for h in self.handoffs if h.from_member == name]


def composition_violations(team: TeamDef) -> list[str]:
    """The union bound: why this team, as composed, exceeds the
    authority its composer holds (empty = within bounds).

    Every member's authority must be a subset of the composer's, and
    the budget must compose: the members' token sub-allocations sum
    to no more than the composer's token budget — fan-out divides
    the composer's budget, never multiplies it."""
    problems: list[str] = []
    for name in sorted(team.members):
        member = team.members[name]
        for problem in authority_violations(
                member.authority, team.composer,
                child_label=f"member '{name}'",
                parent_label="composer"):
            problems.append(problem)
    if team.composer.budget_tokens is not None:
        total = sum(m.authority.budget_tokens or 0
                    for m in team.members.values())
        if total > team.composer.budget_tokens:
            problems.append(
                f"team budget composition: member budget_tokens "
                f"sub-allocations sum to {total}, exceeding the "
                f"composer budget {team.composer.budget_tokens} "
                f"(fan-out divides, never multiplies)")
    return problems


def handoff_violations(handoff: HandoffDef, sender: MemberDef,
                       receiver: MemberDef) -> list[str]:
    """Why this handoff would amplify authority (empty = attenuated).

    The effective grant — the declared grant, or the receiver's
    composed authority when none is declared — must be a subset of
    the sender's authority: a hand-off carries only what the sender
    holds. A declared grant must also fit inside the receiver's own
    composed authority: a handoff cannot hand a member more than
    the composer gave it."""
    effective = handoff.grant if handoff.grant is not None \
        else receiver.authority
    problems: list[str] = []
    for problem in authority_violations(
            effective, sender.authority,
            child_label=f"handoff '{sender.name}' -> '{receiver.name}' "
                        f"grant",
            parent_label=f"sender '{sender.name}'"):
        problems.append(problem)
    if handoff.grant is not None:
        for problem in authority_violations(
                handoff.grant, receiver.authority,
                child_label=f"handoff '{sender.name}' -> "
                            f"'{receiver.name}' grant",
                parent_label=f"receiver '{receiver.name}'"):
            problems.append(problem)
    return problems
