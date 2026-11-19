"""Team orchestration (v1.2). Plain asserts:
    python3 harness/tests/test_team_orchestration.py

Golden scenarios for the Orchestrator (docs/ORCHESTRATION-DESIGN-
2026-10-31.md): a clean handoff chain whose team ledger shows every
effect with intent and outcome in order; an undeclared handoff
refused at build time (and a runtime request for one refused and
recorded); a double-written context key refused at build time; a
member that exhausts its budget sub-allocation stops while team
context stays consistent; a pause policy fires at its handoff and
the run resumes only on the approver's recorded decision (approve
resumes, deny does not); the union bound and grant attenuation are
build errors; delegation depth decrements per handoff and runs out.

All members are handler-based definitions over scripted/fake models:
synthetic content only, fully offline, no API spend.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from adapters.events.stores import InMemoryStore
from adapters.model.models import FnModel
from app.builder import Harness
from app.team_builder import Team
from core.errors import DefinitionError, HarnessError
from core.events import (
    HANDOFF_COMMITTED, HANDOFF_INTENDED, HANDOFF_REFUSED,
    MEMBER_FINISHED, MEMBER_STARTED, RUN_STARTED, TEAM_DECIDED,
    TEAM_FINISHED, TEAM_PAUSED, TEAM_STARTED,
)
from core.team import Authority
from usecases.team import sub_ledger_digest
from usecases.team_decisions import pending_team_pause, record_team_decision


class Clock:
    def __init__(self):
        self.t = 0

    def now_ms(self):
        self.t += 1
        return self.t


def dummy_model():
    return FnModel(lambda prompt, schema: {"event": "go", "data": {},
                                           "tool_calls": []})


def make_member(name, data):
    """A one-state handler member that emits ``data`` and finishes."""
    b = Harness.define(name).initial("work")
    b.state("work", handler=lambda ctx, _d=data: {
        "event": "go", "data": dict(_d), "tool_calls": []})
    b.state("done", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("work", "go", "done")
    b.escalation_state("esc").exhausted_state("exh")
    return b.build()


def make_looper(name):
    """A member whose only state loops forever; only a budget stops it."""
    def loop(ctx):
        return {"event": "again",
                "data": {"loop_result": "partial",
                         "loop_count": (ctx.get("loop_count") or 0) + 1},
                "tool_calls": []}

    b = Harness.define(name).initial("work")
    b.state("work", handler=loop)
    b.state("done", terminal=True)
    b.state("esc", terminal=True)
    b.state("exh", terminal=True)
    b.transition("work", "again", "work")
    # Declared so the graph is well-formed; the handler never emits
    # it, so only the budget ever stops this member.
    b.transition("work", "finish", "done")
    b.escalation_state("esc").exhausted_state("exh")
    b.budget(step_limit=100)  # the definition's own bound; the team's
    return b.build()          # sub-allocation is the operative cap


def expect_error(exc_type, fn, needle):
    try:
        fn()
    except exc_type as exc:
        assert needle in str(exc), f"{needle!r} not in: {exc}"
        return
    raise AssertionError(
        f"expected {exc_type.__name__} containing {needle!r}")


def types_of(events):
    return [e.type for e in events]


# --------------------------------------------------------------- scenarios

def test_clean_handoff_ledger_and_context_discipline():
    composer = Authority(
        capabilities={"read", "write", "search"}, budget_tokens=1000,
        delegation_depth=3,
        context_keys={"task", "brief", "draft", "internal_note"})
    researcher = make_member("researcher",
                             {"brief": "enriched synthetic brief"})
    writer = make_member("writer", {"draft": "synthetic draft v1"})
    grant = Authority(capabilities={"read"}, budget_tokens=300,
                      delegation_depth=0, context_keys={"task", "brief"})
    built = (Team.define("brief-team")
             .composer(composer)
             .entry("researcher")
             .member("researcher", researcher, writes=("brief",),
                     authority=Authority(
                         capabilities={"search", "read"},
                         budget_tokens=300, delegation_depth=1,
                         context_keys={"task", "brief"}))
             .member("writer", writer, writes=("draft",), terminal=True,
                     authority=Authority(
                         capabilities={"read", "write"},
                         budget_tokens=300, delegation_depth=0,
                         context_keys={"task", "brief", "draft"}))
             .handoff("researcher", "writer", reason="brief_ready",
                      grant=grant)
             .build())
    store = InMemoryStore()
    outcome = built.run(dummy_model(), store, Clock(),
                        initial_context={
                            "task": "summarize the synthetic widget "
                                    "market",
                            "brief": "seed brief",
                            "internal_note": "team-only, never carried"},
                        execution_id="team-1")
    assert outcome.success and outcome.status == "completed"
    # Declared writes merged; the team-only key stayed team-side.
    assert outcome.context["brief"] == "enriched synthetic brief"
    assert outcome.context["draft"] == "synthetic draft v1"
    assert outcome.context["internal_note"] == "team-only, never carried"

    events = store.read("team-1")
    assert types_of(events) == [
        TEAM_STARTED, MEMBER_STARTED, MEMBER_FINISHED,
        HANDOFF_INTENDED, HANDOFF_COMMITTED,
        MEMBER_STARTED, MEMBER_FINISHED, TEAM_FINISHED]
    intended = events[3]
    assert intended.payload["from_member"] == "researcher"
    assert intended.payload["to_member"] == "writer"
    assert intended.payload["reason"] == "brief_ready"
    assert intended.payload["context_digest"]
    assert intended.payload["grant"]["capabilities"] == ["read"]
    committed = events[4]
    assert committed.payload["context_digest"] == \
        intended.payload["context_digest"]

    # The writer's sub-ledger: digest in the team log matches the
    # member run it summarizes, and the writer received exactly the
    # grant's keys (the enriched brief, no team-only key, no draft).
    writer_events = store.read("team-1:writer:1")
    finished = events[6]
    assert finished.payload["sub_ledger_digest"] == \
        sub_ledger_digest(writer_events)
    run_started = next(e for e in writer_events if e.type == RUN_STARTED)
    assert run_started.payload["initial_context"] == {
        "task": "summarize the synthetic widget market",
        "brief": "enriched synthetic brief"}
    # The researcher likewise received only its declared keys.
    researcher_started = next(
        e for e in store.read("team-1:researcher:1")
        if e.type == RUN_STARTED)
    assert researcher_started.payload["initial_context"] == {
        "task": "summarize the synthetic widget market",
        "brief": "seed brief"}


def test_undeclared_handoff_is_a_build_error():
    a = make_member("a", {"x": 1})
    b = make_member("b", {"y": 2})

    def ghost_target():
        return (Team.define("t").entry("a")
                .member("a", a)
                .member("b", b, terminal=True)
                .handoff("a", "ghost")
                .build())

    def ghost_source():
        return (Team.define("t").entry("a")
                .member("a", a)
                .member("b", b, terminal=True)
                .handoff("ghost", "b")
                .handoff("a", "b")
                .build())

    expect_error(DefinitionError, ghost_target, "undeclared handoff")
    expect_error(DefinitionError, ghost_source, "undeclared handoff")


def test_double_written_context_key_is_a_build_error():
    a = make_member("a", {"draft": "from a"})
    b = make_member("b", {"draft": "from b"})

    def build():
        return (Team.define("t").entry("a")
                .member("a", a, writes=("draft",))
                .member("b", b, writes=("draft",), terminal=True)
                .handoff("a", "b")
                .build())

    expect_error(DefinitionError, build, "exactly one writer")


def test_member_budget_exhaustion_stops_team():
    composer = Authority(budget_tokens=1000, delegation_depth=2,
                         context_keys={"task", "loop_result"})
    looper = make_looper("looper")
    finisher = make_member("finisher", {"final_result": "finished"})
    built = (Team.define("loop-team")
             .composer(composer)
             .entry("looper")
             .member("looper", looper, writes=("loop_result",),
                     authority=Authority(budget_tokens=100,
                                         delegation_depth=1),
                     budget={"step_limit": 3})
             .member("finisher", finisher, writes=("final_result",),
                     terminal=True,
                     authority=Authority(budget_tokens=100,
                                         delegation_depth=0))
             .handoff("looper", "finisher")
             .build())
    store = InMemoryStore()
    outcome = built.run(dummy_model(), store, Clock(),
                        initial_context={"task": "loop until stopped"},
                        execution_id="team-loop")
    assert outcome.status == "budget_exhausted"
    assert outcome.success is False
    # The exhausted member's partial writes never reached team context.
    assert "loop_result" not in outcome.context
    events = store.read("team-loop")
    assert types_of(events) == [
        TEAM_STARTED, MEMBER_STARTED, MEMBER_FINISHED, TEAM_FINISHED]
    assert events[2].payload["outcome"] == "budget_exhausted"
    assert events[2].payload["member"] == "looper"
    assert events[3].payload["status"] == "budget_exhausted"
    # The finisher never started.
    assert not any(e.type == MEMBER_STARTED
                   and e.payload.get("member") == "finisher"
                   for e in events)


def _pause_team():
    composer = Authority(capabilities={"read"}, budget_tokens=500,
                         delegation_depth=2,
                         context_keys={"brief", "draft"})
    a = make_member("a", {"brief": "brief v1"})
    b = make_member("b", {"draft": "draft v1"})
    return (Team.define("pause-team")
            .composer(composer)
            .entry("a")
            .member("a", a, writes=("brief",),
                    authority=Authority(
                        capabilities={"read"}, budget_tokens=200,
                        delegation_depth=1, context_keys={"brief"}))
            .member("b", b, writes=("draft",), terminal=True,
                    authority=Authority(
                        capabilities={"read"}, budget_tokens=200,
                        delegation_depth=0, context_keys={"brief"}))
            .handoff("a", "b", reason="publish",
                     pause={"approver": "ops-lead"})
            .build())


def test_pause_at_handoff_resumes_only_on_approval():
    built = _pause_team()
    store = InMemoryStore()
    clock = Clock()
    outcome = built.run(dummy_model(), store, clock,
                        initial_context={"brief": "seed"},
                        execution_id="team-pause")
    assert outcome.awaiting_decision is True
    assert outcome.status == "paused"
    events = store.read("team-pause")
    assert events[-1].type == TEAM_PAUSED
    assert events[-1].payload["approver"] == "ops-lead"
    # The receiving member never started; the handoff never committed.
    assert not any(e.type == HANDOFF_COMMITTED for e in events)
    assert pending_team_pause(store, "team-pause") is not None

    # No decision, no resume.
    expect_error(HarnessError,
                 lambda: built.resume(dummy_model(), store, clock,
                                      execution_id="team-pause"),
                 "no decision, no resume")
    # A decision from anyone but the approver does not count.
    expect_error(HarnessError,
                 lambda: record_team_decision(
                     store, clock, "team-pause", decision="approve",
                     decided_by="someone-else"),
                 "awaits its approver")
    # The approver's recorded decision is the fact that resumes it.
    event = record_team_decision(store, clock, "team-pause",
                                 decision="approve",
                                 decided_by="ops-lead")
    assert event.type == TEAM_DECIDED
    outcome = built.resume(dummy_model(), store, clock,
                           execution_id="team-pause")
    assert outcome.success and outcome.status == "completed"
    assert outcome.context["draft"] == "draft v1"
    events = store.read("team-pause")
    order = types_of(events)
    assert order.index(TEAM_PAUSED) < order.index(TEAM_DECIDED) \
        < order.index(HANDOFF_COMMITTED)
    member_b = next(e for e in events if e.type == MEMBER_STARTED
                    and e.payload.get("member") == "b")
    assert order.index(HANDOFF_COMMITTED) < events.index(member_b)


def test_pause_at_handoff_deny_does_not_resume():
    built = _pause_team()
    store = InMemoryStore()
    clock = Clock()
    built.run(dummy_model(), store, clock,
              initial_context={"brief": "seed"},
              execution_id="team-deny")
    record_team_decision(store, clock, "team-deny", decision="deny",
                         decided_by="ops-lead")
    outcome = built.resume(dummy_model(), store, clock,
                           execution_id="team-deny")
    assert outcome.status == "denied" and outcome.success is False
    events = store.read("team-deny")
    assert events[-1].type == TEAM_FINISHED
    assert events[-1].payload["status"] == "denied"
    assert not any(e.type == MEMBER_STARTED
                   and e.payload.get("member") == "b" for e in events)
    assert "draft" not in outcome.context


def test_three_member_ledger_shows_every_effect_in_order():
    composer = Authority(capabilities={"read"}, budget_tokens=1000,
                         delegation_depth=3,
                         context_keys={"x", "y", "z"})
    a = make_member("a", {"x": "x1"})
    b = make_member("b", {"y": "y1"})
    c = make_member("c", {"z": "z1"})

    def auth(depth):
        return Authority(capabilities={"read"}, budget_tokens=100,
                         delegation_depth=depth,
                         context_keys={"x", "y", "z"})

    built = (Team.define("chain-team")
             .composer(composer)
             .entry("a")
             .member("a", a, writes=("x",), authority=auth(2))
             .member("b", b, writes=("y",), authority=auth(1))
             .member("c", c, writes=("z",), authority=auth(0),
                     terminal=True)
             .handoff("a", "b")
             .handoff("b", "c")
             .build())
    store = InMemoryStore()
    outcome = built.run(dummy_model(), store, Clock(),
                        initial_context={}, execution_id="team-chain")
    assert outcome.success
    assert outcome.context == {"x": "x1", "y": "y1", "z": "z1"}
    events = store.read("team-chain")
    assert types_of(events) == [
        TEAM_STARTED,
        MEMBER_STARTED, MEMBER_FINISHED, HANDOFF_INTENDED,
        HANDOFF_COMMITTED,
        MEMBER_STARTED, MEMBER_FINISHED, HANDOFF_INTENDED,
        HANDOFF_COMMITTED,
        MEMBER_STARTED, MEMBER_FINISHED,
        TEAM_FINISHED]
    assert [e.seq for e in events] == list(range(len(events)))
    # Every intent is followed by its outcome, naming the same pair.
    for i, event in enumerate(events):
        if event.type == HANDOFF_INTENDED:
            committed = events[i + 1]
            assert committed.type == HANDOFF_COMMITTED
            assert committed.payload["from_member"] == \
                event.payload["from_member"]
            assert committed.payload["to_member"] == \
                event.payload["to_member"]
    finished = events[-1]
    assert finished.payload["status"] == "completed"
    assert finished.payload["handoffs"] == 2
    assert finished.payload["members_run"] == ["a", "b", "c"]
    assert set(finished.payload["spend"]) == {"steps", "tokens",
                                              "tool_calls"}


def test_runtime_request_for_undeclared_handoff_is_refused():
    composer = Authority(capabilities={"read"}, budget_tokens=1000,
                         delegation_depth=3, context_keys={"task"})
    a = make_member("a", {"next_member": "c"})
    b = make_member("b", {"y": 1})
    c = make_member("c", {"z": 1})

    def auth(depth):
        return Authority(capabilities={"read"}, budget_tokens=100,
                         delegation_depth=depth, context_keys={"task"})

    built = (Team.define("request-team")
             .composer(composer)
             .entry("a")
             .member("a", a, authority=auth(2))
             .member("b", b, authority=auth(1))
             .member("c", c, authority=auth(0), terminal=True)
             .handoff("a", "b")
             .handoff("b", "c")
             .build())
    store = InMemoryStore()
    outcome = built.run(dummy_model(), store, Clock(),
                        initial_context={"task": "t"},
                        execution_id="team-request")
    assert outcome.status == "refused" and outcome.success is False
    events = store.read("team-request")
    refused = next(e for e in events if e.type == HANDOFF_REFUSED)
    assert refused.payload["from_member"] == "a"
    assert refused.payload["to_member"] == "c"
    assert refused.payload["reason"] == "undeclared handoff"
    # Neither the requested member nor the declared successor ran.
    assert not any(e.type == MEMBER_STARTED
                   and e.payload.get("member") in ("b", "c")
                   for e in events)


def test_union_bound_is_a_build_error():
    a = make_member("a", {"x": 1})
    b = make_member("b", {"y": 2})

    def over_capable():
        return (Team.define("t")
                .composer(Authority(capabilities={"read"},
                                    budget_tokens=1000,
                                    delegation_depth=2))
                .entry("a")
                .member("a", a, authority=Authority(
                    capabilities={"read", "write"}, budget_tokens=100,
                    delegation_depth=1))
                .member("b", b, terminal=True, authority=Authority(
                    capabilities={"read"}, budget_tokens=100,
                    delegation_depth=0))
                .handoff("a", "b")
                .build())

    def over_budget():
        return (Team.define("t")
                .composer(Authority(capabilities={"read"},
                                    budget_tokens=150,
                                    delegation_depth=2))
                .entry("a")
                .member("a", a, authority=Authority(
                    capabilities={"read"}, budget_tokens=100,
                    delegation_depth=1))
                .member("b", b, terminal=True, authority=Authority(
                    capabilities={"read"}, budget_tokens=100,
                    delegation_depth=0))
                .handoff("a", "b")
                .build())

    expect_error(DefinitionError, over_capable, "exceeds the composer's")
    expect_error(DefinitionError, over_budget, "fan-out divides")


def test_amplifying_grant_is_a_build_error_attenuated_builds():
    composer = Authority(capabilities={"read", "write"},
                         budget_tokens=1000, delegation_depth=2,
                         context_keys={"task"})
    a = make_member("a", {"x": 1})
    b = make_member("b", {"y": 2})
    sender = Authority(capabilities={"read"}, budget_tokens=200,
                       delegation_depth=1, context_keys={"task"})
    receiver = Authority(capabilities={"read", "write"},
                         budget_tokens=200, delegation_depth=0,
                         context_keys={"task"})

    def no_grant():
        return (Team.define("t").composer(composer).entry("a")
                .member("a", a, authority=sender)
                .member("b", b, terminal=True, authority=receiver)
                .handoff("a", "b")
                .build())

    expect_error(DefinitionError, no_grant, "amplifies authority")

    attenuated = Authority(capabilities={"read"}, budget_tokens=200,
                           delegation_depth=0, context_keys={"task"})
    built = (Team.define("t").composer(composer).entry("a")
             .member("a", a, authority=sender)
             .member("b", b, terminal=True, authority=receiver)
             .handoff("a", "b", grant=attenuated)
             .build())
    store = InMemoryStore()
    outcome = built.run(dummy_model(), store, Clock(),
                        initial_context={"task": "t"},
                        execution_id="team-grant")
    assert outcome.success


def test_delegation_depth_decrements_and_runs_out():
    composer = Authority(capabilities={"read"}, budget_tokens=1000,
                         delegation_depth=2, context_keys={"task"})
    a = make_member("a", {"x": 1})
    b = make_member("b", {"y": 2})
    c = make_member("c", {"z": 3})

    def auth(depth):
        return Authority(capabilities={"read"}, budget_tokens=100,
                         delegation_depth=depth, context_keys={"task"})

    built = (Team.define("depth-team")
             .composer(composer)
             .entry("a")
             .member("a", a, authority=auth(1))
             .member("b", b, authority=auth(1))
             .member("c", c, authority=auth(0), terminal=True)
             .handoff("a", "b")
             .handoff("b", "c")
             .build())
    store = InMemoryStore()
    outcome = built.run(dummy_model(), store, Clock(),
                        initial_context={"task": "t"},
                        execution_id="team-depth")
    assert outcome.status == "refused"
    events = store.read("team-depth")
    refused = next(e for e in events if e.type == HANDOFF_REFUSED)
    assert refused.payload["reason"] == "delegation depth exhausted"
    assert refused.payload["from_member"] == "b"
    # a and b ran; the depth-exhausted handoff never started c.
    started = [e.payload["member"] for e in events
               if e.type == MEMBER_STARTED]
    assert started == ["a", "b"]


TESTS = [
    test_clean_handoff_ledger_and_context_discipline,
    test_undeclared_handoff_is_a_build_error,
    test_double_written_context_key_is_a_build_error,
    test_member_budget_exhaustion_stops_team,
    test_pause_at_handoff_resumes_only_on_approval,
    test_pause_at_handoff_deny_does_not_resume,
    test_three_member_ledger_shows_every_effect_in_order,
    test_runtime_request_for_undeclared_handoff_is_refused,
    test_union_bound_is_a_build_error,
    test_amplifying_grant_is_a_build_error_attenuated_builds,
    test_delegation_depth_decrements_and_runs_out,
]


def main():
    for test in TESTS:
        test()
        print(f"ok: {test.__name__}")
    print(f"team orchestration: {len(TESTS)}/{len(TESTS)} scenarios")


if __name__ == "__main__":
    main()
