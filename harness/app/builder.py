"""Composition root and build-time validation.

Harness.define(...) builds a validated Definition plus its tool registry,
gates, and budget template. Validation implements docs/HARNESS-ARCHITECTURE.md
section 13: if it builds, the graph is well-formed. Malformed definitions
raise DefinitionError here, never at runtime.

Composition also owns logging transport: a persistent event store
(SQLite/JSONL) handed to runner()/resume() is wrapped in the async
logging adapter by default, so event persistence runs on a dedicated
logger thread, decoupled from run execution. Pass
.async_logging(False) at define time for the raw synchronous path.
BuiltHarness.close() (or a with-block) closes every logger it created.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from adapters.events.async_store import AsyncEventStore
from adapters.events.stores import JsonlStore, SqliteStore
from adapters.gates.checks import SchemaGate
from adapters.tools.registry import ToolRegistry
from core.budget import Budget
from core.errors import DefinitionError
from core.ledger import PostingRule
from core.state import PausePolicy, StateDef, Transition
from core.validate import validate_schema
from usecases.ports import ClockPort, EventStorePort, ModelPort
from usecases.runner import Definition, Outcome, StateMachineRunner

# The only keyword arguments .budget() accepts — the Budget entity's
# two limits (core.budget.Budget.from_kwargs passes kwargs through to
# the Budget constructor, so an unknown key used to surface as a
# TypeError at runner() time instead of a build error; critique P12).
_BUDGET_KWARGS = frozenset({"step_limit", "token_limit",
                            "tool_call_limit", "correction_limit"})


@dataclass
class BuiltHarness:
    definition: Definition
    tools: ToolRegistry
    gates: dict
    budget_kwargs: dict = field(default_factory=dict)
    async_logging: bool = True
    _loggers: list = field(default_factory=list)

    def _prepare_events(self, events: EventStorePort) -> EventStorePort:
        """Wrap a persistent store in the async logging adapter.

        One wrapper per inner store instance: re-wrapping the same
        store would split its event stream across two logger threads
        and forfeit write ordering. In-memory stores stay synchronous
        (tests use them directly), as does an already-async store."""
        if not self.async_logging:
            return events
        if not isinstance(events, (SqliteStore, JsonlStore)):
            return events
        for wrapper in self._loggers:
            if wrapper.inner is events and not wrapper.closed:
                return wrapper
        wrapper = AsyncEventStore(events)
        self._loggers.append(wrapper)
        return wrapper

    def runner(self, model: ModelPort, events: EventStorePort,
               clock: ClockPort,
               correction_model: ModelPort | None = None
               ) -> StateMachineRunner:
        return StateMachineRunner(
            self.definition, model=model, tools=self.tools,
            gates=self.gates, events=self._prepare_events(events),
            clock=clock,
            budget=Budget.from_kwargs(**self.budget_kwargs),
            correction_model=correction_model)

    def resume(self, model: ModelPort, events: EventStorePort,
               clock: ClockPort, execution_id: str,
               initial_context: dict | None = None,
               correction_model: ModelPort | None = None) -> Outcome:
        """Resume an interrupted run of this definition from the event
        store (e.g. a SqliteStore that survived the process). Same
        composition as runner(); the runner folds the persisted log and
        continues the run under the same execution_id."""
        return self.runner(model, events, clock,
                           correction_model=correction_model).resume(
            initial_context, execution_id=execution_id)

    def close(self) -> None:
        """Close every async logger this harness created (flush +
        stop its thread). Every logger is closed even if one raises;
        the first error propagates afterwards. Idempotent."""
        first_error: Exception | None = None
        for wrapper in self._loggers:
            try:
                wrapper.close()
            except Exception as exc:  # noqa: BLE001 - close them all
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    def __enter__(self) -> "BuiltHarness":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class HarnessBuilder:
    def __init__(self, name: str, version: str = "1.0.0") -> None:
        self._name = name
        self._version = version
        self._states: dict[str, StateDef] = {}
        self._transitions: list[Transition] = []
        self._initial: str | None = None
        self._escalation: str | None = None
        self._exhausted: str | None = None
        self._tools = ToolRegistry()
        self._tool_schemas: dict = {}
        self._gates: dict = {}
        self._budget_kwargs: dict = {}
        self._checkpoint_kwargs: dict = {
            "every_steps": 0, "every_bytes": 0, "every_ms": 0}
        self._pause_policies: dict = {}
        self._async_logging = True
        self._accounts: list = []
        self._posting_rules: list = []
        self._validated_posting_rules: tuple = ()
        self._redacted_context_keys: list = []
        self._frozen_context_keys: list = []

    # ------------------------------------------------------------ fluent API
    def initial(self, state_name: str) -> "HarnessBuilder":
        self._initial = state_name
        return self

    def state(self, name: str, *, terminal: bool = False,
              prompt_template: str | None = None,
              output_schema: dict | None = None,
              handler=None, tools: tuple[str, ...] = (),
              entry_gates: tuple[str, ...] = (),
              exit_gates: tuple[str, ...] = (),
              max_corrections: int = 2,
              writes: tuple[str, ...] = ()) -> "HarnessBuilder":
        if name in self._states:
            # Redefinition used to silently overwrite the first
            # definition (critique P14); it is a build error now.
            raise DefinitionError(f"state '{name}' is already defined")
        self._states[name] = StateDef(
            name=name, terminal=terminal, prompt_template=prompt_template,
            output_schema=output_schema, handler=handler, tools=tuple(tools),
            entry_gates=tuple(entry_gates), exit_gates=tuple(exit_gates),
            max_corrections=max_corrections, writes=tuple(writes))
        return self

    def transition(self, from_state: str, event: str, to_state: str,
                   guard=None) -> "HarnessBuilder":
        self._transitions.append(Transition(from_state, event, to_state, guard))
        return self

    def escalation_state(self, name: str) -> "HarnessBuilder":
        self._escalation = name
        return self

    def exhausted_state(self, name: str) -> "HarnessBuilder":
        self._exhausted = name
        return self

    def register_tool(self, name: str, handler,
                      legal_in, args_schema: dict | None = None
                      ) -> "HarnessBuilder":
        """Register a tool. ``args_schema`` declares the tool's
        argument schema in the definition (the one declaration home
        for tool contracts): every tool a state may invoke must
        declare one at build time — a no-argument tool declares the
        empty object schema {"type": "object"}. The runner validates
        every call's args against it before invocation."""
        self._tools.register(name, handler, set(legal_in))
        if args_schema is not None:
            self._tool_schemas[name] = args_schema
        return self

    def accounts(self, *names: str) -> "HarnessBuilder":
        """Declare the chart of accounts ledger postings may use.
        Duplicate or malformed declarations fail at build time."""
        self._accounts.extend(names)
        return self

    # Familiar long name for the same declaration.
    chart_of_accounts = accounts

    def redacted_context_keys(self, *keys: str) -> "HarnessBuilder":
        """Declare context keys whose values are hash-only in event
        payloads. RUN_STARTED records the declaration; recreation
        treats those values as opaque (see runner documentation)."""
        self._redacted_context_keys.extend(keys)
        return self

    def frozen_context_keys(self, *keys: str) -> "HarnessBuilder":
        """Declare context keys frozen once set (#11): any state may
        set one while it is unset; no state may change its value
        afterwards. The check runs on the envelope before any tool
        fires, so a violating attempt has no effects."""
        self._frozen_context_keys.extend(keys)
        return self

    def register_ledger_tool(self, name: str, handler, legal_in, *,
                             debit_account: str, credit_account: str,
                             amount_arg: str = "amount_cents",
                             reference_arg: str = "order_id",
                             args_schema: dict | None = None
                             ) -> "HarnessBuilder":
        """Register a tool whose effect is a double-entry posting.

        The posting accounts are validated against the declared chart
        at build time (an undeclared account is a DefinitionError
        there, not here), so declaration order does not matter.
        ``args_schema`` is the tool's argument schema, as in
        register_tool; ledger tools must declare one too."""
        self._tools.register(name, handler, set(legal_in))
        if args_schema is not None:
            self._tool_schemas[name] = args_schema
        self._posting_rules.append(
            (name, debit_account, credit_account, amount_arg,
             reference_arg))
        return self

    def ledger_posting(self, tool: str, *, debit_account: str,
                       credit_account: str,
                       amount_arg: str = "amount_cents",
                       reference_arg: str = "order_id") -> "HarnessBuilder":
        """Attach a posting rule to an already-registered tool."""
        self._posting_rules.append(
            (tool, debit_account, credit_account, amount_arg,
             reference_arg))
        return self

    def register_gate(self, name: str, gate) -> "HarnessBuilder":
        self._gates[name] = gate
        return self

    def budget(self, **kwargs) -> "HarnessBuilder":
        unknown = sorted(set(kwargs) - _BUDGET_KWARGS)
        if unknown:
            raise DefinitionError(
                f"unknown budget argument(s): {', '.join(unknown)} "
                f"(supported: {', '.join(sorted(_BUDGET_KWARGS))})")
        self._budget_kwargs.update(kwargs)
        return self

    def checkpoint(self, *, every_steps: int = 0, every_bytes: int = 0,
                   every_ms: int = 0) -> "HarnessBuilder":
        """Declare the checkpoint policy (the backup schedule): a
        CONTEXT_CHECKPOINT with the folded context is written at a
        commit boundary when any trigger fires. Triggers are
        non-negative integers; 0 disables that trigger. Values are
        validated in build(), like everything else."""
        self._checkpoint_kwargs = {
            "every_steps": every_steps, "every_bytes": every_bytes,
            "every_ms": every_ms}
        return self

    def pause(self, state: str, *, approver: str, channel: str = "cli",
              decisions: tuple[str, ...] = ("approve", "deny"),
              timeout_ms: int | None = None) -> "HarnessBuilder":
        """Declare a human-in-the-loop pause on a state (#10).

        The policy is configurable because deployments differ: the
        trigger is the state, the approver a named role, the allowed
        decisions a subset of approve/deny/amend, the channel the
        transport the deployment renders the request on, and the
        timeout fail-closed. What is NOT configurable is validated
        around it: an empty approver, an empty or unknown decision
        set, or a non-positive timeout fails the build. Invariants
        never vary: the awaiting state is a log event, the decision
        is a log event, and no decision means no resume."""
        self._pause_policies[state] = {
            "approver": approver, "channel": channel,
            "decisions": tuple(decisions), "timeout_ms": timeout_ms}
        return self

    def async_logging(self, enabled: bool = True) -> "HarnessBuilder":
        """Async logging composition (default on): persistent event
        stores are wrapped in the async adapter so persistence runs on
        a logger thread, decoupled from execution. Disable for the raw
        synchronous store path."""
        self._async_logging = bool(enabled)
        return self

    # ----------------------------------------------------------------- build
    def build(self) -> BuiltHarness:
        self._validate()
        definition = Definition(
            name=self._name, version=self._version, states=self._states,
            transitions=self._transitions, initial=self._initial,
            escalation_state=self._escalation,
            exhausted_state=self._exhausted,
            accounts=tuple(self._accounts),
            ledger_postings=self._validated_posting_rules,
            redacted_context_keys=tuple(self._redacted_context_keys),
            frozen_context_keys=tuple(self._frozen_context_keys),
            tool_schemas=dict(self._tool_schemas),
            checkpoint_every_steps=self._checkpoint_kwargs["every_steps"],
            checkpoint_every_bytes=self._checkpoint_kwargs["every_bytes"],
            checkpoint_every_ms=self._checkpoint_kwargs["every_ms"],
            pause_policies={
                state: PausePolicy(state=state, **raw)
                for state, raw in self._pause_policies.items()})
        return BuiltHarness(definition=definition, tools=self._tools,
                            gates=self._gates,
                            budget_kwargs=dict(self._budget_kwargs),
                            async_logging=self._async_logging)

    # ------------------------------------------------------------ validation
    def _fail(self, msg: str) -> None:
        raise DefinitionError(msg)

    def _validate(self) -> None:
        if not self._initial:
            self._fail("no initial state declared")
        if self._initial not in self._states:
            self._fail(f"initial state '{self._initial}' is not defined")
        if self._states[self._initial].terminal:
            self._fail("initial state must not be terminal")

        for role, name in (("escalation", self._escalation),
                           ("exhausted", self._exhausted)):
            if not name:
                self._fail(f"no {role} state declared")
            if name not in self._states:
                self._fail(f"{role} state '{name}' is not defined")
            if not self._states[name].terminal:
                self._fail(f"{role} state '{name}' must be terminal")

        for t in self._transitions:
            for endpoint in (t.from_state, t.to_state):
                if endpoint not in self._states:
                    self._fail(f"transition references undefined state "
                               f"'{endpoint}'")

        outbound: dict[str, int] = {name: 0 for name in self._states}
        for t in self._transitions:
            outbound[t.from_state] += 1
        for name, state in self._states.items():
            if state.terminal and outbound[name]:
                self._fail(f"terminal state '{name}' has outbound transitions")
            if (not state.terminal and not outbound[name]
                    and name not in (self._escalation, self._exhausted)):
                self._fail(f"non-terminal state '{name}' is a dead end: "
                           f"no outbound transitions")

        # Reachability from initial (terminals reached only by jump are exempt).
        seen = set()
        stack = [self._initial]
        adjacency: dict[str, list[str]] = {n: [] for n in self._states}
        for t in self._transitions:
            adjacency[t.from_state].append(t.to_state)
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            stack.extend(adjacency[node])
        exempt = {self._escalation, self._exhausted}
        for name in self._states:
            if name not in seen and name not in exempt:
                self._fail(f"state '{name}' is unreachable from initial")

        # Work-source rules.
        for name, state in self._states.items():
            if state.terminal:
                continue
            has_prompt = state.prompt_template is not None
            has_handler = state.handler is not None
            if has_prompt and has_handler:
                self._fail(f"state '{name}': prompt_template and handler are "
                           f"mutually exclusive")
            if has_prompt and not state.output_schema:
                self._fail(f"state '{name}': prompt states require "
                           f"output_schema")

        # Output schemas must be well-formed, unions included: a type
        # list with an unknown member or an empty anyOf is a definition
        # error here, not a surprise at runtime.
        for name, state in self._states.items():
            if state.output_schema is not None:
                problems = validate_schema(state.output_schema)
                if problems:
                    self._fail(f"state '{name}': invalid output_schema: "
                               + "; ".join(problems))

        # Tools and gates resolve. Capability has exactly ONE source
        # of truth: the states' tools declarations. The registry's
        # legal_in is cross-checked against it in both directions
        # (critique P2a/P2b, design doc section 13); a disagreement
        # is a build error, never a mid-flight registry refusal or a
        # silently unreachable tool.
        registered = set(self._tools._handlers)
        for name, state in self._states.items():
            for tool in state.tools:
                if tool not in registered:
                    self._fail(f"state '{name}' uses unregistered tool '{tool}'")
        declared_in: dict[str, set[str]] = {tool: set()
                                            for tool in registered}
        for name, state in self._states.items():
            for tool in state.tools:
                declared_in[tool].add(name)
        for tool in sorted(registered):
            legal = set(self._tools._legal_in[tool])
            for state_name in sorted(legal - set(self._states)):
                self._fail(f"tool '{tool}' is registered legal in "
                           f"undefined state '{state_name}'")
            declared = declared_in[tool]
            if not declared and not legal:
                self._fail(f"tool '{tool}' is registered but declared "
                           f"in no state and legal in no state")
            for state_name in sorted(legal - declared):
                self._fail(f"tool '{tool}' is registered legal in state "
                           f"'{state_name}', which does not declare it: "
                           f"capability is declared in the states' tools, "
                           f"not the registry")
            for state_name in sorted(declared - legal):
                self._fail(f"tool '{tool}' is declared in state "
                           f"'{state_name}' but not registered legal "
                           f"there: the two capability declarations "
                           f"disagree")
            # Derive the registry's legality from the definition, so
            # the states' declaration is what the registry enforces.
            self._tools._legal_in[tool] = set(declared)

        # Tool argument schemas (critique P3): every tool a state may
        # invoke must declare one in the definition, and the schemas
        # themselves must be well-formed. Effects are guarded by with
        # what arguments, not only by which tool.
        for tool in sorted(self._tool_schemas):
            if tool not in registered:
                self._fail(f"argument schema declared for unregistered "
                           f"tool '{tool}'")
            problems = validate_schema(self._tool_schemas[tool])
            if problems:
                self._fail(f"tool '{tool}': invalid argument schema: "
                           + "; ".join(problems))
        for name, state in self._states.items():
            for tool in state.tools:
                if tool not in self._tool_schemas:
                    self._fail(f"state '{name}' uses tool '{tool}' with "
                               f"no declared argument schema: declare "
                               f"one at registration (a no-argument tool "
                               f"declares the empty object schema)")
        referenced_gates = set()
        for state in self._states.values():
            referenced_gates.update(state.entry_gates)
            referenced_gates.update(state.exit_gates)
        for gate in referenced_gates:
            if gate not in self._gates:
                self._fail(f"referenced gate '{gate}' is not registered")

        # Gate schemas are definition material too (critique P13): a
        # SchemaGate carrying a malformed schema fails the build
        # here, instead of failing closed at runtime and escalating
        # a healthy run for a definition error.
        for gate_name in sorted(self._gates):
            gate = self._gates[gate_name]
            if isinstance(gate, SchemaGate):
                problems = validate_schema(gate.schema)
                if problems:
                    self._fail(f"gate '{gate_name}': invalid schema: "
                               + "; ".join(problems))

        # Chart of accounts + ledger posting rules. Accounts are a
        # closed declaration: duplicates are rejected, and every
        # account a posting rule names must be declared.
        account_names: set[str] = set()
        for account in self._accounts:
            if not isinstance(account, str) or not account:
                self._fail("account names must be non-empty strings")
            if account in account_names:
                self._fail(f"duplicate account declaration '{account}'")
            account_names.add(account)
        redacted: set[str] = set()
        for key in self._redacted_context_keys:
            if not isinstance(key, str) or not key:
                self._fail("redacted context keys must be non-empty strings")
            if key in redacted:
                self._fail(f"duplicate redacted context key '{key}'")
            redacted.add(key)
        frozen: set[str] = set()
        for key in self._frozen_context_keys:
            if not isinstance(key, str) or not key:
                self._fail("frozen context keys must be non-empty strings")
            if key in frozen:
                self._fail(f"duplicate frozen context key '{key}'")
            frozen.add(key)
        for name, state in self._states.items():
            for key in state.writes:
                if not isinstance(key, str) or not key:
                    self._fail(f"state '{name}': declared writes must "
                               f"be non-empty strings")
            if len(set(state.writes)) != len(state.writes):
                self._fail(f"state '{name}': duplicate declared writes")
        posting_rules: list[PostingRule] = []
        posting_tools: set[str] = set()
        for raw in self._posting_rules:
            try:
                rule = PostingRule(tool=raw[0], debit_account=raw[1],
                                   credit_account=raw[2],
                                   amount_arg=raw[3], reference_arg=raw[4])
            except ValueError as exc:
                self._fail(f"invalid ledger posting rule: {exc}")
            if rule.tool in posting_tools:
                self._fail(f"duplicate ledger posting rule for tool "
                           f"'{rule.tool}'")
            posting_tools.add(rule.tool)
            if rule.tool not in registered:
                self._fail(f"ledger posting rule references unregistered "
                           f"tool '{rule.tool}'")
            for account in (rule.debit_account, rule.credit_account):
                if account not in account_names:
                    self._fail(f"ledger posting for tool '{rule.tool}' "
                               f"references undeclared account '{account}'")
            posting_rules.append(rule)
        self._validated_posting_rules = tuple(posting_rules)

        # Pause policies are definition material (the configurable
        # half of #10): the state must exist, the approver must be
        # named, the decision set must be a non-empty subset of the
        # known decisions, the channel must be named, and a timeout
        # must be positive. IT departments configure options; they
        # do not get to configure away the invariants.
        for state_name, raw in self._pause_policies.items():
            if state_name not in self._states:
                self._fail(f"pause policy references undefined state "
                           f"'{state_name}'")
            if self._states[state_name].terminal:
                self._fail(f"pause policy on terminal state "
                           f"'{state_name}': terminal states do no "
                           f"work and never pause")
            if not isinstance(raw["approver"], str) \
                    or not raw["approver"]:
                self._fail(f"pause policy for state '{state_name}': "
                           f"approver must be a non-empty string")
            if not isinstance(raw["channel"], str) \
                    or not raw["channel"]:
                self._fail(f"pause policy for state '{state_name}': "
                           f"channel must be a non-empty string")
            decisions = raw["decisions"]
            if not decisions:
                self._fail(f"pause policy for state '{state_name}': "
                           f"the decision set is empty; a pause no "
                           f"one can decide is a dead end")
            unknown = sorted(set(decisions)
                             - {"approve", "deny", "amend"})
            if unknown:
                self._fail(f"pause policy for state '{state_name}': "
                           f"unknown decision(s): {', '.join(unknown)} "
                           f"(known: amend, approve, deny)")
            if len(set(decisions)) != len(decisions):
                self._fail(f"pause policy for state '{state_name}': "
                           f"duplicate decisions declared")
            timeout = raw["timeout_ms"]
            if timeout is not None and (
                    isinstance(timeout, bool)
                    or not isinstance(timeout, int) or timeout <= 0):
                self._fail(f"pause policy for state '{state_name}': "
                           f"timeout_ms must be a positive integer "
                           f"or None, got {timeout!r}")

        # Checkpoint triggers are definition material: a negative or
        # non-integer trigger is a build error here, not a surprise
        # at the first commit boundary.
        for key, value in self._checkpoint_kwargs.items():
            if isinstance(value, bool) or not isinstance(value, int) \
                    or value < 0:
                self._fail(f"checkpoint trigger '{key}' must be a "
                           f"non-negative integer, got {value!r}")

        # Cycles need a termination bound.
        if self._has_cycle(adjacency) and not self._budget_kwargs.get("step_limit"):
            self._fail("transition graph has a cycle but no step budget: "
                       "declare .budget(step_limit=N) to bound termination")

    @staticmethod
    def _has_cycle(adjacency: dict[str, list[str]]) -> bool:
        visiting, done = set(), set()

        def visit(node: str) -> bool:
            if node in done:
                return False
            if node in visiting:
                return True
            visiting.add(node)
            for nxt in adjacency[node]:
                if visit(nxt):
                    return True
            visiting.remove(node)
            done.add(node)
            return False

        return any(visit(n) for n in adjacency)


class Harness:
    """Entry point: Harness.define("name", version="1.0.0")."""

    @staticmethod
    def define(name: str, version: str = "1.0.0") -> HarnessBuilder:
        return HarnessBuilder(name, version)
