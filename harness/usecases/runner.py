"""StateMachineRunner: the deterministic control plane.

Loop per state: enter -> entry gates -> work (model call or pure handler)
-> validate envelope -> invoke legal tools (idempotent) -> exit gates with
bounded correction -> select declared transition -> next state.

The model never selects the next state and never invokes tools directly.
Every step fires a log event through the single _log funnel. Events are
atomic: each payload carries what recreating its step requires
(RUN_STARTED holds the definition fingerprint and initial context;
MODEL_CALLED the rendered prompt and full output; TOOL_INVOKED the
args and result; TRANSITION the context delta the state committed;
LEDGER_POSTED both legs of a double-entry posting). Context at any
committed boundary is therefore a fold over the log. Persistence is
decoupled from execution: the event store may buffer asynchronously
(see adapters.events.async_store), and the runner flushes it at
completion — a returned Outcome guarantees its full event sequence is
persisted, and flush() raising there is how a persistence failure
surfaces. Every exit path flushes: run()/resume() wrap execution in
try/finally, so an exception unwinding the run (InvalidTransition,
adapter errors, tool exceptions) still leaves every fired event
persisted before the exception propagates. The finally-flush never
masks the in-flight exception; persistence failures it encounters
stay recorded on the store for the next flush()/close() to surface.
Fail-closed throughout.
"""

from __future__ import annotations

import copy
import functools
import hashlib
import json
import types
import uuid
from dataclasses import dataclass, field
from string import Template

from core.budget import Budget
from core.errors import (
    MODEL_OUTPUT, MODEL_PERMANENT, MODEL_TRANSIENT, HarnessError,
    InvalidTransition, ModelCallError,
)
from core.events import (
    BUDGET_EXHAUSTED, COMPLETED, CONTEXT_CHECKPOINT, ESCALATED,
    GATE_VERDICT, LEDGER_POSTED, MODEL_CALLED, MODEL_FAILED, PAUSE_DECIDED,
    PAUSE_REQUESTED, RUN_STARTED, STATE_ENTERED, STEP_COMMITTED,
    TOOL_INTENDED, TOOL_INVOKED, TRANSITION, DomainEvent, make_event,
)
from core.ledger import (
    CREDIT, DEBIT, Leg, PostingRule, Transaction, fold_ledger,
)
from core.state import PausePolicy, StateDef, Transition
from core.validate import validate_data
from usecases.gates import GateEvaluator
from usecases.ports import ClockPort, EventStorePort, ModelPort, ToolCall, ToolPort
from usecases.replay import RunReconstruction, reconstruct_run

EVENT_KEY = "event"
DATA_KEY = "data"
TOOL_CALLS_KEY = "tool_calls"

_ESCALATE = object()  # sentinel: correction budget exhausted


class _ModelFailure(Exception):
    """A model-path failure that the correction loop may retry."""


class _ModelPermanent(Exception):
    """A permanent model failure (#8): correction cannot fix it, so
    the state escalates immediately instead of burning its
    correction budget on retries that cannot help."""


class _BudgetHit(Exception):
    """Token budget exceeded mid-state."""


@dataclass
class Definition:
    name: str
    version: str
    states: dict[str, StateDef]
    transitions: list[Transition]
    initial: str
    escalation_state: str
    exhausted_state: str
    accounts: tuple[str, ...] = ()
    ledger_postings: tuple[PostingRule, ...] = ()
    redacted_context_keys: tuple[str, ...] = ()
    frozen_context_keys: tuple[str, ...] = ()
    tool_schemas: dict = field(default_factory=dict)
    # Checkpoint policy (v1.1, the database backup schedule): a
    # CONTEXT_CHECKPOINT carrying the folded context is written at a
    # commit boundary when any declared trigger fires — every N
    # committed steps, every N bytes of log since the last
    # checkpoint, or every N ms. 0 disables a trigger. Forcing a
    # checkpoint at shutdown or before a pause is future work; in
    # v1.1 checkpoints ride commits only.
    checkpoint_every_steps: int = 0
    checkpoint_every_bytes: int = 0
    checkpoint_every_ms: int = 0
    # HITL pause policies by state name (v1.1, #10). A state with a
    # policy stops before its work until a decision is in the log.
    pause_policies: dict = field(default_factory=dict)


@dataclass
class Outcome:
    execution_id: str
    final_state: str
    context: dict
    events: list[DomainEvent]
    success: bool
    ledger_incomplete: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    awaiting_decision: bool = False


def _stable_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _format_reasons(reasons: list[str]) -> str:
    if not reasons:
        return "(none)"
    return "\n".join(f"- {r}" for r in reasons)


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str,
                      separators=(",", ":"))


def _value_hash(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _sanitize_for_log(value: object, redacted: set[str],
                      key: str | None = None):
    """Deep-copy a value for an event payload, replacing the values of
    declared redacted keys with their sha256 hashes (marked, so the
    redaction is visible in the log and priced by recreation)."""
    if key is not None and key in redacted:
        return {"__redacted__": f"sha256:{_value_hash(value)}"}
    if isinstance(value, dict):
        return {k: _sanitize_for_log(v, redacted, k)
                for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_for_log(v, redacted) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_sanitize_for_log(v, redacted) for v in value),
                      key=_canonical)
    return copy.deepcopy(value)


def _string_forms(value: object):
    """Scalar string forms of a (possibly nested) value, for prompt
    redaction: the exact substrings a rendered prompt could contain."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, bool):
        return
    elif isinstance(value, (int, float)):
        yield str(value)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _string_forms(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _string_forms(item)


def _callable_id(fn) -> str | None:
    if fn is None:
        return None
    return f"{getattr(fn, '__module__', '?')}." \
           f"{getattr(fn, '__qualname__', type(fn).__name__)}"


def _definition_shape(definition: Definition, callable_repr) -> dict:
    """The definition's structural shape: graph, prompts, schemas,
    tools, gates-by-name, ledger chart/postings, and redactions.
    Callables (handlers, guards) are rendered by ``callable_repr`` —
    the legacy v1 fingerprint renders them by name; the v2 base shape
    renders them as None because v2 carries their content digests as
    separate components."""
    states = {}
    for name, state in sorted(definition.states.items()):
        states[name] = {
            "terminal": state.terminal,
            "prompt_template": state.prompt_template,
            "output_schema": state.output_schema,
            "handler": callable_repr(state.handler),
            "tools": list(state.tools),
            "writes": list(state.writes),
            "entry_gates": list(state.entry_gates),
            "exit_gates": list(state.exit_gates),
            "max_corrections": state.max_corrections,
        }
    return {
        "name": definition.name,
        "version": definition.version,
        "initial": definition.initial,
        "escalation_state": definition.escalation_state,
        "exhausted_state": definition.exhausted_state,
        "states": states,
        "transitions": [
            {"from": t.from_state, "event": t.event, "to": t.to_state,
             "guard": callable_repr(t.guard)}
            for t in definition.transitions],
        "accounts": list(definition.accounts),
        "tool_schemas": definition.tool_schemas,
        "checkpoint": {
            "every_steps": definition.checkpoint_every_steps,
            "every_bytes": definition.checkpoint_every_bytes,
            "every_ms": definition.checkpoint_every_ms,
        },
        "pause_policies": {
            name: {"approver": p.approver, "channel": p.channel,
                   "decisions": list(p.decisions),
                   "timeout_ms": p.timeout_ms}
            for name, p in sorted(definition.pause_policies.items())},
        "ledger_postings": [
            {"tool": r.tool, "debit_account": r.debit_account,
             "credit_account": r.credit_account,
             "amount_arg": r.amount_arg, "reference_arg": r.reference_arg}
            for r in definition.ledger_postings],
        "redacted_context_keys": list(definition.redacted_context_keys),
        "frozen_context_keys": list(definition.frozen_context_keys),
    }


def _definition_fingerprint(definition: Definition) -> str:
    """Legacy fingerprint_version 1: handlers/guards identified by
    module + qualified name only, gates not covered at all. Kept so
    resume() can validate logs written before fingerprint_version 2
    under exactly the comparison those logs were written with."""
    return _value_hash(_definition_shape(definition, _callable_id))


# ------------------------------------------------------------ content identity
# fingerprint_version 2: every component that bears execution semantics
# is identified by the digest of its compiled content, not its name.
# Renaming a function, reformatting it, or editing its comments does
# not change its digest; editing its body, a default, or a captured
# closure value does. All serialization is canonical (sorted keys,
# sorted sets) and hashed with sha256 — never the built-in hash(),
# whose string seed is randomized per process.

_MAX_DIGEST_DEPTH = 12


def _hash_canonical(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _is_jsonable(value) -> bool:
    """Strict JSON-ability (finite numbers, string dict keys), used to
    select the instance-__dict__ parts a digest may bind."""
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return value == value and abs(value) != float("inf")
    if isinstance(value, (list, tuple)):
        return all(_is_jsonable(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_jsonable(v)
                   for k, v in value.items())
    return False


def _serialize_value(value, _active=frozenset(), _depth=0):
    """Deterministic, JSON-able structural serialization of a value
    for digesting. Containers are canonicalized (dict entries and set
    members sorted by their serializations). Callables serialize as
    their content descriptors. Anything with no structural form falls
    back to repr() — documented: a default repr embeds a memory
    address, so such a component's digest is process-local; the
    fallback exists so exotic values still digest *something* rather
    than crashing fingerprinting."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return ["float", repr(value)]
    if isinstance(value, (bytes, bytearray)):
        return ["bytes", bytes(value).hex()]
    if _depth > _MAX_DIGEST_DEPTH:
        return ["truncated", type(value).__name__]
    if isinstance(value, types.ModuleType):
        return ["module", value.__name__]
    if isinstance(value, type):
        return ["type", f"{value.__module__}.{value.__qualname__}"]
    if isinstance(value, types.CodeType):
        return ["code", _code_descriptor(value, _active, _depth + 1)]
    if isinstance(value, dict):
        items = [[_serialize_value(k, _active, _depth + 1),
                  _serialize_value(v, _active, _depth + 1)]
                 for k, v in value.items()]
        items.sort(key=lambda pair: json.dumps(pair, sort_keys=True))
        return ["dict", items]
    if isinstance(value, (list, tuple)):
        return ["seq", [_serialize_value(v, _active, _depth + 1)
                        for v in value]]
    if isinstance(value, (set, frozenset)):
        items = [_serialize_value(v, _active, _depth + 1) for v in value]
        items.sort(key=lambda v: json.dumps(v, sort_keys=True))
        return ["set", items]
    if callable(value):
        return ["callable", _callable_descriptor(value, _active,
                                                 _depth + 1)]
    return ["repr", repr(value)]


def _code_descriptor(code, _active=frozenset(), _depth=0) -> dict:
    """Compiled-content descriptor of a code object: bytecode and
    constants and referenced names. Deliberately excludes the name,
    filename, line numbers, and local variable names, so renaming or
    reformatting the source does not change the digest. Nested code
    objects inside co_consts recurse through _serialize_value (a plain
    repr of co_consts would embed their memory addresses)."""
    return {
        "bytecode": code.co_code.hex(),
        "names": list(code.co_names),
        "consts": [_serialize_value(c, _active, _depth + 1)
                   for c in code.co_consts],
    }


def _function_descriptor(fn, _active, _depth) -> dict:
    """Content descriptor of a Python function: its code, defaults,
    keyword defaults, closure cell contents, and the globals its code
    actually references by name (modules skipped; referenced helper
    functions digest recursively, so editing a helper a handler calls
    by global name changes the handler's digest)."""
    code = fn.__code__
    globals_part = {}
    globs = getattr(fn, "__globals__", None) or {}
    for name in sorted(set(code.co_names) & set(globs)):
        val = globs[name]
        if isinstance(val, types.ModuleType):
            continue
        globals_part[name] = _serialize_value(val, _active, _depth + 1)
    closure = []
    for cell in (fn.__closure__ or ()):
        try:
            closure.append(_serialize_value(cell.cell_contents, _active,
                                            _depth + 1))
        except ValueError:  # empty cell
            closure.append(["empty-cell"])
    return {
        "kind": "function",
        "identity_only": False,
        "code": _code_descriptor(code, _active, _depth + 1),
        "defaults": _serialize_value(fn.__defaults__ or (), _active,
                                     _depth + 1),
        "kwdefaults": _serialize_value(fn.__kwdefaults__ or {}, _active,
                                       _depth + 1),
        "closure": closure,
        "globals": globals_part,
    }


def _callable_descriptor(fn, _active=frozenset(), _depth=0) -> dict:
    """Content descriptor for any callable. Falls back to identity
    (module + qualified name, marked identity_only) only when there
    is no Python code to digest — C-level callables and the like."""
    if id(fn) in _active or _depth > _MAX_DIGEST_DEPTH:
        return {"kind": "recursive", "identity_only": False}
    active = _active | {id(fn)}
    if isinstance(fn, functools.partial):
        return {
            "kind": "partial", "identity_only": False,
            "func": _callable_descriptor(fn.func, active, _depth + 1),
            "args": _serialize_value(fn.args, active, _depth + 1),
            "keywords": _serialize_value(fn.keywords or {}, active,
                                         _depth + 1),
        }
    if isinstance(fn, types.MethodType):
        return {
            "kind": "method", "identity_only": False,
            "func": _callable_descriptor(fn.__func__, active, _depth + 1),
            "self": _serialize_value(fn.__self__, active, _depth + 1),
        }
    if isinstance(fn, types.FunctionType):
        return _function_descriptor(fn, active, _depth)
    if isinstance(fn, type):
        parts = {}
        identity_only = True
        for attr in ("__init__", "__call__"):
            target = getattr(fn, attr, None)
            if isinstance(target, types.FunctionType):
                parts[attr] = _function_descriptor(target, active,
                                                   _depth + 1)
                identity_only = False
        descriptor = {"kind": "class", "identity_only": identity_only,
                      "parts": parts}
        if identity_only:
            descriptor["id"] = _callable_id(fn)
        return descriptor
    call = getattr(type(fn), "__call__", None)
    if isinstance(call, types.FunctionType):
        instance = {}
        if hasattr(fn, "__dict__"):
            instance = {
                k: _serialize_value(v, active, _depth + 1)
                for k, v in sorted(vars(fn).items()) if _is_jsonable(v)}
        return {
            "kind": "instance", "identity_only": False,
            "call": _function_descriptor(call, active, _depth + 1),
            "instance": instance,
        }
    return {"kind": "identity", "identity_only": True,
            "id": _callable_id(fn) or type(fn).__name__}


def _callable_digest(fn) -> tuple[str, bool]:
    """(digest, identity_only) for a callable's compiled content."""
    descriptor = _callable_descriptor(fn)
    return _hash_canonical(descriptor), bool(descriptor["identity_only"])


def _gate_component(gate) -> dict:
    """Fingerprint component for one gate: its class's check/__init__
    code plus its serializable configuration (instance __dict__ with
    callables — e.g. PredicateGate's predicate — digested by content,
    so a changed schema, term list, or predicate all register)."""
    cls = type(gate)
    check = getattr(cls, "check", None)
    check_digest = None
    identity_only = True
    if isinstance(check, types.FunctionType):
        check_digest, identity_only = _callable_digest(check)
    init = getattr(cls, "__init__", None)
    init_digest = None
    if isinstance(init, types.FunctionType):
        init_digest, _ = _callable_digest(init)
    config = (_serialize_value(vars(gate)) if hasattr(gate, "__dict__")
              else ["no-instance-dict"])
    descriptor = {
        "class": f"{cls.__module__}.{cls.__qualname__}",
        "check": check_digest,
        "init": init_digest,
        "config": config,
    }
    entry = {"digest": _hash_canonical(descriptor)}
    if identity_only:
        entry["identity_only"] = True
    return entry


_SECRET_CONFIG_KEYS = ("key", "secret", "token", "password", "auth",
                       "credential")


def _model_descriptor(model) -> dict | None:
    """Identity descriptor for the model adapter, or None when the
    adapter exposes no stable model identity to bind.

    An adapter is bound only by what it exposes: a model name
    (``model_name`` / ``model`` / ``_model`` string attribute), a
    temperature attribute if it carries one, and the JSON-able,
    non-secret parts of its client configuration. That binds the
    *named* model and its declared configuration — it cannot bind a
    remote model's server-side weights, and it deliberately does not
    bind opaque local adapters (scripted/function test doubles expose
    no model name): those resume unverified, as before. Credential
    material is never recorded: config keys naming keys/secrets/
    tokens/passwords are excluded outright, not hashed."""
    if model is None:
        return None
    name = None
    for attr in ("model_name", "model", "_model"):
        candidate = getattr(model, attr, None)
        if isinstance(candidate, str) and candidate:
            name = candidate
            break
    temperature = None
    for attr in ("temperature", "_temperature"):
        candidate = getattr(model, attr, None)
        if isinstance(candidate, (int, float)) \
                and not isinstance(candidate, bool):
            temperature = candidate
            break
    client_config = None
    raw_kwargs = getattr(model, "_client_kwargs", None)
    if isinstance(raw_kwargs, dict):
        client_config = {
            k: v for k, v in sorted(raw_kwargs.items())
            if isinstance(k, str) and _is_jsonable(v)
            and not any(s in k.lower() for s in _SECRET_CONFIG_KEYS)}
    if name is None and temperature is None and not client_config:
        return None
    descriptor: dict = {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
    }
    if name is not None:
        descriptor["model"] = name
    if temperature is not None:
        descriptor["temperature"] = temperature
    if client_config:
        descriptor["client_config"] = client_config
    return descriptor


def _definition_components(definition: Definition, gates: dict,
                           model, correction_model=None) -> dict:
    """fingerprint_version 2 component map: a digest for the base
    definition shape, one per state handler, one per transition
    guard, one per gate, and the model descriptor. Values are
    {"digest": hex} (plus "identity_only": True where a component
    could only be identified by name, and the model descriptor
    itself); the model entry is None when the adapter is unbound."""
    components: dict = {
        "definition": {"digest": _hash_canonical(_serialize_value(
            _definition_shape(definition, lambda fn: None)))},
    }
    for state_name in sorted(definition.states):
        handler = definition.states[state_name].handler
        if handler is not None:
            digest, identity_only = _callable_digest(handler)
            entry = {"digest": digest}
            if identity_only:
                entry["identity_only"] = True
            components[f"handler:{state_name}"] = entry
    for idx, t in enumerate(definition.transitions):
        if t.guard is not None:
            digest, identity_only = _callable_digest(t.guard)
            entry = {"digest": digest}
            if identity_only:
                entry["identity_only"] = True
            components[f"guard:{idx}:{t.from_state}:{t.event}:{t.to_state}"] \
                = entry
    for gate_name in sorted(gates):
        components[f"gate:{gate_name}"] = _gate_component(gates[gate_name])
    descriptor = _model_descriptor(model)
    if descriptor is None:
        components["model"] = None
    else:
        components["model"] = {"digest": _hash_canonical(descriptor),
                               "descriptor": descriptor}
    # The correction model is bound the same way when one is
    # declared (#12); None records "the primary model corrects
    # itself", which is what pre-seam logs carry implicitly.
    corr_descriptor = _model_descriptor(correction_model)
    if corr_descriptor is None:
        components["correction_model"] = None
    else:
        components["correction_model"] = {
            "digest": _hash_canonical(corr_descriptor),
            "descriptor": corr_descriptor}
    return components


def _component_label(name: str) -> str:
    if name == "definition":
        return ("the definition (graph, prompts, schemas, tools, "
                "chart, postings, redactions)")
    if name == "model":
        return "the model"
    if name == "correction_model":
        return "the correction model"
    if name.startswith("handler:"):
        return f"handler for state '{name[len('handler:'):]}'"
    if name.startswith("gate:"):
        return f"gate '{name[len('gate:'):]}'"
    if name.startswith("guard:"):
        parts = name.split(":", 4)
        if len(parts) == 5:
            return (f"guard for transition '{parts[2]}' "
                    f"--{parts[3]}--> '{parts[4]}'")
        return "a transition guard"
    return name


def _identity_only_notes(components: dict) -> list[str]:
    return [
        f"fingerprint: {_component_label(name)} is checked by "
        f"identity only (no Python code available to digest)"
        for name in sorted(components)
        if isinstance(components[name], dict)
        and components[name].get("identity_only")]


_V1_FALLBACK_NOTE = (
    "fingerprint v1 log: resumed under the legacy name-based "
    "fingerprint comparison; component-level code identity checks "
    "were not performed (the log predates fingerprint_version 2)")


class StateMachineRunner:
    def __init__(self, definition: Definition, *, model: ModelPort,
                 tools: ToolPort, gates: dict, events: EventStorePort,
                 clock: ClockPort, budget: Budget | None = None,
                 lease_ttl_ms: int = 60_000,
                 correction_model: ModelPort | None = None) -> None:
        self.definition = definition
        self.model = model
        # The correction seam (#12): when a separate correction model
        # is supplied, correction attempts (retries after a failure)
        # go to it instead of the primary model. None means the
        # primary model corrects itself, as before.
        self.correction_model = correction_model
        self.tools = tools
        self.gates = GateEvaluator(gates)
        self._gate_map = dict(gates)
        self.events = events
        self.clock = clock
        self.budget = budget
        self.lease_ttl_ms = lease_ttl_ms
        self._lease_owner: str | None = None
        # Checkpoint bookkeeping: commits and log bytes since the
        # last CONTEXT_CHECKPOINT, its timestamp, and whether any
        # committed work is still uncheckpointed.
        self._ckpt_commits = 0
        self._ckpt_bytes = 0
        self._ckpt_last_ms: int | None = None
        self._ckpt_dirty = False
        # States whose pause an approver waived for the rest of this
        # run (approve-and-remember, recorded in the decision event).
        self._pause_waivers: set[str] = set()
        self._outcome_notes: list[str] = []
        self._posting_rules: dict[str, PostingRule] = {
            rule.tool: rule for rule in definition.ledger_postings}
        # Declared out-events per state: the event names for which a
        # transition exists. The correction loop checks every envelope
        # against this set before any tool invocation (critique P4):
        # an undeclared event is a correctable failure, never a
        # post-effect InvalidTransition crash.
        self._declared_events: dict[str, frozenset] = {}
        for t in definition.transitions:
            self._declared_events.setdefault(t.from_state, set()).add(t.event)
        self._declared_events = {
            name: frozenset(events)
            for name, events in self._declared_events.items()}
        self._redacted: set[str] = set(definition.redacted_context_keys)
        self._frozen: set[str] = set(definition.frozen_context_keys)
        self._stored_events: tuple[DomainEvent, ...] = ()
        self._ledger_finalized: set[str] = set()
        self._terminal_jump_pending = False
        self._initial_context: dict = {}

    # ------------------------------------------------------------- helpers
    def _sanitize(self, value: object, key: str | None = None):
        return _sanitize_for_log(value, self._redacted, key)

    def _redact_prompt(self, prompt: str) -> str:
        """Replace the rendered values of redacted context keys in a
        prompt with their hash markers, for the MODEL_CALLED payload.
        The model received the real prompt; the log carries the
        redacted rendering, and recreation prices that (see the
        redaction note on HarnessBuilder.redacted_context_keys)."""
        replacements: list[tuple[str, str]] = []

        def collect(value: object, key: str | None) -> None:
            if key is not None and key in self._redacted:
                marker = f"[redacted sha256:{_value_hash(value)}]"
                for text in _string_forms(value):
                    if text:
                        replacements.append((text, marker))
                return
            if isinstance(value, dict):
                for k, v in value.items():
                    collect(v, k)
            elif isinstance(value, (list, tuple)):
                for v in value:
                    collect(v, None)

        collect(self._initial_context, None)
        for text, marker in sorted(replacements,
                                   key=lambda pair: -len(pair[0])):
            prompt = prompt.replace(text, marker)
        return prompt

    # ------------------------------------------------------------- leases
    def _acquire_lease(self, execution_id: str, action: str) -> None:
        """Become the single writer for this execution, or refuse.

        The lease is the lock manager's half of the database model:
        one writer per run, held from before the first append until
        the runner exits. A writer that dies without releasing is
        cleaned up by TTL expiry inside the store, never by a second
        writer barging in while the lease is live."""
        acquire = getattr(self.events, "acquire_lease", None)
        if acquire is None:
            return
        owner = f"runner-{uuid.uuid4().hex}"
        if not acquire(execution_id, owner, self.lease_ttl_ms):
            raise HarnessError(
                f"cannot {action}: execution '{execution_id}' is "
                f"leased by another writer (single-writer lease); "
                f"if that writer is dead its lease expires on its TTL")
        self._lease_owner = owner

    def _release_lease(self, execution_id: str) -> None:
        if self._lease_owner is None:
            return
        release = getattr(self.events, "release_lease", None)
        if release is not None:
            try:
                release(execution_id, self._lease_owner)
            except Exception:  # noqa: BLE001 - release must not mask
                pass
        self._lease_owner = None

    # ------------------------------------------------------------------ run
    def run(self, initial_context: dict | None = None,
            execution_id: str | None = None) -> Outcome:
        self._exec = execution_id or uuid.uuid4().hex
        self._acquire_lease(self._exec, "run")
        try:
            ctx = dict(initial_context or {})
            self._seq = 0
            self._local: list[DomainEvent] = []
            self._recovery_events: list[DomainEvent] = []
            self._recovery_index = 0
            self._recovery_plan = None
            self._recovery_budget_verified = True
            self._stored_events = ()
            self._ledger_finalized = set()
            self._terminal_jump_pending = False
            self._ckpt_commits = 0
            self._ckpt_bytes = 0
            self._ckpt_dirty = False
            self._ckpt_last_ms = None
            self._pause_waivers = set()
            self._initial_context = copy.deepcopy(ctx)
            components = _definition_components(
                self.definition, self._gate_map, self.model,
                self.correction_model)
            self._outcome_notes = _identity_only_notes(components)
            model_entry = components.get("model")
            self._log(RUN_STARTED, {
                "definition": self.definition.name,
                "version": self.definition.version,
                "definition_fingerprint": _hash_canonical(
                    {"fingerprint_version": 2, "components": components}),
                "fingerprint_version": 2,
                "commit_records": 2,
                "fingerprint_components": components,
                "model_descriptor": (model_entry["descriptor"]
                                     if isinstance(model_entry, dict)
                                     else None),
                "initial_state": self.definition.initial,
                "initial_context": self._sanitize(ctx),
                "redacted_context_keys":
                    sorted(self.definition.redacted_context_keys),
            })
            self._ckpt_last_ms = self._local[-1].ts_ms
            return self._loop(ctx, self.definition.initial)
        finally:
            # Every fired event is flushed whether the run completed,
            # escalated, or is unwinding through an exception.
            self._flush_quietly()
            self._release_lease(self._exec)

    def resume(self, initial_context: dict | None = None, *,
               execution_id: str) -> Outcome:
        """Resume an interrupted run from its persisted event log.

        Committed history is recreated by folding the log alone:
        RUN_STARTED supplies the initial context and each TRANSITION
        its committed delta, so no model calls are made for committed
        states and the caller's context is never trusted — if
        ``initial_context`` is supplied it must match RUN_STARTED's
        recorded (redaction-aware) context or resume refuses with
        HarnessError. Only the single in-flight state (events after
        the last committed boundary) may be deterministically
        re-driven, with regenerated events matched against the stored
        suffix. An already-completed log returns its folded outcome
        with zero model/tool calls. Raises HarnessError if no events
        exist for execution_id, if any fingerprint_version 2 component
        (definition, handler, guard, gate, model) differs from the
        one recorded at RUN_STARTED — checked before any event is
        appended or any model/tool call happens — or if re-driving
        the in-flight state diverges from the log. Logs written with
        fingerprint_version 1 are validated under the legacy
        name-based comparison only, and the returned Outcome's notes
        say so explicitly.
        """
        self._outcome_notes = []
        stored = sorted(self.events.read(execution_id),
                        key=lambda e: e.seq)
        if not stored:
            raise HarnessError(
                f"cannot resume: no persisted events for execution "
                f"'{execution_id}'")
        discarded_tail = getattr(self.events, "last_discarded_tail", None)
        self._acquire_lease(execution_id, "resume")
        try:
            plan = reconstruct_run(stored)
            self._validate_resume(stored, plan, initial_context)
            if discarded_tail:
                # The store discarded a torn final line while reading:
                # an uncommitted fragment from a crash mid-append. It
                # never happened; say so on the outcome.
                self._outcome_notes = list(self._outcome_notes) + [
                    "recovery: discarded an uncommitted partial event "
                    "at the log tail (torn write); it never happened"]
            if plan.pause_requested is not None \
                    and plan.pause_decision is None \
                    and not plan.completed:
                # An awaiting run: no decision, no resume — unless
                # the declared timeout has already expired, in which
                # case resume proceeds only to close the pause
                # fail-closed (see _handle_pause).
                policy = self.definition.pause_policies.get(
                    plan.pause_requested.payload.get("state"))
                expired = (
                    policy is not None and policy.timeout_ms is not None
                    and self.clock.now_ms() - plan.pause_requested.ts_ms
                    > policy.timeout_ms)
                if not expired:
                    req = plan.pause_requested.payload
                    raise HarnessError(
                        f"cannot resume: execution '{execution_id}' "
                        f"is paused awaiting a decision for state "
                        f"'{req.get('state')}' (approver "
                        f"{req.get('approver')}, channel "
                        f"{req.get('channel')}); record a decision in "
                        f"the log first — no decision, no resume")
            if plan.completed:
                return self._folded_outcome(stored, plan)
            return self._execute_resume(stored, plan, execution_id)
        finally:
            self._flush_quietly()
            self._release_lease(execution_id)

    def _validate_resume(self, stored: list[DomainEvent],
                         plan: RunReconstruction,
                         initial_context: dict | None) -> None:
        first = stored[0]
        if first.type != RUN_STARTED:
            raise HarnessError(
                "cannot resume: log has no RUN_STARTED event; only "
                "self-contained (atomic) logs are resumable")
        recorded_components = first.payload.get("fingerprint_components")
        if first.payload.get("fingerprint_version") == 2 \
                and isinstance(recorded_components, dict):
            current = _definition_components(
                self.definition, self._gate_map, self.model,
                self.correction_model)
            mismatches = []
            for name in sorted(set(recorded_components) | set(current)):
                recorded = recorded_components.get(name)
                presented = current.get(name)
                if _canonical(recorded) == _canonical(presented):
                    continue
                if name == "model":
                    rec_desc = (recorded.get("descriptor")
                                if isinstance(recorded, dict) else None)
                    cur_desc = (presented.get("descriptor")
                                if isinstance(presented, dict) else None)
                    mismatches.append(
                        "model changed since the run started "
                        f"(recorded {rec_desc.get('model') if rec_desc else None!r}, "
                        f"presented {cur_desc.get('model') if cur_desc else None!r})")
                else:
                    mismatches.append(
                        f"{_component_label(name)} changed since the "
                        f"run started")
            if mismatches:
                raise HarnessError(
                    "cannot resume: " + "; ".join(mismatches) +
                    " (fingerprint_version 2 component mismatch: the "
                    "log was produced under different execution "
                    "semantics)")
            self._outcome_notes = _identity_only_notes(current)
        else:
            # fingerprint_version 1 log (or a RUN_STARTED predating
            # structured fingerprints): validate under exactly the
            # legacy comparison the log was written with — handlers
            # and guards by name, gates and model unchecked — and
            # record that weaker check on the outcome rather than
            # pretending the v2 component checks ran.
            recorded_fingerprint = first.payload.get(
                "definition_fingerprint")
            if recorded_fingerprint != _definition_fingerprint(
                    self.definition):
                raise HarnessError(
                    "cannot resume: definition fingerprint differs "
                    "from the one recorded at RUN_STARTED")
            self._outcome_notes = [_V1_FALLBACK_NOTE]
        if initial_context is not None:
            supplied = self._sanitize(dict(initial_context))
            if _canonical(supplied) != _canonical(
                    plan.recorded_initial_context):
                raise HarnessError(
                    "cannot resume: supplied initial_context contradicts "
                    "the initial context recorded at RUN_STARTED; the "
                    "log is authoritative")

    def _folded_outcome(self, stored: list[DomainEvent],
                        plan: RunReconstruction) -> Outcome:
        incomplete = tuple(txn.txn_id
                           for txn in fold_ledger(stored).incomplete)
        success = (plan.final_state not in (self.definition.escalation_state,
                                            self.definition.exhausted_state)
                   and not incomplete)
        return Outcome(execution_id=stored[0].execution_id,
                       final_state=plan.final_state,
                       context=copy.deepcopy(plan.context),
                       events=list(stored), success=success,
                       ledger_incomplete=incomplete,
                       notes=tuple(self._outcome_notes))

    def _execute_resume(self, stored: list[DomainEvent],
                        plan: RunReconstruction,
                        execution_id: str) -> Outcome:
        prefix = [e for e in stored if e.seq < plan.boundary_seq]
        self._exec = execution_id
        self._seq = plan.boundary_seq
        self._local = list(prefix)
        self._recovery_events = list(plan.suffix_events)
        self._recovery_index = 0
        self._recovery_plan = plan
        self._recovery_budget_verified = not plan.suffix_events
        self._stored_events = tuple(stored)
        self._ledger_finalized = set()
        self._pause_waivers = {
            e.payload.get("state")
            for e in stored
            if e.type == PAUSE_DECIDED and e.payload.get("remember")
            and e.payload.get("decision") in ("approve", "amend")}
        self._seed_checkpoint_counters(
            [e for e in stored if e.seq < plan.boundary_seq])
        self._initial_context = copy.deepcopy(plan.recorded_initial_context)
        if self.budget is not None:
            # Committed accounting comes from the fold; re-driving the
            # in-flight suffix reproduces its share, which _log checks
            # against the folded totals before anything new appends.
            self.budget.steps_used = plan.boundary_steps_used
            self.budget.tokens_used = plan.boundary_tokens_used
            self.budget.tool_calls_used = plan.boundary_tool_calls_used
            self.budget.corrections_used = plan.boundary_corrections_used

        start_state = plan.current_state
        jump = False
        for event in plan.suffix_events:
            if event.type == ESCALATED:
                start_state = self.definition.escalation_state
                jump = True
            elif event.type == BUDGET_EXHAUSTED:
                start_state = self.definition.exhausted_state
                jump = True
        if start_state is None:
            raise HarnessError(
                "cannot resume: the log does not determine a current "
                "state")
        if jump:
            # The in-flight state already reached its terminal
            # decision (escalation / exhaustion) before the crash:
            # adopt the whole suffix as logged history instead of
            # re-driving a state whose outcome is already recorded.
            # Only the terminal entry and/or COMPLETED can be missing.
            self._local = list(stored)
            self._seq = plan.next_seq
            self._recovery_events = []
            self._recovery_index = 0
            self._recovery_budget_verified = True
            if self.budget is not None:
                self.budget.steps_used = plan.steps_used
                self.budget.tokens_used = plan.tokens_used
                self.budget.tool_calls_used = plan.tool_calls_used
                self.budget.corrections_used = plan.corrections_used
            if (plan.current_state == start_state
                    and plan.current_state_entered):
                # The terminal entry is already in the log too; only
                # COMPLETED is missing.
                return self._complete(copy.deepcopy(plan.context),
                                      start_state)
            self._terminal_jump_pending = True
            return self._loop(copy.deepcopy(plan.context), start_state)
        self._terminal_jump_pending = False
        return self._loop(copy.deepcopy(plan.context), start_state)

    # ------------------------------------------------------------ checkpoints
    def _seed_checkpoint_counters(self, stored: list[DomainEvent]) -> None:
        """Rebuild checkpoint bookkeeping from a persisted log on
        resume: commits and log bytes since the last checkpoint, and
        when it was taken (RUN_STARTED's timestamp if none)."""
        self._ckpt_commits = 0
        self._ckpt_bytes = 0
        self._ckpt_dirty = False
        self._ckpt_last_ms = stored[0].ts_ms if stored else None
        for event in stored:
            if event.type == CONTEXT_CHECKPOINT:
                self._ckpt_commits = 0
                self._ckpt_bytes = 0
                self._ckpt_dirty = False
                self._ckpt_last_ms = event.ts_ms
                continue
            self._ckpt_bytes += len(_canonical(
                {"type": event.type, "payload": event.payload}))
            if event.type == STEP_COMMITTED:
                self._ckpt_commits += 1
                self._ckpt_dirty = True

    def _checkpoint_policy_on(self) -> bool:
        d = self.definition
        return bool(d.checkpoint_every_steps or d.checkpoint_every_bytes
                    or d.checkpoint_every_ms)

    def _maybe_checkpoint(self, ctx: dict) -> None:
        """Write a CONTEXT_CHECKPOINT if a declared trigger fired.

        Called only immediately after a STEP_COMMITTED, so a
        checkpoint always sits inside committed territory and
        recovery always folds past it: a restore point is a
        committed point. The checkpoint carries the folded context;
        the deltas remain the truth, the checkpoint is the backup
        that lets recovery start closer to the crash."""
        if not self._checkpoint_policy_on():
            return
        d = self.definition
        trigger = None
        if d.checkpoint_every_steps \
                and self._ckpt_commits >= d.checkpoint_every_steps:
            trigger = "steps"
        elif d.checkpoint_every_bytes \
                and self._ckpt_bytes >= d.checkpoint_every_bytes:
            trigger = "bytes"
        elif d.checkpoint_every_ms and self._ckpt_last_ms is not None \
                and self.clock.now_ms() - self._ckpt_last_ms \
                >= d.checkpoint_every_ms:
            trigger = "time"
        if trigger is None:
            return
        self._log(CONTEXT_CHECKPOINT, {
            "context": self._sanitize(ctx),
            "trigger": trigger,
            "commits_since_checkpoint": self._ckpt_commits})
        self._ckpt_commits = 0
        self._ckpt_bytes = 0
        self._ckpt_dirty = False
        self._ckpt_last_ms = self._local[-1].ts_ms

    # ---------------------------------------------------------------- pauses
    def _handle_pause(self, ctx: dict, state_name: str,
                      policy: PausePolicy) -> Outcome | None:
        """The HITL pause point: after entry gates, before any work.

        Fresh run: write PAUSE_REQUESTED (the awaiting state, in the
        log, durable before the run stops) and return an awaiting
        Outcome. The process may exit; the pause survives because it
        is a log event, not a blocked thread.

        Resumed run: the request is already in the log — adopt it,
        then adopt the recorded decision and act on it. Approve
        proceeds to the state's work; amend applies its logged
        context patch first; deny escalates. A request with no
        decision reaches here only when resume() let an expired
        pause through, and expiry is fail-closed: escalate.

        Returns None when the state's work may proceed."""
        pending = self._remaining_event(
            PAUSE_REQUESTED,
            lambda e: e.payload.get("state") == state_name)
        if pending is None:
            self._log(PAUSE_REQUESTED, {
                "state": state_name,
                "approver": policy.approver,
                "channel": policy.channel,
                "allowed_decisions": list(policy.decisions),
                "timeout_ms": policy.timeout_ms,
                "context": self._sanitize(ctx)})
            # The request must be durable before the run reports
            # itself paused: a returned awaiting Outcome guarantees
            # the awaiting state is in the log.
            self.events.flush()
            note = (f"paused: state '{state_name}' awaits a decision "
                    f"from {policy.approver} (channel: "
                    f"{policy.channel})")
            return Outcome(
                execution_id=self._exec, final_state=state_name,
                context=ctx, events=list(self._local), success=False,
                awaiting_decision=True,
                notes=tuple(self._outcome_notes) + (note,))
        self._log(PAUSE_REQUESTED, pending.payload)
        decided = self._remaining_event(
            PAUSE_DECIDED,
            lambda e: e.payload.get("state") == state_name)
        if decided is None:
            return self._escalate(
                ctx, state_name,
                [("pause_timeout",
                  f"no decision within {policy.timeout_ms} ms; "
                  f"expiry is fail-closed")],
                phase="pause")
        self._log(PAUSE_DECIDED, decided.payload)
        decision = decided.payload.get("decision")
        if decided.payload.get("remember") \
                and decision in ("approve", "amend"):
            self._pause_waivers.add(state_name)
        if decision == "deny":
            return self._escalate(
                ctx, state_name,
                [("pause_denied",
                  f"denied by {decided.payload.get('decided_by')}")],
                phase="pause")
        if decision == "amend":
            patch = decided.payload.get("context_patch") or {}
            ctx.update(copy.deepcopy(patch))
        return None

    # ------------------------------------------------------------ main loop
    def _loop(self, ctx: dict, state_name: str) -> Outcome:
        while True:
            state = self.definition.states[state_name]
            self._log(STATE_ENTERED, {"state": state_name})

            if state.terminal and self._terminal_jump_pending:
                # The terminal entry was reached by an escalation /
                # exhaustion jump (possibly before the interruption):
                # jumps do not pass the budget check or gates.
                self._terminal_jump_pending = False
                return self._complete(ctx, state_name)

            if self.budget is not None:
                reason = self.budget.check_step()
                if reason:
                    return self._goto_exhausted(ctx, state_name, reason)
                self.budget.add_step()

            entry_subject = {"context": ctx, "state": state_name}
            failures = self.gates.evaluate(state.entry_gates, entry_subject)
            if failures:
                self._log(GATE_VERDICT, {
                    "state": state_name, "phase": "entry", "passed": False,
                    "failures": [list(f) for f in failures],
                    "subject": self._sanitize(entry_subject)})
                return self._escalate(ctx, state_name, failures, phase="entry")
            if state.entry_gates:
                self._log(GATE_VERDICT, {
                    "state": state_name, "phase": "entry", "passed": True,
                    "subject": self._sanitize(entry_subject)})

            if state.terminal:
                return self._complete(ctx, state_name)

            pause_policy = self.definition.pause_policies.get(state_name)
            if pause_policy is not None:
                # A pause already requested for this entry (a resumed
                # run) must be consumed, waiver or not: the waiver
                # from its decision applies to LATER entries.
                has_pending = self._remaining_event(
                    PAUSE_REQUESTED,
                    lambda e: e.payload.get("state") == state_name) \
                    is not None
                if has_pending or state_name not in self._pause_waivers:
                    pause_outcome = self._handle_pause(ctx, state_name,
                                                       pause_policy)
                    if pause_outcome is not None:
                        return pause_outcome

            before = copy.deepcopy(ctx)
            try:
                result = self._perform_state(state, ctx)
            except _BudgetHit as exc:
                return self._goto_exhausted(ctx, state_name, str(exc))

            if result is _ESCALATE:
                return self._escalate(ctx, state_name, self._last_failures,
                                      phase="work")

            next_state = self._select_transition(state_name, result, ctx)
            delta = {k: self._sanitize(v) for k, v in ctx.items()
                     if k not in before or before[k] != v}
            removed = [k for k in before if k not in ctx]
            transition_seq = self._seq
            self._log(TRANSITION, {"from": state_name, "event": result,
                                   "to": next_state,
                                   "context_delta": delta,
                                   "context_removed": removed})
            # The step is a transaction: the TRANSITION above is its
            # work, this record is its commit. A crash between the
            # two leaves an uncommitted transition, and recovery
            # treats it as never having happened (see reconstruct_run).
            self._log(STEP_COMMITTED, {
                "state": state_name, "event": result, "to": next_state,
                "transition_seq": transition_seq})
            self._ckpt_commits += 1
            self._ckpt_dirty = True
            self._maybe_checkpoint(ctx)
            state_name = next_state

    # ------------------------------------------------------------ tool calls
    def _all_known_events(self) -> list[DomainEvent]:
        """Stored events for this execution merged with locally fired
        ones, deduplicated by seq (recovery matches are in both)."""
        by_seq: dict[int, DomainEvent] = {}
        for event in self._stored_events:
            by_seq[event.seq] = event
        for event in self._local:
            by_seq[event.seq] = event
        return [by_seq[seq] for seq in sorted(by_seq)]

    def _remaining_event(self, type: str, predicate) -> DomainEvent | None:
        for event in self._recovery_events[self._recovery_index:]:
            if event.type == type and predicate(event):
                return event
        return None

    @staticmethod
    def _txn_id(idempotency_key: str) -> str:
        digest = hashlib.sha256(idempotency_key.encode()).hexdigest()
        return f"txn-{digest}"

    def _ledger_payload(self, txn_id: str, key: str, rule: PostingRule,
                        leg: Leg, args: dict, result: dict) -> dict:
        return {
            "txn_id": txn_id,
            "account": leg.account,
            "side": leg.side,
            "amount_cents": leg.amount_cents,
            "reference": self._sanitize(leg.reference, rule.reference_arg),
            "tool": rule.tool,
            "idempotency_key": key,
            "tool_args": self._sanitize(args),
            "tool_result": self._sanitize(result),
        }

    def _log_tool_intended(self, state_name: str, name: str, key: str,
                           args: dict) -> None:
        """The intent record (#2): logged BEFORE the tool fires, so
        the log shows what was about to happen even when the effect
        never lands (a crash between intent and invocation leaves
        intent without invocation — visible, not silent)."""
        stored = self._remaining_event(
            TOOL_INTENDED, lambda e: e.payload.get("key") == key)
        if stored is not None:
            self._log(TOOL_INTENDED, stored.payload)
            return
        if any(e.type == TOOL_INTENDED and e.payload.get("key") == key
               for e in self._local):
            return
        self._log(TOOL_INTENDED, {
            "state": state_name, "tool": name, "key": key,
            "args": self._sanitize(args)})

    def _check_tool_budget(self) -> None:
        """Price the call before it fires (#7). Raises _BudgetHit,
        which the run loop routes to the exhausted state."""
        if self.budget is None:
            return
        reason = self.budget.check_tool_call()
        if reason:
            raise _BudgetHit(reason)
        self.budget.add_tool_call()

    def _log_tool_invoked(self, state_name: str, name: str, key: str,
                          args: dict, result: dict,
                          txn_id: str | None) -> None:
        stored = self._remaining_event(
            TOOL_INVOKED, lambda e: e.payload.get("key") == key)
        if stored is not None:
            self._log(TOOL_INVOKED, stored.payload)
            return
        if any(e.type == TOOL_INVOKED and e.payload.get("key") == key
               for e in self._local):
            return
        self._log(TOOL_INVOKED, {
            "state": state_name, "tool": name, "key": key,
            "args": self._sanitize(args),
            "result": self._sanitize(result),
            "ledger_txn_id": txn_id})

    def _invoke_tool(self, call: dict, state_name: str) -> dict:
        name = call["name"]
        args = call["args"]
        key = (f"{self._exec}:{state_name}:{name}:"
               f"{_stable_hash(args)}")
        rule = self._posting_rules.get(name)
        if rule is not None:
            return self._invoke_ledger_tool(call, state_name, key, rule)
        stored = self._remaining_event(
            TOOL_INVOKED, lambda e: e.payload.get("key") == key)
        if stored is not None:
            # The effect already fired before the interruption and its
            # result is in the log: adopt it, never re-fire. Its
            # intent record precedes it in the log; adopt that too.
            self._log_tool_intended(state_name, name, key, args)
            result = copy.deepcopy(stored.payload.get("result") or {})
            self._log(TOOL_INVOKED, stored.payload)
            return result
        self._check_tool_budget()
        self._log_tool_intended(state_name, name, key, args)
        result = self.tools.invoke(ToolCall(name, args, key), state_name)
        self._log_tool_invoked(state_name, name, key, args, result, None)
        return result

    def _invoke_ledger_tool(self, call: dict, state_name: str, key: str,
                            rule: PostingRule) -> dict:
        """Invoke a ledger-aware tool under the double-entry invariant:
        the effect counts as done only when its transaction balances
        in the log (both legs, equal amounts)."""
        name = call["name"]
        args = call["args"]
        amount = args.get(rule.amount_arg)
        if (not isinstance(amount, int) or isinstance(amount, bool)
                or amount <= 0):
            raise HarnessError(
                f"ledger tool '{name}': amount arg '{rule.amount_arg}' "
                f"must be a positive integer, got {amount!r}")
        reference = args.get(rule.reference_arg)
        txn_id = self._txn_id(key)
        status = fold_ledger(self._all_known_events()).transaction(txn_id)
        # Intent precedes effect on the ledger path too (#2): the
        # record is written (or adopted, on resume) before any leg
        # posts and before the tool fires.
        self._log_tool_intended(state_name, name, key, args)

        if status is not None and status.legs:
            for leg in status.legs:
                expected_account = (rule.debit_account if leg.side == DEBIT
                                    else rule.credit_account)
                if (leg.account != expected_account
                        or leg.amount_cents != amount):
                    raise HarnessError(
                        f"ledger transaction {txn_id} in the log does "
                        f"not match the posting rule for '{name}'; "
                        f"refusing to proceed")
            proof = next(
                (e.payload for e in self._all_known_events()
                 if e.type == LEDGER_POSTED
                 and e.payload.get("txn_id") == txn_id
                 and "tool_result" in e.payload), None)
            if proof is None:
                raise HarnessError(
                    f"ledger transaction {txn_id} is incomplete and "
                    f"carries no effect proof; refusing to re-fire "
                    f"'{name}'")
            if _canonical(self._sanitize(args)) != _canonical(
                    proof.get("tool_args")):
                raise HarnessError(
                    f"ledger transaction {txn_id}: tool args in the log "
                    f"do not match this call; refusing to proceed")
            result = copy.deepcopy(proof["tool_result"])
            # Adopt legs already in the log (consuming any recovery
            # suffix), then post exactly the missing side, once.
            local_seqs = {e.seq for e in self._local}
            for event in self._all_known_events():
                if (event.type == LEDGER_POSTED
                        and event.payload.get("txn_id") == txn_id
                        and event.seq not in local_seqs):
                    self._log(LEDGER_POSTED, event.payload)
            present = {leg.side for leg in status.legs}
            for side, account in ((DEBIT, rule.debit_account),
                                  (CREDIT, rule.credit_account)):
                if side not in present:
                    leg = Leg(account, side, amount, reference)
                    self._log(LEDGER_POSTED, self._ledger_payload(
                        txn_id, key, rule, leg, args, result))
            self._ledger_finalized.add(txn_id)
            self._log_tool_invoked(state_name, name, key, args, result,
                                    txn_id)
            self._verify_ledger_balanced(txn_id)
            return result

        # No legs anywhere: fire the effect, then post both legs.
        self._check_tool_budget()
        result = self.tools.invoke(ToolCall(name, args, key), state_name)
        try:
            txn = Transaction(txn_id, (
                Leg(rule.debit_account, DEBIT, amount, reference),
                Leg(rule.credit_account, CREDIT, amount, reference)))
        except ValueError as exc:
            raise HarnessError(
                f"ledger tool '{name}': cannot build posting: {exc}"
            ) from exc
        for leg in txn.legs:
            self._log(LEDGER_POSTED, self._ledger_payload(
                txn_id, key, rule, leg, args, result))
        self._ledger_finalized.add(txn_id)
        self._log_tool_invoked(state_name, name, key, args, result, txn_id)
        self._verify_ledger_balanced(txn_id)
        return result

    def _verify_ledger_balanced(self, txn_id: str) -> None:
        status = fold_ledger(self._all_known_events()).transaction(txn_id)
        if status is None or not status.balanced:
            raise HarnessError(
                f"ledger transaction {txn_id} does not balance in the "
                f"log; the effect is not complete")

    # ----------------------------------------------------------------- states
    def _perform_state(self, state: StateDef, ctx: dict):
        """Run one non-terminal state. Returns the event name, or _ESCALATE."""
        reasons: list[str] = []
        attempts = state.max_corrections + 1
        for attempt in range(attempts):
            if attempt > 0 and self.budget is not None:
                # Correction attempts are priced actions (#7): the
                # run-global correction budget binds across states,
                # on top of each state's own max_corrections.
                reason = self.budget.add_correction()
                if reason:
                    raise _BudgetHit(reason)
            try:
                envelope = self._produce_envelope(state, ctx, reasons)
            except _ModelPermanent as exc:
                self._last_failures = [("model_permanent", str(exc))]
                return _ESCALATE
            except _ModelFailure as exc:
                reasons.append(str(exc))
                continue

            declared = self._declared_events.get(state.name, frozenset())
            if envelope[EVENT_KEY] not in declared:
                reasons.append(
                    f"event '{envelope[EVENT_KEY]}' is not a declared "
                    f"outcome of state '{state.name}' "
                    f"(declared: {sorted(declared)})")
                continue

            illegal = [c["name"] for c in envelope[TOOL_CALLS_KEY]
                       if c["name"] not in state.tools]
            if illegal:
                reasons.append(
                    f"tools not legal in state '{state.name}': {illegal}")
                continue

            # Argument schemas (critique P3): every call's args are
            # validated against the tool's declared schema before any
            # invocation, so a rejected attempt fires no effects at
            # all. Violations are correctable failures, like an
            # illegal tool or an undeclared event.
            arg_problems: list[str] = []
            for call in envelope[TOOL_CALLS_KEY]:
                schema = self.definition.tool_schemas.get(call["name"])
                if schema is not None:
                    for problem in validate_data(call["args"], schema):
                        arg_problems.append(
                            f"tool '{call['name']}' arguments: {problem}")
            if arg_problems:
                reasons.extend(arg_problems)
                continue

            # Dataflow discipline (#11): every context write flows
            # through the envelope's data, so it can be checked here,
            # before any tool fires. Two declared rules: a state with
            # declared writes may write only those keys, and a frozen
            # context key, once set, may not change value (writing
            # the same value is a no-op, not a change). Violations
            # are correctable failures, like a bad envelope.
            flow_problems: list[str] = []
            for key, value in envelope[DATA_KEY].items():
                if state.writes and key not in state.writes:
                    flow_problems.append(
                        f"state '{state.name}' may not write context "
                        f"key '{key}' (declared writes: "
                        f"{sorted(state.writes)})")
                if key in self._frozen and key in ctx \
                        and ctx[key] != value:
                    flow_problems.append(
                        f"context key '{key}' is frozen: once set it "
                        f"may not change (state '{state.name}' tried "
                        f"to change it)")
            if flow_problems:
                reasons.extend(flow_problems)
                continue

            # Work on a scratch copy; commit only if exit gates pass.
            scratch = dict(ctx)
            tool_results: dict[str, dict] = {}
            for call in envelope[TOOL_CALLS_KEY]:
                result = self._invoke_tool(call, state.name)
                tool_results[call["name"]] = result
            scratch.update(envelope[DATA_KEY])
            scratch["_tool_results"] = tool_results

            exit_subject = {"context": scratch, "data": envelope[DATA_KEY],
                            "state": state.name,
                            "event": envelope[EVENT_KEY]}
            failures = self.gates.evaluate(state.exit_gates, exit_subject)
            self._log(GATE_VERDICT, {
                "state": state.name, "phase": "exit",
                "passed": not failures,
                "failures": [list(f) for f in failures],
                "subject": self._sanitize(exit_subject)})
            if not failures:
                ctx.update(scratch)
                return envelope[EVENT_KEY]
            reasons.extend(f"{name}: {reason}" for name, reason in failures)

        self._last_failures = [("correction_budget", r) for r in reasons]
        return _ESCALATE

    def _produce_envelope(self, state: StateDef, ctx: dict,
                          reasons: list[str]) -> dict:
        if state.handler is not None:
            envelope = state.handler(dict(ctx))
            try:
                self._validate_envelope(envelope)
            except _ModelFailure as exc:
                raise HarnessError(
                    f"handler for state '{state.name}' produced an invalid "
                    f"envelope: {exc}") from exc
            return envelope
        if state.prompt_template is None:
            return {EVENT_KEY: "next", DATA_KEY: {}, TOOL_CALLS_KEY: []}
        mapping = {k: v for k, v in ctx.items()
                   if isinstance(k, str) and k.isidentifier()}
        mapping["correction"] = _format_reasons(reasons)
        prompt = Template(state.prompt_template).safe_substitute(mapping)
        # The correction seam (#12): the first attempt is the primary
        # model's; retries after a failure go to the correction
        # model when one is declared. The role is in the log, so a
        # reader can tell whose output each attempt was.
        if reasons and self.correction_model is not None:
            adapter, role = self.correction_model, "correction"
        else:
            adapter, role = self.model, "primary"
        try:
            result = adapter.generate(prompt, state.output_schema or {})
        except ModelCallError as exc:
            # Typed model failures (#8): every failure is a log
            # event with its kind. Permanent failures escalate at
            # once; transient and output failures enter the
            # correction loop like any other correctable failure.
            self._log(MODEL_FAILED,
                      {"state": state.name, "kind": exc.kind,
                       "error": str(exc), "role": role})
            if exc.kind == MODEL_PERMANENT:
                raise _ModelPermanent(str(exc)) from exc
            raise _ModelFailure(
                f"model {exc.kind} failure: {exc}") from exc
        self._log(MODEL_CALLED,
                  {"state": state.name, "role": role,
                   "prompt": self._redact_prompt(prompt),
                   "output": self._sanitize(result.data),
                   "prompt_tokens": result.prompt_tokens,
                   "completion_tokens": result.completion_tokens})
        if self.budget is not None:
            reason = self.budget.add_tokens(
                result.prompt_tokens + result.completion_tokens)
            if reason:
                raise _BudgetHit(reason)
        self._validate_envelope(result.data)
        if state.output_schema:
            problems = validate_data(result.data[DATA_KEY], state.output_schema)
            if problems:
                raise _ModelFailure("schema: " + "; ".join(problems))
        return result.data

    @staticmethod
    def _validate_envelope(envelope) -> None:
        if not isinstance(envelope, dict):
            raise _ModelFailure("envelope is not an object")
        for key in (EVENT_KEY, DATA_KEY, TOOL_CALLS_KEY):
            if key not in envelope:
                raise _ModelFailure(f"envelope missing key '{key}'")
        if not isinstance(envelope[EVENT_KEY], str):
            raise _ModelFailure("envelope['event'] must be a string")
        if not isinstance(envelope[DATA_KEY], dict):
            raise _ModelFailure("envelope['data'] must be an object")
        if not isinstance(envelope[TOOL_CALLS_KEY], list):
            raise _ModelFailure("envelope['tool_calls'] must be an array")
        for call in envelope[TOOL_CALLS_KEY]:
            if (not isinstance(call, dict) or not isinstance(call.get("name"), str)
                    or not isinstance(call.get("args"), dict)):
                raise _ModelFailure(
                    "each tool_call must be {name: string, args: object}")

    # ------------------------------------------------------------ transitions
    def _select_transition(self, state_name: str, event: str, ctx: dict) -> str:
        for t in self.definition.transitions:
            if t.from_state == state_name and t.event == event:
                if t.guard is None or t.guard(ctx):
                    return t.to_state
        raise InvalidTransition(
            f"no transition from '{state_name}' on event '{event}'")

    # -------------------------------------------------------------- terminals
    def _escalate(self, ctx: dict, state_name: str,
                  failures: list[tuple[str, str]], phase: str) -> Outcome:
        self._log(ESCALATED, {"state": state_name, "phase": phase,
                              "failures": [list(f) for f in failures]})
        return self._goto_terminal(ctx, self.definition.escalation_state)

    def _goto_exhausted(self, ctx: dict, state_name: str, reason: str) -> Outcome:
        self._log(BUDGET_EXHAUSTED, {"state": state_name, "reason": reason})
        return self._goto_terminal(ctx, self.definition.exhausted_state)

    def _goto_terminal(self, ctx: dict, state_name: str) -> Outcome:
        self._log(STATE_ENTERED, {"state": state_name})
        return self._complete(ctx, state_name)

    def _complete(self, ctx: dict, state_name: str) -> Outcome:
        self._log(COMPLETED, {"state": state_name,
                              "context": self._sanitize(ctx)})
        if (self._recovery_events
                and self._recovery_index == len(self._recovery_events)):
            self._verify_recovery_budget()
        # Every terminal path funnels through here. Flush before the
        # Outcome exists: a returned Outcome guarantees its full event
        # sequence is persisted, and a store that failed to persist
        # raises here (HarnessError from an async store's flush) rather
        # than letting the run report success over a hole in the log.
        self.events.flush()
        # Double-entry completion invariant: if any ledger transaction
        # in this run's log does not balance, no effect it belongs to
        # counts as complete — the outcome carries the incomplete
        # transaction ids and cannot report success.
        incomplete = tuple(
            txn.txn_id
            for txn in fold_ledger(self._all_known_events()).incomplete)
        success = (state_name not in (self.definition.escalation_state,
                                      self.definition.exhausted_state)
                   and not incomplete)
        return Outcome(execution_id=self._exec, final_state=state_name,
                       context=ctx, events=list(self._local), success=success,
                       ledger_incomplete=incomplete,
                       notes=tuple(self._outcome_notes))

    def _flush_quietly(self) -> None:
        """Flush from a finally block: drain every queued event before
        the run's result — or its exception — reaches the caller.
        Never raises: on an unwinding exception the original error is
        the signal, and on the happy path _complete already surfaced
        any persistence failure. Failures stay recorded on the store
        (flush reports each batch once), inspectable via its errors
        list and raised by the next explicit flush()/close()."""
        try:
            self.events.flush()
        except Exception:  # noqa: BLE001 - must not mask the real error
            pass

    # ---------------------------------------------------------------- logging
    def _verify_recovery_budget(self) -> None:
        if self._recovery_budget_verified:
            return
        plan = self._recovery_plan
        if plan is not None and self.budget is not None:
            actual = (self.budget.steps_used, self.budget.tokens_used)
            expected = (plan.steps_used, plan.tokens_used)
            if actual != expected:
                raise HarnessError(
                    "cannot resume: rebuilt budget (steps, tokens) "
                    f"{actual} diverges from the persisted log {expected}")
        self._recovery_budget_verified = True

    def _log(self, type: str, payload: dict) -> None:
        event = make_event(self._exec, self._seq, self.clock.now_ms(),
                           type, payload)
        self._seq += 1
        if self._recovery_index < len(self._recovery_events):
            expected = self._recovery_events[self._recovery_index]
            if (event.seq != expected.seq or event.type != expected.type
                    or event.payload != expected.payload
                    or event.schema_version != expected.schema_version):
                raise HarnessError(
                    "cannot resume: re-driven event at seq "
                    f"{expected.seq} diverges from the persisted log "
                    f"(expected {expected.type} {expected.payload!r}, "
                    f"got {event.type} {event.payload!r})")
            self._local.append(expected)
            self._recovery_index += 1
            return
        if self._recovery_events:
            # First event beyond the stored prefix: the prefix's budget
            # accounting must be fully reproduced before we append.
            self._verify_recovery_budget()
        self._local.append(event)
        self.events.append(event)
        self._ckpt_bytes += len(_canonical(
            {"type": event.type, "payload": event.payload}))
