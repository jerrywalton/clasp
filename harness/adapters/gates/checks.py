"""Gate adapters: concrete checks implementing GatePort."""

from __future__ import annotations

from core.validate import validate_data
from usecases.ports import Verdict


class SchemaGate:
    """Validates subject['data'] against a fixed schema (core.validate)."""

    def __init__(self, schema: dict, name: str = "schema") -> None:
        self.schema = schema
        self.name = name

    def check(self, subject: dict) -> Verdict:
        problems = validate_data(subject.get("data", {}), self.schema)
        if problems:
            return Verdict(False, "; ".join(problems))
        return Verdict(True)


class ContentGate:
    """Substring policy over a text field of the subject: required present,
    forbidden absent. Deterministic, no model involved.

    `path` locates the text inside the subject dict; the runner's exit-gate
    subject carries model output under ("data", <field>), which is the
    default.
    """

    def __init__(self, required: tuple[str, ...] = (),
                 forbidden: tuple[str, ...] = (),
                 path: tuple[str, ...] = ("data", "message")) -> None:
        self.required = tuple(required)
        self.forbidden = tuple(forbidden)
        self.path = tuple(path)

    def check(self, subject: dict) -> Verdict:
        node = subject
        for key in self.path:
            node = node.get(key, {}) if isinstance(node, dict) else {}
        message = node if isinstance(node, str) else ""
        lowered = message.lower()
        for needle in self.required:
            if needle.lower() not in lowered:
                return Verdict(False, f"message must mention '{needle}'")
        for needle in self.forbidden:
            if needle.lower() in lowered:
                return Verdict(False, f"message must not contain '{needle}'")
        return Verdict(True)


class PredicateGate:
    """Wraps a pure predicate for custom checks."""

    def __init__(self, predicate, description: str = "predicate") -> None:
        self.predicate = predicate
        self.description = description

    def check(self, subject: dict) -> Verdict:
        ok, reason = self.predicate(subject)
        return Verdict(bool(ok), "" if ok else (reason or self.description))
