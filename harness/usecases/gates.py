"""Gate evaluation orchestration. The checks themselves live in adapters;
this runs a named pipeline and aggregates failures."""

from __future__ import annotations

from usecases.ports import GatePort


class GateEvaluator:
    def __init__(self, gates: dict[str, GatePort]) -> None:
        self._gates = dict(gates)

    def evaluate(self, names: tuple[str, ...], subject: dict) -> list[tuple[str, str]]:
        """Return [(gate_name, reason)] for each failing gate, in order.
        Short-circuits on first failure: fail fast, diagnose precisely."""
        failures: list[tuple[str, str]] = []
        for name in names:
            verdict = self._gates[name].check(subject)
            if not verdict.passed:
                failures.append((name, verdict.reason))
                break
        return failures
