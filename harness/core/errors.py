"""Harness error hierarchy. Failures are typed so callers can distinguish
a malformed definition (build time) from a runtime violation."""


class HarnessError(Exception):
    """Base for all harness-raised errors."""


class DefinitionError(HarnessError):
    """Raised at build time: the agent definition is malformed."""


class InvalidTransition(HarnessError):
    """No declared transition matches (state, event). Fail-closed."""


class GateRejected(HarnessError):
    """A gate rejected and no correction path remained (should normally
    route to escalation instead of raising; reserved for handler states)."""


class BudgetExceeded(HarnessError):
    """A budget limit was hit (normally routes to the exhausted state)."""


# Model failure taxonomy (#8): a model call fails in one of three
# typed ways, and the kind decides the response — correction can fix
# output, retry can fix transient, nothing fixes permanent.
MODEL_TRANSIENT = "transient"   # timeout, rate limit, 5xx: retry may help
MODEL_PERMANENT = "permanent"   # auth, invalid request: retry cannot help
MODEL_OUTPUT = "output"         # unparseable or invalid output: correctable

MODEL_FAILURE_KINDS = (MODEL_TRANSIENT, MODEL_PERMANENT, MODEL_OUTPUT)


class ModelCallError(HarnessError):
    """A model adapter's typed failure. Adapters raise this (never a
    bare SDK exception) so the runner can route by kind: transient
    and output failures enter the correction loop, permanent
    failures escalate immediately."""

    def __init__(self, kind: str, message: str) -> None:
        if kind not in MODEL_FAILURE_KINDS:
            raise ValueError(f"unknown model failure kind: {kind!r}")
        super().__init__(message)
        self.kind = kind
