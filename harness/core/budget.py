"""Budget: hard limits as an entity, not a config comment. The runner
consults the budget *before* priced actions; exhaustion is a first-class
transition, not an exception."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Budget:
    step_limit: int | None = None
    token_limit: int | None = None
    tool_call_limit: int | None = None
    correction_limit: int | None = None
    steps_used: int = 0
    tokens_used: int = 0
    tool_calls_used: int = 0
    corrections_used: int = 0

    def check_step(self) -> str | None:
        """None if a further step is allowed, else the exhaustion reason."""
        if self.step_limit is not None and self.steps_used >= self.step_limit:
            return f"step budget exhausted ({self.steps_used}/{self.step_limit})"
        return None

    def add_step(self) -> None:
        self.steps_used += 1

    def add_tokens(self, n: int) -> str | None:
        self.tokens_used += n
        if self.token_limit is not None and self.tokens_used > self.token_limit:
            return (f"token budget exceeded "
                    f"({self.tokens_used}/{self.token_limit})")
        return None

    def check_tool_call(self) -> str | None:
        """None if a further tool call is allowed (#7: tool calls are
        priced actions and are counted, like steps and tokens)."""
        if self.tool_call_limit is not None \
                and self.tool_calls_used >= self.tool_call_limit:
            return (f"tool-call budget exhausted "
                    f"({self.tool_calls_used}/{self.tool_call_limit})")
        return None

    def add_tool_call(self) -> None:
        self.tool_calls_used += 1

    def add_correction(self) -> str | None:
        """Count one correction attempt (#7); returns the exhaustion
        reason when the correction budget is exceeded. Correction
        attempts are priced: each one is another model call, another
        chance to wander."""
        self.corrections_used += 1
        if self.correction_limit is not None \
                and self.corrections_used > self.correction_limit:
            return (f"correction budget exceeded "
                    f"({self.corrections_used}/{self.correction_limit})")
        return None

    @classmethod
    def from_kwargs(cls, **kwargs) -> "Budget | None":
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if not kwargs:
            return None
        return cls(**kwargs)
