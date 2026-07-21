"""Public action interface for memoryless robot policies."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from warehouse.observation import Observation


class Action(StrEnum):
    WAIT = "wait"
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"
    PICKUP = "pickup"
    DROP = "drop"


MOVEMENT_ACTIONS = frozenset(
    {
        Action.WAIT,
        Action.UP,
        Action.DOWN,
        Action.LEFT,
        Action.RIGHT,
    }
)


class ActFunction(Protocol):
    """Pure policy callable implemented by each team.

    The observation is passed positionally, so a policy may name its parameter
    however it likes (``obs``, ``observation``, ``state`` ...).
    """

    def __call__(self, observation: Observation, /) -> Action:
        """Return exactly one action for the given immutable observation."""


class PolicyTimeBudgetExceededError(RuntimeError):
    """Raised when a policy exhausts its cumulative compute-time budget.

    Unlike regular policy exceptions (which the simulator downgrades to a
    blocked WAIT), this error must abort the whole evaluation, so the
    simulator re-raises it instead of swallowing it.
    """

    def __init__(self, spent_seconds: float, budget_seconds: float) -> None:
        super().__init__(
            f"policy compute budget exceeded: "
            f"{spent_seconds:.3f}s > {budget_seconds:.3f}s"
        )
        self.spent_seconds = spent_seconds
        self.budget_seconds = budget_seconds


def coerce_action(value: Action | str) -> Action:
    """Normalize team-returned actions while keeping invalid values explicit."""
    if isinstance(value, Action):
        return value
    return Action(value)
