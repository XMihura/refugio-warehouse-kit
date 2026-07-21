"""Stable public API for warehouse challenge agents.

Participant policies should import from this package rather than from simulator
internals:

    from warehouse_api import Action, Observation

    def act(observation: Observation) -> Action:
        return Action.WAIT
"""

from __future__ import annotations

from warehouse.actions import ActFunction, Action
from warehouse.layout import CellType
from warehouse.observation import GridView, Observation
from warehouse.state import Position

__all__ = [
    "ActFunction",
    "Action",
    "CellType",
    "GridView",
    "Observation",
    "Position",
]
