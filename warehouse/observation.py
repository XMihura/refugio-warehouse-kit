"""Observation model exposed to memoryless robot policies."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from warehouse.layout import CellType, generate_grid
from warehouse.state import Position, RobotState

GridView = tuple[tuple[CellType, ...], ...]


@dataclass(frozen=True, slots=True)
class Observation:
    tick: int
    robot_id: int
    position: Position
    base_position: Position
    target_item_position: Position
    carrying_item: bool
    grid: GridView
    all_robot_positions: Mapping[int, Position]


def freeze_grid(grid: list[list[CellType]]) -> GridView:
    return tuple(tuple(row) for row in grid)


def build_observation(
    tick: int,
    robot: RobotState,
    robots: tuple[RobotState, ...],
    grid: GridView | None = None,
    all_robot_positions: Mapping[int, Position] | None = None,
) -> Observation:
    """Build the per-robot observation.

    The observation includes all robot positions, but only this robot's base,
    target and carrying flag. Other robots' targets, bases and carrying flags
    remain hidden.

    ``all_robot_positions`` may be supplied so a batch can share one read-only
    snapshot; otherwise it is derived from ``robots``.
    """
    if tick < 0:
        raise ValueError(f"tick must be non-negative, got {tick}")

    grid_view = grid if grid is not None else freeze_grid(generate_grid())
    positions = (
        all_robot_positions
        if all_robot_positions is not None
        else _positions_view(robots)
    )

    return Observation(
        tick=tick,
        robot_id=robot.robot_id,
        position=robot.position,
        base_position=robot.base_position,
        target_item_position=robot.target_item_position,
        carrying_item=robot.carrying_item,
        grid=grid_view,
        all_robot_positions=positions,
    )


def build_observations(
    tick: int,
    robots: tuple[RobotState, ...],
    grid: GridView | None = None,
) -> tuple[Observation, ...]:
    grid_view = grid if grid is not None else freeze_grid(generate_grid())
    # One immutable positions snapshot is shared by every observation in the
    # tick: it is identical for all robots and never mutated by policies.
    positions = _positions_view(robots)
    return tuple(
        build_observation(tick, robot, robots, grid_view, positions)
        for robot in robots
    )


def _positions_view(robots: tuple[RobotState, ...]) -> Mapping[int, Position]:
    return MappingProxyType({state.robot_id: state.position for state in robots})
