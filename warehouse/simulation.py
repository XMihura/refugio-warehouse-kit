"""Tick-level simulator orchestration."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass, replace

from warehouse.actions import (
    ActFunction,
    Action,
    PolicyTimeBudgetExceededError,
    coerce_action,
)
from warehouse.layout import (
    CellType,
    build_layout,
    build_layout_from_shelves,
    generate_grid,
    grid_from_layout,
    grid_from_shelves,
    iter_shelf_cells,
    shelf_cells_from_layout,
)
from warehouse.observation import Observation, build_observations, freeze_grid
from warehouse.state import (
    ActionResult,
    RobotState,
    initial_robot_states,
    locked_shelf_positions,
    resolve_action,
    resolve_collisions,
    validate_robot_states,
)
from warehouse.targets import target_for

PolicySet = ActFunction | Mapping[int, ActFunction]


@dataclass(frozen=True, slots=True)
class TickResult:
    tick: int
    robots_before: tuple[RobotState, ...]
    actions: Mapping[int, Action]
    action_results: Mapping[int, ActionResult]
    robots_after: tuple[RobotState, ...]


@dataclass(frozen=True, slots=True)
class SimulationResult:
    global_seed: int | str
    ticks: int
    layout: Mapping[str, object]
    grid: list[list[CellType]]
    shelf_cells: tuple[tuple[int, int], ...]
    initial_robots: tuple[RobotState, ...]
    final_robots: tuple[RobotState, ...]
    tick_results: tuple[TickResult, ...]


def step_tick(
    tick: int,
    robots: Sequence[RobotState],
    policies: PolicySet,
    global_seed: int | str,
    grid: list[list[CellType]] | None = None,
    shelf_cells: Sequence[tuple[int, int]] | None = None,
) -> TickResult:
    """Resolve one full simultaneous simulation tick."""
    if tick < 0:
        raise ValueError(f"tick must be non-negative, got {tick}")

    global_seed = str(global_seed)
    grid = grid if grid is not None else generate_grid()
    target_shelves = (
        tuple(iter_shelf_cells()) if shelf_cells is None else tuple(shelf_cells)
    )
    robots_before = validate_robot_states(grid, robots)
    observations = build_observations(tick, robots_before, freeze_grid(grid))

    locked_shelves = locked_shelf_positions(robots_before)

    actions: dict[int, Action] = {}
    action_results: dict[int, ActionResult] = {}
    for robot, observation in zip(robots_before, observations, strict=True):
        action, invalid_reason = _call_policy(policies, robot.robot_id, observation)
        actions[robot.robot_id] = action
        if invalid_reason is not None:
            action_results[robot.robot_id] = _blocked_wait(robot, invalid_reason)
        else:
            action_results[robot.robot_id] = resolve_action(
                grid,
                robot,
                action,
                locked_shelves=locked_shelves,
            )

    action_results = _resolve_pickup_conflicts(robots_before, action_results)
    collision_results = resolve_collisions(robots_before, action_results)
    moved_robots = tuple(
        robot.with_position(collision_results[robot.robot_id].final_position)
        for robot in robots_before
    )
    robots_after = _apply_item_actions(
        grid,
        moved_robots,
        collision_results,
        global_seed,
        target_shelves,
    )

    robots_after = validate_robot_states(grid, robots_after)

    return TickResult(
        tick=tick,
        robots_before=robots_before,
        actions=actions,
        action_results=collision_results,
        robots_after=robots_after,
    )


def run_simulation(
    global_seed: int | str,
    policies: PolicySet,
    ticks: int = 10_000,
    robots: Sequence[RobotState] | None = None,
    grid: list[list[CellType]] | None = None,
    shelf_cells: Sequence[tuple[int, int]] | None = None,
    layout: Mapping[str, object] | None = None,
    record_ticks: bool = False,
) -> SimulationResult:
    """Run a deterministic simulation for a fixed number of ticks."""
    if ticks < 0:
        raise ValueError(f"ticks must be non-negative, got {ticks}")

    global_seed = str(global_seed)
    layout_json, grid, target_shelves = _resolve_layout_inputs(
        layout=layout,
        grid=grid,
        shelf_cells=shelf_cells,
    )
    initial_robots = (
        validate_robot_states(grid, robots, require_complete=True)
        if robots is not None
        else initial_robot_states(global_seed, grid=grid, shelf_cells=target_shelves)
    )
    current_robots = initial_robots
    recorded: list[TickResult] = []

    for tick in range(ticks):
        result = step_tick(
            tick,
            current_robots,
            policies,
            global_seed,
            grid,
            target_shelves,
        )
        current_robots = result.robots_after
        if record_ticks:
            recorded.append(result)

    return SimulationResult(
        global_seed=global_seed,
        ticks=ticks,
        layout=layout_json,
        grid=grid,
        shelf_cells=target_shelves,
        initial_robots=initial_robots,
        final_robots=current_robots,
        tick_results=tuple(recorded),
    )


def _resolve_pickup_conflicts(
    robots: tuple[RobotState, ...],
    action_results: Mapping[int, ActionResult],
) -> dict[int, ActionResult]:
    """Allow only one same-tick pickup per shelf, with deterministic priority."""
    resolved = dict(action_results)
    robot_by_id = {robot.robot_id: robot for robot in robots}
    pickups_by_shelf: dict[tuple[int, int], list[int]] = {}
    for robot_id, result in resolved.items():
        if result.blocked or result.action != Action.PICKUP:
            continue
        shelf = robot_by_id[robot_id].target_item_position
        pickups_by_shelf.setdefault(shelf, []).append(robot_id)

    for robot_ids in pickups_by_shelf.values():
        if len(robot_ids) <= 1:
            continue
        for robot_id in sorted(robot_ids)[1:]:
            result = resolved[robot_id]
            resolved[robot_id] = replace(result, blocked=True, reason="shelf_locked")
    return resolved


def _apply_item_actions(
    grid: list[list[CellType]],
    robots: tuple[RobotState, ...],
    action_results: Mapping[int, ActionResult],
    global_seed: int | str,
    shelf_cells: Sequence[tuple[int, int]],
) -> tuple[RobotState, ...]:
    next_robots: list[RobotState] = []
    for robot in robots:
        result = action_results[robot.robot_id]
        if result.blocked or result.action not in {Action.PICKUP, Action.DROP}:
            next_robots.append(robot)
            continue
        if result.action == Action.PICKUP:
            next_robots.append(robot.with_carrying(True))
            continue

        deliveries = robot.deliveries + 1
        next_robots.append(
            replace(
                robot,
                target_item_position=target_for(
                    global_seed,
                    robot.robot_id,
                    deliveries,
                    shelf_cells,
                ),
                carrying_item=False,
                deliveries=deliveries,
            )
        )
    return tuple(next_robots)


def _call_policy(
    policies: PolicySet,
    robot_id: int,
    observation: Observation,
) -> tuple[Action, str | None]:
    try:
        policy = _policy_for_robot(policies, robot_id)
    except KeyError:
        return Action.WAIT, "invalid_action"

    try:
        raw_action = policy(observation)
    except PolicyTimeBudgetExceededError:
        raise
    except Exception:
        return Action.WAIT, "policy_error"

    try:
        return coerce_action(raw_action), None
    except (TypeError, ValueError):
        reason = (
            f"invalid_action: {raw_action}"
            if isinstance(raw_action, str)
            else "invalid_action"
        )
        return Action.WAIT, reason


def _policy_for_robot(policies: PolicySet, robot_id: int) -> ActFunction:
    if isinstance(policies, MappingABC):
        return policies[robot_id]
    return policies


def _blocked_wait(robot: RobotState, reason: str) -> ActionResult:
    return ActionResult(
        Action.WAIT,
        robot.position,
        robot.position,
        robot.position,
        True,
        reason,
    )


def _resolve_layout_inputs(
    *,
    layout: Mapping[str, object] | None,
    grid: list[list[CellType]] | None,
    shelf_cells: Sequence[tuple[int, int]] | None,
) -> tuple[Mapping[str, object], list[list[CellType]], tuple[tuple[int, int], ...]]:
    if layout is not None:
        target_shelves = shelf_cells_from_layout(layout)
        return layout, grid_from_layout(layout), target_shelves

    if shelf_cells is None:
        target_shelves = tuple(iter_shelf_cells())
        layout_json = build_layout()
    else:
        target_shelves = tuple(shelf_cells)
        layout_json = build_layout_from_shelves(target_shelves)

    resolved_grid = grid if grid is not None else grid_from_shelves(target_shelves)
    return layout_json, resolved_grid, target_shelves
