"""Robot state and static movement rules."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from warehouse.actions import MOVEMENT_ACTIONS, Action, coerce_action
from warehouse.layout import (
    GRID_SIZE,
    ROBOT_COUNT,
    BaseCell,
    CellType,
    generate_grid,
    in_bounds,
    in_walkable_area,
    is_walkable_cell,
    iter_base_cells,
    iter_shelf_cells,
)
from warehouse.targets import target_for

Position = tuple[int, int]


class Movement(StrEnum):
    WAIT = "wait"
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


MOVE_DELTAS: dict[Movement, Position] = {
    Movement.WAIT: (0, 0),
    Movement.UP: (0, -1),
    Movement.DOWN: (0, 1),
    Movement.LEFT: (-1, 0),
    Movement.RIGHT: (1, 0),
}

ACTION_TO_MOVEMENT: dict[Action, Movement] = {
    Action.WAIT: Movement.WAIT,
    Action.UP: Movement.UP,
    Action.DOWN: Movement.DOWN,
    Action.LEFT: Movement.LEFT,
    Action.RIGHT: Movement.RIGHT,
}


@dataclass(frozen=True, slots=True)
class RobotState:
    robot_id: int
    position: Position
    base_position: Position
    target_item_position: Position
    carrying_item: bool = False
    deliveries: int = 0

    def with_position(self, position: Position) -> RobotState:
        return replace(self, position=position)

    def with_target(self, target_item_position: Position) -> RobotState:
        return replace(self, target_item_position=target_item_position)

    def with_carrying(self, carrying_item: bool) -> RobotState:
        return replace(self, carrying_item=carrying_item)

    def with_deliveries(self, deliveries: int) -> RobotState:
        return replace(self, deliveries=deliveries)


@dataclass(frozen=True, slots=True)
class StaticMovementResult:
    movement: Movement
    start_position: Position
    intended_position: Position
    final_position: Position
    blocked: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class ActionResult:
    action: Action
    start_position: Position
    intended_position: Position
    final_position: Position
    blocked: bool
    reason: str | None = None


CollisionResults = dict[int, ActionResult]


def are_adjacent(a: Position, b: Position) -> bool:
    ax, ay = a
    bx, by = b
    return abs(ax - bx) + abs(ay - by) == 1


def locked_shelf_positions(robots: Sequence[RobotState]) -> frozenset[Position]:
    """Shelf cells reserved by robots that picked up but have not delivered yet."""
    return frozenset(
        robot.target_item_position for robot in robots if robot.carrying_item
    )


def validate_robot_states(
    grid: list[list[CellType]],
    robots: Sequence[RobotState],
    *,
    require_complete: bool = False,
) -> tuple[RobotState, ...]:
    """Validate robot states before they enter the simulator tick loop.

    The simulator accepts small custom robot tuples in unit tests and visual
    fixtures, so completeness is optional. Every provided state must still obey
    the official geometry and identity invariants.
    """
    robot_states = tuple(robots)
    if require_complete and len(robot_states) != ROBOT_COUNT:
        raise ValueError(
            f"expected {ROBOT_COUNT} robot states, got {len(robot_states)}"
        )

    seen_ids: set[int] = set()
    seen_positions: set[Position] = set()
    for robot in robot_states:
        _validate_robot_state(grid, robot)
        if robot.robot_id in seen_ids:
            raise ValueError(f"duplicate robot_id: {robot.robot_id}")
        seen_ids.add(robot.robot_id)
        if robot.position in seen_positions:
            raise ValueError(f"duplicate robot position: {robot.position}")
        seen_positions.add(robot.position)
    return robot_states


def movement_destination(position: Position, movement: Movement) -> Position:
    dx, dy = MOVE_DELTAS[movement]
    x, y = position
    return (x + dx, y + dy)


def drop_position_for_base(base_position: Position) -> Position:
    """Return the unique walkable cell adjacent to an external base."""
    x, y = base_position
    if x == 0:
        return (1, y)
    if x == GRID_SIZE - 1:
        return (GRID_SIZE - 2, y)
    if y == 0:
        return (x, 1)
    if y == GRID_SIZE - 1:
        return (x, GRID_SIZE - 2)
    raise ValueError(f"base_position must be on the external boundary: {base_position}")


def is_valid_robot_position(grid: list[list[CellType]], position: Position) -> bool:
    return is_walkable_cell(grid, position)


def resolve_static_movement(
    grid: list[list[CellType]],
    position: Position,
    movement: Movement,
) -> StaticMovementResult:
    intended = movement_destination(position, movement)

    if movement == Movement.WAIT:
        return StaticMovementResult(movement, position, intended, position, False)

    if not in_bounds(intended):
        return StaticMovementResult(
            movement, position, intended, position, True, "out_of_bounds"
        )

    if not in_walkable_area(intended):
        return StaticMovementResult(
            movement, position, intended, position, True, "outside_walkable_area"
        )

    x, y = intended
    if grid[y][x] != CellType.EMPTY:
        return StaticMovementResult(
            movement,
            position,
            intended,
            position,
            True,
            f"blocked_by_{grid[y][x].value}",
        )

    return StaticMovementResult(movement, position, intended, intended, False)


def apply_static_movement(
    grid: list[list[CellType]],
    robot: RobotState,
    movement: Movement,
) -> tuple[RobotState, StaticMovementResult]:
    result = resolve_static_movement(grid, robot.position, movement)
    return robot.with_position(result.final_position), result


def resolve_action(
    grid: list[list[CellType]],
    robot: RobotState,
    action: Action | str,
    locked_shelves: frozenset[Position] = frozenset(),
) -> ActionResult:
    """Resolve a single robot action before multi-robot collision handling."""
    normalized_action = coerce_action(action)

    if normalized_action in MOVEMENT_ACTIONS:
        movement_result = resolve_static_movement(
            grid,
            robot.position,
            ACTION_TO_MOVEMENT[normalized_action],
        )
        return ActionResult(
            Action.WAIT if movement_result.blocked else normalized_action,
            movement_result.start_position,
            movement_result.intended_position,
            movement_result.final_position,
            movement_result.blocked,
            movement_result.reason,
        )

    if normalized_action == Action.PICKUP:
        return resolve_pickup(grid, robot, locked_shelves)

    if normalized_action == Action.DROP:
        return resolve_drop(robot)

    raise ValueError(f"unsupported action: {normalized_action}")


def resolve_pickup(
    grid: list[list[CellType]],
    robot: RobotState,
    locked_shelves: frozenset[Position] = frozenset(),
) -> ActionResult:
    if robot.carrying_item:
        return ActionResult(
            Action.PICKUP,
            robot.position,
            robot.position,
            robot.position,
            True,
            "already_carrying",
        )

    if robot.target_item_position in locked_shelves:
        return ActionResult(
            Action.PICKUP,
            robot.position,
            robot.position,
            robot.position,
            True,
            "shelf_locked",
        )

    tx, ty = robot.target_item_position
    if not in_bounds(robot.target_item_position):
        return ActionResult(
            Action.PICKUP,
            robot.position,
            robot.position,
            robot.position,
            True,
            "target_out_of_bounds",
        )

    if grid[ty][tx] != CellType.SHELF:
        return ActionResult(
            Action.PICKUP,
            robot.position,
            robot.position,
            robot.position,
            True,
            "target_not_shelf",
        )

    if not are_adjacent(robot.position, robot.target_item_position):
        return ActionResult(
            Action.PICKUP,
            robot.position,
            robot.position,
            robot.position,
            True,
            "not_adjacent_to_target",
        )

    return ActionResult(
        Action.PICKUP,
        robot.position,
        robot.position,
        robot.position,
        False,
    )


def resolve_drop(robot: RobotState) -> ActionResult:
    if not robot.carrying_item:
        return ActionResult(
            Action.DROP,
            robot.position,
            robot.position,
            robot.position,
            True,
            "not_carrying",
        )

    drop_position = drop_position_for_base(robot.base_position)
    if robot.position != drop_position:
        return ActionResult(
            Action.DROP,
            robot.position,
            drop_position,
            robot.position,
            True,
            "not_at_base_drop_position",
        )

    return ActionResult(
        Action.DROP,
        robot.position,
        robot.position,
        robot.position,
        False,
    )


def apply_action(
    grid: list[list[CellType]],
    robot: RobotState,
    action: Action | str,
    global_seed: int | str,
    locked_shelves: frozenset[Position] = frozenset(),
    shelf_cells: Sequence[Position] | None = None,
) -> tuple[RobotState, ActionResult]:
    """Apply a single robot action, excluding simultaneous collision rules."""
    result = resolve_action(grid, robot, action, locked_shelves)
    if result.blocked:
        return robot.with_position(result.final_position), result

    if result.action in MOVEMENT_ACTIONS:
        return robot.with_position(result.final_position), result

    if result.action == Action.PICKUP:
        return robot.with_carrying(True), result

    if result.action == Action.DROP:
        deliveries = robot.deliveries + 1
        target_shelves = (
            tuple(iter_shelf_cells()) if shelf_cells is None else tuple(shelf_cells)
        )
        return (
            replace(
                robot,
                carrying_item=False,
                deliveries=deliveries,
                target_item_position=target_for(
                    global_seed,
                    robot.robot_id,
                    deliveries,
                    target_shelves,
                ),
            ),
            result,
        )

    raise ValueError(f"unsupported action: {result.action}")


def resolve_collisions(
    robots: Sequence[RobotState],
    action_results: Mapping[int, ActionResult],
) -> CollisionResults:
    """Apply simultaneous vertex and edge-swap collision rules.

    Resolution is deterministic and independent of robot ordering. Only robots
    that are actually moving can be blocked; stationary robots (``WAIT``,
    ``PICKUP``, ``DROP`` or statically blocked moves) always keep their result
    and their current cell, so a moving robot can never cancel a neighbour's
    pickup or drop simply by bumping into it.

    A moving robot is blocked when its destination coincides with the final
    cell of any other robot. Because reverting a blocked robot to its start
    cell can create a *new* conflict (a chain of robots following a leader that
    stays put), the vertex rule is iterated to a fixpoint. This guarantees the
    core invariant that no two robots ever occupy the same cell after a tick.
    """
    robot_by_id = _robot_map(robots)
    resolved = dict(action_results)

    for robot_id in resolved:
        if robot_id not in robot_by_id:
            raise ValueError(f"missing robot state for robot_id {robot_id}")

    robot_ids = sorted(resolved)
    start = {robot_id: resolved[robot_id].start_position for robot_id in robot_ids}
    intended_final = {
        robot_id: resolved[robot_id].final_position for robot_id in robot_ids
    }
    is_mover = {
        robot_id: _can_be_collision_blocked(resolved[robot_id])
        for robot_id in robot_ids
    }
    final = dict(intended_final)
    blocked_reason: dict[int, str] = {}

    # Edge swaps are detected on the originally intended destinations so that a
    # later revert cannot mask or fabricate a swap.
    for left_index, left_id in enumerate(robot_ids):
        if not is_mover[left_id]:
            continue
        for right_id in robot_ids[left_index + 1 :]:
            if not is_mover[right_id]:
                continue
            if (
                start[left_id] == intended_final[right_id]
                and start[right_id] == intended_final[left_id]
            ):
                blocked_reason.setdefault(left_id, "edge_swap_conflict")
                blocked_reason.setdefault(right_id, "edge_swap_conflict")

    for robot_id in blocked_reason:
        final[robot_id] = start[robot_id]

    # Vertex conflicts are resolved to a fixpoint. Recomputing occupancy each
    # pass lets a revert cascade backwards through a chain of followers, while
    # using a per-pass snapshot keeps symmetric conflicts mutually blocking.
    changed = True
    while changed:
        changed = False
        occupancy: defaultdict[Position, int] = defaultdict(int)
        for robot_id in robot_ids:
            occupancy[final[robot_id]] += 1
        for robot_id in robot_ids:
            if robot_id in blocked_reason or not is_mover[robot_id]:
                continue
            if occupancy[final[robot_id]] > 1:
                blocked_reason[robot_id] = "vertex_conflict"
                final[robot_id] = start[robot_id]
                changed = True

    for robot_id, reason in blocked_reason.items():
        result = resolved[robot_id]
        resolved[robot_id] = replace(
            result,
            final_position=start[robot_id],
            blocked=True,
            reason=reason,
        )

    return resolved


def apply_collision_results(
    robots: Sequence[RobotState],
    action_results: Mapping[int, ActionResult],
) -> tuple[RobotState, ...]:
    """Return robot states after movement collisions are resolved."""
    resolved = resolve_collisions(robots, action_results)
    next_states = []
    for robot in robots:
        result = resolved[robot.robot_id]
        next_states.append(robot.with_position(result.final_position))
    return tuple(next_states)


def _robot_map(robots: Sequence[RobotState]) -> dict[int, RobotState]:
    robot_by_id = {robot.robot_id: robot for robot in robots}
    if len(robot_by_id) != len(robots):
        raise ValueError("robot ids must be unique")
    return robot_by_id


def _validate_robot_state(grid: list[list[CellType]], robot: RobotState) -> None:
    if not isinstance(robot.robot_id, int) or not 0 <= robot.robot_id < ROBOT_COUNT:
        raise ValueError(f"invalid robot_id: {robot.robot_id}")

    if robot.deliveries < 0:
        raise ValueError(
            f"robot {robot.robot_id} has negative deliveries: {robot.deliveries}"
        )

    if not is_valid_robot_position(grid, robot.position):
        raise ValueError(
            f"robot {robot.robot_id} has invalid position: {robot.position}"
        )

    bx, by = robot.base_position
    if not in_bounds(robot.base_position) or grid[by][bx] != CellType.BASE:
        raise ValueError(
            f"robot {robot.robot_id} has invalid base: {robot.base_position}"
        )

    drop_position = drop_position_for_base(robot.base_position)
    if not is_valid_robot_position(grid, drop_position):
        raise ValueError(
            f"robot {robot.robot_id} has invalid base drop cell: {drop_position}"
        )

    tx, ty = robot.target_item_position
    if not in_bounds(robot.target_item_position) or grid[ty][tx] != CellType.SHELF:
        raise ValueError(
            f"robot {robot.robot_id} has invalid target shelf: "
            f"{robot.target_item_position}"
        )


def _can_be_collision_blocked(result: ActionResult) -> bool:
    return (
        not result.blocked
        and result.action in MOVEMENT_ACTIONS
        and result.final_position != result.start_position
    )


def initial_robot_states(
    global_seed: int | str,
    *,
    grid: list[list[CellType]] | None = None,
    shelf_cells: Sequence[Position] | None = None,
) -> tuple[RobotState, ...]:
    """Build deterministic initial robot states from bases and initial targets."""
    grid = grid if grid is not None else generate_grid()
    target_shelves = (
        tuple(iter_shelf_cells()) if shelf_cells is None else tuple(shelf_cells)
    )
    states: list[RobotState] = []
    for base in iter_base_cells():
        states.append(
            initial_robot_state(
                global_seed,
                base,
                grid,
                shelf_cells=target_shelves,
            )
        )
    return tuple(states)


def initial_robot_state(
    global_seed: int | str,
    base: BaseCell,
    grid: list[list[CellType]] | None = None,
    *,
    shelf_cells: Sequence[Position] | None = None,
) -> RobotState:
    if not 0 <= base.robot_id < ROBOT_COUNT:
        raise ValueError(f"invalid robot_id: {base.robot_id}")

    grid = grid if grid is not None else generate_grid()
    target_shelves = (
        tuple(iter_shelf_cells()) if shelf_cells is None else tuple(shelf_cells)
    )
    start_position = drop_position_for_base(base.position)
    if not is_valid_robot_position(grid, start_position):
        raise ValueError(
            f"base {base.position} does not have a valid adjacent start cell: "
            f"{start_position}"
        )

    return RobotState(
        robot_id=base.robot_id,
        position=start_position,
        base_position=base.position,
        target_item_position=target_for(global_seed, base.robot_id, 0, target_shelves),
    )
