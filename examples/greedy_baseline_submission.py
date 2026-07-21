"""Greedy baseline submission.

This file is intentionally self-contained and imports only from `warehouse_api`,
so it can be uploaded directly as a participant submission.
"""

from __future__ import annotations

from collections import deque

from warehouse_api import Action, CellType, GridView, Observation, Position

ACTION_DELTAS: tuple[tuple[Action, Position], ...] = (
    (Action.UP, (0, -1)),
    (Action.RIGHT, (1, 0)),
    (Action.DOWN, (0, 1)),
    (Action.LEFT, (-1, 0)),
)

_DELTA_BY_ACTION: dict[Action, Position] = dict(ACTION_DELTAS)
_GRID_INDEX_CACHE_LIMIT = 8


class _GridIndex:
    """Static pathfinding structures precomputed once per grid value."""

    __slots__ = ("grid", "passable", "neighbors")

    def __init__(self, grid: GridView) -> None:
        self.grid = grid
        passable = frozenset(
            (x, y)
            for y, row in enumerate(grid)
            for x, cell in enumerate(row)
            if cell == CellType.EMPTY
        )
        self.passable = passable
        self.neighbors: dict[Position, tuple[tuple[Action, Position], ...]] = {
            position: tuple(
                (action, candidate)
                for action, (dx, dy) in ACTION_DELTAS
                if (candidate := (position[0] + dx, position[1] + dy)) in passable
            )
            for position in passable
        }


_GRID_INDEXES: dict[int, _GridIndex] = {}


def act(observation: Observation) -> Action:
    """Return a deterministic greedy action for the current robot."""
    if not observation.carrying_item and _are_adjacent(
        observation.position,
        observation.target_item_position,
    ):
        return Action.PICKUP

    base_drop_position = _drop_position_for_base(observation.base_position)
    if observation.carrying_item and observation.position == base_drop_position:
        return Action.DROP

    index = _grid_index(observation.grid)
    blocked = {
        position
        for other_id, position in observation.all_robot_positions.items()
        if other_id != observation.robot_id
    }

    if not observation.carrying_item:
        departure_action = _base_departure_action(observation, index, blocked)
        if departure_action is not None:
            return departure_action

    goals = (
        (base_drop_position,)
        if observation.carrying_item
        else _pickup_positions(observation, index, blocked)
    )
    if not goals:
        return Action.WAIT

    return _shortest_path_step(observation.position, goals, index, blocked)


def _grid_index(grid: GridView) -> _GridIndex:
    key = id(grid)
    index = _GRID_INDEXES.get(key)
    if index is None:
        index = next(
            (cached for cached in _GRID_INDEXES.values() if cached.grid == grid),
            None,
        ) or _GridIndex(grid)
        _GRID_INDEXES[key] = index
        while len(_GRID_INDEXES) > _GRID_INDEX_CACHE_LIMIT:
            del _GRID_INDEXES[next(iter(_GRID_INDEXES))]
    return index


def _are_adjacent(left: Position, right: Position) -> bool:
    return abs(left[0] - right[0]) + abs(left[1] - right[1]) == 1


def _drop_position_for_base(base_position: Position) -> Position:
    x, y = base_position
    if y == 0:
        return (x, y + 1)
    if y == 51:
        return (x, y - 1)
    if x == 0:
        return (x + 1, y)
    if x == 51:
        return (x - 1, y)
    raise ValueError(f"base position must be on the perimeter: {base_position}")


def _base_departure_action(
    observation: Observation,
    index: _GridIndex,
    blocked: set[Position],
) -> Action | None:
    base_drop_position = _drop_position_for_base(observation.base_position)
    if observation.position != base_drop_position:
        return None

    bx, by = observation.base_position
    px, py = observation.position
    candidates = (
        (Action.DOWN, by < py),
        (Action.UP, by > py),
        (Action.RIGHT, bx < px),
        (Action.LEFT, bx > px),
    )
    for action, matches in candidates:
        if not matches:
            continue
        dx, dy = _DELTA_BY_ACTION[action]
        candidate = (px + dx, py + dy)
        if candidate in index.passable and candidate not in blocked:
            return action
        return None
    return None


def _pickup_positions(
    observation: Observation,
    index: _GridIndex,
    blocked: set[Position],
) -> tuple[Position, ...]:
    px, py = observation.position
    tx, ty = observation.target_item_position
    candidates = [
        candidate
        for candidate in ((tx + 1, ty), (tx, ty + 1), (tx - 1, ty), (tx, ty - 1))
        if candidate in index.passable and candidate not in blocked
    ]
    candidates.sort(
        key=lambda candidate: (
            abs(px - candidate[0]) + abs(py - candidate[1]),
            candidate[1],
            candidate[0],
        )
    )
    return tuple(candidates)


def _shortest_path_step(
    start: Position,
    goals: tuple[Position, ...],
    index: _GridIndex,
    blocked: set[Position],
) -> Action:
    if start in goals:
        return Action.WAIT

    goal_set = frozenset(goals)
    neighbors = index.neighbors
    visited = {start}
    queue: deque[tuple[Position, Action]] = deque()

    start_neighbors = neighbors.get(start)
    if start_neighbors is None:
        start_neighbors = tuple(
            (action, candidate)
            for action, (dx, dy) in ACTION_DELTAS
            if (candidate := (start[0] + dx, start[1] + dy)) in index.passable
        )

    current_neighbors = start_neighbors
    current_first_action: Action | None = None
    expansions: list[tuple[int, int, int, Action, Position]] = []
    while True:
        expansions.clear()
        for action, candidate in current_neighbors:
            if candidate in visited or candidate in blocked:
                continue
            cx, cy = candidate
            best = min(abs(cx - gx) + abs(cy - gy) for gx, gy in goals)
            expansions.append((best, cy, cx, action, candidate))
        expansions.sort()

        for _best, _cy, _cx, action, candidate in expansions:
            visited.add(candidate)
            first_action = action if current_first_action is None else current_first_action
            if candidate in goal_set:
                return first_action
            queue.append((candidate, first_action))

        if not queue:
            return Action.WAIT
        current, current_first_action = queue.popleft()
        current_neighbors = neighbors[current]
