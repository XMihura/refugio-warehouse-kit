"""Generate visual debugger scenarios from simulator movement rules."""

from __future__ import annotations

import json
from pathlib import Path

from warehouse.actions import Action
from warehouse.layout import generate_grid
from warehouse.state import (
    ActionResult,
    Movement,
    Position,
    RobotState,
    StaticMovementResult,
    initial_robot_states,
    resolve_action,
    resolve_collisions,
    resolve_static_movement,
)


def _robot_step(
    robot_id: int,
    start: Position,
    movement: Movement,
    carrying: bool = False,
) -> dict[str, object]:
    result = resolve_static_movement(generate_grid(), start, movement)
    return _format_robot_result(robot_id, movement.value, result, carrying)


def _format_robot_result(
    robot_id: int,
    action: str,
    result: ActionResult | StaticMovementResult,
    carrying: bool = False,
) -> dict[str, object]:
    return {
        "id": robot_id,
        "movement": action,
        "carrying": carrying,
        "start": list(result.start_position),
        "intended": list(result.intended_position),
        "final": list(result.final_position),
        "blocked": result.blocked,
        "reason": result.reason,
    }


def _robot_action_step(
    robot: RobotState,
    action: Action,
    result: ActionResult | None = None,
) -> dict[str, object]:
    resolved = result if result is not None else resolve_action(generate_grid(), robot, action)
    return _format_robot_result(robot.robot_id, action.value, resolved, robot.carrying_item)


def _collision_steps(
    robots: tuple[RobotState, ...],
    actions: tuple[Action, ...],
) -> list[dict[str, object]]:
    grid = generate_grid()
    action_results = {
        robot.robot_id: resolve_action(grid, robot, action)
        for robot, action in zip(robots, actions, strict=True)
    }
    collision_results = resolve_collisions(robots, action_results)
    return [
        _robot_action_step(robot, action, collision_results[robot.robot_id])
        for robot, action in zip(robots, actions, strict=True)
    ]


def build_movement_debug_scenarios() -> dict[str, object]:
    initial_states = initial_robot_states("debug-scenarios")
    top_start = initial_states[0].position
    bottom_start = initial_states[24].position
    right_start = initial_states[72].position
    pickup_robot = RobotState(
        robot_id=0,
        position=(2, 3),
        base_position=(3, 0),
        target_item_position=(3, 3),
    )
    drop_robot = RobotState(
        robot_id=0,
        position=top_start,
        base_position=(3, 0),
        target_item_position=(3, 3),
        carrying_item=True,
    )

    scenarios = [
        {
            "id": "valid_outer_corridor_move",
            "title": "Valid move in outer corridor",
            "description": (
                "Robot 0 starts next to its top base and moves deeper "
                "into the warehouse."
            ),
            "robots": [_robot_step(0, top_start, Movement.DOWN)],
        },
        {
            "id": "blocked_by_external_base",
            "title": "Blocked by external base",
            "description": (
                "Robot 0 tries to move upward into its base dock. "
                "Bases are outside the walkable floor."
            ),
            "robots": [_robot_step(0, top_start, Movement.UP)],
        },
        {
            "id": "blocked_by_shelf",
            "title": "Blocked by shelf",
            "description": "Robot 12 tries to enter the first shelf block and remains in place.",
            "robots": [_robot_step(12, (2, 3), Movement.RIGHT)],
        },
        {
            "id": "wait_action",
            "title": "Wait action",
            "description": "WAIT keeps the robot in the same cell and is never statically blocked.",
            "robots": [_robot_step(24, bottom_start, Movement.WAIT, carrying=True)],
        },
        {
            "id": "small_batch",
            "title": "Small batch of independent static moves",
            "description": "A tiny fixture for visually checking multiple requested actions.",
            "robots": [
                _robot_step(0, top_start, Movement.DOWN),
                _robot_step(12, (2, 3), Movement.RIGHT),
                _robot_step(72, right_start, Movement.LEFT),
            ],
        },
        {
            "id": "vertex_conflict",
            "title": "Vertex conflict",
            "description": (
                "Two robots request the same final cell; both moving robots "
                "are blocked."
            ),
            "robots": _collision_steps(
                (
                    RobotState(0, (1, 1), (3, 0), (3, 3)),
                    RobotState(1, (3, 1), (5, 0), (3, 3)),
                ),
                (Action.RIGHT, Action.LEFT),
            ),
        },
        {
            "id": "edge_swap_conflict",
            "title": "Edge swap conflict",
            "description": "Two robots try to exchange cells in one tick; both are blocked.",
            "robots": _collision_steps(
                (
                    RobotState(0, (1, 1), (3, 0), (3, 3)),
                    RobotState(1, (2, 1), (5, 0), (3, 3)),
                ),
                (Action.RIGHT, Action.LEFT),
            ),
        },
        {
            "id": "pickup_adjacent_to_shelf",
            "title": "Pickup adjacent to shelf",
            "description": "A robot next to its target shelf cell can pick up without moving.",
            "robots": [_robot_action_step(pickup_robot, Action.PICKUP)],
        },
        {
            "id": "drop_at_base",
            "title": "Drop at base",
            "description": (
                "A carrying robot on its base-adjacent drop cell can drop "
                "without moving."
            ),
            "robots": [_robot_action_step(drop_robot, Action.DROP)],
        },
    ]
    return {
        "schema_version": 1,
        "layout_url": "/data/warehouse_layout.json",
        "scenarios": scenarios,
    }


def write_movement_debug_scenarios(path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_movement_debug_scenarios(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    write_movement_debug_scenarios(Path("data") / "movement_debug_scenarios.json")
