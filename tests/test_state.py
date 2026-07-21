import unittest
from dataclasses import replace

from warehouse.actions import Action
from warehouse.layout import CellType, generate_grid, iter_base_cells
from warehouse.state import (
    ActionResult,
    Movement,
    RobotState,
    apply_action,
    apply_collision_results,
    apply_static_movement,
    are_adjacent,
    drop_position_for_base,
    initial_robot_state,
    initial_robot_states,
    is_valid_robot_position,
    locked_shelf_positions,
    movement_destination,
    resolve_action,
    resolve_collisions,
    resolve_pickup,
    resolve_static_movement,
    validate_robot_states,
)
from warehouse.targets import target_for


class RobotStateTests(unittest.TestCase):
    def test_drop_positions_are_adjacent_inside_walkable_area(self) -> None:
        bases = list(iter_base_cells())

        self.assertEqual(drop_position_for_base(bases[0].position), (3, 1))
        self.assertEqual(drop_position_for_base(bases[24].position), (2, 50))
        self.assertEqual(drop_position_for_base(bases[48].position), (1, 2))
        self.assertEqual(drop_position_for_base(bases[72].position), (50, 3))

    def test_initial_robot_states(self) -> None:
        states = initial_robot_states("round-0")
        positions = [state.position for state in states]

        self.assertEqual(len(states), 96)
        self.assertEqual(len(set(positions)), 96)
        self.assertEqual(states[0].robot_id, 0)
        self.assertEqual(states[0].base_position, (3, 0))
        self.assertEqual(states[0].position, (3, 1))
        self.assertFalse(states[0].carrying_item)
        self.assertEqual(states[0].deliveries, 0)

    def test_initial_robot_state_requires_valid_start_cell(self) -> None:
        grid = generate_grid()
        base = list(iter_base_cells())[0]
        robot = initial_robot_state("round-0", base, grid)

        self.assertTrue(is_valid_robot_position(grid, robot.position))
        self.assertEqual(grid[robot.position[1]][robot.position[0]], CellType.EMPTY)

    def test_validate_robot_states_rejects_invalid_states(self) -> None:
        grid = generate_grid()
        valid = RobotState(0, (2, 3), (3, 0), (3, 3))

        self.assertEqual(validate_robot_states(grid, (valid,)), (valid,))

        with self.assertRaisesRegex(ValueError, "duplicate robot_id"):
            validate_robot_states(grid, (valid, replace(valid, position=(1, 1))))

        with self.assertRaisesRegex(ValueError, "duplicate robot position"):
            validate_robot_states(grid, (valid, replace(valid, robot_id=1, base_position=(5, 0))))

        with self.assertRaisesRegex(ValueError, "invalid position"):
            validate_robot_states(grid, (replace(valid, position=(3, 3)),))

        with self.assertRaisesRegex(ValueError, "invalid target shelf"):
            validate_robot_states(grid, (replace(valid, target_item_position=(2, 3)),))

        with self.assertRaisesRegex(ValueError, "negative deliveries"):
            validate_robot_states(grid, (replace(valid, deliveries=-1),))


class MovementTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = generate_grid()

    def test_movement_destination(self) -> None:
        self.assertEqual(movement_destination((10, 10), Movement.WAIT), (10, 10))
        self.assertEqual(movement_destination((10, 10), Movement.UP), (10, 9))
        self.assertEqual(movement_destination((10, 10), Movement.DOWN), (10, 11))
        self.assertEqual(movement_destination((10, 10), Movement.LEFT), (9, 10))
        self.assertEqual(movement_destination((10, 10), Movement.RIGHT), (11, 10))

    def test_valid_movement_into_empty_corridor(self) -> None:
        result = resolve_static_movement(self.grid, (1, 1), Movement.RIGHT)

        self.assertFalse(result.blocked)
        self.assertEqual(result.intended_position, (2, 1))
        self.assertEqual(result.final_position, (2, 1))
        self.assertIsNone(result.reason)

    def test_wait_never_blocks(self) -> None:
        result = resolve_static_movement(self.grid, (1, 1), Movement.WAIT)

        self.assertFalse(result.blocked)
        self.assertEqual(result.final_position, (1, 1))

    def test_cannot_enter_external_boundary_or_base(self) -> None:
        top_boundary = resolve_static_movement(self.grid, (1, 1), Movement.UP)
        left_boundary = resolve_static_movement(self.grid, (1, 2), Movement.LEFT)

        self.assertTrue(top_boundary.blocked)
        self.assertEqual(top_boundary.reason, "outside_walkable_area")
        self.assertEqual(top_boundary.final_position, (1, 1))

        self.assertTrue(left_boundary.blocked)
        self.assertEqual(left_boundary.reason, "outside_walkable_area")
        self.assertEqual(left_boundary.final_position, (1, 2))

    def test_cannot_enter_shelf(self) -> None:
        result = resolve_static_movement(self.grid, (2, 3), Movement.RIGHT)

        self.assertTrue(result.blocked)
        self.assertEqual(result.intended_position, (3, 3))
        self.assertEqual(result.final_position, (2, 3))
        self.assertEqual(result.reason, "blocked_by_shelf")

    def test_apply_static_movement_updates_position_only_when_valid(self) -> None:
        robot = initial_robot_states("round-0")[0]
        moved, result = apply_static_movement(self.grid, robot, Movement.DOWN)
        blocked, blocked_result = apply_static_movement(self.grid, robot, Movement.UP)

        self.assertFalse(result.blocked)
        self.assertEqual(moved.position, (3, 2))
        self.assertEqual(moved.base_position, robot.base_position)
        self.assertEqual(moved.target_item_position, robot.target_item_position)

        self.assertTrue(blocked_result.blocked)
        self.assertEqual(blocked.position, robot.position)


class ActionTransitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = generate_grid()
        self.seed = "round-0"

    def test_movement_actions_delegate_to_static_movement(self) -> None:
        robot = initial_robot_states(self.seed)[0]

        moved, result = apply_action(self.grid, robot, Action.DOWN, self.seed)

        self.assertFalse(result.blocked)
        self.assertEqual(result.action, Action.DOWN)
        self.assertEqual(moved.position, (3, 2))
        self.assertFalse(moved.carrying_item)

    def test_pickup_requires_adjacency_to_target_shelf(self) -> None:
        robot = RobotState(
            robot_id=0,
            position=(2, 3),
            base_position=(3, 0),
            target_item_position=(3, 3),
        )

        picked, result = apply_action(self.grid, robot, Action.PICKUP, self.seed)

        self.assertFalse(result.blocked)
        self.assertTrue(are_adjacent(robot.position, robot.target_item_position))
        self.assertTrue(picked.carrying_item)
        self.assertEqual(picked.deliveries, 0)
        self.assertEqual(picked.target_item_position, robot.target_item_position)

    def test_pickup_blocks_when_not_adjacent_or_already_carrying(self) -> None:
        far_robot = RobotState(
            robot_id=0,
            position=(1, 1),
            base_position=(3, 0),
            target_item_position=(3, 3),
        )
        carrying_robot = replace(far_robot, position=(2, 3), carrying_item=True)

        _, far_result = apply_action(self.grid, far_robot, Action.PICKUP, self.seed)
        _, carrying_result = apply_action(
            self.grid,
            carrying_robot,
            Action.PICKUP,
            self.seed,
        )

        self.assertTrue(far_result.blocked)
        self.assertEqual(far_result.reason, "not_adjacent_to_target")
        self.assertTrue(carrying_result.blocked)
        self.assertEqual(carrying_result.reason, "already_carrying")

    def test_pickup_blocks_when_target_shelf_is_locked_by_carrying_robot(self) -> None:
        shelf = (3, 3)
        carrier = RobotState(
            robot_id=0,
            position=(2, 3),
            base_position=(3, 0),
            target_item_position=shelf,
            carrying_item=True,
        )
        waiter = RobotState(
            robot_id=1,
            position=(4, 3),
            base_position=(5, 0),
            target_item_position=shelf,
        )

        result = resolve_pickup(self.grid, waiter, locked_shelf_positions((carrier,)))

        self.assertTrue(result.blocked)
        self.assertEqual(result.reason, "shelf_locked")

    def test_drop_requires_carrying_item_at_base_drop_position(self) -> None:
        robot = replace(
            initial_robot_states(self.seed)[0],
            carrying_item=True,
            deliveries=0,
        )

        dropped, result = apply_action(self.grid, robot, "drop", self.seed)

        self.assertFalse(result.blocked)
        self.assertEqual(result.action, Action.DROP)
        self.assertFalse(dropped.carrying_item)
        self.assertEqual(dropped.deliveries, 1)
        self.assertEqual(dropped.position, robot.position)
        self.assertEqual(dropped.target_item_position, target_for(self.seed, 0, 1))

    def test_drop_blocks_when_not_carrying_or_not_at_base(self) -> None:
        robot = initial_robot_states(self.seed)[0]
        away_robot = replace(robot, position=(10, 10), carrying_item=True)

        _, not_carrying = apply_action(self.grid, robot, Action.DROP, self.seed)
        _, away_result = apply_action(self.grid, away_robot, Action.DROP, self.seed)

        self.assertTrue(not_carrying.blocked)
        self.assertEqual(not_carrying.reason, "not_carrying")
        self.assertTrue(away_result.blocked)
        self.assertEqual(away_result.reason, "not_at_base_drop_position")
        self.assertEqual(away_result.intended_position, (3, 1))

    def test_resolve_action_rejects_target_that_is_not_shelf(self) -> None:
        robot = RobotState(
            robot_id=0,
            position=(1, 1),
            base_position=(3, 0),
            target_item_position=(2, 1),
        )

        result = resolve_action(self.grid, robot, Action.PICKUP)

        self.assertTrue(result.blocked)
        self.assertEqual(result.reason, "target_not_shelf")


class CollisionResolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = generate_grid()
        self.seed = "round-0"

    def _robot(self, robot_id: int, position: tuple[int, int]) -> RobotState:
        return RobotState(
            robot_id=robot_id,
            position=position,
            base_position=(3, 0),
            target_item_position=(3, 3),
        )

    def _results(
        self,
        robots: tuple[RobotState, ...],
        actions: tuple[Action, ...],
    ) -> dict[int, ActionResult]:
        return {
            robot.robot_id: resolve_action(self.grid, robot, action)
            for robot, action in zip(robots, actions, strict=True)
        }

    def test_vertex_conflict_blocks_all_moving_robots_entering_same_cell(self) -> None:
        robots = (self._robot(0, (1, 1)), self._robot(1, (3, 1)))
        results = self._results(robots, (Action.RIGHT, Action.LEFT))

        resolved = resolve_collisions(robots, results)

        self.assertTrue(resolved[0].blocked)
        self.assertTrue(resolved[1].blocked)
        self.assertEqual(resolved[0].reason, "vertex_conflict")
        self.assertEqual(resolved[1].reason, "vertex_conflict")
        self.assertEqual(resolved[0].final_position, (1, 1))
        self.assertEqual(resolved[1].final_position, (3, 1))

    def test_moving_into_waiting_robot_blocks_only_the_mover(self) -> None:
        # A stationary (WAIT) robot must never be reverted/relabeled by a robot
        # bumping into it; only the mover is blocked.
        robots = (self._robot(0, (1, 1)), self._robot(1, (2, 1)))
        results = self._results(robots, (Action.RIGHT, Action.WAIT))

        resolved = resolve_collisions(robots, results)

        self.assertTrue(resolved[0].blocked)
        self.assertEqual(resolved[0].reason, "vertex_conflict")
        self.assertEqual(resolved[0].final_position, (1, 1))

        self.assertFalse(resolved[1].blocked)
        self.assertIsNone(resolved[1].reason)
        self.assertEqual(resolved[1].final_position, (2, 1))

    def test_moving_into_statically_blocked_robot_keeps_static_reason(self) -> None:
        robots = (self._robot(0, (1, 1)), self._robot(1, (2, 1)))
        results = self._results(robots, (Action.RIGHT, Action.UP))

        resolved = resolve_collisions(robots, results)

        self.assertTrue(resolved[0].blocked)
        self.assertEqual(resolved[0].reason, "vertex_conflict")
        self.assertEqual(resolved[0].final_position, (1, 1))

        # Robot 1's UP move is statically blocked; it must keep its own reason.
        self.assertTrue(resolved[1].blocked)
        self.assertEqual(resolved[1].reason, "outside_walkable_area")
        self.assertEqual(resolved[1].final_position, (2, 1))

    def test_following_chain_blocks_when_leader_waits(self) -> None:
        # Classic MAPF cascade: a chain of robots each trying to advance into
        # the next robot's cell while the leader waits. A naive single pass
        # would let the tail move into a cell the blocked middle robot reverts
        # to, producing two robots on one cell. The fixpoint must block both
        # followers.
        robots = (
            self._robot(0, (1, 1)),
            self._robot(1, (2, 1)),
            self._robot(2, (3, 1)),
        )
        results = self._results(robots, (Action.RIGHT, Action.RIGHT, Action.WAIT))

        resolved = resolve_collisions(robots, results)
        finals = [resolved[robot_id].final_position for robot_id in sorted(resolved)]

        self.assertEqual(finals, [(1, 1), (2, 1), (3, 1)])
        self.assertEqual(len(set(finals)), len(finals))
        self.assertTrue(resolved[0].blocked)
        self.assertEqual(resolved[0].reason, "vertex_conflict")
        self.assertTrue(resolved[1].blocked)
        self.assertEqual(resolved[1].reason, "vertex_conflict")
        self.assertFalse(resolved[2].blocked)

    def test_following_chain_advances_when_leader_moves(self) -> None:
        # When the leader vacates into empty space, the whole chain advances.
        robots = (
            self._robot(0, (1, 1)),
            self._robot(1, (2, 1)),
            self._robot(2, (3, 1)),
        )
        results = self._results(robots, (Action.RIGHT, Action.RIGHT, Action.RIGHT))

        resolved = resolve_collisions(robots, results)
        finals = [resolved[robot_id].final_position for robot_id in sorted(resolved)]

        self.assertEqual(finals, [(2, 1), (3, 1), (4, 1)])
        self.assertEqual(len(set(finals)), len(finals))
        self.assertFalse(any(resolved[robot_id].blocked for robot_id in resolved))

    def test_rotation_cycle_is_allowed(self) -> None:
        # A pure rotation has all-distinct destinations and no 2-cycles, so it
        # must be permitted (no false vertex/edge blocking).
        robots = (
            self._robot(0, (1, 1)),
            self._robot(1, (2, 1)),
            self._robot(2, (2, 2)),
            self._robot(3, (1, 2)),
        )
        results = self._results(
            robots,
            (Action.RIGHT, Action.DOWN, Action.LEFT, Action.UP),
        )

        resolved = resolve_collisions(robots, results)
        finals = [resolved[robot_id].final_position for robot_id in sorted(resolved)]

        self.assertEqual(finals, [(2, 1), (2, 2), (1, 2), (1, 1)])
        self.assertEqual(len(set(finals)), len(finals))
        self.assertFalse(any(resolved[robot_id].blocked for robot_id in resolved))

    def test_edge_swap_blocks_both_robots(self) -> None:
        robots = (self._robot(0, (1, 1)), self._robot(1, (2, 1)))
        results = self._results(robots, (Action.RIGHT, Action.LEFT))

        resolved = resolve_collisions(robots, results)

        self.assertTrue(resolved[0].blocked)
        self.assertTrue(resolved[1].blocked)
        self.assertEqual(resolved[0].reason, "edge_swap_conflict")
        self.assertEqual(resolved[1].reason, "edge_swap_conflict")
        self.assertEqual(resolved[0].final_position, (1, 1))
        self.assertEqual(resolved[1].final_position, (2, 1))

    def test_independent_moves_are_applied(self) -> None:
        robots = (self._robot(0, (1, 1)), self._robot(1, (3, 1)))
        results = self._results(robots, (Action.DOWN, Action.RIGHT))

        next_states = apply_collision_results(robots, results)
        resolved = resolve_collisions(robots, results)

        self.assertFalse(resolved[0].blocked)
        self.assertFalse(resolved[1].blocked)
        self.assertEqual([state.position for state in next_states], [(1, 2), (4, 1)])


if __name__ == "__main__":
    unittest.main()
