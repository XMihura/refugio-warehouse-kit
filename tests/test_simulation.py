import random
import unittest
from dataclasses import replace
from typing import cast

from warehouse.actions import ActFunction, Action
from warehouse.layout import (
    LAYOUT_SCHEMA_VERSION,
    CellType,
    iter_shelf_cells,
    validate_submitted_layout,
)
from warehouse.observation import Observation
from warehouse.simulation import run_simulation, step_tick
from warehouse.state import RobotState, initial_robot_states
from warehouse.targets import target_for


class SimulationTests(unittest.TestCase):
    def test_step_tick_applies_valid_movement(self) -> None:
        robot = initial_robot_states("round-0")[0]

        result = step_tick(0, (robot,), lambda _obs: Action.DOWN, "round-0")

        self.assertEqual(result.actions[0], Action.DOWN)
        self.assertFalse(result.action_results[0].blocked)
        self.assertEqual(result.robots_after[0].position, (3, 2))

    def test_step_tick_resolves_collision_conflicts(self) -> None:
        robots = (
            RobotState(0, (1, 1), (3, 0), (3, 3)),
            RobotState(1, (3, 1), (5, 0), (3, 3)),
        )
        policies = {0: lambda _obs: Action.RIGHT, 1: lambda _obs: Action.LEFT}

        result = step_tick(0, robots, policies, "round-0")

        self.assertEqual([robot.position for robot in result.robots_after], [(1, 1), (3, 1)])
        self.assertEqual(result.action_results[0].reason, "vertex_conflict")
        self.assertEqual(result.action_results[1].reason, "vertex_conflict")

    def test_following_chain_does_not_produce_overlap(self) -> None:
        # Regression: a chain of robots advancing while the leader waits used to
        # leave two robots on one cell, which then tripped post-tick validation.
        robots = (
            RobotState(0, (1, 1), (3, 0), (3, 3)),
            RobotState(1, (2, 1), (5, 0), (3, 3)),
            RobotState(2, (3, 1), (7, 0), (3, 3)),
        )
        policies = {
            0: lambda _obs: Action.RIGHT,
            1: lambda _obs: Action.RIGHT,
            2: lambda _obs: Action.WAIT,
        }

        result = step_tick(0, robots, policies, "round-0")
        positions = [robot.position for robot in result.robots_after]

        self.assertEqual(positions, [(1, 1), (2, 1), (3, 1)])
        self.assertEqual(len(set(positions)), len(positions))

    def test_pickup_is_not_cancelled_by_a_bumping_robot(self) -> None:
        # A stationary robot performing a PICKUP must keep its pickup even if a
        # neighbour tries to move onto its cell in the same tick.
        picker = RobotState(0, (2, 3), (3, 0), (3, 3))
        bumper = RobotState(1, (1, 3), (5, 0), (7, 3))
        policies = {0: lambda _obs: Action.PICKUP, 1: lambda _obs: Action.RIGHT}

        result = step_tick(0, (picker, bumper), policies, "round-0")

        self.assertFalse(result.action_results[0].blocked)
        self.assertTrue(result.robots_after[0].carrying_item)
        self.assertTrue(result.action_results[1].blocked)
        self.assertEqual(result.action_results[1].reason, "vertex_conflict")
        self.assertEqual(result.robots_after[1].position, (1, 3))

    def test_step_tick_applies_pickup_after_action_resolution(self) -> None:
        robot = RobotState(
            robot_id=0,
            position=(2, 3),
            base_position=(3, 0),
            target_item_position=(3, 3),
        )

        result = step_tick(0, (robot,), lambda _obs: Action.PICKUP, "round-0")

        self.assertFalse(result.action_results[0].blocked)
        self.assertTrue(result.robots_after[0].carrying_item)
        self.assertEqual(result.robots_after[0].deliveries, 0)

    def test_same_tick_pickup_race_allows_lowest_robot_id(self) -> None:
        robots = (
            RobotState(0, (2, 3), (3, 0), (3, 3)),
            RobotState(1, (3, 2), (5, 0), (3, 3)),
        )
        policies = {0: lambda _obs: Action.PICKUP, 1: lambda _obs: Action.PICKUP}

        result = step_tick(0, robots, policies, "round-0")

        self.assertFalse(result.action_results[0].blocked)
        self.assertTrue(result.action_results[1].blocked)
        self.assertEqual(result.action_results[1].reason, "shelf_locked")
        self.assertTrue(result.robots_after[0].carrying_item)
        self.assertFalse(result.robots_after[1].carrying_item)

    def test_shelf_lock_releases_after_drop(self) -> None:
        carrier = RobotState(
            0,
            (3, 1),
            (3, 0),
            (3, 3),
            carrying_item=True,
        )
        waiter = RobotState(1, (2, 3), (5, 0), (3, 3))
        policies = {0: lambda _obs: Action.DROP, 1: lambda _obs: Action.PICKUP}

        first = step_tick(0, (carrier, waiter), policies, "round-0")

        self.assertFalse(first.action_results[0].blocked)
        self.assertEqual(first.action_results[1].reason, "shelf_locked")
        self.assertFalse(first.robots_after[0].carrying_item)
        self.assertFalse(first.robots_after[1].carrying_item)

        second = step_tick(1, first.robots_after, policies, "round-0")

        self.assertFalse(second.action_results[1].blocked)
        self.assertTrue(second.robots_after[1].carrying_item)

    def test_step_tick_applies_drop_and_reassigns_target(self) -> None:
        robot = replace(initial_robot_states("round-0")[0], carrying_item=True)

        result = step_tick(0, (robot,), lambda _obs: Action.DROP, "round-0")

        self.assertFalse(result.action_results[0].blocked)
        self.assertFalse(result.robots_after[0].carrying_item)
        self.assertEqual(result.robots_after[0].deliveries, 1)
        self.assertEqual(result.robots_after[0].target_item_position, target_for("round-0", 0, 1))

    def test_invalid_action_and_policy_error_are_blocked(self) -> None:
        robots = (
            RobotState(0, (1, 1), (3, 0), (3, 3)),
            RobotState(1, (2, 1), (5, 0), (3, 3)),
        )

        def raising_policy(_observation: Observation) -> Action:
            raise RuntimeError("boom")

        def teleport_policy(_observation: Observation) -> str:
            return "teleport"

        policies: dict[int, ActFunction] = {
            0: cast(ActFunction, teleport_policy),
            1: cast(ActFunction, raising_policy),
        }
        result = step_tick(0, robots, policies, "round-0")

        self.assertTrue(result.action_results[0].blocked)
        self.assertEqual(result.action_results[0].reason, "invalid_action: teleport")
        self.assertTrue(result.action_results[1].blocked)
        self.assertEqual(result.action_results[1].reason, "policy_error")
        self.assertEqual([robot.position for robot in result.robots_after], [(1, 1), (2, 1)])

    def test_step_tick_rejects_invalid_robot_state(self) -> None:
        robots = (
            RobotState(0, (1, 1), (3, 0), (3, 3)),
            RobotState(1, (1, 1), (5, 0), (3, 3)),
        )

        with self.assertRaisesRegex(ValueError, "duplicate robot position"):
            step_tick(0, robots, lambda _obs: Action.WAIT, "round-0")

    def test_run_simulation_is_deterministic_and_can_record_ticks(self) -> None:
        def wait_policy(_obs: Observation) -> Action:
            return Action.WAIT

        first = run_simulation("round-0", wait_policy, ticks=3, record_ticks=True)
        second = run_simulation("round-0", wait_policy, ticks=3, record_ticks=True)

        self.assertEqual(first.final_robots, second.final_robots)
        self.assertEqual(len(first.tick_results), 3)
        self.assertEqual(first.ticks, 3)
        self.assertEqual(first.initial_robots, first.final_robots)

    def test_run_simulation_normalizes_seed_to_string(self) -> None:
        result = run_simulation(12345, lambda _obs: Action.WAIT, ticks=0)

        self.assertEqual(result.global_seed, "12345")

    def test_run_simulation_uses_submitted_layout_targets(self) -> None:
        layout = _custom_layout_with_first_shelf_at_one_one()

        result = run_simulation(
            "custom-layout-905",
            lambda _obs: Action.WAIT,
            ticks=0,
            layout=layout,
        )

        self.assertEqual(result.shelf_cells[0], (1, 1))
        self.assertEqual(result.layout["shelf_cells"][0], [1, 1])
        self.assertEqual(result.grid[1][1], CellType.SHELF)
        self.assertEqual(result.initial_robots[0].target_item_position, (1, 1))

    def test_random_policies_never_violate_core_invariants(self) -> None:
        # Property test over the full 96-robot map: under adversarial/random
        # play the simulator must never place two robots on one cell, never let
        # two robots swap, and never raise. This locks in the collision-engine
        # soundness fixpoint.
        actions = list(Action)

        for seed in range(4):
            rng = random.Random(seed)

            def policy(_obs: Observation, _rng: random.Random = rng) -> Action:
                return _rng.choice(actions)

            robots = initial_robot_states(f"invariant-{seed}")
            for tick in range(150):
                result = step_tick(tick, robots, policy, f"invariant-{seed}")
                before = {r.robot_id: r.position for r in result.robots_before}
                after = {r.robot_id: r.position for r in result.robots_after}

                positions = list(after.values())
                self.assertEqual(
                    len(set(positions)), len(positions), "two robots share a cell"
                )
                for left in result.robots_after:
                    for right in result.robots_after:
                        if left.robot_id >= right.robot_id:
                            continue
                        swapped = (
                            after[left.robot_id] == before[right.robot_id]
                            and after[right.robot_id] == before[left.robot_id]
                            and before[left.robot_id] != before[right.robot_id]
                        )
                        self.assertFalse(swapped, "two robots swapped cells")
                robots = result.robots_after

    def test_random_play_is_deterministic(self) -> None:
        actions = list(Action)

        def run_once() -> tuple[tuple[int, int], ...]:
            rng = random.Random(1234)

            def policy(_obs: Observation, _rng: random.Random = rng) -> Action:
                return _rng.choice(actions)

            result = run_simulation("determinism", policy, ticks=120)
            return tuple(robot.position for robot in result.final_robots)

        self.assertEqual(run_once(), run_once())


def _custom_layout_with_first_shelf_at_one_one() -> dict[str, object]:
    shelves = [cell for cell in iter_shelf_cells() if cell != (3, 3)]
    shelves.append((1, 1))
    return validate_submitted_layout(
        {
            "schema_version": LAYOUT_SCHEMA_VERSION,
            "shelves": [list(cell) for cell in shelves],
        }
    )


if __name__ == "__main__":
    unittest.main()
