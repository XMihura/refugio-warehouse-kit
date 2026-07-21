import hashlib
import unittest
from dataclasses import replace

from warehouse.actions import Action
from warehouse.baseline_agent import act
from warehouse.layout import generate_grid
from warehouse.observation import Observation, build_observation, freeze_grid
from warehouse.simulation import run_simulation
from warehouse.state import RobotState, initial_robot_states

# Captured from the original (pre-optimization) baseline implementation.
# Guards that performance work never changes the agent's behavior.
GOLDEN_ROUND0_3TICKS_SHA256 = (
    "99a19675c159e7c1bf3b2e7526e25783160a3ebbdf6692325418e35dbcbdd7ef"
)


class BaselineAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = freeze_grid(generate_grid())

    def _observation(self, robot: RobotState) -> Observation:
        return build_observation(0, robot, (robot,), self.grid)

    def _fleet_observation(
        self,
        robot: RobotState,
        robots: tuple[RobotState, ...],
    ) -> Observation:
        return build_observation(0, robot, robots, self.grid)

    def test_pickup_when_adjacent_to_target_shelf(self) -> None:
        robot = RobotState(
            robot_id=0,
            position=(2, 3),
            base_position=(3, 0),
            target_item_position=(3, 3),
        )

        self.assertEqual(act(self._observation(robot)), Action.PICKUP)

    def test_drop_when_carrying_at_base_drop_position(self) -> None:
        robot = replace(initial_robot_states("round-0")[0], carrying_item=True)

        self.assertEqual(act(self._observation(robot)), Action.DROP)

    def test_carrying_robot_moves_toward_own_base_drop_position(self) -> None:
        robot = replace(
            initial_robot_states("round-0")[0],
            position=(3, 3),
            carrying_item=True,
        )

        self.assertEqual(act(self._observation(robot)), Action.UP)

    def test_empty_robot_moves_toward_nearest_pickup_cell(self) -> None:
        robot = RobotState(
            robot_id=0,
            position=(1, 3),
            base_position=(3, 0),
            target_item_position=(3, 3),
        )

        self.assertEqual(act(self._observation(robot)), Action.RIGHT)

    def test_empty_robot_does_not_step_into_shelf(self) -> None:
        robot = RobotState(
            robot_id=0,
            position=(2, 4),
            base_position=(3, 0),
            target_item_position=(3, 3),
        )

        self.assertEqual(act(self._observation(robot)), Action.UP)

    def test_bottom_side_robot_leaves_base_row(self) -> None:
        robot = initial_robot_states("round-0")[24]

        self.assertEqual(act(self._observation(robot)), Action.UP)

    def test_top_side_robot_leaves_base_row_instead_of_sideways_gridlock(self) -> None:
        robots = initial_robot_states("round-0")
        robot = robots[3]

        self.assertEqual(act(self._fleet_observation(robot, robots)), Action.DOWN)

    def test_static_pathfinding_can_move_around_shelf_blocks(self) -> None:
        robot = RobotState(
            robot_id=0,
            position=(5, 4),
            base_position=(3, 0),
            target_item_position=(3, 4),
        )

        self.assertEqual(act(self._observation(robot)), Action.UP)

    def test_action_stream_matches_original_implementation(self) -> None:
        result = run_simulation("round-0", act, ticks=3, record_ticks=True)
        stream = tuple(
            (tick_result.tick, robot_id, tick_result.actions[robot_id].value)
            for tick_result in result.tick_results
            for robot_id in sorted(tick_result.actions)
        )
        digest = hashlib.sha256(repr(stream).encode("utf-8")).hexdigest()

        self.assertEqual(digest, GOLDEN_ROUND0_3TICKS_SHA256)


if __name__ == "__main__":
    unittest.main()
