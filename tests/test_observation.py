import unittest
from dataclasses import FrozenInstanceError

from warehouse.layout import CellType, generate_grid
from warehouse.observation import build_observation, build_observations, freeze_grid
from warehouse.state import initial_robot_states


class ObservationTests(unittest.TestCase):
    def test_freeze_grid_makes_tuple_grid(self) -> None:
        grid = freeze_grid(generate_grid())

        self.assertIsInstance(grid, tuple)
        self.assertIsInstance(grid[0], tuple)
        self.assertEqual(grid[3][3], CellType.SHELF)

    def test_build_observation_contains_allowed_fields(self) -> None:
        robots = initial_robot_states("round-0")
        obs = build_observation(7, robots[0], robots)

        self.assertEqual(obs.tick, 7)
        self.assertEqual(obs.robot_id, robots[0].robot_id)
        self.assertEqual(obs.position, robots[0].position)
        self.assertEqual(obs.base_position, robots[0].base_position)
        self.assertEqual(obs.target_item_position, robots[0].target_item_position)
        self.assertEqual(obs.carrying_item, robots[0].carrying_item)
        self.assertEqual(len(obs.all_robot_positions), len(robots))
        self.assertEqual(obs.all_robot_positions[robots[1].robot_id], robots[1].position)

        self.assertFalse(hasattr(obs, "all_robot_targets"))
        self.assertFalse(hasattr(obs, "all_robot_bases"))
        self.assertFalse(hasattr(obs, "all_robot_carrying"))

    def test_observation_is_frozen_and_position_mapping_is_readonly(self) -> None:
        robots = initial_robot_states("round-0")
        obs = build_observation(0, robots[0], robots)

        with self.assertRaises(FrozenInstanceError):
            obs.tick = 1  # type: ignore[misc]

        with self.assertRaises(TypeError):
            obs.all_robot_positions[0] = (9, 9)  # type: ignore[index]

    def test_build_observations_shares_grid(self) -> None:
        robots = initial_robot_states("round-0")
        observations = build_observations(3, robots)

        self.assertEqual(len(observations), len(robots))
        self.assertEqual(observations[0].tick, 3)
        self.assertIs(observations[0].grid, observations[1].grid)
        self.assertNotEqual(
            observations[0].target_item_position,
            observations[1].target_item_position,
        )

    def test_negative_tick_rejected(self) -> None:
        robots = initial_robot_states("round-0")

        with self.assertRaises(ValueError):
            build_observation(-1, robots[0], robots)


if __name__ == "__main__":
    unittest.main()
