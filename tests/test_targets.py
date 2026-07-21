import unittest

from warehouse.layout import ROBOT_COUNT, CellType, generate_grid, iter_shelf_cells
from warehouse.targets import TargetGenerator, initial_targets, target_for, target_index


class TargetGeneratorTests(unittest.TestCase):
    def test_known_target_fixtures(self) -> None:
        self.assertEqual(target_index("official-seed", 0, 0), 330)
        self.assertEqual(target_for("official-seed", 0, 0), (20, 3))
        self.assertEqual(target_for("official-seed", 0, 1), (11, 35))
        self.assertEqual(target_for("official-seed", 95, 0), (27, 39))
        self.assertEqual(target_for(12345, 42, 7), (4, 21))

    def test_initial_targets_are_reproducible_and_shelf_cells(self) -> None:
        grid = generate_grid()
        first = initial_targets("round-0")
        second = initial_targets("round-0")

        self.assertEqual(first, second)
        self.assertEqual(len(first), ROBOT_COUNT)
        for x, y in first:
            self.assertEqual(grid[y][x], CellType.SHELF)

    def test_initial_targets_can_use_submitted_shelf_order(self) -> None:
        shelves = sorted(
            [cell for cell in iter_shelf_cells() if cell != (3, 3)] + [(1, 1)],
            key=lambda cell: (cell[1], cell[0]),
        )

        self.assertEqual(target_index("custom-layout-905", 0, 0), 0)
        self.assertEqual(initial_targets("custom-layout-905", shelves)[0], (1, 1))

    def test_wrapper_matches_function_api(self) -> None:
        generator = TargetGenerator("round-1")

        self.assertEqual(generator.target_for(17, 3), target_for("round-1", 17, 3))
        self.assertEqual(generator.initial_targets(), initial_targets("round-1"))

    def test_invalid_inputs(self) -> None:
        with self.assertRaises(ValueError):
            target_for("seed", -1, 0)
        with self.assertRaises(ValueError):
            target_for("seed", ROBOT_COUNT, 0)
        with self.assertRaises(ValueError):
            target_for("seed", 0, -1)


if __name__ == "__main__":
    unittest.main()
