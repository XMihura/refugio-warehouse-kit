import unittest
from typing import Any, cast

from warehouse.layout import (
    GRID_SIZE,
    LAYOUT_SCHEMA_VERSION,
    ROBOT_COUNT,
    SHELF_COUNT,
    CellType,
    LayoutValidationError,
    base_entry_position,
    build_layout,
    fixed_base_cells,
    generate_grid,
    iter_base_cells,
    iter_shelf_cells,
    load_submitted_layout,
    serialize_grid,
    validate_submitted_layout,
)


class LayoutTests(unittest.TestCase):
    def test_grid_dimensions(self) -> None:
        grid = generate_grid()

        self.assertEqual(len(grid), GRID_SIZE)
        self.assertTrue(all(len(row) == GRID_SIZE for row in grid))

    def test_shelf_geometry(self) -> None:
        shelf_cells = set(iter_shelf_cells())

        self.assertEqual(len(shelf_cells), 12 * 4 * 2 * 10)
        self.assertIn((3, 3), shelf_cells)
        self.assertIn((48, 48), shelf_cells)
        self.assertNotIn((5, 3), shelf_cells)
        self.assertNotIn((3, 13), shelf_cells)
        self.assertNotIn((3, 14), shelf_cells)

    def test_base_count_uniqueness_and_chirality(self) -> None:
        bases = list(iter_base_cells())
        positions = [base.position for base in bases]

        self.assertEqual(len(bases), ROBOT_COUNT)
        self.assertEqual(len(set(positions)), ROBOT_COUNT)

        self.assertEqual(bases[0].side, "top")
        self.assertEqual(bases[0].position, (3, 0))
        self.assertEqual(bases[23].position, (49, 0))

        self.assertEqual(bases[24].side, "bottom")
        self.assertEqual(bases[24].position, (2, 51))
        self.assertEqual(bases[47].position, (48, 51))

        self.assertEqual(bases[48].side, "left")
        self.assertEqual(bases[48].position, (0, 2))
        self.assertEqual(bases[71].position, (0, 48))

        self.assertEqual(bases[72].side, "right")
        self.assertEqual(bases[72].position, (51, 3))
        self.assertEqual(bases[95].position, (51, 49))

    def test_static_cell_types(self) -> None:
        grid = generate_grid()

        self.assertEqual(grid[3][3], CellType.SHELF)
        self.assertEqual(grid[0][3], CellType.BASE)
        self.assertEqual(grid[51][2], CellType.BASE)
        self.assertEqual(grid[2][0], CellType.BASE)
        self.assertEqual(grid[3][51], CellType.BASE)
        self.assertEqual(grid[1][1], CellType.EMPTY)

    def test_serialized_layout_shape(self) -> None:
        layout = build_layout()
        grid_rows = cast(list[str], layout["grid"])
        bases = cast(list[Any], layout["bases"])
        shelf_cells = cast(list[Any], layout["shelf_cells"])

        self.assertEqual(layout["width"], 52)
        self.assertEqual(layout["height"], 52)
        self.assertEqual(layout["robot_count"], 96)
        self.assertEqual(len(bases), 96)
        self.assertEqual(len(shelf_cells), 960)
        self.assertEqual(shelf_cells, [list(position) for position in iter_shelf_cells()])
        self.assertEqual(grid_rows, serialize_grid(generate_grid()))
        self.assertTrue(all(len(row) == 52 for row in grid_rows))

    def test_layout_parameters_match_base_chirality(self) -> None:
        params = cast(dict[str, Any], build_layout()["layout_parameters"])

        self.assertEqual(params["top_base_x_values"], list(range(3, 50, 2)))
        self.assertEqual(params["bottom_base_x_values"], list(range(2, 49, 2)))
        self.assertEqual(params["left_base_y_values"], list(range(2, 49, 2)))
        self.assertEqual(params["right_base_y_values"], list(range(3, 50, 2)))

    def test_submitted_layout_validates_and_normalizes(self) -> None:
        payload = {
            "schema_version": LAYOUT_SCHEMA_VERSION,
            "shelves": [list(cell) for cell in iter_shelf_cells()],
        }

        layout = validate_submitted_layout(payload)
        shelf_cells = cast(list[list[int]], layout["shelf_cells"])

        self.assertEqual(layout["width"], GRID_SIZE)
        self.assertEqual(layout["height"], GRID_SIZE)
        self.assertEqual(layout["robot_count"], ROBOT_COUNT)
        self.assertEqual(layout["shelf_count"], SHELF_COUNT)
        self.assertEqual(len(cast(list[Any], layout["bases"])), ROBOT_COUNT)
        self.assertEqual(len(shelf_cells), SHELF_COUNT)
        self.assertEqual(shelf_cells, sorted(shelf_cells, key=lambda cell: (cell[1], cell[0])))

    def test_submitted_layout_can_be_loaded_from_file(self) -> None:
        from tempfile import TemporaryDirectory
        from pathlib import Path
        import json

        payload = {
            "schema_version": LAYOUT_SCHEMA_VERSION,
            "shelves": [list(cell) for cell in iter_shelf_cells()],
        }

        with TemporaryDirectory() as directory:
            path = Path(directory) / "layout.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            layout = load_submitted_layout(path)

        self.assertEqual(layout["shelf_count"], SHELF_COUNT)

    def test_submitted_layout_rejects_wrong_schema_version(self) -> None:
        payload = {
            "schema_version": 999,
            "shelves": [list(cell) for cell in iter_shelf_cells()],
        }

        with self.assertRaisesRegex(LayoutValidationError, "schema_version"):
            validate_submitted_layout(payload)

    def test_submitted_layout_rejects_wrong_shelf_count(self) -> None:
        payload = {
            "schema_version": LAYOUT_SCHEMA_VERSION,
            "shelves": [list(cell) for cell in list(iter_shelf_cells())[:-1]],
        }

        with self.assertRaisesRegex(LayoutValidationError, "exactly 960"):
            validate_submitted_layout(payload)

    def test_submitted_layout_rejects_duplicate_shelves(self) -> None:
        shelves = list(iter_shelf_cells())
        shelves[-1] = shelves[0]
        payload = {
            "schema_version": LAYOUT_SCHEMA_VERSION,
            "shelves": [list(cell) for cell in shelves],
        }

        with self.assertRaisesRegex(LayoutValidationError, "duplicate"):
            validate_submitted_layout(payload)

    def test_submitted_layout_rejects_out_of_bounds_shelf(self) -> None:
        shelves = list(iter_shelf_cells())
        shelves[0] = (0, 0)
        payload = {
            "schema_version": LAYOUT_SCHEMA_VERSION,
            "shelves": [list(cell) for cell in shelves],
        }

        with self.assertRaisesRegex(LayoutValidationError, "outside"):
            validate_submitted_layout(payload)

    def test_submitted_layout_rejects_boolean_coordinates(self) -> None:
        shelves = [list(cell) for cell in iter_shelf_cells()]
        shelves[0] = [True, 3]
        payload = {
            "schema_version": LAYOUT_SCHEMA_VERSION,
            "shelves": shelves,
        }

        with self.assertRaisesRegex(LayoutValidationError, "integer pair"):
            validate_submitted_layout(payload)

    def test_submitted_layout_accepts_tuple_coordinates(self) -> None:
        payload = {
            "schema_version": LAYOUT_SCHEMA_VERSION,
            "shelves": tuple(iter_shelf_cells()),
        }

        layout = validate_submitted_layout(payload)

        self.assertEqual(layout["shelf_count"], SHELF_COUNT)

    def test_submitted_layout_rejects_blocked_base_entry(self) -> None:
        shelves = list(iter_shelf_cells())
        shelves[0] = base_entry_position(fixed_base_cells()[0])
        payload = {
            "schema_version": LAYOUT_SCHEMA_VERSION,
            "shelves": [list(cell) for cell in shelves],
        }

        with self.assertRaisesRegex(LayoutValidationError, "base entry"):
            validate_submitted_layout(payload)

    def test_submitted_layout_rejects_shelf_without_pickup_cell(self) -> None:
        shelves = set(iter_shelf_cells())
        center = (25, 25)
        required = {center, (24, 25), (26, 25), (25, 24), (25, 26)}
        shelves.update(required)
        for cell in list(shelves):
            if len(shelves) <= SHELF_COUNT:
                break
            if cell not in required:
                shelves.remove(cell)
        payload = {
            "schema_version": LAYOUT_SCHEMA_VERSION,
            "shelves": [list(cell) for cell in shelves],
        }

        with self.assertRaisesRegex(LayoutValidationError, "no adjacent"):
            validate_submitted_layout(payload)

    def test_submitted_layout_rejects_disconnected_walkable_region(self) -> None:
        shelves = {(25, y) for y in range(2, 51)}
        shelves.update({(24, 1), (26, 1)})
        for x in range(3, 50, 2):
            if x == 25:
                continue
            for y in range(2, 51):
                if len(shelves) >= SHELF_COUNT:
                    break
                shelves.add((x, y))
            if len(shelves) >= SHELF_COUNT:
                break
        payload = {
            "schema_version": LAYOUT_SCHEMA_VERSION,
            "shelves": [list(cell) for cell in shelves],
        }

        with self.assertRaisesRegex(LayoutValidationError, "connected"):
            validate_submitted_layout(payload)


if __name__ == "__main__":
    unittest.main()
