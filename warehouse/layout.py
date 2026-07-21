"""Canonical warehouse layout generator.

This module is the source of truth for the static 52x52 challenge map.
It intentionally mirrors the formal specification and the SVG diagram in
``images/warehouse_layout.svg``.
"""

from __future__ import annotations

from collections import deque
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

GRID_SIZE = 52
WALKABLE_MIN = 1
WALKABLE_MAX = 50
WALKABLE_SIZE = 50
ROBOT_COUNT = 96
SHELF_COUNT = 960
LAYOUT_SCHEMA_VERSION = 1

Position = tuple[int, int]

SHELF_COLUMN_STARTS: tuple[int, ...] = tuple(range(3, 48, 4))
SHELF_ROW_BANDS: tuple[tuple[int, int], ...] = (
    (3, 12),
    (15, 24),
    (27, 36),
    (39, 48),
)


class CellType(StrEnum):
    EMPTY = "empty"
    SHELF = "shelf"
    BASE = "base"


CELL_TO_CHAR = {
    CellType.EMPTY: ".",
    CellType.SHELF: "#",
    CellType.BASE: "B",
}

CHAR_TO_CELL = {value: key for key, value in CELL_TO_CHAR.items()}


@dataclass(frozen=True, slots=True)
class BaseCell:
    robot_id: int
    side: str
    position: tuple[int, int]

    def to_json(self) -> dict[str, object]:
        return {
            "robot_id": self.robot_id,
            "side": self.side,
            "position": list(self.position),
        }


class LayoutValidationError(ValueError):
    """Raised when a submitted warehouse layout violates the public contract."""


def in_bounds(position: Position) -> bool:
    x, y = position
    return 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE


def in_walkable_area(position: Position) -> bool:
    x, y = position
    return WALKABLE_MIN <= x <= WALKABLE_MAX and WALKABLE_MIN <= y <= WALKABLE_MAX


def is_walkable_cell(grid: list[list[CellType]], position: Position) -> bool:
    if not in_walkable_area(position):
        return False
    x, y = position
    return grid[y][x] == CellType.EMPTY


def iter_shelf_cells() -> Iterable[Position]:
    """Yield all shelf cells in deterministic row-major block order."""
    for x0 in SHELF_COLUMN_STARTS:
        for y0, y1 in SHELF_ROW_BANDS:
            for x in (x0, x0 + 1):
                for y in range(y0, y1 + 1):
                    yield (x, y)


def iter_base_cells() -> Iterable[BaseCell]:
    """Yield the 96 canonical bases in robot-id order.

    Ordering is side-major and stable: top, bottom, left, right.  Chirality
    offsets break symmetry: bottom +1 right, left +1 down, top +2 right,
    right +2 down.
    """
    robot_id = 0

    for x in range(3, 50, 2):
        yield BaseCell(robot_id, "top", (x, 0))
        robot_id += 1

    for x in range(2, 49, 2):
        yield BaseCell(robot_id, "bottom", (x, 51))
        robot_id += 1

    for y in range(2, 49, 2):
        yield BaseCell(robot_id, "left", (0, y))
        robot_id += 1

    for y in range(3, 50, 2):
        yield BaseCell(robot_id, "right", (51, y))
        robot_id += 1


def fixed_base_cells() -> tuple[BaseCell, ...]:
    """Return the fixed public base cells in robot-id order."""
    return tuple(iter_base_cells())


def base_entry_position(base: BaseCell) -> Position:
    """Return the walkable interior cell adjacent to an external base."""
    x, y = base.position
    if x == 0:
        return (WALKABLE_MIN, y)
    if x == GRID_SIZE - 1:
        return (WALKABLE_MAX, y)
    if y == 0:
        return (x, WALKABLE_MIN)
    if y == GRID_SIZE - 1:
        return (x, WALKABLE_MAX)
    raise ValueError(f"base must be on the external boundary: {base.position}")


def generate_grid() -> list[list[CellType]]:
    return grid_from_shelves(tuple(iter_shelf_cells()))


def grid_from_shelves(
    shelf_cells: Sequence[Position],
) -> list[list[CellType]]:
    grid = [[CellType.EMPTY for _x in range(GRID_SIZE)] for _y in range(GRID_SIZE)]

    for x, y in shelf_cells:
        grid[y][x] = CellType.SHELF

    for base in fixed_base_cells():
        x, y = base.position
        grid[y][x] = CellType.BASE

    return grid


def shelf_cells_from_layout(layout: Mapping[str, object]) -> tuple[Position, ...]:
    """Extract normalized shelf cells from a full layout JSON object."""
    raw_shelves = layout.get("shelf_cells")
    if raw_shelves is None:
        raise LayoutValidationError("layout.shelf_cells must be present")
    return _parse_shelf_cells(raw_shelves, field_name="layout.shelf_cells")


def grid_from_layout(layout: Mapping[str, object]) -> list[list[CellType]]:
    """Build a simulator grid from a normalized full layout JSON object."""
    return grid_from_shelves(shelf_cells_from_layout(layout))


def serialize_grid(grid: list[list[CellType]]) -> list[str]:
    return ["".join(CELL_TO_CHAR[cell] for cell in row) for row in grid]


def build_layout() -> dict[str, object]:
    layout = build_layout_from_shelves(tuple(iter_shelf_cells()), sort_shelves=False)
    layout_parameters = layout["layout_parameters"]
    if isinstance(layout_parameters, dict):
        layout_parameters.update(
            {
                "shelf_column_starts": list(SHELF_COLUMN_STARTS),
                "shelf_row_bands_inclusive": [list(band) for band in SHELF_ROW_BANDS],
            },
        )
    return layout


def build_layout_from_shelves(
    shelf_cells: Sequence[Position],
    *,
    sort_shelves: bool = True,
) -> dict[str, object]:
    normalized_shelves = (
        tuple(sorted(shelf_cells, key=lambda cell: (cell[1], cell[0])))
        if sort_shelves
        else tuple(shelf_cells)
    )
    grid = grid_from_shelves(normalized_shelves)
    bases = fixed_base_cells()

    return {
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "width": GRID_SIZE,
        "height": GRID_SIZE,
        "robot_count": ROBOT_COUNT,
        "cell_encoding": {
            "empty": CELL_TO_CHAR[CellType.EMPTY],
            "shelf": CELL_TO_CHAR[CellType.SHELF],
            "base": CELL_TO_CHAR[CellType.BASE],
        },
        "grid": serialize_grid(grid),
        "bases": [base.to_json() for base in bases],
        "shelf_count": len(normalized_shelves),
        "shelf_cells": [list(position) for position in normalized_shelves],
        "layout_parameters": {
            "base_order": ["top", "bottom", "left", "right"],
            "top_base_x_values": list(range(3, 50, 2)),
            "bottom_base_x_values": list(range(2, 49, 2)),
            "left_base_y_values": list(range(2, 49, 2)),
            "right_base_y_values": list(range(3, 50, 2)),
            "walkable_min": WALKABLE_MIN,
            "walkable_max": WALKABLE_MAX,
            "walkable_width": WALKABLE_SIZE,
            "walkable_height": WALKABLE_SIZE,
        },
    }


def load_submitted_layout(path: str | Path) -> dict[str, object]:
    """Load, validate, and normalize a participant layout JSON file."""
    layout_path = Path(path)
    try:
        payload = json.loads(layout_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LayoutValidationError(f"layout JSON is invalid: {exc}") from exc
    return validate_submitted_layout(payload)


def validate_submitted_layout(payload: object) -> dict[str, object]:
    """Return normalized full layout JSON for a public participant payload.

    Accepted participant shape:

        {"schema_version": 1, "shelves": [[x, y], ...]}

    The returned shape includes fixed bases, serialized grid rows, and sorted
    shelf cells so all downstream systems use one deterministic layout object.
    """
    if not isinstance(payload, dict):
        raise LayoutValidationError("layout must be a JSON object")

    schema_version = payload.get("schema_version")
    if schema_version != LAYOUT_SCHEMA_VERSION:
        raise LayoutValidationError(
            f"layout schema_version must be {LAYOUT_SCHEMA_VERSION}",
        )

    raw_shelves = payload.get("shelves")
    if raw_shelves is None and "shelf_cells" in payload:
        raw_shelves = payload["shelf_cells"]
    shelves = _parse_shelf_cells(raw_shelves)
    _validate_shelf_contract(shelves)
    return build_layout_from_shelves(shelves)


def _parse_shelf_cells(
    raw_shelves: object,
    *,
    field_name: str = "layout.shelves",
) -> tuple[Position, ...]:
    if not isinstance(raw_shelves, (list, tuple)):
        raise LayoutValidationError(f"{field_name} must be a list or tuple")

    shelves: list[Position] = []
    for index, raw_cell in enumerate(raw_shelves):
        if (
            not isinstance(raw_cell, (list, tuple))
            or len(raw_cell) != 2
            or not all(type(value) is int for value in raw_cell)
        ):
            raise LayoutValidationError(
                f"{field_name}[{index}] must be an [x, y] integer pair",
            )
        shelves.append((raw_cell[0], raw_cell[1]))
    return tuple(shelves)


def _validate_shelf_contract(shelves: tuple[Position, ...]) -> None:
    if len(shelves) != SHELF_COUNT:
        raise LayoutValidationError(
            f"layout must contain exactly {SHELF_COUNT} shelves, got {len(shelves)}",
        )

    unique_shelves = set(shelves)
    if len(unique_shelves) != len(shelves):
        raise LayoutValidationError("layout contains duplicate shelf cells")

    for cell in shelves:
        if not in_walkable_area(cell):
            raise LayoutValidationError(
                f"shelf {cell} is outside the walkable area "
                f"[{WALKABLE_MIN}, {WALKABLE_MAX}]",
            )

    base_entries = {base_entry_position(base) for base in fixed_base_cells()}
    blocked_entries = sorted(
        unique_shelves & base_entries,
        key=lambda cell: (cell[1], cell[0]),
    )
    if blocked_entries:
        raise LayoutValidationError(
            f"layout blocks base entry cell {blocked_entries[0]}",
        )

    pickup_blocked = _shelves_without_pickup(unique_shelves)
    if pickup_blocked:
        raise LayoutValidationError(
            f"shelf {pickup_blocked[0]} has no adjacent walkable pickup cell",
        )

    if not _walkable_cells_connected(unique_shelves):
        raise LayoutValidationError("walkable cells must form one connected region")


def _shelves_without_pickup(
    shelf_set: set[Position],
) -> list[Position]:
    blocked: list[Position] = []
    for shelf in sorted(shelf_set, key=lambda cell: (cell[1], cell[0])):
        if not any(
            in_walkable_area(neighbor) and neighbor not in shelf_set
            for neighbor in _adjacent_positions(shelf)
        ):
            blocked.append(shelf)
    return blocked


def _walkable_cells_connected(shelf_set: set[Position]) -> bool:
    all_walkable = {
        (x, y)
        for y in range(WALKABLE_MIN, WALKABLE_MAX + 1)
        for x in range(WALKABLE_MIN, WALKABLE_MAX + 1)
        if (x, y) not in shelf_set
    }
    if not all_walkable:
        return False

    queue: deque[Position] = deque([next(iter(all_walkable))])
    seen = {queue[0]}
    while queue:
        current = queue.popleft()
        for neighbor in _adjacent_positions(current):
            if neighbor in all_walkable and neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return len(seen) == len(all_walkable)


def _adjacent_positions(position: Position) -> tuple[Position, ...]:
    x, y = position
    return ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1))


def write_layout_json(path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(build_layout(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    write_layout_json(Path("data") / "warehouse_layout.json")
