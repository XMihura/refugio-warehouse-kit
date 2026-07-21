"""Deterministic per-robot target generation."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from warehouse.layout import ROBOT_COUNT, iter_shelf_cells

Position = tuple[int, int]

DEFAULT_SHELF_CELLS: tuple[Position, ...] = tuple(iter_shelf_cells())


def _validate_robot_id(robot_id: int) -> None:
    if not 0 <= robot_id < ROBOT_COUNT:
        raise ValueError(f"robot_id must be in [0, {ROBOT_COUNT - 1}], got {robot_id}")


def _validate_delivery_count(delivery_count: int) -> None:
    if delivery_count < 0:
        raise ValueError(f"delivery_count must be non-negative, got {delivery_count}")


def target_index(
    global_seed: int | str,
    robot_id: int,
    delivery_count: int,
    shelf_count: int = len(DEFAULT_SHELF_CELLS),
) -> int:
    """Return the deterministic shelf index for a robot's next request.

    The generator is counter-based rather than stateful: target `k` for robot
    `r` is `H(seed, r, k) mod |S|`. This makes every target reproducible from
    public inputs and avoids depending on Python's `random` implementation.
    """
    _validate_robot_id(robot_id)
    _validate_delivery_count(delivery_count)
    if shelf_count <= 0:
        raise ValueError("shelf_count must be positive")

    key = f"{global_seed}|{robot_id}|{delivery_count}".encode()
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], byteorder="big") % shelf_count


def target_for(
    global_seed: int | str,
    robot_id: int,
    delivery_count: int,
    shelf_cells: Sequence[Position] = DEFAULT_SHELF_CELLS,
) -> Position:
    if not shelf_cells:
        raise ValueError("shelf_cells must not be empty")
    return shelf_cells[target_index(global_seed, robot_id, delivery_count, len(shelf_cells))]


def initial_targets(
    global_seed: int | str,
    shelf_cells: Sequence[Position] = DEFAULT_SHELF_CELLS,
) -> tuple[Position, ...]:
    return tuple(
        target_for(global_seed, robot_id, 0, shelf_cells)
        for robot_id in range(ROBOT_COUNT)
    )


class TargetGenerator:
    """Small convenience wrapper around the counter-based target function."""

    def __init__(
        self,
        global_seed: int | str,
        shelf_cells: Sequence[Position] = DEFAULT_SHELF_CELLS,
    ) -> None:
        if not shelf_cells:
            raise ValueError("shelf_cells must not be empty")
        self.global_seed = global_seed
        self.shelf_cells = tuple(shelf_cells)

    def target_for(self, robot_id: int, delivery_count: int) -> Position:
        return target_for(self.global_seed, robot_id, delivery_count, self.shelf_cells)

    def initial_targets(self) -> tuple[Position, ...]:
        return tuple(self.target_for(robot_id, 0) for robot_id in range(ROBOT_COUNT))
