"""Example participant submission for the local platform prototype."""

from __future__ import annotations

from warehouse_api import Action, Observation


def create_layout() -> dict[str, object]:
    """Return the canonical shelf layout, generated programmatically."""
    shelves: list[list[int]] = []
    for x0 in range(3, 48, 4):
        for y0, y1 in ((3, 12), (15, 24), (27, 36), (39, 48)):
            for x in (x0, x0 + 1):
                for y in range(y0, y1 + 1):
                    shelves.append([x, y])
    return {"schema_version": 1, "shelves": shelves}


def act(_observation: Observation) -> Action:
    """Minimal valid policy: wait forever."""
    return Action.WAIT

