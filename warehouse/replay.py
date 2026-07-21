"""Replay JSON generation helpers."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from warehouse.actions import ActFunction
from warehouse.baseline_agent import act as baseline_act
from warehouse.simulation import run_simulation
from warehouse.state import RobotState

SCHEMA_VERSION = 1


def frame_from_robots(tick: int, robots: Sequence[RobotState]) -> dict[str, object]:
    """Return one JSON-safe replay frame.

    Robot targets are intentionally omitted while carrying because the source
    shelf remains reserved but should no longer be drawn as an active pickup
    target in the viewer.
    """
    if tick < 0:
        raise ValueError(f"tick must be non-negative, got {tick}")

    robot_frames = []
    for robot in robots:
        entry: dict[str, object] = {
            "id": robot.robot_id,
            "pos": list(robot.position),
            "carrying": robot.carrying_item,
            "deliveries": robot.deliveries,
        }
        if not robot.carrying_item:
            entry["target"] = list(robot.target_item_position)
        robot_frames.append(entry)
    return {"tick": tick, "robots": robot_frames}


def build_replay(
    global_seed: int | str,
    policy: ActFunction,
    ticks: int,
    name: str,
) -> dict[str, object]:
    global_seed = str(global_seed)
    result = run_simulation(global_seed, policy, ticks=ticks, record_ticks=True)
    frames = [frame_from_robots(0, result.initial_robots)]
    frames.extend(
        frame_from_robots(tick_result.tick + 1, tick_result.robots_after)
        for tick_result in result.tick_results
    )
    total_deliveries = sum(robot.deliveries for robot in result.final_robots)

    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "global_seed": str(global_seed),
        "ticks": ticks,
        "total_deliveries": total_deliveries,
        "layout": result.layout,
        "frames": frames,
    }


def write_replay(path: str | Path, replay: dict[str, object]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(replay, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_baseline_replay(
    path: str | Path = Path("data") / "baseline_replay.json",
    global_seed: int | str = "baseline-demo",
    ticks: int = 300,
) -> None:
    write_replay(path, build_replay(global_seed, baseline_act, ticks, "Baseline policy"))


if __name__ == "__main__":
    write_baseline_replay()
