import json
import unittest
from typing import Any, cast

from warehouse.actions import Action
from warehouse.replay import SCHEMA_VERSION, build_replay, frame_from_robots
from warehouse.state import RobotState


class ReplayTests(unittest.TestCase):
    def test_frame_omits_target_only_while_carrying(self) -> None:
        waiting = RobotState(0, (2, 3), (3, 0), (3, 3))
        carrying = RobotState(1, (3, 2), (5, 0), (3, 3), carrying_item=True)

        frame = frame_from_robots(7, (waiting, carrying))
        robots = cast(list[dict[str, Any]], frame["robots"])

        self.assertEqual(frame["tick"], 7)
        self.assertEqual(robots[0]["target"], [3, 3])
        self.assertNotIn("target", robots[1])

    def test_frame_rejects_negative_tick(self) -> None:
        with self.assertRaisesRegex(ValueError, "tick must be non-negative"):
            frame_from_robots(-1, ())

    def test_build_replay_schema_and_ticks(self) -> None:
        replay = build_replay(12345, lambda _obs: Action.WAIT, ticks=2, name="Demo")

        self.assertEqual(replay["schema_version"], SCHEMA_VERSION)
        self.assertEqual(replay["name"], "Demo")
        self.assertEqual(replay["global_seed"], "12345")
        self.assertEqual(replay["ticks"], 2)
        frames = cast(list[dict[str, Any]], replay["frames"])
        self.assertEqual([frame["tick"] for frame in frames], [0, 1, 2])
        self.assertEqual(len(frames), 3)
        self.assertIn("layout", replay)
        self.assertEqual(replay["total_deliveries"], 0)

    def test_build_replay_is_json_serializable(self) -> None:
        replay = build_replay("round-0", lambda _obs: Action.WAIT, ticks=1, name="Demo")

        encoded = json.dumps(replay, sort_keys=True)

        self.assertIn('"schema_version"', encoded)


if __name__ == "__main__":
    unittest.main()
