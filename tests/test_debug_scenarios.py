import unittest
from typing import Any, cast

from warehouse.debug_scenarios import build_movement_debug_scenarios


class DebugScenarioTests(unittest.TestCase):
    def test_movement_debug_scenarios_have_expected_shape(self) -> None:
        data = build_movement_debug_scenarios()
        scenarios = cast(list[dict[str, Any]], data["scenarios"])

        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["layout_url"], "/data/warehouse_layout.json")
        self.assertGreaterEqual(len(scenarios), 5)

        for scenario in scenarios:
            self.assertIn("id", scenario)
            self.assertIn("title", scenario)
            self.assertIn("description", scenario)
            self.assertTrue(scenario["robots"])
            for robot in cast(list[dict[str, Any]], scenario["robots"]):
                self.assertIn("id", robot)
                self.assertIn("movement", robot)
                self.assertIn("start", robot)
                self.assertIn("intended", robot)
                self.assertIn("final", robot)
                self.assertIn("blocked", robot)

    def test_static_blocking_fixtures_are_present(self) -> None:
        scenarios_list = cast(list[dict[str, Any]], build_movement_debug_scenarios()["scenarios"])
        scenarios = {
            scenario["id"]: scenario
            for scenario in scenarios_list
        }

        base_block = cast(list[dict[str, Any]], scenarios["blocked_by_external_base"]["robots"])[0]
        shelf_block = cast(list[dict[str, Any]], scenarios["blocked_by_shelf"]["robots"])[0]
        valid_move = cast(list[dict[str, Any]], scenarios["valid_outer_corridor_move"]["robots"])[0]

        self.assertTrue(base_block["blocked"])
        self.assertEqual(base_block["reason"], "outside_walkable_area")
        self.assertEqual(base_block["start"], base_block["final"])

        self.assertTrue(shelf_block["blocked"])
        self.assertEqual(shelf_block["reason"], "blocked_by_shelf")
        self.assertEqual(shelf_block["start"], shelf_block["final"])

        self.assertFalse(valid_move["blocked"])
        self.assertNotEqual(valid_move["start"], valid_move["final"])

    def test_collision_and_item_fixtures_are_present(self) -> None:
        scenarios_list = cast(list[dict[str, Any]], build_movement_debug_scenarios()["scenarios"])
        scenarios = {
            scenario["id"]: scenario
            for scenario in scenarios_list
        }

        vertex = cast(list[dict[str, Any]], scenarios["vertex_conflict"]["robots"])
        edge_swap = cast(list[dict[str, Any]], scenarios["edge_swap_conflict"]["robots"])
        pickup = cast(list[dict[str, Any]], scenarios["pickup_adjacent_to_shelf"]["robots"])[0]
        drop = cast(list[dict[str, Any]], scenarios["drop_at_base"]["robots"])[0]

        self.assertTrue(all(robot["blocked"] for robot in vertex))
        self.assertEqual({robot["reason"] for robot in vertex}, {"vertex_conflict"})
        self.assertTrue(all(robot["blocked"] for robot in edge_swap))
        self.assertEqual(
            {robot["reason"] for robot in edge_swap},
            {"edge_swap_conflict"},
        )

        self.assertFalse(pickup["blocked"])
        self.assertEqual(pickup["movement"], "pickup")
        self.assertEqual(pickup["start"], pickup["final"])

        self.assertFalse(drop["blocked"])
        self.assertEqual(drop["movement"], "drop")
        self.assertEqual(drop["start"], drop["final"])


if __name__ == "__main__":
    unittest.main()
