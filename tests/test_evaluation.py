import json
import time
import unittest
from typing import cast

from warehouse.actions import Action
from warehouse.evaluation import (
    evaluate_policy,
    evaluate_policy_across_seeds,
    score_simulation,
)
from warehouse.layout import LAYOUT_SCHEMA_VERSION, iter_shelf_cells, validate_submitted_layout
from warehouse.observation import Observation
from warehouse.simulation import run_simulation


class EvaluationTests(unittest.TestCase):
    def test_evaluate_policy_returns_result_and_replay(self) -> None:
        result, replay = evaluate_policy(
            _wait_policy,
            submission_id="job-1",
            team_name="Team One",
            global_seed="eval-test",
            ticks=2,
        )

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(result["submission_id"], "job-1")
        self.assertEqual(result["team_name"], "Team One")
        self.assertEqual(result["status"], "succeeded")
        self.assertEqual(result["global_seeds"], ["eval-test"])
        self.assertEqual(result["replay_seed"], "eval-test")
        self.assertEqual(result["score"], 0)
        self.assertEqual(result["score_breakdown"]["deliveries"], 0)
        self.assertEqual(len(result["seed_results"]), 1)
        self.assertIn("runtime_seconds", result)
        self.assertIn("policy_time_seconds", result)
        self.assertIn("policy_time_seconds", result["seed_results"][0])
        assert replay is not None
        self.assertEqual(replay["schema_version"], 1)
        self.assertEqual(replay["global_seed"], "eval-test")
        self.assertEqual(replay["ticks"], 2)
        frames = cast(list[object], replay["frames"])
        self.assertEqual(len(frames), 3)
        json.dumps(result, sort_keys=True)
        json.dumps(replay, sort_keys=True)

    def test_policy_time_budget_marks_timed_out(self) -> None:
        def slow_policy(_observation: Observation) -> Action:
            time.sleep(0.005)
            return Action.WAIT

        result, replay = evaluate_policy_across_seeds(
            slow_policy,
            submission_id="job-slow",
            team_name="Team Slow",
            global_seeds=("budget-test",),
            ticks=5,
            replay_seed="budget-test",
            policy_time_budget_seconds=0.02,
        )

        self.assertEqual(result["status"], "timed_out")
        self.assertEqual(result["score"], 0)
        self.assertIn("policy time budget exceeded", result["error"])
        self.assertGreater(result["policy_time_seconds"], 0.02)
        self.assertIsNone(replay)
        json.dumps(result, sort_keys=True)

    def test_budget_keeps_replay_from_completed_replay_seed(self) -> None:
        calls = {"count": 0}

        def degrading_policy(_observation: Observation) -> Action:
            calls["count"] += 1
            if calls["count"] > 96 * 2:
                time.sleep(0.01)
            return Action.WAIT

        result, replay = evaluate_policy_across_seeds(
            degrading_policy,
            submission_id="job-degrade",
            team_name="Team Degrade",
            global_seeds=("seed-a", "seed-b"),
            ticks=2,
            replay_seed="seed-a",
            policy_time_budget_seconds=0.05,
        )

        self.assertEqual(result["status"], "timed_out")
        self.assertEqual(len(result["seed_results"]), 1)
        self.assertIsNotNone(replay)

    def test_score_simulation_is_deterministic(self) -> None:
        first = run_simulation("score-test", _wait_policy, ticks=2, record_ticks=True)
        second = run_simulation("score-test", _wait_policy, ticks=2, record_ticks=True)

        self.assertEqual(score_simulation(first), score_simulation(second))

    def test_evaluation_replay_uses_submitted_layout(self) -> None:
        layout = _custom_layout_with_first_shelf_at_one_one()

        result, replay = evaluate_policy(
            _wait_policy,
            submission_id="job-layout",
            team_name="Layout Team",
            global_seed="custom-layout-905",
            ticks=0,
            layout=layout,
        )

        self.assertEqual(result["status"], "succeeded")
        assert replay is not None
        replay_layout = cast(dict[str, object], replay["layout"])
        self.assertEqual(cast(list[list[int]], replay_layout["shelf_cells"])[0], [1, 1])
        frames = cast(list[dict[str, object]], replay["frames"])
        robots = cast(list[dict[str, object]], frames[0]["robots"])
        self.assertEqual(robots[0]["target"], [1, 1])


def _wait_policy(_observation: Observation) -> Action:
    return Action.WAIT


def _custom_layout_with_first_shelf_at_one_one() -> dict[str, object]:
    shelves = [cell for cell in iter_shelf_cells() if cell != (3, 3)]
    shelves.append((1, 1))
    return validate_submitted_layout(
        {
            "schema_version": LAYOUT_SCHEMA_VERSION,
            "shelves": [list(cell) for cell in shelves],
        }
    )


if __name__ == "__main__":
    unittest.main()

