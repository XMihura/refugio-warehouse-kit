import json
import tempfile
import unittest
from pathlib import Path
from typing import cast

from warehouse.eval_runner import run_evaluation
from warehouse.layout import LAYOUT_SCHEMA_VERSION, iter_shelf_cells


class EvalRunnerTests(unittest.TestCase):
    def test_run_evaluation_hides_cli_args_from_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            submission_path = directory / "policy.py"
            result_path = directory / "result.json"
            replay_path = directory / "replay.json"

            submission_path.write_text(
                _policy_source_that_checks_argv(),
                encoding="utf-8",
            )

            result = run_evaluation(
                submission_path,
                submission_id="job-argv",
                team_name="Argv Team",
                seeds=("secret-seed",),
                ticks=1,
                replay_seed="secret-seed",
                policy_budget_seconds=None,
                result_out=result_path,
                replay_out=replay_path,
            )

            self.assertEqual(result["status"], "succeeded")

    def test_run_evaluation_uses_create_layout_from_submission(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            submission_path = directory / "policy.py"
            result_path = directory / "result.json"
            replay_path = directory / "replay.json"

            submission_path.write_text(
                _policy_source_with_custom_layout(),
                encoding="utf-8",
            )

            result = run_evaluation(
                submission_path,
                submission_id="job-create-layout",
                team_name="Layout Team",
                seeds=("custom-layout-905",),
                ticks=0,
                replay_seed="custom-layout-905",
                policy_budget_seconds=None,
                result_out=result_path,
                replay_out=replay_path,
            )

            self.assertEqual(result["status"], "succeeded")
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            replay_layout = cast(dict[str, object], replay["layout"])
            self.assertEqual(cast(list[list[int]], replay_layout["shelf_cells"])[0], [1, 1])
            robots = cast(list[dict[str, object]], replay["frames"][0]["robots"])
            self.assertEqual(robots[0]["target"], [1, 1])
            self.assertGreater(result["layout_time_seconds"], 0)
            self.assertGreater(result["policy_time_seconds"], 0)

    def test_run_evaluation_uses_submitted_layout_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            directory = Path(temp_dir)
            submission_path = directory / "policy.py"
            layout_path = directory / "layout.json"
            result_path = directory / "result.json"
            replay_path = directory / "replay.json"

            submission_path.write_text(
                "\n".join(
                    [
                        "from warehouse_api import Action, Observation",
                        "",
                        "def act(_observation: Observation) -> Action:",
                        "    return Action.WAIT",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            layout_path.write_text(
                json.dumps(_custom_layout_payload_with_first_shelf_at_one_one()),
                encoding="utf-8",
            )

            result = run_evaluation(
                submission_path,
                layout_path=layout_path,
                submission_id="job-layout",
                team_name="Layout Team",
                seeds=("custom-layout-905",),
                ticks=0,
                replay_seed="custom-layout-905",
                policy_budget_seconds=None,
                result_out=result_path,
                replay_out=replay_path,
            )

            self.assertEqual(result["status"], "succeeded")
            replay = json.loads(replay_path.read_text(encoding="utf-8"))
            replay_layout = cast(dict[str, object], replay["layout"])
            self.assertEqual(cast(list[list[int]], replay_layout["shelf_cells"])[0], [1, 1])
            robots = cast(list[dict[str, object]], replay["frames"][0]["robots"])
            self.assertEqual(robots[0]["target"], [1, 1])


def _custom_layout_payload_with_first_shelf_at_one_one() -> dict[str, object]:
    shelves = [cell for cell in iter_shelf_cells() if cell != (3, 3)]
    shelves.append((1, 1))
    return {
        "schema_version": LAYOUT_SCHEMA_VERSION,
        "shelves": [list(cell) for cell in shelves],
    }


def _policy_source_with_custom_layout() -> str:
    return (
        "\n".join(
            [
                "from warehouse_api import Action",
                "",
                "def create_layout():",
                "    shelves = []",
                "    for x0 in range(3, 48, 4):",
                "        for y0, y1 in ((3, 12), (15, 24), (27, 36), (39, 48)):",
                "            for x in (x0, x0 + 1):",
                "                for y in range(y0, y1 + 1):",
                "                    shelves.append([x, y])",
                "    shelves = [cell for cell in shelves if cell != [3, 3]]",
                "    shelves.append([1, 1])",
                "    return {'schema_version': 1, 'shelves': shelves}",
                "",
                "def act(_observation):",
                "    return Action.WAIT",
            ]
        )
        + "\n"
    )


def _policy_source_that_checks_argv() -> str:
    return (
        "\n".join(
            [
                "from warehouse_api import Action",
                "import sys",
                "",
                "def _assert_no_secret_argv():",
                "    if any('secret-seed' in arg for arg in sys.argv):",
                "        raise RuntimeError('secret seed leaked through argv')",
                "",
                "def create_layout():",
                "    _assert_no_secret_argv()",
                "    shelves = []",
                "    for x0 in range(3, 48, 4):",
                "        for y0, y1 in ((3, 12), (15, 24), (27, 36), (39, 48)):",
                "            for x in (x0, x0 + 1):",
                "                for y in range(y0, y1 + 1):",
                "                    shelves.append([x, y])",
                "    return {'schema_version': 1, 'shelves': shelves}",
                "",
                "def act(_observation):",
                "    _assert_no_secret_argv()",
                "    return Action.WAIT",
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    unittest.main()
