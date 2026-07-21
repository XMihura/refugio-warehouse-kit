import tempfile
import unittest
from pathlib import Path

from warehouse.actions import Action
from warehouse.observation import Observation
from warehouse.submission_loader import (
    SubmissionLoadError,
    SubmissionSetupBudgetExceededError,
    load_submission,
    load_submission_with_layout,
)


class SubmissionLoaderTests(unittest.TestCase):
    def test_loads_valid_act_function(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            submission = Path(temp_dir) / "submission.py"
            submission.write_text(
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

            policy = load_submission(submission)

        self.assertEqual(policy(_dummy_observation()), Action.WAIT)

    def test_rejects_file_without_act(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            submission = Path(temp_dir) / "submission.py"
            submission.write_text("VALUE = 1\n", encoding="utf-8")

            with self.assertRaisesRegex(SubmissionLoadError, "act"):
                load_submission(submission)

    def test_loads_create_layout_function(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            submission = Path(temp_dir) / "submission.py"
            submission.write_text(
                _submission_source_with_layout(),
                encoding="utf-8",
            )

            loaded = load_submission_with_layout(submission)

        self.assertEqual(loaded.act(_dummy_observation()), Action.WAIT)
        self.assertEqual(loaded.layout["shelf_count"], 960)

    def test_rejects_missing_create_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            submission = Path(temp_dir) / "submission.py"
            submission.write_text(
                "\n".join(
                    [
                        "from warehouse_api import Action",
                        "",
                        "def act(_observation):",
                        "    return Action.WAIT",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(SubmissionLoadError, "create_layout"):
                load_submission_with_layout(submission)

    def test_rejects_invalid_create_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            submission = Path(temp_dir) / "submission.py"
            submission.write_text(
                "\n".join(
                    [
                        "from warehouse_api import Action",
                        "",
                        "def create_layout():",
                        "    return {'schema_version': 1, 'shelves': []}",
                        "",
                        "def act(_observation):",
                        "    return Action.WAIT",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(SubmissionLoadError, "invalid layout"):
                load_submission_with_layout(submission)

    def test_rejects_nondeterministic_create_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            submission = Path(temp_dir) / "submission.py"
            submission.write_text(
                _nondeterministic_submission_source(),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(SubmissionLoadError, "same layout"):
                load_submission_with_layout(submission)

    def test_setup_budget_includes_create_layout_time(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            submission = Path(temp_dir) / "submission.py"
            submission.write_text(
                _slow_submission_source(),
                encoding="utf-8",
            )

            with self.assertRaises(SubmissionSetupBudgetExceededError):
                load_submission_with_layout(submission, setup_budget_seconds=0.001)


def _dummy_observation() -> Observation:
    return Observation(
        tick=0,
        robot_id=0,
        position=(1, 1),
        base_position=(3, 0),
        target_item_position=(3, 3),
        carrying_item=False,
        grid=(),
        all_robot_positions={0: (1, 1)},
    )


def _submission_source_with_layout() -> str:
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
                "    return {'schema_version': 1, 'shelves': shelves}",
                "",
                "def act(_observation):",
                "    return Action.WAIT",
            ]
        )
        + "\n"
    )


def _nondeterministic_submission_source() -> str:
    return (
        "\n".join(
            [
                "from warehouse_api import Action",
                "CALLS = 0",
                "",
                "def create_layout():",
                "    global CALLS",
                "    CALLS += 1",
                "    shelves = []",
                "    for x0 in range(3, 48, 4):",
                "        for y0, y1 in ((3, 12), (15, 24), (27, 36), (39, 48)):",
                "            for x in (x0, x0 + 1):",
                "                for y in range(y0, y1 + 1):",
                "                    shelves.append([x, y])",
                "    if CALLS == 2:",
                "        shelves = [cell for cell in shelves if cell != [3, 3]]",
                "        shelves.append([1, 1])",
                "    return {'schema_version': 1, 'shelves': shelves}",
                "",
                "def act(_observation):",
                "    return Action.WAIT",
            ]
        )
        + "\n"
    )


def _slow_submission_source() -> str:
    return (
        "\n".join(
            [
                "from warehouse_api import Action",
                "import time",
                "",
                "def create_layout():",
                "    time.sleep(0.02)",
                "    shelves = []",
                "    for x0 in range(3, 48, 4):",
                "        for y0, y1 in ((3, 12), (15, 24), (27, 36), (39, 48)):",
                "            for x in (x0, x0 + 1):",
                "                for y in range(y0, y1 + 1):",
                "                    shelves.append([x, y])",
                "    return {'schema_version': 1, 'shelves': shelves}",
                "",
                "def act(_observation):",
                "    return Action.WAIT",
            ]
        )
        + "\n"
    )


if __name__ == "__main__":
    unittest.main()

