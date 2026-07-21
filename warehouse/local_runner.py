"""Local submission runner for teams to estimate score and runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from warehouse.evaluation import (
    DEFAULT_EVAL_SEEDS,
    DEFAULT_EVAL_TICKS,
    DEFAULT_POLICY_TIME_BUDGET_SECONDS,
    EvaluationResult,
    evaluate_policy_across_seeds,
)
from warehouse.layout import load_submitted_layout
from warehouse.submission_loader import (
    load_submission,
    load_submission_with_layout,
    sanitized_submission_argv,
)


def run_local(
    submission_path: Path,
    *,
    layout_path: Path | None = None,
    seeds: tuple[str, ...] = DEFAULT_EVAL_SEEDS,
    ticks: int = DEFAULT_EVAL_TICKS,
    policy_budget_seconds: float | None = DEFAULT_POLICY_TIME_BUDGET_SECONDS,
) -> EvaluationResult:
    """Evaluate locally with the same policy-time budget the server enforces."""
    setup_time_seconds = 0.0
    with sanitized_submission_argv(submission_path):
        if layout_path is None:
            loaded_submission = load_submission_with_layout(
                submission_path,
                setup_budget_seconds=policy_budget_seconds,
            )
            policy = loaded_submission.act
            layout = loaded_submission.layout
            setup_time_seconds = loaded_submission.setup_time_seconds
            remaining_policy_budget = _remaining_policy_budget(
                policy_budget_seconds,
                setup_time_seconds,
            )
        else:
            policy = load_submission(submission_path)
            layout = load_submitted_layout(layout_path)
            remaining_policy_budget = policy_budget_seconds
        result, _replay = evaluate_policy_across_seeds(
            policy,
            submission_id="local",
            team_name="local",
            global_seeds=seeds,
            ticks=ticks,
            replay_seed=seeds[0],
            layout=layout,
            policy_time_budget_seconds=remaining_policy_budget,
        )
        _add_setup_timing(result, setup_time_seconds)
    return result


def _remaining_policy_budget(
    policy_budget_seconds: float | None,
    setup_time_seconds: float,
) -> float | None:
    if policy_budget_seconds is None:
        return None
    return max(0.0, policy_budget_seconds - setup_time_seconds)


def _add_setup_timing(result: EvaluationResult, setup_time_seconds: float) -> None:
    if setup_time_seconds <= 0:
        return
    rounded_setup = round(setup_time_seconds, 6)
    result["layout_time_seconds"] = rounded_setup
    result["runtime_seconds"] = round(result["runtime_seconds"] + setup_time_seconds, 6)
    result["policy_time_seconds"] = round(
        result.get("policy_time_seconds", 0.0) + setup_time_seconds,
        6,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a submission locally against representative seeds.",
    )
    parser.add_argument("submission", type=Path)
    parser.add_argument(
        "--layout",
        type=Path,
        help="Optional standalone layout.json override; official submissions use create_layout().",
    )
    parser.add_argument(
        "--seeds",
        default=",".join(DEFAULT_EVAL_SEEDS),
        help="Comma-separated local test seeds.",
    )
    parser.add_argument("--ticks", type=int, default=DEFAULT_EVAL_TICKS)
    parser.add_argument(
        "--policy-budget-seconds",
        type=float,
        default=DEFAULT_POLICY_TIME_BUDGET_SECONDS,
        help="Same act() compute budget the server enforces; <= 0 disables.",
    )
    args = parser.parse_args()

    policy_budget: float | None = args.policy_budget_seconds
    if policy_budget is not None and policy_budget <= 0:
        policy_budget = None

    result = run_local(
        args.submission,
        layout_path=args.layout,
        seeds=parse_seed_arg(args.seeds),
        ticks=args.ticks,
        policy_budget_seconds=policy_budget,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def parse_seed_arg(raw: str) -> tuple[str, ...]:
    seeds = tuple(seed.strip() for seed in raw.split(",") if seed.strip())
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


if __name__ == "__main__":
    main()

