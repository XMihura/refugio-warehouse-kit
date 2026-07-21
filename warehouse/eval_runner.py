"""Isolated single-evaluation runner: the process the eval worker spawns.

This CLI is the trust/isolation boundary. It imports untrusted submission
code, so production infra should run it inside a constrained environment
(container/VM: no network, read-only repo, memory/CPU limits) and enforce a
hard wall-clock kill from outside — the worker already does the kill part.

The runner always tries to leave a result JSON behind (status ``succeeded``,
``timed_out`` or ``failed``) and exits 0 for handled outcomes. A non-zero exit
means something unexpected happened and the parent should synthesize a failed
result from stderr.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from warehouse.evaluation import (
    DEFAULT_POLICY_TIME_BUDGET_SECONDS,
    EvaluationResult,
    build_failed_result,
    evaluate_policy_across_seeds,
)
from warehouse.layout import load_submitted_layout
from warehouse.submission_loader import (
    SubmissionSetupBudgetExceededError,
    load_submission,
    load_submission_with_layout,
    sanitized_submission_argv,
)


def run_evaluation(
    submission_path: Path,
    *,
    layout_path: Path | None = None,
    submission_id: str,
    team_name: str,
    seeds: tuple[str, ...],
    ticks: int,
    replay_seed: str,
    policy_budget_seconds: float | None,
    result_out: Path,
    replay_out: Path,
) -> EvaluationResult:
    """Evaluate one submission and write result/replay artifacts."""
    replay: dict[str, object] | None = None
    setup_time_seconds = 0.0
    try:
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
            result, replay = evaluate_policy_across_seeds(
                policy,
                submission_id=submission_id,
                team_name=team_name,
                global_seeds=seeds,
                ticks=ticks,
                replay_seed=replay_seed,
                layout=layout,
                policy_time_budget_seconds=remaining_policy_budget,
            )
            _add_setup_timing(result, setup_time_seconds)
    except SubmissionSetupBudgetExceededError as exc:
        result = build_failed_result(
            submission_id=submission_id,
            team_name=team_name,
            seeds=seeds,
            replay_seed=replay_seed,
            ticks=ticks,
            status="timed_out",
            error=str(exc),
        )
        _add_setup_timing(result, exc.spent_seconds)
    except Exception as exc:
        result = build_failed_result(
            submission_id=submission_id,
            team_name=team_name,
            seeds=seeds,
            replay_seed=replay_seed,
            ticks=ticks,
            error=str(exc),
        )

    atomic_write_json(result_out, result)
    if replay is not None:
        atomic_write_json(replay_out, replay)
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


def atomic_write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=path.parent,
        delete=False,
    ) as temp_file:
        temp_file.write(encoded)
        temp_name = temp_file.name
    os.replace(temp_name, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one isolated submission evaluation.")
    parser.add_argument("submission", type=Path)
    parser.add_argument(
        "--layout",
        type=Path,
        help="Optional standalone layout.json override; official submissions use create_layout().",
    )
    parser.add_argument("--submission-id", required=True)
    parser.add_argument("--team-name", required=True)
    parser.add_argument("--seeds", required=True, help="Comma-separated evaluation seeds.")
    parser.add_argument("--ticks", type=int, required=True)
    parser.add_argument("--replay-seed", required=True)
    parser.add_argument(
        "--policy-budget-seconds",
        type=float,
        default=DEFAULT_POLICY_TIME_BUDGET_SECONDS,
        help="Cumulative act() compute budget across all seeds; <= 0 disables.",
    )
    parser.add_argument("--result-out", type=Path, required=True)
    parser.add_argument("--replay-out", type=Path, required=True)
    args = parser.parse_args()

    seeds = tuple(seed.strip() for seed in args.seeds.split(",") if seed.strip())
    if not seeds:
        raise SystemExit("at least one seed is required")
    policy_budget: float | None = args.policy_budget_seconds
    if policy_budget is not None and policy_budget <= 0:
        policy_budget = None

    result = run_evaluation(
        args.submission,
        layout_path=args.layout,
        submission_id=args.submission_id,
        team_name=args.team_name,
        seeds=seeds,
        ticks=args.ticks,
        replay_seed=args.replay_seed,
        policy_budget_seconds=policy_budget,
        result_out=args.result_out,
        replay_out=args.replay_out,
    )
    print(f"{result['submission_id']}: {result['status']}")


if __name__ == "__main__":
    main()
