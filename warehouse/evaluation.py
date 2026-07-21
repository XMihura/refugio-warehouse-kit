"""Evaluation helpers for scoring a submitted warehouse policy."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import NotRequired, TypedDict

from warehouse.actions import ActFunction, Action, PolicyTimeBudgetExceededError
from warehouse.layout import CellType, generate_grid, is_walkable_cell
from warehouse.observation import Observation
from warehouse.replay import SCHEMA_VERSION, frame_from_robots
from warehouse.simulation import SimulationResult, run_simulation
from warehouse.state import Position, RobotState, drop_position_for_base

SCORING_SCHEMA_VERSION = 1
DEFAULT_EVAL_SEED = "round-0"
DEFAULT_EVAL_SEEDS = ("round-0", "round-1", "round-2")
DEFAULT_EVAL_TICKS = 300

# Total compute budget for the team's act() calls, summed across every seed of
# one evaluation. The simulator's own bookkeeping does not count against it.
DEFAULT_POLICY_TIME_BUDGET_SECONDS = 180.0

BLOCKABLE_MOVEMENT_ACTIONS = frozenset(
    {
        Action.UP,
        Action.DOWN,
        Action.LEFT,
        Action.RIGHT,
    }
)


class ScoreBreakdown(TypedDict):
    deliveries: int
    blocked_moves: int
    remaining_distance: int


class SeedEvaluationResult(TypedDict):
    global_seed: str
    score: int
    score_breakdown: ScoreBreakdown
    runtime_seconds: float
    policy_time_seconds: NotRequired[float]
    replay_saved: bool


class EvaluationResult(TypedDict):
    schema_version: int
    submission_id: str
    team_name: str
    status: str
    global_seeds: list[str]
    replay_seed: str
    ticks: int
    score: int
    score_breakdown: ScoreBreakdown
    runtime_seconds: float
    policy_time_seconds: NotRequired[float]
    layout_time_seconds: NotRequired[float]
    seed_results: list[SeedEvaluationResult]
    error: NotRequired[str]


class _TimedPolicy:
    """Wrap a policy to meter cumulative act() time and enforce a budget.

    The check runs after each call returns, so it cannot interrupt a single
    call that never returns (e.g. an infinite loop). That case is handled by
    the hard wall-clock kill in the eval worker, outside this interpreter.
    """

    __slots__ = ("_policy", "_budget_seconds", "spent_seconds")

    def __init__(self, policy: ActFunction, budget_seconds: float | None) -> None:
        self._policy = policy
        self._budget_seconds = budget_seconds
        self.spent_seconds = 0.0

    def __call__(self, observation: Observation, /) -> Action:
        started = perf_counter()
        try:
            return self._policy(observation)
        finally:
            self.spent_seconds += perf_counter() - started
            if (
                self._budget_seconds is not None
                and self.spent_seconds > self._budget_seconds
            ):
                raise PolicyTimeBudgetExceededError(
                    self.spent_seconds, self._budget_seconds
                )


def evaluate_policy(
    policy: ActFunction,
    *,
    submission_id: str,
    team_name: str,
    global_seed: int | str = DEFAULT_EVAL_SEED,
    ticks: int = DEFAULT_EVAL_TICKS,
    layout: Mapping[str, object] | None = None,
    policy_time_budget_seconds: float | None = None,
) -> tuple[EvaluationResult, dict[str, object] | None]:
    """Run a policy for one seed and return a compact result plus replay JSON."""
    return evaluate_policy_across_seeds(
        policy,
        submission_id=submission_id,
        team_name=team_name,
        global_seeds=(str(global_seed),),
        ticks=ticks,
        replay_seed=str(global_seed),
        layout=layout,
        policy_time_budget_seconds=policy_time_budget_seconds,
    )


def evaluate_policy_across_seeds(
    policy: ActFunction,
    *,
    submission_id: str,
    team_name: str,
    global_seeds: Sequence[int | str] = DEFAULT_EVAL_SEEDS,
    ticks: int = DEFAULT_EVAL_TICKS,
    replay_seed: int | str | None = None,
    layout: Mapping[str, object] | None = None,
    policy_time_budget_seconds: float | None = None,
) -> tuple[EvaluationResult, dict[str, object] | None]:
    """Evaluate one policy across hidden seeds, saving one representative replay.

    The budget meters only the time spent inside the team's act() calls,
    summed across all seeds. When it is exhausted the evaluation stops and the
    result is returned with status ``timed_out`` and score 0. The replay is
    ``None`` unless the replay seed had already completed.
    """
    seeds = normalize_seeds(global_seeds)
    selected_replay_seed = str(replay_seed) if replay_seed is not None else seeds[0]
    if selected_replay_seed not in seeds:
        raise ValueError("replay_seed must be one of the evaluation seeds")

    timed_policy = _TimedPolicy(policy, policy_time_budget_seconds)
    total_runtime = 0.0
    total_breakdown: ScoreBreakdown = {
        "deliveries": 0,
        "blocked_moves": 0,
        "remaining_distance": 0,
    }
    seed_results: list[SeedEvaluationResult] = []
    replay: dict[str, object] | None = None

    for seed in seeds:
        seed_started = perf_counter()
        policy_time_before = timed_policy.spent_seconds
        try:
            seed_result, seed_replay = evaluate_single_seed(
                timed_policy,
                submission_id=submission_id,
                team_name=team_name,
                global_seed=seed,
                ticks=ticks,
                layout=layout,
                save_replay=seed == selected_replay_seed,
            )
        except PolicyTimeBudgetExceededError as exc:
            total_runtime += perf_counter() - seed_started
            result = build_failed_result(
                submission_id=submission_id,
                team_name=team_name,
                seeds=seeds,
                replay_seed=selected_replay_seed,
                ticks=ticks,
                status="timed_out",
                error=(
                    f"policy time budget exceeded during seed '{seed}' "
                    f"({len(seed_results)}/{len(seeds)} seeds completed): "
                    f"{exc.spent_seconds:.3f}s > {exc.budget_seconds:.3f}s"
                ),
            )
            result["runtime_seconds"] = round(total_runtime, 6)
            result["policy_time_seconds"] = round(timed_policy.spent_seconds, 6)
            result["seed_results"] = seed_results
            return result, replay

        seed_result["policy_time_seconds"] = round(
            timed_policy.spent_seconds - policy_time_before, 6
        )
        total_runtime += seed_result["runtime_seconds"]
        total_breakdown["deliveries"] += seed_result["score_breakdown"]["deliveries"]
        total_breakdown["blocked_moves"] += seed_result["score_breakdown"]["blocked_moves"]
        total_breakdown["remaining_distance"] += seed_result["score_breakdown"][
            "remaining_distance"
        ]
        seed_results.append(seed_result)
        if seed_replay is not None:
            replay = seed_replay

    result = {
        "schema_version": SCORING_SCHEMA_VERSION,
        "submission_id": submission_id,
        "team_name": team_name,
        "status": "succeeded",
        "global_seeds": seeds,
        "replay_seed": selected_replay_seed,
        "ticks": ticks,
        "score": total_breakdown["deliveries"],
        "score_breakdown": total_breakdown,
        "runtime_seconds": round(total_runtime, 6),
        "policy_time_seconds": round(timed_policy.spent_seconds, 6),
        "seed_results": seed_results,
    }
    return result, replay


def build_failed_result(
    *,
    submission_id: str,
    team_name: str,
    seeds: Sequence[str],
    replay_seed: str,
    ticks: int,
    error: str,
    status: str = "failed",
) -> EvaluationResult:
    """Build a zero-score result for failed/timed-out/rejected evaluations."""
    return {
        "schema_version": SCORING_SCHEMA_VERSION,
        "submission_id": submission_id,
        "team_name": team_name,
        "status": status,
        "global_seeds": list(seeds),
        "replay_seed": replay_seed,
        "ticks": ticks,
        "score": 0,
        "score_breakdown": {
            "deliveries": 0,
            "blocked_moves": 0,
            "remaining_distance": 0,
        },
        "runtime_seconds": 0.0,
        "seed_results": [],
        "error": error,
    }


def evaluate_single_seed(
    policy: ActFunction,
    *,
    submission_id: str,
    team_name: str,
    global_seed: str,
    ticks: int,
    layout: Mapping[str, object] | None,
    save_replay: bool,
) -> tuple[SeedEvaluationResult, dict[str, object] | None]:
    started = perf_counter()
    simulation = run_simulation(
        global_seed,
        policy,
        ticks=ticks,
        layout=layout,
        record_ticks=True,
    )
    runtime_seconds = perf_counter() - started
    score_breakdown = score_simulation(simulation)
    score = score_breakdown["deliveries"]

    result: SeedEvaluationResult = {
        "global_seed": str(global_seed),
        "score": score,
        "score_breakdown": score_breakdown,
        "runtime_seconds": round(runtime_seconds, 6),
        "replay_saved": save_replay,
    }
    replay = (
        replay_from_simulation(
            simulation,
            name=f"{team_name} / {submission_id} / {global_seed}",
        )
        if save_replay
        else None
    )
    return result, replay


def normalize_seeds(global_seeds: Sequence[int | str]) -> list[str]:
    seeds = [str(seed) for seed in global_seeds]
    if not seeds:
        raise ValueError("at least one evaluation seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("evaluation seeds must be unique")
    return seeds


def score_simulation(simulation: SimulationResult) -> ScoreBreakdown:
    """Compute the primary score and first tie-breakers from a simulation."""
    deliveries = sum(robot.deliveries for robot in simulation.final_robots)
    blocked_moves = sum(
        1
        for tick in simulation.tick_results
        for action_result in tick.action_results.values()
        if action_result.blocked and action_result.action in BLOCKABLE_MOVEMENT_ACTIONS
    )
    remaining_distance = sum_remaining_distance(simulation.final_robots, simulation.grid)
    return {
        "deliveries": deliveries,
        "blocked_moves": blocked_moves,
        "remaining_distance": remaining_distance,
    }


def replay_from_simulation(
    simulation: SimulationResult,
    *,
    name: str,
) -> dict[str, object]:
    """Build replay JSON from an already-executed simulation result."""
    frames = [frame_from_robots(0, simulation.initial_robots)]
    frames.extend(
        frame_from_robots(tick_result.tick + 1, tick_result.robots_after)
        for tick_result in simulation.tick_results
    )
    total_deliveries = sum(robot.deliveries for robot in simulation.final_robots)
    return {
        "schema_version": SCHEMA_VERSION,
        "name": name,
        "global_seed": str(simulation.global_seed),
        "ticks": simulation.ticks,
        "total_deliveries": total_deliveries,
        "layout": simulation.layout,
        "frames": frames,
    }


def sum_remaining_distance(
    robots: tuple[RobotState, ...],
    grid: list[list[CellType]] | None = None,
) -> int:
    grid = grid if grid is not None else generate_grid()
    return sum(remaining_distance(robot, grid) for robot in robots)


def remaining_distance(robot: RobotState, grid: list[list[CellType]]) -> int:
    """Distance to the next useful interaction cell for tie-breaking."""
    if robot.carrying_item:
        return manhattan(robot.position, drop_position_for_base(robot.base_position))

    pickup_positions = valid_pickup_positions(robot.target_item_position, grid)
    if not pickup_positions:
        return 0
    return min(manhattan(robot.position, position) for position in pickup_positions)


def valid_pickup_positions(
    shelf_position: Position,
    grid: list[list[CellType]],
) -> tuple[Position, ...]:
    return tuple(
        position
        for position in adjacent_positions(shelf_position)
        if is_walkable_cell(grid, position)
    )


def adjacent_positions(position: Position) -> tuple[Position, Position, Position, Position]:
    x, y = position
    return ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1))


def manhattan(left: Position, right: Position) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def summarize_result_for_leaderboard(
    result: EvaluationResult,
    *,
    result_url: str,
    replay_url: str,
    code_url: str,
    updated_at: str,
    rank: int = 0,
) -> Mapping[str, object]:
    """Return a JSON-safe leaderboard row for a succeeded evaluation."""
    breakdown = result["score_breakdown"]
    return {
        "rank": rank,
        "team_name": result["team_name"],
        "job_id": result["submission_id"],
        "status": result["status"],
        "score": result["score"],
        "deliveries": breakdown["deliveries"],
        "blocked_moves": breakdown["blocked_moves"],
        "remaining_distance": breakdown["remaining_distance"],
        "runtime_seconds": result["runtime_seconds"],
        "result_url": result_url,
        "replay_url": replay_url,
        "code_url": code_url,
        "updated_at": updated_at,
    }

