"""Load participant submissions from standalone Python files."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import sys
from collections.abc import Callable, Iterator, Mapping
from time import perf_counter
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import cast

from warehouse.actions import ActFunction
from warehouse.layout import LayoutValidationError, validate_submitted_layout


class SubmissionLoadError(ValueError):
    """Raised when a submitted Python file cannot provide a usable act()."""


class SubmissionSetupBudgetExceededError(SubmissionLoadError):
    """Raised when import or create_layout() exceeds the setup budget."""

    def __init__(self, spent_seconds: float, budget_seconds: float) -> None:
        self.spent_seconds = spent_seconds
        self.budget_seconds = budget_seconds
        super().__init__(
            "submission setup time budget exceeded "
            f"({spent_seconds:.3f}s > {budget_seconds:.3f}s)",
        )


@dataclass(frozen=True, slots=True)
class LoadedSubmission:
    act: ActFunction
    layout: Mapping[str, object]
    setup_time_seconds: float


def load_submission(path: str | Path) -> ActFunction:
    """Import a submitted `.py` file and return its `act` callable.

    This loader intentionally does not attempt to sandbox Python. The real
    trust boundary is the worker process/container/VM that runs this code.
    """
    submission_path = Path(path).resolve()
    if not submission_path.is_file():
        raise SubmissionLoadError(f"submission file does not exist: {submission_path}")
    if submission_path.suffix != ".py":
        raise SubmissionLoadError("submission must be a single .py file")

    module = _load_module(submission_path)
    act = getattr(module, "act", None)
    if not callable(act):
        raise SubmissionLoadError("submission must define a callable act(observation)")
    return cast(ActFunction, act)


def load_submission_with_layout(
    path: str | Path,
    *,
    setup_budget_seconds: float | None = None,
) -> LoadedSubmission:
    """Import a submitted `.py` file and resolve its policy plus layout."""
    submission_path = Path(path).resolve()
    started = perf_counter()
    module = _load_submission_module(submission_path)
    _check_setup_budget(started, setup_budget_seconds)
    act = _require_callable(module, "act", "submission must define a callable act(observation)")
    create_layout = cast(
        Callable[[], object],
        _require_callable(
            module,
            "create_layout",
            "submission must define a callable create_layout()",
        ),
    )
    layout = _load_and_validate_layout(create_layout)
    _check_setup_budget(started, setup_budget_seconds)
    repeated_layout = _load_and_validate_layout(create_layout)
    _check_setup_budget(started, setup_budget_seconds)
    if repeated_layout != layout:
        raise SubmissionLoadError("create_layout() must return the same layout every time")
    return LoadedSubmission(
        act=cast(ActFunction, act),
        layout=layout,
        setup_time_seconds=perf_counter() - started,
    )


@contextmanager
def sanitized_submission_argv(path: str | Path) -> Iterator[None]:
    """Hide evaluator command-line arguments while participant code runs."""
    original = sys.argv
    sys.argv = [str(Path(path).resolve())]
    try:
        yield
    finally:
        sys.argv = original


def _load_submission_module(path: Path) -> ModuleType:
    if not path.is_file():
        raise SubmissionLoadError(f"submission file does not exist: {path}")
    if path.suffix != ".py":
        raise SubmissionLoadError("submission must be a single .py file")
    return _load_module(path)


def _require_callable(module: ModuleType, name: str, error: str) -> object:
    value = getattr(module, name, None)
    if not callable(value):
        raise SubmissionLoadError(error)
    return value


def _load_and_validate_layout(create_layout: Callable[[], object]) -> Mapping[str, object]:
    try:
        layout_payload = create_layout()
    except Exception as exc:
        raise SubmissionLoadError(f"create_layout() failed: {exc}") from exc
    try:
        return validate_submitted_layout(layout_payload)
    except LayoutValidationError as exc:
        raise SubmissionLoadError(f"create_layout() returned an invalid layout: {exc}") from exc


def _check_setup_budget(started: float, budget_seconds: float | None) -> None:
    if budget_seconds is None:
        return
    spent_seconds = perf_counter() - started
    if spent_seconds > budget_seconds:
        raise SubmissionSetupBudgetExceededError(spent_seconds, budget_seconds)


def _load_module(path: Path) -> ModuleType:
    module_name = _module_name_for_path(path)
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise SubmissionLoadError(f"cannot import submission: {path}")

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise SubmissionLoadError(f"submission import failed: {exc}") from exc
    return module


def _module_name_for_path(path: Path) -> str:
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:16]
    return f"_warehouse_submission_{digest}"

