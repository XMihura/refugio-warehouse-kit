"""CLI for validating participant warehouse layouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import cast

from warehouse.layout import LayoutValidationError, load_submitted_layout


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a REFUGIO layout.json file.")
    parser.add_argument("layout", type=Path)
    parser.add_argument(
        "--normalized-out",
        type=Path,
        help="Optional path to write the normalized full layout JSON.",
    )
    args = parser.parse_args()

    try:
        normalized = load_submitted_layout(args.layout)
    except LayoutValidationError as exc:
        raise SystemExit(f"INVALID: {exc}") from exc

    if args.normalized_out:
        args.normalized_out.parent.mkdir(parents=True, exist_ok=True)
        args.normalized_out.write_text(
            json.dumps(normalized, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    shelf_count = cast(int, normalized["shelf_count"])
    width = cast(int, normalized["width"])
    height = cast(int, normalized["height"])
    walkable_count = width * height - shelf_count
    print(f"VALID: {shelf_count} shelves, {walkable_count} non-shelf cells")


if __name__ == "__main__":
    main()
