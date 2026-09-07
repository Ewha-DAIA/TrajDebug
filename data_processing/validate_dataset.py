"""Validate detector-ready unified trajectories from a registry key or path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_processing.dataset_registry import dataset_path
from data_processing.schema import validate_unified


def validate_directory(path: Path) -> int:
    files = sorted(path.glob("*.json"))
    if not files:
        raise ValueError(f"no JSON trajectories found in {path}")
    task_ids: set[str] = set()
    for file_path in files:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
        validate_unified(payload)
        task_id = payload["metadata"]["task_id"]
        if task_id in task_ids:
            raise ValueError(f"duplicate metadata.task_id {task_id!r} in {path}")
        task_ids.add(task_id)
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dataset", help="registered dataset key")
    group.add_argument("--path", type=Path, help="trajectory directory")
    args = parser.parse_args()

    path = dataset_path(args.dataset) if args.dataset else args.path.resolve()
    count = validate_directory(path)
    print(f"validated {count} trajectories: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
