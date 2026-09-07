"""Canonical dataset keys and detector-ready trajectory locations.

The public data layout groups related subsets by benchmark identity, while
pipeline outputs need flat, filesystem-safe keys.  This registry is the single
source of truth connecting those two views.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    profile: str
    released: bool = True

    def absolute_path(self, root: Path = REPO_ROOT) -> Path:
        return (root / self.path).resolve()


DATASETS: Dict[str, DatasetSpec] = {
    "trajerrbench_zh_tau2bench": DatasetSpec(
        "data/trajerrbench/zh/tau2bench", "tau2bench"
    ),
    "trajerrbench_zh_swebenchpro": DatasetSpec(
        "data/trajerrbench/zh/swebenchpro", "swebenchpro"
    ),
    "trajerrbench_en_tau2bench": DatasetSpec(
        "data/trajerrbench/en/tau2bench", "tau2bench"
    ),
    "trajerrbench_en_swebenchpro": DatasetSpec(
        "data/trajerrbench/en/swebenchpro", "swebenchpro"
    ),
    "agentdebugbench_alfworld": DatasetSpec(
        "data/agentdebugbench/alfworld", "alfworld"
    ),
    "agentdebugbench_gaia": DatasetSpec("data/agentdebugbench/gaia", "gaia"),
    "agentdebugbench_webshop": DatasetSpec(
        "data/agentdebugbench/webshop", "webshop"
    ),
    "whoandwhen_handcrafted": DatasetSpec(
        "data/whoandwhen/handcrafted", "whoandwhen"
    ),
    "whoandwhen_algorithm_generated": DatasetSpec(
        "data/whoandwhen/algorithm_generated", "whoandwhen_algorithm"
    ),
}

# Keep old commands useful without retaining duplicate data directories.
ALIASES = {
    "tau2bench": "trajerrbench_zh_tau2bench",
    "swebenchpro": "trajerrbench_zh_swebenchpro",
    "alfworld": "agentdebugbench_alfworld",
    "gaia": "agentdebugbench_gaia",
    "webshop": "agentdebugbench_webshop",
    "whoandwhen": "whoandwhen_handcrafted",
    "whoandwhen_algorithm": "whoandwhen_algorithm_generated",
}


def canonical_key(key: str) -> str:
    resolved = ALIASES.get(key, key)
    if resolved not in DATASETS:
        choices = ", ".join(sorted(DATASETS))
        raise KeyError(f"unknown dataset {key!r}; choose one of: {choices}")
    return resolved


def get_dataset(key: str) -> DatasetSpec:
    return DATASETS[canonical_key(key)]


def dataset_path(key: str, root: Path = REPO_ROOT) -> Path:
    return get_dataset(key).absolute_path(root)


def available_datasets(root: Path = REPO_ROOT) -> Iterable[str]:
    return (
        key for key, spec in DATASETS.items() if spec.absolute_path(root).is_dir()
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Resolve a TRAJDEBUG dataset key.")
    parser.add_argument("dataset", nargs="?")
    parser.add_argument(
        "--field", choices=("key", "path", "profile", "json"), default="path"
    )
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    if args.list:
        for key in available_datasets():
            print(key)
        return 0
    if not args.dataset:
        parser.error("dataset is required unless --list is used")

    try:
        key = canonical_key(args.dataset)
    except KeyError as exc:
        parser.error(str(exc))
    spec = DATASETS[key]
    if args.field == "key":
        print(key)
    elif args.field == "path":
        print(spec.absolute_path())
    elif args.field == "profile":
        print(spec.profile)
    else:
        print(
            json.dumps(
                {
                    "key": key,
                    "path": str(spec.absolute_path()),
                    "profile": spec.profile,
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
