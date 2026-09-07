"""CLI: convert raw datasets into the unified schema.

Usage examples::

    python -m data_processing.build_unified_dataset --all
    python -m data_processing.build_unified_dataset --dataset trajerrbench_zh_tau2bench
    python -m data_processing.build_unified_dataset \
        --dataset trajerrbench_zh_tau2bench --src /path/to/tau2bench \
        --out data/generated/trajerrbench_zh_tau2bench

The CLI is idempotent: it re-emits one JSON per source trajectory and
overwrites files in its generated output directory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from data_processing import (
    alfworld,
    gaia,
    swebenchpro,
    tau2bench,
    webshop,
    whoandwhen,
)
from data_processing.schema import validate_unified


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_SOURCE_DIR = DEFAULT_DATA_DIR / "sources"
DEFAULT_GENERATED_DIR = DEFAULT_DATA_DIR / "generated"


# Each entry: (src_subdir, label_filename_or_None, converter_fn)
DatasetSpec = Tuple[str, Optional[str], Callable[..., List[str]]]


DATASETS: Dict[str, DatasetSpec] = {
    "whoandwhen_handcrafted": (
        "Who_and_When/Hand-Crafted", None, whoandwhen.convert_directory,
    ),
    "whoandwhen_algorithm_generated": (
        "Who_and_When/Algorithm-Generated", None, whoandwhen.convert_directory_algorithm,
    ),
    "agentdebugbench_alfworld": (
        "AgentDebugBench/ALFWorld", "AgentDebugBench/alfworld_labels.json", alfworld.convert_directory,
    ),
    "agentdebugbench_gaia": (
        "AgentDebugBench/GAIA", "AgentDebugBench/gaia_labels.json", gaia.convert_directory,
    ),
    "agentdebugbench_webshop": (
        "AgentDebugBench/WebShop", "AgentDebugBench/webshop_labels.json", webshop.convert_directory,
    ),
    "trajerrbench_zh_tau2bench": (
        "TrajErrBench/Tau2Bench", None, tau2bench.convert_directory,
    ),
    "trajerrbench_zh_swebenchpro": (
        "TrajErrBench/SWEBenchPro", None, swebenchpro.convert_directory,
    ),
}

_METADATA_IDENTITY = {
    "agentdebugbench_alfworld": ("agentdebugbench_alfworld", "alfworld", None),
    "agentdebugbench_gaia": ("agentdebugbench_gaia", "gaia", None),
    "agentdebugbench_webshop": ("agentdebugbench_webshop", "webshop", None),
    "whoandwhen_handcrafted": ("whoandwhen_handcrafted", "handcrafted", None),
    "whoandwhen_algorithm_generated": (
        "whoandwhen_algorithm_generated", "algorithm_generated", None,
    ),
    "trajerrbench_zh_tau2bench": ("trajerrbench", "tau2bench", "zh"),
    "trajerrbench_zh_swebenchpro": ("trajerrbench", "swebenchpro", "zh"),
}


def _convert_one(
    name: str,
    *,
    src: Optional[Path],
    out: Optional[Path],
    labels: Optional[Path],
) -> int:
    spec = DATASETS[name]
    src_subdir, default_label, fn = spec
    src_dir = src or (DEFAULT_SOURCE_DIR / src_subdir)
    out_dir = out or (DEFAULT_GENERATED_DIR / name)
    label_path = labels
    if label_path is None and default_label:
        candidate = DEFAULT_SOURCE_DIR / default_label
        label_path = candidate if candidate.exists() else None

    print(f"[{name}] src={src_dir} out={out_dir} labels={label_path}")
    if not (src_dir.is_dir() or src_dir.is_file()):
        print(f"[{name}] WARNING: source path does not exist, skipping")
        return 0

    if default_label:
        written = fn(src_dir, out_dir, label_path)
    else:
        written = fn(src_dir, out_dir)
    dataset_name, subset, language = _METADATA_IDENTITY[name]
    for written_path in written:
        path = Path(written_path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            metadata = payload.get("metadata", {})
            metadata["dataset"] = dataset_name
            extra = metadata.setdefault("extra", {})
            extra["source_benchmark"] = subset
            if language:
                extra["language"] = language
                extra["pair_id"] = f"{subset}:{metadata.get('task_id', path.stem)}"
            validate_unified(payload)
            path.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception as exc:
            raise RuntimeError(f"failed to normalize {path}: {exc}") from exc
    print(f"[{name}] wrote {len(written)} unified trajectories")
    return len(written)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build unified-schema trajectories.")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS.keys()),
        help="Convert a single dataset (mutually exclusive with --all).",
    )
    parser.add_argument("--all", action="store_true", help="Convert every supported dataset.")
    parser.add_argument("--src", type=Path, default=None, help="Override source directory.")
    parser.add_argument("--out", type=Path, default=None, help="Override unified output directory.")
    parser.add_argument("--labels", type=Path, default=None, help="Override label JSON path.")
    args = parser.parse_args(argv)

    if not args.all and not args.dataset:
        parser.error("either --all or --dataset is required")
    if args.all and (args.src or args.out or args.labels):
        parser.error("--src/--out/--labels can only be used with --dataset")

    targets = sorted(DATASETS.keys()) if args.all else [args.dataset]
    total = 0
    for name in targets:
        total += _convert_one(name, src=args.src, out=args.out, labels=args.labels)
    print(f"Done. Wrote {total} unified trajectories total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
