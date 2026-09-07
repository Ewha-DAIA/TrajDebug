"""Normalize public-release metadata and build the TrajErrBench pair manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from data_processing.dataset_registry import DATASETS, dataset_path
from data_processing.schema import validate_unified


_PUBLIC_METADATA = {
    "agentdebugbench_alfworld": ("agentdebugbench_alfworld", "alfworld", None),
    "agentdebugbench_gaia": ("agentdebugbench_gaia", "gaia", None),
    "agentdebugbench_webshop": ("agentdebugbench_webshop", "webshop", None),
    "whoandwhen_handcrafted": ("whoandwhen_handcrafted", "handcrafted", None),
    "whoandwhen_algorithm_generated": (
        "whoandwhen_algorithm_generated",
        "algorithm_generated",
        None,
    ),
    "trajerrbench_zh_tau2bench": ("trajerrbench", "tau2bench", "zh"),
    "trajerrbench_zh_swebenchpro": ("trajerrbench", "swebenchpro", "zh"),
    "trajerrbench_en_tau2bench": ("trajerrbench", "tau2bench", "en"),
    "trajerrbench_en_swebenchpro": ("trajerrbench", "swebenchpro", "en"),
}


def _safe_annotation(raw: Any) -> Dict[str, Any]:
    annotation = raw if isinstance(raw, dict) else {}
    return {
        "critical_error_step": annotation.get("critical_error_step"),
        "critical_error_type": annotation.get("critical_error_type"),
    }


def normalize_payload(payload: Dict[str, Any], dataset_key: str) -> Dict[str, Any]:
    dataset_name, subset, language = _PUBLIC_METADATA[dataset_key]
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("missing metadata")

    if language:
        # TrajErrBench metadata is rebuilt from an allowlist. In particular,
        # never publish raw summaries, annotator rationales, model predictions,
        # internal grouping IDs, or processing diagnostics.
        old_extra = metadata.get("extra")
        old_extra = old_extra if isinstance(old_extra, dict) else {}
        task_id = str(metadata.get("task_id", "")).strip()
        metadata = {
            "dataset": dataset_name,
            "task_id": task_id,
            "task_description": str(metadata.get("task_description", "") or ""),
            "reward": metadata.get("reward"),
            "annotation": _safe_annotation(metadata.get("annotation")),
            "extra": {
                "agent_framework_description": str(
                    old_extra.get("agent_framework_description", "") or ""
                ),
                "language": language,
                "source_benchmark": subset,
                "pair_id": f"{subset}:{task_id}",
            },
        }
        payload = {"messages": payload.get("messages"), "metadata": metadata}
    else:
        metadata["dataset"] = dataset_name
        extra = metadata.setdefault("extra", {})
        if not isinstance(extra, dict):
            extra = {}
            metadata["extra"] = extra
        extra["source_benchmark"] = subset

    validate_unified(payload)
    return payload


def normalize_dataset(dataset_key: str) -> int:
    directory = dataset_path(dataset_key)
    count = 0
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        normalized = normalize_payload(payload, dataset_key)
        path.write_text(
            json.dumps(normalized, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        count += 1
    return count


def write_manifest(root: Path) -> int:
    records = []
    for subset in ("tau2bench", "swebenchpro"):
        zh_dir = root / "zh" / subset
        en_dir = root / "en" / subset
        zh_by_id = {}
        en_by_id = {}
        for language, directory, target in (
            ("zh", zh_dir, zh_by_id),
            ("en", en_dir, en_by_id),
        ):
            if not directory.is_dir():
                continue
            for path in sorted(directory.glob("*.json")):
                payload = json.loads(path.read_text(encoding="utf-8"))
                pair_id = payload["metadata"]["extra"]["pair_id"]
                if pair_id in target:
                    raise ValueError(f"duplicate {language} pair_id: {pair_id}")
                target[pair_id] = path
        if en_by_id and set(zh_by_id) != set(en_by_id):
            missing_en = sorted(set(zh_by_id) - set(en_by_id))
            missing_zh = sorted(set(en_by_id) - set(zh_by_id))
            raise ValueError(
                f"{subset} pairing mismatch: missing_en={missing_en[:3]}, "
                f"missing_zh={missing_zh[:3]}"
            )
        for pair_id, zh_path in sorted(zh_by_id.items()):
            en_path = en_by_id.get(pair_id)
            records.append(
                {
                    "pair_id": pair_id,
                    "source_benchmark": subset,
                    "task_id": pair_id.split(":", 1)[1],
                    "zh_path": str(zh_path.relative_to(root)),
                    "en_path": (
                        str(en_path.relative_to(root)) if en_path is not None else None
                    ),
                }
            )
    manifest = root / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
        encoding="utf-8",
    )
    return len(records)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="*",
        choices=sorted(_PUBLIC_METADATA),
        default=None,
    )
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()

    keys = args.datasets or [
        key for key in DATASETS if dataset_path(key).is_dir()
    ]
    for key in keys:
        print(f"{key}: normalized {normalize_dataset(key)} files")
    if args.write_manifest:
        root = Path(__file__).resolve().parents[1] / "data" / "trajerrbench"
        print(f"manifest: wrote {write_manifest(root)} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
