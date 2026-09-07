"""Build paired English TrajErrBench files from aligned processed trajectories."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from data_processing.release_metadata import normalize_payload, write_manifest
from data_processing.schema import validate_unified


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAJERRBENCH_ROOT = REPO_ROOT / "data" / "trajerrbench"


def _extract_swe_task_description(messages: List[Dict[str, Any]]) -> str:
    user_text = next(
        (
            str(message.get("content", ""))
            for message in messages
            if str(message.get("role", "")).lower() == "user"
        ),
        "",
    )
    match = re.search(
        r"<issue_description>\s*(.*?)\s*</issue_description>",
        user_text,
        flags=re.DOTALL,
    )
    return match.group(1).strip() if match else user_text.strip()


def _aligned_messages(
    base_messages: List[Dict[str, Any]],
    translated_messages: List[Dict[str, Any]],
    source: Path,
) -> List[Dict[str, Any]]:
    translated_by_step = {
        int(message["step_id"]): message for message in translated_messages
    }
    if len(translated_by_step) != len(translated_messages):
        raise ValueError(f"duplicate step_id in {source}")
    if len(base_messages) != len(translated_messages):
        raise ValueError(
            f"message count mismatch for {source}: "
            f"{len(base_messages)} != {len(translated_messages)}"
        )

    aligned = []
    for base in base_messages:
        step = int(base["step"])
        translated = translated_by_step.get(step)
        if translated is None:
            raise ValueError(f"missing translated step {step} in {source}")
        if translated.get("role") != base.get("role"):
            raise ValueError(
                f"role mismatch at step {step} in {source}: "
                f"{translated.get('role')} != {base.get('role')}"
            )
        message = dict(base)
        message["content"] = str(translated.get("content", ""))
        aligned.append(message)
    return aligned


def build_swebenchpro(processed_en: Path) -> int:
    zh_dir = TRAJERRBENCH_ROOT / "zh" / "swebenchpro"
    out_dir = TRAJERRBENCH_ROOT / "en" / "swebenchpro"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for zh_path in sorted(zh_dir.glob("*.json")):
        if "_instance_" not in zh_path.stem:
            raise ValueError(f"cannot split model and instance: {zh_path.name}")
        model, instance_tail = zh_path.stem.split("_instance_", 1)
        source = processed_en / model / f"instance_{instance_tail}.json"
        if not source.is_file():
            raise FileNotFoundError(f"missing English pair: {source}")

        base = json.loads(zh_path.read_text(encoding="utf-8"))
        translated = json.loads(source.read_text(encoding="utf-8"))
        raw_messages = translated.get("messages")
        if not isinstance(raw_messages, list):
            raise ValueError(f"missing messages in {source}")

        payload = {
            "messages": _aligned_messages(base["messages"], raw_messages, source),
            "metadata": {
                **base["metadata"],
                "task_description": _extract_swe_task_description(raw_messages),
            },
        }
        payload = normalize_payload(payload, "trajerrbench_en_swebenchpro")
        validate_unified(payload)
        (out_dir / zh_path.name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written += 1
    return written


def _tau2_message_content(message: Dict[str, Any]) -> str:
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list) and tool_calls:
        lines = ["[tool_calls]"]
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name", "") or "")
            arguments = call.get("arguments")
            if not isinstance(arguments, dict):
                arguments = {}
            lines.append(
                f"- {name}({json.dumps(arguments, ensure_ascii=False)})"
            )
        return "\n".join(lines)
    content = message.get("content")
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def build_tau2bench(failed_trajectory_dir: Path) -> int:
    raw_by_task_id: Dict[str, tuple[str, Dict[str, Any]]] = {}
    for source in sorted(failed_trajectory_dir.glob("*.json")):
        raw = json.loads(source.read_text(encoding="utf-8"))
        policy = str(
            ((raw.get("info") or {}).get("environment_info") or {}).get(
                "policy", ""
            )
            or ""
        )
        prefix = source.stem
        simulations = raw.get("failed_simulations")
        if not isinstance(simulations, list):
            continue
        for simulation in simulations:
            if not isinstance(simulation, dict):
                continue
            simulation_id = str(simulation.get("id", "") or "")
            if simulation_id:
                raw_by_task_id[f"{prefix}::{simulation_id}"] = (
                    policy,
                    simulation,
                )

    zh_dir = TRAJERRBENCH_ROOT / "zh" / "tau2bench"
    out_dir = TRAJERRBENCH_ROOT / "en" / "tau2bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for zh_path in sorted(zh_dir.glob("*.json")):
        base = json.loads(zh_path.read_text(encoding="utf-8"))
        task_id = base["metadata"]["task_id"]
        pair = raw_by_task_id.get(task_id)
        if pair is None:
            raise FileNotFoundError(f"missing English Tau2 trajectory: {task_id}")
        policy, simulation = pair
        raw_messages = simulation.get("messages")
        if not isinstance(raw_messages, list):
            raise ValueError(f"missing messages for {task_id}")

        expected_roles = ["system"] + [
            str(message.get("role", "")) for message in raw_messages
        ]
        base_roles = [message["role"] for message in base["messages"]]
        if expected_roles != base_roles:
            raise ValueError(f"message role alignment failed for {task_id}")

        messages = []
        for index, base_message in enumerate(base["messages"]):
            message = dict(base_message)
            message["content"] = (
                policy
                if index == 0
                else _tau2_message_content(raw_messages[index - 1])
            )
            messages.append(message)

        first_user = next(
            (
                _tau2_message_content(message)
                for message in raw_messages
                if message.get("role") == "user"
            ),
            "",
        )
        payload = {
            "messages": messages,
            "metadata": {
                **base["metadata"],
                "task_description": first_user,
            },
        }
        payload = normalize_payload(payload, "trajerrbench_en_tau2bench")
        validate_unified(payload)
        (out_dir / zh_path.name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written += 1
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--swebenchpro-processed-en",
        type=Path,
        help="directory containing per-model English processed SWE-Bench Pro JSON",
    )
    parser.add_argument(
        "--tau2bench-failed-trajectories",
        type=Path,
        help="directory containing English Tau2 failed_simulations JSON files",
    )
    args = parser.parse_args()
    if not args.swebenchpro_processed_en and not args.tau2bench_failed_trajectories:
        parser.error("provide at least one English source directory")
    if args.swebenchpro_processed_en:
        count = build_swebenchpro(args.swebenchpro_processed_en.resolve())
        print(f"trajerrbench_en_swebenchpro: wrote {count} files")
    if args.tau2bench_failed_trajectories:
        count = build_tau2bench(args.tau2bench_failed_trajectories.resolve())
        print(f"trajerrbench_en_tau2bench: wrote {count} files")
    print(f"manifest: wrote {write_manifest(TRAJERRBENCH_ROOT)} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
