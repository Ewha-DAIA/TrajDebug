import json
from pathlib import Path

from data_processing.schema import validate_unified


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAJERRBENCH = REPO_ROOT / "data" / "trajerrbench"
SAFE_TRAJERRBENCH_EXTRA = {
    "agent_framework_description",
    "language",
    "source_benchmark",
    "pair_id",
}


def _assert_trajerrbench_release(subset, language, expected):
    directory = TRAJERRBENCH / language / subset
    paths = sorted(directory.glob("*.json"))
    assert len(paths) == expected

    task_ids = set()
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_unified(payload)
        messages = payload["messages"]
        metadata = payload["metadata"]
        annotation = metadata["annotation"]
        step = annotation["critical_error_step"]

        assert metadata["dataset"] == "trajerrbench"
        assert metadata["task_id"] not in task_ids
        assert isinstance(step, int) and 0 <= step < len(messages)
        assert messages[step]["role"] != "user"
        assert set(annotation) == {
            "critical_error_step",
            "critical_error_type",
        }
        assert set(metadata) == {
            "dataset",
            "task_id",
            "task_description",
            "reward",
            "annotation",
            "extra",
        }
        assert set(metadata["extra"]) == SAFE_TRAJERRBENCH_EXTRA
        assert metadata["extra"]["language"] == language
        assert metadata["extra"]["source_benchmark"] == subset
        assert metadata["extra"]["pair_id"] == f"{subset}:{metadata['task_id']}"
        task_ids.add(metadata["task_id"])


def test_trajerrbench_chinese_release_is_valid_and_sanitized():
    _assert_trajerrbench_release("tau2bench", "zh", 400)
    _assert_trajerrbench_release("swebenchpro", "zh", 86)


def test_trajerrbench_english_release_is_valid_and_sanitized():
    _assert_trajerrbench_release("tau2bench", "en", 400)
    _assert_trajerrbench_release("swebenchpro", "en", 86)


def test_trajerrbench_language_pairs_have_identical_structure_and_labels():
    for subset in ("tau2bench", "swebenchpro"):
        zh_dir = TRAJERRBENCH / "zh" / subset
        en_dir = TRAJERRBENCH / "en" / subset
        assert {path.name for path in zh_dir.glob("*.json")} == {
            path.name for path in en_dir.glob("*.json")
        }
        for zh_path in zh_dir.glob("*.json"):
            zh = json.loads(zh_path.read_text(encoding="utf-8"))
            en = json.loads((en_dir / zh_path.name).read_text(encoding="utf-8"))
            assert [m["role"] for m in zh["messages"]] == [
                m["role"] for m in en["messages"]
            ]
            assert [m["step"] for m in zh["messages"]] == [
                m["step"] for m in en["messages"]
            ]
            assert zh["metadata"]["task_id"] == en["metadata"]["task_id"]
            assert zh["metadata"]["reward"] == en["metadata"]["reward"]
            assert zh["metadata"]["annotation"] == en["metadata"]["annotation"]
            assert (
                zh["metadata"]["extra"]["pair_id"]
                == en["metadata"]["extra"]["pair_id"]
            )


def test_removed_swebenchpro_variant_is_absent():
    removed_variant = "swebenchpro" + "4model"
    assert not (TRAJERRBENCH / "zh" / removed_variant).exists()


def test_trajerrbench_manifest_covers_release_and_english_pairs():
    records = [
        json.loads(line)
        for line in (TRAJERRBENCH / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    assert len(records) == 486
    assert len({record["pair_id"] for record in records}) == 486
    assert sum(record["en_path"] is not None for record in records) == 486
    for record in records:
        assert (TRAJERRBENCH / record["zh_path"]).is_file()
        if record["en_path"] is not None:
            assert (TRAJERRBENCH / record["en_path"]).is_file()
