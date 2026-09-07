from data_processing.dataset_registry import (
    DATASETS,
    canonical_key,
    dataset_path,
    get_dataset,
)


def test_registered_release_paths_exist():
    assert set(DATASETS) == {
        "trajerrbench_zh_tau2bench",
        "trajerrbench_zh_swebenchpro",
        "trajerrbench_en_tau2bench",
        "trajerrbench_en_swebenchpro",
        "agentdebugbench_alfworld",
        "agentdebugbench_gaia",
        "agentdebugbench_webshop",
        "whoandwhen_handcrafted",
        "whoandwhen_algorithm_generated",
    }
    for key in DATASETS:
        assert dataset_path(key).is_dir()


def test_legacy_alias_resolves_without_duplicate_data():
    assert canonical_key("alfworld") == "agentdebugbench_alfworld"
    assert dataset_path("tau2bench") == dataset_path(
        "trajerrbench_zh_tau2bench"
    )
    assert get_dataset("whoandwhen_algorithm").profile == "whoandwhen_algorithm"
