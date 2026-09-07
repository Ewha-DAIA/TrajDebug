# Data processing

> [中文说明](README_zh.md)

This package converts heterogeneous agent traces into the single schema consumed by the detector.

## Unified schema

Each detector-ready trajectory JSON contains:

```json
{
  "messages": [
    {"step": 0, "role": "user", "name": "human", "content": "..."}
  ],
  "metadata": {
    "dataset": "example",
    "task_id": "task-1",
    "task_description": "...",
    "reward": 0,
    "annotation": {
      "critical_error_step": 7,
      "critical_error_type": "act.WrongTool"
    },
    "extra": {}
  }
}
```

Required invariants:

- `messages` is non-empty and `messages[i].step == i`.
- `role` is one of `user`, `assistant`, `tool`, or `system`.
- `content` is a string.
- `reward` is `0` or `1`.
- A non-null `critical_error_step` is a valid unified-message index and does not point to a `user` message.
- Successful trajectories have null critical-error annotations.

`schema.py` defines and validates this contract.

## Released data versus rebuilding

The checked-in files under `data/trajerrbench`, `data/agentdebugbench`, and
`data/whoandwhen` are already converted and can be passed directly to the
detector. Rebuilding is optional and does not download upstream data.

To rebuild, obtain the relevant upstream release under its own terms and place
the source files under:

```text
data/sources/
├── AgentDebugBench/{ALFWorld,GAIA,WebShop}/
├── Who_and_When/{Hand-Crafted,Algorithm-Generated}/
└── TrajErrBench/{Tau2Bench,SWEBenchPro}/
```

You may instead pass `--src` and `--labels` explicitly. Generated files default
to `data/generated/<dataset-key>/`, so rebuilding never overwrites the
checked-in public release.

## Build datasets

From the repository root:

```bash
python -m data_processing.build_unified_dataset --all
python -m data_processing.build_unified_dataset --dataset trajerrbench_zh_tau2bench
```

Registered conversion keys are `agentdebugbench_alfworld`,
`agentdebugbench_gaia`, `agentdebugbench_webshop`,
`whoandwhen_handcrafted`, `whoandwhen_algorithm_generated`,
`trajerrbench_zh_tau2bench`, and `trajerrbench_zh_swebenchpro`.

## Add a dataset

1. Add `data_processing/<name>.py` with `convert_directory(src, out[, labels])`.
2. Normalize roles and assign contiguous zero-based `step` values.
3. Construct the required metadata and map labels to `<module>.<subtype>` when available.
4. Validate every output with `validate_unified`.
5. Register the converter in `DATASETS` in `build_unified_dataset.py`.
6. Add dataset-specific judging/compression settings to `dataset_config.json` when needed.

For message-style sources similar to ALFWorld, GAIA, or WebShop, reuse `_standard_messages.convert_standard_record`. For custom structures, construct the schema directly; `whoandwhen.py`, `tau2bench.py`, and `swebenchpro.py` are examples.

Build and run:

```bash
python -m data_processing.build_unified_dataset \
  --dataset my_dataset \
  --src data/MyDataset \
  --out data/generated/my_dataset

python -m data_processing.validate_dataset --path data/generated/my_dataset
```

Detector outputs default to `outputs`; data conversion never writes detector outputs.
