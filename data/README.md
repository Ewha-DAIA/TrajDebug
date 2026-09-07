# Data

All JSON files in this directory already use the unified trajectory schema
consumed by TRAJDEBUG; no conversion is required to run the released data.

```text
data/
├── trajerrbench/
│   ├── zh/{tau2bench,swebenchpro}/
│   └── en/{tau2bench,swebenchpro}/
├── agentdebugbench/{alfworld,gaia,webshop}/
└── whoandwhen/{handcrafted,algorithm_generated}/
```

TrajErrBench consists of 400 failed trajectories derived from τ²-Bench and 86
failed trajectories derived from SWE-Bench Pro.
`trajerrbench/manifest.jsonl` records release membership and the one-to-one
Chinese/English pairing available for both subsets. The paper evaluates the
Chinese edition. The remaining folders reproduce the
paper's evaluations on Who&When and AgentDebugBench.

Registered pipeline keys and physical paths are defined in
`data_processing/dataset_registry.py`. Validate all available data without
calling an LLM:

```bash
VALIDATE_ONLY=1 bash run_pipeline.sh
```

The repository's MIT license applies to TRAJDEBUG code and original annotations.
It does not relicense task text, trajectories, environments, repositories, or
other material originating from the upstream benchmarks. Those components remain
subject to their respective licenses and terms:

- τ²-Bench
- SWE-Bench Pro and the repositories represented by its tasks
- WhoAndWhen
- AgentDebugBench (ALFWorld, GAIA, and WebShop subsets)

Before redistributing or using the data beyond research reproduction, review
the current upstream terms. The converters in `data_processing/` can rebuild
the unified representation from separately obtained source datasets. Upstream
source acquisition is not automated: obtain the data under its original terms,
place it under `data/sources/` as documented in
`data_processing/README.md`, and write rebuilt files to `data/generated/` so
the checked-in release is not overwritten.
