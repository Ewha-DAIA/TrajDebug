# TRAJDEBUG

> [中文说明](README_zh.md)

**TRAJDEBUG: Tracing Error Lifecycle to Identify Critical Failures in Long-Horizon Agent Trajectories** is an evidence-grounded framework for locating the earliest decisive error responsible for a failed LLM-agent trajectory. This repository provides the detector, unified data adapters, evaluation tools, and a local viewer.

## News

- **2026-09-07:** We released the English edition of TrajErrBench.
- **2026-08-20:** Our work was accepted to Findings of EMNLP 2026. The paper is available on [arXiv](https://arxiv.org/abs/2608.06346).

## Method

TRAJDEBUG first builds multi-granularity trajectory views, then performs three
auditable stages:

1. **Error trigger detection:** identifies wrong commitments and requires verbatim evidence for both the commitment and violated reference.
2. **Error state classification:** clusters triggers by violated object and classifies resolution and terminal impact.
3. **Causal attribution:** selects the failure-responsible origin from terminal-relevant candidates.

![Overview of the TRAJDEBUG pipeline](assets/%20pipeline.png)

See [`detector/README.md`](detector/README.md) for implementation details and the [`paper`](main.pdf).

## Results

![Critical error detection results](assets/results.png)

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# Include the local viewer:
pip install -e ".[viewer]"
```

Copy `.env.example` to `.env`, fill in your endpoint settings, and export them in your shell. The pipeline uses only OpenAI-compatible chat-completions APIs.

## Data

Detector inputs are one unified-schema JSON file per trajectory. Detector-ready
data is included directly in the repository:

| Dataset key | Trajectories |
|---|---:|
| `agentdebugbench_alfworld` | 100 |
| `agentdebugbench_gaia` | 50 |
| `agentdebugbench_webshop` | 50 |
| `whoandwhen_handcrafted` | 58 |
| `whoandwhen_algorithm_generated` | 126 |
| `trajerrbench_zh_tau2bench` | 400 |
| `trajerrbench_zh_swebenchpro` | 86 |
| `trajerrbench_en_tau2bench` | 400 |
| `trajerrbench_en_swebenchpro` | 86 |

We release our **TrajErrBench** dataset under
[`data/trajerrbench`](data/trajerrbench), grouped by language and source
benchmark. Our paper uses the Chinese (`zh`) edition; we have also recently
released a paired English (`en`) edition with the same task IDs and
critical-error labels. AgentDebugBench and Who&When evaluation subsets are grouped under
[`data/agentdebugbench`](data/agentdebugbench) and
[`data/whoandwhen`](data/whoandwhen).

The included JSON files can be used directly. To rebuild data from separately
obtained upstream sources:

```bash
python -m data_processing.build_unified_dataset --all
```

Schema and custom-adapter instructions are in [`data_processing/README.md`](data_processing/README.md). Data licensing and attribution notes are in [`data/README.md`](data/README.md).

## OpenAI-compatible API

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
export DETECTOR_MODEL="your-model-name"
```

`DETECTOR_API_KEY` and `DETECTOR_BASE_URL` may be used instead of the corresponding `OPENAI_*` variables.

## Self-hosting with SGLang

Install SGLang separately, then launch one OpenAI-compatible server:

```bash
MODEL_PATH=/models/your-model \
SERVED_MODEL_NAME=your-model \
TP_SIZE=8 \
bash deploy_qwen_router.sh
```

The default endpoint is `http://127.0.0.1:30000/v1`.

Run the detector against it with:

```bash
DETECTOR_BASE_URL=http://127.0.0.1:30000/v1 \
DETECTOR_API_KEY=EMPTY \
DETECTOR_MODEL=your-model \
bash run_pipeline.sh
```

## Run

```bash
DETECTOR_MODEL=your-model bash run_pipeline.sh

# Select datasets or tune concurrency:
DATASETS="trajerrbench_zh_tau2bench" \
FILE_CONCURRENCY=2 \
LLM_CONCURRENCY=8 \
DETECTOR_MODEL=your-model \
bash run_pipeline.sh
```

Dataset keys are resolved through `data_processing/dataset_registry.py`;
outputs default to `outputs`. Each run produces `<dataset>_stage_a`,
`<dataset>_stage_b`, `<dataset>_phase1`, `<dataset>_phase2`,
`<dataset>_final`, and `score_<dataset>.json`. Validate dataset discovery and
all input schemas without making API calls with
`VALIDATE_ONLY=1 bash run_pipeline.sh`.

## Evaluation

`run_pipeline.sh` runs exact-step evaluation automatically. To score existing predictions:

```bash
python detector/score_steps.py \
  --unified-dir data/trajerrbench/zh/tau2bench \
  --pred-dir outputs/trajerrbench_zh_tau2bench_final \
  --out outputs/score_trajerrbench_zh_tau2bench.json
```

## Feedback generation

The identified critical error step can be converted into actionable feedback to improve agent self-repair and failure-memory transfer (see Section 6 of the paper):

```bash
python applications/generate_feedback.py \
  --final_dir outputs/trajerrbench_zh_tau2bench_final \
  --trajectory_dir data/trajerrbench/zh/tau2bench \
  --stage_a_dir outputs/trajerrbench_zh_tau2bench_stage_a \
  --output_dir outputs/trajerrbench_zh_tau2bench_report \
  --base_url "$OPENAI_BASE_URL" \
  --model "$DETECTOR_MODEL" \
  --api_key "$OPENAI_API_KEY" \
  --resume
```

The generated feedback is available in `outputs/<dataset>_report/<task_id>_report.json` under `fix_suggestion.hint_sentence`.

## Viewer

```bash
python -m viewer.server --dataset trajerrbench_zh_tau2bench --output-dir outputs
```

Open <http://localhost:8000>. See [`viewer/README.md`](viewer/README.md).

## Citation

If you use TRAJDEBUG or TrajErrBench, please cite our paper:

```bibtex
@article{qi2026trajdebug,
  title={TRAJDEBUG: Tracing Error Lifecycle to Identify Critical Failures in Long-Horizon Agent Trajectories},
  author={Qi, Yunjia and Yin, Zehua and Shi, Xintong and Peng, Hao and Lu, Songyuanyi and Liu, Yixian and Xuan, Richeng and Liu, Yuhong and Hu, Zhichao and Wang, Xiaozhi and others},
  journal={arXiv preprint arXiv:2608.06346},
  year={2026}
}
```

## License

Code is released under the [MIT License](LICENSE). Dataset components remain subject to their original licenses and terms.
