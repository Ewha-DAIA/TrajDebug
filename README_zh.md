# TRAJDEBUG

> [English](README.md)

**TRAJDEBUG：通过追踪错误生命周期定位长程 Agent 轨迹中的关键失败**，是一个基于证据的关键错误定位框架。本仓库提供 detector、统一数据适配器、评测工具与本地 viewer。

## 最新动态

- **2026-09-07：**我们发布了 TrajErrBench 英文版。
- **2026-08-20：**我们的工作被 EMNLP 2026 Findings 接收。论文已发布在 [arXiv](https://arxiv.org/abs/2608.06346)。

## 方法

TRAJDEBUG 先构建多粒度轨迹视图，再执行三个可审计阶段：

1. **错误触发检测：**识别错误承诺，并要求逐字引用错误内容与被违反的参照。
2. **错误状态分类：**按被违反对象聚合 trigger，判断错误是否修复及其终态影响。
3. **因果归因：**从与终态相关的候选中选择对失败负责的起源步。

![TRAJDEBUG pipeline 总览](assets/%20pipeline.png)

实现细节见 [`detector/README_zh.md`](detector/README_zh.md)，论文见 [`main.pdf`](main.pdf)。

## 结果

![关键错误检测结果](assets/results.png)

## 安装

推荐 Python 3.10+。

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
# 同时安装 viewer：
pip install -e ".[viewer]"
```

将 `.env.example` 复制为 `.env`，填写端点信息并在 shell 中导出。本项目仅使用 OpenAI-compatible Chat Completions API。

## 数据

detector 输入为每条轨迹一个统一 schema JSON。仓库直接提供可运行数据：

| 数据集键 | 轨迹数 |
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

我们的数据集 **TrajErrBench** 发布在
[`data/trajerrbench`](data/trajerrbench)，按语言与来源 benchmark 分层。
论文使用中文（`zh`）版本；同时最近我们公布英文（`en`）版本，与中文的
task ID 和关键错误标注一一对应。
AgentDebugBench 与 Who&When 测试子集分别位于
[`data/agentdebugbench`](data/agentdebugbench) 和
[`data/whoandwhen`](data/whoandwhen)。

仓库内 JSON 可直接运行。若要从另行获取的上游原始数据重新转换：

```bash
python -m data_processing.build_unified_dataset --all
```

统一 schema 与自定义适配器说明见 [`data_processing/README_zh.md`](data_processing/README_zh.md)，数据许可与归属说明见 [`data/README.md`](data/README.md)。

## OpenAI-compatible API

```bash
export OPENAI_API_KEY="..."
export OPENAI_BASE_URL="https://api.openai.com/v1"
export DETECTOR_MODEL="your-model-name"
```

也可用 `DETECTOR_API_KEY`、`DETECTOR_BASE_URL` 覆盖对应的 `OPENAI_*` 变量。

## 使用 SGLang 自部署

单独安装 SGLang 后，启动一个 OpenAI-compatible 服务：

```bash
MODEL_PATH=/models/your-model \
SERVED_MODEL_NAME=your-model \
TP_SIZE=8 \
bash deploy_qwen_router.sh
```

默认端点为 `http://127.0.0.1:30000/v1`。

使用该端点运行 detector：

```bash
DETECTOR_BASE_URL=http://127.0.0.1:30000/v1 \
DETECTOR_API_KEY=EMPTY \
DETECTOR_MODEL=your-model \
bash run_pipeline.sh
```

## 运行

```bash
DETECTOR_MODEL=your-model bash run_pipeline.sh

DATASETS="trajerrbench_zh_tau2bench" \
FILE_CONCURRENCY=2 \
LLM_CONCURRENCY=8 \
DETECTOR_MODEL=your-model \
bash run_pipeline.sh
```

数据集键通过 `data_processing/dataset_registry.py` 解析，输出默认位于
`outputs`。每个数据集会生成 `<dataset>_stage_a`、`<dataset>_stage_b`、
`<dataset>_phase1`、`<dataset>_phase2`、`<dataset>_final` 与
`score_<dataset>.json`。可用 `VALIDATE_ONLY=1 bash run_pipeline.sh`
在不调用 API 的情况下检查数据发现和全部输入 schema。

## 评测

`run_pipeline.sh` 会自动执行严格步匹配评测。也可单独评分：

```bash
python detector/score_steps.py \
  --unified-dir data/trajerrbench/zh/tau2bench \
  --pred-dir outputs/trajerrbench_zh_tau2bench_final \
  --out outputs/score_trajerrbench_zh_tau2bench.json
```

## 生成反馈

可以根据定位出的 critical error step 生成可操作的反馈，用于提升 Agent 的自我修复与失败记忆迁移能力（详见论文 Section 6）：

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

生成的 feedback 位于 `outputs/<dataset>_report/<task_id>_report.json` 的 `fix_suggestion.hint_sentence` 字段。

## Viewer

```bash
python -m viewer.server --dataset trajerrbench_zh_tau2bench --output-dir outputs
```

浏览器打开 <http://localhost:8000>。详见 [`viewer/README.md`](viewer/README.md)。

## 引用

如果您使用 TRAJDEBUG 或 TrajErrBench，请引用我们的论文：

```bibtex
@article{qi2026trajdebug,
  title={TRAJDEBUG: Tracing Error Lifecycle to Identify Critical Failures in Long-Horizon Agent Trajectories},
  author={Qi, Yunjia and Yin, Zehua and Shi, Xintong and Peng, Hao and Lu, Songyuanyi and Liu, Yixian and Xuan, Richeng and Liu, Yuhong and Hu, Zhichao and Wang, Xiaozhi and others},
  journal={arXiv preprint arXiv:2608.06346},
  year={2026}
}
```

## 许可证

代码采用 [MIT License](LICENSE)。各数据集组件仍受其原始许可证与使用条款约束。
