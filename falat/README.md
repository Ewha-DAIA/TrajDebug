# FALAT reproduction

Re-implementation of **FALAT: Tracing Failures in LLM Agent Trajectories via
Dependency-Guided Search** (Rafi et al., arXiv:2606.00765). The authors did not
release code; this follows the paper's §2 and the prompts in Appendix A.

The **method** is FALAT. **Inputs, outputs and scoring** are TrajDebug's
(unified trajectories under `data/`, `<stem>_critical_error.json` predictions,
`detector/score_steps.py`), so FALAT can be compared directly with the TrajDebug
pipeline and the minimal-prompt baseline.

## Layout

```
falat/
├── prompts/                               one file per prompt (paper text in `template`)
│   ├── stage0_task_spec.yaml              assumption, not in the paper
│   ├── stage1_A1_prior.yaml               A.1 external prior π
│   ├── stage1_A1_hierarchical.yaml        A.1 hierarchical representation M
│   ├── stage2_A2_typed_dependency.yaml    A.2
│   ├── stage3_A3_candidate_selection.yaml A.3 (+ appended_output_format, assumption)
│   ├── stage3_A3_candidate_verification.yaml A.3 (+ appended_output_format, assumption)
│   └── stage4_A4_local_research.yaml      A.4
├── stages/                                one module per paper stage
│   ├── _stage_common.py                   StageRunner: LLM call, parse retries, fallback
│   ├── stage0_task_spec.py
│   ├── stage1_prior.py
│   ├── stage1_hierarchy.py                per-window M, candidate set C, regions R
│   ├── stage2_typed_dependency.py
│   ├── stage3a_candidate_selection.py
│   ├── stage3b_candidate_verification.py
│   └── stage4_local_research.py
├── pipeline.py                            runs the stages, builds the prediction
├── run_falat.py                           CLI: one run directory per invocation
├── score_falat.py                         paper metrics + stage diagnostics
├── RESULTS.md                             reproduction report (GPT-5.4-Mini)
├── common/
│   ├── prompts.py               Prompt loader (inserts appended_output_format, checks placeholders)
│   ├── llm_client.py            OpenAI-compatible client, typed errors, retry policy
│   ├── trajectory.py            Step rendering, agents block, step index, final answer, windows
│   ├── parsing.py               Section / step-entry / transition / JSON parsers
│   ├── run_logger.py            logs/run.log + logs/errors.jsonl
│   └── stub_client.py           Offline stub for --dry_run
└── outputs/                     (git-ignored) results
run_pipeline_falat.sh            Entry point (repo root), mirrors run_pipeline.sh
```

## Running

`.env` in the repo root (git-ignored):

```bash
OPENAI_API_KEY=sk-or-...                      # OpenRouter key
OPENAI_BASE_URL=https://openrouter.ai/api/v1
DETECTOR_MODEL=openai/gpt-5.4-mini
```

```bash
DRY_RUN=1 LIMIT=3 bash run_pipeline_falat.sh                      # offline check, no API calls
DATASETS=whoandwhen_handcrafted LIMIT=5 bash run_pipeline_falat.sh # small real run
bash run_pipeline_falat.sh                                          # both Who&When subsets
CLIP_CHARS=20000 TAG=clip20k bash run_pipeline_falat.sh             # clipped variant
REASONING_EFFORT=medium bash run_pipeline_falat.sh                 # reasoning on (run dir gets _reasoning-medium)
DETECTOR_MODEL=deepseek/deepseek-v3.2 bash run_pipeline_falat.sh    # other backbone (command line overrides .env)
DETECTOR_MODEL=minimax/minimax-m2.5 bash run_pipeline_falat.sh
RUN_DIR=falat/outputs/<dataset>/<run> bash run_pipeline_falat.sh    # resume
```

Each run writes `falat/outputs/<dataset>/<model>_<YYYYmmdd-HHMMSS>[_<tag>]/`:

| Path | Content |
|---|---|
| `config.json` | model, k, K, clip, retries, client params, git commit, prompt-file hash |
| `predictions/` | `<stem>_critical_error.json` (TrajDebug format + `critical_agent`) |
| `traces/` | per-stage prompt, raw responses, parsed result, token usage, status |
| `logs/run.log` | progress |
| `logs/errors.jsonl` | one line per retry / failure / fallback, with `error_type` |
| `summary.json` | counts, token usage, elapsed time |
| `score_falat.json` | `metrics` (final t̃ / ã), `metrics_stage3_t_hat` (t̂ / α̂), stage diagnostics, per-trajectory rows |
| `score_steps.json` | TrajDebug `score_steps.py` report |

`error_type` values: `auth`, `rate_limit`, `timeout`, `connection`,
`context_length`, `bad_request`, `api_error`, `truncated`, `empty_response`,
`parse_error`, `clipped`, `fallback`, `pipeline_error`.

## Pipeline

| Stage | Prompt | Output |
|---|---|---|
| 0 (assumption) | `task_spec` | answer_type, constraints |
| 1 | `prior` (A.1) | π: QUESTION_ANALYSIS / AGENT_ROLES / ERROR_SIGNALS |
| 1 | `hierarchical` (A.1), once per k-window | M_hi, M_mid (= C), M_lo |
| 1 | code | consolidation into contiguous regions R |
| 2 | `typed_dependency` (A.2) | transition labels, pruning summary (C′), value flow |
| 3a | `candidate_selection` (A.3) | C_H (top K = 3) |
| 3b | `candidate_verification` (A.3) | t̂ |
| 4 | `local_research` (A.4) | t̃ (CONFIRM / REVISE) |

## Reproduction assumptions (not specified in the paper)

1. **Window size k = 10.** §2.1 defines k but gives no value. The A.1
   hierarchical prompt is used as a per-window template (`{steps_block}` and
   `{step_index}` hold one window). Algorithm-generated trajectories have ≤ 10
   steps, so k only affects hand-crafted ones. Note: the paper's K = 3 is |C_H|,
   not k.
2. **R usage.** Candidate regions spanning a window boundary are grouped in the
   A.2 `{mid_level}` under a "(cross-window region: Steps a-b)" header.
3. **Candidate set C** = steps with a MID_LEVEL or LOW_LEVEL entry.
4. **3a / 3b JSON schemas** were added (the paper only says "Output only the JSON
   object"). 3b's `description` / `backward_chain` feed A.4's
   `{predicted_desc}` / `{backward_chain}`.
5. **`{answer_type}`, `{constraints}`** come from a Stage-0 call that sees only
   the question (never the ground truth).
6. **Pruning C → C′ is left to the LLM.** The full A.2 output is passed to 3a as
   `{transitions_section}`; no rule-based filter.
7. **`{candidates_block}` (3b, A.4).** Prompt wording is unchanged; material the
   paper says is provided but has no slot goes here:
   - 3b: π's ERROR_SIGNALS + per candidate [3a reasoning, MID entry, typed
     transitions involving the step]. Raw ±1 steps go to `{raw_steps}`, linked
     LOW entries to `{low_level}`.
   - A.4: π's ERROR_SIGNALS + per step in [max(1, t̂−2), min(t̂+2, T)]: raw
     content, MID, LOW, typed transitions; t̂ is marked.
8. **`{agents_block}`** (paper's A, P, T): algorithm-generated →
   `metadata.extra.system_prompt`; hand-crafted → the team description in the
   Orchestrator's initial plan.
9. **`{final_answer}`** (ô): the `FINAL ANSWER:` line if present, otherwise the
   last non-`TERMINATE` message.
10. **Step rendering** follows the minimal baseline's `_render_trajectory_block`
    (`Step {step} [role: {name}]:`); step numbers are the unified `step` field,
    which equals `metadata.annotation.critical_error_step` numbering.
11. **Predicted agent** = agent of the predicted step, with the parenthesised
    suffix removed (`Orchestrator (thought)` → `Orchestrator`); compared to
    `metadata.extra.mistake_agent` case- and underscore-insensitively.
12. **Clipping** is off by default (`CLIP_CHARS=0`); when on, each step is
    clipped with `detector/utils/llm_compression.clip_text_middle` and logged.
13. **Sampling / reasoning.** The paper states no temperature or reasoning
    setting. Temperature is not sent (GPT-5-family models only accept the
    default); `max_completion_tokens` is used. Reasoning is the provider default
    unless `REASONING_EFFORT` is set (OpenRouter: `reasoning.effort`; OpenAI:
    `reasoning_effort`). With GPT-5.4-Mini on OpenRouter the default produced
    0 reasoning tokens.
14. **Robustness.** API errors (rate limit / timeout / connection / 5xx): up to 3
    attempts with exponential backoff. Parse failures: up to 3 re-calls.
    Context-length / bad-request errors: no retry. When a stage still fails it
    falls back to the previous stage's result and the prediction is marked
    `fallback_used`. Auth / quota errors abort the run (resume with `RUN_DIR`).
15. **Prompt text.** The six Appendix-A prompts (`verbatim: true`) are copied
    character-for-character from the paper, including `{t̂}` / `{â}` and
    typographic quotes / dashes; only PDF line wrapping and page numbers are
    dropped. Anything we added lives outside `template`
    (`appended_output_format`, Stage 0).
16. **Step indexing in Stage 4.** The paper's window is [max(1, t̂−2), …] with
    steps counted from 1. Our steps start at 0, so the lower bound is the first
    non-human step: 1 for hand-crafted (step 0 = the human task) and 0 for
    algorithm-generated (step 0 is an agent turn; gold = 0 for 20/126).
17. **Which answer is scored.** §3 (Metrics) writes t̂ / α̂, while §2.4 calls the
    Stage-4 output t̃ / ã the verified answer. `metrics` scores t̃ / ã;
    `metrics_stage3_t_hat` scores t̂ / α̂.


## Paper reference numbers (Who&When, Table 2)

| Backbone | Alg.-generated agent / step | Hand-crafted agent / step |
|---|---|---|
| GPT-5.4-Mini | 66.5 / 42.0 | 65.5 / 22.2 |
| DeepSeek V3.2 | 59.7 / 38.0 | 73.0 / 29.1 |
| MiniMax 2.5 | 68.0 / 46.0 | 66.3 / 25.9 |
