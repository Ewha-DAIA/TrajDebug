#!/usr/bin/env bash
# Run the FALAT reproduction (arXiv:2606.00765) with any OpenAI-compatible endpoint
# and score it with both the paper's metrics and TrajDebug's score_steps.py.
#
# Settings are read from .env (OPENAI_API_KEY, OPENAI_BASE_URL, DETECTOR_MODEL)
# and can be overridden on the command line:
#
#   bash run_pipeline_falat.sh                              # both Who&When subsets
#   DATASETS="whoandwhen_handcrafted" LIMIT=5 bash run_pipeline_falat.sh   # quick test
#   DRY_RUN=1 LIMIT=3 bash run_pipeline_falat.sh            # offline, no API calls
#   CLIP_CHARS=20000 TAG=clip20k bash run_pipeline_falat.sh # clipped variant
#   REASONING_EFFORT=medium bash run_pipeline_falat.sh      # reasoning on (none|minimal|low|medium|high)
#   RUN_DIR=falat/outputs/<dataset>/<run> bash run_pipeline_falat.sh   # resume one run
#
# Optional:
#   DETECTOR_API_KEY / DETECTOR_BASE_URL   override the OPENAI_* values
#   DATASETS     (default: "whoandwhen_handcrafted whoandwhen_algorithm_generated")
#   OUTPUT_ROOT  (default: falat/outputs)
#   WINDOW_K=10  TOP_K=3  CLIP_CHARS=0  MAX_TOKENS=32768  TEMPERATURE=(model default)
#   REASONING_EFFORT=(provider default)
#   MAX_ATTEMPTS=3  PARSE_ATTEMPTS=3  FILE_CONCURRENCY=4  LIMIT=0  IDS=""  TAG=""

set -euo pipefail
cd "$(dirname "$0")"

# Load .env without overriding variables already set on the command line,
# e.g. `DETECTOR_MODEL=deepseek/deepseek-v3.2 bash run_pipeline_falat.sh`.
if [[ -f .env ]]; then
  while IFS='=' read -r key value; do
    key="${key//[[:space:]]/}"
    [[ -z "$key" || "$key" == \#* ]] && continue
    value="${value%\"}"; value="${value#\"}"
    if [[ -z "${!key+x}" ]]; then
      export "$key=$value"
    fi
  done < .env
fi

if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x .venv/bin/python ]]; then PYTHON_BIN=.venv/bin/python; else PYTHON_BIN=python3; fi
fi
BASE_URL="${DETECTOR_BASE_URL:-${OPENAI_BASE_URL:-https://api.openai.com/v1}}"
MODEL="${DETECTOR_MODEL:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-falat/outputs}"
WINDOW_K="${WINDOW_K:-10}"
TOP_K="${TOP_K:-3}"
CLIP_CHARS="${CLIP_CHARS:-0}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
TEMPERATURE="${TEMPERATURE:-}"
REASONING_EFFORT="${REASONING_EFFORT:-}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"
PARSE_ATTEMPTS="${PARSE_ATTEMPTS:-3}"
FILE_CONCURRENCY="${FILE_CONCURRENCY:-4}"
LIMIT="${LIMIT:-0}"
IDS="${IDS:-}"
TAG="${TAG:-}"
DRY_RUN="${DRY_RUN:-0}"
RUN_DIR="${RUN_DIR:-}"
read -r -a DATASET_LIST <<< "${DATASETS:-whoandwhen_handcrafted whoandwhen_algorithm_generated}"

if [[ "$DRY_RUN" != "1" && -z "$MODEL" ]]; then
  echo "DETECTOR_MODEL is required (set it in .env)." >&2
  exit 2
fi

score_run() {
  local run_dir="$1" input_dir="$2"
  echo "[score] FALAT metrics (agent / step / tolerance)"
  "$PYTHON_BIN" -m falat.score_falat --run_dir "$run_dir"
  echo "[score] TrajDebug score_steps.py"
  # Partial runs (LIMIT / IDS): restrict score_steps to the selected trajectories.
  local id_args=()
  if "$PYTHON_BIN" -c "import json,sys;c=json.load(open(sys.argv[1]));sys.exit(0 if (c.get('limit') or c.get('ids')) else 1)" "$run_dir/config.json"; then
    "$PYTHON_BIN" -c "
import json, sys
from falat.run_falat import _select_sources
cfg = json.load(open(sys.argv[1]))
json.dump([p.stem for p in _select_sources(cfg)], open(sys.argv[2], 'w'))" "$run_dir/config.json" "$run_dir/selected_ids.json"
    id_args=(--id-list-file "$run_dir/selected_ids.json")
  fi
  "$PYTHON_BIN" detector/score_steps.py \
    --unified-dir "$input_dir" --pred-dir "$run_dir/predictions" "${id_args[@]}" \
    --out "$run_dir/score_steps.json" > "$run_dir/logs/score_steps.log" 2>&1 \
    || echo "  score_steps.py failed; see $run_dir/logs/score_steps.log" >&2
}

if [[ -n "$RUN_DIR" ]]; then
  echo "FALAT: resuming $RUN_DIR"
  "$PYTHON_BIN" -m falat.run_falat --run_dir "$RUN_DIR" --concurrency "$FILE_CONCURRENCY"
  input_dir="$("$PYTHON_BIN" -c "import json,sys;print(json.load(open(sys.argv[1]))['input_dir'])" "$RUN_DIR/config.json")"
  score_run "$RUN_DIR" "$input_dir"
  exit 0
fi

echo "FALAT reproduction"
echo "  endpoint: $([[ "$DRY_RUN" == "1" ]] && echo 'dry run (no API calls)' || echo "$BASE_URL")"
echo "  model:    $([[ "$DRY_RUN" == "1" ]] && echo 'dry-run-stub' || echo "$MODEL")"
echo "  datasets: ${DATASET_LIST[*]}"
echo "  k=$WINDOW_K K=$TOP_K clip=$CLIP_CHARS reasoning=${REASONING_EFFORT:-default} limit=$LIMIT tag=${TAG:-none}"

for requested_dataset in "${DATASET_LIST[@]}"; do
  dataset="$("$PYTHON_BIN" -m data_processing.dataset_registry "$requested_dataset" --field key)"
  input_dir="$("$PYTHON_BIN" -m data_processing.dataset_registry "$requested_dataset" --field path)"
  if [[ ! -d "$input_dir" ]]; then
    echo "[$dataset] skipped: missing $input_dir" >&2
    continue
  fi

  run_dir_file="$(mktemp)"
  extra_args=()
  [[ "$DRY_RUN" == "1" ]] && extra_args+=(--dry_run)
  [[ -n "$TEMPERATURE" ]] && extra_args+=(--temperature "$TEMPERATURE")
  [[ -n "$REASONING_EFFORT" ]] && extra_args+=(--reasoning_effort "$REASONING_EFFORT")
  [[ -n "$IDS" ]] && extra_args+=(--ids "$IDS")
  [[ -n "$TAG" ]] && extra_args+=(--tag "$TAG")

  echo "[$dataset] FALAT stages 0-4"
  set +e
  "$PYTHON_BIN" -m falat.run_falat \
    --dataset "$dataset" --output_root "$OUTPUT_ROOT" \
    --model "$MODEL" --base_url "$BASE_URL" \
    --k "$WINDOW_K" --top_k "$TOP_K" --clip_chars "$CLIP_CHARS" \
    --max_output_tokens "$MAX_TOKENS" --max_attempts "$MAX_ATTEMPTS" \
    --parse_attempts "$PARSE_ATTEMPTS" --concurrency "$FILE_CONCURRENCY" \
    --limit "$LIMIT" --run_dir_file "$run_dir_file" "${extra_args[@]}"
  status=$?
  set -e
  run_dir="$(cat "$run_dir_file" 2>/dev/null || true)"
  rm -f "$run_dir_file"
  if [[ -z "$run_dir" ]]; then
    echo "[$dataset] run failed before creating a run directory" >&2
    exit 1
  fi
  if [[ $status -eq 3 ]]; then
    echo "[$dataset] aborted (API key / quota). Fix .env, then resume: RUN_DIR=$run_dir bash run_pipeline_falat.sh" >&2
    exit 3
  fi

  echo "[$dataset] Evaluation"
  score_run "$run_dir" "$input_dir"
done

echo "Done. Results are under $OUTPUT_ROOT/."
