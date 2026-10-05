#!/usr/bin/env python3
"""Run FALAT on a TrajDebug dataset and write TrajDebug-compatible predictions.

Each invocation creates its own run directory:

    falat/outputs/<dataset>/<model>_<YYYYmmdd-HHMMSS>[_<tag>]/
        config.json      run settings (model, k, K, clip, retries, git commit, prompt hash)
        predictions/     <stem>_critical_error.json   (scored by detector/score_steps.py)
        traces/          <stem>_trace.json            (every stage: prompt, raw responses, parsed, usage)
        logs/run.log     progress log
        logs/errors.jsonl one JSON line per error / retry / fallback
        summary.json     counts, token usage, elapsed time

Usage (from the TrajDebug root):

    python -m falat.run_falat --dataset whoandwhen_handcrafted --limit 5
    python -m falat.run_falat --dataset whoandwhen_handcrafted --dry_run --limit 3
    python -m falat.run_falat --run_dir <existing run dir>      # resume
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_processing.dataset_registry import canonical_key, dataset_path  # noqa: E402
from falat.common.llm_client import LLMClient  # noqa: E402
from falat.common.prompts import load_prompts, prompts_fingerprint  # noqa: E402
from falat.common.run_logger import RunLogger  # noqa: E402
from falat.common.stub_client import StubClient  # noqa: E402
from falat.common.trajectory import load_trajectory  # noqa: E402
from falat.pipeline import AbortRun, FALATPipeline, StageRunner  # noqa: E402

DEFAULT_PROMPTS = REPO_ROOT / "falat" / "prompts"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "falat" / "outputs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_argument_group("input / output")
    src.add_argument("--dataset", help="TrajDebug dataset key, e.g. whoandwhen_handcrafted")
    src.add_argument("--input", help="Directory of unified trajectories (overrides --dataset path)")
    src.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    src.add_argument("--run_dir", help="Resume an existing run directory (settings are re-read from its config.json)")
    src.add_argument("--tag", default="", help="Suffix for the run directory, e.g. noclip, clip20k, k5")
    src.add_argument("--limit", type=int, default=0, help="Only the first N trajectories (0 = all)")
    src.add_argument("--ids", default="", help="Comma-separated trajectory stems, e.g. 10,11,30")
    src.add_argument("--prompts_dir", default=str(DEFAULT_PROMPTS), help="Directory of per-stage prompt YAML files")
    src.add_argument("--run_dir_file", default="", help="Write the run directory path to this file (used by run_pipeline_falat.sh)")

    m = p.add_argument_group("model")
    m.add_argument("--model", default=os.getenv("DETECTOR_MODEL", ""))
    m.add_argument("--base_url", default=os.getenv("DETECTOR_BASE_URL") or os.getenv("OPENAI_BASE_URL")
                   or "https://api.openai.com/v1")
    m.add_argument("--max_output_tokens", type=int, default=32768)
    m.add_argument("--temperature", type=float, default=None,
                   help="Omit to use the model default (required for GPT-5-family models)")
    m.add_argument("--reasoning_effort", default=None,
                   choices=("none", "minimal", "low", "medium", "high"),
                   help="Reasoning effort sent to the model (omit = provider default; not stated in the paper)")
    m.add_argument("--request_timeout", type=float, default=600.0)
    m.add_argument("--dry_run", action="store_true", help="Use an offline stub instead of the API")

    f = p.add_argument_group("FALAT")
    f.add_argument("--k", type=int, default=10, help="Window size k for the hierarchical representation (§2.1)")
    f.add_argument("--top_k", type=int, default=3, help="|C_H| (paper: K = 3, §2.3)")
    f.add_argument("--clip_chars", type=int, default=0,
                   help="Clip each step to this many characters with clip_text_middle (0 = no clipping)")

    r = p.add_argument_group("robustness")
    r.add_argument("--max_attempts", type=int, default=3, help="API retries for rate limit / timeout / 5xx")
    r.add_argument("--parse_attempts", type=int, default=3, help="Re-calls when a response cannot be parsed")
    r.add_argument("--concurrency", type=int, default=4, help="Trajectories processed in parallel")
    r.add_argument("--quiet", action="store_true")
    return p.parse_args()


def _git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True).strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001
        return "unknown"


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-") or "model"


def _resolve_run(args: argparse.Namespace) -> Dict[str, Any]:
    """Create (or reload) the run directory and its config."""
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
        cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        cfg["resumed_at"] = datetime.now().isoformat(timespec="seconds")
        cfg["dry_run"] = cfg.get("dry_run", False)
        return {"run_dir": run_dir, "config": cfg}

    if not args.dataset and not args.input:
        sys.exit("Provide --dataset or --input (or --run_dir to resume).")
    dataset = canonical_key(args.dataset) if args.dataset else Path(args.input).name
    input_dir = Path(args.input).resolve() if args.input else dataset_path(dataset)
    model = "dry-run-stub" if args.dry_run else args.model
    if not model:
        sys.exit("DETECTOR_MODEL (or --model) is required.")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{_safe(model)}_{stamp}"
    if args.reasoning_effort and not args.dry_run:
        name += f"_reasoning-{args.reasoning_effort}"
    if args.tag:
        name += f"_{_safe(args.tag)}"
    run_dir = Path(args.output_root).resolve() / dataset / name
    cfg = {
        "method": "FALAT (arXiv:2606.00765) reproduction",
        "dataset": dataset,
        "input_dir": str(input_dir),
        "model": model,
        "base_url": None if args.dry_run else args.base_url,
        "dry_run": args.dry_run,
        "window_k": args.k,
        "top_k": args.top_k,
        "clip_chars": args.clip_chars or None,
        "max_output_tokens": args.max_output_tokens,
        "temperature": args.temperature,
        "reasoning_effort": None if args.dry_run else args.reasoning_effort,
        "max_attempts": args.max_attempts,
        "parse_attempts": args.parse_attempts,
        "request_timeout": args.request_timeout,
        "limit": args.limit,
        "ids": args.ids,
        "prompts_dir": str(Path(args.prompts_dir).resolve()),
        "prompts_sha256": prompts_fingerprint(Path(args.prompts_dir)),
        "git_commit": _git_commit(),
        "started_at": datetime.now().isoformat(timespec="seconds"),
    }
    return {"run_dir": run_dir, "config": cfg}


def _select_sources(cfg: Dict[str, Any]) -> List[Path]:
    files = sorted(Path(cfg["input_dir"]).glob("*.json"),
                   key=lambda p: (int(p.stem) if p.stem.isdigit() else 10**9, p.stem))
    if cfg.get("ids"):
        wanted = {s.strip() for s in str(cfg["ids"]).split(",") if s.strip()}
        files = [f for f in files if f.stem in wanted]
    if cfg.get("limit"):
        files = files[: int(cfg["limit"])]
    return files


def main() -> int:
    args = parse_args()
    resolved = _resolve_run(args)
    run_dir: Path = resolved["run_dir"]
    cfg = resolved["config"]
    pred_dir, trace_dir = run_dir / "predictions", run_dir / "traces"
    for d in (pred_dir, trace_dir):
        d.mkdir(parents=True, exist_ok=True)
    log = RunLogger(run_dir / "logs", verbose=not args.quiet)
    if args.run_dir_file:
        Path(args.run_dir_file).write_text(str(run_dir), encoding="utf-8")

    if cfg["dry_run"]:
        client = StubClient()
    else:
        api_key = os.getenv("DETECTOR_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not api_key:
            sys.exit("OPENAI_API_KEY (or DETECTOR_API_KEY) is not set. Put it in TrajDebug/.env.")
        client = LLMClient(base_url=cfg["base_url"], api_key=api_key, model=cfg["model"],
                           max_output_tokens=cfg["max_output_tokens"], temperature=cfg["temperature"],
                           reasoning_effort=cfg.get("reasoning_effort"),
                           max_attempts=cfg["max_attempts"], request_timeout=cfg["request_timeout"])
    cfg["client_params"] = client.describe_params()
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    prompts = load_prompts(Path(cfg["prompts_dir"]))
    cfg["prompt_files"] = {pid: spec["file"] for pid, spec in prompts.items()}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    runner = StageRunner(client=client, prompts=prompts, logger=log, window_k=cfg["window_k"],
                         top_k=cfg["top_k"], parse_attempts=cfg["parse_attempts"])
    pipeline = FALATPipeline(runner)

    sources = _select_sources(cfg)
    def _needs_run(src: Path) -> bool:
        """No prediction yet, or a prediction without a step (e.g. empty candidate set):
        resuming a run retries these and keeps every completed trajectory."""
        path = pred_dir / f"{src.stem}_critical_error.json"
        if not path.exists():
            return True
        try:
            return json.loads(path.read_text(encoding="utf-8")).get("critical_error") is None
        except (OSError, json.JSONDecodeError):
            return True

    todo = [s for s in sources if _needs_run(s)]
    log.info(f"FALAT run: {run_dir}")
    log.info(f"model={cfg['model']} dataset={cfg['dataset']} k={cfg['window_k']} K={cfg['top_k']} "
             f"reasoning={cfg.get('reasoning_effort') or 'default'} clip={cfg['clip_chars']} | {len(todo)} to run, {len(sources) - len(todo)} already done")

    abort = threading.Event()
    stats_lock = threading.Lock()
    stats = {"ok": 0, "with_fallback": 0, "failed": 0, "aborted": 0,
             "usage": {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "calls": 0}}
    t0 = time.time()

    def work(src: Path) -> None:
        if abort.is_set():
            with stats_lock:
                stats["aborted"] += 1
            return
        stem = src.stem

        def on_clip(step: int, n_chars: int) -> None:
            log.error(trajectory=stem, stage="load", error_type="clipped",
                      message=f"step {step}: {n_chars} chars clipped to {cfg['clip_chars']}", action="clipped")

        try:
            traj = load_trajectory(str(src), clip_chars=cfg["clip_chars"], on_clip=on_clip)
            pred, trace = pipeline.run(traj)
        except AbortRun as exc:
            abort.set()
            log.error(trajectory=stem, stage="run", error_type="auth", message=str(exc), action="abort_run")
            with stats_lock:
                stats["aborted"] += 1
            return
        except Exception as exc:  # noqa: BLE001
            log.error(trajectory=stem, stage="run", error_type="pipeline_error",
                      message=f"{type(exc).__name__}: {exc}", action="skipped")
            with stats_lock:
                stats["failed"] += 1
            return

        usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "calls": 0}
        for rec in trace["stages"].values():
            for u in rec.get("usage", []):
                usage["calls"] += 1
                for key in ("input_tokens", "output_tokens", "reasoning_tokens"):
                    usage[key] += u.get(key, 0) or 0
        trace["usage"] = usage
        pred["falat_trace"] = str((trace_dir / f"{stem}_trace.json").relative_to(run_dir))
        (trace_dir / f"{stem}_trace.json").write_text(json.dumps(trace, indent=2, ensure_ascii=False),
                                                      encoding="utf-8")
        (pred_dir / f"{stem}_critical_error.json").write_text(json.dumps(pred, indent=2, ensure_ascii=False),
                                                              encoding="utf-8")
        with stats_lock:
            stats["with_fallback" if pred["fallback_used"] else "ok"] += 1
            for key in usage:
                stats["usage"][key] += usage[key]
        ce = pred["critical_error"] or {}
        log.info(f"[{stem}] done: step={ce.get('critical_step')} agent={ce.get('critical_agent')} "
                 f"fallbacks={trace['fallback_stages'] or '-'} calls={usage['calls']} "
                 f"in/out tokens={usage['input_tokens']}/{usage['output_tokens']}")

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            futures = [pool.submit(work, s) for s in todo]
            for _ in as_completed(futures):
                pass
    finally:
        client.close()

    summary = {"run_dir": str(run_dir), "trajectories_selected": len(sources),
               "processed_this_invocation": len(todo), **stats,
               "elapsed_sec": round(time.time() - t0, 1),
               "finished_at": datetime.now().isoformat(timespec="seconds")}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info(f"summary: {json.dumps(summary)}")
    if abort.is_set():
        log.warning("Run aborted (authentication / quota). Fix the API key and resume with --run_dir.")
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
