#!/usr/bin/env python3
"""Score a FALAT run with the paper's metrics plus stage-level diagnostics.

Paper metrics (FALAT §3, following Who&When), for the final answer t̃ / ã
(``metrics``) and for the Stage-3 answer t̂ / α̂ (``metrics_stage3_t_hat``):
  * agent-level accuracy        predicted agent == metadata.extra.mistake_agent
  * step-level accuracy         predicted step  == metadata.annotation.critical_error_step
  * step accuracy, tolerance k  |pred - gold| <= k for k = 1..5

Trajectories without a prediction count as wrong (same convention as
detector/score_steps.py). The TrajDebug-native report is produced separately by
detector/score_steps.py (run_pipeline_falat.sh runs both).

Diagnostics from traces (where in the search the gold step was lost):
  * gold in C      : gold step among the Stage-1 candidates
  * gold in C_H    : gold step among the Stage-3a high-confidence candidates
  * stage3 correct : t̂ (before local re-search) == gold
  * stage4 effect  : how often Stage 4 fixed / broke / kept the Stage-3 answer

Usage:
    python -m falat.score_falat --run_dir falat/outputs/whoandwhen_handcrafted/<run>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from falat.common.trajectory import agent_key, normalize_agent  # noqa: E402

TOLERANCES = (1, 2, 3, 4, 5)


def _pct(num: int, den: int) -> Optional[float]:
    return round(100.0 * num / den, 2) if den else None


def score(run_dir: Path) -> Dict[str, Any]:
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    gold_files = sorted(Path(cfg["input_dir"]).glob("*.json"))
    selected = {p.stem for p in (run_dir / "predictions").glob("*_critical_error.json")}
    if cfg.get("limit") or cfg.get("ids"):
        # Partial runs: score only the trajectories that were selected for this run.
        from falat.run_falat import _select_sources
        wanted = {p.stem for p in _select_sources(cfg)}
        gold_files = [g for g in gold_files if g.stem in wanted]

    n = 0
    c = {"step": 0, "agent": 0, "missing": 0, "fallback": 0,
         "gold_in_C": 0, "gold_in_CH": 0, "stage3": 0,
         "s4_fixed": 0, "s4_broke": 0, "s4_kept_right": 0, "s4_kept_wrong": 0, "s4_moved_wrong": 0}
    tol = {k: 0 for k in TOLERANCES}
    s3 = {"step": 0, "agent": 0, **{f"tol{k}": 0 for k in TOLERANCES}}
    per_traj = []
    for g in gold_files:
        data = json.loads(g.read_text(encoding="utf-8"))
        meta = data.get("metadata") or {}
        gold_step = (meta.get("annotation") or {}).get("critical_error_step")
        if gold_step is None:
            continue
        gold_step = int(gold_step)
        gold_agent = (meta.get("extra") or {}).get("mistake_agent")
        n += 1

        pred_path = run_dir / "predictions" / f"{g.stem}_critical_error.json"
        pred = json.loads(pred_path.read_text(encoding="utf-8")) if pred_path.exists() else None
        ce = (pred or {}).get("critical_error") or {}
        p_step, p_agent = ce.get("critical_step"), ce.get("critical_agent")
        if pred is None or p_step is None:
            c["missing"] += 1
        if pred and pred.get("fallback_used"):
            c["fallback"] += 1

        step_ok = p_step == gold_step
        agent_ok = p_agent is not None and agent_key(p_agent) == agent_key(gold_agent)
        c["step"] += step_ok
        c["agent"] += agent_ok
        for k in TOLERANCES:
            tol[k] += p_step is not None and abs(p_step - gold_step) <= k

        trace_path = run_dir / "traces" / f"{g.stem}_trace.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8")) if trace_path.exists() else {}
        in_c = gold_step in (trace.get("candidates_C") or [])
        in_ch = gold_step in (trace.get("C_H") or [])
        t_hat = trace.get("t_hat")
        c["gold_in_C"] += in_c
        c["gold_in_CH"] += in_ch
        c["stage3"] += t_hat == gold_step
        t_hat_agent = None
        if t_hat is not None:
            msgs = data.get("messages") or []
            if 0 <= t_hat < len(msgs):
                t_hat_agent = normalize_agent(msgs[t_hat].get("name") or msgs[t_hat].get("role"))
            s3["step"] += t_hat == gold_step
            s3["agent"] += t_hat_agent is not None and agent_key(t_hat_agent) == agent_key(gold_agent)
            for k in TOLERANCES:
                s3[f"tol{k}"] += abs(t_hat - gold_step) <= k
        if t_hat is not None and p_step is not None:
            if t_hat != gold_step and p_step == gold_step:
                c["s4_fixed"] += 1
            elif t_hat == gold_step and p_step != gold_step:
                c["s4_broke"] += 1
            elif t_hat == p_step == gold_step:
                c["s4_kept_right"] += 1
            elif t_hat == p_step:
                c["s4_kept_wrong"] += 1
            else:
                c["s4_moved_wrong"] += 1
        per_traj.append({"trajectory": g.stem, "gold_step": gold_step, "pred_step": p_step,
                         "gold_agent": gold_agent, "pred_agent": p_agent,
                         "step_correct": step_ok, "agent_correct": agent_ok,
                         "gold_in_C": in_c, "gold_in_C_H": in_ch, "t_hat": t_hat,
                         "fallback_stages": (pred or {}).get("fallback_stages")})

    usage = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "calls": 0}
    for tp in (run_dir / "traces").glob("*_trace.json"):
        for key, val in (json.loads(tp.read_text(encoding="utf-8")).get("usage") or {}).items():
            usage[key] = usage.get(key, 0) + (val or 0)

    return {
        "run_dir": str(run_dir),
        # Summed over the final trace of every trajectory (summary.json only covers
        # the last invocation when a run was resumed).
        "usage_total": usage,
        "dataset": cfg["dataset"],
        "model": cfg["model"],
        "num_labeled": n,
        "num_predictions": len(selected),
        "metrics": {
            "agent_accuracy": _pct(c["agent"], n),
            "step_accuracy": _pct(c["step"], n),
            **{f"step_accuracy_tol{k}": _pct(tol[k], n) for k in TOLERANCES},
        },
        # Paper §3 (Metrics) is written with t̂ / α̂ (Stage 3), while §2.4 calls the
        # Stage-4 output t̃ / ã the verified answer. Both are reported.
        "metrics_stage3_t_hat": {
            "agent_accuracy": _pct(s3["agent"], n),
            "step_accuracy": _pct(s3["step"], n),
            **{f"step_accuracy_tol{k}": _pct(s3[f"tol{k}"], n) for k in TOLERANCES},
        },
        "diagnostics": {
            "missing_predictions": c["missing"],
            "trajectories_with_fallback": c["fallback"],
            "gold_in_C_pct": _pct(c["gold_in_C"], n),
            "gold_in_C_H_pct": _pct(c["gold_in_CH"], n),
            "stage3_step_accuracy": _pct(c["stage3"], n),
            "stage4_effect": {k[3:]: v for k, v in c.items() if k.startswith("s4_")},
        },
        "per_trajectory": per_traj,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", default="", help="Default: <run_dir>/score_falat.json")
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    report = score(run_dir)
    out = Path(args.out) if args.out else run_dir / "score_falat.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    m, d = report["metrics"], report["diagnostics"]
    print(f"[{report['dataset']} | {report['model']}] n={report['num_labeled']}")
    print(f"  agent acc {m['agent_accuracy']} | step acc {m['step_accuracy']} | "
          + " ".join(f"±{k} {m[f'step_accuracy_tol{k}']}" for k in TOLERANCES))
    m3 = report["metrics_stage3_t_hat"]
    print(f"  (stage-3 t̂/α̂) agent acc {m3['agent_accuracy']} | step acc {m3['step_accuracy']}")
    print(f"  gold in C {d['gold_in_C_pct']} | gold in C_H {d['gold_in_C_H_pct']} | "
          f"stage3 acc {d['stage3_step_accuracy']} | stage4 {d['stage4_effect']}")
    print(f"  missing {d['missing_predictions']} | with fallback {d['trajectories_with_fallback']}")
    print(f"  -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
