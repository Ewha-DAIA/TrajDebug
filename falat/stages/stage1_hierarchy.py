"""Stage 1b: hierarchical representation M and candidate set C (paper §2.1).

1. Split the trajectory into non-overlapping k-step windows (k = 10, assumption #1).
2. Per window, call the A.1 hierarchical prompt -> HIGH / MID / LOW.
3. Consolidate (code, no LLM):
     C = steps with a MID or LOW entry
     R = maximal runs of adjacent candidate steps; runs spanning a window
         boundary are grouped in {mid_level} (assumption #2).

Output: ctx["high_level"], ctx["mid_level"], ctx["low_level"],
        ctx["mid_entries"], ctx["low_entries"] ({step: entry}), ctx["candidates"] (C)
"""

from __future__ import annotations

import re
from typing import Dict, List

from ._stage_common import StageRunner
from ..common.parsing import (
    contiguous_regions,
    extract_json,
    parse_step_entries,
    split_sections,
    to_int,
)
from ..common.trajectory import render_steps, step_index, windows

SECTIONS = ["HIGH_LEVEL", "MID_LEVEL", "LOW_LEVEL"]


def run(runner: StageRunner, traj, trace, ctx) -> None:
    valid = set(traj.step_numbers)
    highs: List[str] = []
    mid: Dict[int, str] = {}
    low: Dict[int, str] = {}
    step_window: Dict[int, int] = {}

    def parse(text):
        sec = split_sections(text, SECTIONS)
        if not (sec["HIGH_LEVEL"] or sec["MID_LEVEL"]):
            sec = _sections_from_json(text)
        if not (sec["HIGH_LEVEL"] or sec["MID_LEVEL"]):
            sec = _sections_from_bare_steps(text)
        if not (sec["HIGH_LEVEL"] or sec["MID_LEVEL"]):
            return None
        return {"sections": sec,
                "mid": parse_step_entries(sec["MID_LEVEL"]),
                "low": parse_step_entries(sec["LOW_LEVEL"])}

    for w_idx, w in enumerate(windows(traj, runner.window_k)):
        for n in w:
            step_window[n] = w_idx
        values = {
            "context_analysis": ctx["prior_text"],
            "question": ctx["question"], "answer_type": ctx["answer_type"],
            "constraints": ctx["constraints"], "agents_block": ctx["agents_block"],
            "final_answer": ctx["final_answer"],
            "num_steps": len(w),
            "step_index": step_index(traj, w),
            "steps_block": render_steps(traj, w),
        }
        stage = f"1_hierarchical_w{w_idx}_steps{w[0]}-{w[-1]}"
        out = runner.call(traj, trace, stage, "hierarchical", values, parse)
        if out is None:
            runner.mark_fallback(trace, stage, "window skipped")
            continue
        highs.append(f"[Steps {w[0]}-{w[-1]}]\n{out['sections']['HIGH_LEVEL']}")
        _merge(mid, out["mid"], valid)
        _merge(low, out["low"], valid)

    candidates = sorted(set(mid) | set(low))
    regions = contiguous_regions(candidates)
    cross = [r for r in regions if len({step_window.get(n) for n in r}) > 1]
    ctx.update({
        "high_level": "\n\n".join(highs) or "(none)",
        "mid_entries": mid,
        "low_entries": low,
        "candidates": candidates,
        "mid_level": render_mid_level(mid, regions, cross),
        "low_level": "\n\n".join(low[n] for n in sorted(low)) or "(none)",
    })
    trace["candidates_C"] = candidates
    trace["regions_R"] = regions
    trace["cross_window_regions"] = cross


def _sections_from_json(text: str) -> Dict[str, str]:
    """Some responses come back as JSON ({"high_level": [...], "mid_level": [...], ...})
    instead of ===SECTIONS===; convert them to the same text layout."""
    obj = extract_json(text) or {}
    lowered = {str(k).lower().replace("-", "_"): v for k, v in obj.items()}
    out = {}
    for name in SECTIONS:
        value = lowered.get(name.lower(), "")
        if isinstance(value, list):
            parts = []
            for item in value:
                if isinstance(item, dict):
                    step = to_int(item.get("step"))
                    agent = item.get("agent") or item.get("agent_name") or ""
                    body = "\n".join(f"- {k}: {v}" for k, v in item.items() if k not in ("step", "agent"))
                    parts.append(f"### Step {step} ({agent})\n{body}" if step is not None else body)
                else:
                    parts.append(f"- {item}" if name == "HIGH_LEVEL" else str(item))
            value = "\n".join(parts)
        out[name] = str(value or "").strip()
    return out


def _sections_from_bare_steps(text: str) -> Dict[str, str]:
    """No section headings at all, only '### Step N (Agent)' entries: keep the
    entries as MID_LEVEL (they cannot be told apart from LOW) and any leading
    text as HIGH_LEVEL."""
    if not parse_step_entries(text):
        return {name: "" for name in SECTIONS}
    first = re.search(r"^#{2,4}\s*Step\s+\d+", text, re.MULTILINE)
    return {"HIGH_LEVEL": text[: first.start()].strip(), "MID_LEVEL": text[first.start():].strip(),
            "LOW_LEVEL": ""}


def _merge(into: Dict[int, str], entries: Dict[int, str], valid: set) -> None:
    for n, entry in entries.items():
        if n in valid:
            into[n] = (into[n] + "\n" + entry) if n in into else entry


def render_mid_level(mid: Dict[int, str], regions: List[List[int]], cross: List[List[int]]) -> str:
    """MID entries in step order; window-crossing regions are grouped and labelled
    so Stage 2 sees them as one suspicious span (§2.1 consolidation)."""
    cross_keys = {tuple(r) for r in cross}
    parts: List[str] = []
    for r in regions:
        entries = [mid[n] for n in r if n in mid]
        if not entries:
            continue
        if tuple(r) in cross_keys:
            parts.append(f"#### (cross-window region: Steps {r[0]}-{r[-1]})\n" + "\n\n".join(entries))
        else:
            parts.extend(entries)
    return "\n\n".join(parts) or "(none)"
