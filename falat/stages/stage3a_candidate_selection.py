"""Stage 3a: narrow C′ to the high-confidence set C_H (paper §2.3, Appendix A.3).

Keeps the top K (= 3) valid, de-duplicated steps in rank order.
Fallback: candidates that are the target of an error_shift first, then earliest.

Output: ctx["C_H"] = [{step, agent, reasoning}, ...]
"""

from __future__ import annotations

from ._stage_common import StageRunner
from ..common.parsing import extract_json, to_int
from ..common.trajectory import step_index

STAGE = "3a_candidate_selection"


def run(runner: StageRunner, traj, trace, ctx) -> None:
    valid = set(traj.step_numbers)

    def parse(text):
        obj = extract_json(text)
        items = obj.get("candidates") if obj else None
        if not isinstance(items, list):
            return None
        ranked = sorted((c for c in items if isinstance(c, dict)),
                        key=lambda c: to_int(c.get("rank")) or 99)
        picked, seen = [], set()
        for c in ranked:
            n = to_int(c.get("step"))
            if n in valid and n not in seen:
                seen.add(n)
                picked.append({"step": n, "agent": c.get("agent"),
                               "reasoning": str(c.get("reasoning") or "")})
        return picked or None

    section = ("## Typed Dependencies\n" + ctx["dependency_text"]) if ctx["dependency_text"] else ""
    values = {k: ctx[k] for k in ("question", "answer_type", "final_answer", "agents_block",
                                  "high_level", "mid_level")}
    values.update({"step_index": step_index(traj), "transitions_section": section})
    picked = runner.call(traj, trace, STAGE, "candidate_selection", values, parse)
    if picked is None:
        shift_targets = {t["to"] for t in ctx["transitions"] if t["label"] == "error_shift"}
        order = sorted(ctx["candidates"], key=lambda n: (n not in shift_targets, n))
        picked = [{"step": n, "agent": traj.agent_of(n), "reasoning": ""} for n in order]
        runner.mark_fallback(trace, STAGE, "error_shift targets first, then earliest")
    ctx["C_H"] = picked[: runner.top_k]
    trace["C_H"] = [c["step"] for c in ctx["C_H"]]
