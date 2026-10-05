"""Stage 4: local re-search / adversarial verification t̂ -> t̃ (paper §2.4, Appendix A.4).

Window = [max(1, t̂ − 2), min(t̂ + 2, T)]. {candidates_block} holds, for every
step in the window, raw content, MID / LOW entries and typed transitions, plus
π's ERROR_SIGNALS (assumption #7; the prompt has no other evidence slot).

The verdict's decisive_step must lie in the window; otherwise (or on failure)
t̃ = t̂.
Output: ctx["t_tilde"], ctx["local"]
"""

from __future__ import annotations

from ._stage_common import StageRunner
from ..common.parsing import extract_json, format_transitions, to_int, transitions_involving

STAGE = "4_local_research"


def local_window(traj, t_hat: int):
    """[max(first, t̂ − 2), min(t̂ + 2, T)].

    The paper indexes steps from 1 (s_1 = first step). Our step numbers start at 0,
    and step 0 is the human task in hand-crafted data but a real agent turn in
    algorithm-generated data (where the gold step is 0 for 20/126 trajectories).
    So the lower bound is the first non-human step, not the literal 1.
    """
    nums = traj.step_numbers
    first = traj.first_agent_step
    lo, hi = max(first, t_hat - 2), min(t_hat + 2, nums[-1])
    return [n for n in nums if lo <= n <= hi]


def build_candidates_block(traj, ctx, window, t_hat: int) -> str:
    blocks = [f"## Expected behavior (from prior π)\n{ctx['error_signals']}"]
    for n in window:
        tag = "  ← PREDICTED DECISIVE STEP" if n == t_hat else ""
        blocks.append(
            f"### Step {n} ({traj.agent_of(n)}){tag}\n"
            f"- Raw content:\n{traj.by_step(n)['content']}\n"
            f"- Summary (MID): {ctx['mid_entries'].get(n, '(not a key moment)')}\n"
            f"- Evidence (LOW): {ctx['low_entries'].get(n, '(none)')}\n"
            f"- Typed transitions:\n{format_transitions(transitions_involving(ctx['transitions'], n))}"
        )
    return "\n\n".join(blocks)


def run(runner: StageRunner, traj, trace, ctx) -> None:
    t_hat = ctx["t_hat"]
    window = local_window(traj, t_hat)
    v = ctx["verification"]

    def parse(text):
        obj = extract_json(text)
        fv = obj.get("final_verdict") if obj else None
        n = to_int(fv.get("decisive_step")) if isinstance(fv, dict) else None
        if n is None:
            return None
        return {"step": n, "decision": str(fv.get("decision") or "").upper(),
                "agent": fv.get("decisive_agent"), "failure_mode": fv.get("failure_mode"),
                "description": str(fv.get("description") or ""),
                "confidence": str(fv.get("confidence") or "").lower(),
                "reasoning": str(fv.get("reasoning") or ""),
                "ranked_steps": obj.get("ranked_steps"),
                "candidates": obj.get("candidates"),
                "counterfactual_tests": obj.get("counterfactual_tests")}

    values = {"question": ctx["question"], "final_answer": ctx["final_answer"],
              "agents_block": ctx["agents_block"], "total_steps": traj.total_steps,
              "t̂": t_hat, "â": traj.agent_of(t_hat),
              "predicted_desc": v["description"] or "(no description)",
              "backward_chain": v["backward_chain"] or "(not available)",
              "candidates_block": build_candidates_block(traj, ctx, window, t_hat)}
    out = runner.call(traj, trace, STAGE, "local_research", values, parse)
    if out is not None and out["step"] not in window:
        runner.log.error(trajectory=traj.stem, stage=STAGE, error_type="parse_error",
                         message=f"decisive_step {out['step']} outside window {window}; keeping t̂",
                         action="fallback")
        out = None
    if out is None:
        runner.mark_fallback(trace, STAGE, "keep t̂")
        out = {"step": t_hat, "decision": "FALLBACK", "agent": None, "failure_mode": None,
               "description": v["description"], "confidence": "", "reasoning": v["reasoning"]}
    ctx["t_tilde"] = out["step"]
    ctx["local"] = out
    trace["local_window"] = window
    trace["t_tilde"] = out["step"]
