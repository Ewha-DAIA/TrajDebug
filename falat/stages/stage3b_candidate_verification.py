"""Stage 3b: counterfactual verification C_H -> t̂ (paper §2.3, Appendix A.3).

Prompt slots (assumption #7):
  {candidates_block} π's ERROR_SIGNALS + per candidate: 3a reasoning, MID entry,
                     typed transitions involving the step
  {raw_steps}        raw steps [i-1, i, i+1] around each candidate, candidates marked
  {low_level}        LOW entries of the candidates

Fallback: the top-ranked C_H candidate.
Output: ctx["t_hat"], ctx["verification"] (description / backward_chain feed Stage 4)
"""

from __future__ import annotations

from ._stage_common import StageRunner
from ..common.parsing import extract_json, format_transitions, to_int, transitions_involving
from ..common.trajectory import render_steps, step_index

STAGE = "3b_candidate_verification"


def build_candidates_block(traj, ctx) -> str:
    blocks = [f"## Expected behavior (from prior π)\n{ctx['error_signals']}"]
    for i, c in enumerate(ctx["C_H"], 1):
        n = c["step"]
        blocks.append(
            f"### Candidate {i}: Step {n} ({traj.agent_of(n)})\n"
            f"- Why selected: {c['reasoning'] or '(not given)'}\n"
            f"- Summary (MID): {ctx['mid_entries'].get(n, '(not a key moment)')}\n"
            f"- Typed transitions:\n{format_transitions(transitions_involving(ctx['transitions'], n))}"
        )
    return "\n\n".join(blocks)


def run(runner: StageRunner, traj, trace, ctx) -> None:
    valid = set(traj.step_numbers)
    cand_steps = [c["step"] for c in ctx["C_H"]]
    neighborhood = sorted({m for n in cand_steps for m in (n - 1, n, n + 1) if m in valid})
    linked_low = [ctx["low_entries"][n] for n in cand_steps if n in ctx["low_entries"]]

    def parse(text):
        obj = extract_json(text)
        de = obj.get("decisive_error") if obj else None
        n = to_int(de.get("step")) if isinstance(de, dict) else None
        if n is None or n not in valid:
            return None
        return {"step": n, "agent": de.get("agent"),
                "description": str(de.get("description") or ""),
                "backward_chain": str(de.get("backward_chain") or ""),
                "reasoning": str(de.get("reasoning") or ""),
                "evaluations": obj.get("evaluations")}

    values = {"question": ctx["question"], "final_answer": ctx["final_answer"],
              "step_index": step_index(traj),
              "candidates_block": build_candidates_block(traj, ctx),
              "raw_steps": render_steps(traj, neighborhood, marked=set(cand_steps)),
              "low_level": "\n\n".join(linked_low) or "(none)"}
    out = runner.call(traj, trace, STAGE, "candidate_verification", values, parse)
    if out is None:
        top = ctx["C_H"][0]
        out = {"step": top["step"], "agent": top["agent"], "description": top["reasoning"],
               "backward_chain": "(not available)", "reasoning": "fallback: top-ranked C_H candidate"}
        runner.mark_fallback(trace, STAGE, "top-ranked C_H candidate")
    ctx["t_hat"] = out["step"]
    ctx["verification"] = out
    trace["t_hat"] = out["step"]
