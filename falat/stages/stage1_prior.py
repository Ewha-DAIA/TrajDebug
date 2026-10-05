"""Stage 1a: external prior π (paper §2.1, Appendix A.1 external-prior prompt).

Input : question, answer_type, constraints, agents_block (A, P, T), final_answer (ô).
        The intermediate trajectory is NOT shown.
Output: ctx["prior_text"]    full π (fed to the hierarchical prompt as {context_analysis})
        ctx["error_signals"] π's ERROR_SIGNALS section (used in 3b / 4 candidate blocks)
"""

from __future__ import annotations

from ._stage_common import StageRunner
from ..common.parsing import split_sections

STAGE = "1_prior"
SECTIONS = ["QUESTION_ANALYSIS", "AGENT_ROLES", "ERROR_SIGNALS"]


def run(runner: StageRunner, traj, trace, ctx) -> None:
    def parse(text):
        sec = split_sections(text, SECTIONS)
        return {"text": text.strip(), "sections": sec} if sec["ERROR_SIGNALS"] else None

    values = {k: ctx[k] for k in ("question", "answer_type", "constraints", "agents_block", "final_answer")}
    out = runner.call(traj, trace, STAGE, "prior", values, parse)
    if out is None:
        runner.mark_fallback(trace, STAGE, "prior unavailable")
        out = {"text": "(prior unavailable)", "sections": {s: "" for s in SECTIONS}}
    ctx["prior_text"] = out["text"]
    ctx["error_signals"] = out["sections"]["ERROR_SIGNALS"] or out["text"]
