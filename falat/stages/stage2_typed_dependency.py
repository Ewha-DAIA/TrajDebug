"""Stage 2: typed dependencies for candidate pruning, C -> C′ (paper §2.2, Appendix A.2).

The LLM labels ordered candidate pairs (follow_up, redundancy, no_influence,
error_shift, correction, dead_end) and writes the Candidate-Pruning Summary and
Value Flow. Pruning is left to the LLM (assumption #6); the full output is
passed to Stage 3a as {transitions_section}.

Output: ctx["dependency_text"]  full A.2 output
        ctx["transitions"]      [{from, to, label, relationship}] (parsed, for 3b / 4)
"""

from __future__ import annotations

from ._stage_common import StageRunner
from ..common.parsing import parse_transitions
from ..common.trajectory import step_index

STAGE = "2_typed_dependency"


def run(runner: StageRunner, traj, trace, ctx) -> None:
    def parse(text):
        transitions = parse_transitions(text)
        if transitions or "Typed Dependencies" in text or "Candidate-Pruning" in text:
            return {"text": text.strip(), "transitions": transitions}
        return None

    values = {"high_level": ctx["high_level"], "mid_level": ctx["mid_level"],
              "low_level": ctx["low_level"], "num_steps": traj.total_steps,
              "step_index": step_index(traj)}
    out = runner.call(traj, trace, STAGE, "typed_dependency", values, parse)
    if out is None:
        runner.mark_fallback(trace, STAGE, "no typed dependencies")
        out = {"text": "", "transitions": []}
    ctx["dependency_text"] = out["text"]
    ctx["transitions"] = out["transitions"]
    trace["transitions"] = out["transitions"]
