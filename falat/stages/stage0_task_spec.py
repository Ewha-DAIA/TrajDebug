"""Stage 0 (assumption, not in the paper): fill {answer_type} and {constraints}.

Prompt: prompts/stage0_task_spec.yaml. Sees only the question.
Fallback: both values become "not specified".
"""

from __future__ import annotations

from ._stage_common import StageRunner
from ..common.parsing import extract_json

STAGE = "0_task_spec"


def run(runner: StageRunner, traj, trace, ctx) -> None:
    def parse(text):
        obj = extract_json(text)
        if obj and obj.get("answer_type"):
            return {"answer_type": str(obj["answer_type"]),
                    "constraints": str(obj.get("constraints") or "None stated")}
        return None

    out = runner.call(traj, trace, STAGE, "task_spec", {"question": traj.question}, parse)
    if out is None:
        runner.mark_fallback(trace, STAGE, "answer_type/constraints = not specified")
        out = {"answer_type": "not specified", "constraints": "not specified"}
    ctx.update(out)
