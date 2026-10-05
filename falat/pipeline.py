"""FALAT pipeline: runs the stage modules in order and builds the prediction.

    stage0_task_spec               (assumption) answer_type / constraints
    stage1_prior                   π                                   A.1
    stage1_hierarchy               M per k-window, C, R                A.1 + code
    stage2_typed_dependency        labels, pruning (C′), value flow    A.2
    stage3a_candidate_selection    C_H (top K)                         A.3
    stage3b_candidate_verification t̂                                    A.3
    stage4_local_research          t̃                                    A.4

The prediction file follows TrajDebug's ``<stem>_critical_error.json`` format
(``critical_error.critical_step``) plus ``critical_agent`` for agent accuracy.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from .stages import (
    stage0_task_spec,
    stage1_hierarchy,
    stage1_prior,
    stage2_typed_dependency,
    stage3a_candidate_selection,
    stage3b_candidate_verification,
    stage4_local_research,
)
from .stages._stage_common import AbortRun, StageRunner  # noqa: F401  (re-exported)
from .common.trajectory import Trajectory, agents_block, final_answer

STAGES_AFTER_C = (
    stage2_typed_dependency,
    stage3a_candidate_selection,
    stage3b_candidate_verification,
    stage4_local_research,
)


class FALATPipeline:
    def __init__(self, runner: StageRunner):
        self.runner = runner

    def run(self, traj: Trajectory) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        trace: Dict[str, Any] = {"trajectory": traj.stem, "source": traj.source,
                                 "total_steps": traj.total_steps, "stages": {},
                                 "fallback_stages": [], "clipped_steps": traj.clipped_steps}
        ctx: Dict[str, Any] = {
            "question": traj.question,
            "agents_block": agents_block(traj),
            "final_answer": final_answer(traj),
        }
        trace["inputs"] = {"agents_block": ctx["agents_block"], "final_answer": ctx["final_answer"]}

        stage0_task_spec.run(self.runner, traj, trace, ctx)
        stage1_prior.run(self.runner, traj, trace, ctx)
        stage1_hierarchy.run(self.runner, traj, trace, ctx)
        if not ctx["candidates"]:
            self.runner.mark_fallback(trace, "consolidation", "empty candidate set C")
            self.runner.log.error(trajectory=traj.stem, stage="consolidation", error_type="fallback",
                                  message="empty candidate set C; no prediction", action="skipped")
            return build_prediction(traj, trace, ctx, step=None), trace
        for stage in STAGES_AFTER_C:
            stage.run(self.runner, traj, trace, ctx)
        return build_prediction(traj, trace, ctx, step=ctx["t_tilde"]), trace


def build_prediction(traj: Trajectory, trace: Dict[str, Any], ctx: Dict[str, Any],
                     step: Optional[int]) -> Dict[str, Any]:
    local = ctx.get("local") or {}
    confidence = {"low": 0.3, "moderate": 0.6, "high": 0.9}.get(local.get("confidence", ""), 0.5)
    critical = None
    if step is not None:
        critical = {
            "critical_step": step,
            "critical_agent": traj.agent_of(step),
            "llm_agent": local.get("agent"),
            "reason": local.get("description") or ctx.get("verification", {}).get("description", ""),
            "failure_mode": local.get("failure_mode"),
            "stage3_step": ctx.get("t_hat"),
            "stage4_decision": local.get("decision"),
        }
    return {
        "task_id": traj.task_id,
        "trajectory_source": traj.source,
        "critical_error": critical,
        "error_summary": {
            "total_steps": traj.total_steps,
            "critical_at": f"Step {step}" if step is not None else None,
            "confidence": confidence,
        },
        "baseline": "falat",
        "fallback_used": bool(trace["fallback_stages"]),
        "fallback_stages": trace["fallback_stages"],
    }
