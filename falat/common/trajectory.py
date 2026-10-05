"""Turn a TrajDebug unified trajectory into the inputs FALAT's prompts need.

Step numbers always come from the unified ``step`` field (same numbering as
``metadata.annotation.critical_error_step``), never from list positions.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DETECTOR_DIR = _REPO_ROOT / "detector"
if str(_DETECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(_DETECTOR_DIR))

from utils.llm_compression import clip_text_middle  # noqa: E402
from utils.trajectory_utils import flatten_unified_trajectory  # noqa: E402

_TEAM_START = "we have assembled the following team:"
_TEAM_END_MARKERS = ("Here is an initial fact sheet", "Here is the fact sheet", "Here is the plan")
_FINAL_ANSWER = re.compile(r"FINAL ANSWER:\s*(.+?)(?:\s*SCENARIO\.PY COMPLETE|\n|$)", re.IGNORECASE)
_TERMINATE_ONLY = re.compile(r"^\s*(terminate\.?|exit|done\.?)?\s*$", re.IGNORECASE)


def normalize_agent(name: Optional[str]) -> str:
    """'Orchestrator (thought)' / 'Orchestrator (-> WebSurfer)' -> 'Orchestrator'."""
    if not name:
        return "unknown"
    return re.sub(r"\s*\(.*\)\s*$", "", str(name)).strip() or str(name).strip()


def agent_key(name: Optional[str]) -> str:
    """Comparison key for agent accuracy (case/underscore insensitive)."""
    return re.sub(r"[^a-z0-9]", "", normalize_agent(name).lower())


@dataclass
class Trajectory:
    source: str
    stem: str
    task_id: str
    question: str
    steps: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    clipped_steps: List[int] = field(default_factory=list)

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def step_numbers(self) -> List[int]:
        return [s["step"] for s in self.steps]

    @property
    def first_agent_step(self) -> int:
        """First step that is not the human task message (paper's s_1)."""
        for s in self.steps:
            if s["role"] != "user" and (s.get("name") or "").lower() != "human":
                return s["step"]
        return self.steps[0]["step"] if self.steps else 0

    def by_step(self, n: int) -> Optional[Dict[str, Any]]:
        for s in self.steps:
            if s["step"] == n:
                return s
        return None

    def agent_of(self, n: int) -> str:
        s = self.by_step(n)
        return normalize_agent(s["message_role"]) if s else "unknown"


def load_trajectory(path: str, clip_chars: Optional[int] = None,
                    on_clip: Optional[Callable[[int, int], None]] = None) -> Trajectory:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    meta = data.get("metadata") or {}
    steps = flatten_unified_trajectory(data)
    clipped: List[int] = []
    if clip_chars:
        for s in steps:
            if len(s["content"]) > clip_chars:
                if on_clip:
                    on_clip(s["step"], len(s["content"]))
                s["content"] = clip_text_middle(s["content"], clip_chars)
                clipped.append(s["step"])
    question = str(meta.get("task_description") or "").strip()
    if not question and steps:
        question = steps[0]["content"].strip()
    return Trajectory(
        source=str(path),
        stem=Path(path).stem,
        task_id=str(meta.get("task_id") or Path(path).stem),
        question=question,
        steps=steps,
        metadata=meta,
        clipped_steps=clipped,
    )


# ---------------------------------------------------------------- prompt inputs


def render_steps(traj: Trajectory, step_numbers: List[int], marked: Optional[set] = None) -> str:
    """Same layout as jiyul's minimal baseline ``_render_trajectory_block``:
    ``Step {step} [role: {message_role}]:`` followed by the content."""
    marked = marked or set()
    out: List[str] = []
    for n in step_numbers:
        s = traj.by_step(n)
        if s is None:
            continue
        tag = "  <<< CANDIDATE" if n in marked else ""
        out.append(f"Step {s['step']} [role: {s['message_role']}]:{tag}\n{s['content']}")
    return "\n\n".join(out)


def step_index(traj: Trajectory, step_numbers: Optional[List[int]] = None) -> str:
    nums = step_numbers if step_numbers is not None else traj.step_numbers
    return "\n".join(f"Step {n}: {traj.agent_of(n)}" for n in nums)


def agents_block(traj: Trajectory) -> str:
    """Agent roster with roles (paper's A, P, T).

    * algorithm-generated: ``metadata.extra.system_prompt`` (per-agent system prompts).
    * hand-crafted (Magentic-One): the team description in the Orchestrator's
      initial plan ("we have assembled the following team: ...").
    * otherwise: just the agent names.
    """
    names: List[str] = []
    for s in traj.steps:
        if s["role"] == "user" or (s.get("name") or "").lower() == "human":
            continue
        a = normalize_agent(s["message_role"])
        if a not in names:
            names.append(a)

    extra = traj.metadata.get("extra") or {}
    sys_prompts = extra.get("system_prompt")
    if isinstance(sys_prompts, dict) and sys_prompts:
        parts = []
        for agent, prompt in sys_prompts.items():
            parts.append(f"### {agent}\n{str(prompt).strip()}")
        missing = [n for n in names if agent_key(n) not in {agent_key(k) for k in sys_prompts}]
        if missing:
            parts.append("### Other participants\n" + "\n".join(f"- {n}" for n in missing))
        return "\n\n".join(parts)

    team = _magentic_team(traj)
    if team:
        lines = [team]
        if any(agent_key(n) == "orchestrator" for n in names):
            lines.insert(0, "Orchestrator: plans the task, delegates sub-tasks to the team members "
                            "below, tracks progress, and decides when to give the final answer.")
        return "\n".join(lines)

    return "\n".join(f"- {n}" for n in names)


def _magentic_team(traj: Trajectory) -> str:
    for s in traj.steps[:4]:
        text = s["content"]
        i = text.lower().find(_TEAM_START)
        if i < 0:
            continue
        body = text[i + len(_TEAM_START):]
        end = len(body)
        for marker in _TEAM_END_MARKERS:
            j = body.find(marker)
            if 0 <= j < end:
                end = j
        return body[:end].strip()
    return ""


def final_answer(traj: Trajectory, limit: int = 3000) -> str:
    """The trajectory's observed (wrong) final output, ô.

    Prefers an explicit ``FINAL ANSWER: ...`` line (Magentic-One logs it in the
    termination message); otherwise the last message that is not a bare
    TERMINATE marker (e.g. runs cut off by max rounds / max time).
    """
    for s in reversed(traj.steps):
        m = _FINAL_ANSWER.search(s["content"])
        if m:
            return (f"(Step {s['step']}, {normalize_agent(s['message_role'])})\n"
                    f"FINAL ANSWER: {m.group(1).strip()}")
    for s in reversed(traj.steps):
        text = s["content"].strip()
        if not text or _TERMINATE_ONLY.match(text):
            continue
        text = re.sub(r"\s*TERMINATE\s*$", "", text).strip()
        return f"(Step {s['step']}, {normalize_agent(s['message_role'])})\n" + clip_text_middle(text, limit)
    return "(no final answer found)"


def windows(traj: Trajectory, k: int) -> List[List[int]]:
    """Non-overlapping windows of k consecutive steps (paper §2.1)."""
    nums = traj.step_numbers
    return [nums[i:i + k] for i in range(0, len(nums), k)]
