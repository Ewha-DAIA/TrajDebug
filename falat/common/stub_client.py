"""Offline stand-in for LLMClient used by ``--dry_run``.

Returns syntactically valid, content-free answers for every FALAT stage so the
whole pipeline (rendering, parsing, consolidation, fallbacks, output files,
scoring) can be exercised without an API key or cost. Predictions from a dry
run are meaningless.
"""

from __future__ import annotations

import json
import re

from .llm_client import LLMResult


class StubClient:
    model = "dry-run-stub"

    def describe_params(self):
        return {"stub": True}

    def chat(self, prompt: str, on_retry=None) -> LLMResult:
        return LLMResult(text=self._answer(prompt), usage={"input_tokens": len(prompt) // 4,
                                                          "output_tokens": 0, "reasoning_tokens": 0})

    def close(self):
        pass

    @staticmethod
    def _steps_after(prompt: str, header: str):
        body = prompt.split(header, 1)[-1]
        return [int(n) for n in re.findall(r"^Step (\d+)", body, re.MULTILINE)]

    def _answer(self, p: str) -> str:
        if "# PRELIMINARY DIAGNOSIS" in p:
            t = int(re.search(r"Predicted decisive error: Step (\d+)", p).group(1))
            return json.dumps({"candidates": [], "counterfactual_tests": [], "ranked_steps": [t],
                               "final_verdict": {"decision": "CONFIRM", "decisive_step": t,
                                                 "decisive_agent": "stub", "failure_mode": "other",
                                                 "description": "stub", "confidence": "low",
                                                 "reasoning": "stub"}})
        if "# CANDIDATES TO VERIFY" in p:
            t = int(re.search(r"### Candidate 1: Step (\d+)", p).group(1))
            return json.dumps({"evaluations": [], "decisive_error": {
                "step": t, "agent": "stub", "description": "stub",
                "backward_chain": f"Step {t} -> final answer", "reasoning": "stub"}})
        if "Identify the 2–3 steps" in p:
            steps = [int(n) for n in re.findall(r"^#{3,4} Step (\d+)", p, re.MULTILINE)]
            steps = sorted(set(steps))[:3] or [1]
            return "```json\n" + json.dumps({"candidates": [
                {"rank": i + 1, "step": n, "agent": "stub", "reasoning": "stub"}
                for i, n in enumerate(steps)]}) + "\n```"
        if "# HIERARCHICAL REPRESENTATION" in p:
            steps = sorted({int(n) for n in re.findall(r"^### Step (\d+)", p, re.MULTILINE)})
            lines = ["## Typed Dependencies"]
            for a, b in zip(steps, steps[1:]):
                lines += [f"### Step {a} → Step {b}", "- Type: follow_up", "- Relationship: stub"]
            lines += ["## Candidate-Pruning Summary", "stub", "## Value Flow", "stub"]
            return "\n".join(lines)
        if "===MID_LEVEL===\nWrite one entry" in p:
            steps = [n for n in self._steps_after(p, "# FULL TRAJECTORY") if n > 0]
            picks = sorted({steps[0], steps[-1]}) if steps else []
            mid = "\n".join(f"### Step {n} (stub)\nstub moment." for n in picks)
            low = "\n".join(f"### Step {n} (stub)\n- stub evidence" for n in picks[:1])
            return f"===HIGH_LEVEL===\n- stub\n===MID_LEVEL===\n{mid}\n===LOW_LEVEL===\n{low}"
        if "Stay observational" in p:
            return ("===QUESTION_ANALYSIS===\n- stub\n===AGENT_ROLES===\n- stub\n"
                    "===ERROR_SIGNALS===\n- stub signal")
        if "Do not try to answer it" in p:
            return json.dumps({"answer_type": "stub", "constraints": "None stated"})
        return "stub"
