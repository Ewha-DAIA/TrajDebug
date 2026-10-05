"""Shared machinery for the FALAT stage modules.

Every stage module exposes ``run(runner, traj, trace, ctx)``:

* ``runner``  – :class:`StageRunner` (LLM call + retries + logging + settings)
* ``traj``    – :class:`falat.common.trajectory.Trajectory`
* ``trace``   – per-trajectory trace dict written to ``traces/<stem>_trace.json``
* ``ctx``     – values passed from one stage to the next (question, π, M, C, C_H, t̂, …)

A stage that still fails after all attempts falls back to the previous stage's
result and records it with :meth:`StageRunner.mark_fallback`.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ..common.llm_client import LLMCallError
from ..common.parsing import render
from ..common.run_logger import RunLogger
from ..common.trajectory import Trajectory


class AbortRun(Exception):
    """Raised on errors that will affect every trajectory (invalid API key, no quota)."""


class StageRunner:
    def __init__(
        self,
        *,
        client,
        prompts: Dict[str, Dict[str, Any]],
        logger: RunLogger,
        window_k: int = 10,
        top_k: int = 3,
        parse_attempts: int = 3,
    ):
        self.client = client
        self.prompts = prompts
        self.log = logger
        self.window_k = window_k
        self.top_k = top_k
        self.parse_attempts = parse_attempts

    def call(
        self,
        traj: Trajectory,
        trace: Dict[str, Any],
        stage: str,
        prompt_id: str,
        values: Dict[str, Any],
        parser: Callable[[str], Any],
    ) -> Optional[Any]:
        """Render the prompt, call the model, parse; retry on parse failure.

        Returns the parsed value, or None when the stage should fall back.
        API-level retries (rate limit, timeout, ...) happen inside the client.
        """
        prompt = render(self.prompts[prompt_id]["text"], values)
        rec: Dict[str, Any] = {"prompt_file": self.prompts[prompt_id]["file"], "prompt": prompt,
                               "responses": [], "usage": [], "status": "failed", "errors": []}
        trace["stages"][stage] = rec

        def on_retry(attempt: int, etype: str, msg: str) -> None:
            rec["errors"].append({"attempt": attempt, "error_type": etype, "message": msg[:500]})
            self.log.error(trajectory=traj.stem, stage=stage, error_type=etype, message=msg,
                           action="retry", attempt=attempt)

        for attempt in range(1, self.parse_attempts + 1):
            try:
                result = self.client.chat(prompt, on_retry=on_retry)
            except LLMCallError as exc:
                rec["errors"].append({"attempt": attempt, "error_type": exc.error_type,
                                      "message": str(exc)[:500]})
                action = "abort_run" if exc.error_type == "auth" else "fallback"
                self.log.error(trajectory=traj.stem, stage=stage, error_type=exc.error_type,
                               message=str(exc), action=action, attempt=attempt,
                               extra={"prompt_chars": len(prompt)})
                if exc.error_type == "auth":
                    raise AbortRun(f"{exc.error_type}: {exc}") from exc
                return None
            rec["responses"].append(result.text)
            rec["usage"].append(result.usage)
            parsed = parser(result.text)
            if parsed is not None:
                rec["status"] = "ok"
                rec["parsed"] = parsed
                return parsed
            action = "retry" if attempt < self.parse_attempts else "fallback"
            rec["errors"].append({"attempt": attempt, "error_type": "parse_error"})
            self.log.error(trajectory=traj.stem, stage=stage, error_type="parse_error",
                           message=result.text[:500], action=action, attempt=attempt)
        return None

    @staticmethod
    def mark_fallback(trace: Dict[str, Any], stage: str, detail: str) -> None:
        rec = trace["stages"].setdefault(stage, {})
        rec["status"] = "fallback"
        rec["fallback_detail"] = detail
        trace["fallback_stages"].append(stage)
