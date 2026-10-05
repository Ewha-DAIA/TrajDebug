"""Run-level logging: human-readable run.log plus machine-readable errors.jsonl."""

from __future__ import annotations

import json
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

ERROR_TYPES = (
    "auth",            # invalid / missing API key (aborts the run)
    "rate_limit",      # 429
    "timeout",         # request timed out
    "connection",      # network error before a response
    "context_length",  # prompt exceeds the model context window
    "bad_request",     # other 400 errors (e.g. unsupported parameter)
    "api_error",       # 5xx / other API failures
    "truncated",       # finish_reason == "length"
    "empty_response",  # no content returned
    "parse_error",     # response could not be parsed into the expected structure
    "clipped",         # a step was clipped by --clip_chars (informational)
    "fallback",        # a stage gave up and reused the previous stage's result
    "pipeline_error",  # unexpected exception inside the pipeline
)


class RunLogger:
    """Thread-safe writer for ``logs/run.log`` and ``logs/errors.jsonl``."""

    def __init__(self, log_dir: Path, verbose: bool = True):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._errors_path = self.log_dir / "errors.jsonl"
        self._lock = threading.Lock()

        self.logger = logging.getLogger(f"falat.{self.log_dir}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        file_handler = logging.FileHandler(self.log_dir / "run.log", encoding="utf-8")
        file_handler.setFormatter(fmt)
        self.logger.addHandler(file_handler)
        if verbose:
            console = logging.StreamHandler()
            console.setFormatter(fmt)
            self.logger.addHandler(console)

    def info(self, msg: str) -> None:
        self.logger.info(msg)

    def warning(self, msg: str) -> None:
        self.logger.warning(msg)

    def error(
        self,
        *,
        trajectory: str,
        stage: str,
        error_type: str,
        message: str,
        action: str,
        attempt: Optional[int] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        record: Dict[str, Any] = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "trajectory": trajectory,
            "stage": stage,
            "attempt": attempt,
            "error_type": error_type,
            "message": str(message)[:2000],
            "action": action,
        }
        if extra:
            record.update(extra)
        with self._lock:
            with self._errors_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        level = logging.INFO if error_type == "clipped" else logging.WARNING
        self.logger.log(
            level,
            f"[{trajectory}] {stage} {error_type} (attempt={attempt}, action={action}): "
            f"{str(message)[:300]}",
        )
