"""OpenAI-compatible chat client with typed errors and per-type retry policy.

TrajDebug's ``detector/utils/model.APIModel`` swallows exceptions and returns
``None``, which hides *why* a call failed. FALAT needs the failure reason in
``errors.jsonl``, so this client raises :class:`LLMCallError` with an
``error_type`` instead.

Retry policy (see falat/README.md):
  * rate_limit / timeout / connection / api_error -> up to ``max_attempts``
    tries with exponential backoff.
  * context_length / auth / bad_request -> no retry (the same request would
    fail again).
  * Parse failures are retried by the caller (``stages.py``), not here.

Reasoning: ``reasoning_effort`` (none / minimal / low / medium / high) is sent
only when set. OpenRouter takes it as ``reasoning: {"effort": ...}``; the OpenAI
API takes ``reasoning_effort``. Unset = the provider's default (the paper does
not state a reasoning setting).

Some models (e.g. reasoning models) reject ``temperature`` or ``max_tokens``.
On such a 400 the client drops / renames the offending parameter once and
remembers the choice for the rest of the run.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

try:
    import openai
    from openai import OpenAI
except ImportError:  # pragma: no cover
    openai = None
    OpenAI = None

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
REASONING_EFFORTS = ("none", "minimal", "low", "medium", "high")
RETRYABLE = {"rate_limit", "timeout", "connection", "api_error", "empty_response"}


class LLMCallError(Exception):
    def __init__(self, error_type: str, message: str, attempts: int = 1):
        super().__init__(message)
        self.error_type = error_type
        self.attempts = attempts


@dataclass
class LLMResult:
    text: str
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    attempts: int = 1


def classify_exception(exc: Exception) -> str:
    msg = str(exc).lower()
    if openai is not None:
        if isinstance(exc, openai.AuthenticationError) or isinstance(exc, openai.PermissionDeniedError):
            return "auth"
        if isinstance(exc, openai.RateLimitError):
            if "insufficient_quota" in msg:
                return "auth"  # out of credit: retrying will not help
            return "rate_limit"
        if isinstance(exc, openai.APITimeoutError):
            return "timeout"
        if isinstance(exc, openai.APIConnectionError):
            return "connection"
        if isinstance(exc, openai.BadRequestError):
            if "context_length" in msg or "maximum context" in msg or "too many tokens" in msg:
                return "context_length"
            return "bad_request"
        if isinstance(exc, openai.APIStatusError):
            return "api_error"
    if isinstance(exc, (httpx.TimeoutException,)):
        return "timeout"
    return "api_error"


class LLMClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        max_output_tokens: int = 32768,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
        max_attempts: int = 3,
        request_timeout: float = 600.0,
        backoff_base: float = 5.0,
    ):
        if OpenAI is None:
            raise ImportError("`openai` is required: pip install -e . (inside TrajDebug)")
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
            raise ValueError(f"reasoning_effort must be one of {REASONING_EFFORTS}")
        self.reasoning_effort = reasoning_effort
        self._openrouter = "openrouter.ai" in base_url
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.http_client = httpx.Client(follow_redirects=True, timeout=request_timeout)
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url.rstrip("/"),
            http_client=self.http_client,
            max_retries=0,  # retries are handled (and logged) here
        )
        # Parameter style, auto-adjusted on the first incompatible 400.
        self._lock = threading.Lock()
        self._token_param = "max_completion_tokens" if _looks_like_reasoning_model(model) else "max_tokens"
        self._send_temperature = temperature is not None

    def describe_params(self) -> Dict[str, Any]:
        return {
            "token_param": self._token_param,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature if self._send_temperature else "model default",
            "reasoning_effort": self.reasoning_effort or "provider default",
            "reasoning_param": (None if not self.reasoning_effort
                                else "extra_body.reasoning.effort" if self._openrouter
                                else "reasoning_effort"),
        }

    def _kwargs(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"model": self.model, "messages": messages}
        kwargs[self._token_param] = self.max_output_tokens
        if self._send_temperature:
            kwargs["temperature"] = self.temperature
        if self.reasoning_effort:
            if self._openrouter:
                kwargs["extra_body"] = {"reasoning": {"effort": self.reasoning_effort}}
            else:
                kwargs["reasoning_effort"] = self.reasoning_effort
        return kwargs

    def _maybe_adapt(self, exc: Exception) -> bool:
        """Adjust parameters after an 'unsupported parameter' 400. Returns True if changed."""
        msg = str(exc).lower()
        with self._lock:
            if "max_tokens" in msg and "max_completion_tokens" in msg and self._token_param == "max_tokens":
                self._token_param = "max_completion_tokens"
                return True
            if "temperature" in msg and self._send_temperature:
                self._send_temperature = False
                return True
        return False

    def chat(self, prompt: str, on_retry=None) -> LLMResult:
        """Send one user message. Raises LLMCallError on final failure.

        ``on_retry(attempt, error_type, message)`` is called before each retry
        so the caller can log it.
        """
        messages = [{"role": "user", "content": prompt}]
        attempt = 0
        adapted = 0
        while True:
            attempt += 1
            try:
                resp = self.client.chat.completions.create(**self._kwargs(messages))
            except Exception as exc:  # noqa: BLE001
                etype = classify_exception(exc)
                if etype == "bad_request" and adapted < 2 and self._maybe_adapt(exc):
                    adapted += 1
                    attempt -= 1  # parameter fix, not a real attempt
                    if on_retry:
                        on_retry(attempt, "bad_request", f"adapted params -> {self.describe_params()}: {exc}")
                    continue
                if etype in RETRYABLE and attempt < self.max_attempts:
                    if on_retry:
                        on_retry(attempt, etype, str(exc))
                    time.sleep(self.backoff_base * (2 ** (attempt - 1)))
                    continue
                raise LLMCallError(etype, str(exc), attempt) from exc

            choice = resp.choices[0] if resp.choices else None
            text = (choice.message.content if choice and choice.message else None) or ""
            # Thinking models (e.g. MiniMax) may inline their reasoning; keep only the answer.
            text = _THINK_BLOCK.sub("", text).strip()
            finish = getattr(choice, "finish_reason", None)
            usage = _usage(resp)
            if finish == "length":
                # Output budget exhausted (often by reasoning tokens). Retrying the
                # identical request would hit the same limit, so surface it.
                raise LLMCallError("truncated", f"finish_reason=length usage={usage}", attempt)
            if not text.strip():
                if attempt < self.max_attempts:
                    if on_retry:
                        on_retry(attempt, "empty_response", f"empty content, finish_reason={finish}")
                    time.sleep(self.backoff_base)
                    continue
                raise LLMCallError("empty_response", f"empty content, finish_reason={finish}", attempt)
            return LLMResult(text=text, usage=usage, finish_reason=finish, attempts=attempt)

    def close(self) -> None:
        try:
            self.client.close()
        finally:
            self.http_client.close()


def _looks_like_reasoning_model(model: str) -> bool:
    name = model.lower().split("/")[-1]
    return name.startswith(("gpt-5", "o1", "o3", "o4"))


def _usage(resp: Any) -> Dict[str, int]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return {}
    details = getattr(usage, "completion_tokens_details", None)
    return {
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "reasoning_tokens": (getattr(details, "reasoning_tokens", 0) or 0) if details else 0,
    }
