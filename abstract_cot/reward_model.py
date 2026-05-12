"""Generative reward model client for GRPO.

Scores (prompt, completion) pairs against a MindRouter-hosted reward model
(default: openai/gpt-oss-20b at reasoning_effort=medium, per paper Appendix B).

Layered rate limiting:
  - Process-global AsyncTokenBucket at 200 req/min (MindRouter per-account cap)
  - Per-model asyncio.Semaphore at concurrency=4 (gpt-oss-20b per-model cap)
  - Hash-based cache to skip repeats (Dolci-Think-RL-7B has duplicate prompts)

The bucket pattern is borrowed from
_RCDS/compression/corpus/generation/generate_reasoning.py — see compression's
CLAUDE.md for the gory history of 429 cascades on April 17, 2026.

See PLAN.md section 6.4.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)


class AsyncTokenBucket:
    """Process-global async token bucket. Acquire blocks until a token frees up.

    Pattern lifted verbatim from compression/corpus/generation/generate_reasoning.py.
    """

    def __init__(self, rate_per_sec: float, burst: float):
        self.rate = rate_per_sec
        self.burst = max(1.0, burst)
        self._tokens = self.burst
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            wait = 0.0
            async with self._lock:
                now = time.monotonic()
                elapsed = now - self._last_refill
                self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
                self._last_refill = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                wait = (1.0 - self._tokens) / self.rate
            await asyncio.sleep(wait)


def _extract_message_text(message: dict | object) -> str:
    """Pull text out of a chat message, handling gpt-oss-style `reasoning_content`.

    gpt-oss-* return text in `reasoning_content` with `content: None`. Standard
    OpenAI clients only read `content` and silently see empty replies.
    """
    if hasattr(message, "model_dump"):
        message = message.model_dump()
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    if content:
        return content
    if reasoning:
        return reasoning
    return ""


_JSON_BLOCK = re.compile(r"\{[^{}]*\"score\"\s*:\s*([\d.]+)[^{}]*\}", re.DOTALL)
_SCORE_FALLBACK = re.compile(r"\"?score\"?\s*[:=]\s*([\d.]+)")


def parse_reward_json(text: str) -> Optional[float]:
    """Parse a reward score from a model response.

    Tries (1) strict JSON parse of the first {...} block; (2) regex fallback
    matching a "score": N pattern. Returns the score normalized to [0, 1],
    or None if no score could be extracted.
    """
    if not text:
        return None
    # Try strict JSON first — look for outermost {...}
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            score = float(obj.get("score"))
            return max(0.0, min(1.0, score / 10.0))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    # Fallback regex
    m = _SCORE_FALLBACK.search(text)
    if m:
        try:
            score = float(m.group(1))
            return max(0.0, min(1.0, score / 10.0))
        except ValueError:
            return None
    return None


def load_prompt_template(path: str | Path) -> str:
    """Load the reward prompt template (paper Appendix B, verbatim)."""
    return Path(path).read_text(encoding="utf-8")


def _cache_key(template_hash: str, prompt: str, completion: str) -> str:
    h = hashlib.sha256()
    h.update(template_hash.encode())
    h.update(b"\x1e")
    h.update(prompt.encode("utf-8", errors="replace"))
    h.update(b"\x1e")
    h.update(completion.encode("utf-8", errors="replace"))
    return h.hexdigest()


@dataclass
class RewardConfig:
    model: str = "openai/gpt-oss-20b"
    base_url: str = "https://mindrouter.uidaho.edu/v1"
    api_key_env: str = "MINDROUTER_API_KEY"
    reasoning_effort: str = "medium"  # paper Appendix B
    rate_per_min: float = 200.0       # MindRouter per-account cap
    burst: float = 10.0
    concurrency: int = 4              # gpt-oss-20b per-model cap
    max_completion_tokens: int = 512
    max_retries: int = 4
    base_backoff_sec: float = 1.0
    cache_enabled: bool = True
    trace_path: Optional[str] = None  # JSONL audit log; one record per call


@dataclass
class AbstractCotRewardModel:
    """Async reward client suitable for GRPO rollout scoring.

    Usage:
        rm = AbstractCotRewardModel(
            api_key=os.environ["MINDROUTER_API_KEY"],
            template_path="configs/reward_prompt.txt",
        )
        score = await rm.score(prompt="...", completion="...")
    """

    api_key: str
    template_path: str | Path
    config: RewardConfig = field(default_factory=RewardConfig)

    _client: AsyncOpenAI = field(init=False)
    _bucket: AsyncTokenBucket = field(init=False)
    _sem: asyncio.Semaphore = field(init=False)
    _cache: dict[str, float] = field(default_factory=dict)
    _template: str = field(init=False)
    _template_hash: str = field(init=False)
    _trace_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self):
        self._client = AsyncOpenAI(
            api_key=self.api_key, base_url=self.config.base_url
        )
        self._bucket = AsyncTokenBucket(
            rate_per_sec=self.config.rate_per_min / 60.0, burst=self.config.burst
        )
        self._sem = asyncio.Semaphore(self.config.concurrency)
        self._template = load_prompt_template(self.template_path)
        self._template_hash = hashlib.sha256(self._template.encode()).hexdigest()[:16]

    def _render(self, prompt: str, completion: str) -> str:
        return self._template.replace("{CONVERSATION_HISTORY}", prompt).replace(
            "{RESPONSE_TO_SCORE}", completion
        )

    async def _trace(self, record: dict) -> None:
        if not self.config.trace_path:
            return
        async with self._trace_lock:
            with open(self.config.trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

    async def score(self, prompt: str, completion: str) -> float:
        """Score a (prompt, completion) pair. Returns r ∈ [0, 1].

        Returns 0.0 on persistent failure (logged + traced).
        """
        if self.config.cache_enabled:
            key = _cache_key(self._template_hash, prompt, completion)
            if key in self._cache:
                return self._cache[key]
        else:
            key = None

        message_content = self._render(prompt, completion)

        last_error: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            await self._bucket.acquire()
            async with self._sem:
                try:
                    t0 = time.monotonic()
                    resp = await self._client.chat.completions.create(
                        model=self.config.model,
                        messages=[{"role": "user", "content": message_content}],
                        max_completion_tokens=self.config.max_completion_tokens,
                        extra_body={"reasoning_effort": self.config.reasoning_effort},
                    )
                    elapsed = time.monotonic() - t0
                    text = _extract_message_text(resp.choices[0].message)
                    r = parse_reward_json(text)
                    if r is None:
                        last_error = ValueError(f"could not parse score from: {text[:200]!r}")
                        await self._trace({"ok": False, "reason": "parse_fail",
                                           "text": text[:500], "attempt": attempt,
                                           "elapsed": elapsed})
                        # Retry on parse failure — model may have produced bad JSON
                        await asyncio.sleep(self.config.base_backoff_sec * (2 ** (attempt - 1)))
                        continue
                    await self._trace({"ok": True, "score": r,
                                       "text": text[:500], "elapsed": elapsed})
                    if key is not None:
                        self._cache[key] = r
                    return r
                except Exception as e:
                    last_error = e
                    backoff = self.config.base_backoff_sec * (2 ** (attempt - 1))
                    logger.warning(
                        "reward call attempt %d/%d failed: %s; backing off %.1fs",
                        attempt, self.config.max_retries, e, backoff,
                    )
                    await asyncio.sleep(backoff)

        logger.error("reward call exhausted retries: %s", last_error)
        await self._trace({"ok": False, "reason": "exhausted_retries",
                           "error": str(last_error)})
        return 0.0  # graceful fallback so GRPO can keep going

    async def score_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[float]:
        """Score K rollouts concurrently. Limited by the semaphore + bucket."""
        return await asyncio.gather(*(self.score(p, c) for p, c in pairs))
