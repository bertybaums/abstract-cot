"""Unit tests for the reward-model client.

These tests do not hit MindRouter. They exercise:
  - JSON / regex score-extraction edge cases
  - reasoning_content fallback for gpt-oss-* style responses
  - The AsyncTokenBucket rate limiter
  - The hash cache (no second API call for repeated inputs)
  - Retry-on-parse-failure
  - Graceful fallback to r=0.0 after exhausted retries

A separate manual test (`tests/manual_test_mindrouter.py`) hits the real
endpoint — that one is opt-in and not part of the default pytest run.
"""
from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from abstract_cot.reward_model import (
    AbstractCotRewardModel,
    AsyncTokenBucket,
    RewardConfig,
    _extract_message_text,
    parse_reward_json,
)


# ---------- pure functions ----------


def test_parse_reward_json_strict():
    text = '{"score": 7, "reasoning": "looks good"}'
    assert parse_reward_json(text) == pytest.approx(0.7)


def test_parse_reward_json_with_surrounding_text():
    text = 'Here is my evaluation:\n```json\n{"score": 9, "reasoning": "x"}\n```\nDone.'
    assert parse_reward_json(text) == pytest.approx(0.9)


def test_parse_reward_json_fractional_score():
    text = '{"score": 4.5, "reasoning": "x"}'
    assert parse_reward_json(text) == pytest.approx(0.45)


def test_parse_reward_json_clamps_above_10():
    text = '{"score": 15, "reasoning": "x"}'
    assert parse_reward_json(text) == 1.0


def test_parse_reward_json_clamps_below_0():
    text = '{"score": -2, "reasoning": "x"}'
    assert parse_reward_json(text) == 0.0


def test_parse_reward_json_regex_fallback():
    text = 'The score: 6 — but also lots of garbage that breaks JSON'
    assert parse_reward_json(text) == pytest.approx(0.6)


def test_parse_reward_json_empty_returns_none():
    assert parse_reward_json("") is None
    assert parse_reward_json("nothing here") is None


def test_extract_message_text_prefers_content():
    msg = SimpleNamespace(model_dump=lambda: {"content": "hi", "reasoning_content": "ignored"})
    assert _extract_message_text(msg) == "hi"


def test_extract_message_text_falls_back_to_reasoning_content():
    msg = SimpleNamespace(model_dump=lambda: {"content": None, "reasoning_content": "actual text"})
    assert _extract_message_text(msg) == "actual text"


def test_extract_message_text_handles_plain_dict():
    msg = {"content": None, "reasoning_content": "from dict"}
    assert _extract_message_text(msg) == "from dict"


# ---------- async pieces ----------


@pytest.mark.asyncio
async def test_token_bucket_enforces_rate():
    bucket = AsyncTokenBucket(rate_per_sec=5.0, burst=1.0)
    # Burst=1 means: first acquire is instant, then we wait ~0.2s per call.
    t0 = time.monotonic()
    for _ in range(3):
        await bucket.acquire()
    elapsed = time.monotonic() - t0
    # Expect: 0 + 0.2 + 0.2 = ~0.4s minimum
    assert 0.3 < elapsed < 0.7, f"expected ~0.4s, got {elapsed:.3f}s"


@pytest.mark.asyncio
async def test_reward_model_caches_repeated_calls(tmp_path):
    template = tmp_path / "p.txt"
    template.write_text("{CONVERSATION_HISTORY} :: {RESPONSE_TO_SCORE}")

    rm = AbstractCotRewardModel(
        api_key="fake",
        template_path=template,
        config=RewardConfig(rate_per_min=10000.0, max_retries=1),
    )

    # Mock the AsyncOpenAI client to count calls
    fake_message = SimpleNamespace(
        model_dump=lambda: {"content": '{"score": 8, "reasoning": "ok"}',
                            "reasoning_content": None}
    )
    fake_response = SimpleNamespace(
        choices=[SimpleNamespace(message=fake_message)]
    )
    rm._client.chat.completions.create = AsyncMock(return_value=fake_response)

    r1 = await rm.score("hello", "world")
    r2 = await rm.score("hello", "world")
    r3 = await rm.score("hello", "different")

    assert r1 == r2 == pytest.approx(0.8)
    assert r3 == pytest.approx(0.8)
    assert rm._client.chat.completions.create.await_count == 2, "second call should hit cache"


@pytest.mark.asyncio
async def test_reward_model_retries_on_parse_failure(tmp_path):
    template = tmp_path / "p.txt"
    template.write_text("{CONVERSATION_HISTORY} {RESPONSE_TO_SCORE}")
    rm = AbstractCotRewardModel(
        api_key="fake",
        template_path=template,
        config=RewardConfig(
            rate_per_min=10000.0, max_retries=3, base_backoff_sec=0.001, cache_enabled=False
        ),
    )

    bad = SimpleNamespace(model_dump=lambda: {"content": "no score here", "reasoning_content": None})
    good = SimpleNamespace(model_dump=lambda: {"content": '{"score": 5, "reasoning": "x"}', "reasoning_content": None})
    bad_resp = SimpleNamespace(choices=[SimpleNamespace(message=bad)])
    good_resp = SimpleNamespace(choices=[SimpleNamespace(message=good)])

    rm._client.chat.completions.create = AsyncMock(side_effect=[bad_resp, bad_resp, good_resp])

    r = await rm.score("p", "c")
    assert r == pytest.approx(0.5)
    assert rm._client.chat.completions.create.await_count == 3


@pytest.mark.asyncio
async def test_reward_model_returns_zero_on_exhausted_retries(tmp_path):
    template = tmp_path / "p.txt"
    template.write_text("x")
    rm = AbstractCotRewardModel(
        api_key="fake",
        template_path=template,
        config=RewardConfig(
            rate_per_min=10000.0, max_retries=2, base_backoff_sec=0.001, cache_enabled=False
        ),
    )
    rm._client.chat.completions.create = AsyncMock(side_effect=RuntimeError("connection refused"))

    r = await rm.score("p", "c")
    assert r == 0.0
    assert rm._client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_score_batch_runs_concurrently(tmp_path):
    template = tmp_path / "p.txt"
    template.write_text("x")
    rm = AbstractCotRewardModel(
        api_key="fake",
        template_path=template,
        config=RewardConfig(
            rate_per_min=10000.0, concurrency=4, max_retries=1, cache_enabled=False
        ),
    )

    async def slow_create(*args, **kwargs):
        await asyncio.sleep(0.1)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                model_dump=lambda: {"content": '{"score": 7}', "reasoning_content": None}
            ))]
        )

    rm._client.chat.completions.create = AsyncMock(side_effect=slow_create)

    pairs = [(f"p{i}", f"c{i}") for i in range(4)]
    t0 = time.monotonic()
    scores = await rm.score_batch(pairs)
    elapsed = time.monotonic() - t0
    assert all(s == pytest.approx(0.7) for s in scores)
    # 4 calls at 0.1s each, concurrency=4 → ~0.1s, not 0.4s
    assert elapsed < 0.3, f"batch should run concurrently, took {elapsed:.3f}s"
