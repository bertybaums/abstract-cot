"""End-to-end probe against the real MindRouter endpoint.

This is NOT part of the default pytest run — it's an opt-in script for
verifying the reward-model layer works against live infrastructure.

Run locally on fortyfive after `source ~/compression/.env` (or however else
you set MINDROUTER_API_KEY):

    python tests/manual_test_mindrouter.py

Expects MINDROUTER_API_KEY in env. Hits openai/gpt-oss-20b with the verbatim
Appendix B reward prompt and reports score, latency, and any parse failures.
"""
from __future__ import annotations

import asyncio
import os
import statistics
import time
from pathlib import Path

from abstract_cot.reward_model import AbstractCotRewardModel, RewardConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = REPO_ROOT / "configs" / "reward_prompt.txt"

# Two contrasting (prompt, completion) pairs. Reward model should distinguish.
GOOD = (
    "What is the sum of the first 10 positive integers?",
    "The sum is 55. Reasoning: 1+2+...+10 = 10*11/2 = 55.",
)
BAD = (
    "What is the sum of the first 10 positive integers?",
    "potato",
)


async def main():
    api_key = os.environ.get("MINDROUTER_API_KEY")
    if not api_key:
        raise SystemExit(
            "MINDROUTER_API_KEY not set. "
            "Try: source ~/compression/.env && python tests/manual_test_mindrouter.py"
        )

    rm = AbstractCotRewardModel(
        api_key=api_key,
        template_path=TEMPLATE_PATH,
        config=RewardConfig(reasoning_effort="medium"),
    )

    print(f"Hitting {rm.config.base_url} with model {rm.config.model} "
          f"(reasoning_effort={rm.config.reasoning_effort})")

    times = []
    for label, (prompt, completion) in [("GOOD", GOOD), ("BAD", BAD)]:
        t0 = time.monotonic()
        score = await rm.score(prompt, completion)
        elapsed = time.monotonic() - t0
        times.append(elapsed)
        print(f"  {label}: score={score:.3f}  latency={elapsed:.2f}s")

    print(f"\nMean latency: {statistics.mean(times):.2f}s "
          f"(min {min(times):.2f}, max {max(times):.2f})")
    print("If GOOD > BAD, the reward signal is working.")


if __name__ == "__main__":
    asyncio.run(main())
