"""Reasoning-effort latency × discrimination study.

Question: does gpt-oss-20b at `reasoning_effort: low` (or even `minimal`)
discriminate completion quality as well as `medium` (the paper's choice)?

If yes, we save 3-5x on the dominant cost in RL training. If no, the budget
for Tier-1 reproduction is painful and we have to make scope cuts.

Method:
  - 8 contrasting (prompt, completion) pairs, hand-ranked 0-7 from best (0)
    to worst (7).
  - Score every pair at every effort level, twice (16 calls per level).
  - Report: mean latency per level; Spearman rank correlation between
    intended rank and model score; visualization of score ranges.

Not part of the default pytest run. Opt-in:
    source ~/compression/.env
    python tests/latency_study.py
"""
from __future__ import annotations

import asyncio
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

from abstract_cot.reward_model import AbstractCotRewardModel, RewardConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_PATH = REPO_ROOT / "configs" / "reward_prompt.txt"

# Pairs ordered intended-best to intended-worst. The reward model should
# produce a (roughly) monotonically decreasing score down this list.
PROMPT_MATH = "What is the sum of the first 10 positive integers?"
PROMPT_CAPITAL = "What is the capital of France?"

PAIRS = [
    # rank 0 — clear correct + clean reasoning
    (PROMPT_MATH, "The sum is 55. Using the formula n(n+1)/2 with n=10: 10*11/2 = 55."),
    # rank 1 — correct, no reasoning shown
    (PROMPT_MATH, "55."),
    # rank 2 — correct answer, slightly off reasoning (arithmetic mistake en route to right answer)
    (PROMPT_MATH, "1+2=3, 3+3=6, 6+4=10, ..., final sum 55. (Some steps glossed.)"),
    # rank 3 — wrong answer but reasoning looks plausible
    (PROMPT_MATH, "Using n(n+1)/2 with n=10: 10*11/2 = 60. So the sum is 60."),
    # rank 4 — correct but with irrelevant filler
    (PROMPT_MATH, "I love math! Anyway, 55. By the way, did you know Gauss did this as a child?"),
    # rank 5 — off-topic but coherent
    (PROMPT_MATH, "The Pythagorean theorem states that a^2 + b^2 = c^2 for right triangles."),
    # rank 6 — single word non-answer
    (PROMPT_MATH, "potato"),
    # rank 7 — near-empty
    (PROMPT_MATH, ""),
]

# Sanity-check pair on a non-math topic to make sure we are not measuring
# math-specific behaviour
CAPITAL_PAIRS = [
    (PROMPT_CAPITAL, "The capital of France is Paris."),
    (PROMPT_CAPITAL, "Lyon."),
    (PROMPT_CAPITAL, "I do not know."),
    (PROMPT_CAPITAL, "banana"),
]


@dataclass
class Trial:
    effort: str
    pair_idx: int
    score: float
    latency: float


def spearman(a: list[float], b: list[float]) -> float:
    """Spearman rank correlation (no scipy needed)."""
    n = len(a)
    if n < 2:
        return float("nan")

    def ranks(xs):
        sorted_idx = sorted(range(n), key=lambda i: xs[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and xs[sorted_idx[j + 1]] == xs[sorted_idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[sorted_idx[k]] = avg
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    num = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n))
    den_a = (sum((r - mean_a) ** 2 for r in ra)) ** 0.5
    den_b = (sum((r - mean_b) ** 2 for r in rb)) ** 0.5
    if den_a == 0 or den_b == 0:
        return float("nan")
    return num / (den_a * den_b)


async def score_one(rm: AbstractCotRewardModel, pair_idx: int, prompt: str, completion: str) -> Trial:
    t0 = time.monotonic()
    score = await rm.score(prompt, completion)
    return Trial(effort=rm.config.reasoning_effort, pair_idx=pair_idx, score=score,
                 latency=time.monotonic() - t0)


async def run_effort(api_key: str, effort: str, pairs: list, repeats: int = 2) -> list[Trial]:
    rm = AbstractCotRewardModel(
        api_key=api_key,
        template_path=TEMPLATE_PATH,
        config=RewardConfig(reasoning_effort=effort, cache_enabled=False),
    )
    print(f"\n=== reasoning_effort={effort} ===")
    tasks = []
    for rep in range(repeats):
        for i, (p, c) in enumerate(pairs):
            tasks.append(score_one(rm, i, p, c))
    trials = await asyncio.gather(*tasks)
    return trials


def summarize(trials: list[Trial], pair_count: int, label: str = ""):
    by_pair = [[] for _ in range(pair_count)]
    for t in trials:
        by_pair[t.pair_idx].append(t.score)
    mean_scores = [statistics.mean(s) for s in by_pair]
    intended_ranks = list(range(pair_count))  # 0 = best
    # Lower intended rank → higher expected score → negative Spearman would be
    # a perfect-but-flipped match. Compute correlation of (-rank) vs score so
    # +1 = perfect agreement.
    rho = spearman([-r for r in intended_ranks], mean_scores)
    latencies = [t.latency for t in trials]
    print(f"\n--- {label} summary ---")
    print(f"  trials: {len(trials)} ({pair_count} pairs × {len(trials)//pair_count} repeats)")
    print(f"  mean latency: {statistics.mean(latencies):.2f}s "
          f"(median {statistics.median(latencies):.2f}, "
          f"min {min(latencies):.2f}, max {max(latencies):.2f})")
    print(f"  rank-vs-score Spearman ρ: {rho:+.3f}  (closer to +1.0 = better discrimination)")
    print(f"  per-pair mean score (rank 0=best, descending):")
    for i, s in enumerate(mean_scores):
        bar = "█" * int(round(s * 20))
        print(f"    pair {i}: {s:.2f}  {bar}")
    return rho, statistics.mean(latencies)


async def main():
    api_key = os.environ.get("MINDROUTER_API_KEY")
    if not api_key:
        raise SystemExit("MINDROUTER_API_KEY not set — try `source ~/compression/.env`")

    EFFORTS = ["minimal", "low", "medium", "high"]
    results = {}
    for effort in EFFORTS:
        trials = await run_effort(api_key, effort, PAIRS, repeats=2)
        rho, mean_lat = summarize(trials, len(PAIRS), label=f"MATH pairs / {effort}")
        results[effort] = (rho, mean_lat)

    print("\n\n=== HEADLINE ===")
    print(f"{'effort':>10s}  {'mean_latency':>14s}  {'discrimination_rho':>20s}")
    for effort, (rho, lat) in results.items():
        print(f"{effort:>10s}  {lat:>12.2f}s   {rho:>+18.3f}")

    # Cost-multiplier vs medium
    if "medium" in results:
        med_lat = results["medium"][1]
        print("\nLatency relative to medium:")
        for effort, (_, lat) in results.items():
            print(f"  {effort:>8s}: {lat/med_lat:.2f}× medium")


if __name__ == "__main__":
    asyncio.run(main())
