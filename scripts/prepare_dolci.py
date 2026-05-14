"""Stream Dolci-Think-SFT-7B and produce a length-filtered subsample.

Dolci-Think-SFT-7B is a HuggingFace dataset with 2.27 M examples in
chat-messages format:
    {messages: [{role: "user", content: prompt},
                {role: "assistant", content: "<think>VERBAL_COT</think>\\n\\nANSWER"}],
     dataset_source: str, id: str}

We extract (prompt, verbal_cot, answer) by parsing the <think>...</think>
wrapper. Examples with empty <think> blocks (wildchat instruction-following
without reasoning) are skipped — they lack the verbal CoT we need for
bottlenecked SFT.

Output: JSONL at data/dolci_<n>.jsonl with one record per line:
    {"prompt": "...", "verbal_cot": "...", "answer": "...", "source": "..."}

Length filtering: we tokenize with the target model's tokenizer and skip
examples where prompt + cot + answer + 134 (delimiters + max abstract span)
exceeds --max-seq-length.

Usage on fortyfive:
    source ~/venvs/abscot/bin/activate && cd ~/abstract-cot
    python scripts/prepare_dolci.py --n 100000 --max-seq-length 2048 \\
        --out data/dolci_warmup_100k.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

THINK_RE = re.compile(r"<think>(.*?)</think>(.*)$", re.DOTALL)


def parse_assistant(content: str) -> tuple[str, str] | None:
    """Return (verbal_cot, answer) or None if no usable think block."""
    m = THINK_RE.match(content)
    if not m:
        return None
    think = m.group(1).strip()
    answer = m.group(2).strip()
    if not think or not answer:
        return None
    return think, answer


def parse_example(ex: dict) -> dict | None:
    """Extract (prompt, verbal_cot, answer, source). None if not parseable."""
    msgs = ex.get("messages") or []
    if len(msgs) < 2:
        return None
    user = next((m for m in msgs if m.get("role") == "user"), None)
    asst = next((m for m in msgs if m.get("role") == "assistant"), None)
    if not user or not asst:
        return None
    prompt = (user.get("content") or "").strip()
    parsed = parse_assistant(asst.get("content") or "")
    if not parsed:
        return None
    cot, answer = parsed
    if not prompt:
        return None
    return {
        "prompt": prompt,
        "verbal_cot": cot,
        "answer": answer,
        "source": ex.get("dataset_source") or "",
        "id": ex.get("id") or "",
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100_000, help="target passing examples")
    p.add_argument("--max-seq-length", type=int, default=2048)
    p.add_argument("--abstract-budget", type=int, default=134,
                   help="reserved tokens: <begin>+<end> + max m_max (128) + slack")
    p.add_argument("--tokenizer", default="ibm-granite/granite-4.0-micro")
    p.add_argument("--dataset", default="allenai/Dolci-Think-SFT-7B")
    p.add_argument("--out", default="data/dolci_warmup.jsonl")
    p.add_argument("--max-scan", type=int, default=2_000_000,
                   help="max examples to scan before giving up")
    p.add_argument("--report-every", type=int, default=2000)
    p.add_argument("--shuffle-buffer", type=int, default=50_000,
                   help="streaming shuffle buffer size (0 disables)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer {args.tokenizer}…")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    max_text_tokens = args.max_seq_length - args.abstract_budget
    print(f"Per-example token budget for (prompt + cot + answer): {max_text_tokens}")

    print(f"Streaming {args.dataset}…")
    ds = load_dataset(args.dataset, split="train", streaming=True)
    if args.shuffle_buffer > 0:
        print(f"  shuffle buffer={args.shuffle_buffer}, seed={args.seed}")
        ds = ds.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)

    kept = 0
    scanned = 0
    n_no_think = 0
    n_empty_think = 0
    n_too_long = 0
    source_counts: dict[str, int] = {}
    cot_lens: list[int] = []
    total_lens: list[int] = []
    t0 = time.monotonic()

    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds:
            scanned += 1
            if scanned > args.max_scan:
                print(f"Hit max-scan={args.max_scan}; stopping.")
                break

            parsed = parse_example(ex)
            if parsed is None:
                # Either no <think> wrapper or empty think/answer
                msgs = ex.get("messages") or []
                asst = next((m for m in msgs if m.get("role") == "assistant"), None)
                c = (asst.get("content") or "") if asst else ""
                if "<think>" in c:
                    n_empty_think += 1
                else:
                    n_no_think += 1
                if scanned % args.report_every == 0:
                    elapsed = time.monotonic() - t0
                    rate = scanned / max(elapsed, 1e-6)
                    print(f"  scanned={scanned:>9d} kept={kept:>7d} "
                          f"empty_think={n_empty_think} no_think={n_no_think} "
                          f"too_long={n_too_long} rate={rate:.0f}/s")
                continue

            # Tokenize and length-filter
            p_ids = tok.encode(parsed["prompt"], add_special_tokens=False)
            c_ids = tok.encode(parsed["verbal_cot"], add_special_tokens=False)
            a_ids = tok.encode(parsed["answer"], add_special_tokens=False)
            total = len(p_ids) + len(c_ids) + len(a_ids)
            if total > max_text_tokens:
                n_too_long += 1
                if scanned % args.report_every == 0:
                    elapsed = time.monotonic() - t0
                    rate = scanned / max(elapsed, 1e-6)
                    print(f"  scanned={scanned:>9d} kept={kept:>7d} "
                          f"empty_think={n_empty_think} no_think={n_no_think} "
                          f"too_long={n_too_long} rate={rate:.0f}/s")
                continue

            # Pre-tokenize and store for fast loading at train time
            parsed["prompt_ids"] = p_ids
            parsed["cot_ids"] = c_ids
            parsed["answer_ids"] = a_ids
            parsed["total_text_tokens"] = total
            cot_lens.append(len(c_ids))
            total_lens.append(total)
            source_counts[parsed["source"]] = source_counts.get(parsed["source"], 0) + 1

            f.write(json.dumps(parsed) + "\n")
            kept += 1
            if scanned % args.report_every == 0:
                elapsed = time.monotonic() - t0
                rate = scanned / max(elapsed, 1e-6)
                print(f"  scanned={scanned:>9d} kept={kept:>7d} "
                      f"empty_think={n_empty_think} no_think={n_no_think} "
                      f"too_long={n_too_long} rate={rate:.0f}/s")
            if kept >= args.n:
                break

    elapsed = time.monotonic() - t0
    print()
    print(f"=== done in {elapsed/60:.1f} min ===")
    print(f"scanned: {scanned:,}    kept: {kept:,}")
    print(f"  rejection: empty_think={n_empty_think:,}  no_think={n_no_think:,}  "
          f"too_long={n_too_long:,}")
    print(f"  retention: {kept / scanned * 100:.1f}%")

    if cot_lens:
        cot_lens.sort()
        total_lens.sort()
        def pct(xs, p):
            return xs[min(len(xs) - 1, int(len(xs) * p / 100))]
        print(f"\n  CoT token length:   "
              f"p10={pct(cot_lens, 10):>6}  p50={pct(cot_lens, 50):>6}  "
              f"p90={pct(cot_lens, 90):>6}  p99={pct(cot_lens, 99):>6}  "
              f"max={cot_lens[-1]}")
        print(f"  Total text length:  "
              f"p10={pct(total_lens, 10):>6}  p50={pct(total_lens, 50):>6}  "
              f"p90={pct(total_lens, 90):>6}  p99={pct(total_lens, 99):>6}  "
              f"max={total_lens[-1]}")

    print(f"\n  top 10 sources:")
    for src, count in sorted(source_counts.items(), key=lambda kv: -kv[1])[:10]:
        print(f"    {count:>6}  {src}")

    print(f"\nWrote {kept} examples to {out_path}")


if __name__ == "__main__":
    main()
