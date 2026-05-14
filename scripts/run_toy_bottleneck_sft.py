"""Phase 0 exit-gate toy: 4-example bottlenecked SFT on Granite 4.0 Micro.

What this proves:
  1. Extended Granite + 4D bottleneck mask runs forward on a real GPU
  2. Loss is finite and decreases over a small number of steps
  3. Constrained-decode generation from the trained-1-step model produces
     legal abstract spans
  4. End-to-end memory + time fits in a sensible budget

Pass criterion: loss at step 5 < loss at step 0; constrained generation
produces only V_abs ∪ {<endabstract>} inside the abstract span.

Run on fortyfive:
    srun --partition=gpu-volatile --gres=gpu:1 --time=15 \\
         --cpus-per-task=4 --mem=32G --pty \\
         bash -lc "source ~/venvs/abscot/bin/activate && cd ~/abstract-cot && python scripts/run_toy_bottleneck_sft.py"
"""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

from abstract_cot.bottlenecked_sft import (
    BottleneckCollator,
    BottleneckExample,
    bottleneck_loss,
    random_abstract_ids,
)
from abstract_cot.constrained_decoding import (
    AbstractCotLogitsProcessor,
    assert_abstract_span_legal,
)
from abstract_cot.tokenizer import extend_tokenizer, resize_model_for_vocab

MODEL_ID = "ibm-granite/granite-4.0-micro"
M_DEFAULT = 64
M_MAX = 32   # short for the toy
STEPS = 5
LR = 5e-5
SEED = 0


TOY_EXAMPLES = [
    {
        "prompt": "Question: A car travels from A to B at 60 km/h, rests 30 minutes at B, "
                  "then returns at 80 km/h. The total trip takes 4 hours. Find the distance from A to B.",
        "verbal_cot": "Let d be the A-B distance in km. Going: d/60 hours. Rest: 0.5 hours. "
                      "Returning: d/80 hours. Total: d/60 + 0.5 + d/80 = 4. So 7d/240 = 7/2, d = 120.",
        "abstract_count": 8,
        "answer": "The distance from A to B is 120 km.",
    },
    {
        "prompt": "Question: Solve for x: 3x + 7 = 22.",
        "verbal_cot": "Subtract 7 from both sides: 3x = 15. Divide by 3: x = 5.",
        "abstract_count": 6,
        "answer": "x = 5.",
    },
    {
        "prompt": "Question: What is 12 * 13?",
        "verbal_cot": "12 * 13 = 12 * 10 + 12 * 3 = 120 + 36 = 156.",
        "abstract_count": 5,
        "answer": "12 * 13 = 156.",
    },
    {
        "prompt": "Question: A box has 24 apples. 1/3 are red, the rest are green. How many are green?",
        "verbal_cot": "Red: 24 * 1/3 = 8. Green: 24 - 8 = 16.",
        "abstract_count": 6,
        "answer": "There are 16 green apples.",
    },
]


def build_examples(vocab, rng):
    out = []
    for ex in TOY_EXAMPLES:
        out.append(
            BottleneckExample(
                prompt=ex["prompt"],
                verbal_cot=ex["verbal_cot"],
                abstract_ids=random_abstract_ids(vocab, ex["abstract_count"], rng),
                answer=ex["answer"],
            )
        )
    return out


def smoke_generate(model, tok, vocab, prompt: str, m_max: int = 16, max_new: int = 60):
    """Sample one constrained abstract span + a few response tokens."""
    prompt_ids = tok.encode(prompt, return_tensors="pt").to(model.device)
    # Append <beginabstract> manually so the policy starts inside the abstract span.
    begin = torch.tensor([[vocab.begin_id]], device=model.device, dtype=torch.long)
    input_ids = torch.cat([prompt_ids, begin], dim=1)

    proc = AbstractCotLogitsProcessor(vocab=vocab, m_max=m_max, start_inside=True)
    with torch.no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new,
            do_sample=True,
            top_p=0.95,
            temperature=1.0,
            logits_processor=LogitsProcessorList([proc]),
            pad_token_id=tok.pad_token_id or tok.eos_token_id,
        )
    seq = out[0].tolist()
    # Validate the abstract span
    assert_abstract_span_legal(seq, vocab, m_max=m_max)
    return seq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_ID)
    p.add_argument("--M", type=int, default=M_DEFAULT)
    p.add_argument("--steps", type=int, default=STEPS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--device", default=None, help="cuda/cpu; auto if unset")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.startswith("cuda"):
        print(f"  GPU: {torch.cuda.get_device_name()}")
        print(f"  Free mem: {torch.cuda.mem_get_info()[0] / 1e9:.1f} GB")

    print(f"Loading {args.model} (bf16)…")
    t0 = time.monotonic()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="eager"
    ).to(device)
    print(f"  loaded in {time.monotonic() - t0:.1f}s; "
          f"params={sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    tok, vocab = extend_tokenizer(tok, M=args.M)
    model = resize_model_for_vocab(model, tok, seed=args.seed)
    print(f"  extended vocab: {len(tok)}  (M={args.M})")

    examples = build_examples(vocab, rng)
    collator = BottleneckCollator(tokenizer=tok, vocab=vocab, max_length=args.max_length,
                                  dtype=torch.bfloat16)
    batch = collator(examples)
    print(f"  batch shape: input_ids={tuple(batch.input_ids.shape)}, "
          f"loss-tokens={int(batch.loss_mask.sum())}")

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    loss_traj = []
    for step in range(args.steps):
        t = time.monotonic()
        opt.zero_grad()
        loss = bottleneck_loss(model, batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_val = float(loss.detach().item())
        loss_traj.append(loss_val)
        print(f"  step {step}: loss={loss_val:.4f}   wall={time.monotonic() - t:.2f}s")

    drop = loss_traj[0] - loss_traj[-1]
    print(f"\nLoss trajectory: {[f'{l:.3f}' for l in loss_traj]}")
    print(f"Step 0 → step {args.steps-1}: drop = {drop:+.4f}")

    assert drop > 0, f"loss did not decrease (drop={drop:+.4f}) — bug somewhere"
    assert all(l == l for l in loss_traj), "NaN in loss"  # NaN != NaN

    # Smoke-test constrained generation on a held-out prompt
    print("\nConstrained-decode sanity check:")
    model.eval()
    held = "Question: What is 5 + 7?"
    seq = smoke_generate(model, tok, vocab, held, m_max=10, max_new=40)
    decoded = tok.decode(seq, skip_special_tokens=False)
    print(f"  prompt: {held!r}")
    print(f"  generated: {decoded[:300]}…" if len(decoded) > 300 else f"  generated: {decoded}")

    print("\n✅ Phase 0 exit gate PASSED.")


if __name__ == "__main__":
    main()
