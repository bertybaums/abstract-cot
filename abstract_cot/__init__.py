"""Abstract Chain-of-Thought reproduction (Ramji et al. 2026, arXiv:2604.22709).

Modules implemented:
    tokenizer            — extend tokenizer with M reserved abstract tokens + 2 delimiters
    attention_masks      — block-structured mask for bottlenecked SFT
    constrained_decoding — LogitsProcessor restricting abstract span to V_abs ∪ {<endabstract>}
    reward_model         — MindRouter client for gpt-oss-20b generative reward

Modules to be implemented (per PLAN.md §6):
    bottlenecked_sft     — Phase 1a trainer
    self_distillation    — Phase 1b trainer
    grpo_trainer         — Phase 2 RL trainer (TRL GRPOTrainer subclass)
    policy_iteration     — top-level driver for warm-up loop (Algorithm 1)
"""

__version__ = "0.0.0"
