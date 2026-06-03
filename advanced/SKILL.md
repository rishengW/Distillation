---
name: advanced-distillation-dev
description: >
  Use this skill whenever working in the advanced/ folder of the Distillation
  project, or when asked to fix, audit, improve, or review any file under
  advanced/. This skill catalogs every known bug, missing validation, technical
  debt item, and planned feature for the multi-signal distillation pipeline.
  It provides exact code fixes for each issue rather than just describing them,
  so you can apply changes directly without re-auditing the codebase. Trigger
  on any mention of advanced/distill_advanced.py, advanced/distill_advanced_bert.py,
  advanced/losses.py, advanced/components.py, or the advanced/ folder itself.
---

# Advanced Distillation — Development Guide

This folder implements a five-signal knowledge distillation pipeline for causal
LLMs (Qwen2.5-1.5B → Qwen2.5-0.5B) with a BERT classifier variant kept for
comparison. The code is structurally sound but has known gaps documented below.

## Quick Reference

| Fact | Value |
|---|---|
| Teacher | Qwen2.5-1.5B (28L × 1536d, 12 Q-heads, 2 KV-heads) |
| Student | Qwen2.5-0.5B (24L × 896d, 14 Q-heads, 2 KV-heads) |
| gcd(Q-dims) | 128 |
| Valid `minilm_relation_heads` | 1, 2, 4, 8, 16, 32, 64, 128 |
| Layer map (24→28) | `[0,1,3,4,5,6,7,8,9,11,12,13,14,15,17,18,19,20,21,22,23,25,26,27]` |
| N_LOSS_TERMS | 6 (logit, hidden, attn, minilm, rkd, ce) |

## File Map

```
advanced/
├── SKILL.md                    ← this file
├── README.md                   ← user-facing documentation (run instructions, knob guide)
├── distill_advanced.py         ← LLM pipeline, ~645 lines — the main script
├── distill_advanced_bert.py    ← BERT pipeline, ~540 lines — refactored to use
│                                 shared imports (see Priority 3)
├── losses.py                   ← pure distillation loss functions, 247 lines
├── components.py               ← HiddenProjector, UncertaintyWeights, EMAModel,
│                                 build_param_groups, 116 lines
├── __init__.py                 ← created (Fix 3)
└── tests/
    ├── __init__.py
    └── test_losses.py          ← 47 tests covering all loss functions (Priority 2)
```

---

## Priority 1 — Apply These Fixes Immediately (defensive, <5 min each)

> **STATUS: DONE ✓** — All four fixes applied 2026-06-03.

When working in this folder, apply all four fixes before making other changes.
They prevent silent runtime failures deep in training.

### Fix 1: Validate `minilm_relation_heads` at startup ✓

**Status: DONE.** Inserted at `distill_advanced.py:293-302`.

### Fix 2: Validate attention outputs are available ✓

**Status: DONE.** Inserted at `distill_advanced.py:304-311`.

### Fix 3: Create `advanced/__init__.py` ✓

**Status: DONE.** Created `advanced/__init__.py` with minimal docstring.

### Fix 4: Add `.gitignore` for Python artifacts ✓

**Status: DONE.** Created `.gitignore` at repo root.

---

## Priority 2 — Unit Tests for `losses.py`

> **STATUS: DONE ✓** — Created 2026-06-03.

`advanced/tests/test_losses.py` contains 47 tests covering every public
function in `losses.py`. Run with: `pytest advanced/tests/test_losses.py -v`

---

## Priority 3 — Refactor `distill_advanced_bert.py`

> **STATUS: DONE ✓** — Applied 2026-06-03.

### Refactor 3a: Shared imports ✓
All 8 duplicated functions/classes replaced with imports from `losses.py`
and `components.py`. Kept inline: `minilm_relation_loss` (Q+K+V for BERT)
and `rkd_loss` (uses [CLS] token).

### Refactor 3b: MiniLMv2 duplicate forward pass fixed ✓
`minilm_relation_loss` now accepts cached `hidden_states` tuples instead of
re-running the BERT encoder.

---

## Priority 4 — Future Work

### 4a: Generalize `_qwen_last_q` beyond Qwen2

> **STATUS: DONE ✓** — Applied 2026-06-03.

Extracted architecture dispatch into `advanced/arch_utils.py` as `extract_last_q()`
with a registry (`ARCH_LAYER_PATHS`). Supports Qwen2, Llama, Mistral, Gemma, OPT,
GPT-NeoX, and GPT-2 via `model.config.model_type` dispatch. Combined QKV
projections (GPT-NeoX, GPT-2) are auto-split to extract Q. This module has no
HuggingFace `datasets` dependency and can be imported independently.

### 4b: Standardize `optimizer.zero_grad(set_to_none=True)`

> **STATUS: DONE ✓** — All four `zero_grad()` calls across both scripts now
use `set_to_none=True`.

### 4c: README-listed features

From `README.md:83–89`. Each is a multi-day effort:

| Feature | What it does |
|---|---|
| **On-policy / GKD** | Sample from student during training, score with teacher. Replaces offline KL with on-policy distributions for better mode coverage. |
| **Sequence-level distillation** | Generate teacher continuations offline, train student on those completions rather than next-token matching. |
| **DPO/preference distillation** | Match teacher preferences over completions — useful when the teacher was RLHF-trained. |
| **TAKD** | Insert a 0.7B teacher-assistant when the 1.5B→0.5B capacity gap causes instability. |
| **LoRA on student** | Use low-rank adaptation instead of full fine-tuning — needed for students too large to fully finetune. |

---

## Priority 5 — Cleanup (resolved 2026-06-03)

All housekeeping items from the original audit are resolved:
- **5a:** Parent repo files were already committed/reverted by a prior session.
- **5b:** `optimizer.zero_grad(set_to_none=True)` standardized in both scripts.
- Tracked `__pycache__/` files removed from git index (now `.gitignore`'d).
- All changes committed on `main` as commit `d5d00f4`.

---

## Known Remaining Issues

These are known limitations discovered during audit. They're not urgent enough
to block a release but should be tracked for the next maintainer.

| # | Issue | Severity | Notes |
|---|---|---|---|
| 1 | `_triplet_angles` O(n³) memory | Medium | Produces [B, B, B] tensor. B=32 → 32K entries, B=128 → 2M entries. Documented but not mitigated. Consider subsampling or gradient accumulation for large batches. |
| 2 | `causal_ce_loss` all-padding guard | Low | Returns 0.0 when all labels are -100. Correct but silently drops loss signal — prefer `clamp(min=1)` pattern like other loss functions, though F.cross_entropy doesn't expose the token count directly. |
| 3 | `datasets` + `pyarrow` segfault on Windows | Low | Some pyarrow builds crash on import inside pytest/subprocess. Workaround: import `arch_utils` (no datasets dependency) for architecture dispatch; import full pipeline scripts from the `advanced/` directory directly. |
| 4 | No GPU/quantization tests | Low | All tests run on CPU with tiny mock models. No coverage of bf16 autocast, GradScaler, or int4/int8 teacher quantization paths. |
| 5 | `distill_advanced_bert.py` still imports `math` | Trivial | Only used in inline `minilm_relation_loss`. Import is scoped to that one function; could be moved inside if desired. |
