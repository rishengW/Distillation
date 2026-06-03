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
├── distill_advanced.py         ← LLM pipeline, 577 lines — the main script
├── distill_advanced_bert.py    ← BERT pipeline, 757 lines — ~300 lines duplicated,
│                                 kept for comparison (see Priority 3)
├── losses.py                   ← pure distillation loss functions, 247 lines
├── components.py               ← HiddenProjector, UncertaintyWeights, EMAModel,
│                                 build_param_groups, 116 lines
└── __init__.py                 ← MISSING — create this first (see Fix 3)
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

**Why:** The `losses.py` docstring claims functions are "straightforward to
unit-test with hand-computed expected values on tiny inputs" and the README
calls the BERT version a "unit-test surface area" — but no tests exist. These
functions are pure tensor math with no model coupling, so tests are genuinely
easy to write and catch regressions when loss formulas change.

**File to create:** `advanced/tests/test_losses.py`

Test each function on tiny tensors (B=2, T=3, V=4, D=8) with hand-computed
expected values:

| Function | What to verify |
|---|---|
| `kl_logits_per_token` | KL math against `F.kl_div` reference; verify T² scaling; test both forward and reverse KL; verify padding mask correctly excludes positions |
| `kl_logits_classification` | Same as above but on [B, C] classifier logits |
| `hidden_mse` | Create a mock `HiddenProjector` (identity or known weights), verify MSE = mean((proj(s) - t)²); verify attention_mask excludes padding tokens |
| `attention_kl` | Known attention matrices; verify per-row re-normalization after key-masking; verify query-mask excludes padding queries |
| `minilm_relation_kl` | Verify relation-distribution softmax with/without padding keys; verify KL is averaged over relation_heads and query positions |
| `rkd_loss_pooled` | Known embeddings → hand-computed pairwise distances and triplet angles; verify both distance and angle terms return scalar |
| `rkd_loss_token_pool` | Verify mean-pooling over attention_mask produces correct pooled embeddings |
| `causal_ce_loss` | Verify shift logic: position i's logit predicts token i+1; verify ignore_index=-100 masks correctly |
| `build_layer_map` | Test all edge ratios: 1→1, 1→12, 2→12, 6→12, 12→12, 24→28; verify ValueError on non-positive inputs |
| `reshape_to_relation_heads` | Verify [B,T,D] → [B,H,T,D/H] permutation; verify ValueError when D % H != 0 |
| `relation_distribution` | Verify softmax sums to 1 over key dimension; verify padding keys produce near-zero probability |

Use `pytest` conventions. Each test should be <20 lines and run on CPU.

---

## Priority 3 — Refactor `distill_advanced_bert.py`

### Refactor 3a: Replace duplicated code with imports from shared modules

The BERT version defines ~300 lines of functions and classes that already exist
in `losses.py` and `components.py`. Replace each with an import. The shared
modules are the canonical versions — bugs fixed there won't propagate to the
BERT script otherwise.

**Replacements to make in `distill_advanced_bert.py`:**

Remove the inline definition and add the import:

| Remove (lines) | Add import |
|---|---|
| `build_layer_map` (217–223) | `from advanced.losses import build_layer_map` |
| `HiddenProjector` class (233–244) | `from advanced.components import HiddenProjector` |
| `kl_logits` function (255–271) | `from advanced.losses import kl_logits_classification as kl_logits` |
| `hidden_mse` function (274–302) | `from advanced.losses import hidden_mse` |
| `attention_kl` function (305–339) | `from advanced.losses import attention_kl` |
| `UncertaintyWeights` class (464–483) | `from advanced.components import UncertaintyWeights` |
| `EMAModel` class (489–514) | `from advanced.components import EMAModel` |
| `build_param_groups` function (521–542) | `from advanced.components import build_param_groups` |

**Keep inline (deliberately different from shared version):**

- `minilm_relation_loss` (342–415) — BERT uses Q+K+V (full MHA) with a
  separate forward pass. The shared `minilm_relation_kl` in `losses.py` is
  Q-only (for GQA models). These are different enough that sharing doesn't help.
- `rkd_loss` (418–461) — BERT uses `[CLS]` token (position 0), shared version
  uses `rkd_loss_pooled` on arbitrary pooled embeddings. Can replace with
  `rkd_loss_pooled(s_cls, t_cls)` from `losses.py`, or keep inline for clarity.

### Refactor 3b: Fix BERT MiniLMv2 duplicate forward pass

**Why:** `minilm_relation_loss` at lines 366–377 re-runs the BERT encoder
(`teacher_model.bert(...)` and `student_model.bert(...)`) to get last-layer
hidden states. The main training loop already has these from
`student_out.hidden_states` and `teacher_out.hidden_states`. This double-pass
costs 1.5–2× throughput, as noted in the README.

**How:** Follow the same pattern used by `_qwen_last_q` in `distill_advanced.py`
(lines 304–315). Instead of re-running the encoder, pass the cached
`hidden_states` tuple into `minilm_relation_loss` and extract `hidden_states[-2]`
(the input to the last transformer block). Apply the last layer's attention
projections to that cached tensor.

The signature changes from:
```python
def minilm_relation_loss(student_model, teacher_model, input_ids,
                         attention_mask, relation_heads, eps=1e-8)
```
to:
```python
def minilm_relation_loss(student_model, teacher_model,
                         student_hidden_states, teacher_hidden_states,
                         attention_mask, relation_heads, eps=1e-8)
```

And the callsite changes from:
```python
l_minilm = minilm_relation_loss(
    student, teacher, input_ids, attention_mask, CFG.minilm_relation_heads)
```
to:
```python
l_minilm = minilm_relation_loss(
    student, teacher,
    student_out.hidden_states, teacher_out.hidden_states,
    attention_mask, CFG.minilm_relation_heads)
```

---

## Priority 4 — Future Work

These are larger efforts referenced in `README.md:83–89` or discovered during
audit. They require design decisions and are not simple code fixes.

### 4a: Generalize `_qwen_last_q` beyond Qwen2

`distill_advanced.py:304–315` hardcodes Qwen2 internals: `model.model.layers[-1]`,
`.input_layernorm`, `.self_attn.q_proj`. To support Llama, GPT-NeoX, or other
architectures, add an architecture-detection dispatch. Llama uses the same
`input_layernorm` naming but a different `.model.` path; other models differ more.

### 4b: Standardize `optimizer.zero_grad(set_to_none=True)`

`distill_advanced.py` uses `set_to_none=True` (more memory-efficient) while
`distill_advanced_bert.py` uses the default `set_to_none=False`. Standardize on
`True` in both scripts for consistency and lower peak memory.

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

## When Applying Fixes

> **Fix 1–4 are DONE (2026-06-03).** Start from Priority 3.

Follow this order to avoid conflicts between changes:

1. **Fix 3** (`__init__.py`) and **Fix 4** (`.gitignore`) first — they're standalone. **DONE ✓**
2. **Fix 1** and **Fix 2** next — they go into nearby lines in `distill_advanced.py`, so do them together. **DONE ✓**
3. **Priority 3** (BERT refactor) — depends on Fix 3 being done so imports resolve.
4. **Priority 2** (tests) — can be done in parallel with everything else.
5. **Priority 5** (parent repo housekeeping) — review and commit pending changes.
6. **Priority 4** — only after 1–3 are done, as these require design discussion.

---

## Priority 5 — Parent Repo Housekeeping (detected 2026-06-03)

### 5a: Commit or revert 10 modified files in parent repo

The git repo at `/mnt/e/Distillation/` has 10 modified-but-uncommitted files:

| File | Notes |
|---|---|
| `README.md` | Modified; needs review |
| `distill.py` | Modified; needs review |
| `distill_qwen.py` | Modified; needs review |
| `distill_transformers.py` | Modified; needs review |
| `models.py` | Modified; needs review |
| `requirements.txt` | Modified; needs review |
| `teacher_pretrained.py` | Modified; needs review |
| `train_student.py` | Modified; needs review |
| `train_teacher.py` | Modified; needs review |
| `train_teacher_pretrained.py` | Modified; needs review |

These are pending changes that should be reviewed, committed, or reverted
before they diverge further. Run `git diff` in `/mnt/e/Distillation/` to
assess what changed.

### 5b: Zero_grad inconsistency in `distill_advanced_bert.py`

The BERT script is inconsistent about `set_to_none`:
- Line 601: `optimizer.zero_grad()` — uses default `set_to_none=False`
- Line 687: `optimizer.zero_grad(set_to_none=True)` — correct, more memory-efficient

Standardize both to `set_to_none=True` as done in `distill_advanced.py:417,506`. The SKILL.md already mentions this at Priority 4b but the inconsistency is in the same file and should be fixed alongside the refactor.
