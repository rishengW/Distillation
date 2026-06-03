# Advanced Distillation Project — Status Report & Priority Checklist

## Project Overview

The `advanced/` folder implements a five-signal knowledge distillation pipeline for causal LLMs (Qwen2.5-1.5B to Qwen2.5-0.5B) with a BERT classifier variant kept for comparison. The code is structurally sound — it correctly implements six loss terms (logit KD, hidden-state MSE, attention-map KL, MiniLMv2 Q-relation KD, pooled RKD, and causal CE), includes gradient checkpointing, uncertainty-based loss balancing, EMA shadow weights, and early stopping. However, the project has known gaps across four priority levels.

## Current State

| Area | Status |
|---|---|
| **distill_advanced.py** (LLM pipeline) | Complete and functional, but missing startup validations |
| **distill_advanced_bert.py** (BERT pipeline) | Functional but has ~300 lines of duplicated code from losses.py/components.py |
| **losses.py** | Complete, pure tensor functions, no model coupling |
| **components.py** | Complete, reusable torch components |
| **advanced/__init__.py** | MISSING — folder is an implicit namespace package |
| **.gitignore** (repo root) | MISSING — __pycache__/ appears in git status as untracked |
| **Unit tests** | MISSING — no tests/ directory exists |
| **README-listed features** | NONE started — all future work items remain unaddressed |

---

## Priority-Ordered Checklist

### PRIORITY 1 — Apply These Fixes Immediately (defensive, <5 min each)

These prevent silent runtime failures deep in training that waste significant time. Apply all four before making any other changes.

#### [ ] 1.1 Validate `minilm_relation_heads` at startup
- **File:** `E:\Distillation\advanced\distill_advanced.py`
- **Location:** After the model-info print block (after line ~286, the `print(f"Layer map...")` line)
- **What to do:** Insert an assertion that `CFG.minilm_relation_heads` divides both teacher and student Q-hidden dims (1536 and 896 respectively). The gcd is 128, so valid values are divisors of 128: {1, 2, 4, 8, 16, 32, 64, 128}.
- **Why:** Without this, `reshape_to_relation_heads()` in `losses.py:138` crashes with a `ValueError` after dataset loading, model init, and baseline eval — wasting 5-10 minutes per failed run.

#### [ ] 1.2 Validate attention outputs are available
- **File:** `E:\Distillation\advanced\distill_advanced.py`
- **Location:** Same block as 1.1, below the minilm_relation_heads check
- **What to do:** Assert that both `teacher.config._attn_implementation` and `student.config._attn_implementation` equal `"eager"`. The pipeline requires `attn_implementation="eager"` to return attention weight tensors. If someone changes this to `"sdpa"` or `"flash_attention_2"` for throughput, `student_out.attentions` and `teacher_out.attentions` become `None`, causing `attention_kl()` to crash with an opaque TypeError in the first training step.
- **Why:** Catches the misconfiguration before the first training step rather than after everything has been set up.

#### [ ] 1.3 Create `advanced/__init__.py`
- **File:** `E:\Distillation\advanced\__init__.py` (create new)
- **Content:** A minimal docstring: `"""Advanced multi-signal knowledge distillation pipeline."""`
- **Why:** Without it, `from advanced.losses import ...` fails. The folder is currently an implicit namespace package. Adding `__init__.py` makes it a proper Python package, enabling clean imports from other scripts and tests. This is also a prerequisite for the BERT refactor (Priority 3).

#### [ ] 1.4 Add `.gitignore` for Python artifacts
- **File:** `E:\Distillation\.gitignore` (create new at repo root)
- **Content:**
  ```
  __pycache__/
  *.pyc
  *.pyo
  *.pyd
  .Python
  *.so
  *.egg-info/
  dist/
  build/
  ```
- **Why:** `__pycache__/` directories and `.pyc` files currently appear in `git status` as untracked (visible right now: `advanced/__pycache__/distill_advanced.cpython-311.pyc`). There is no `.gitignore` anywhere in the repo.

---

### PRIORITY 2 — Unit Tests for `losses.py`

#### [ ] 2.1 Create `advanced/tests/test_losses.py`
- **File to create:** `E:\Distillation\advanced\tests\test_losses.py`
- **Why:** The `losses.py` docstring explicitly claims functions are "straightforward to unit-test with hand-computed expected values on tiny inputs" and the README calls the BERT version a "unit-test surface area" — but no tests exist. These functions are pure tensor math with no model coupling, so tests are genuinely easy to write and catch regressions when loss formulas change.
- **What to test** (each on tiny tensors B=2, T=3, V=4, D=8 with hand-computed expected values):

  | Function | What to verify |
  |---|---|
  | `kl_logits_per_token` | KL math against `F.kl_div` reference; verify T^2 scaling; test both forward and reverse KL; verify padding mask correctly excludes positions |
  | `kl_logits_classification` | Same as above but on [B, C] classifier logits |
  | `hidden_mse` | Create a mock HiddenProjector (identity or known weights); verify MSE = mean((proj(s) - t)^2); verify attention_mask excludes padding tokens |
  | `attention_kl` | Known attention matrices; verify per-row re-normalization after key-masking; verify query-mask excludes padding queries |
  | `minilm_relation_kl` | Verify relation-distribution softmax with/without padding keys; verify KL is averaged over relation_heads and query positions |
  | `rkd_loss_pooled` | Known embeddings to hand-computed pairwise distances and triplet angles; verify both distance and angle terms return scalar |
  | `rkd_loss_token_pool` | Verify mean-pooling over attention_mask produces correct pooled embeddings |
  | `causal_ce_loss` | Verify shift logic: position i's logit predicts token i+1; verify ignore_index=-100 masks correctly |
  | `build_layer_map` | Test all edge ratios: 1-to-1, 1-to-12, 2-to-12, 6-to-12, 12-to-12, 24-to-28; verify ValueError on non-positive inputs |
  | `reshape_to_relation_heads` | Verify [B,T,D] to [B,H,T,D/H] permutation; verify ValueError when D % H != 0 |
  | `relation_distribution` | Verify softmax sums to 1 over key dimension; verify padding keys produce near-zero probability |

- **Tooling:** Use `pytest` conventions. Each test should be <20 lines and run on CPU.

---

### PRIORITY 3 — Refactor `distill_advanced_bert.py`

This file (757 lines) has ~300 lines of functions and classes that are duplicated from `losses.py` and `components.py`.

#### [ ] 3a: Replace duplicated code with imports from shared modules
- **File:** `E:\Distillation\advanced\distill_advanced_bert.py`
- **Replacements:**

  | Lines to Remove | Function/Class | Replace With |
  |---|---|---|
  | 217-223 | `build_layer_map` | `from advanced.losses import build_layer_map` |
  | 233-244 | `HiddenProjector` class | `from advanced.components import HiddenProjector` |
  | 255-271 | `kl_logits` function | `from advanced.losses import kl_logits_classification as kl_logits` |
  | 274-302 | `hidden_mse` function | `from advanced.losses import hidden_mse` |
  | 305-339 | `attention_kl` function | `from advanced.losses import attention_kl` |
  | 464-483 | `UncertaintyWeights` class | `from advanced.components import UncertaintyWeights` |
  | 489-514 | `EMAModel` class | `from advanced.components import EMAModel` |
  | 521-542 | `build_param_groups` function | `from advanced.components import build_param_groups` |

- **Keep inline (deliberately different from shared version):**
  - `minilm_relation_loss` (lines 342-415) — BERT uses Q+K+V (full MHA) with a separate forward pass. The shared `minilm_relation_kl` in losses.py is Q-only (for GQA models). These are different enough that sharing doesn't help.
  - `rkd_loss` (lines 418-461) — BERT uses [CLS] token (position 0), shared version uses `rkd_loss_pooled` on arbitrary pooled embeddings. Can optionally replace with `rkd_loss_pooled(s_cls, t_cls)` from losses.py.

- **Prerequisite:** Priority 1.3 (__init__.py must exist).

#### [ ] 3b: Fix BERT MiniLMv2 duplicate forward pass
- **File:** `E:\Distillation\advanced\distill_advanced_bert.py`
- **Lines:** 366-377
- **What to do:** The `minilm_relation_loss` function re-runs the BERT encoder (`teacher_model.bert(...)` and `student_model.bert(...)`) to get last-layer hidden states. The main training loop already has these from `student_out.hidden_states` and `teacher_out.hidden_states`. This double-pass costs 1.5-2x throughput.
- **Solution:** Follow the pattern from `_qwen_last_q` in `distill_advanced.py` (lines 304-315). Change the function signature to accept `student_hidden_states` and `teacher_hidden_states` tuples instead of `input_ids`, and extract `hidden_states[-2]` (the input to the last transformer block). Apply the last layer's attention projections to that cached tensor.
- **Why:** Eliminates the duplicate encoder forward pass, roughly doubling throughput for the MiniLMv2 loss term.

---

### PRIORITY 4 — Future Work (design decisions required, multi-day each)

These are larger efforts referenced in `README.md:83-89` or discovered during audit. They require design discussion before implementation.

#### [ ] 4a: Generalize `_qwen_last_q` beyond Qwen2
- **File:** `E:\Distillation\advanced\distill_advanced.py`, lines 304-315
- **What to do:** The function hardcodes Qwen2 internals: `model.model.layers[-1]`, `.input_layernorm`, `.self_attn.q_proj`. To support Llama, GPT-NeoX, or other architectures, add an architecture-detection dispatch. Llama uses the same `input_layernorm` naming but a different `.model.` path; other models differ more.

#### [ ] 4b: Standardize `optimizer.zero_grad(set_to_none=True)`
- **Files:** `E:\Distillation\advanced\distill_advanced.py` (already uses `set_to_none=True`) and `E:\Distillation\advanced\distill_advanced_bert.py` (uses default `set_to_none=False`)
- **What to do:** Change `distill_advanced_bert.py` to use `set_to_none=True` for consistency and lower peak memory.

#### [ ] 4c: README-listed features
- **Source:** `README.md:83-89`
- **Features (each multi-day):**

  | Feature | Description | Effort |
  |---|---|---|
  | **On-policy / GKD** | Sample from student during training, score with teacher. Replaces offline KL with on-policy distributions for better mode coverage. | Large |
  | **Sequence-level distillation** | Generate teacher continuations offline, train student on those completions rather than next-token matching. | Large |
  | **DPO/preference distillation** | Match teacher preferences over completions — useful when the teacher was RLHF-trained. | Large |
  | **TAKD** | Insert a 0.7B teacher-assistant when the 1.5B-to-0.5B capacity gap causes instability. | Large |
  | **LoRA on student** | Use low-rank adaptation instead of full fine-tuning — needed for students too large to fully finetune. | Medium |

---

## Recommendation

**Immediate (Priority 1):** Apply Fixes 1.1-1.4 before any training runs. These take <20 minutes total and prevent expensive silent failures.

**Short-term (Priority 2-3):** Write the unit tests for `losses.py` (straightforward given the pure-function design) and refactor the BERT file to eliminate the duplicated code and the 1.5-2x throughput regression in its MiniLMv2 forward pass.

**Long-term (Priority 4):** Pursue the future features once Priorities 1-3 are complete, as these require design discussion and are multi-day efforts.

## Files Summary

| File | Path | Lines | Notes |
|---|---|---|---|
| Main LLM script | `E:\Distillation\advanced\distill_advanced.py` | 577 | Needs Fixes 1.1, 1.2; generalization 4a |
| BERT script | `E:\Distillation\advanced\distill_advanced_bert.py` | 757 | Needs refactor 3a, 3b; standardize 4b |
| Loss functions | `E:\Distillation\advanced\losses.py` | 247 | Needs unit tests (Priority 2) |
| Components | `E:\Distillation\advanced\components.py` | 116 | Already clean |
| __init__.py | `E:\Distillation\advanced\__init__.py` | MISSING | Needs creation (Fix 1.3) |
| .gitignore | `E:\Distillation\.gitignore` | MISSING | Needs creation (Fix 1.4) |
| Skill file | `E:\Distillation\advanced\SKILL.md` | 290 | Already complete — the source of truth for all fixes |
| README | `E:\Distillation\advanced\README.md` | 98 | User-facing docs; already accurate |
| Evals | `E:\Distillation\advanced\evals\evals.json` | 23 | Contains eval #3 matching this exact prompt |
