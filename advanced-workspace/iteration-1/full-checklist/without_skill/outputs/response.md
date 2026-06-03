# Advanced Distillation Project -- Status Report and Priority Checklist

## Project Summary

The advanced distillation folder (`advanced/`) implements a five-signal knowledge distillation pipeline for causal LLMs (Qwen2.5-1.5B teacher -> Qwen2.5-0.5B student), with a BERT classifier variant kept for comparison. The code is structurally sound, implements five complementary distillation signals (logit KD, hidden-state matching, attention-map KD, MiniLMv2 Q-relation KD, and pooled RKD) plus a full engineering toolbox (EMA, uncertainty loss balancing, gradient checkpointing, gradient accumulation, early stopping, etc.). The main scripts are feature-complete and ready for training, but several known gaps and bugs still need attention.

## Current Status

**Done:**
- Main LLM distillation script (`distill_advanced.py`, 577 lines) -- fully functional
- BERT classifier distillation script (`distill_advanced_bert.py`, 757 lines) -- fully functional but duplicated code
- Shared loss functions (`losses.py`, 248 lines) -- clean, model-agnostic tensor functions
- Shared components (`components.py`, 117 lines) -- HiddenProjector, UncertaintyWeights, EMAModel, build_param_groups
- User-facing documentation (`README.md`) covering all five signals, configuration knobs, VRAM estimates, and planned features
- Developer skill guide (`SKILL.md`) with detailed fixes for every known issue
- Eval framework metadata (`evals/evals.json`) with three evaluation cases defined

**Not done (all items below):**

---

## Priority 1 -- Quick Defensive Fixes (estimated <20 min total)

These prevent silent runtime failures deep in training that waste GPU time.

- [ ] **1a. Validate `minilm_relation_heads` at startup**
  - File: `distill_advanced.py`
  - The `reshape_to_relation_heads()` call in `losses.py:137` crashes with `ValueError` if `minilm_relation_heads` doesn't divide both Q-hidden-dims. This crash happens after dataset loading, model init, and baseline eval.
  - Insert validation after the layer-map print block (~line 291) checking both teacher_hidden % CFG.minilm_relation_heads == 0 and student_hidden % CFG.minilm_relation_heads == 0.
  - Valid values: divisors of gcd(1536, 896)=128, i.e. {1,2,4,8,16,32,64,128}. Default (64) is valid.

- [ ] **1b. Validate attention outputs are available**
  - File: `distill_advanced.py`
  - The script requires `attn_implementation="eager"` to return attention weights. If changed to `"sdpa"` or `"flash_attention_2"`, `student_out.attentions` and `teacher_out.attentions` become `None`, and `attention_kl()` crashes with an opaque TypeError.
  - Insert checks that `teacher.config._attn_implementation == "eager"` and `student.config._attn_implementation == "eager"` alongside the minilm validation (~line 291).

- [ ] **1c. Create `advanced/__init__.py`**
  - File: `advanced/__init__.py` (does not exist -- directory is currently an implicit namespace package)
  - Without it, `from advanced.losses import ...` will fail from outside the `advanced/` context. The BERT refactor (Priority 3) depends on this.
  - Content: a simple docstring `"""Advanced multi-signal knowledge distillation pipeline."""`

- [ ] **1d. Add `.gitignore`**
  - File: `.gitignore` at the repo root (does not exist anywhere in the repo)
  - `__pycache__/` directories and `.pyc` files show in `git status` as untracked.
  - Content: `__pycache__/`, `*.pyc`, `*.pyo`, `*.pyd`, `.Python`, `*.so`, `*.egg-info/`, `dist/`, `build/`

---

## Priority 2 -- Unit Tests for `losses.py`

- [ ] **2. Create `advanced/tests/test_losses.py`**
  - The `losses.py` docstring explicitly says functions are "straightforward to unit-test with hand-computed expected values on tiny inputs" and the README calls the BERT version a "unit-test surface area" -- but no tests exist.
  - These are pure tensor functions with no model coupling, so tests are easy to write and catch regressions.
  - Functions to test (use pytest, B=2, T=3, V=4, D=8 tiny tensors, run on CPU):

| Function | What to verify |
|---|---|
| `kl_logits_per_token` | KL math against `F.kl_div` reference; T^2 scaling; forward and reverse KL; padding mask correctness |
| `kl_logits_classification` | Same as above but on [B, C] logits |
| `hidden_mse` | Mock `HiddenProjector` (identity/known weights); verify mask excludes padding |
| `attention_kl` | Known attention matrices; per-row re-normalization after key-masking |
| `minilm_relation_kl` | Relation-distribution softmax with/without padding keys; averaging over heads and queries |
| `rkd_loss_pooled` | Hand-computed pairwise distances and triplet angles; verify both terms return scalar |
| `rkd_loss_token_pool` | Verify mean-pooling over attention mask produces correct embeddings |
| `causal_ce_loss` | Verify i->i+1 shift logic; ignore_index=-100 masking |
| `build_layer_map` | Test edge ratios: 1->1, 1->12, 2->12, 6->12, 12->12, 24->28; ValueError on non-positive |
| `reshape_to_relation_heads` | [B,T,D] -> [B,H,T,D/H] permutation; ValueError when D % H != 0 |
| `relation_distribution` | Softmax sums to 1 over keys; padding keys produce near-zero probability |

---

## Priority 3 -- Refactor `distill_advanced_bert.py`

- [ ] **3a. Replace duplicated code with imports from shared modules**
  - The BERT script defines ~300 lines of functions and classes that already exist in `losses.py` and `components.py`. Replace each with an import so bug fixes don't propagate separately.
  - Replacements (remove inline, add import):

| Remove (lines) | Add import |
|---|---|
| `build_layer_map` (217-223) | `from advanced.losses import build_layer_map` |
| `HiddenProjector` class (233-244) | `from advanced.components import HiddenProjector` |
| `kl_logits` function (255-271) | `from advanced.losses import kl_logits_classification as kl_logits` |
| `hidden_mse` function (274-302) | `from advanced.losses import hidden_mse` |
| `attention_kl` function (305-339) | `from advanced.losses import attention_kl` |
| `UncertaintyWeights` class (464-483) | `from advanced.components import UncertaintyWeights` |
| `EMAModel` class (489-514) | `from advanced.components import EMAModel` |
| `build_param_groups` function (521-542) | `from advanced.components import build_param_groups` |

  - Keep inline: `minilm_relation_loss` (342-415) because BERT uses Q+K+V (full MHA) with a separate forward pass, which is different from the Q-only `minilm_relation_kl` in `losses.py`. `rkd_loss` (418-461) can optionally stay or be replaced with `rkd_loss_pooled(s_cls, t_cls)`.

  - Depends on Priority 1c (`advanced/__init__.py` existing) for clean imports.

- [ ] **3b. Fix BERT MiniLMv2 duplicate forward pass**
  - Lines 366-377 in `distill_advanced_bert.py` re-run the BERT encoder to get last-layer hidden states. The main training loop already has these from `student_out.hidden_states` and `teacher_out.hidden_states`.
  - Refactor `minilm_relation_loss` signature to accept cached `hidden_states` tuples instead of `input_ids`, reusing `hidden_states[-2]` (the input to the last transformer block). This eliminates a 1.5-2x throughput cost.
  - Follow the same pattern used by `_qwen_last_q` in `distill_advanced.py` (lines 304-315).

---

## Priority 4 -- Future Work (larger efforts, multi-day)

- [ ] **4a. Generalize `_qwen_last_q` beyond Qwen2**
  - `distill_advanced.py:304-315` hardcodes Qwen2 internals (`model.model.layers[-1]`, `.input_layernorm`, `.self_attn.q_proj`). Add architecture-detection dispatch to support Llama, GPT-NeoX, or other architectures.

- [ ] **4b. Standardize `optimizer.zero_grad(set_to_none=True)`**
  - `distill_advanced.py` uses `set_to_none=True` (more memory-efficient) while `distill_advanced_bert.py` uses default `set_to_none=False` at line 601. Standardize on `True` in both scripts.

- [ ] **4c. On-policy / Generalized Knowledge Distillation (GKD)**
  - Sample from the student during training, score with teacher logits. Replaces offline KL with on-policy distributions for better mode coverage. (Referenced in README.md:83)

- [ ] **4d. Sequence-level distillation**
  - Generate teacher continuations offline, train the student on those completions rather than next-token matching. (Referenced in README.md:84)

- [ ] **4e. DPO / Preference distillation**
  - Match teacher preferences over completions -- useful when the teacher was RLHF-trained. (Referenced in README.md:85)

- [ ] **4f. Teacher-Assistant Knowledge Distillation (TAKD)**
  - Insert a 0.7B teacher-assistant when the 1.5B -> 0.5B capacity gap causes instability. (Referenced in README.md:86)

- [ ] **4g. LoRA on the student**
  - Use low-rank adaptation instead of full fine-tuning -- needed for students too large to fully finetune. (Referenced in README.md:87)

---

## Blockers / Dependencies

- Priority 1 items have no dependencies -- can be done in any order and validated independently.
- Priority 3a (BERT dedup imports) depends on Priority 1c (`advanced/__init__.py` being created).
- Priority 3a and 3b should be done together to avoid conflicting edits on `distill_advanced_bert.py`.
- Priority 2 (tests) and Priority 4 have no dependencies on other items.
- Recommendation: apply Priority 1 items first (including the standalone 1c and 1d), then tackle Priority 3 and Priority 2 in parallel, then discuss Priority 4 at a design level.
