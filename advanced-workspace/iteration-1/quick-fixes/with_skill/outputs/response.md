# distill_advanced.py — Pre-Training Audit

After reading the skill file (SKILL.md) and auditing the entire codebase in `advanced/`, here is what needs to be fixed before training starts, ordered by severity.

---

## Priority 1 — Apply These Immediately (defensive, <5 min each)

These four issues cause silent failures deep into training or create project-level friction. They should be fixed before the first `python advanced/distill_advanced.py` run.

### 1. Validate `minilm_relation_heads` at startup (no startup check)

**File:** `distill_advanced.py` (around line 291)

**Problem:** Line 128 sets `minilm_relation_heads = 64`. If someone changes this to a value that does not divide both Q hidden dims (teacher 1536, student 896), the crash happens in `reshape_to_relation_heads()` inside `losses.py:138` — after dataset loading, model init, and baseline eval. That wastes 30+ seconds every time.

Valid values are divisors of gcd(1536, 896) = 128: {1, 2, 4, 8, 16, 32, 64, 128}. The default (64) is fine, but there is no guard.

**Fix:** Insert validation after the model-info print block (~line 288, after `print(f"Layer map...")`) :

```python
if CFG.minilm_relation_heads > 0:
    assert teacher_hidden % CFG.minilm_relation_heads == 0, (
        f"minilm_relation_heads ({CFG.minilm_relation_heads}) must divide "
        f"teacher hidden size ({teacher_hidden})"
    )
    assert student_hidden % CFG.minilm_relation_heads == 0, (
        f"minilm_relation_heads ({CFG.minilm_relation_heads}) must divide "
        f"student hidden size ({student_hidden})"
    )
```

---

### 2. Validate attention outputs are available (no guard on attn_implementation)

**File:** `distill_advanced.py` (same location as Fix 1)

**Problem:** The script sets `attn_implementation="eager"` on both models (lines 238, 267), which is required to get non-None `attentions` tuples. But if someone edits these to `"sdpa"` or `"flash_attention_2"` (e.g., to improve throughput), `student_out.attentions` and `teacher_out.attentions` become `None`. The `attention_kl()` call at line 460 then crashes with an opaque TypeError on the first training step — after baseline eval, so the user has already waited.

**Fix:** Insert these assertions after the validation block from Fix 1:

```python
assert teacher.config._attn_implementation == "eager", (
    "teacher must use attn_implementation='eager' to return attention weights"
)
assert student.config._attn_implementation == "eager", (
    "student must use attn_implementation='eager' to return attention weights"
)
```

---

### 3. Missing `advanced/__init__.py`

**File:** `E:\Distillation\advanced\__init__.py` (needs to be created)

**Problem:** The `advanced/` directory is an implicit namespace package — there is no `__init__.py`. Currently `distill_advanced.py` gets away with `from losses import ...` because it is co-located in the same directory, but `from advanced.losses import ...` will fail. This blocks:

- Writing tests in `advanced/tests/` that import via `advanced.losses`.
- Any script outside the `advanced/` folder that tries to import these modules.
- Proper Python packaging conventions.

**Fix:** Create `advanced/__init__.py` with minimal content:

```python
"""Advanced multi-signal knowledge distillation pipeline."""
```

---

### 4. Missing `.gitignore` at repo root

**File:** `E:\Distillation\.gitignore` (needs to be created)

**Problem:** Running the script generates `__pycache__/` directories and `.pyc` files. Without a `.gitignore`, these show up as untracked in `git status`. The repo currently has no `.gitignore` at all.

**Fix:** Create a `.gitignore` at the repo root:

```gitignore
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

---

## Priority 2 — Missing Unit Tests for `losses.py`

**File:** `advanced/tests/test_losses.py` (needs to be created)

**Problem:** The docstring in `losses.py` (lines 3-6) says the functions are "straightforward to unit-test with hand-computed expected values on tiny inputs", and the README describes the BERT script as a "unit-test surface area" — but no tests exist anywhere. These functions are pure tensor math with zero model coupling, so tests are easy to write and would catch regressions when loss formulas change.

Functions that should be tested (tiny tensors, B=2, T=3, V=4, D=8):

| Function | What to verify |
|---|---|
| `kl_logits_per_token` | KL math, T^2 scaling, padding mask |
| `kl_logits_classification` | Same, on [B,C] logits |
| `hidden_mse` | MSE with mock HiddenProjector, padding exclusion |
| `attention_kl` | Row re-normalization, query masking |
| `minilm_relation_kl` | Relation softmax, KL averaged over heads |
| `rkd_loss_pooled` | Pairwise distances + triplet angles |
| `rkd_loss_token_pool` | Mean-pooling correctness |
| `causal_ce_loss` | Shift logic, ignore_index=-100 |
| `build_layer_map` | Edge ratios (1->1, 24->28, etc.), ValueError |
| `reshape_to_relation_heads` | Permutation correctness, ValueError |
| `relation_distribution` | Softmax sums to 1, padding keys |

---

## Priority 3 — Refactor `distill_advanced_bert.py`

**File:** `distill_advanced_bert.py`

**Problem:** ~300 lines of code are duplicated from `losses.py` and `components.py`. The BERT script defines its own `HiddenProjector`, `kl_logits`, `hidden_mse`, `attention_kl`, `build_layer_map`, `UncertaintyWeights`, `EMAModel`, and `build_param_groups`. Bugs fixed in one copy do not propagate to the other.

There is also a duplicate forward pass in `minilm_relation_loss` (BERT script, lines 366-377) that re-runs the BERT encoder to get last-layer hidden states, even though the main training loop already has them from `student_out.hidden_states` and `teacher_out.hidden_states`. This costs 1.5-2x throughput.

**Fix:** Replace duplicated definitions with `from advanced.losses import ...` and `from advanced.components import ...` (depends on Fix 3 being done first). Refactor `minilm_relation_loss` to accept cached `hidden_states` tuples instead of re-running the encoder.

---

## Priority 4 — Future Work (needs design discussion)

These are not blockers for training but are known technical debt items:

1. **`_qwen_last_q` hardcodes Qwen2 internals** (`distill_advanced.py:304-315`): references `model.model.layers[-1]`, `input_layernorm`, `q_proj`. To support Llama, GPT-NeoX, or other architectures, an architecture-detection dispatch is needed.

2. **`optimizer.zero_grad()` inconsistency**: `distill_advanced.py` uses `set_to_none=True` (line 417, line 506) which is more memory-efficient. `distill_advanced_bert.py` uses the default `set_to_none=False`. Should standardize on `True`.

3. **Unimplemented features from README**: GKD / on-policy KD, sequence-level distillation, DPO/preference distillation, TAKD (teacher-assistant), and LoRA on the student are all listed in README.md but not implemented.

---

## Summary

| # | Issue | Severity | File | Est. fix time |
|---|---|---|---|---|
| 1 | No `minilm_relation_heads` startup validation | Critical | distill_advanced.py | 2 min |
| 2 | No guard on attention implementation | Critical | distill_advanced.py | 2 min |
| 3 | Missing `__init__.py` | High | advanced/ | <1 min |
| 4 | Missing `.gitignore` | Medium | Repo root | <1 min |
| 5 | No unit tests | Medium | advanced/tests/ | 1-2 hours |
| 6 | BERT script code duplication | Low | distill_advanced_bert.py | 30 min |
| 7 | Qwen2-hardcoded internals | Low | distill_advanced.py | design needed |
| 8 | `zero_grad` inconsistency | Low | both scripts | <1 min |

**Recommended order:** Apply Fixes 1-4 (Priority 1) before your first training run. Start with Fixes 3 and 4 (standalone), then Fixes 1 and 2 together in `distill_advanced.py`. After that, the script is safe to train with.
