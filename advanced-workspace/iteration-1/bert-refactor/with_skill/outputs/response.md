# Refactoring Analysis: `distill_advanced_bert.py` -> shared imports

I compared every function and class in `distill_advanced_bert.py` against `losses.py` and `components.py`. Below is the full mapping: which definitions can be replaced with imports, which should stay inline, and the reasoning for each.

---

## REPLACE with imports (8 items)

These are direct duplicates of the shared modules. Replacing them means bug fixes in `losses.py` / `components.py` automatically propagate to the BERT script.

| Inline definition (lines) | Import to add | Notes |
|---|---|---|
| `build_layer_map` (217-223) | `from advanced.losses import build_layer_map` | The shared version adds input validation (`ValueError` on non-positive counts) that the BERT script lacks. Strict improvement. |
| `HiddenProjector` class (233-244) | `from advanced.components import HiddenProjector` | Functionally identical. The shared version exposes a `bias` kwarg (default `False`) that the BERT version hardcodes — BERT's instantiation call is unaffected since it doesn't pass `bias`. |
| `kl_logits` (255-271) | `from advanced.losses import kl_logits_classification as kl_logits` | Same logic (classification logits, no token dim). The shared version casts to `.float()` for fp32 stability; the BERT version doesn't — the shared one is safer. |
| `hidden_mse` (274-302) | `from advanced.losses import hidden_mse` | Same algorithm. Shared version adds `.float()` casts on tensors. |
| `attention_kl` (305-339) | `from advanced.losses import attention_kl` | Same algorithm. Shared version adds `.float()` casts on attention tensors. |
| `UncertaintyWeights` class (464-483) | `from advanced.components import UncertaintyWeights` | Identical classes. |
| `EMAModel` class (489-514) | `from advanced.components import EMAModel` | Shared version adds a guard (`if self.decay <= 0: return` in `update()`) for disabling EMA via non-positive decay. The BERT version lacks this — shared is more robust. |
| `build_param_groups` (521-542) | `from advanced.components import build_param_groups` | Shared version also excludes `"norm.weight"` (RMSNorm) from weight decay, in addition to `"bias"`, `"LayerNorm.weight"`, `"layer_norm.weight"` that the BERT version handles. More complete. |

---

## KEEP INLINE (1 item strongly recommended + 1 optional)

### `minilm_relation_loss` (lines 342-415) — STRONGLY KEEP INLINE

**Why it differs from the shared version:**

| Aspect | BERT inline (`distill_advanced_bert.py`) | Shared (`losses.py` `minilm_relation_kl`) |
|---|---|---|
| Projections | Q + K + V (full MHA: `bert.encoder.layer[-1].attention.self`) | Q-only (designed for GQA models like Qwen) |
| Encoder interaction | Re-runs `model.bert(...)` to get last-layer hidden states | Takes already-projected tensors (`s_proj`, `t_proj`) — caller is responsible for providing them |
| Loop over terms | Loops over `[(s_q, t_q), (s_k, t_k), (s_v, t_v)]`, averages all 3 | Single call per (s, t) pair — caller loops if needed |
| Architecture coupling | Accesses `model.bert.encoder.layer[-1].attention.self` directly | No model coupling — pure tensor math |

The BERT version and the shared version solve different problems: the BERT version is a self-contained function that knows about BERT's internal structure and computes Q/K/V; the shared version is a generic KL-on-relation-distributions helper for GQA models that expects the caller to have already extracted projections. Replacing one with the other would require rewriting the callsite and breaking the shared module's GQA focus.

**The SKILL.md Priority 3b fix** (duplicate forward pass in minilm_relation_loss lines 366-377) is a separate concern from the import question. Lines 366-377 re-run the BERT encoder inside the loss function when the hidden states are already available from the main `model(...)` call. This is a performance bug (1.5-2x throughput cost) but not an import-vs-inline question — it needs a signature change to accept cached hidden states, not an import replacement.

---

### `rkd_loss` (lines 418-461) — OPTIONAL REPLACE

**Comparison with shared `rkd_loss_pooled` (losses.py:192-205):**

The BERT version:
- Takes `student_cls` and `teacher_cls` (already-pooled [CLS] embeddings, shape `[B, D]`)
- Defines `pdist()` and `pangle()` as inner functions
- Computes distance + angle loss

The shared `rkd_loss_pooled`:
- Takes `student_emb` and `teacher_emb` (generic pre-pooled embeddings, same shape contract)
- Delegates to module-level helpers `_scale_normalized_pdist` and `_triplet_angles`
- Computes the exact same distance + angle loss

**They are functionally identical.** The BERT callsite:
```python
s_cls = student_out.hidden_states[-1][:, 0, :]
t_cls = teacher_out.hidden_states[-1][:, 0, :]
l_rkd = rkd_loss(s_cls, t_cls)
```
could be replaced with:
```python
s_cls = student_out.hidden_states[-1][:, 0, :]
t_cls = teacher_out.hidden_states[-1][:, 0, :]
l_rkd = rkd_loss_pooled(s_cls, t_cls)
```
without any behavior change.

**Verdict from the skill:** "Can replace with `rkd_loss_pooled(s_cls, t_cls)` from `losses.py`, or keep inline for clarity." I lean toward replacing it — it removes more duplicated code and the shared module's `_scale_normalized_pdist` handles the edge case of all-zero distances (where `.mean()` would crash) better than the BERT version's `s_d[s_d > 0].mean()`.

---

## Summary table

```
Function/Class            Lines      Shared in        Replace?     Reasoning
───────                   ─────      ────────         ───────      ─────────
build_layer_map           217-223    losses.py        YES          Identical + validation
HiddenProjector           233-244    components.py    YES          Identical
kl_logits                 255-271    losses.py        YES          Same as kl_logits_classification
hidden_mse                274-302    losses.py        YES          Same logic + fp32 casts
attention_kl              305-339    losses.py        YES          Same logic + fp32 casts
minilm_relation_loss      342-415    losses.py        NO           Q+K+V vs Q-only; model-coupled
rkd_loss                  418-461    losses.py        OPTIONAL     Same math; rkd_loss_pooled works
UncertaintyWeights        464-483    components.py    YES          Identical
EMAModel                  489-514    components.py    YES          Identical + decay guard
build_param_groups        521-542    components.py    YES          Identical + RMSNorm coverage
```

**Bottom line:** 8 definitions should be replaced with imports, 1 (`minilm_relation_loss`) must stay inline, and 1 (`rkd_loss`) is optional but recommended for replacement. This removes roughly 180 lines of duplicated code from the BERT script while keeping the architecturally distinct logic in place.
