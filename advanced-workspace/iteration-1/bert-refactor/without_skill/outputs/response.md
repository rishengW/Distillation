# Analysis: What to import vs. what to keep inline

The BERT script (`distill_advanced_bert.py`) has ten functions or classes that overlap with the shared library files.

## Category 1: Replace with imports from `losses.py` (5 items)

| In BERT script | In `losses.py` | Verdict |
|---|---|---|
| `kl_logits` (line 255) | `kl_logits_classification` (line 49) | Replace. Identical logic. |
| `hidden_mse` (line 274) | `hidden_mse` (line 67) | Replace. Same signature, same loop. |
| `attention_kl` (line 305) | `attention_kl` (line 97) | Replace. Same masking pattern. |
| `rkd_loss` (line 418) | `rkd_loss_pooled` (line 192) | Replace. Same math. |
| `build_layer_map` (line 217) | `build_layer_map` (line 236) | Replace. Shared adds input validation. |

## Category 2: Replace with imports from `components.py` (4 items)

| In BERT script | In `components.py` | Verdict |
|---|---|---|
| `HiddenProjector` (line 233) | `HiddenProjector` (line 19) | Replace. Same interface. |
| `UncertaintyWeights` (line 465) | `UncertaintyWeights` (line 34) | Replace. Identical. |
| `EMAModel` (line 490) | `EMAModel` (line 55) | Replace. Shared adds decay guard. |
| `build_param_groups` (line 522) | `build_param_groups` (line 93) | Replace. Shared adds RMSNorm. |

## Category 3: `minilm_relation_loss` — partial replacement only

Two layers:
- **BERT-specific orchestration** (must stay inline): Accessing `bert.encoder.layer[-1].attention.self`, running separate BERT encoder forward passes, calling `.query()/.key()/.value()`.
- **Pure tensor math** (can be replaced): `reshape_to_relation_heads`, `relation_distribution`, and the KL computation — all available in `losses.py`.

Recommendation: Replace inner math with shared helpers but keep model-coupling logic inline.

## Category 4: Stay inline (no shared equivalent)

Config, set_seed, tokenize, evaluate, model loading, training loop — all script-specific.
