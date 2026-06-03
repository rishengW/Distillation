# Advanced Distillation

A multi-signal distillation pipeline for **causal LLMs** that goes well beyond the vanilla "logit KL + label CE" recipe in `distill_qwen.py`.

```bash
python advanced/distill_advanced.py        # Qwen2.5-1.5B → Qwen2.5-0.5B (LLM)
python advanced/distill_advanced_bert.py   # BERT-base → BERT-tiny on SST-2 (kept for comparison)
```

## Teacher / Student

| | Model | Architecture | Params |
|---|---|---|---|
| Teacher | `Qwen/Qwen2.5-1.5B-Instruct` | 28L × 1536d, 12 Q-heads, 2 KV-heads (GQA) | ~1.5B |
| Student | `Qwen/Qwen2.5-0.5B-Instruct` | 24L × 896d, 14 Q-heads, 2 KV-heads (GQA)  | ~0.5B |

Both are decoder-only, instruction-tuned LLMs. Training task: causal language modeling on WikiText-2, evaluated by perplexity.

## Five distillation signals

| # | Signal | What it matches | Where |
|---|---|---|---|
| 1 | **Per-token logit KD** | KL on temperature-softened next-token distributions, masked over real tokens (forward or reverse KL) | `kl_logits_per_token()` |
| 2 | **Hidden-state matching** | Per-token MSE between projected student hiddens and teacher hiddens at mapped layers (FitNets / Patient-KD) | `hidden_mse()` + `HiddenProjector` |
| 3 | **Attention-map KD** | KL between teacher and student attention probability matrices (heads averaged, padding-key re-normalized) | `attention_kl()` |
| 4 | **MiniLMv2 Q-relation KD** | Match Q·Q^T self-relation distributions at the last layer via fixed "relation heads" — Q-only because Qwen uses GQA (Q-dim ≠ K/V-dim) | `minilm_q_relation_loss()` |
| 5 | **Pooled RKD** | Match pairwise distances and triplet angles of mean-pooled token embeddings across a batch | `pooled_rkd_loss()` |

A 6th term — next-token cross-entropy with `i → i+1` shift — keeps the student grounded in ground truth.

## What changed vs the BERT version

The five signals translate cleanly, but causal LMs need adjustments:

- **Per-token KL**, not per-sample. Padding masked via `labels != -100`.
- **No [CLS]** — RKD pools tokens with the attention mask instead.
- **Q-only MiniLMv2** — Qwen2.5 uses Grouped-Query Attention. Q has 12 or 14 heads while K/V have just 2. K/V live in a different hidden dim than Q so a single shared `relation_heads` count can't apply uniformly. Matching Q-relations is the standard MiniLM-for-GQA recipe.
- **No duplicate forward pass** — MiniLMv2 reuses the cached `hidden_states[-2]` from the main forward instead of re-running the encoder. Roughly 1.5–2× throughput improvement vs the BERT version.
- **`attn_implementation="eager"`** required to actually return attention weights (the default `sdpa` / FlashAttention paths don't expose them).
- **Perplexity** for eval, not accuracy.
- **bf16 throughout** (Qwen native dtype). `GradScaler` is auto-disabled when bf16 autocast is active.

## Engineering toolbox

- bf16 / fp16 autocast (auto-detected via `torch.cuda.is_bf16_supported()`)
- Gradient checkpointing on the student
- Gradient accumulation
- Optional int4 / int8 teacher quantization via `bitsandbytes` (`teacher_quant` in Config)
- AdamW with no-decay groups for bias / LayerNorm / RMSNorm
- Linear warmup + cosine LR schedule
- **Uncertainty loss balancing** (Kendall et al., 2018) — learn one log-variance per loss term so they self-balance
- **EMA shadow weights** for stable eval and final save
- Skip-stride layer mapping for any depth ratio (24 student layers ↔ 28 teacher layers)
- Hidden-state projection heads bridging differing hidden sizes (896 ↔ 1536)
- Early stopping on EMA-evaluated val perplexity with best-checkpoint restore
- Reproducible seed; padding-aware masking everywhere

## Knobs worth knowing

All in the `Config` dataclass at the top of `distill_advanced.py`:

- `temperature` — soft-target temperature (default 4.0)
- `use_reverse_kl` — flip to mode-seeking KL (often preferred for generation; see GKD)
- `w_logit / w_hidden / w_attn / w_minilm / w_rkd / w_ce` — initial weights, used as priors when `learn_loss_weights=True`
- `minilm_relation_heads` — must divide both teacher and student Q-hidden dims. With teacher Q-dim = 1536 and student Q-dim = 896, gcd = 128 → valid values are divisors of 128: {1, 2, 4, 8, 16, 32, 64, 128}. Default 64.
- `teacher_quant` — `"none"` / `"int8"` / `"int4"` for VRAM control
- `ema_decay` — set to 0 to disable EMA on extreme-VRAM setups (EMA shadow lives on the model device)
- `batch_size`, `grad_accum`, `max_len` — VRAM/throughput tradeoffs
- `early_stop_patience`, `early_stop_min_delta` — halt on stalled val PPL

## VRAM ballpark

Single GPU, `max_len=512`, `batch_size=2`:

| `teacher_quant` | Approximate total VRAM |
|---|---|
| `"none"` (bf16) | ~14 GB |
| `"int8"` | ~12 GB |
| `"int4"` | ~10 GB |

Most of the cost is the student forward + backward + EMA shadow. Set `ema_decay=0` to save ~1 GB.

## What's still on the table

- **On-policy / GKD** — sample from the student during training, score with teacher
- **Sequence-level distillation** — generate teacher continuations offline, train the student on those
- **DPO/preference distillation** — match teacher preferences over completions
- **TAKD** — chain a 0.7B teacher-assistant when the gap is too wide
- **LoRA on the student** — for very large students where full-finetune is impossible

This script is the strong on-policy-free baseline you'd build those on top of.

## Files

- `distill_advanced.py` — LLM version (Qwen2.5)
- `distill_advanced_bert.py` — BERT classifier version, kept for comparison and unit-test surface area
- `README.md` — this file
