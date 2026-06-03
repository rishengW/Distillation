"""
Advanced multi-signal knowledge distillation for causal LLMs.

This is the LLM counterpart to distill_advanced_bert.py — same five-signal
strategy, but adapted for decoder-only transformers and language modeling.

Teacher : Qwen/Qwen2.5-1.5B-Instruct  (28 layers × 1536d, 12 Q-heads, 2 KV-heads)
Student : Qwen/Qwen2.5-0.5B-Instruct  (24 layers ×  896d, 14 Q-heads, 2 KV-heads)
Task    : Causal LM on WikiText-2  (evaluated by perplexity)

Distillation signals
====================
1. Per-token logit KD       — KL(student || teacher) on temperature-softened
                              next-token distributions, masked & averaged over
                              non-padding positions.
2. Hidden-state matching    — Per-token MSE between projected student hiddens
                              and teacher hiddens at mapped layers. Hidden sizes
                              differ (896 vs 1536) so we learn projections.
3. Attention-map KD         — KL between teacher and student attention
                              probability matrices at mapped layers, with
                              heads averaged (12 vs 14) and re-normalized
                              over real keys after padding-mask.
4. MiniLMv2 Q-relation KD   — Match Q·Q^T self-relation distributions at the
                              last layer via fixed "relation heads". Q-only
                              because Qwen2.5 uses Grouped-Query Attention
                              (2 KV heads vs 12/14 Q heads), so K/V have a
                              different hidden dim than Q and would need a
                              separate, smaller relation-head count.
5. Pooled RKD               — Match the geometry of mean-pooled token
                              embeddings (no [CLS] in a causal LM): pairwise
                              distances + triplet angles across the batch.

Hard supervision: standard next-token cross-entropy with i → i+1 shift.

Engineering
===========
- bf16 autocast (preferred) or fp16 + GradScaler
- Gradient checkpointing on the student
- Gradient accumulation
- AdamW with no-decay groups for bias / LayerNorm / RMSNorm
- Linear warmup + cosine LR schedule
- Uncertainty-based loss balancing (Kendall et al., 2018)
- EMA shadow weights for stable eval & save
- Optional int4/int8 teacher quantization via bitsandbytes
- Reuses cached hidden_states for MiniLMv2 — no duplicate forward pass
- Perplexity-based early stopping with best-checkpoint restore
- Reproducible seed; padding-aware masking everywhere

Run:
    python advanced/distill_advanced.py
"""

from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset

# Make sure the advanced/ directory is on sys.path so bare-name imports
# (losses, components) work regardless of how the script is invoked.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from losses import (
    kl_logits_per_token,
    hidden_mse,
    attention_kl,
    minilm_relation_kl,
    rkd_loss_token_pool,
    causal_ce_loss,
    build_layer_map,
)
from components import (
    HiddenProjector,
    UncertaintyWeights,
    EMAModel,
    build_param_groups,
)
from arch_utils import extract_last_q

# Use HuggingFace mirror for stable access from mainland China.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    teacher_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    student_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    max_len:      int = 512        # token sequence length per training chunk

    # Optimization
    epochs:        int   = 3
    batch_size:    int   = 2       # tune to your VRAM
    grad_accum:    int   = 16      # effective batch = batch_size * grad_accum
    lr:            float = 2e-5
    weight_decay:  float = 0.01
    warmup_ratio:  float = 0.1
    grad_clip:     float = 1.0
    seed:          int   = 42

    # Distillation
    temperature:    float = 4.0
    use_reverse_kl: bool  = False    # forward = mode-covering (Hinton 2015);
                                     # reverse = mode-seeking (GKD-style).

    # Initial loss weights — also serve as priors for the loss balancer.
    w_logit:  float = 1.0
    w_hidden: float = 1.0
    w_attn:   float = 0.5
    w_minilm: float = 1.0
    w_rkd:    float = 0.25
    w_ce:     float = 0.5
    learn_loss_weights: bool = True

    # MiniLMv2 relation heads. Must divide the Q hidden dim of *both* models.
    # Qwen2.5-1.5B Q-dim = 1536, Qwen2.5-0.5B Q-dim = 896.
    # gcd(1536, 896) = 128, so divisors of 128 are valid: {1,2,4,8,16,32,64,128}.
    minilm_relation_heads: int = 64

    # Quantization for the (frozen) teacher. Saves VRAM at negligible loss.
    #   "none" - bf16/fp16 full precision     (~3.0 GB for 1.5B)
    #   "int8" - bitsandbytes 8-bit            (~1.7 GB)
    #   "int4" - bitsandbytes 4-bit NF4        (~1.0 GB, recommended <=16GB)
    teacher_quant: str = "none"

    # EMA of student weights for smoother eval/save.
    ema_decay: float = 0.999

    # Early stopping (on validation perplexity, lower is better).
    early_stop_patience:  int   = 2
    early_stop_min_delta: float = 0.01    # absolute PPL improvement

    # Eval cap — running PPL over the full WikiText-2 val set every epoch is
    # cheap, but during smoke runs you may want to clip it.
    eval_max_batches: int = 50

    # I/O
    save_dir: str = "advanced/student_qwen_advanced"

    # Devices / dtypes — populated at runtime.
    device: torch.device = field(default_factory=lambda: torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    use_bf16: bool = field(default_factory=lambda: (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ))


CFG = Config()
DTYPE = (
    torch.bfloat16 if CFG.use_bf16
    else (torch.float16 if torch.cuda.is_available() else torch.float32)
)


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CFG.seed)
print(f"Device: {CFG.device}  |  dtype: {DTYPE}  |  bf16 autocast: {CFG.use_bf16}")


# ── Tokenizer (Qwen2.5 teacher & student share vocabulary) ────────────────────
print(f"\nLoading tokenizer from {CFG.teacher_name} ...")
tokenizer = AutoTokenizer.from_pretrained(CFG.teacher_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# ── Dataset ───────────────────────────────────────────────────────────────────
print("Loading WikiText-2 ...")
raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")


def tokenize(batch):
    return tokenizer(batch["text"], add_special_tokens=False)


tokenized = raw.map(
    tokenize, batched=True, remove_columns=raw["train"].column_names
)


def group_texts(examples):
    """Concatenate then split into fixed-length MAX_LEN chunks (standard
    causal-LM recipe — no padding, uniform batches)."""
    concatenated = {k: sum(examples[k], []) for k in examples.keys()}
    total_len = (len(concatenated["input_ids"]) // CFG.max_len) * CFG.max_len
    result = {
        k: [t[i:i + CFG.max_len] for i in range(0, total_len, CFG.max_len)]
        for k, t in concatenated.items()
    }
    result["labels"] = [ids.copy() for ids in result["input_ids"]]
    return result


lm_dataset = tokenized.map(group_texts, batched=True)
lm_dataset.set_format("torch")
print(
    f"  Train chunks: {len(lm_dataset['train']):,}  "
    f"Val chunks: {len(lm_dataset['validation']):,}  "
    f"(each {CFG.max_len} tokens)"
)

collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
train_loader = DataLoader(
    lm_dataset["train"], batch_size=CFG.batch_size, shuffle=True, collate_fn=collator
)
val_loader = DataLoader(
    lm_dataset["validation"],
    batch_size=CFG.batch_size * 2, shuffle=False, collate_fn=collator,
)


# ── Load teacher (frozen) ─────────────────────────────────────────────────────
print(f"\nLoading teacher: {CFG.teacher_name}  (quant={CFG.teacher_quant})")
teacher_kwargs = dict(
    torch_dtype=DTYPE,
    trust_remote_code=True,
    device_map="auto" if torch.cuda.is_available() else None,
    output_hidden_states=True,
    output_attentions=True,
    attn_implementation="eager",   # required to actually return attn weights
)
if torch.cuda.is_available() and CFG.teacher_quant in ("int4", "int8"):
    if CFG.teacher_quant == "int4":
        teacher_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=DTYPE,
            bnb_4bit_use_double_quant=True,
        )
    else:
        teacher_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    teacher_kwargs.pop("torch_dtype", None)

teacher = AutoModelForCausalLM.from_pretrained(CFG.teacher_name, **teacher_kwargs)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)


# ── Load student ──────────────────────────────────────────────────────────────
print(f"Loading student: {CFG.student_name}")
student = AutoModelForCausalLM.from_pretrained(
    CFG.student_name,
    torch_dtype=DTYPE,
    trust_remote_code=True,
    device_map="auto" if torch.cuda.is_available() else None,
    output_hidden_states=True,
    output_attentions=True,
    attn_implementation="eager",
)
student.train()
student.gradient_checkpointing_enable()
# Required when grad checkpointing is on for HF causal LMs that use cache.
if hasattr(student, "config"):
    student.config.use_cache = False

teacher_hidden = teacher.config.hidden_size
student_hidden = student.config.hidden_size
teacher_layers = teacher.config.num_hidden_layers
student_layers = student.config.num_hidden_layers
teacher_params = sum(p.numel() for p in teacher.parameters())
student_params = sum(p.numel() for p in student.parameters())

print(
    f"  Teacher: {teacher_layers}L × {teacher_hidden}d   params: {teacher_params:,}\n"
    f"  Student: {student_layers}L × {student_hidden}d   params: {student_params:,}  "
    f"({student_params / teacher_params * 100:.1f}% of teacher)"
)


# ── Layer mapping (skip-stride) ───────────────────────────────────────────────
LAYER_MAP = build_layer_map(student_layers, teacher_layers)
print(f"  Layer map (student → teacher): {list(enumerate(LAYER_MAP))}")

# Validate MiniLMv2 relation_heads before training.
if CFG.minilm_relation_heads > 0:
    assert teacher_hidden % CFG.minilm_relation_heads == 0, (
        f"minilm_relation_heads ({CFG.minilm_relation_heads}) must divide "
        f"teacher hidden size ({teacher_hidden})"
    )
    assert student_hidden % CFG.minilm_relation_heads == 0, (
        f"minilm_relation_heads ({CFG.minilm_relation_heads}) must divide "
        f"student hidden size ({student_hidden})"
    )

# Guard: eager attention is required for attention-map KD.
# sdpa / flash_attention_2 do not return attention weights.
assert teacher.config._attn_implementation == "eager", (
    "teacher must use attn_implementation='eager' to return attention weights"
)
assert student.config._attn_implementation == "eager", (
    "student must use attn_implementation='eager' to return attention weights"
)

# ── Hidden-state projection heads (student dim → teacher dim) ─────────────────
hidden_proj = HiddenProjector(
    n_layers=student_layers, in_dim=student_hidden, out_dim=teacher_hidden,
).to(CFG.device).to(DTYPE)


# ── MiniLMv2 Q-relation wrapper ─────────────────────────────────────────────
# extract_last_q (from arch_utils.py) handles architecture dispatch across
# Qwen2, Llama, Mistral, Gemma, OPT, GPT-NeoX, and GPT-2.
# minilm_relation_kl (from losses.py) computes the self-relation KL.
# Together they extract the last-layer Q from cached hidden_states
# (no extra forward pass) and compute MiniLMv2 loss in one call.


def minilm_q_relation_loss(
    student_model, teacher_model,
    student_hidden_states, teacher_hidden_states,
    attention_mask, relation_heads,
) -> torch.Tensor:
    """
    MiniLMv2 self-relation distillation, Q-only variant for GQA models.

    Qwen2.5 uses Grouped-Query Attention: K and V have far fewer heads than Q
    (2 KV heads vs 12-14 Q heads), so K/V live in a different hidden dim than
    Q and a single shared `relation_heads` count can't apply uniformly.
    Q-only is the standard MiniLM-for-GQA recipe.
    """
    s_q = extract_last_q(student_model, student_hidden_states)
    with torch.no_grad():
        t_q = extract_last_q(teacher_model, teacher_hidden_states)
    return minilm_relation_kl(s_q, t_q, attention_mask, relation_heads)


# ── Loss balancer & EMA ───────────────────────────────────────────────────────
N_LOSS_TERMS = 6   # logit, hidden, attn, minilm, rkd, ce
loss_balancer = UncertaintyWeights(N_LOSS_TERMS).to(CFG.device)
ema           = EMAModel(student, CFG.ema_decay)





# ── Optimizer & scheduler ─────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    build_param_groups(student, hidden_proj, loss_balancer, weight_decay=CFG.weight_decay),
    lr=CFG.lr,
    betas=(0.9, 0.95),
    eps=1e-8,
)

steps_per_epoch = max(1, len(train_loader) // CFG.grad_accum)
total_steps  = steps_per_epoch * CFG.epochs
warmup_steps = int(CFG.warmup_ratio * total_steps)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)


# ── Perplexity evaluation ─────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_perplexity(model, loader, max_batches: int = 50) -> float:
    """Lower is better; random 50k-vocab LM ≈ 50000."""
    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        input_ids      = batch["input_ids"].to(CFG.device).long()
        labels         = batch["labels"].to(CFG.device).long()
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(CFG.device).long()

        out = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )
        n_tokens = (labels != -100).sum().item()
        total_loss   += out.loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return float(torch.exp(torch.tensor(avg_loss)))


# ── Baseline ──────────────────────────────────────────────────────────────────
print("\n=== Baseline perplexity (before distillation) ===")
teacher_ppl = evaluate_perplexity(teacher, val_loader, CFG.eval_max_batches)
student_ppl = evaluate_perplexity(student, val_loader, CFG.eval_max_batches)
print(f"  Teacher PPL : {teacher_ppl:.2f}")
print(f"  Student PPL : {student_ppl:.2f}\n")


# ── Distillation training loop ────────────────────────────────────────────────
print("=== Starting distillation ===")
scaler = torch.amp.GradScaler(
    "cuda", enabled=torch.cuda.is_available() and not CFG.use_bf16
)

best_ppl          = float("inf")
best_state        = None
epochs_no_improve = 0
global_step       = 0

for epoch in range(1, CFG.epochs + 1):
    student.train()
    hidden_proj.train()
    loss_balancer.train()

    optimizer.zero_grad(set_to_none=True)
    epoch_loss  = 0.0
    epoch_terms = torch.zeros(N_LOSS_TERMS)
    t0 = time.time()

    for step, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(CFG.device, non_blocking=True).long()
        labels         = batch["labels"].to(CFG.device, non_blocking=True).long()
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(CFG.device, non_blocking=True).long()

        with torch.amp.autocast(
            "cuda", enabled=torch.cuda.is_available(), dtype=DTYPE
        ):
            student_out = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
                use_cache=False,
            )
            with torch.no_grad():
                teacher_out = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=True,
                    use_cache=False,
                )

            # Non-padding token mask (labels != -100).
            label_mask = (labels != -100)

            # Five distillation signals.
            l_logit = kl_logits_per_token(
                student_out.logits, teacher_out.logits, label_mask,
                CFG.temperature, reverse=CFG.use_reverse_kl,
            )
            l_hidden = hidden_mse(
                student_out.hidden_states, teacher_out.hidden_states,
                LAYER_MAP, hidden_proj, attention_mask,
            )
            l_attn = attention_kl(
                student_out.attentions, teacher_out.attentions,
                LAYER_MAP, attention_mask,
            )
            l_minilm = minilm_q_relation_loss(
                student, teacher,
                student_out.hidden_states, teacher_out.hidden_states,
                attention_mask, CFG.minilm_relation_heads,
            )
            l_rkd = rkd_loss_token_pool(
                student_out.hidden_states[-1],
                teacher_out.hidden_states[-1],
                attention_mask,
            )
            l_ce = causal_ce_loss(student_out.logits, labels)

            term_losses = [l_logit, l_hidden, l_attn, l_minilm, l_rkd, l_ce]
            init_weights = torch.tensor(
                [CFG.w_logit, CFG.w_hidden, CFG.w_attn,
                 CFG.w_minilm, CFG.w_rkd, CFG.w_ce],
                device=CFG.device,
            )

            if CFG.learn_loss_weights:
                weighted = [w * l for w, l in zip(init_weights, term_losses)]
                loss = loss_balancer(weighted)
            else:
                loss = sum(w * l for w, l in zip(init_weights, term_losses))

            loss = loss / CFG.grad_accum

        scaler.scale(loss).backward()

        epoch_loss  += loss.item() * CFG.grad_accum
        epoch_terms += torch.tensor([l.detach().float().item() for l in term_losses])

        # Gradient accumulation step.
        if (step + 1) % CFG.grad_accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]],
                CFG.grad_clip,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            ema.update(student)
            global_step += 1

        if (step + 1) % 100 == 0:
            avg = epoch_loss / (step + 1)
            print(
                f"  Epoch {epoch} | step {step+1}/{len(train_loader)} | "
                f"loss={avg:.4f} | lr={scheduler.get_last_lr()[0]:.2e}"
            )

    # ── End-of-epoch eval (using EMA weights) ─────────────────────────────────
    original = ema.apply_to(student)
    val_ppl  = evaluate_perplexity(student, val_loader, CFG.eval_max_batches)
    EMAModel.restore(student, original)

    avg_loss   = epoch_loss / len(train_loader)
    avg_terms  = epoch_terms / len(train_loader)
    elapsed    = time.time() - t0
    term_names = ["logit", "hidden", "attn", "minilm", "rkd", "ce"]
    term_str   = "  ".join(f"{n}={v:.3f}" for n, v in zip(term_names, avg_terms))

    print(
        f"\nEpoch {epoch}/{CFG.epochs}  loss={avg_loss:.4f}  "
        f"val_ppl(EMA)={val_ppl:.2f}  ({elapsed:.1f}s)\n"
        f"   per-term: {term_str}"
    )

    if CFG.learn_loss_weights:
        learned = torch.exp(-loss_balancer.log_var).detach().cpu().tolist()
        learned_str = "  ".join(f"{n}={w:.2f}" for n, w in zip(term_names, learned))
        print(f"   learned weights (exp(-s)): {learned_str}")

    # Early stopping on EMA-evaluated validation perplexity.
    if val_ppl < best_ppl - CFG.early_stop_min_delta:
        best_ppl          = val_ppl
        epochs_no_improve = 0
        best_state = {k: v.clone() for k, v in ema.shadow.items()}
        print(f"   ↳ new best val_ppl={best_ppl:.2f}  (EMA snapshot saved)\n")
    else:
        epochs_no_improve += 1
        print(
            f"   ↳ no improvement for {epochs_no_improve}/"
            f"{CFG.early_stop_patience} epoch(s)  (best={best_ppl:.2f})\n"
        )
        if (
            CFG.early_stop_patience > 0
            and epochs_no_improve >= CFG.early_stop_patience
        ):
            print(f"Early stopping at epoch {epoch}.\n")
            break


# ── Restore best EMA weights, save, final eval ────────────────────────────────
if best_state is not None:
    student.load_state_dict(best_state, strict=False)
    print(f"Restored best EMA weights (val_ppl={best_ppl:.2f}).")

os.makedirs(CFG.save_dir, exist_ok=True)
student.save_pretrained(CFG.save_dir)
tokenizer.save_pretrained(CFG.save_dir)
print(f"Distilled student saved to '{CFG.save_dir}/'")

final_ppl = evaluate_perplexity(student, val_loader, CFG.eval_max_batches)
print("\n=== Final Results ===")
print(f"Teacher  PPL : {teacher_ppl:.2f}   params : {teacher_params:,}")
print(f"Student  PPL : {final_ppl:.2f}   params : {student_params:,}")
print(f"PPL gap      : {final_ppl - teacher_ppl:+.2f}")
print(f"Param reduction : {(1 - student_params / teacher_params) * 100:.1f}%")



