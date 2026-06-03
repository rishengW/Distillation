"""
Advanced multi-signal knowledge distillation for transformer classifiers.

This script goes well beyond the vanilla "logit KL + label CE" recipe used in
the rest of the repo. It combines five complementary distillation signals and
a small engineering toolbox so the student gets every drop of supervision the
teacher can offer.

Distillation signals
====================
1. Logit KD                — KL(student || teacher) on temperature-softened
                             output logits. The classic Hinton-2015 signal.
                             Optionally swap to *reverse* KL for mode-seeking.
2. Hidden-state matching   — MSE between student layer outputs and teacher
                             layer outputs. Hidden sizes differ (e.g. 128 vs
                             768) so we learn a per-layer projection head.
                             Aka FitNets / Patient-KD.
3. Attention-map KD        — KL between teacher and student attention
                             probability matrices at mapped layers. Heads are
                             averaged to handle differing head counts.
4. MiniLMv2 relation KD    — Match self-relation matrices Q·Q^T, K·K^T, V·V^T
                             at the last layer. Head-agnostic via "relation
                             heads" reshaping, so it works across very
                             different architectures.
5. Relational KD (RKD)     — Match the *geometry* of [CLS] embeddings across
                             a batch: pairwise distances and triplet angles.
                             Forces the student to preserve sample-to-sample
                             relationships, not just absolute features.

Engineering
===========
- bf16 / fp16 autocast (auto-detected)
- Gradient checkpointing on the student
- Gradient accumulation
- EMA "shadow" student weights for smoother eval & final save
- AdamW with decoupled weight-decay (no decay on bias/LayerNorm)
- Linear warmup + cosine LR schedule
- Loss balancing via uncertainty weighting (Kendall et al., 2018) so the
  five loss terms don't have to be hand-tuned to the same scale
- Layer-mapping helper for any teacher/student depth ratio
- Early stopping with best-checkpoint restore
- Reproducible: deterministic seed, cuDNN benchmark off in eval

Task        : SST-2 sentiment classification
Teacher     : textattack/bert-base-uncased-SST-2  (12L, 768 hidden, 12 heads)
Student     : prajjwal1/bert-tiny                  (2L,  128 hidden,  2 heads)
Run         : python advanced/distill_advanced.py

Most knobs live in the Config block below — change them there, not in the
loops, so the script stays grep-friendly.
"""

from __future__ import annotations

import math
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup,
)
from datasets import load_dataset

# Shared distillation losses and components — canonical versions from
# losses.py and components.py. Imported here to avoid duplication.
from losses import (
    build_layer_map,
    kl_logits_classification as kl_logits,
    hidden_mse,
    attention_kl,
)
from components import (
    HiddenProjector,
    UncertaintyWeights,
    EMAModel,
    build_param_groups,
)

# Use HuggingFace mirror for stable access from mainland China; matches the
# convention used by distill_transformers.py at the repo root. Comment out
# this line if you're outside China.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    teacher_name: str = "textattack/bert-base-uncased-SST-2"
    student_name: str = "prajjwal1/bert-tiny"
    num_labels:   int = 2
    max_len:      int = 128

    # Optimization
    epochs:        int   = 4
    batch_size:    int   = 32
    grad_accum:    int   = 2          # effective batch = 64
    lr:            float = 5e-5
    weight_decay:  float = 0.01
    warmup_ratio:  float = 0.1
    grad_clip:     float = 1.0
    seed:          int   = 42

    # Distillation
    temperature:   float = 4.0
    use_reverse_kl: bool = False      # forward KL is mode-covering; reverse
                                      # KL is mode-seeking. Try both.

    # Initial loss weights. With `learn_loss_weights=True` these become the
    # *starting points* for the uncertainty-weighted balancer.
    w_logit:       float = 1.0
    w_hidden:      float = 1.0
    w_attn:        float = 1.0
    w_minilm:      float = 1.0
    w_rkd:         float = 0.5
    w_ce:          float = 0.5        # ground-truth label CE (kept small;
                                      # the soft signals dominate)
    learn_loss_weights: bool = True

    # MiniLMv2 relation heads — head-count-agnostic via reshaping.
    # Must divide BOTH teacher and student hidden sizes. With teacher=768 and
    # student=128, valid choices are {1, 2, 4, 8, 16, 32, 64, 128}. The
    # original MiniLMv2 paper uses 48 for BERT-base; 64 is the closest value
    # that also divides 128.
    minilm_relation_heads: int = 64

    # EMA of student weights for stable evaluation
    ema_decay: float = 0.999

    # Early stopping
    early_stop_patience: int = 2
    early_stop_min_delta: float = 0.0005   # absolute val-acc improvement

    # I/O
    save_dir: str = "advanced/student_advanced"

    # Devices / dtypes — populated at runtime
    device: torch.device = field(default_factory=lambda: torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ))
    use_bf16: bool = field(default_factory=lambda: (
        torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    ))


CFG = Config()


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(CFG.seed)
print(f"Device: {CFG.device}  |  bf16 autocast: {CFG.use_bf16}")


# ── Data ──────────────────────────────────────────────────────────────────────
print("\nLoading SST-2 dataset...")
raw = load_dataset("glue", "sst2")
tokenizer = AutoTokenizer.from_pretrained(CFG.teacher_name)


def tokenize(batch):
    return tokenizer(
        batch["sentence"],
        truncation=True,
        padding="max_length",
        max_length=CFG.max_len,
    )


encoded = raw.map(tokenize, batched=True)
encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

train_loader = DataLoader(encoded["train"], batch_size=CFG.batch_size, shuffle=True)
val_loader   = DataLoader(encoded["validation"], batch_size=64, shuffle=False)


# ── Models (request hidden states & attentions) ───────────────────────────────
print(f"\nLoading teacher: {CFG.teacher_name}")
teacher = AutoModelForSequenceClassification.from_pretrained(
    CFG.teacher_name,
    num_labels=CFG.num_labels,
    output_hidden_states=True,
    output_attentions=True,
).to(CFG.device)
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)

print(f"Loading student: {CFG.student_name}")
student = AutoModelForSequenceClassification.from_pretrained(
    CFG.student_name,
    num_labels=CFG.num_labels,
    output_hidden_states=True,
    output_attentions=True,
    ignore_mismatched_sizes=True,
).to(CFG.device)
# Trade compute for memory — required if you scale this up to a 12L student.
student.gradient_checkpointing_enable()

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


# ── Layer mapping (skip-stride, via shared helper) ───────────────────────────
LAYER_MAP = build_layer_map(student_layers, teacher_layers)
print(f"  Layer map (student → teacher): {list(enumerate(LAYER_MAP))}")


# ── Hidden-state projection heads (student dim → teacher dim) ─────────────────
hidden_proj = HiddenProjector(
    n_layers=student_layers,
    in_dim=student_hidden,
    out_dim=teacher_hidden,
).to(CFG.device)


# ── Distillation losses ───────────────────────────────────────────────────────
def minilm_relation_loss(
    student_model,
    teacher_model,
    student_hidden_states: tuple[torch.Tensor, ...],
    teacher_hidden_states: tuple[torch.Tensor, ...],
    attention_mask: torch.Tensor,
    relation_heads: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    MiniLMv2 last-layer self-relation distillation.

    Uses cached hidden_states from the main forward pass (no duplicate BERT
    encoder run). Extracts Q, K, V from the last layer's attention projections
    applied to hidden_states[-2] (the input to the final transformer block).

    Concept: instead of matching attention probabilities (which depend on head
    count), we form Q·Q^T / sqrt(d_h), K·K^T / sqrt(d_h), V·V^T / sqrt(d_h)
    at the last transformer layer and match those distributions. By reshaping
    to a fixed `relation_heads` count, the loss is independent of the actual
    head count of either model.
    """
    s_enc = student_model.bert.encoder.layer[-1].attention.self
    t_enc = teacher_model.bert.encoder.layer[-1].attention.self

    # Extract Q, K, V from the cached last-layer inputs — no re-forward.
    s_input = student_hidden_states[-2]                          # [B, T, Ds]
    t_input = teacher_hidden_states[-2]                          # [B, T, Dt]

    def project(enc, x):
        return enc.query(x), enc.key(x), enc.value(x)

    s_q, s_k, s_v = project(s_enc, s_input)
    with torch.no_grad():
        t_q, t_k, t_v = project(t_enc, t_input)

    def reshape_to_relation_heads(x: torch.Tensor) -> torch.Tensor:
        """[B, T, D] → [B, relation_heads, T, D / relation_heads]."""
        B, T, D = x.shape
        assert D % relation_heads == 0, (
            f"hidden size {D} not divisible by relation_heads={relation_heads}"
        )
        d_h = D // relation_heads
        return x.view(B, T, relation_heads, d_h).permute(0, 2, 1, 3).contiguous()

    def relation_distribution(x: torch.Tensor) -> torch.Tensor:
        """x: [B, H, T, d_h] → softmax over keys of x · x^T / sqrt(d_h)."""
        d_h = x.size(-1)
        scores = torch.matmul(x, x.transpose(-1, -2)) / math.sqrt(d_h)
        # Mask padding keys before softmax.
        key_mask = attention_mask.unsqueeze(1).unsqueeze(2)        # [B, 1, 1, T]
        scores = scores.masked_fill(key_mask == 0, float("-inf"))
        return F.softmax(scores, dim=-1)

    losses = []
    for s_x, t_x in [(s_q, t_q), (s_k, t_k), (s_v, t_v)]:
        s_rel = relation_distribution(reshape_to_relation_heads(s_x))
        t_rel = relation_distribution(reshape_to_relation_heads(t_x))
        # KL per (head, query) row, masked & averaged over real queries.
        kl = (t_rel * (t_rel.clamp(min=eps).log() - s_rel.clamp(min=eps).log())).sum(-1)
        q_mask = attention_mask.unsqueeze(1).float()               # [B, 1, T]
        kl = (kl * q_mask).sum() / (q_mask.sum() * relation_heads).clamp(min=1.0)
        losses.append(kl)

    return torch.stack(losses).mean()


def rkd_loss(
    student_cls: torch.Tensor,
    teacher_cls: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Relational KD on [CLS] embeddings (Park et al., 2019).

    Distance term: match pairwise L2 distances between samples in a batch
                   (after dividing by the mean nonzero distance for scale
                   invariance — student & teacher live in different geometries).
    Angle term:    match the cosine of the angle formed by every triplet
                   (i, j, k) where j is the apex.

    This forces the student to preserve sample-to-sample relationships,
    which is exactly what the classifier head needs.
    """
    # Project teacher CLS down to student CLS dim for fair comparison? No —
    # RKD is scale-invariant by construction, so we can compare raw vectors.

    # ── Distance term ─────────────────────────────────────────────────────────
    def pdist(x):
        diff = x.unsqueeze(0) - x.unsqueeze(1)                     # [B, B, D]
        return diff.norm(dim=-1)                                    # [B, B]

    s_d = pdist(student_cls)
    t_d = pdist(teacher_cls)
    s_d = s_d / s_d[s_d > 0].mean().clamp(min=eps)
    t_d = t_d / t_d[t_d > 0].mean().clamp(min=eps)
    dist_loss = F.smooth_l1_loss(s_d, t_d)

    # ── Angle term ────────────────────────────────────────────────────────────
    def pangle(x):
        # vector from j to i and from j to k, then cosine for each (i, j, k)
        vd = x.unsqueeze(0) - x.unsqueeze(1)                        # [B, B, D]
        vd = F.normalize(vd, p=2, dim=-1)
        # angle = vd_ij · vd_kj^T → [B, B, B]
        return torch.einsum("ijd,kjd->ijk", vd, vd)

    s_a = pangle(student_cls)
    t_a = pangle(teacher_cls)
    angle_loss = F.smooth_l1_loss(s_a, t_a)

    return dist_loss + angle_loss


# ── Loss balancer & EMA ───────────────────────────────────────────────────────
N_LOSS_TERMS = 6   # logit, hidden, attn, minilm, rkd, ce
loss_balancer = UncertaintyWeights(N_LOSS_TERMS).to(CFG.device)
ema = EMAModel(student, CFG.ema_decay)


# ── Optimizer & schedule ──────────────────────────────────────────────────────
optimizer = torch.optim.AdamW(
    build_param_groups(student, hidden_proj, loss_balancer, weight_decay=CFG.weight_decay),
    lr=CFG.lr,
    betas=(0.9, 0.999),
    eps=1e-8,
)

steps_per_epoch = max(1, len(train_loader) // CFG.grad_accum)
total_steps  = steps_per_epoch * CFG.epochs
warmup_steps = int(CFG.warmup_ratio * total_steps)
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
)


# ── Eval helper ───────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader) -> float:
    model.eval()
    correct = total = 0
    for batch in loader:
        input_ids      = batch["input_ids"].to(CFG.device)
        attention_mask = batch["attention_mask"].to(CFG.device)
        labels         = batch["label"].to(CFG.device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total   += labels.size(0)
    return correct / total


# ── Baseline ──────────────────────────────────────────────────────────────────
print("\n=== Baselines ===")
teacher_acc = evaluate(teacher, val_loader)
student_acc = evaluate(student, val_loader)
print(f"  Teacher val acc : {teacher_acc:.4f}")
print(f"  Student val acc : {student_acc:.4f}  (before distillation)\n")


# ── Distillation training loop ────────────────────────────────────────────────
print("=== Distilling student from teacher ===")
amp_dtype = torch.bfloat16 if CFG.use_bf16 else torch.float16
scaler = torch.amp.GradScaler(
    "cuda",
    enabled=torch.cuda.is_available() and not CFG.use_bf16,
)

best_acc          = 0.0
best_state        = None
epochs_no_improve = 0
global_step       = 0

for epoch in range(1, CFG.epochs + 1):
    student.train()
    hidden_proj.train()
    loss_balancer.train()

    optimizer.zero_grad(set_to_none=True)
    epoch_loss   = 0.0
    epoch_terms  = torch.zeros(N_LOSS_TERMS)
    t0 = time.time()

    for step, batch in enumerate(train_loader):
        input_ids      = batch["input_ids"].to(CFG.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(CFG.device, non_blocking=True)
        labels         = batch["label"].to(CFG.device, non_blocking=True)

        with torch.amp.autocast(
            "cuda",
            enabled=torch.cuda.is_available(),
            dtype=amp_dtype,
        ):
            student_out = student(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                output_attentions=True,
            )
            with torch.no_grad():
                teacher_out = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=True,
                )

            # --- Five distillation signals ---------------------------------
            l_logit  = kl_logits(
                student_out.logits, teacher_out.logits,
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
            # MiniLMv2 uses cached hidden_states — no duplicate BERT forward pass.
            l_minilm = minilm_relation_loss(
                student, teacher,
                student_out.hidden_states, teacher_out.hidden_states,
                attention_mask, CFG.minilm_relation_heads,
            )
            # CLS token = position 0 of the last hidden state
            s_cls = student_out.hidden_states[-1][:, 0, :]
            t_cls = teacher_out.hidden_states[-1][:, 0, :]
            l_rkd = rkd_loss(s_cls, t_cls)

            # Ground-truth supervision
            l_ce = F.cross_entropy(student_out.logits, labels)

            term_losses = [l_logit, l_hidden, l_attn, l_minilm, l_rkd, l_ce]
            init_weights = torch.tensor(
                [CFG.w_logit, CFG.w_hidden, CFG.w_attn,
                 CFG.w_minilm, CFG.w_rkd, CFG.w_ce],
                device=CFG.device,
            )

            if CFG.learn_loss_weights:
                # Apply user-supplied weights as fixed prior multipliers, then
                # let the balancer adapt their relative scales.
                weighted = [w * l for w, l in zip(init_weights, term_losses)]
                loss = loss_balancer(weighted)
            else:
                loss = sum(w * l for w, l in zip(init_weights, term_losses))

            loss = loss / CFG.grad_accum

        scaler.scale(loss).backward()

        epoch_loss  += loss.item() * CFG.grad_accum
        epoch_terms += torch.tensor([l.detach().float().item() for l in term_losses])

        # Gradient accumulation step
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

    # ── End-of-epoch eval (using EMA weights for stability) ──────────────────
    original = ema.apply_to(student)
    val_acc  = evaluate(student, val_loader)
    EMAModel.restore(student, original)

    avg_loss   = epoch_loss / len(train_loader)
    avg_terms  = epoch_terms / len(train_loader)
    elapsed    = time.time() - t0
    term_names = ["logit", "hidden", "attn", "minilm", "rkd", "ce"]
    term_str   = "  ".join(f"{n}={v:.3f}" for n, v in zip(term_names, avg_terms))

    print(
        f"\nEpoch {epoch}/{CFG.epochs}  loss={avg_loss:.4f}  "
        f"val_acc(EMA)={val_acc:.4f}  ({elapsed:.1f}s)\n"
        f"   per-term: {term_str}"
    )

    if CFG.learn_loss_weights:
        learned = torch.exp(-loss_balancer.log_var).detach().cpu().tolist()
        learned_str = "  ".join(f"{n}={w:.2f}" for n, w in zip(term_names, learned))
        print(f"   learned weights (exp(-s)): {learned_str}")

    # Early stopping on EMA-evaluated validation accuracy
    if val_acc > best_acc + CFG.early_stop_min_delta:
        best_acc          = val_acc
        epochs_no_improve = 0
        # Snapshot EMA weights — those are what we'll save & report.
        best_state = {k: v.clone() for k, v in ema.shadow.items()}
        print(f"   ↳ new best val_acc={best_acc:.4f}  (EMA snapshot saved)\n")
    else:
        epochs_no_improve += 1
        print(
            f"   ↳ no improvement for {epochs_no_improve}/"
            f"{CFG.early_stop_patience} epoch(s)  (best={best_acc:.4f})\n"
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
    print(f"Restored best EMA weights (val_acc={best_acc:.4f}).")

os.makedirs(CFG.save_dir, exist_ok=True)
student.save_pretrained(CFG.save_dir)
tokenizer.save_pretrained(CFG.save_dir)
print(f"Distilled student saved to '{CFG.save_dir}/'")

final_acc = evaluate(student, val_loader)
print("\n=== Final Results ===")
print(f"Teacher  acc: {teacher_acc:.4f}   params: {teacher_params:,}")
print(f"Student  acc: {final_acc:.4f}   params: {student_params:,}")
print(f"Accuracy retention : {final_acc / teacher_acc * 100:.1f}%")
print(f"Parameter reduction: {(1 - student_params / teacher_params) * 100:.1f}%")
