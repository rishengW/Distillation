"""
Pure distillation loss functions.

Every function here takes tensors and returns a scalar tensor. No model
coupling, no HuggingFace types — that makes them straightforward to unit-test
with hand-computed expected values on tiny inputs.

Used by both distill_advanced.py (Qwen LLM) and distill_advanced_bert.py
(BERT classifier).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Logit KD ──────────────────────────────────────────────────────────────────
def kl_logits_per_token(
    student_logits: torch.Tensor,    # [B, T, V]
    teacher_logits: torch.Tensor,    # [B, T, V]
    label_mask: torch.Tensor,        # [B, T]  bool/int: 1 where real token
    temperature: float,
    reverse: bool = False,
) -> torch.Tensor:
    """
    Token-level KL on temperature-softened next-token distributions, masked
    and averaged over real (non-padding) positions. Math is in fp32 for
    stability regardless of model dtype.

    Forward KL  KL(teacher || student) = mode-covering (Hinton 2015 default).
    Reverse KL  KL(student || teacher) = mode-seeking (sharper students,
                                         used in GKD-style on-policy KD).
    """
    s = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t = F.log_softmax(teacher_logits.float() / temperature, dim=-1)

    if reverse:
        kl = (s.exp() * (s - t)).sum(dim=-1)        # [B, T]
    else:
        kl = (t.exp() * (t - s)).sum(dim=-1)        # [B, T]

    mask = label_mask.float()
    return (kl * mask).sum() / mask.sum().clamp(min=1.0) * (temperature ** 2)


def kl_logits_classification(
    student_logits: torch.Tensor,    # [B, C]
    teacher_logits: torch.Tensor,    # [B, C]
    temperature: float,
    reverse: bool = False,
) -> torch.Tensor:
    """KL on classifier logits (no token dimension). Same scaling as above."""
    s = F.log_softmax(student_logits.float() / temperature, dim=-1)
    t = F.log_softmax(teacher_logits.float() / temperature, dim=-1)

    if reverse:
        kl = F.kl_div(t, s, reduction="batchmean", log_target=True)
    else:
        kl = F.kl_div(s, t, reduction="batchmean", log_target=True)
    return kl * (temperature ** 2)


# ── Hidden-state matching ─────────────────────────────────────────────────────
def hidden_mse(
    student_hidden_states: tuple[torch.Tensor, ...],
    teacher_hidden_states: tuple[torch.Tensor, ...],
    layer_map: list[int],
    proj: "HiddenProjector",         # forward declaration (in components.py)
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Per-token MSE between projected student hiddens and teacher hiddens at
    mapped layers. Both `*_hidden_states` are tuples of length (num_layers+1)
    where index 0 is the embedding output.
    """
    mask     = attention_mask.unsqueeze(-1).float()              # [B, T, 1]
    n_tokens = mask.sum().clamp(min=1.0)
    losses   = []

    # Embedding-layer alignment.
    s_emb = proj(0, student_hidden_states[0].float())
    t_emb = teacher_hidden_states[0].float()
    losses.append(((s_emb - t_emb) ** 2 * mask).sum() / (n_tokens * t_emb.size(-1)))

    for s_idx, t_idx in enumerate(layer_map):
        s_h = proj(s_idx + 1, student_hidden_states[s_idx + 1].float())
        t_h = teacher_hidden_states[t_idx + 1].float()
        losses.append(((s_h - t_h) ** 2 * mask).sum() / (n_tokens * t_h.size(-1)))

    return torch.stack(losses).mean()


# ── Attention-map KD ──────────────────────────────────────────────────────────
def attention_kl(
    student_attn: tuple[torch.Tensor, ...],
    teacher_attn: tuple[torch.Tensor, ...],
    layer_map: list[int],
    attention_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    KL on attention probability matrices at mapped layers, with heads
    averaged so the loss is invariant to head-count mismatches.

    Per-row re-normalization after key-masking is the part most "advanced"
    repos get wrong: the softmax was computed by the model with padding keys
    included (they contribute exp(-inf) ≈ 0), but downstream you still want
    the surviving rows to sum to 1 — otherwise the KL is over distributions
    that don't normalize and the gradient gets noisy near the mask boundary.
    """
    q_mask = attention_mask.unsqueeze(-1).float()                 # [B, T, 1]
    k_mask = attention_mask.unsqueeze(1).float()                  # [B, 1, T]
    losses = []

    for s_idx, t_idx in enumerate(layer_map):
        s_a = student_attn[s_idx].float().mean(dim=1)             # [B, T, T]
        t_a = teacher_attn[t_idx].float().mean(dim=1)
        s_a = s_a * k_mask
        t_a = t_a * k_mask
        s_a = s_a / s_a.sum(dim=-1, keepdim=True).clamp(min=eps)
        t_a = t_a / t_a.sum(dim=-1, keepdim=True).clamp(min=eps)

        kl = (t_a * (t_a.clamp(min=eps).log() - s_a.clamp(min=eps).log())).sum(-1)
        kl = (kl * q_mask.squeeze(-1)).sum() / q_mask.sum().clamp(min=1.0)
        losses.append(kl)

    return torch.stack(losses).mean()


# ── MiniLMv2 self-relation ────────────────────────────────────────────────────
def reshape_to_relation_heads(x: torch.Tensor, relation_heads: int) -> torch.Tensor:
    """[B, T, D] → [B, relation_heads, T, D / relation_heads]."""
    B, T, D = x.shape
    if D % relation_heads != 0:
        raise ValueError(
            f"hidden size {D} not divisible by relation_heads={relation_heads}"
        )
    d_h = D // relation_heads
    return x.view(B, T, relation_heads, d_h).permute(0, 2, 1, 3).contiguous()


def relation_distribution(
    x: torch.Tensor,                 # [B, H, T, d_h]
    attention_mask: torch.Tensor,    # [B, T]
) -> torch.Tensor:
    """Softmax(x · x^T / sqrt(d_h)) with padding keys masked to -inf."""
    d_h = x.size(-1)
    scores = torch.matmul(x, x.transpose(-1, -2)) / math.sqrt(d_h)
    key_mask = attention_mask.unsqueeze(1).unsqueeze(2)            # [B,1,1,T]
    scores = scores.masked_fill(key_mask == 0, float("-inf"))
    return F.softmax(scores.float(), dim=-1)


def minilm_relation_kl(
    s_proj: torch.Tensor,            # [B, T, Ds] — Q (or K, V) of student
    t_proj: torch.Tensor,            # [B, T, Dt] — Q (or K, V) of teacher
    attention_mask: torch.Tensor,
    relation_heads: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    KL between teacher and student self-relation distributions, head-agnostic
    via reshaping. Pass Q for Q-relation (use this for GQA models), or K/V
    for K-relation / V-relation (works for full-MHA models like BERT).
    """
    s_rel = relation_distribution(reshape_to_relation_heads(s_proj, relation_heads), attention_mask)
    t_rel = relation_distribution(reshape_to_relation_heads(t_proj, relation_heads), attention_mask)
    kl = (t_rel * (t_rel.clamp(min=eps).log() - s_rel.clamp(min=eps).log())).sum(-1)
    q_mask = attention_mask.unsqueeze(1).float()                   # [B, 1, T]
    return (kl * q_mask).sum() / (q_mask.sum() * relation_heads).clamp(min=1.0)


# ── Relational KD (RKD) ───────────────────────────────────────────────────────
def _scale_normalized_pdist(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Pairwise L2 distances, divided by their mean nonzero value."""
    d = (x.unsqueeze(0) - x.unsqueeze(1)).norm(dim=-1)
    nonzero = d[d > 0]
    scale = nonzero.mean() if nonzero.numel() > 0 else d.new_tensor(1.0)
    return d / scale.clamp(min=eps)


def _triplet_angles(x: torch.Tensor) -> torch.Tensor:
    """Cosines of triplet angles (i, j, k) with j as apex."""
    vd = x.unsqueeze(0) - x.unsqueeze(1)               # [B, B, D]
    vd = F.normalize(vd, p=2, dim=-1)
    return torch.einsum("ijd,kjd->ijk", vd, vd)


def rkd_loss_pooled(
    student_emb: torch.Tensor,       # [B, D_s]
    teacher_emb: torch.Tensor,       # [B, D_t]
) -> torch.Tensor:
    """RKD distance + angle loss on pre-pooled embeddings."""
    s_d = _scale_normalized_pdist(student_emb.float())
    t_d = _scale_normalized_pdist(teacher_emb.float())
    dist_loss = F.smooth_l1_loss(s_d, t_d)

    s_a = _triplet_angles(student_emb.float())
    t_a = _triplet_angles(teacher_emb.float())
    angle_loss = F.smooth_l1_loss(s_a, t_a)

    return dist_loss + angle_loss


def rkd_loss_token_pool(
    student_hidden: torch.Tensor,    # [B, T, D_s]
    teacher_hidden: torch.Tensor,    # [B, T, D_t]
    attention_mask: torch.Tensor,    # [B, T]
) -> torch.Tensor:
    """RKD on mean-pooled token embeddings (causal LMs have no [CLS])."""
    mask = attention_mask.unsqueeze(-1).float()
    s_pool = (student_hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    t_pool = (teacher_hidden.float() * mask).sum(1) / mask.sum(1).clamp(min=1.0)
    return rkd_loss_pooled(s_pool, t_pool)


# ── Hard supervision ──────────────────────────────────────────────────────────
def causal_ce_loss(
    student_logits: torch.Tensor,    # [B, T, V]
    labels:         torch.Tensor,    # [B, T]  (-100 at padding)
) -> torch.Tensor:
    """Standard next-token CE with i → i+1 shift."""
    shift_logits = student_logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


# ── Layer-mapping helper ──────────────────────────────────────────────────────
def build_layer_map(n_student: int, n_teacher: int) -> list[int]:
    """
    Skip-stride mapping: for each student layer index, the teacher layer it
    should imitate. With 2 student layers and 12 teacher layers this gives
    [5, 11] — the student matches the middle and final teacher representation.
    """
    if n_student <= 0 or n_teacher <= 0:
        raise ValueError(f"layer counts must be positive ({n_student}, {n_teacher})")
    if n_student == 1:
        return [n_teacher - 1]
    stride = n_teacher / n_student
    return [int(round((i + 1) * stride)) - 1 for i in range(n_student)]
