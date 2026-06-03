"""
Reusable components used by the advanced distillation scripts.

  HiddenProjector       — per-layer linear maps from student to teacher hidden dim
  UncertaintyWeights    — Kendall et al. 2018 multi-task loss balancer
  EMAModel              — exponential moving average of model parameters
  build_param_groups    — AdamW groups with no-decay on bias/LayerNorm/RMSNorm

All free of HuggingFace coupling — just torch.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── Hidden-state projection heads (student dim → teacher dim) ─────────────────
class HiddenProjector(nn.Module):
    """One Linear per (mapped) student layer + one for the embedding layer."""

    def __init__(self, n_layers: int, in_dim: int, out_dim: int, bias: bool = False):
        super().__init__()
        # +1 for the embedding layer (index 0 in HuggingFace `hidden_states`).
        self.projs = nn.ModuleList(
            [nn.Linear(in_dim, out_dim, bias=bias) for _ in range(n_layers + 1)]
        )

    def forward(self, idx: int, x: torch.Tensor) -> torch.Tensor:
        return self.projs[idx](x)


# ── Uncertainty loss balancer (Kendall et al., 2018) ──────────────────────────
class UncertaintyWeights(nn.Module):
    """
    Combined loss = Σ_i ( exp(-s_i) · L_i + s_i ).

    s_i is a learnable log-variance per loss term. The optimizer can shrink
    unhelpful terms (large s_i → small precision) while the +s_i regularizer
    prevents collapse to zero. Initialize log_var = 0 so each term starts
    with weight 1.
    """

    def __init__(self, n: int, init_log_var: float = 0.0):
        super().__init__()
        self.log_var = nn.Parameter(torch.full((n,), init_log_var))

    def forward(self, losses: list[torch.Tensor]) -> torch.Tensor:
        stacked   = torch.stack(losses)
        precision = torch.exp(-self.log_var)
        return (precision * stacked + self.log_var).sum()


# ── EMA shadow ────────────────────────────────────────────────────────────────
class EMAModel:
    """
    Exponential moving average of model parameters.

    Set decay <= 0 to disable updates (the shadow keeps its initial copy and
    `update()` becomes a no-op). Useful on tight VRAM budgets where the
    shadow itself is the issue.

    Non-floating-point buffers (counters, masks) are copied verbatim so the
    EMA state always loads cleanly into the live model.
    """

    def __init__(self, model: nn.Module, decay: float):
        self.decay  = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        if self.decay <= 0:
            return
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
            else:
                self.shadow[k] = v.detach().clone()

    def apply_to(self, model: nn.Module) -> dict:
        """Swap in EMA weights, returning the originals for later restore."""
        original = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)
        return original

    @staticmethod
    def restore(model: nn.Module, original: dict) -> None:
        model.load_state_dict(original, strict=False)


# ── AdamW parameter groups ────────────────────────────────────────────────────
def build_param_groups(*modules: nn.Module, weight_decay: float):
    """
    Standard transformer recipe: no weight decay on bias / LayerNorm.weight /
    RMSNorm.weight. Decay is applied only to matrix-shaped weights.

    The keyword list covers BERT (LayerNorm), Qwen / Llama (RMSNorm via
    `*_norm.weight`), and any other model that follows either convention.
    """
    no_decay_keywords = ("bias", "LayerNorm.weight", "layer_norm.weight", "norm.weight")
    decay_params, no_decay_params = [], []
    seen = set()
    for m in modules:
        for n, p in m.named_parameters():
            if not p.requires_grad or id(p) in seen:
                continue
            seen.add(id(p))
            if any(k in n for k in no_decay_keywords):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
    return [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
