"""
Architecture-agnostic layer extraction utilities for MiniLMv2 Q-relation KD.

Extracts Q at the last transformer layer of any supported HuggingFace model
type from cached hidden_states, without requiring a model forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# Registry of architecture paths: (layers_container, norm_attr, q_proj_attr)
# Each path is resolved via getattr traversal from the model root.
ARCH_LAYER_PATHS = {
    "qwen2":    ("model.layers", "input_layernorm", "self_attn.q_proj"),
    "llama":    ("model.layers", "input_layernorm", "self_attn.q_proj"),
    "mistral":  ("model.layers", "input_layernorm", "self_attn.q_proj"),
    "gemma":    ("model.layers", "input_layernorm", "self_attn.q_proj"),
    "gemma2":   ("model.layers", "input_layernorm", "self_attn.q_proj"),
    "opt":      ("model.decoder.layers", "self_attn_layer_norm", "self_attn.q_proj"),
    "gpt_neox": ("gpt_neox.layers", "input_layernorm", "attention.query_key_value"),
    "gpt2":     ("transformer.h", "ln_1", "attn.c_attn"),
}


def _resolve_arch_paths(model):
    """Return (layers_container, norm_attr_name, q_projector_name) for model type."""
    arch = getattr(model.config, "model_type", "").lower()
    paths = ARCH_LAYER_PATHS.get(arch)
    if paths is None:
        raise NotImplementedError(
            f"Architecture '{arch}' not supported by MiniLMv2 Q-extraction. "
            f"Supported: {list(ARCH_LAYER_PATHS)}. "
            f"Add an entry to ARCH_LAYER_PATHS for '{arch}'."
        )
    # Resolve the dotted layers path from the model root.
    layer_container = model
    for part in paths[0].split("."):
        layer_container = getattr(layer_container, part)
    return layer_container, paths[1], paths[2]


def _split_q_from_qkv(weight: torch.Tensor, hidden_size: int) -> torch.nn.Linear:
    """Split a combined QKV weight (GPT-NeoX / GPT-2 style) into Q-only."""
    q_weight = weight[:hidden_size, :]
    q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
    q_proj.weight.data = q_weight
    return q_proj


def extract_last_q(model, hidden_states):
    """
    Extract Q at the last transformer layer of any supported architecture.

    HuggingFace ``output_hidden_states`` caches the *output* of each layer
    plus the embedding output at index 0, so ``hidden_states[-2]`` is the
    input to the final transformer block. Apply that block's pre-attention
    norm to match what self-attention actually sees.

    Parameters
    ----------
    model:
        HuggingFace model supporting ``.config.model_type``.
    hidden_states: list or tuple of torch.Tensor
        Cached hidden states from a forward pass with
        ``output_hidden_states=True``.  Length = num_layers + 1.

    Returns
    -------
    torch.Tensor
        Q at the last layer, shape ``[B, T, hidden_size]``.
    """
    layers, norm_attr, q_attr = _resolve_arch_paths(model)
    last_block = layers[-1]
    norm = getattr(last_block, norm_attr)
    normed = norm(hidden_states[-2])

    # Resolve the Q projection (dotted attr name).
    q_projector = last_block
    for part in q_attr.split("."):
        q_projector = getattr(q_projector, part)

    hidden = model.config.hidden_size

    # GPT-2 style: `Conv1D` combined QKV projection. Its weight is laid out as
    # (in_features, out_features) = [H, 3H], so the shape[0] == H guard below
    # would wrongly skip splitting. Conv1D.forward yields [B, T, 3H] with Q
    # first, so slice the leading hidden block to recover Q-only.
    if hasattr(q_projector, "nf"):  # transformers.pytorch_utils.Conv1D
        return q_projector(normed)[..., :hidden]

    # Linear combined QKV (GPT-NeoX style): weight is [3H, H]; split rows for Q.
    if hasattr(q_projector, "weight") and q_projector.weight.shape[0] != hidden:
        q_proj = _split_q_from_qkv(q_projector.weight.data, hidden)
        return q_proj(normed.to(q_proj.weight.dtype))

    return q_projector(normed)
