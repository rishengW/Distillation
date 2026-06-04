# HuggingFace Internals Review Patterns

Common pitfalls when code reaches into HuggingFace model internals (attention
projections, weight splitting, architecture dispatch). These patterns catch bugs
that linters cannot.

## Conv1D vs nn.Linear Weight Layout

HuggingFace uses `Conv1D` (from `transformers.pytorch_utils`) for GPT-2 and
older models, while newer architectures use `nn.Linear`. Their weight shapes
differ:

| Type | Weight shape | Forward output |
|---|---|---|
| `nn.Linear(H, D)` | `[D, H]` | `[..., D]` |
| `Conv1D(nf=D, nx=H)` | `[H, D]` | `[..., D]` |

The bug pattern: code checks `weight.shape[0] != hidden_size` to detect a
combined QKV projection (NeoX style, weight `[3H, H]`), but this guard fails
for Conv1D because `Conv1D.weight.shape[0] == H == hidden_size`.

### Detection

- `hasattr(proj, 'nf')` → Conv1D. Weight is `[H, D]`, forward returns `[..., D]`.
- `isinstance(proj, nn.Linear)` → Standard linear. Weight is `[D, H]`.
- Neither → unknown, raise or handle.

### Correct Fix Pattern

```python
hidden = model.config.hidden_size

# Conv1D (GPT-2 style): forward returns QKV concatenated [B,T,3H].
# Q occupies the first hidden_size positions.
if hasattr(q_projector, "nf"):
    return q_projector(normed)[..., :hidden]

# Linear combined QKV (GPT-NeoX style): weight [3H, H], split rows.
if (isinstance(q_projector, nn.Linear) and
    q_projector.weight.shape[0] != hidden):
    return split_q(q_projector.weight[:hidden, :], normed)

# Standard separate Q-proj: nn.Linear(H, H).
return q_projector(normed)
```

### What the Wrong Guard Looks Like

```python
# BROKEN — misses Conv1D because shape[0] == H == hidden_size
if hasattr(q_projector, "weight") and q_projector.weight.shape[0] != model.config.hidden_size:
    ...  # Conv1D never enters this branch
return q_projector(normed)  # returns [B,T,3H] instead of Q-only [B,T,H]
```

## Architecture Dispatch Tables

When code supports multiple architectures, verify that:
- Every architecture in the table has the correct attribute traversal path
- QKV-combined architectures (GPT-NeoX, GPT-2) are handled before standard ones
- The `model_type` string matches HuggingFace's `config.model_type` exactly

## Frozen Teacher Gradient Leaks

When teacher tensors are used in loss computations, verify they're inside
`torch.no_grad()` blocks. Common leak: tests that extract teacher projections
without `no_grad()`. While `requires_grad=False` prevents actual gradient
accumulation, the autograd graph still tracks the operations — wasting memory
and compute.

## Gradient Checkpointing + use_cache

When `gradient_checkpointing_enable()` is called:
- Verify `model.config.use_cache = False` is set (cached KV values can't be
  checkpointed)
- For HuggingFace causal LMs, this must be set BEFORE the first forward pass
