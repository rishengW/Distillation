"""
End-to-end smoke test for the advanced distillation pipeline.

Constructs minimal mock models and runs one training step of the
full pipeline to verify all components compose correctly without
crashes. Does NOT require network access or real HuggingFace models.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from advanced.losses import (
    build_layer_map,
    kl_logits_per_token,
    hidden_mse,
    attention_kl,
    minilm_relation_kl,
    rkd_loss_token_pool,
    causal_ce_loss,
)
from advanced.components import (
    HiddenProjector,
    UncertaintyWeights,
    EMAModel,
)


# ── Helper: tiny mock models ─────────────────────────────────────────────────


class MockConfig:
    """Emulate HuggingFace model.config for the pipeline."""
    def __init__(self, hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                 model_type="mock", _attn_implementation="eager"):
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.model_type = model_type
        self._attn_implementation = _attn_implementation


class MockAttention:
    """Mock self-attention returning dummy Q from q_proj."""
    def __init__(self, hidden_size):
        self.q_proj = nn.Linear(hidden_size, hidden_size)


class MockBlock(nn.Module):
    """Mock a single transformer block with self-attention."""
    def __init__(self, hidden_size):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.self_attn = MockAttention(hidden_size)


class MockModel(nn.Module):
    """
    Minimal model that produces outputs mimicking HuggingFace causal LM:
      - .logits        [B, T, V]
      - .hidden_states tuple of length (L+1)
      - .attentions    tuple of length L
      - .config        MockConfig
      - .model.layers  list of MockBlock
      - loss for eval
    """
    def __init__(self, hidden_size=16, num_layers=2, num_heads=2, vocab_size=16,
                 model_type="qwen2"):
        super().__init__()
        self.config = MockConfig(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            model_type=model_type,
        )
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        # Head that produces logits
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        # Embedding
        self.wte = nn.Embedding(vocab_size, hidden_size)

        # Model layers structure matching HF's model.model.layers
        class LayerContainer:
            pass
        self.model = LayerContainer()
        self.model.layers = nn.ModuleList(
            [MockBlock(hidden_size) for _ in range(num_layers)]
        )

    @staticmethod
    def gradient_checkpointing_enable():
        """No-op for mock models, mimics HuggingFace API."""
        pass

    def forward(self, input_ids, attention_mask=None, labels=None,
                output_hidden_states=False, output_attentions=False,
                use_cache=False, **kwargs):
        B, T = input_ids.shape
        # Embed
        h = self.wte(input_ids)  # [B, T, D]

        # Generate dummy hidden_states: (L+1) tensors
        hidden_states = [h]
        for i in range(self.num_layers):
            # Apply layernorm + q_proj like real models do in hidden_states
            block = self.model.layers[i]
            h = block.input_layernorm(h)
            _ = block.self_attn.q_proj(h)  # touch q_proj so it's "used"
            # Simple linear transform per layer
            h = h + 0.01 * torch.randn_like(h)  # random perturbation
            hidden_states.append(h)

        # Logits
        logits = self.lm_head(h)  # [B, T, V]

        # Dummy attentions: (L) tensors [B, H, T, T]
        attentions = tuple(
            F.softmax(torch.randn(B, self.config.num_attention_heads, T, T), dim=-1)
            for _ in range(self.num_layers)
        )

        # Loss (for perplexity eval)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.reshape(-1, self.vocab_size),
                shift_labels.reshape(-1),
                ignore_index=-100,
            )

        class Output:
            pass
        out = Output()
        out.logits = logits
        out.hidden_states = tuple(hidden_states)
        out.attentions = attentions
        out.loss = loss
        return out


# ── Smoke test ───────────────────────────────────────────────────────────────


@pytest.fixture
def pipeline_components():
    """Build all pipeline components with tiny dimensions for a single step."""
    B, T = 2, 8
    teacher_hidden, student_hidden = 16, 8
    teacher_layers, student_layers = 3, 2
    vocab_size = 16

    teacher = MockModel(
        hidden_size=teacher_hidden, num_layers=teacher_layers,
        num_heads=2, vocab_size=vocab_size, model_type="qwen2",
    )
    student = MockModel(
        hidden_size=student_hidden, num_layers=student_layers,
        num_heads=2, vocab_size=vocab_size, model_type="qwen2",
    )

    input_ids = torch.randint(0, vocab_size, (B, T)).long()
    attention_mask = torch.ones(B, T).long()
    labels = input_ids.clone()
    labels[:, T // 2:] = -100  # mask some as padding

    # Teacher always eval
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student.train()
    student.gradient_checkpointing_enable()

    return {
        "teacher": teacher,
        "student": student,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "B": B,
        "T": T,
        "teacher_hidden": teacher_hidden,
        "student_hidden": student_hidden,
        "teacher_layers": teacher_layers,
        "student_layers": student_layers,
        "vocab_size": vocab_size,
    }


class TestPipelineSmoke:
    """End-to-end smoke: verify all components compose without crash."""

    def test_build_layer_map(self, pipeline_components):
        t_layers = pipeline_components["teacher_layers"]
        s_layers = pipeline_components["student_layers"]
        layer_map = build_layer_map(s_layers, t_layers)
        assert len(layer_map) == s_layers
        assert all(0 <= i < t_layers for i in layer_map)

    def test_projection_head(self, pipeline_components):
        s_hidden = pipeline_components["student_hidden"]
        t_hidden = pipeline_components["teacher_hidden"]
        s_layers = pipeline_components["student_layers"]
        proj = HiddenProjector(s_layers, s_hidden, t_hidden)
        x = torch.randn(2, 8, s_hidden)
        out = proj(0, x)
        assert out.shape == (2, 8, t_hidden)

    def test_loss_balancer(self, pipeline_components):
        N = 6
        balancer = UncertaintyWeights(N)
        dummy_losses = [torch.tensor(0.5 * i) for i in range(N)]
        loss = balancer(dummy_losses)
        assert loss.ndim == 0
        assert loss.isfinite()

    def test_ema(self, pipeline_components):
        student = pipeline_components["student"]
        ema = EMAModel(student, decay=0.999)
        ema.update(student)
        original = ema.apply_to(student)
        EMAModel.restore(student, original)

    def test_six_loss_terms_compose(self, pipeline_components):
        """Run all six loss signals — the core of the pipeline."""
        tc = pipeline_components
        teacher, student = tc["teacher"], tc["student"]
        input_ids = tc["input_ids"]
        attention_mask = tc["attention_mask"]
        labels = tc["labels"]
        s_layers, t_layers = tc["student_layers"], tc["teacher_layers"]
        s_hidden, t_hidden = tc["student_hidden"], tc["teacher_hidden"]

        layer_map = build_layer_map(s_layers, t_layers)
        hidden_proj = HiddenProjector(s_layers, s_hidden, t_hidden)
        relation_heads = 2  # divides both 8 and 16

        # Forward passes (what the training loop does)
        student_out = student(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, output_attentions=True,
            use_cache=False,
        )
        with torch.no_grad():
            teacher_out = teacher(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, output_attentions=True,
                use_cache=False,
            )

        label_mask = (labels != -100)

        # --- Each loss term ---
        l_logit = kl_logits_per_token(
            student_out.logits, teacher_out.logits, label_mask,
            temperature=4.0, reverse=False,
        )
        assert l_logit.isfinite()

        l_hidden = hidden_mse(
            student_out.hidden_states, teacher_out.hidden_states,
            layer_map, hidden_proj, attention_mask,
        )
        assert l_hidden.isfinite()

        l_attn = attention_kl(
            student_out.attentions, teacher_out.attentions,
            layer_map, attention_mask,
        )
        assert l_attn.isfinite()

        # MiniLMv2 Q-only (for GQA models like Qwen2)
        from advanced.arch_utils import extract_last_q
        s_q = extract_last_q(student, student_out.hidden_states)
        with torch.no_grad():
            t_q = extract_last_q(teacher, teacher_out.hidden_states)
        l_minilm = minilm_relation_kl(
            s_q, t_q, attention_mask, relation_heads,
        )
        assert l_minilm.isfinite()

        l_rkd = rkd_loss_token_pool(
            student_out.hidden_states[-1],
            teacher_out.hidden_states[-1],
            attention_mask,
        )
        assert l_rkd.isfinite()

        l_ce = causal_ce_loss(student_out.logits, labels)
        assert l_ce.isfinite()

        # Combined loss
        term_losses = [l_logit, l_hidden, l_attn, l_minilm, l_rkd, l_ce]
        combined = sum(term_losses)
        assert combined.isfinite()
        assert combined > 0

    def test_architecture_dispatch(self):
        """Verify extract_last_q works on supported arch model_types."""
        from advanced.arch_utils import extract_last_q, ARCH_LAYER_PATHS
        assert "qwen2" in ARCH_LAYER_PATHS
        assert "llama" in ARCH_LAYER_PATHS
        assert "mistral" in ARCH_LAYER_PATHS
        assert "gemma" in ARCH_LAYER_PATHS
        assert "opt" in ARCH_LAYER_PATHS
        assert "gpt_neox" in ARCH_LAYER_PATHS
        assert "gpt2" in ARCH_LAYER_PATHS

        # Test Qwen2-like (our MockModel uses model.model.layers)
        model = MockModel(
            hidden_size=16, num_layers=2, num_heads=2, vocab_size=16,
            model_type="qwen2",
        )
        B, T, D = 2, 4, 16
        h = torch.randn(B, T, D)
        hidden_states = [h, h + 0.1, h + 0.2]
        q = extract_last_q(model, hidden_states)
        assert q.shape == (B, T, D)
        assert q.isfinite().all()

        # Test Llama-like (same structure, different model_type)
        model.config.model_type = "llama"
        q = extract_last_q(model, hidden_states)
        assert q.shape == (B, T, D)

    def test_gradient_flows_through_all_losses(self, pipeline_components):
        """Backprop through all losses and verify student gets gradients."""
        tc = pipeline_components
        teacher, student = tc["teacher"], tc["student"]
        input_ids = tc["input_ids"]
        attention_mask = tc["attention_mask"]
        labels = tc["labels"]
        s_layers, t_layers = tc["student_layers"], tc["teacher_layers"]
        s_hidden, t_hidden = tc["student_hidden"], tc["teacher_hidden"]

        layer_map = build_layer_map(s_layers, t_layers)
        hidden_proj = HiddenProjector(s_layers, s_hidden, t_hidden)
        loss_balancer = UncertaintyWeights(6)

        student_out = student(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, output_attentions=True,
            use_cache=False,
        )
        with torch.no_grad():
            teacher_out = teacher(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, output_attentions=True,
                use_cache=False,
            )

        label_mask = (labels != -100)
        from advanced.arch_utils import extract_last_q

        losses = [
            kl_logits_per_token(student_out.logits, teacher_out.logits, label_mask, 4.0),
            hidden_mse(student_out.hidden_states, teacher_out.hidden_states,
                       layer_map, hidden_proj, attention_mask),
            attention_kl(student_out.attentions, teacher_out.attentions,
                         layer_map, attention_mask),
            minilm_relation_kl(
                extract_last_q(student, student_out.hidden_states),
                extract_last_q(teacher, teacher_out.hidden_states),
                attention_mask, 2,
            ),
            rkd_loss_token_pool(student_out.hidden_states[-1],
                                teacher_out.hidden_states[-1], attention_mask),
            causal_ce_loss(student_out.logits, labels),
        ]

        init_weights = torch.tensor([1.0] * 6)
        weighted = [w * l for w, l in zip(init_weights, losses)]
        loss = loss_balancer(weighted)

        # Backprop
        loss.backward()

        # Verify student got gradients
        student_grads = [p.grad for p in student.parameters() if p.grad is not None]
        assert len(student_grads) > 0, "No gradients flowed to student"
        assert all(g.isfinite().all() for g in student_grads)

        # Verify teacher has NO gradients (frozen)
        teacher_grads = [p.grad for p in teacher.parameters() if p.grad is not None]
        assert len(teacher_grads) == 0, "Teacher should be frozen"

    def test_unknown_architecture_raises(self):
        """Unsupported arch should raise NotImplementedError."""
        from advanced.arch_utils import extract_last_q
        model = MockModel(model_type="unknown_arch_xyz")
        h = torch.randn(2, 4, 16)
        hidden_states = [h, h + 0.1, h + 0.2]
        with pytest.raises(NotImplementedError, match="not supported"):
            extract_last_q(model, hidden_states)


# ── Run via pytest ────────────────────────────────────────────────────────────
# $ python -m pytest advanced/tests/test_smoke.py -v
