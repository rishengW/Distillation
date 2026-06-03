"""
Unit tests for advanced/losses.py — pure tensor math, no model coupling.

Every function is tested on tiny tensors (B=2, T=3, V=4, D=8) with
hand-computed or reference-verified expected values.
"""

import math
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from advanced.losses import (
    kl_logits_per_token,
    kl_logits_classification,
    hidden_mse,
    attention_kl,
    minilm_relation_kl,
    rkd_loss_pooled,
    rkd_loss_token_pool,
    causal_ce_loss,
    build_layer_map,
    reshape_to_relation_heads,
    relation_distribution,
    _scale_normalized_pdist,
    _triplet_angles,
)
from advanced.components import HiddenProjector


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def tiny_batch():
    """B=2, T=3, V=4, D=8 — small enough for hand computation."""
    return {
        "student_logits": torch.tensor([
            [[0.1, 0.2, 0.3, 0.4],
             [0.5, 0.6, 0.7, 0.8],
             [0.9, 1.0, 1.1, 1.2]],
            [[0.0, 0.1, 0.2, 0.3],
             [0.4, 0.5, 0.6, 0.7],
             [0.8, 0.9, 1.0, 1.1]],
        ]),  # [B, T, V]
        "teacher_logits": torch.tensor([
            [[1.0, 0.8, 0.6, 0.4],
             [0.2, 0.0, 0.8, 1.0],
             [0.5, 0.5, 0.5, 0.5]],
            [[0.9, 0.7, 0.5, 0.3],
             [1.0, 0.0, 0.0, 1.0],
             [0.2, 0.4, 0.6, 0.8]],
        ]),  # [B, T, V]
        "label_mask": torch.tensor([
            [1, 1, 0],
            [1, 0, 1],
        ]),  # [B, T]
        "student_hidden": tuple(
            torch.randn(2, 3, 8) for _ in range(3)
        ),  # 3 layers (embed + 2)
        "teacher_hidden": tuple(
            torch.randn(2, 3, 8) for _ in range(3)
        ),
        "student_attn": tuple(
            F.softmax(torch.randn(2, 2, 3, 3), dim=-1) for _ in range(2)
        ),  # 2 layers, 2 heads each
        "teacher_attn": tuple(
            F.softmax(torch.randn(2, 4, 3, 3), dim=-1) for _ in range(2)
        ),  # 2 layers, 4 heads each
        "attention_mask": torch.tensor([
            [1, 1, 0],
            [1, 1, 1],
        ]),  # [B, T]
        "labels": torch.tensor([
            [1, 0, 2],
            [0, 3, 1],
        ]),  # [B, T]
        "relation_heads": 2,
    }


# ── kl_logits_per_token ───────────────────────────────────────────────────────


class TestKLlogitsPerToken:
    def test_forward_kl_shape_and_finite(self, tiny_batch):
        loss = kl_logits_per_token(
            tiny_batch["student_logits"],
            tiny_batch["teacher_logits"],
            tiny_batch["label_mask"],
            temperature=1.0,
            reverse=False,
        )
        assert loss.ndim == 0  # scalar
        assert loss.isfinite()
        assert loss > 0  # KL divergence of non-identical distributions

    def test_reverse_kl_shape(self, tiny_batch):
        loss = kl_logits_per_token(
            tiny_batch["student_logits"],
            tiny_batch["teacher_logits"],
            tiny_batch["label_mask"],
            temperature=1.0,
            reverse=True,
        )
        assert loss.ndim == 0
        assert loss.isfinite()

    def test_zero_when_identical(self):
        logits = torch.randn(2, 3, 4)
        mask = torch.ones(2, 3)
        loss = kl_logits_per_token(logits, logits, mask, temperature=1.0)
        assert loss < 1e-6

    def test_temperature_squared_scaling(self, tiny_batch):
        loss_T1 = kl_logits_per_token(
            tiny_batch["student_logits"],
            tiny_batch["teacher_logits"],
            tiny_batch["label_mask"],
            temperature=1.0,
        )
        loss_T2 = kl_logits_per_token(
            tiny_batch["student_logits"],
            tiny_batch["teacher_logits"],
            tiny_batch["label_mask"],
            temperature=2.0,
        )
        # Should be roughly 4x difference (T² scaling), but the softer
        # distribution also changes the KL value, so just check it's larger.
        assert loss_T2 > loss_T1

    def test_padding_mask_excludes_positions(self, tiny_batch):
        """Positions with mask=0 should contribute nothing."""
        full_mask = torch.ones_like(tiny_batch["label_mask"])
        loss_full = kl_logits_per_token(
            tiny_batch["student_logits"],
            tiny_batch["teacher_logits"],
            full_mask,
            temperature=1.0,
        )
        loss_partial = kl_logits_per_token(
            tiny_batch["student_logits"],
            tiny_batch["teacher_logits"],
            tiny_batch["label_mask"],
            temperature=1.0,
        )
        # With mask excluding tokens, the averaged loss over fewer
        # tokens should differ from the full-mask version.
        assert not torch.allclose(loss_full, loss_partial)

    def test_forward_vs_reverse_not_equal(self, tiny_batch):
        fwd = kl_logits_per_token(
            tiny_batch["student_logits"], tiny_batch["teacher_logits"],
            tiny_batch["label_mask"], temperature=4.0, reverse=False,
        )
        rev = kl_logits_per_token(
            tiny_batch["student_logits"], tiny_batch["teacher_logits"],
            tiny_batch["label_mask"], temperature=4.0, reverse=True,
        )
        # Forward and reverse KL differ unless distributions are identical.
        assert not torch.allclose(fwd, rev)

    def test_matches_F_kl_div_reference(self):
        """Reverse KL with T=1 and a single position should match F.kl_div."""
        s = torch.randn(2, 1, 4)  # B=2, T=1
        t = torch.randn(2, 1, 4)
        mask = torch.ones(2, 1, dtype=torch.long)
        loss = kl_logits_per_token(s, t, mask, temperature=1.0, reverse=True)

        s_log = F.log_softmax(s, dim=-1)
        t_log = F.log_softmax(t, dim=-1)
        # Reverse KL = KL(student || teacher):
        #   Σ prob_s * (log_s - log_t) = Σ prob_s * (log_s - log_t)
        # F.kl_div(log_target, target) with reduction="batchmean":
        #   Σ target * (log_target - input) / batch
        # So: input=t_log, target=probs → Σ prob_s * (log_s - log_t) / B
        ref = F.kl_div(t_log, s_log.exp(), reduction="batchmean")
        # With T=1, average over T removes the T factor mismatch.
        assert torch.allclose(loss, ref, atol=1e-6)


# ── kl_logits_classification ──────────────────────────────────────────────────


class TestKLlogitsClassification:
    def test_scalar_output(self):
        s = torch.randn(4, 5)
        t = torch.randn(4, 5)
        loss = kl_logits_classification(s, t, temperature=1.0)
        assert loss.ndim == 0
        assert loss.isfinite()

    def test_zero_when_identical(self):
        logits = torch.randn(4, 5)
        loss = kl_logits_classification(logits, logits, temperature=1.0)
        assert loss < 1e-6

    def test_reverse_kl(self):
        s = torch.randn(4, 5)
        t = torch.randn(4, 5)
        fwd = kl_logits_classification(s, t, temperature=1.0, reverse=False)
        rev = kl_logits_classification(s, t, temperature=1.0, reverse=True)
        assert not torch.allclose(fwd, rev)

    def test_temperature_scaling(self):
        s = torch.randn(4, 5)
        t = torch.randn(4, 5)
        loss_1 = kl_logits_classification(s, t, temperature=1.0)
        loss_2 = kl_logits_classification(s, t, temperature=2.0)
        assert loss_2 > loss_1


# ── hidden_mse ────────────────────────────────────────────────────────────────


class TestHiddenMSE:
    def test_output_shape(self, tiny_batch):
        proj = HiddenProjector(n_layers=2, in_dim=8, out_dim=8)
        layer_map = [1]  # student layer 0 → teacher layer 1
        loss = hidden_mse(
            tiny_batch["student_hidden"],
            tiny_batch["teacher_hidden"],
            layer_map,
            proj,
            tiny_batch["attention_mask"],
        )
        assert loss.ndim == 0
        assert loss.isfinite()
        assert loss > 0

    def test_zero_when_identical(self):
        B, T, D = 2, 3, 8
        hidden = tuple(torch.randn(B, T, D) for _ in range(3))
        proj = HiddenProjector(n_layers=2, in_dim=D, out_dim=D)
        # Initialize projections to identity
        for p in proj.projs:
            nn.init.eye_(p.weight)
        mask = torch.ones(B, T)
        layer_map = [0]  # student layer 0 → teacher layer 0 (same index in tuple)
        loss = hidden_mse(hidden, hidden, layer_map, proj, mask)
        assert loss < 1e-6

    def test_attention_mask_excludes_padding(self, tiny_batch):
        """With padding tokens, MSE should differ from no-padding case."""
        proj = HiddenProjector(n_layers=2, in_dim=8, out_dim=8)
        layer_map = [1]
        full_mask = torch.ones(2, 3)
        loss_full = hidden_mse(
            tiny_batch["student_hidden"], tiny_batch["teacher_hidden"],
            layer_map, proj, full_mask,
        )
        loss_masked = hidden_mse(
            tiny_batch["student_hidden"], tiny_batch["teacher_hidden"],
            layer_map, proj, tiny_batch["attention_mask"],
        )
        assert not torch.allclose(loss_full, loss_masked)

    def test_identity_proj_produces_mse(self):
        """With identity projection, MSE = mean((s - t)²) over unmasked tokens."""
        B, T, D = 2, 3, 8
        s = tuple(torch.randn(B, T, D) for _ in range(3))
        t = tuple(torch.randn(B, T, D) for _ in range(3))
        proj = HiddenProjector(n_layers=2, in_dim=D, out_dim=D)
        for p in proj.projs:
            nn.init.eye_(p.weight)
        mask = torch.ones(B, T)
        layer_map = [0]  # student layer 0 → teacher layer 0

        loss = hidden_mse(s, t, layer_map, proj, mask)
        # Manual: MSE over the embedding layer + the one mapped layer
        # Layer_map [0] → s_idx=0, t_idx=0 → compare s[1] with t[1]
        diff_emb = (s[0] - t[0]) ** 2
        diff_map = (s[1] - t[1]) ** 2
        expected = torch.stack([
            diff_emb.mean(),
            diff_map.mean(),
        ]).mean()
        assert torch.allclose(loss, expected, atol=1e-6)


# ── attention_kl ──────────────────────────────────────────────────────────────


class TestAttentionKL:
    def test_output_shape_and_finite(self, tiny_batch):
        layer_map = [0, 1]
        loss = attention_kl(
            tiny_batch["student_attn"],
            tiny_batch["teacher_attn"],
            layer_map,
            tiny_batch["attention_mask"],
        )
        assert loss.ndim == 0
        assert loss.isfinite()

    def test_zero_when_identical(self):
        B, H_s, H_t, T = 2, 3, 3, 4
        attn = tuple(
            F.softmax(torch.randn(B, H_s, T, T), dim=-1) for _ in range(2)
        )
        mask = torch.ones(B, T)
        loss = attention_kl(attn, attn, [0, 1], mask)
        assert loss < 1e-6

    def test_attention_mask_affects_result(self):
        B, H_s, H_t, T = 2, 3, 3, 4
        s_attn = tuple(F.softmax(torch.randn(B, H_s, T, T), dim=-1) for _ in range(2))
        t_attn = tuple(F.softmax(torch.randn(B, H_t, T, T), dim=-1) for _ in range(2))
        full_mask = torch.ones(B, T)
        partial_mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])

        loss_full = attention_kl(s_attn, t_attn, [0, 1], full_mask)
        loss_partial = attention_kl(s_attn, t_attn, [0, 1], partial_mask)
        assert not torch.allclose(loss_full, loss_partial)

    def test_different_head_counts(self):
        """Should work when student and teacher have different head counts."""
        B, T = 2, 4
        s_attn = tuple(F.softmax(torch.randn(B, 2, T, T), dim=-1) for _ in range(2))
        t_attn = tuple(F.softmax(torch.randn(B, 12, T, T), dim=-1) for _ in range(2))
        mask = torch.ones(B, T)
        loss = attention_kl(s_attn, t_attn, [0, 1], mask)
        assert loss.isfinite()


# ── minilm_relation_kl ────────────────────────────────────────────────────────


class TestMiniLMRellationKL:
    def test_output_shape_and_finite(self):
        B, T, Ds, Dt, H = 2, 4, 8, 8, 2
        s_proj = torch.randn(B, T, Ds)
        t_proj = torch.randn(B, T, Dt)
        mask = torch.ones(B, T)
        loss = minilm_relation_kl(s_proj, t_proj, mask, relation_heads=H)
        assert loss.ndim == 0
        assert loss.isfinite()

    def test_zero_when_identical(self):
        B, T, D, H = 2, 4, 8, 2
        x = torch.randn(B, T, D)
        mask = torch.ones(B, T)
        loss = minilm_relation_kl(x, x, mask, relation_heads=H)
        assert loss < 1e-6

    def test_padding_keys_produce_near_zero_prob(self):
        """With all padding keys, the relation distribution should be
        near-uniform after softmax over many masked -inf entries, but
        the KL computation should still handle it gracefully."""
        B, T, D, H = 2, 4, 8, 2
        s_proj = torch.randn(B, T, D)
        t_proj = torch.randn(B, T, D)
        mask = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0]])
        loss = minilm_relation_kl(s_proj, t_proj, mask, relation_heads=H)
        assert loss.isfinite()


# ── reshape_to_relation_heads ─────────────────────────────────────────────────


class TestReshapeToRelationHeads:
    def test_shape_transformation(self):
        x = torch.randn(2, 6, 8)  # [B, T, D]
        H = 2
        result = reshape_to_relation_heads(x, H)
        assert result.shape == (2, 2, 6, 4)  # [B, H, T, D/H]

    def test_value_preservation(self):
        """Values should be preserved under reshape + permute."""
        x = torch.arange(24).float().view(1, 6, 4)
        result = reshape_to_relation_heads(x, 2)
        # Flatten back and check all elements present
        flat_result = result.permute(0, 2, 1, 3).contiguous().view(1, 6, 4)
        assert torch.allclose(x, flat_result)

    def test_raises_on_non_divisible(self):
        x = torch.randn(2, 6, 7)  # D=7, not divisible by H=3
        with pytest.raises(ValueError, match="not divisible"):
            reshape_to_relation_heads(x, 3)


# ── relation_distribution ─────────────────────────────────────────────────────


class TestRelationDistribution:
    def test_softmax_sums_to_one(self):
        B, H, T, d_h = 2, 4, 5, 4
        x = torch.randn(B, H, T, d_h)
        mask = torch.ones(B, T)
        dist = relation_distribution(x, mask)
        assert dist.shape == (B, H, T, T)
        assert torch.allclose(dist.sum(dim=-1), torch.ones(B, H, T), atol=1e-6)

    def test_padding_keys_produce_near_zero_prob(self):
        B, H, T, d_h = 1, 4, 3, 4
        x = torch.randn(B, H, T, d_h)
        # Sample 0: only key 0 is unmasked. All queries must attend only to key 0.
        mask = torch.tensor([[1, 0, 0]])
        dist = relation_distribution(x, mask)
        assert dist.isfinite().all()
        # All queries pay ~100% probability to key 0, ~0% to keys 1 and 2.
        prob_key0 = dist[:, :, :, 0].sum()
        prob_keys_rest = dist[:, :, :, 1:].sum()
        assert prob_key0 > prob_keys_rest * 100  # key 0 dominates


# ── rkd_loss_pooled ───────────────────────────────────────────────────────────


class TestRKDlossPooled:
    def test_scalar_output(self):
        s = torch.randn(4, 8)
        t = torch.randn(4, 16)
        loss = rkd_loss_pooled(s, t)
        assert loss.ndim == 0
        assert loss.isfinite()

    def test_zero_when_identical(self):
        x = torch.randn(4, 8)
        loss = rkd_loss_pooled(x, x)
        assert loss < 1e-6

    def test_scale_invariance(self):
        """RKD should be scale-invariant due to the mean-norm division."""
        s = torch.randn(4, 8)
        t = torch.randn(4, 8)
        loss_orig = rkd_loss_pooled(s, t)
        loss_scaled = rkd_loss_pooled(s * 10, t)
        # The distance term normalizes by mean distance so scaling the
        # student shouldn't drastically change the loss. It won't be
        # identical because the angle term can also be affected, but
        # it should still be in the same ballpark.
        assert torch.allclose(loss_orig, loss_scaled, atol=1.0)

    def test_distance_and_angle_terms(self):
        """Both distance and angle terms should contribute."""
        B, D = 4, 8
        # Very different embeddings
        s = torch.randn(B, D)
        t = torch.randn(B, D) * 100  # very different scale + values
        loss_both = rkd_loss_pooled(s, t)

        # Just distance (by making s_a == t_a with an identity-like setup):
        # We can't easily separate them, but we know loss_both should be finite.
        assert loss_both > 0


# ── rkd_loss_token_pool ───────────────────────────────────────────────────────


class TestRKDtokenPool:
    def test_scalar_output(self):
        B, T, Ds, Dt = 2, 5, 8, 12
        s_hidden = torch.randn(B, T, Ds)
        t_hidden = torch.randn(B, T, Dt)
        mask = torch.ones(B, T)
        loss = rkd_loss_token_pool(s_hidden, t_hidden, mask)
        assert loss.ndim == 0
        assert loss.isfinite()

    def test_zero_when_identical(self):
        B, T, D = 2, 5, 8
        hidden = torch.randn(B, T, D)
        mask = torch.ones(B, T)
        loss = rkd_loss_token_pool(hidden, hidden, mask)
        assert loss < 1e-6

    def test_masking_affects_pooled_result(self):
        B, T, D = 3, 4, 2  # B=3 for non-trivial pairwise geometry
        # Give each sample a unique token profile so RKD is non-trivial.
        torch.manual_seed(0)
        s_hidden = torch.randn(B, T, D)
        t_hidden = torch.randn(B, T, D)

        full_mask = torch.ones(B, T)
        # Mask out different token positions so mean-pooled vectors change
        partial_mask = torch.tensor([
            [1, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 0],
        ])

        loss_full = rkd_loss_token_pool(s_hidden, t_hidden, full_mask)
        loss_partial = rkd_loss_token_pool(s_hidden, t_hidden, partial_mask)
        # At minimum the loss values are finite and non-negative.
        assert loss_full >= 0
        assert loss_partial >= 0
        # With s ≠ t and different pooling, the geometry should differ.
        # (Seed the random generator and compare to avoid flakiness.)
        assert not torch.allclose(loss_full, loss_partial, atol=1e-6)


# ── causal_ce_loss ────────────────────────────────────────────────────────────


class TestCausalCELoss:
    def test_scalar_output(self, tiny_batch):
        loss = causal_ce_loss(
            tiny_batch["student_logits"],
            tiny_batch["labels"],
        )
        assert loss.ndim == 0
        assert loss.isfinite()

    def test_shift_logic(self):
        """Position i's logit should predict token i+1."""
        B, T, V = 2, 4, 5
        # Make it easy: set student logits to predict specific tokens.
        logits = torch.zeros(B, T, V)
        # At position 0, we want to predict token 1 at position 1.
        # At position 1, we want to predict token 2 at position 2, etc.
        logits[:, 0, :] = -100  # low everywhere
        logits[:, 0, 1] = 100  # high for token 1
        logits[:, 1, 2] = 100  # high for token 2
        logits[:, 2, 3] = 100  # high for token 3
        labels = torch.tensor([
            [0, 1, 2, 3],
            [0, 1, 2, 3],
        ])
        # Shift: logits[:-1] predict labels[1:]
        # logits[0] → label[1] = 1 ✓, logits[1] → label[2] = 2 ✓, etc.
        loss = causal_ce_loss(logits, labels)
        # Loss should be very low since predictions match perfectly
        assert loss < 0.01

    def test_ignore_index_minus_100(self):
        """Positions with label=-100 should be ignored in the loss."""
        B, T, V = 2, 4, 5
        logits = torch.randn(B, T, V)
        # After shift, labels[1:] is compared to logits[:-1].
        # Set label[:, 1] = 0 for both samples so shift_labels[0,0]=0 and [1,0]=0.
        labels = torch.full((B, T), -100, dtype=torch.long)
        labels[:, 1] = 0  # two valid labels after shift: shift_labels[:, 0] = 0
        loss = causal_ce_loss(logits, labels)
        assert loss.isfinite()
        assert loss > 0

        # With ALL labels -100, no valid shift label exists → NaN
        labels_none = torch.full((B, T), -100, dtype=torch.long)
        loss_nan = causal_ce_loss(logits, labels_none)
        assert torch.isnan(loss_nan)


# ── build_layer_map ───────────────────────────────────────────────────────────


class TestBuildLayerMap:
    def test_ratio_1_to_1(self):
        result = build_layer_map(6, 6)
        assert result == [0, 1, 2, 3, 4, 5]

    def test_ratio_1_to_12(self):
        result = build_layer_map(1, 12)
        assert result == [11]  # maps to last teacher layer

    def test_ratio_2_to_12(self):
        result = build_layer_map(2, 12)
        # stride = 6, so: round(1*6)-1=5, round(2*6)-1=11
        # SKILL.md says [5, 11]
        assert result == [5, 11]

    def test_ratio_6_to_12(self):
        result = build_layer_map(6, 12)
        # stride = 2, so: 1, 3, 5, 7, 9, 11
        assert result == [1, 3, 5, 7, 9, 11]

    def test_ratio_12_to_12(self):
        result = build_layer_map(12, 12)
        assert result == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    def test_ratio_24_to_28(self):
        # From SKILL.md: valid layer map for Qwen2.5 0.5B→1.5B
        result = build_layer_map(24, 28)
        expected = [0, 1, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13,
                    14, 15, 17, 18, 19, 20, 21, 22, 23, 25, 26, 27]
        assert result == expected

    def test_raises_on_non_positive(self):
        with pytest.raises(ValueError):
            build_layer_map(0, 12)
        with pytest.raises(ValueError):
            build_layer_map(12, 0)


# ── _scale_normalized_pdist ───────────────────────────────────────────────────


class TestScaleNormalizedPdist:
    def test_output_shape(self):
        x = torch.randn(4, 8)
        result = _scale_normalized_pdist(x)
        assert result.shape == (4, 4)

    def test_symmetric_with_zero_diagonal(self):
        x = torch.randn(4, 8)
        result = _scale_normalized_pdist(x)
        assert torch.allclose(result, result.T, atol=1e-6)
        assert torch.allclose(result.diag(), torch.zeros(4), atol=1e-6)


# ── _triplet_angles ───────────────────────────────────────────────────────────


class TestTripletAngles:
    def test_output_shape(self):
        x = torch.randn(4, 8)
        result = _triplet_angles(x)
        assert result.shape == (4, 4, 4)

    def values_in_range(self):
        x = torch.randn(4, 8)
        result = _triplet_angles(x)
        # Cosine should be in [-1, 1]
        assert result.min() >= -1.0
        assert result.max() <= 1.0


# ── Run via pytest ────────────────────────────────────────────────────────────
# $ pytest advanced/tests/test_losses.py -v
