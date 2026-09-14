"""CPU tests for the cross-model FINAL-attention-matrix extractors (rank bench, #94 C2).

Builds a tiny REAL instance of each baseline layer and runs its extractor, asserting the
returned M is [B,H,L,L], finite, and respects the feature-dim rank bound — for BOTH causal
settings (unmasked rank-law object + the literal causal operator). Every extractor but GLA's uses only the layers'
projections + feature maps and runs on CPU; GLA's runs its Triton kernel and needs CUDA.

Run: pytest tests/similarity/test_extractors_cpu.py
"""
import pytest
import torch

from rola_bench.similarity import evaluator as E

B, L, DM, H = 2, 8, 32, 2


def _eval_x(device="cpu"):
    torch.manual_seed(0)
    return torch.randn(B, L, DM, device=device)


def _check(layer, build_extractor, dbound, device="cpu"):
    x = _eval_x(device)
    last = None
    for causal in (False, True):
        M = build_extractor(causal=causal)(layer.eval(), x)
        assert M.shape == (B, H, L, L), f"shape {tuple(M.shape)}"
        assert torch.isfinite(M).all(), "non-finite M"
        r = E.rank_stats(M)["rank_1e-03"]
        assert r <= dbound + 1, f"causal={causal}: rank {r} > bound {dbound}"
        last = r
    return last


@pytest.mark.parametrize("fmap", ["elu", "hedgehog"])
def test_fla_linear_attn_extractor(fmap):
    from fla.layers import LinearAttention
    la = LinearAttention(hidden_size=DM, expand_k=1.0, expand_v=1.0, num_heads=H,
                         feature_map=fmap, do_feature_map_norm=True)
    dbound = la.head_k_dim * (2 if fmap == "hedgehog" else 1)
    _check(la, lambda causal: E.build_fla_linear_attn_extractor(causal=causal), dbound)


def test_softmax_attention_extractor():
    from zoology.mixers.attention import MHA
    _check(MHA(d_model=DM, num_heads=H), lambda causal: E.build_softmax_attention_extractor(causal=causal), L)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GLA's extractor runs fla's Triton kernel")
def test_gla_extractor():
    """The one extractor here that runs a kernel (chunk_gla on identity values), so it runs on CUDA."""
    from zoology.mixers.gla import GatedLinearAttention
    gla = GatedLinearAttention(d_model=DM, expand_k=1.0, expand_v=1.0, num_heads=H, use_short_conv=False).cuda()
    _check(gla, lambda causal: E.build_gla_extractor(causal=causal), gla.head_k_dim, device="cuda")


def test_based_extractor():
    from zoology.mixers.based import Based
    based = Based(d_model=DM, feature_dim=4, num_heads=H, num_key_value_heads=H,
                  feature_name="taylor_exp", use_short_conv=False)
    _check(based, lambda causal: E.build_based_extractor(causal=causal), 21)  # taylor 4 -> 1+4+16
