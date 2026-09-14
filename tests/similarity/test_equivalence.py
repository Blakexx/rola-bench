"""Equivalence tests: prove each extractor's FINAL matrix M is IDENTICAL to the canonical
operator, i.e. `M @ V == kernel_output` for the model's own kernel (the fla chunked-vs-naive
analog). Since every kernel here is linear in V, agreement of `M @ V` with the kernel's O on
generic V means M *is* the operator — faithfulness proven, not asserted.

Softmax is CPU-exact (we replicate the exact SelfAttention formula). The linear-attention family
runs the real (Triton) kernels, so those tests require CUDA and are skipped on CPU.

Run:  pytest tests/similarity/test_equivalence.py            # softmax (CPU) always
      pytest tests/similarity/test_equivalence.py -m gpu     # + kernel equivalence (needs CUDA)
"""
import pytest
import torch

from rola_bench.similarity import evaluator as E

B, L, DM, H = 2, 8, 32, 2
CUDA = torch.cuda.is_available()
gpu = pytest.mark.skipif(not CUDA, reason="needs CUDA (runs the real Triton kernels)")


def _apply(M, v):
    """O[b,i,h,d] = sum_j M[b,h,i,j] v[b,j,h,d] — apply the attention matrix to values."""
    return torch.einsum("bhij,bjhd->bihd", M, v)


def test_softmax_equiv_cpu():
    """M @ v must equal zoology SelfAttention's context (causal), exactly."""
    from zoology.mixers.attention import MHA
    torch.manual_seed(0)
    mha = MHA(d_model=DM, num_heads=H).eval()
    x = torch.randn(B, L, DM)
    from einops import rearrange
    qkv = rearrange(mha.Wqkv(x), "... (three h d) -> ... three h d", three=3, d=mha.head_dim)
    _q, _k, v = qkv.unbind(2)
    o_ref = mha.inner_attn(qkv)                              # [B,L,H,head_dim], the real context
    M = E.build_softmax_attention_extractor(causal=True)(mha, x)
    o_ours = _apply(M.to(v.dtype), v)
    rel = (o_ours - o_ref).abs().max() / (o_ref.abs().max() + 1e-9)
    assert rel < 1e-5, f"softmax M@v vs SelfAttention: rel {rel:.2e}"


@gpu
@pytest.mark.parametrize("fmap,norm", [("elu", False), ("elu", True), ("hedgehog", True)])
def test_fla_linear_equiv_gpu(fmap, norm):
    """M @ v must equal chunk_linear_attn(phi(q), phi(k), v) — our M is the kernel's operator."""
    from fla.layers import LinearAttention
    from fla.ops.linear_attn import chunk_linear_attn
    torch.manual_seed(0)
    dev = "cuda"
    la = LinearAttention(hidden_size=DM, expand_k=1.0, expand_v=1.0, num_heads=H,
                         feature_map=fmap, do_feature_map_norm=norm).to(dev).eval()
    x = torch.randn(B, L, DM, device=dev)
    from einops import rearrange
    d = la.head_k_dim
    q = rearrange(la.q_proj(x), "... (h d) -> ... h d", d=d)
    k = rearrange(la.k_proj(x), "... (h d) -> ... h d", d=d)
    qf, kf = la.feature_map_q(q), la.feature_map_k(k)
    v = torch.randn(B, L, H, la.head_v_dim, device=dev)
    o_ref, _ = chunk_linear_attn(qf, kf, v, scale=1.0, normalize=norm, output_final_state=False)
    M = E.build_fla_linear_attn_extractor(causal=True)(la, x)   # causal = the literal operator
    o_ours = _apply(M.float(), v.float())
    rel = (o_ours - o_ref.float()).abs().max() / (o_ref.float().abs().max() + 1e-9)
    assert rel < 2e-2, f"fla-linear/{fmap} norm={norm} M@v vs kernel: rel {rel:.2e}"


@gpu
def test_gdn_equiv_gpu():
    """GDN probe: M (= kernel output at v=identity) must satisfy M@v_rand == kernel(v_rand) for the
    SAME q/k/beta/g — proving the recovered M is the delta-rule operator (output linear in v)."""
    import torch.nn.functional as F
    from einops import rearrange
    from fla.layers import GatedDeltaNet
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    torch.manual_seed(0)
    dev = "cuda"
    gdn = GatedDeltaNet(hidden_size=DM, num_heads=H, head_dim=DM // H, use_short_conv=False).to(dev).eval()
    x = torch.randn(B, L, DM, device=dev)
    M = E.build_gdn_extractor()(gdn, x)                          # [B,H,L,L] via v=identity probe
    # replicate the extractor's q/k/beta/g, run the kernel on a RANDOM v
    dk = gdn.head_k_dim
    q = rearrange(F.silu(gdn.q_proj(x)), "... (h d) -> ... h d", d=dk)
    k = rearrange(F.silu(gdn.k_proj(x)), "... (h d) -> ... h d", d=dk)
    beta = gdn.b_proj(x).sigmoid()
    if getattr(gdn, "allow_neg_eigval", False):
        beta = beta * 2.0
    g = -gdn.A_log.float().exp() * F.softplus(gdn.a_proj(x).float() + gdn.dt_bias)
    v = torch.randn(B, L, q.shape[-2], gdn.head_v_dim, device=dev)
    o_ref, _ = chunk_gated_delta_rule(q, k, v, g=g.to(q.dtype), beta=beta.to(q.dtype),
                                      use_qk_l2norm_in_kernel=True, output_final_state=False)
    o_ours = _apply(M.float(), v.float())
    rel = (o_ours - o_ref.float()).abs().max() / (o_ref.float().abs().max() + 1e-9)
    assert rel < 2e-2, f"GDN M@v vs chunk_gated_delta_rule: rel {rel:.2e}"


@gpu
def test_gla_equiv_gpu():
    """GLA's recovered operator: M @ v must equal chunk_gla(q, k, v, g) on the same projections and the same log-decay
    the layer computes -- the extractor runs that kernel on identity values, so this is its linearity check."""
    import torch.nn.functional as F
    from einops import rearrange
    from fla.ops.gla import chunk_gla
    from zoology.mixers.gla import GatedLinearAttention
    torch.manual_seed(0)
    dev = "cuda"
    gla = GatedLinearAttention(d_model=DM, expand_k=1.0, expand_v=1.0, num_heads=H,
                               use_short_conv=False).to(dev).eval()
    x = torch.randn(B, L, DM, device=dev)
    q = rearrange(gla.q_proj(x), "... (h d) -> ... h d", d=gla.head_k_dim)
    k = rearrange(gla.k_proj(x), "... (h d) -> ... h d", d=gla.head_k_dim)
    if gla.feature_map_fn is not None:
        q, k = gla.feature_map_fn(q), gla.feature_map_fn(k)
    gk = F.logsigmoid(rearrange(gla.gk_proj(x), "... (h d) -> ... h d", d=gla.head_k_dim)) / gla.gate_logit_normalizer
    v = torch.randn(B, L, H, gla.head_v_dim, device=dev)
    o_ref, _ = chunk_gla(q, k, v, g=gk, output_final_state=False)
    M = E.build_gla_extractor(causal=True)(gla, x)
    rel = (_apply(M.float(), v.float()) - o_ref.float()).abs().max() / (o_ref.float().abs().max() + 1e-9)
    assert rel < 2e-2, f"GLA M@v vs chunk_gla: rel {rel:.2e}"


def test_based_equiv_cpu():
    """Based (train_view='quadratic', the MQAR config): M@v must equal Based's OWN quadratic
    readout (NOT fla's fused_chunk_based — a different kernel). Based computes, per its forward:
        A = (phi(q) phi(k)^T) ∘ tril ;  y = (A @ v) / (sum_{m<=n} phi(q_n).phi(k_m) + eps)
    which is exactly causal-row-normalized phi-gram = our M. CPU-exact (pure einsum)."""
    from zoology.mixers.based import Based
    torch.manual_seed(0)
    based = Based(d_model=DM, feature_dim=4, num_heads=H, num_key_value_heads=H,
                  feature_name="taylor_exp", train_view="quadratic", use_short_conv=False).eval()
    x = torch.randn(B, L, DM)
    fd, eps = based.feature_dim, based.eps
    # Based layout: [B,H,L,fd] (forward transposes), feature_map on last dim.
    q = based.proj_q(x).view(B, L, H, fd).transpose(1, 2)
    k = based.proj_k(x).view(B, L, H, fd).transpose(1, 2)
    v = torch.randn(B, H, L, based.head_dim)
    qf, kf = based.feature_map(q), based.feature_map(k)
    tril = torch.tril(torch.ones(L, L))
    A = torch.einsum("bhnd,bhmd->bhnm", qf, kf) * tril
    out = torch.einsum("bhnm,bhme->bhne", A, v)
    z = 1.0 / (torch.einsum("bhld,bhld->bhl", qf, kf.cumsum(2)) + eps)
    y_ref = (out * z.unsqueeze(-1)).transpose(1, 2)               # -> [B,L,H,d]
    M = E.build_based_extractor(causal=True)(based, x)
    o_ours = _apply(M.to(v.dtype), v.transpose(1, 2))            # v as [B,L,H,d]
    rel = (o_ours - y_ref).abs().max() / (y_ref.abs().max() + 1e-9)
    assert rel < 1e-4, f"Based M@v vs its quadratic readout: rel {rel:.2e}"
