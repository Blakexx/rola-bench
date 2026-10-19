"""THE ATTENTION REFERENCE: causal attention through torch's forced flash backend, on a central QKV cell.

    flash(cell, params) -> Timed    a timing entry of rola-devtools' timing system: the call, what it built, its stopwatch

The root (`declare.py`) registers it on every QKV cell of the selected groups. The cell is rola-devtools'
(`rola_devtools.cells.qkv`: tokens, value width, heads, batch, dtype, causal) and its tensors are the central draw
(`rola_devtools.cells.qkv.realize`), so every library timed on a QKV cell reads the same queries, keys and values. The
call runs through Dao's FlashAttention-2 as torch compiles it, or FA3 (Hopper) and FA4 (Blackwell) once
`torch.nn.attention.activate_flash_attention_impl` has registered them. The backend is forced, so torch refuses a call
flash cannot take rather than timing its math or memory-efficient backend; building the entry makes one call, so that
refusal is recorded as the entry's failure before anything is timed. Each sample is one call between two CUDA events; a
call carries no state, so it needs no reset. What the entry reports records torch's version and the active
implementation, so a number cites its reference.

THE INCREMENTAL CELL (rola's docs/measurement.md, ruled 2026-10-18): with parameter `context` = T the cell's tokens are c
NEW tokens after T already absorbed -- their queries attend over a KV cache of T + c keys, bottom-right causal (each new
token sees the whole context and the chunk before it); the context's keys and values are drawn once at build from the
cell's seed plus one, the chunk's are the cell's. With `hidden` = h the call also projects the chunk: one cuBLAS GEMM
of the c tokens' hidden states `[batch, c, h]` by the QKV weights `[h, 3 x heads x dv]` for every head at once, the new
keys and values written into the cache's tail -- the layer level, the side a RoLA layer's producer stands beside.
`heads` overrides the cell's heads (a layer's projection is one GEMM for all of them). Without parameters the call is
the whole causal sequence from zero tokens, N = L's whole-sequence case.

Which attention cell stands beside which RoLA cells is a group's (`groups.py`).
"""
from __future__ import annotations

#: the parameters the entry reads: the context T, the hidden width h (0: attention alone), the heads (the cell's when 0)
PARAMS = ("context", "hidden", "heads")


def flash(cell, params):
    import torch
    import torch.nn.functional as F
    from rola_devtools.cells.qkv import QKVCell, realize
    from rola_devtools.timing import Timed
    from torch.nn.attention import SDPBackend, current_flash_attention_impl, sdpa_kernel

    if not isinstance(cell, QKVCell):
        raise TypeError(f"the attention reference takes a QKV cell (rola_devtools.cells.qkv), got {type(cell).__name__}")
    unknown = sorted(set(params) - set(PARAMS))
    if unknown:
        raise ValueError(f"the attention reference reads {PARAMS}, got {unknown}")
    context, hidden = int(params.get("context", 0)), int(params.get("hidden", 0))
    heads = int(params.get("heads", 0)) or cell.heads
    if context < 0 or hidden < 0 or heads < 1:
        raise ValueError(f"context and hidden are counts, heads at least 1: got {context}, {hidden}, {heads}")
    if heads != cell.heads and not hidden:
        raise ValueError("`heads` sizes a projection: give `hidden` with it")
    c, dv, b = cell.tokens, cell.dv, cell.batch
    dtype = getattr(torch, cell.dtype)
    q, k, v = realize(cell) if heads == cell.heads else (None, None, None)
    generator = torch.Generator(device="cuda").manual_seed(cell.seed + 1)

    def draw(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", dtype=torch.float32, generator=generator) * scale).to(dtype)

    if context or hidden:
        cache_k = draw(b, heads, context + c, dv)
        cache_v = draw(b, heads, context + c, dv)
        if q is not None:
            cache_k[:, :, context:] = k
            cache_v[:, :, context:] = v
    if hidden:
        x = draw(b, c, hidden)
        w = draw(hidden, 3 * heads * dv, scale=hidden ** -0.5)

    #: the chunk over the cache: torch's own flash op, whose `is_causal` is bottom-right -- the op torch's
    #: `causal_lower_right` dispatches to, called directly because that bias is a tensor subclass whose construction
    #: allocates an uninitialized c x (T + c) host tensor (36 GB at T = 1M, c = 16K)
    def attend(query):
        with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            if not (context or hidden):
                return F.scaled_dot_product_attention(q, k, v, is_causal=cell.causal)
            return torch.ops.aten._scaled_dot_product_flash_attention(query, cache_k, cache_v, 0.0,
                                                                      is_causal=cell.causal, return_debug_mask=False,
                                                                      scale=dv ** -0.5)[0]

    def launch():
        if not hidden:
            return attend(q)
        with torch.no_grad():
            qkv = torch.matmul(x, w).view(b, c, 3, heads, dv)
            new_q, new_k, new_v = (qkv[:, :, i].transpose(1, 2) for i in range(3))
            cache_k[:, :, context:] = new_k
            cache_v[:, :, context:] = new_v
            return attend(new_q.contiguous())

    launch()

    def call() -> float:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        launch()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    #: A LAYER-LEVEL ENTRY (Blake, 2026-09-15): attention as a library gives it, which is the only level a comparison
    #: between two libraries is fair at -- RoLA's own layer holds its projections and its state, and a reading pairs
    #: entries of one level or refuses. With `hidden` the entry holds its projection too (2026-10-18).
    built = {"cell": cell.name, "level": "layer", "library": "torch", "torch": torch.__version__, "backend": "flash",
             "implementation": current_flash_attention_impl() or "FA2", "tokens": c, "context": context,
             "hidden": hidden, "projection": "cublas" if hidden else None, "d_v": dv, "heads": heads, "batch": b,
             "dtype": cell.dtype, "causal": cell.causal, "mask": "bottom-right" if (context or hidden) else "full",
             "device": torch.cuda.get_device_name(0)}
    return Timed(call=call, built=built, instrument="cuda_events")
