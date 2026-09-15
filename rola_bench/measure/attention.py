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

Which attention cell stands beside which RoLA cells is a group's (`groups.py`): the capacity-fair comparison is at N = L
(rola's docs/measurement.md), and a group states whether its RoLA cells' state count equals their tokens.
"""
from __future__ import annotations


def flash(cell, params):
    import torch
    import torch.nn.functional as F
    from rola_devtools.cells.qkv import QKVCell, realize
    from rola_devtools.timing import Timed
    from torch.nn.attention import SDPBackend, current_flash_attention_impl, sdpa_kernel

    if not isinstance(cell, QKVCell):
        raise TypeError(f"the attention reference takes a QKV cell (rola_devtools.cells.qkv), got {type(cell).__name__}")
    if params:
        raise ValueError(f"the attention reference reads no parameters, got {sorted(params)}")
    q, k, v = realize(cell)

    def launch():
        with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, is_causal=cell.causal)

    launch()

    def call() -> float:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        launch()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    built = {"cell": cell.name, "library": "torch", "torch": torch.__version__, "backend": "flash",
             "implementation": current_flash_attention_impl() or "FA2", "tokens": cell.tokens, "d_v": cell.dv,
             "heads": cell.heads, "batch": cell.batch, "dtype": cell.dtype, "causal": cell.causal,
             "device": torch.cuda.get_device_name(0)}
    return Timed(call=call, built=built, instrument="cuda_events")
