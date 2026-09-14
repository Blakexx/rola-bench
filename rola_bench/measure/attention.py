"""THE ATTENTION REFERENCE: causal attention as a runner of the interleaving driver (`rola_devtools.interleave`).

    ArmSpec("attention", "rola_bench.measure.attention:arms", "flash", runner="attention",
            python=<a venv with torch>, cwd=<rola-bench>)

The runner takes a `QKV` cell (`rola_bench.measure.cells:qkv`: tokens, value width, heads, batch, dtype, causal) and
times it through torch's flash backend -- Dao's FlashAttention-2 as torch compiles it, or FA3 (Hopper) and FA4
(Blackwell) once `torch.nn.attention.activate_flash_attention_impl` has registered them. The backend is forced, so torch
refuses a call flash cannot take rather than timing its math or memory-efficient backend; building the arm makes one
call, so that refusal names the arm before anything is timed. Each sample is one call between two CUDA events. What the
arm reports records torch's version and the active implementation, so a number cites its reference.

Which attention cell stands beside which RoLA cells is a point's (`registry.json`): the capacity-fair comparison is at
N = L (rola's docs/measurement.md), and a point states whether its RoLA cells' state count equals their tokens.
"""
from __future__ import annotations

from functools import partial


def arms(data) -> dict:
    from .cells import QKV

    if not isinstance(data, QKV):
        raise TypeError(f"the attention runner takes a QKV cell (rola_bench.measure.cells:qkv), got {type(data).__name__}")
    return {"flash": partial(_flash, data)}


def _flash(cell):
    import torch
    import torch.nn.functional as F
    from rola_devtools.interleave import Arm
    from torch.nn.attention import SDPBackend, current_flash_attention_impl, sdpa_kernel

    generator = torch.Generator(device="cuda").manual_seed(0)
    dtype = getattr(torch, cell.dtype)
    q, k, v = (torch.randn(cell.batch, cell.heads, cell.tokens, cell.dv, device="cuda", dtype=dtype, generator=generator)
               for _ in range(3))

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
    return Arm(cell=built, call=call, instrument="cuda_events")
