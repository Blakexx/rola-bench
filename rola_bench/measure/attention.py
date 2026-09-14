"""THE ATTENTION REFERENCE: causal attention as an arm of the interleaving driver (`rola_devtools.interleave`).

    ArmSpec("attention", "rola_bench.measure.attention:arms", "flash", python=<a venv with torch>, cwd=<rola-bench>)

A point states `tokens` and `d_v`; the arm realizes them as attention's own cell: one head of width `d_v` over `tokens`
tokens, bf16, causal, through torch's flash backend -- Dao's FlashAttention-2 as torch compiles it, or FA3 (Hopper) and
FA4 (Blackwell) once `torch.nn.attention.activate_flash_attention_impl` has registered them. The backend is forced, so
torch refuses a call flash cannot take rather than timing its math or memory-efficient backend; building the arm makes
one call, so that refusal names the arm before anything is timed. Each sample is one call between two CUDA events. The
cell records torch's version and the active implementation, so a number cites its reference.

The capacity-fair comparison against RoLA is at N = L (rola's docs/measurement.md): the point is the tokens and value
width, and whether a RoLA cell's state count equals its tokens is in that arm's own cell.
"""
from __future__ import annotations

from functools import partial


def arms(point: dict) -> dict:
    return {"flash": partial(_flash, int(point["tokens"]), int(point["d_v"]))}


def _flash(tokens: int, d_v: int):
    import torch
    import torch.nn.functional as F
    from rola_devtools.interleave import Arm
    from torch.nn.attention import SDPBackend, current_flash_attention_impl, sdpa_kernel

    generator = torch.Generator(device="cuda").manual_seed(0)
    q, k, v = (torch.randn(1, 1, tokens, d_v, device="cuda", dtype=torch.bfloat16, generator=generator)
               for _ in range(3))

    def launch():
        with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)

    launch()

    def call() -> float:
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        launch()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end)

    cell = {"library": "torch", "torch": torch.__version__, "backend": "flash",
            "implementation": current_flash_attention_impl() or "FA2", "tokens": tokens, "d_v": d_v, "heads": 1,
            "batch": 1, "dtype": "bfloat16", "causal": True, "device": torch.cuda.get_device_name(0)}
    return Arm(cell=cell, call=call, instrument="cuda_events")
