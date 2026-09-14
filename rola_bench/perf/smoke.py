"""The perf smoke: every scaling arm built on the GPU at a short sequence, one prefill and one training step each.

    python -m rola_bench.perf.smoke

A real run is the fleet job (`python -m rola_bench.fleet perf paper_v1`) or `python -m rola_bench.perf.scaling`. What
rola refuses (prefill while its kernel is rebuilt, training before its native backward), the smoke reports as a failure
with rola's own error.
"""
from __future__ import annotations

import sys


def run() -> bool:
    import torch

    from rola_bench.perf import _common as C
    from rola_bench.perf import scaling as sc

    if not torch.cuda.is_available():
        print("SMOKE FAILED: no CUDA device")
        return False
    ok = True
    x = torch.randn(1, 64, sc.DM, device=C.DEV, dtype=C.DT)
    for kind in ("attn", *sc.BASELINE_GEOM, *sc.ROLA_ARMS):
        try:
            module = sc.build_arm({"kind": kind}, n=64, dv=64)
            with torch.inference_mode():
                sc._fwd(module, x)
            sc._train_rep(module, x)
            print(f"OK   {kind}")
        except Exception as ex:  # noqa: BLE001 -- each arm's failure is reported and the rest still run
            ok = False
            print(f"FAIL {kind:36s} {type(ex).__name__}: {str(ex)[:160]}")
    print("SMOKE " + ("OK" if ok else "FAILED"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
