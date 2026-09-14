"""The MQAR smoke: every wiring's mixer built on the GPU, one forward pass each.

    python -m rola_bench.mqar.smoke --list     # the cells the smoke builds
    python -m rola_bench.mqar.smoke            # build and run them

Each cell is the MQAR geometry (`rola_bench.mqar.MQAR_GEOM`) at the smallest state count every wiring spells (N = 16),
built through `rola_bench.models.rola` as zoology's mixer over fla. What rola refuses, the smoke reports as a
failure with rola's own error. Real runs are `rola_bench.mqar.run` over `experiments/*.yaml`.
"""
from __future__ import annotations

import argparse
import sys

from rola_bench.models import rola as cells
from rola_bench.mqar import MQAR_GEOM

SMOKE_N = 16


def smoke_cells() -> list[cells.Cell]:
    return [cells.cell(wiring, SMOKE_N) for wiring in cells.WIRINGS]


def run() -> bool:
    import importlib

    import torch

    if not torch.cuda.is_available():
        print("SMOKE FAILED: no CUDA device")
        return False
    ok = True
    d_model = MQAR_GEOM["n_heads"] * MQAR_GEOM["d_v"]
    for cell in smoke_cells():
        config = cells.mixer_config(cell, **MQAR_GEOM)
        module_path, name = config["name"].rsplit(".", 1)
        try:
            mixer = getattr(importlib.import_module(module_path), name)(d_model=d_model, layer_idx=0,
                                                                         **config["kwargs"]).cuda().eval()
            with torch.no_grad():
                out = mixer(torch.randn(1, 64, d_model, device="cuda"))
            print(f"OK   {cell.wiring:34s} N={cell.n} -> {tuple(out.shape)}")
        except Exception as ex:  # noqa: BLE001 -- each cell's failure is reported and the rest still run
            ok = False
            print(f"FAIL {cell.wiring:34s} N={cell.n} -> {type(ex).__name__}: {str(ex)[:160]}")
    print("SMOKE " + ("OK" if ok else "FAILED"))
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the cells and exit")
    a = ap.parse_args(argv)
    if a.list:
        for cell in smoke_cells():
            print(cell.wiring, cell.widths)
        return 0
    return 0 if run() else 1


if __name__ == "__main__":
    sys.exit(main())
