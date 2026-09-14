"""The fleet's boot gate (FLEET_VERIFY_CMD): a rented box proves its GPU runs RoLA before it takes work.

One small RoLA layer (fla over rola) runs a forward and backward pass on the box's device, so a host with a broken
GPU, driver or extension fails here and is blocked instead of failing every cell of a batch. Prints "VERIFY OK", the
token the box server reads. While rola refuses a pass (its kernel rebuilt, its backward not yet built), every box
fails this gate with rola's own error: a fleet cannot train RoLA until rola can.
"""
import sys

import torch

from rola_bench.models import rola as cells


def main() -> int:
    if not torch.cuda.is_available():
        print("VERIFY FAILED: no CUDA device")
        return 1
    torch.manual_seed(0)
    layer = cells.layer(cells.cell("rola-d1-dense", 4), hidden_size=64, num_heads=1, d_v=64).cuda()
    x = torch.randn(2, 64, 64, device="cuda", requires_grad=True)
    try:
        out = layer(x)[0]
        out.float().pow(2).mean().backward()
    except Exception as ex:  # noqa: BLE001 -- the gate reports the refusal and fails
        print(f"VERIFY FAILED: {type(ex).__name__}: {ex}")
        return 1
    ok = bool(torch.isfinite(out).all()) and x.grad is not None and bool(torch.isfinite(x.grad).all())
    print(f"VERIFY {'OK' if ok else 'FAILED'} | RoLA forward and backward on {torch.cuda.get_device_name(0)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
