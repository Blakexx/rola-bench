"""The LM smoke: every arm's HF model built on the GPU at a small backbone, one training step each.

    python -m rola_bench.lm.smoke --list     # the arms
    python -m rola_bench.lm.smoke            # build and step them

Real runs are `rola_bench.lm.run` over `experiments/*.yaml`. What rola refuses (training before its native backward),
the smoke reports as a failure with rola's own error.
"""
from __future__ import annotations

import argparse
import sys

#: the small backbone the smoke builds every arm at: the registry's geometry, two layers wide 256
SMOKE_ENV = {"LM_HIDDEN": "256", "LM_LAYERS": "2", "LM_MAX_POS": "128", "LM_ROLA_NH": "4", "LM_ROLA_DV": "64",
             "LM_GDN_NH": "4", "LM_GDN_HEAD_DIM": "64", "LM_GLA_NH": "4", "LM_ATTN_HEADS": "4"}


def run() -> bool:
    import torch

    from rola_bench.lm import arms

    if not torch.cuda.is_available():
        print("SMOKE FAILED: no CUDA device")
        return False
    g = arms.geometry(SMOKE_ENV)
    ok = True
    for arm in arms.ARMS:
        try:
            model = arms.AutoModelForCausalLM.from_config(arms.arm_config(arm, vocab_size=256, g=g)).cuda()
            tokens = torch.randint(0, 256, (2, 64), device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(input_ids=tokens, labels=tokens).loss
            loss.backward()
            print(f"OK   {arm:36s} loss={loss.item():.3f}")
        except Exception as ex:  # noqa: BLE001 -- each arm's failure is reported and the rest still run
            ok = False
            print(f"FAIL {arm:36s} {type(ex).__name__}: {str(ex)[:160]}")
    print("SMOKE " + ("OK" if ok else "FAILED"))
    return ok


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the arms and exit")
    a = ap.parse_args(argv)
    if a.list:
        from rola_bench.lm import arms

        print("\n".join(arms.ARMS))
        return 0
    return 0 if run() else 1


if __name__ == "__main__":
    sys.exit(main())
