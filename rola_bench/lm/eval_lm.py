"""Evaluate a RoLA (or any AutoModel) checkpoint with lm-evaluation-harness (paper-grade eval).
Every arm is an fla HF model, registered with the auto-classes on import, so lm_eval's HFLM loads it directly.

  python eval_lm.py --pretrained <run>                       # GENERAL zero-shot + held-out ppl
  python eval_lm.py --pretrained <run> --recall              # + SWDE/FDA/SQUAD recall suite
  python eval_lm.py --pretrained <run> --recall --max_length 2048   # recall docs can exceed 1024

GENERAL suite mirrors the GLA/GDN/Mamba Table-2 norm: lambada_openai (acc + PERPLEXITY), hellaswag,
piqa, arc_easy, arc_challenge, winogrande, openbookqa, + wikitext (held-out PPL — a corpus-agnostic
standard, NOT the training corpus's own val split). RECALL suite = the Based/Zoology info-extraction
tasks (swde, fda, squad_completion).
"""
import argparse
import json

import fla.models  # noqa: F401 -- registers fla's architectures so HFLM can load them
from lm_eval import simple_evaluate
from lm_eval.models.huggingface import HFLM

# GLA Table-2 / GDN zero-shot norm + held-out perplexity standards (wikitext, lambada).
GENERAL = "lambada_openai,hellaswag,piqa,arc_easy,arc_challenge,winogrande,openbookqa,wikitext"
RECALL = "swde,fda,squad_completion"   # Based/Zoology associative-recall (long-context info extraction)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained", required=True)
    p.add_argument("--tasks", default=GENERAL)
    p.add_argument("--recall", action="store_true",
                   help="append the SWDE/FDA/SQUAD recall suite (consider --max_length 2048: recall docs are long)")
    # NOT "auto": on WSL2 the accelerate batch-finder probes a huge batch, OOMs at the
    # lm_head (vocab 50257), and the OOM surfaces as cudaErrorUnknown — a string the finder
    # doesn't recognize as OOM, so it dies instead of halving. Use an explicit batch.
    p.add_argument("--batch_size", default="8")
    # Match the training context (block_size=1024). The model's config max_position_embeddings
    # is 2048, but it trained at 1024; lm-eval would otherwise roll at 2048 and the
    # linear/recurrent head degrades out-of-distribution past its trained context.
    p.add_argument("--max_length", type=int, default=2048)   # = train context; recall docs fit
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    bs = int(a.batch_size) if str(a.batch_size).isdigit() else a.batch_size
    tasks = a.tasks.split(",") + (RECALL.split(",") if a.recall else [])
    lm = HFLM(pretrained=a.pretrained, backend="causal", batch_size=bs, max_length=a.max_length)
    res = simple_evaluate(model=lm, tasks=tasks, limit=a.limit)
    rows = res["results"]
    print("=== lm-eval results ===")
    for task, m in rows.items():
        print(f"  {task}: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()
                                         if isinstance(v, (int, float))))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rows, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
