"""Post-hoc token-mixing-matrix RANK evaluator for trained MQAR checkpoints.

Public API:
  rank_stats(attn)                       — numerical / stable / effective rank of [B,H,L,L]
  evaluate_rank(ckpt, builder, extractor, batch) — generic checkpoint-driven driver
  make_mqar_eval(...)                    — real MQAR eval batch (per kv-slice) for the rank probe
  build_*_extractor()                    — each baseline family's final matrix (attention, GDN, GLA, Based, linear
                                           attention); RoLA's is carded

Metric definitions are mirrored verbatim from rola.py:311-401 (see evaluator.py docstring for the exact line map).
"""
from .evaluator import (
    DEFAULT_TOLS,
    evaluate_rank,
    make_mqar_eval,
    rank_stats,
)

__all__ = [
    "DEFAULT_TOLS",
    "evaluate_rank",
    "make_mqar_eval",
    "rank_stats",
]
