"""ROLA-BENCH'S DATA PROVIDERS: the cells of libraries other than rola (`rola_devtools.cells`).

A cell names one of these and its parameters in `registry.json`; the provider returns a description, and the runner the
point sends it to builds the tensors in its own worker.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QKV:
    """Attention's inputs: `batch` sequences of `tokens` tokens, `heads` heads of width `dv`, in `dtype`, causal or not;
    drawn from a device generator seeded from nothing but the cell (seed 0)."""

    name: str
    tokens: int
    dv: int
    heads: int = 1
    batch: int = 1
    dtype: str = "bfloat16"
    causal: bool = True


def qkv(name: str, **params) -> QKV:
    return QKV(name, **params)
