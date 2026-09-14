"""GATES for `zoology.data.multiquery_ar`'s seed-determinism fix (pre-campaign, 2026-08-02).

Companion to `test_cue_consistency.py`'s "determinism and the tokenization" section (§4), whose
`cue_consistency` generator already used a per-call `np.random.default_rng(seed)` stream. Auditing
`multiquery_ar.py` against that pattern (the 2026-08-02 cue-consistency record, "DELIBERATE DEPARTURES") found it
still read TWO global RNGs: `np.random.seed(seed)` followed by
`np.random.choice`/`np.apply_along_axis` (global numpy), and an unseeded `torch.randint(...)` for the
`random_non_queries` filler (global torch, not even keyed off `seed` at all). Both are fixed the same
way `cue_consistency` was built: every draw now comes from one `np.random.default_rng(seed)` local to
the call. See `zoology/data/multiquery_ar.py`'s `multiquery_ar` docstring, "DETERMINISM", for the
full audit.

This is a PRE-CAMPAIGN fix: no cells trained on `multiquery_ar` (the `ext` preset — `canonical.yaml`,
`dmodel64.yaml`, `router_bias.yaml`) have been published, so the fact that a given `seed` now produces
different bytes than the old code is intentional and accepted here, and is not to be repeated once
cells are published. `data/utils.py`'s on-disk cache key hashes `MQARConfig.model_dump()`, which now
includes `gen_version` (bumped alongside the fix), so stale pre-fix caches are invalidated by key
mismatch rather than by deleting anyone's files.

CPU only. A visible GPU is needed only because pytest collection imports Triton.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

BASE = {"vocab_size": 200, "num_examples": 64, "input_seq_len": 48, "num_kv_pairs": 4}


def _gen(**kw):
    from zoology.data.multiquery_ar import multiquery_ar
    return multiquery_ar(**{**BASE, **kw})


def test_determinism_per_seed():
    a = _gen(seed=7)
    b = _gen(seed=7)
    c = _gen(seed=8)
    assert (a.inputs == b.inputs).all() and (a.labels == b.labels).all()
    assert not (a.inputs == c.inputs).all()


def test_determinism_survives_an_unrelated_global_numpy_consumer():
    """The generator must not read `np.random`'s global stream. Mirrors
    `test_cue_consistency.test_determinism_survives_an_unrelated_global_numpy_consumer`."""
    a = _gen(seed=7)
    np.random.seed(999)
    np.random.random(1000)
    b = _gen(seed=7)
    assert (a.inputs == b.inputs).all() and (a.labels == b.labels).all()


def test_determinism_survives_an_unrelated_global_torch_consumer():
    """The `random_non_queries` filler used `torch.randint(...)` against torch's GLOBAL RNG with no
    seed of its own — the specific bug this fix closes (it depended on whatever else in the process
    had touched torch's RNG first, not just on being unseeded). Gated directly: perturbing torch's
    global generator between two same-seed calls must not change either one's output."""
    a = _gen(seed=7, random_non_queries=True)
    torch.manual_seed(999)
    torch.randint(0, 1000, (10_000,))
    b = _gen(seed=7, random_non_queries=True)
    assert (a.inputs == b.inputs).all() and (a.labels == b.labels).all()


def test_no_global_rng_reads_in_source():
    """AST-verifiable form of the audit: no CALL in `multiquery_ar.py` may invoke the bare global
    `np.random.*` (other than `default_rng`, which is fine — it is not global state) or `torch.rand*`.
    Walks the parsed call graph rather than grepping raw text, so mentions inside comments/docstrings
    (e.g. this module's own "DETERMINISM" note, which names the old bug by its call) don't false-fail
    the gate."""
    import ast
    import inspect

    from zoology.data import multiquery_ar as mod

    tree = ast.parse(inspect.getsource(mod))
    banned = {
        ("np", "random", "seed"), ("np", "random", "choice"), ("np", "random", "randint"),
        ("torch", "randint"), ("torch", "rand"),
    }
    hits = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        parts = []
        cur = node.func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        path = tuple(reversed(parts))
        if path in banned or path[-2:] in {("random", "seed"), ("random", "choice"),
                                            ("random", "randint")}:
            hits.append(".".join(path))
    assert not hits, f"global-RNG call(s) found in multiquery_ar.py: {hits}"


@pytest.mark.parametrize("random_non_queries", [True, False])
def test_shapes_and_labels_unaffected_by_the_rng_source_change(random_non_queries):
    """The fix changes WHICH numbers are drawn, never the generator's structural contract: shapes,
    dtypes, and which positions carry a label."""
    seg = _gen(seed=3, random_non_queries=random_non_queries)
    assert seg.inputs.shape == (BASE["num_examples"], BASE["input_seq_len"])
    assert seg.labels.shape == (BASE["num_examples"], BASE["input_seq_len"])
    assert seg.inputs.dtype == torch.int64 and seg.labels.dtype == torch.int64
    n_labelled = (seg.labels != -100).sum(dim=1)
    assert (n_labelled == BASE["num_kv_pairs"]).all()
