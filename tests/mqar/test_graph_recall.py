"""GATES for `zoology.data.graph_recall` — the graph-structured recall generator (2026-08-02).

The generator's whole value is that the sequence distribution has a KNOWN co-occurrence graph, so
these gates check the three things that claim rests on, and nothing decorative:

  §1 KNOWN DEMAND. For the three constructed families the realized graph's demand (its chromatic
     number, measured off the arrays the renderer actually consumes) equals the construction's
     formula: `complete(k) -> k`, `clustered(k, m) -> k` for every m, `overlap(w, ...) -> w` for
     every birth count and both lifetime profiles. For `reuse` there is no formula and the gate is
     that the reported INTERVAL is well-formed and that the cell carries no nominal claim.
  §2 COVERAGE. The model sees the UNION of the sampled patches, a subgraph of the target, so the
     stamp must MOVE with the sampling rather than echo the knob.
  §3 THE CONTRACT. Every written key is queried, after it was written, with the value it was written
     with, and nothing else in the sequence is labelled — which is what makes the evaluation land on
     REALIZED EDGES (pairs some sequence forced apart) rather than on pairs that never conflicted.
  §4 DETERMINISM. Same pattern as `test_mqar_determinism.py` / `test_cue_consistency.py`: every draw
     through `np.random.default_rng(seed)`, gated behaviourally AND by walking the module's parsed
     call graph for global-RNG reads.

Symbols: `k` = keys a sequence forces apart; `m` = keys per cluster / per birth slot; `w` = maximum
simultaneously live keys; `B` = birth slots per sequence; `Lg` = long-lived slots (the mixed cell);
`chi` = the graph's chromatic number = the demand; `|V|` = realized node count.

CPU only. A visible GPU is needed only because pytest collection imports Triton.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

BASE = {"vocab_size": 1024, "num_examples": 2000, "input_seq_len": 128}


def _episodes(**kw):
    from zoology.data.graph_recall import build_episodes
    return build_episodes(**{**BASE, "seed": 0, **kw})


def _stamp(**kw):
    from zoology.data.graph_recall import stamp
    return stamp(_episodes(**kw))


def _segment(**kw):
    from zoology.data.graph_recall import graph_recall
    return graph_recall(**{**BASE, "seed": 0, **kw})


# =============================================================== 1. demand is known by construction
@pytest.mark.parametrize("k", [4, 8, 16])
def test_complete_realizes_the_complete_graph_on_k_keys(k):
    s = _stamp(construction="complete", demand=k)
    assert (s["graph_nodes"], s["graph_edges"]) == (k, k * (k - 1) // 2)
    assert s["graph_demand_exact"] and s["graph_demand_lower"] == k == s["graph_demand_upper"]


@pytest.mark.parametrize("m", [1, 2, 4, 8])
def test_clustered_demand_is_the_cluster_count_regardless_of_cluster_size(m):
    """THE DECOUPLING. chi = k at every m while |V| = k*m and the edge count grows as m^2: the
    realized graph is the complete k-partite K_{m,...,m}. If this ever failed, `clustered` would be
    measuring node count rather than demand and the contrast against `complete(k)` would be void."""
    k = 8
    s = _stamp(construction="clustered", demand=k, keys_per_class=m)
    assert s["graph_nodes"] == k * m
    assert s["graph_edges"] == k * (k - 1) // 2 * m * m      # every cross-cluster pair realized
    assert s["graph_demand_exact"] and s["graph_demand_lower"] == k == s["graph_demand_upper"]


@pytest.mark.parametrize("w,births,m,long_lived", [
    (4, 8, 1, 0), (4, 16, 2, 0), (8, 16, 1, 0), (8, 24, 2, 0),
    (8, 16, 1, 2), (8, 16, 2, 4),                            # the MIXED-lifetime cells
])
def test_overlap_demand_is_the_liveness_window(w, births, m, long_lived):
    """chi = w = max simultaneous liveness, independent of how many lifetimes the sequence starts
    (B) and of the lifetime PROFILE (Lg long-lived slots + a churn window of width w - Lg)."""
    s = _stamp(construction="overlap", demand=w, births=births, keys_per_class=m,
               long_lived=long_lived)
    assert s["graph_nodes"] == births * m
    assert s["graph_demand_exact"] and s["graph_demand_lower"] == w == s["graph_demand_upper"]


def test_overlap_writes_more_keys_than_the_demand():
    """The point of the family: B keys pass through a window of width w << B, so the item count and
    the demand are different numbers. Without this the cell would just be `complete(B)`."""
    ep = _episodes(construction="overlap", demand=8, births=24, keys_per_class=1)
    assert ep.write_key.shape[1] == 24
    from zoology.data.graph_recall import stamp
    assert stamp(ep)["graph_demand_upper"] == 8


def test_reuse_reports_bounds_and_claims_no_nominal_demand():
    """`reuse` has no formula: its chi is emergent, is far above its keys-per-sequence knob, and is
    reported as an interval. `graph_demand_nominal == 0` is the explicit 'no claim' — stamping the
    knob there would let a cell be read at a demand it does not have."""
    s = _stamp(construction="reuse", demand=8, pool_size=128, zipf_a=1.5)
    assert s["graph_demand_nominal"] == 0
    assert 8 <= s["graph_demand_lower"] <= s["graph_demand_upper"] <= s["graph_nodes"] == 128
    assert s["graph_demand_lower"] > 8, "a Zipf-reuse pool whose chi equals k is not a reuse cell"


def test_demand_bounds_are_correct_on_hand_checkable_graphs():
    """The bound machinery itself, against graphs whose chi is known by inspection rather than by
    the generator that produced them ([[matching-the-naive-antipattern]])."""
    from zoology.data.graph_recall import demand_bounds
    # one clique of 5 -> chi = 5; two disjoint triangles -> chi = 3; a 4-cycle (as two groups) -> 2
    assert demand_bounds(np.array([[[10, 11, 12, 13, 14]]]))["upper"] == 5
    assert demand_bounds(np.array([[[1, 2, 3]], [[4, 5, 6]]]))["upper"] == 3
    four_cycle = np.array([[[1, 2, -1]], [[2, 3, -1]], [[3, 4, -1]], [[4, 1, -1]]])
    b = demand_bounds(four_cycle)
    assert (b["nodes"], b["edges"], b["lower"], b["upper"]) == (4, 4, 2, 2)


# =============================================================== 2. coverage, not the knob
def test_the_stamp_measures_the_realized_union_and_not_the_target():
    """An under-sampled `clustered` cell realizes a strict SUBGRAPH of K_{m,...,m}: the same knobs
    with 8 episodes and with 4000 realize different edge counts, and both segments carry their own
    number. This is the failure mode the stamp exists for — a cell reading its demand off the knob
    would report the target graph it never actually presented."""
    kw = {"construction": "clustered", "demand": 8, "keys_per_class": 8}
    small = _stamp(num_examples=8, **kw)
    full = _stamp(num_examples=4000, **kw)
    assert small["graph_edges"] < full["graph_edges"] == 8 * 7 // 2 * 64
    assert small["graph_nodes"] <= full["graph_nodes"] == 64


def test_reuse_demand_grows_with_the_sample_and_is_stamped_per_segment():
    kw = {"construction": "reuse", "demand": 8, "pool_size": 128, "zipf_a": 1.5}
    assert _stamp(num_examples=50, **kw)["graph_demand_upper"] < \
           _stamp(num_examples=4000, **kw)["graph_demand_upper"]


def test_every_stamp_field_reaches_the_segment_slices():
    """`DataSegment` carries `(inputs, labels, slices)` and the on-disk cache stores exactly those,
    so `slices` is the only cache-safe home for the stamp — and it is what `slice_keys` can report
    per-slice accuracy against."""
    from zoology.data.graph_recall import STAMP_FIELDS
    seg = _segment(construction="clustered", demand=8, keys_per_class=4)
    assert set(STAMP_FIELDS) <= set(seg.slices)
    assert seg.slices["graph_demand_upper"] == 8 and seg.slices["graph_nodes"] == 32


# =============================================================== 3. the retrieval contract
@pytest.mark.parametrize("kw", [
    {"construction": "complete", "demand": 8},
    {"construction": "clustered", "demand": 8, "keys_per_class": 4},
    {"construction": "overlap", "demand": 8, "births": 16, "keys_per_class": 2},
    {"construction": "overlap", "demand": 8, "births": 16, "keys_per_class": 2, "long_lived": 2},
    {"construction": "reuse", "demand": 8, "pool_size": 64, "zipf_a": 1.5},
])
def test_every_written_key_is_queried_exactly_once_and_after_its_write(kw):
    """Recall on a pair that never conflicted tests nothing. Querying EVERY written key is what puts
    the evaluation on realized edges: an edge exists because two keys were live together, and both
    are then asked for in that same sequence."""
    ep = _episodes(num_examples=200, **kw)
    assert ep.write_key.shape == ep.query_key.shape
    assert (np.sort(ep.write_key, 1) == np.sort(ep.query_key, 1)).all()
    w_ord, q_ord = np.argsort(ep.write_key, 1), np.argsort(ep.query_key, 1)   # align by key
    assert (np.take_along_axis(ep.query_slot, q_ord, 1) >
            np.take_along_axis(ep.write_slot, w_ord, 1)).all()
    # the queried value is the value that key was written with
    assert (np.take_along_axis(ep.query_value, q_ord, 1) ==
            np.take_along_axis(ep.write_value, w_ord, 1)).all()


@pytest.mark.parametrize("kw", [
    {"construction": "complete", "demand": 8},
    {"construction": "overlap", "demand": 8, "births": 16, "keys_per_class": 2, "long_lived": 2},
])
def test_labels_sit_on_query_keys_and_nowhere_else(kw):
    from zoology.data.graph_recall import build_episodes, graph_recall
    ep = build_episodes(**{**BASE, "num_examples": 64, "seed": 3, **kw})
    seg = graph_recall(**{**BASE, "num_examples": 64, "seed": 3, **kw})
    labelled = (seg.labels != -100).numpy()
    assert labelled.sum(1).tolist() == [ep.query_key.shape[1]] * 64
    rows = np.arange(64)[:, None]
    assert labelled[rows, 2 * ep.query_slot].all()
    assert (seg.inputs.numpy()[rows, 2 * ep.query_slot] == ep.query_key).all()
    assert (seg.labels.numpy()[rows, 2 * ep.query_slot] == ep.query_value).all()


def test_filler_never_draws_a_key_token():
    """`multiquery_ar` fills from the whole vocabulary, so a filler token can coincide with a live
    key and act as an uninstructed query. Excluding the (few dozen token) key pool costs nothing and
    removes the ambiguity — the same reason `cue_consistency` keeps WILDCARD out of its filler."""
    ep = _episodes(num_examples=64, construction="complete", demand=8)
    seg = _segment(num_examples=64, construction="complete", demand=8)
    inputs = seg.inputs.numpy()
    rows = np.arange(64)[:, None]
    structural = np.zeros_like(inputs, dtype=bool)
    for pos in (2 * ep.write_slot, 2 * ep.write_slot + 1, 2 * ep.query_slot):
        structural[rows, pos] = True
    assert not np.isin(inputs[~structural], ep.key_pool).any()


def test_the_mixed_lifetime_cell_writes_early_and_queries_at_the_end():
    """The cell that separates a write-mass clock from a time clock: `Lg` keys are written once at
    the very start and asked for at the very end, with nothing re-touching them in between, while
    the rest churn. Without this property the cell is just another uniform-lifetime cell."""
    ep = _episodes(num_examples=200, construction="overlap", demand=8, births=16, long_lived=3)
    long_keys = ep.write_key[:, :3]                        # the schedule writes them first, ...
    assert (ep.write_slot[:, :3] < ep.write_slot[:, 3:].min(1, keepdims=True)).all()
    # ... and asks for them last, after every churn event of the episode has passed
    assert (np.sort(ep.query_key[:, -3:], 1) == np.sort(long_keys, 1)).all()
    assert (ep.query_slot[:, -3:] > ep.query_slot[:, :-3].max(1, keepdims=True)).all()
    assert (ep.query_slot[:, -3:] > ep.write_slot.max(1, keepdims=True)).all()


@pytest.mark.parametrize("kw,reason", [
    ({"construction": "overlap", "demand": 8, "births": 4}, "births"),
    ({"construction": "overlap", "demand": 8, "long_lived": 8}, "long_lived"),
    ({"construction": "complete", "demand": 64, "input_seq_len": 64}, "event slots"),
    ({"construction": "reuse", "demand": 8, "pool_size": 4}, "pool_size"),
    ({"construction": "nonesuch", "demand": 8}, "construction"),
])
def test_unbuildable_cells_refuse_by_name(kw, reason):
    """Structural impossibilities raise HERE, by name, rather than asserting deep in the schedule —
    the house rule `make_law_data` follows for MQAR's `4*kv <= L`."""
    with pytest.raises(ValueError, match=reason):
        _episodes(**kw)


# =============================================================== 4. determinism
def test_determinism_per_seed():
    a, b, c = _segment(seed=7), _segment(seed=7), _segment(seed=8)
    assert (a.inputs == b.inputs).all() and (a.labels == b.labels).all()
    assert not (a.inputs == c.inputs).all()


def test_determinism_survives_unrelated_global_consumers():
    """Mirrors `test_mqar_determinism` — the bug it gates was a generator reading numpy's and
    torch's GLOBAL streams, so a same-seed segment could be perturbed by anything else in the
    process that touched either one first."""
    a = _segment(seed=7)
    np.random.seed(999)
    np.random.random(1000)
    torch.manual_seed(999)
    torch.randint(0, 1000, (10_000,))
    b = _segment(seed=7)
    assert (a.inputs == b.inputs).all() and (a.labels == b.labels).all()


def test_no_global_rng_reads_in_source():
    """AST form of the audit: no CALL in `graph_recall.py` may invoke the bare global `np.random.*`
    (other than `default_rng`, which is not global state) or `torch.rand*`. Walks the parsed call
    graph rather than grepping text, so a docstring naming the old bug cannot false-fail it."""
    import ast
    import inspect

    from zoology.data import graph_recall as mod

    banned = {("np", "random", "seed"), ("np", "random", "choice"), ("np", "random", "randint"),
              ("torch", "randint"), ("torch", "rand")}
    hits = []
    for node in ast.walk(ast.parse(inspect.getsource(mod))):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        parts, cur = [], node.func
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        path = tuple(reversed(parts))
        if path in banned or path[-2:] in {("random", "seed"), ("random", "choice"),
                                           ("random", "randint")}:
            hits.append(".".join(path))
    assert not hits, f"global-RNG call(s) found in graph_recall.py: {hits}"


# =============================================================== 4b. the contract (train/eval split)
def test_the_default_contract_reads_every_item_of_every_sequence():
    """TOTAL COVERAGE is the default on both sides: the per-sequence read set is a PERMUTATION of the
    live items. That is what makes write-time predictability harmless — every item is asked about —
    and it is what makes the eval threshold sharp rather than sampled."""
    for split in ("train", "eval"):
        ep = _episodes(num_examples=200, construction="complete", demand=16, split=split)
        assert (np.sort(ep.query_key, 1) == np.sort(ep.write_key, 1)).all()
    assert _stamp(num_examples=200, construction="complete", demand=16)["graph_contract_coverage"] \
        == 1.0


def test_the_sampled_contract_variant_is_flat_over_items_and_over_write_positions():
    """The SAMPLED variant (`queries_per_sequence` < the item count) is a control, not the default,
    and it is the one setting where write-time predictability could matter: a selection correlated
    with anything visible at write time would let the model alias the low-probability pairs and drop
    the effective demand below the solved chi. Uniform-without-replacement has no such correlation,
    and it is gated empirically on the arrays the renderer consumes rather than argued."""
    n, k, q = 4000, 16, 4
    ep = _episodes(num_examples=n, construction="complete", demand=k, queries_per_sequence=q)
    per_item = np.bincount(ep.query_key.ravel() - ep.key_pool.min(), minlength=k)
    expected = n * q / k
    assert np.abs(per_item / expected - 1).max() < 0.12, f"item bias: {per_item}"
    # ... and flat over WRITE POSITION, which is the correlation a positional heuristic would need.
    # `argsort(write_key)` maps a key's rank (= its class, since the pool ascends) to the position it
    # was written at, so gathering it at the queried classes gives each query's write position.
    pos_of_class = np.argsort(ep.write_key, axis=1)
    qpos = np.take_along_axis(pos_of_class, ep.query_key - ep.key_pool.min(), axis=1)
    per_pos = np.bincount(qpos.ravel(), minlength=k)
    assert np.abs(per_pos / expected - 1).max() < 0.12, f"write-position bias: {per_pos}"


def test_eval_queries_are_exhaustive_over_the_contract():
    """Eval must ask about EVERY covered item of every sequence, so every aliased pair is counted
    every time and the threshold is sharp against chi. Sampling eval targets would smear sub-chi
    degradation by sampling luck."""
    ep = _episodes(num_examples=200, construction="complete", demand=16,
                   queries_per_sequence=4, split="eval")
    assert ep.query_key.shape[1] == 16
    s = _stamp(num_examples=200, construction="complete", demand=16, queries_per_sequence=4,
               split="eval")
    assert s["graph_contract_coverage"] == 1.0 and s["graph_split"] == "eval"


def test_an_edge_is_enforced_node_wise_by_either_endpoint():
    """A point lookup demands separation NODE-wise: querying `u` demands `u` apart from everything
    live beside it. So covering half the items of `complete(k)` keeps every edge that TOUCHES a
    covered item — K_cov joined to an independent rest — and the solved demand is cov + 1, not cov.
    That asymmetry is the whole reason the contract graph is solved rather than assumed."""
    k = 16
    s = _stamp(num_examples=1000, construction="complete", demand=k, contract_fraction=0.5)
    cov = k // 2
    assert s["graph_demand_lower"] == s["graph_demand_upper"] == cov + 1
    assert s["graph_contract_edges"] == k * (k - 1) // 2 - cov * (cov - 1) // 2
    assert 0 < s["graph_contract_coverage"] < 1 and not s["graph_contract_full"]


def test_a_sampled_contract_does_not_reduce_the_demand_but_a_covered_one_does():
    """The two ways to 'ask fewer questions', side by side — the pair the grid's contract tier runs.
    Re-drawing targets per sequence still demands every pair apart in SOME sequence, so the union
    contract graph, and the solved chi, is the full one. Restricting WHICH items are ever asked about
    genuinely shrinks it. A cell that confused the two would report a threshold shift as a contract
    effect when it was only a query budget."""
    sampled = _stamp(num_examples=2000, construction="complete", demand=16, queries_per_sequence=4)
    covered = _stamp(num_examples=2000, construction="complete", demand=16, contract_fraction=0.5)
    assert sampled["graph_contract_coverage"] == 1.0 and sampled["graph_demand_upper"] == 16
    assert covered["graph_demand_upper"] == 9


def test_the_partial_contract_cell_is_not_gated_against_the_closed_form_but_is_announced(capsys):
    """Under a partial contract the stamped demand is chi of the demanded-apart subgraph, which is
    deliberately NOT the construction's formula — so the closed-form gate must stand down, and the
    segment must say so loudly rather than let a reader assume the nominal number."""
    seg = _segment(num_examples=200, construction="complete", demand=16, contract_fraction=0.5)
    assert seg.slices["graph_demand_upper"] == 9 and seg.slices["graph_demand_nominal"] == 16
    assert "GRAPH_RECALL_PARTIAL_CONTRACT" in capsys.readouterr().out


def test_the_contract_reduces_on_both_sides():
    """A reduced contract must restrict the TRAINING targets and the EXHAUSTIVE EVAL alike: an eval
    that asked about uncovered items would be measuring a contract the model was never trained to
    serve."""
    for split in ("train", "eval"):
        ep = _episodes(num_examples=100, construction="complete", demand=16, contract_fraction=0.25,
                       split=split)
        covered = {int(x) for x in ep.key_pool[:4]}
        assert set(np.unique(ep.query_key)) == covered


def test_temporal_separation_is_asserted_on_every_construction():
    """A read adjacent to its own write is served out of the residual stream, so the generator
    ASSERTS separation rather than hoping the layout provides it. Checked here as a property of the
    emitted arrays: every read trails the write it answers by at least `min_separation` event slots,
    and at most one item per episode (the last-written, which nothing can follow) is read with no
    other item's write in between."""
    for kw in ({"construction": "complete", "demand": 8},
               {"construction": "clustered", "demand": 8, "keys_per_class": 4},
               {"construction": "overlap", "demand": 8, "births": 16, "keys_per_class": 2},
               {"construction": "overlap", "demand": 8, "births": 16, "long_lived": 2},
               {"construction": "reuse", "demand": 8, "pool_size": 64, "zipf_a": 1.5}):
        for sep in (2, 4):
            ep = _episodes(num_examples=200, min_separation=sep, **kw)
            lut = np.full((len(ep.write_key), int(ep.write_key.max()) + 2), -1, dtype=np.int64)
            np.put_along_axis(lut, ep.write_key, ep.write_slot, axis=1)
            src = np.take_along_axis(lut, ep.query_target_key, axis=1)
            assert (ep.query_slot - src).min() >= sep, f"{kw} at min_separation={sep}"
            no_write_after = (src == ep.write_slot.max(1, keepdims=True)).sum(1)
            assert no_write_after.max() <= 1


def test_the_separation_assertion_actually_bites():
    """Gated by asking for a separation the layout cannot provide: the assertion must raise, not
    quietly emit a shorter gap."""
    with pytest.raises(ValueError, match="min_separation"):
        _episodes(construction="complete", demand=8, min_separation=0)
    with pytest.raises((AssertionError, ValueError)):
        # 60 event slots of a 64-slot grid cannot separate 8 writes from 8 reads by 60
        _episodes(construction="complete", demand=8, input_seq_len=128, min_separation=60)


def test_the_cue_permutation_is_frozen_per_cell_and_shared_by_train_and_eval():
    """pi must come from `cue_permutation_seed`, NEVER from the segment seed: a per-segment map would
    make the cell unlearnable by construction rather than by capacity, and the failure would look
    like a capacity result."""
    a = _episodes(num_examples=64, seed=1, construction="complete", demand=8, cue_permutation=True)
    b = _episodes(num_examples=64, seed=99, construction="complete", demand=8, cue_permutation=True,
                  split="eval")

    def cue_to_target(ep):
        return {int(c): int(t) for row_c, row_t in zip(ep.query_key, ep.query_target_key, strict=True)
                for c, t in zip(row_c, row_t, strict=True)}
    assert cue_to_target(a) == cue_to_target(b)
    assert any(c != t for c, t in cue_to_target(a).items()), "pi is the identity"
    # a different frozen seed is a different map
    c = _episodes(num_examples=64, seed=1, construction="complete", demand=8, cue_permutation=True,
                  cue_permutation_seed=7)
    assert cue_to_target(c) != cue_to_target(a)


def test_the_cue_permutation_leaves_the_storage_demand_untouched():
    """The point of the cell: same distribution, same chi, different addressing map. If chi moved,
    the cell would be confounding the map with the storage question it is meant to isolate."""
    plain = _stamp(num_examples=500, construction="complete", demand=16)
    permuted = _stamp(num_examples=500, construction="complete", demand=16, cue_permutation=True)
    for key in ("graph_nodes", "graph_edges", "graph_demand_upper", "graph_contract_coverage"):
        assert plain[key] == permuted[key], key
    assert permuted["graph_cue_permuted"] and not plain["graph_cue_permuted"]


def test_the_answer_under_a_permuted_cue_is_the_value_of_the_mapped_item():
    """The label must be `pi(u)`'s value, not `u`'s — the one thing that makes this a different
    contract rather than a relabelling of the same one."""
    ep = _episodes(num_examples=64, construction="complete", demand=8, cue_permutation=True)
    seg = _segment(num_examples=64, construction="complete", demand=8, cue_permutation=True)
    rows = np.arange(64)[:, None]
    lut = np.full((64, int(ep.write_key.max()) + 2), -1, dtype=np.int64)
    np.put_along_axis(lut, ep.write_key, ep.write_value, axis=1)
    assert (seg.labels.numpy()[rows, 2 * ep.query_slot]
            == np.take_along_axis(lut, ep.query_target_key, axis=1)).all()
    assert not (ep.query_key == ep.query_target_key).all()


def test_the_cue_permutation_is_refused_where_it_is_not_well_defined():
    with pytest.raises(ValueError, match="cue_permutation"):
        _episodes(construction="overlap", demand=8, births=16, cue_permutation=True)
    with pytest.raises(ValueError, match="cue_permutation"):
        _episodes(construction="reuse", demand=8, pool_size=64, cue_permutation=True)


# =============================================================== 5. the chi solver
@pytest.mark.parametrize("kw,chi,method", [
    ({"construction": "complete", "demand": 6}, 6, "disjoint_cliques"),
    ({"construction": "clustered", "demand": 6, "keys_per_class": 1}, 6, "disjoint_cliques"),
    ({"construction": "clustered", "demand": 6, "keys_per_class": 4}, 6, "complete_multipartite"),
    ({"construction": "overlap", "demand": 6, "births": 12, "keys_per_class": 1}, 6, "chordal"),
    ({"construction": "overlap", "demand": 6, "births": 12, "keys_per_class": 3}, 6, "chordal_quotient"),
    ({"construction": "overlap", "demand": 8, "births": 20, "keys_per_class": 2, "long_lived": 3}, 8,
     "chordal_quotient"),
])
def test_solver_agrees_with_the_closed_form_and_says_which_argument_it_used(kw, chi, method):
    """The solver must reach each construction's formula BY RECOGNIZING ITS STRUCTURE, not by a
    heuristic that happens to agree. `method` is asserted because "greedy got the right number" and
    "the graph is a complete multipartite graph, so k colors is optimal" are different claims, and
    only the second one survives a change of parameters."""
    from zoology.data.graph_chi import solve_chi
    from zoology.data.graph_recall import realized_graph
    _nodes, adj = realized_graph(_episodes(num_examples=500, **kw).groups)
    res = solve_chi(adj)
    assert (res["lower"], res["upper"], res["exact"], res["method"]) == (chi, chi, True, method)


def test_solver_is_exact_on_graphs_whose_chi_is_known_by_hand():
    """Independent of the generator entirely: the ladder's general rungs, on classical graphs.
    C5 needs 3 (odd hole), the Petersen graph needs 3, K_{3,3} needs 2."""
    from zoology.data.graph_chi import solve_chi

    def g(n, edges):
        a = np.zeros((n, n), dtype=bool)
        for u, v in edges:
            a[u, v] = a[v, u] = True
        return a
    c5 = solve_chi(g(5, [(i, (i + 1) % 5) for i in range(5)]))
    assert (c5["lower"], c5["upper"], c5["method"]) == (3, 3, "exact_branch_and_bound")
    petersen = [(i, (i + 1) % 5) for i in range(5)] + [(i, i + 5) for i in range(5)] + \
               [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    p = solve_chi(g(10, petersen))
    assert (p["lower"], p["upper"], p["exact"]) == (3, 3, True)
    k33 = solve_chi(g(6, [(i, 3 + j) for i in range(3) for j in range(3)]))
    assert (k33["lower"], k33["upper"], k33["method"]) == (2, 2, "complete_multipartite")


def test_both_certificates_are_verified_not_trusted():
    """A bound is only as good as its witness, so the solver re-checks the clique and the coloring it
    is about to return. Gated by handing the checkers a wrong witness."""
    from zoology.data.graph_chi import _check_clique, _check_coloring, solve_chi
    from zoology.data.graph_recall import realized_graph
    _nodes, adj = realized_graph(_episodes(num_examples=200, construction="clustered", demand=4,
                                           keys_per_class=3).groups)
    res = solve_chi(adj)
    assert _check_clique(adj, res["clique"]) == res["lower"]
    assert _check_coloring(adj, np.array(res["coloring"])) == res["upper"]
    with pytest.raises(AssertionError, match="not a clique"):
        _check_clique(adj, [0, 1, 2])                      # 0 and 1 share a cluster: non-adjacent
    bad = np.zeros_like(np.array(res["coloring"]))
    with pytest.raises(AssertionError, match="not proper"):
        _check_coloring(adj, bad)


def test_the_coverage_gate_catches_a_realized_demand_below_the_nominal_one():
    """THE COVERAGE CHECK, gated on a graph that under-realizes its construction. A truncated sample
    — here, one partial live window instead of the full schedule — realizes a smaller graph than
    `overlap(w=8)` claims, and `verify_construction` must refuse it by name rather than let a cell
    ship with an x-coordinate of 8 that the model was never shown."""
    from zoology.data.graph_chi import DemandMismatch, verify_construction
    from zoology.data.graph_recall import demand_bounds
    ep = _episodes(num_examples=200, construction="overlap", demand=8, births=16)
    full = demand_bounds(ep.groups)
    verify_construction("overlap", {"demand": 8}, full)              # the real sample passes
    starved = demand_bounds(ep.groups[:2, :1, :3])                    # a partial window only
    assert starved["upper"] < 8
    with pytest.raises(DemandMismatch, match="coverage shortfall"):
        verify_construction("overlap", {"demand": 8}, starved)


def test_the_closed_form_demand_is_invariant_to_the_sample_size():
    """A property of the CONSTRUCTIONS, worth pinning because it is why the coverage gate can be
    strict: their key pools are structured (per-cluster / per-birth-slot disjoint sub-pools), so the
    realized chi is the nominal one at any number of episodes and under-sampling shows up in the node
    and edge counts instead. An unstructured pool — `multiquery_ar`'s free draw from vocab/2 — would
    not have this property, which is exactly why this generator does not use one."""
    kw = {"construction": "clustered", "demand": 6, "keys_per_class": 8}
    assert _stamp(num_examples=2, **kw)["graph_demand_upper"] == 6
    assert _stamp(num_examples=2000, **kw)["graph_demand_upper"] == 6
    assert _stamp(num_examples=2, **kw)["graph_edges"] < _stamp(num_examples=2000, **kw)["graph_edges"]


def test_reuse_is_solved_exactly_where_the_budget_allows_and_bounded_otherwise():
    from zoology.data.graph_chi import solve_chi
    from zoology.data.graph_recall import realized_graph
    _nodes, adj = realized_graph(_episodes(num_examples=2000, construction="reuse", demand=8,
                                           pool_size=96, zipf_a=1.5).groups)
    res = solve_chi(adj)
    assert res["method"] in ("exact_branch_and_bound", "certified_bounds")
    assert res["lower"] <= res["upper"]
    # whatever the rung, the interval is certified on both sides
    from zoology.data.graph_chi import _check_clique, _check_coloring
    assert _check_clique(adj, res["clique"]) == res["lower"]
    assert _check_coloring(adj, np.array(res["coloring"])) == res["upper"]


def test_predicted_N_is_stamped_for_every_construction():
    """The comparison the grid's headline statistic is built on (`rola_bench.mqar.analysis.graph`): the cell
    carries the demand the theory predicts it needs, as its own field."""
    assert _stamp(construction="complete", demand=8)["graph_predicted_N"] == "8"
    assert _stamp(construction="overlap", demand=8, births=16)["graph_predicted_N"] == "8"
    reuse = _stamp(num_examples=500, construction="reuse", demand=8, pool_size=128, zipf_a=1.5)
    lo, hi = reuse["graph_demand_lower"], reuse["graph_demand_upper"]
    assert reuse["graph_predicted_N"] == (str(lo) if lo == hi else f"{lo}-{hi}")


# =============================================================== 6. the grid around the generator
def _spec():
    import os

    from rola_bench.mqar.build_configs import load_spec
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    here = os.path.join(repo, "rola_bench", "mqar", "experiments")
    return load_spec(os.path.join(here, "graph_grid.yaml"))


#: the header's sizes net of the rungs a D=2 wiring has no uniform spelling at (N = 32 and N = 128: squares only)
@pytest.mark.parametrize("tier,cells", [("decoupling", 36), ("resize", 28), ("overlap", 32),
                                        ("decay", 32), ("reuse", 20), ("contract", 36),
                                        ("cue_map", 32), ("attn_reference", 10),
                                        ("attn_capacity", 28)])
def test_every_tier_is_the_size_its_header_states(tier, cells):
    """The header's arithmetic is a schedule commitment (each tier is one overnight unit on the local
    card). If a tier silently doubles, the night it was planned for is not the night it takes."""
    import os

    from rola_bench.mqar.build_configs import build_configs
    old = os.environ.get("GRID_TIERS")
    os.environ["GRID_TIERS"] = tier
    try:
        configs, _ = build_configs(_spec())
    finally:
        os.environ.pop("GRID_TIERS") if old is None else os.environ.update(GRID_TIERS=old)
    assert len(configs) == cells


def test_the_slate_is_the_three_wirings_plus_the_control_plus_an_attention_reference():
    """The fixed-state class baselines (rla / gla / gdn) are established on the standard law grid and
    no claim in this file is about them; a stray `baselines` arm would silently triple a tier. The
    attention arm is admitted in EXACTLY ONE tier, as the instrument floor — if it ever appeared on a
    capacity tier it would be read as a contender on the N ladder, which it cannot be."""
    from rola_bench.models.rola import CANONICAL_WIRINGS
    for tier, arms in _spec()["tiers"].items():
        kinds = {a["build"] for a in arms}
        if tier.startswith("attn"):
            assert kinds == {"attn"}
            continue
        assert kinds == {"routed"}, f"{tier} carries a non-routed arm"
        assert {a["instance"] for a in arms} == set(CANONICAL_WIRINGS), f"{tier} is not the slate"


def test_the_attention_reference_covers_every_family_and_is_never_on_the_n_ladder():
    """One capacity-adequate cell per FAMILY (construction), so no family's routed failure can be
    read as a capacity result without a solvability check behind it. And no `ncs`: attention has no
    rung on the state axis."""
    spec = _spec()
    conds = {c["tag"]: c for c in spec["conditions"]}
    arms = spec["tiers"]["attn_reference"]
    covered = {conds[t]["construction"] for a in arms for t in a["conditions"]}
    assert covered == {"complete", "clustered", "overlap", "reuse"}
    assert all("ncs" not in a for a in arms)
    # ... and every attention cell's L matches the length its condition's routed cells train at
    routed_len = {}
    for tier, tarms in spec["tiers"].items():
        if tier == "attn_reference":
            continue
        for a in tarms:
            for t in a["conditions"]:
                routed_len[t] = a.get("seq_lens", spec["seq_lens"])[0]
    for a in arms:
        for t in a["conditions"]:
            assert a.get("seq_lens", spec["seq_lens"])[0] == routed_len[t], (
                f"attention runs {t} at a different L than the routed arms do; the floor would not "
                "be the floor of the cells it is supposed to calibrate")


def test_every_run_id_parses_for_the_analysis():
    """The analysis keys its rows off the run_id, so a template change that the parser does not know
    about would silently produce an empty headline table rather than an error."""
    import os

    from rola_bench.mqar.analysis.graph import parse_run_id
    from rola_bench.mqar.build_configs import build_configs
    os.environ["GRID_TIERS"] = "all"
    configs, _ = build_configs(_spec())
    conds = {c["tag"] for c in _spec()["conditions"]}
    for cfg in configs:
        got = parse_run_id(cfg.run_id)
        assert got is not None, f"analysis cannot parse {cfg.run_id}"
        assert got[4] in conds, f"{cfg.run_id} parsed to unknown condition {got[4]!r}"


def test_the_attention_capacity_tier_sweeps_d_qk_over_three_distinct_demands():
    """A slope needs three points. The d_qk ladder must also be a divisor ladder of d_model, since
    zoology's MHA derives head_dim = d_model // num_heads."""
    spec = _spec()
    arms = spec["tiers"]["attn_capacity"]
    d_model = spec["backbone"]["d_model"]
    demands = set()
    for a in arms:
        assert a["head_dims"], "the capacity tier must sweep d_qk"
        for d in a["head_dims"]:
            assert d_model % d == 0, f"d_qk={d} does not divide d_model={d_model}"
        for t in a["conditions"]:
            demands.add(t)
    # the three demand levels the fit needs: chi = 16 (complete-k16 / clustered), 33 (the contract
    # cell) and 64 (complete-k64)
    assert {"complete-k16", "complete-k64", "complete-k64-contract50"} <= demands


def test_the_N_ladders_bracket_the_demand_they_are_pointed_at():
    """A tier whose N ladder sits entirely above (or below) its condition's chi cannot see a collapse
    threshold at all, which would make the headline statistic unmeasurable for that cell."""
    spec = _spec()
    conds = {c["tag"]: c for c in spec["conditions"]}
    for tier, arms in spec["tiers"].items():
        for arm in arms:
            ncs = arm.get("ncs", spec["ncs"])
            for tag in arm["conditions"]:
                demand = conds[tag].get("demand")
                if conds[tag]["construction"] == "reuse":
                    continue                     # its chi is emergent (~125); checked by measurement
                if conds[tag].get("contract_fraction", 1.0) < 1.0:
                    # the demanded-apart graph is K_cov joined to an independent rest, so its chi is
                    # cov + 1, not `demand` — the ladder is checked against the SOLVED number
                    demand = int(round(conds[tag]["contract_fraction"] * demand)) + 1
                assert min(ncs) <= demand, f"{tier}/{tag}: ladder {ncs} starts above chi={demand}"
                assert max(ncs) >= demand, f"{tier}/{tag}: ladder {ncs} ends below chi={demand}"


def test_the_grid_stamps_the_realized_demand_as_a_slice_key():
    from zoology.data.graph_recall import STAMP_FIELDS
    keys = _spec()["train"]["slice_keys"]
    assert "graph_demand_upper" in keys and "graph_construction" in keys
    assert set(keys) <= set(STAMP_FIELDS) | {"num_kv_pairs", "input_seq_len"}


def test_gen_version_is_in_the_cache_key():
    """`DataSegment.from_config` hashes `config.model_dump()`; a plain (non-private) `gen_version`
    field therefore invalidates stale on-disk caches by key mismatch when the draw sequence changes,
    without deleting anyone's files. Same mechanism as `MQARConfig.gen_version`."""
    from zoology.data.graph_recall import GRAPH_RECALL_GEN_VERSION, GraphRecallConfig
    cfg = GraphRecallConfig(vocab_size=1024, num_examples=32, input_seq_len=128, demand=4)
    assert cfg.model_dump()["gen_version"] == GRAPH_RECALL_GEN_VERSION
    seg = cfg.build(seed=5)
    assert seg.inputs.shape == (32, 128) and seg.inputs.dtype == torch.int64
    assert seg.labels.shape == (32, 128) and seg.labels.dtype == torch.int64
