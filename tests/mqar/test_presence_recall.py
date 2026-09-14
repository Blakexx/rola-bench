"""GATES for `zoology.data.presence_recall` — v1's theory-confirmation instrument (2026-08-03).

The task: a frozen map `pi` over the key vocabulary with `pi(x) != x`, a sequence of key tokens, and
one bit at every key position — has `pi(x_t)` appeared STRICTLY BEFORE t? Everything the instrument
claims rests on five properties, and these gates check those and nothing decorative:

  §1 THE LABEL IS THE TASK. Every emitted label equals a brute-force prefix membership check written
     independently of the emission machinery ([[matching-the-naive-antipattern]] — a second copy of
     the fast path would only ever prove itself self-consistent), and the labels land on the key
     positions, in the reserved token pair, with the first event of each sequence UNLABELLED.
  §2 pi IS FROZEN AND IS A DERANGEMENT. `pi(x) != x` everywhere, drawn from `pi_seed` and never from
     the segment seed, so a cell's train and eval segments share one map; a different `pi_seed` is a
     different map, and the map is stamped.
  §3 THE CLASS BALANCE IS PLANTED, not hoped for: ~50/50 overall AND flat in position (the stamp's
     `presence_position_bias`), which is what makes the 0.5 chance line sharp and what stops a
     position-only model from scoring above it.
  §4 THE DEMAND IS chi AND IT IS SOLVED. The realized union graph is the complete graph on the pool,
     solved by `graph_chi` on the `disjoint_cliques` rung and gated against the closed form; an
     under-sampled segment is REFUSED rather than shipped with the wrong x-coordinate.
  §5 DETERMINISM. Same pattern as `test_graph_recall.py` / `test_mqar_determinism.py`: every draw
     through `np.random.default_rng(seed)`, gated behaviourally AND by walking the module's parsed
     call graph for global-RNG reads.

Symbols: `pi` = the frozen read map; `K` = `demand` = key-pool size = this task's chi; `s` =
`keys_per_sequence`; `L` = `input_seq_len` in tokens; `chi` = the graph's chromatic number.

CPU only. A visible GPU is needed only because pytest collection imports Triton.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

BASE = {"vocab_size": 1024, "num_examples": 2000, "input_seq_len": 64}


def _episodes(**kw):
    from zoology.data.presence_recall import build_episodes
    return build_episodes(**{**BASE, "seed": 0, **kw})


def _stamp(**kw):
    from zoology.data.presence_recall import stamp
    return stamp(_episodes(**kw))


def _segment(**kw):
    from zoology.data.presence_recall import presence_recall
    return presence_recall(**{**BASE, "seed": 0, **kw})


# =============================================================== 1. the label IS the task
@pytest.mark.parametrize("kw", [
    {"demand": 8},
    {"demand": 16, "keys_per_sequence": 8},
    {"demand": 16, "keys_per_sequence": 12},
    {"demand": 32, "keys_per_sequence": 16, "input_seq_len": 128},
])
def test_labels_are_a_prefix_membership_check_on_pi(kw):
    """The whole task, checked one position at a time against a from-the-definition implementation.
    The fast path plants a class and then draws a key to realize it; this asserts that what it
    realized is what the definition says, on the arrays the renderer consumes."""
    from zoology.data.presence_recall import brute_force_labels
    ep = _episodes(num_examples=300, **kw)
    assert (brute_force_labels(ep.key_token, ep.key_pool, ep.pi) == ep.label).all()


def test_a_key_is_emitted_at_most_once_per_sequence():
    """A repeat is MONOTONE in the presence relation (once `pi(x)` has appeared it has appeared for
    good), so a second occurrence carries no new question and would only dilute the metric."""
    ep = _episodes(num_examples=500, demand=16)
    for row in ep.key_token:
        assert len(set(row.tolist())) == len(row)


def test_the_labels_land_on_the_key_positions_in_the_reserved_pair():
    from zoology.data.presence_recall import label_tokens
    n = 128
    ep = _episodes(num_examples=n, seed=3, demand=16)
    seg = _segment(num_examples=n, seed=3, demand=16)
    present, absent = label_tokens(BASE["vocab_size"])
    rows = np.arange(n)[:, None]
    inputs, labels = seg.inputs.numpy(), seg.labels.numpy()
    assert (inputs[rows, ep.event_pos] == ep.key_token).all()
    want = np.where(ep.label[:, 1:], present, absent)
    assert (labels[rows, ep.event_pos[:, 1:]] == want).all()
    assert set(np.unique(labels).tolist()) <= {-100, present, absent}


def test_the_first_event_of_every_sequence_is_unlabelled():
    """Its prefix is empty, so its answer is the constant ABSENT: scoring it would be a free 1/s of
    accuracy for no addressing AND would pull the realized balance off the 0.5 the metric is read
    against. Exactly `s - 1` positions per sequence carry a label."""
    n, s = 128, 8
    ep = _episodes(num_examples=n, seed=5, demand=16, keys_per_sequence=s)
    seg = _segment(num_examples=n, seed=5, demand=16, keys_per_sequence=s)
    labels = seg.labels.numpy()
    rows = np.arange(n)[:, None]
    assert (labels != -100).sum(1).tolist() == [s - 1] * n
    assert (labels[rows, ep.event_pos[:, :1]] == -100).all()
    assert not ep.label[:, 0].any(), "the first event must be structurally ABSENT"


def test_the_label_tokens_never_appear_in_the_input_and_filler_never_draws_a_key():
    """Two ambiguities removed at once: a filler that coincided with a live key would be an
    uninstructed event silently rewriting the presence set, and an input token that IS the answer
    would be a copy channel."""
    from zoology.data.presence_recall import label_tokens
    n = 128
    ep = _episodes(num_examples=n, seed=1, demand=16)
    seg = _segment(num_examples=n, seed=1, demand=16)
    inputs = seg.inputs.numpy()
    present, absent = label_tokens(BASE["vocab_size"])
    assert not np.isin(inputs, [present, absent]).any()
    structural = np.zeros_like(inputs, dtype=bool)
    structural[np.arange(n)[:, None], ep.event_pos] = True
    assert not np.isin(inputs[~structural], ep.key_pool).any()


@pytest.mark.parametrize("sep", [1, 2, 4, 8])
def test_every_pair_of_key_events_is_at_least_min_separation_apart(sep):
    """A read whose target sits in the immediately preceding tokens is served by the backbone's
    short convolution (kernel 3) and measures nothing about the state. The gap is enforced between
    CONSECUTIVE events, which makes the statement true of every pair."""
    ep = _episodes(num_examples=500, demand=16, min_separation=sep)
    gaps = np.diff(ep.event_pos, axis=1)
    assert gaps.min() >= sep
    assert ep.event_pos.max() < BASE["input_seq_len"]


# =============================================================== 2. pi is frozen and deranged
def test_pi_is_a_derangement_at_every_pool_size():
    from zoology.data.presence_recall import frozen_pi
    for k in (2, 3, 8, 16, 64):
        pi = frozen_pi(k, 12345)
        assert sorted(pi.tolist()) == list(range(k)), "pi must be a bijection"
        assert (pi != np.arange(k)).all(), "pi(x) == x has no reading in this task"


def test_pi_is_frozen_per_cell_and_shared_by_train_and_eval():
    """pi must come from `pi_seed`, NEVER from the segment seed: a per-segment map would make the
    cell unlearnable by construction rather than by capacity, and the failure would read as a
    capacity result."""
    a = _episodes(num_examples=64, seed=1, demand=16)
    b = _episodes(num_examples=64, seed=99, demand=16, split="eval")
    c = _episodes(num_examples=64, seed=1, demand=16, pi_seed=7)
    assert (a.pi == b.pi).all()
    assert not (a.pi == c.pi).all(), "a different pi_seed must be a different map"
    assert a.meta["pi_hash"] == b.meta["pi_hash"] != c.meta["pi_hash"]


def test_the_frozen_map_is_stamped():
    """The map is a property of the CELL, so it travels with the segment rather than living only in
    whoever's memory drew it."""
    s = _stamp(num_examples=200, demand=16)
    assert s["presence_pi_derangement"] is True
    assert len(s["presence_pi_hash"]) == 12
    assert s["presence_pi_hash"] == _stamp(num_examples=64, demand=16)["presence_pi_hash"]


# =============================================================== 3. the planted class balance
@pytest.mark.parametrize("k,s", [(16, 8), (32, 16), (64, 32), (16, 12)])
def test_the_realized_balance_is_planted_at_50_50_and_is_stamped(k, s):
    """The 0.5 chance line is only sharp if the realized balance IS 0.5. Controlled by CONSTRUCTION
    (the class is planted and the key is drawn to realize it), never by rejection sampling — and
    MEASURED into the stamp, because a construction that silently drifted would move the line every
    number in the grid is read against."""
    st = _stamp(num_examples=2000, demand=k, keys_per_sequence=s, input_seq_len=256)
    assert abs(st["presence_positive_rate"] - 0.5) < 0.03
    assert st["presence_chance"] == 0.5
    assert st["presence_keys_per_sequence"] == s


@pytest.mark.parametrize("k,s", [(16, 8), (64, 32)])
def test_the_balance_is_flat_in_POSITION_which_is_the_confound_that_matters(k, s):
    """THE LOAD-BEARING ONE. If the label rate rose with position — which it does under the naive
    'draw keys and read off the labels' construction, because a fuller prefix is likelier to contain
    `pi(x)` — a model that only knows WHERE it is would score far above 0.5 without addressing
    anything, and the whole grid would be measuring that. The planted draw removes it and the stamp
    measures what is left."""
    ep = _episodes(num_examples=4000, demand=k, keys_per_sequence=s, input_seq_len=256)
    from zoology.data.presence_recall import stamp
    per_pos = ep.label[:, 1:].mean(axis=0)
    assert np.abs(per_pos - 0.5).max() < 0.05, f"positional prior: {per_pos.round(3)}"
    assert stamp(ep)["presence_position_bias"] < 0.05
    # ... and the naive construction really does have the prior this one removes, so the gate is
    # not vacuous: emitting a uniformly random subset in random order gives a rate that RISES.
    rng = np.random.default_rng(0)
    naive = np.argsort(rng.random((4000, k)), axis=1)[:, :s]
    lab = np.zeros((4000, s), dtype=bool)
    for r in range(4000):
        seen = set()
        for t, x in enumerate(naive[r]):
            lab[r, t] = int(ep.pi[x]) in seen
            seen.add(int(x))
    assert lab[:, -1].mean() - lab[:, 1].mean() > 0.2, "the naive draw should carry the prior"


def test_the_planted_rate_is_honoured_when_it_is_not_a_half():
    """`positive_rate` is a knob so the balance can be moved deliberately; the default is the only
    one the grid uses, and a cell that moved it would be reading against a different chance line."""
    st = _stamp(num_examples=2000, demand=32, keys_per_sequence=16, positive_rate=0.3,
                input_seq_len=128)
    assert abs(st["presence_positive_rate"] - 0.3) < 0.03


def test_the_fallback_rate_is_measured_and_is_small_at_the_default_geometry():
    """Where a planted class has no candidate the other is taken — construction, not rejection — so
    the honest object is the RATE at which that happened, and it is stamped rather than assumed
    negligible."""
    assert _stamp(num_examples=2000, demand=16, keys_per_sequence=8)["presence_forced_rate"] == 0.0
    crowded = _stamp(num_examples=2000, demand=16, keys_per_sequence=15)["presence_forced_rate"]
    assert crowded > 0, "a nearly-full pool must show the tail running out of negatives"


# =============================================================== 4. the demand
@pytest.mark.parametrize("k", [4, 8, 16, 32])
def test_the_union_graph_is_the_complete_graph_on_the_pool(k):
    """A false positive is an error, so every key a sequence draws must be separately distinguishable
    at once: the conflict group is the drawn key set and the union of those cliques over sequences is
    K_K. chi = K, and the demand of the cell is that number."""
    s = _stamp(num_examples=2000, demand=k)
    assert (s["graph_nodes"], s["graph_edges"]) == (k, k * (k - 1) // 2)
    assert s["graph_demand_exact"] and s["graph_demand_lower"] == k == s["graph_demand_upper"]
    assert s["graph_demand_method"] == "disjoint_cliques"
    assert s["graph_predicted_N"] == str(k) and s["graph_demand_nominal"] == k


def test_the_demand_and_the_per_sequence_item_count_are_different_numbers():
    """The decoupling this task gets for free: a sequence draws `s = K/2` keys but the pool is K, so
    'pays for the items of a sequence' and 'pays for the demand' predict thresholds a factor of two
    apart. Without it the grid could not tell the two accounts apart at all."""
    ep = _episodes(num_examples=500, demand=32)
    from zoology.data.presence_recall import stamp
    assert ep.key_token.shape[1] == 16 and stamp(ep)["graph_demand_upper"] == 32


def test_the_coverage_gate_refuses_an_under_realized_sample():
    """The drawn subset varies, so union completeness is a property of the SAMPLE: a handful of
    episodes realizes a strict subgraph of K_K. That cell is refused on the dataset path rather than
    discovered later as an unexplained curve."""
    from zoology.data.graph_chi import DemandMismatch
    with pytest.raises(DemandMismatch, match="coverage shortfall"):
        _segment(num_examples=3, demand=32)


def test_every_stamp_field_reaches_the_segment_slices():
    """`DataSegment` carries `(inputs, labels, slices)` and the on-disk cache stores exactly those,
    so `slices` is the only cache-safe home for the stamp — and it is what `slice_keys` reports
    per-slice accuracy against."""
    from zoology.data.presence_recall import STAMP_FIELDS
    seg = _segment(num_examples=500, demand=16)
    assert set(STAMP_FIELDS) <= set(seg.slices)
    assert seg.slices["graph_demand_upper"] == 16 and seg.slices["graph_nodes"] == 16


@pytest.mark.parametrize("kw,reason", [
    ({"demand": 1}, "demand"),
    ({"demand": 16, "keys_per_sequence": 16}, "keys_per_sequence"),
    ({"demand": 16, "keys_per_sequence": 20}, "keys_per_sequence"),
    ({"demand": 16, "keys_per_sequence": 12, "input_seq_len": 16}, "input_seq_len"),
    ({"demand": 16, "min_separation": 0}, "min_separation"),
    ({"demand": 16, "positive_rate": 0.0}, "positive_rate"),
    ({"demand": 16, "split": "nonesuch"}, "split"),
    ({"demand": 1021, "keys_per_sequence": 2}, "vocab_size"),
])
def test_unbuildable_cells_refuse_by_name(kw, reason):
    """Structural impossibilities raise HERE, by name, rather than asserting deep in the draw — the
    house rule `make_law_data` follows for MQAR's `4*kv <= L`. `keys_per_sequence == demand` is in
    the list on purpose: it is buildable but its tail has no negatives left, so the planted balance
    would decay into the positional prior the subset draw exists to remove."""
    with pytest.raises(ValueError, match=reason):
        _episodes(**kw)


# =============================================================== 5. determinism
def test_determinism_per_seed():
    a, b, c = _segment(num_examples=64), _segment(num_examples=64), _segment(num_examples=64, seed=8)
    assert (a.inputs == b.inputs).all() and (a.labels == b.labels).all()
    assert not (a.inputs == c.inputs).all()


def test_determinism_survives_unrelated_global_consumers():
    """Mirrors `test_mqar_determinism` — the bug it gates was a generator reading numpy's and
    torch's GLOBAL streams, so a same-seed segment could be perturbed by anything else in the
    process that touched either one first."""
    a = _segment(num_examples=64, seed=7)
    np.random.seed(999)
    np.random.random(1000)
    torch.manual_seed(999)
    torch.randint(0, 1000, (10_000,))
    b = _segment(num_examples=64, seed=7)
    assert (a.inputs == b.inputs).all() and (a.labels == b.labels).all()


def test_no_global_rng_reads_in_source():
    """AST form of the audit: no CALL in `presence_recall.py` may invoke the bare global
    `np.random.*` (other than `default_rng`, which is not global state) or `torch.rand*`. Walks the
    parsed call graph rather than grepping text, so a docstring naming the old bug cannot
    false-fail it."""
    import ast
    import inspect

    from zoology.data import presence_recall as mod

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
    assert not hits, f"global-RNG call(s) found in presence_recall.py: {hits}"


def test_gen_version_is_in_the_cache_key():
    """`DataSegment.from_config` hashes `config.model_dump()`; a plain (non-private) `gen_version`
    field therefore invalidates stale on-disk caches by key mismatch when the draw sequence changes,
    without deleting anyone's files. Same mechanism as `MQARConfig.gen_version`."""
    from zoology.data.presence_recall import PRESENCE_RECALL_GEN_VERSION, PresenceRecallConfig
    cfg = PresenceRecallConfig(vocab_size=1024, num_examples=500, input_seq_len=64, demand=8)
    assert cfg.model_dump()["gen_version"] == PRESENCE_RECALL_GEN_VERSION
    seg = cfg.build(seed=5)
    assert seg.inputs.shape == (500, 64) and seg.inputs.dtype == torch.int64
    assert seg.labels.shape == (500, 64) and seg.labels.dtype == torch.int64


# =============================================================== 6. the grid around the generator
def _spec():
    import os

    from rola_bench.mqar.build_configs import load_spec
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    here = os.path.join(repo, "rola_bench", "mqar", "experiments")
    return load_spec(os.path.join(here, "presence_grid.yaml"))


@pytest.mark.parametrize("tier,cells", [("presence16", 48), ("presence64", 48),
                                        ("attn_capacity", 60), ("presence_decay16", 36)])
def test_every_tier_is_the_size_its_header_states(tier, cells):
    """The header's arithmetic is a schedule commitment (each tier is an overnight unit on the local
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


def test_the_slate_is_the_three_wirings_plus_the_control_plus_an_attention_column():
    """The fixed-state class baselines (rla / gla / gdn) are established on the law grid and no claim
    in this file is about them; a stray `baselines` arm would silently triple a tier. Attention is
    admitted in EXACTLY ONE tier — on a capacity tier it would be read as a contender on the N
    ladder, which it cannot be (its state is a growing KV cache)."""
    from rola_bench.models.rola import CANONICAL_WIRINGS
    for tier, arms in _spec()["tiers"].items():
        kinds = {a["build"] for a in arms}
        if tier.startswith("attn"):
            assert kinds == {"attn"}
            continue
        assert kinds == {"routed"}, f"{tier} carries a non-routed arm"
        #: the decay column carries the three structured wirings only (d1's dense write would decay everything every step)
        slate = set(CANONICAL_WIRINGS) - ({"rola-d1-dense"} if "decay" in tier else set())
        assert {a["instance"] for a in arms} == slate, f"{tier} is not the slate"


def test_the_N_ladders_are_pointed_at_the_solved_demand():
    """chi = the pool size, so a ladder entirely above (or below) it cannot see a collapse threshold.
    The routed floor N = 16 (the pre-K31 engine's readout tile) means the chi = 16 tier can only run AT the demand
    and upward, which is stated in the file; the chi = 64 tier must genuinely bracket, since it is
    where the crossing is measured."""
    spec = _spec()
    conds = {c["tag"]: c for c in spec["conditions"]}
    for tier, arms in spec["tiers"].items():
        if tier.startswith("attn"):
            continue
        for arm in arms:
            ncs = arm.get("ncs", spec["ncs"])
            for tag in arm["conditions"]:
                chi = conds[tag]["demand"]
                assert max(ncs) >= chi, f"{tier}/{tag}: ladder {ncs} ends below chi={chi}"
                assert min(ncs) <= chi, f"{tier}/{tag}: ladder {ncs} starts above chi={chi}"
    k64 = list(spec["tiers"]["presence64"])
    assert all(min(a["ncs"]) < 64 for a in k64), "the resize tier must bracket chi from BELOW"


def test_the_attention_axis_sweeps_d_qk_over_three_distinct_demands():
    """A slope needs three points (`mqar.analysis.graph.fit_packing_constant` refuses two by name), and
    the d_qk ladder must be a divisor ladder of d_model, since zoology's MHA derives
    head_dim = d_model // num_heads."""
    spec = _spec()
    conds = {c["tag"]: c for c in spec["conditions"]}
    d_model = spec["backbone"]["d_model"]
    demands = set()
    for a in spec["tiers"]["attn_capacity"]:
        assert a["head_dims"], "the capacity tier must sweep d_qk"
        for d in a["head_dims"]:
            assert d_model % d == 0, f"d_qk={d} does not divide d_model={d_model}"
        for t in a["conditions"]:
            demands.add(conds[t]["demand"])
    assert len(demands) >= 3, f"the packing-constant fit needs three demands, got {sorted(demands)}"


def test_the_attention_column_runs_each_condition_at_its_routed_length():
    """The floor must be the floor of the cells it calibrates: attention at a different L would be
    answering a different question about the same tag."""
    spec = _spec()
    routed_len = {}
    for tier, arms in spec["tiers"].items():
        if tier.startswith("attn"):
            continue
        for a in arms:
            for t in a["conditions"]:
                routed_len[t] = a.get("seq_lens", spec["seq_lens"])[0]
    for a in spec["tiers"]["attn_capacity"]:
        for t in a["conditions"]:
            if t in routed_len:
                assert a.get("seq_lens", spec["seq_lens"])[0] == routed_len[t]


def test_every_run_id_parses_for_the_analysis():
    """The analysis keys its rows off the run_id, so a template change the parser does not know
    about would silently produce an empty headline table rather than an error."""
    import os

    from rola_bench.mqar.analysis.graph import parse_run_id
    from rola_bench.mqar.build_configs import build_configs
    os.environ["GRID_TIERS"] = "all"
    configs, _ = build_configs(_spec())
    tags = {c["tag"] for c in _spec()["conditions"]}
    for cfg in configs:
        got = parse_run_id(cfg.run_id)
        assert got is not None, f"analysis cannot parse {cfg.run_id}"
        assert got[4] in tags, f"{cfg.run_id} parsed to unknown condition {got[4]!r}"


def test_the_grid_stamps_the_realized_demand_as_a_slice_key():
    from zoology.data.presence_recall import STAMP_FIELDS
    keys = _spec()["train"]["slice_keys"]
    assert "graph_demand_upper" in keys and "graph_construction" in keys
    assert set(keys) <= set(STAMP_FIELDS) | {"num_kv_pairs", "input_seq_len"}


def test_the_spec_conditions_build_the_data_they_name():
    """The knob set is validated by `common.make_presence_data`, so a renamed knob refuses here
    rather than being silently dropped and reported as a condition that was never run."""
    from rola_bench.mqar import common as C
    spec = _spec()
    for cond in spec["conditions"]:
        knobs = {k: v for k, v in cond.items() if k != "tag"}
        assert set(knobs) <= set(C.PRESENCE_KNOBS), cond["tag"]
    with pytest.raises(ValueError, match="unknown presence-recall knob"):
        C.make_presence_data(1024, 8, 8, None, train_seq_len=64, test_seq_lens=[64],
                             train_examples=8, test_examples=8, construction="complete")


# =============================================================== 7. the analysis reads it correctly
def test_the_analysis_knows_this_task_and_its_chance_line():
    """A presence accuracy of 0.55 is noise off a coin flip and a graph-recall accuracy of 0.55 is
    most of the task, so the criterion cannot be shared between them. The task is read off the
    spec's own `data.protocol.kind`, and a criterion at or below the chance line is refused —
    otherwise the reported crossing would be the threshold of GUESSING."""
    from rola_bench.mqar.analysis.graph import TASKS, analyse, task_of
    spec = _spec()
    task = task_of(spec)
    assert (task.chance, task.criterion) == (0.5, 0.75)
    assert TASKS["graph_recall"].criterion == 0.5, "the graph grid's historical criterion must hold"
    with pytest.raises(ValueError, match="chance line"):
        analyse("presence_grid", criterion=0.5)


def test_the_analysis_resolves_the_demand_with_the_presence_generator():
    """`predicted_chi` REBUILDS each condition's episodes rather than reading a knob (the stamp is
    absent on a segment-cache hit), so it must dispatch to the generator the spec names — solving a
    presence condition with the graph-recall builder would raise or, worse, answer."""
    from rola_bench.mqar.analysis.graph import predicted_chi
    spec = _spec()
    conds = {c["tag"]: c for c in spec["conditions"]}
    for tag, L in (("presence-k16", 64), ("presence-k64", 256)):
        chi = predicted_chi(spec, conds[tag], L, 2000)
        assert chi["lower"] == chi["upper"] == conds[tag]["demand"], tag
