"""GATES for the cue-consistency instrument (Phase 2, 2026-08-02).

`zoology.data.cue_consistency` is the data generator for the paper's SECOND named boundary
(Sec. 7.1): address agreement, Eq. (7), `addr_r(u) = addr_w(pi(u))`. Nothing downstream can detect a
knob that does not do what its name says, because the whole experiment IS the knob: a cell reports
"accuracy at cue_digits = 1" and there is no independent witness that the cue actually carried one
digit's worth of address. So these gates measure the MECHANISM, not the plumbing:

  * knob (a) is gated in NATS. Measured mutual information between the query's cue and the target's
    write address, normalized by the address entropy, must equal `q / D`, and the conditional entropy
    must equal `(D - q) * log P` — literally the count of open digits the read marginalizes over.
  * knob (b) is gated by the two properties it is supposed to move (is token -> slot a function; is
    the write's slot order stable across episodes) AND by the property it must NOT move (the knob (a)
    measurement above).
  * knob (c) is gated by the REALIZED match count and by the contract's target, both recomputed from
    the episode arrays rather than trusted from the config.

Symbols: D = key_digits; q = cue_digits; P = digit_vocab; kv = num_kv_pairs; m = matches_per_cue;
G = kv/m groups (= queries per episode); L = input_seq_len.

CPU only. A visible GPU is needed only because pytest collection imports Triton.
"""
from __future__ import annotations

import os
from itertools import pairwise

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_EXP_DIR = os.path.join(_REPO, "rola_bench", "mqar", "experiments")

# A small, statistically clean instance: P = 4 keeps the plug-in entropy estimator's bias far below
# the tolerance (16 address states over 80k samples), and kv = 4 keeps G <= P as the generator
# requires. The knob semantics are P-independent; the arithmetic is not, which is why P is small here
# and 64 in the grid.
BASE = {"vocab_size": 200, "input_seq_len": 48, "num_kv_pairs": 4, "key_digits": 2, "digit_vocab": 4}


def _episodes(n=20_000, seed=0, **kw):
    from zoology.data.cue_consistency import build_episodes
    return build_episodes(num_examples=n, seed=seed, **{**BASE, **kw})


def _entropy(rows):
    """Plug-in entropy in nats of the empirical distribution of integer tuples `rows` [n, k]."""
    if rows.shape[1] == 0:
        return 0.0
    _, counts = np.unique(rows, axis=0, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def _cue_address_information(ep):
    """(normalized MI, H(addr | cue) in nats) between the query's cue and the TARGET's write address.

    Both are measured off the arrays the renderer tokenizes, never re-derived from the config."""
    n, G = ep.target_item.shape
    rows = np.arange(n)[:, None]
    addr = ep.digits[rows, ep.target_item].reshape(n * G, -1)     # [n*G, D]
    cue = ep.cue.reshape(n * G, -1)                               # [n*G, q]
    h_addr = _entropy(addr)
    h_joint = _entropy(np.concatenate([cue, addr], axis=1))
    h_cue = _entropy(cue)
    mi = h_cue + h_addr - h_joint
    return mi / h_addr, h_addr - mi


# ================================================== 1. knob (a): cue informativeness, measured in nats
@pytest.mark.parametrize("q,m", [(2, 1), (1, 1), (0, 4)])
def test_knob_a_moves_cue_address_mutual_information_to_exactly_q_over_D(q, m):
    ep = _episodes(cue_digits=q, matches_per_cue=m)
    D, P = BASE["key_digits"], BASE["digit_vocab"]
    norm_mi, h_open = _cue_address_information(ep)
    assert norm_mi == pytest.approx(q / D, abs=0.03), (
        f"cue_digits={q} claims to fix {q} of {D} address digits; measured normalized "
        f"I(cue; addr) = {norm_mi:.4f}, expected {q / D}")
    # the conditional entropy IS the marginalization width: (D - q) open digits of log P nats each.
    assert h_open == pytest.approx((D - q) * np.log(P), abs=0.05)


def test_knob_a_is_monotone_and_spans_full_agreement_to_none():
    D = BASE["key_digits"]
    mis = [_cue_address_information(_episodes(cue_digits=q, matches_per_cue=(1 if q else 4)))[0]
           for q in range(D + 1)]
    assert mis[0] == pytest.approx(0.0, abs=0.03)   # no cue at all
    assert mis[-1] == pytest.approx(1.0, abs=0.03)  # full agreement: the cue IS the address
    assert all(a < b - 0.1 for a, b in pairwise(mis))


# ============================================ 2. knob (b): consistency, and its ORTHOGONALITY to (a)
@pytest.mark.parametrize("consistency", ["consistent", "shared_pool", "episode_shuffled"])
@pytest.mark.parametrize("q", [2, 1])
def test_knob_b_does_not_move_knob_a(consistency, q):
    """(b) changes LEARNABILITY of the address map, never the information the cue carries. If it moved
    (a) as well, every (b) cell would be confounded and PREDICTED-5 would be unreadable."""
    ep = _episodes(cue_digits=q, slot_consistency=consistency, matches_per_cue=1)
    norm_mi, _ = _cue_address_information(ep)
    assert norm_mi == pytest.approx(q / BASE["key_digits"], abs=0.03)


def test_consistent_rung_makes_token_identity_determine_the_digit_slot():
    ep = _episodes(n=64, slot_consistency="consistent")
    D, P = BASE["key_digits"], BASE["digit_vocab"]
    assert ep.digit_tokens.shape == (D, P)
    assert len(set(ep.digit_tokens.ravel().tolist())) == D * P, "slot pools must be DISJOINT"


@pytest.mark.parametrize("consistency", ["shared_pool", "episode_shuffled"])
def test_shared_rungs_break_token_identity_as_a_slot_name(consistency):
    ep = _episodes(n=64, slot_consistency=consistency)
    D = BASE["key_digits"]
    for s in range(1, D):
        assert (ep.digit_tokens[s] == ep.digit_tokens[0]).all(), (
            "the shared rungs must draw every slot from ONE pool — that is what makes the same token "
            "digit 0 in one item and digit 1 in another")


@pytest.mark.parametrize("consistency,shuffled", [("consistent", False), ("shared_pool", False),
                                                  ("episode_shuffled", True)])
def test_only_the_shuffled_rung_permutes_the_write_side_slot_order(consistency, shuffled):
    ep = _episodes(n=2000, slot_consistency=consistency)
    identity = (ep.write_perm == np.arange(BASE["key_digits"])[None, :]).all(axis=1)
    if shuffled:
        # per EPISODE, not per item: within one sequence the correspondence is stable, so depth can
        # in principle recover it while a static layer-1 map cannot (theory notes §4).
        assert 0.3 < identity.mean() < 0.7, identity.mean()
        assert ep.write_perm.shape == (2000, BASE["key_digits"])
    else:
        assert identity.all()


def test_the_query_side_is_always_canonical_order():
    """The shuffle is write-side ONLY. If the query were permuted too, agreement would be restored
    per episode and the condition would measure nothing."""
    from zoology.data.cue_consistency import WILDCARD_TOKEN, cue_consistency
    ep = _episodes(n=200, cue_digits=1, slot_consistency="episode_shuffled")
    seg = cue_consistency(num_examples=200, seed=0, cue_digits=1,
                          slot_consistency="episode_shuffled", random_non_queries=False, **BASE)
    x = seg.inputs.numpy()
    stride, ctx = ep.meta["stride"], ep.meta["context_size"]
    tail = x[:, ctx:]
    starts = np.where(tail[:, 0::stride] != 0)  # every emitted query begins at a stride boundary
    # the cue always occupies slot 0 and the wildcard slot 1, in every episode, whatever write_perm did
    assert (tail[starts[0], starts[1] * stride + 1] == WILDCARD_TOKEN).all()


# ======================================================== 3. knob (c): multi-match and the contract
@pytest.mark.parametrize("q,m", [(1, 1), (1, 2), (1, 4), (0, 4), (2, 1)])
def test_knob_c_realizes_exactly_m_matching_stored_items(q, m):
    """Recomputed from the episode arrays: how many stored items actually agree with the query on the
    digits the cue fixes. A cue tuple colliding between groups would silently double it."""
    ep = _episodes(n=500, cue_digits=q, matches_per_cue=m)
    n, G = ep.cue.shape[0], ep.meta["groups"]
    for i in range(0, n, 37):
        for g in range(G):
            visible = ep.digits[i, :, :q]                 # [kv, q] canonical-order cue digits
            hits = (visible == ep.cue[i, g][None, :]).all(axis=1).sum() if q else len(visible)
            assert hits == m, f"episode {i} group {g}: {hits} matches, expected {m}"


def test_recency_contract_names_the_last_written_match():
    ep = _episodes(n=500, cue_digits=1, matches_per_cue=4, match_resolution="recency")
    slot_of_item = np.argsort(ep.context_order, axis=1)
    for i in range(0, 500, 13):
        for g in range(ep.meta["groups"]):
            members = np.where(ep.group_of_item[i] == g)[0]
            assert ep.target_item[i, g] == members[np.argmax(slot_of_item[i, members])]
            # the m matches carry DISTINCT values, so only the write order can pick the target
            assert len(set(ep.values[i, members].tolist())) == len(members)


def test_consensus_contract_makes_the_marginalization_harmless():
    ep = _episodes(n=500, cue_digits=1, matches_per_cue=4, match_resolution="consensus")
    for i in range(0, 500, 13):
        for g in range(ep.meta["groups"]):
            members = np.where(ep.group_of_item[i] == g)[0]
            vals = set(ep.values[i, members].tolist())
            assert len(vals) == 1, "consensus means every matching leaf holds the SAME value"
            assert ep.values[i, ep.target_item[i, g]] in vals


def test_consensus_and_recency_have_identical_marginalization_width():
    """The pair is only a clean separation if the ONLY thing that differs is the values. Same q, same
    m, same measured cue<->address information."""
    kw = {"cue_digits": 1, "matches_per_cue": 4}
    a = _cue_address_information(_episodes(match_resolution="recency", **kw))
    b = _cue_address_information(_episodes(match_resolution="consensus", **kw))
    assert a[0] == pytest.approx(b[0], abs=0.03) and a[1] == pytest.approx(b[1], abs=0.05)


# ============================================================ 4. determinism and the tokenization
def test_determinism_per_seed():
    from zoology.data.cue_consistency import cue_consistency
    kw = dict(num_examples=64, cue_digits=1, matches_per_cue=2, **BASE)
    a = cue_consistency(seed=7, **kw)
    b = cue_consistency(seed=7, **kw)
    c = cue_consistency(seed=8, **kw)
    assert (a.inputs == b.inputs).all() and (a.labels == b.labels).all()
    assert not (a.inputs == c.inputs).all()


def test_determinism_survives_an_unrelated_global_numpy_consumer():
    """The generator uses its own Generator, not the global `np.random.seed`. A shared global stream
    would make a segment's content depend on what else ran first in the process."""
    from zoology.data.cue_consistency import cue_consistency
    kw = dict(num_examples=64, seed=7, cue_digits=1, matches_per_cue=2, **BASE)
    a = cue_consistency(**kw)
    np.random.seed(999)
    np.random.random(1000)
    b = cue_consistency(**kw)
    assert (a.inputs == b.inputs).all()


@pytest.mark.parametrize("q,m", [(2, 1), (1, 2), (0, 4)])
def test_tokens_decode_back_to_the_episode(q, m):
    """The rendered sequence IS the episode: context items in order, digits under the write
    permutation, value last. Gated by decoding rather than by inspection."""
    from zoology.data.cue_consistency import cue_consistency
    kw = dict(cue_digits=q, matches_per_cue=m, **BASE)
    ep = _episodes(n=200, cue_digits=q, matches_per_cue=m)
    x = cue_consistency(num_examples=200, seed=0, random_non_queries=False, **kw).inputs.numpy()
    D, stride, ctx = BASE["key_digits"], ep.meta["stride"], ep.meta["context_size"]
    for i in range(0, 200, 17):
        for p in range(BASE["num_kv_pairs"]):
            item = ep.context_order[i, p]
            assert x[i, p * stride + D] == ep.values[i, item]
            for s in range(D):
                canonical = ep.write_perm[i, s]
                expected = ep.digit_tokens[s, ep.digits[i, item, canonical]]
                assert x[i, p * stride + s] == expected
        assert (x[i, :ctx] != 0).all()


@pytest.mark.parametrize("q", [2, 1, 0])
def test_wildcard_marks_exactly_the_open_digits_and_occurs_nowhere_else(q):
    from zoology.data.cue_consistency import WILDCARD_TOKEN, cue_consistency
    D, kv = BASE["key_digits"], BASE["num_kv_pairs"]
    m = 1 if q else kv
    seg = cue_consistency(num_examples=300, seed=3, cue_digits=q, matches_per_cue=m,
                          random_non_queries=True, **BASE)
    x = seg.inputs.numpy()
    G = kv // m
    assert (x == WILDCARD_TOKEN).sum(axis=1).tolist() == [G * (D - q)] * 300, (
        "filler must never draw the wildcard: it is the only marker of an open digit")


def test_labels_are_the_contract_target_and_nothing_else():
    from zoology.data.cue_consistency import cue_consistency
    kw = dict(cue_digits=1, matches_per_cue=2, **BASE)
    ep = _episodes(n=300, **kw)
    seg = cue_consistency(num_examples=300, seed=0, random_non_queries=False, **kw)
    lab = seg.labels.numpy()
    G = ep.meta["groups"]
    assert (lab != -100).sum(axis=1).tolist() == [G] * 300
    rows = np.arange(300)[:, None]
    want = np.sort(ep.values[rows, ep.target_item], axis=1)
    got = np.sort(lab[lab != -100].reshape(300, G), axis=1)
    assert (want == got).all()


def test_slices_report_the_condition():
    from zoology.data.cue_consistency import cue_consistency
    seg = cue_consistency(num_examples=8, seed=0, cue_digits=1, matches_per_cue=2, **BASE)
    assert seg.slices["cue_digits"] == 1 and seg.slices["matches_per_cue"] == 2
    assert seg.slices["num_kv_pairs"] == 4 and seg.slices["input_seq_len"] == 48


# ================================================================ 5. the structural refusals
@pytest.mark.parametrize("kw,msg", [
    ({"matches_per_cue": 3}, "divide"),
    ({"cue_digits": 2, "matches_per_cue": 2}, "agreement MEANS"),
    ({"cue_digits": 0, "matches_per_cue": 2}, "EVERY stored item matches"),
    ({"cue_digits": 3}, "cue_digits"),
    ({"input_seq_len": 18}, "does not fit the episode"),
    ({"digit_vocab": 2}, "cue tuples are made distinct"),
    ({"digit_vocab": 4000}, "key half of the vocabulary"),
    ({"slot_consistency": "sometimes"}, "slot_consistency"),
    ({"match_resolution": "oldest"}, "match_resolution"),
])
def test_unrealizable_settings_refuse_by_name(kw, msg):
    from zoology.data.cue_consistency import build_episodes
    with pytest.raises(ValueError, match=msg):
        build_episodes(num_examples=8, seed=0, **{**BASE, **kw})


def test_unknown_knob_is_refused_at_the_spec_surface():
    from rola_bench.mqar.common import make_cue_data
    with pytest.raises(ValueError, match="unknown cue-consistency knob"):
        make_cue_data(8192, 8, 8, None, train_seq_len=48, test_seq_lens=[48], train_examples=8,
                      test_examples=8, cue_digit=1)


# ============================================================== 6. the grid spec
def _spec():
    from rola_bench.mqar.build_configs import load_spec
    return load_spec(os.path.join(_EXP_DIR, "cue_consistency.yaml"))


def test_spec_expands_to_the_cell_count_its_header_states():
    from rola_bench.mqar.build_configs import build_configs
    cfgs, _ = build_configs(_spec())
    assert len(cfgs) == 450, "the header's arithmetic (378 + 72) is a load-bearing claim"
    assert len({c.run_id for c in cfgs}) == len(cfgs), "condition tags must separate the result rows"


def test_every_condition_is_reachable_and_carries_its_knobs():
    from rola_bench.mqar.build_configs import build_configs
    spec = _spec()
    tags = [c["tag"] for c in spec["conditions"]]
    assert tags == ["agree", "agree-shuffled", "partial-m1", "partial-m2", "partial-m4",
                    "partial-m4-consensus", "nocue"]
    cfgs, _ = build_configs(spec)
    for tag in tags:
        cells = [c for c in cfgs if c.run_id.endswith(f"-{tag}")]
        assert cells, f"condition {tag} expanded to zero cells"
        seg = cells[0].data.train_configs[0]
        want = next(c for c in spec["conditions"] if c["tag"] == tag)
        for k, v in want.items():
            if k != "tag":
                assert getattr(seg, k) == v


def test_the_knob_axes_are_actually_crossed_not_confounded():
    """Every condition shares L, kv, vocab and geometry with every other; only the cue knobs move."""
    from rola_bench.mqar.build_configs import build_configs
    cfgs, _ = build_configs(_spec())
    segs = {c.run_id.rsplit("-s", 1)[-1]: c.data.train_configs[0] for c in cfgs}
    fixed = {(s.input_seq_len, s.num_kv_pairs, s.vocab_size, s.key_digits, s.digit_vocab)
             for s in segs.values()}
    assert len(fixed) == 1, f"a condition moved something other than the cue knobs: {fixed}"


def test_capacity_is_held_ample_so_the_boundary_is_not_capacity():
    """PREDICTED-7's precondition: beta = N / kv >= 8 at every rung, so no cell is starved."""
    spec = _spec()
    kv = spec["data"]["protocol"]["num_kv_pairs"]
    assert min(spec["ncs"]) / kv >= 8


def test_full_cue_and_no_cue_are_both_present():
    spec = _spec()
    D = spec["data"]["protocol"]["key_digits"]
    qs = {c["cue_digits"] for c in spec["conditions"]}
    assert D in qs and 0 in qs, "knob (a) must span full agreement to none (§7.1's 'at any N')"


def test_the_multi_match_ladder_has_at_least_two_points():
    """PREDICTED-3 is a 1/m LAW; one point cannot show a law."""
    spec = _spec()
    ms = sorted({c["matches_per_cue"] for c in spec["conditions"]
                 if c["cue_digits"] == 1 and c.get("match_resolution", "recency") == "recency"})
    assert ms == [1, 2, 4]


def test_the_harmless_aliasing_control_matches_a_harmful_cell_exactly():
    spec = _spec()
    by_tag = {c["tag"]: c for c in spec["conditions"]}
    a, b = by_tag["partial-m4"], by_tag["partial-m4-consensus"]
    assert {k: v for k, v in a.items() if k not in ("tag", "match_resolution")} == \
           {k: v for k, v in b.items() if k not in ("tag", "match_resolution")}
    assert b["match_resolution"] == "consensus"


def test_slate_is_the_three_wirings_plus_d1_plus_the_four_contenders():
    from rola_bench.models.rola import CANONICAL_WIRINGS
    spec = _spec()
    routed = [a["instance"] for a in spec["tiers"]["knobs"] if a["build"] == "routed"]
    assert set(routed) == set(CANONICAL_WIRINGS)
    assert any(a["build"] == "attn" for a in spec["tiers"]["knobs"])
    assert spec["tiers"]["contenders"][0]["methods"] == ["rla", "gla", "gdn"]


def test_predictions_are_stated_in_the_header_and_marked():
    with open(os.path.join(_EXP_DIR, "cue_consistency.yaml")) as fh:
        head = fh.read().split("project_name:")[0]
    assert "ADVANCE PREDICTIONS" in head
    for n in range(1, 8):
        assert f"PREDICTED-{n} " in head, f"prediction {n} is missing from the spec header"
    assert head.count("FALSIFIED BY") >= 5


# =========================================================== 7. the axis must not leak into old specs
@pytest.mark.parametrize("name", ["law_grid", "ood_recall", "canonical", "router_bias"])
def test_specs_without_conditions_are_untouched_by_the_new_axis(name):
    from rola_bench.mqar.build_configs import _conditions, load_spec
    spec = load_spec(os.path.join(_EXP_DIR, f"{name}.yaml"))
    assert _conditions(spec) == [(None, {})]
    assert "cond" not in str(spec)


def test_a_condition_axis_without_cue_protocol_refuses():
    from rola_bench.mqar.build_configs import build_configs
    spec = _spec()
    spec["data"]["protocol"] = {"train_kv": [4], "test_seq_lens": [128],
                                "test_kv_by_len": {128: [4]}, "train_examples": 8, "test_examples": 8}
    with pytest.raises(ValueError, match="silently ignored"):
        build_configs(spec)


def test_an_explicit_run_id_without_cond_refuses():
    from rola_bench.mqar.build_configs import build_configs
    spec = _spec()
    spec["tiers"] = {"knobs": [{"build": "routed", "instance": "rola-d1-dense", "tag": "d1",
                                "run_id": "cell-{tag}-{nc}-{seed}"}]}
    with pytest.raises(ValueError, match=r"must contain \{cond\}"):
        build_configs(spec)


def test_an_arm_naming_an_undeclared_condition_refuses():
    from rola_bench.mqar.build_configs import build_configs
    spec = _spec()
    spec["tiers"]["contenders"][0]["conditions"] = ["agree", "typo-tag"]
    with pytest.raises(ValueError, match="does not declare"):
        build_configs(spec)


def test_duplicate_condition_tags_refuse():
    from rola_bench.mqar.build_configs import build_configs
    spec = _spec()
    spec["conditions"].append(dict(spec["conditions"][0]))
    with pytest.raises(ValueError, match="duplicate condition tag"):
        build_configs(spec)
