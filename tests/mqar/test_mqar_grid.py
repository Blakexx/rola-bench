"""GATES for the MQAR grid: the wirings, the contender ladder, the realized-state readback, and every experiment spec.

Mostly CPU. RoLA's forward runs only on rola's CUDA op, so the one gate that trains a wiring is CUDA-marked; while rola
refuses training (before its native backward) that gate fails with rola's own error, on purpose.
"""
from __future__ import annotations

import os

import pytest

_EXP_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "rola_bench", "mqar",
                        "experiments")
_SPECS = ("law_grid", "separation", "ood_recall", "canonical", "dmodel64", "cue_consistency", "graph_grid",
          "presence_grid", "router_bias")
_N = 64


def _mixer(wiring, n=_N):
    from zoology.mixers.rola import RoLAMixer

    from rola_bench.models import rola as cells
    from rola_bench.mqar import MQAR_GEOM

    config = cells.mixer_config(cells.cell(wiring, n), **MQAR_GEOM)
    return RoLAMixer(d_model=MQAR_GEOM["n_heads"] * MQAR_GEOM["d_v"], layer_idx=0, **config["kwargs"])


# =============================================================== 1. the wirings
def test_slate_is_the_three_wirings_plus_the_d1_control():
    from rola_bench.models import rola as cells

    assert cells.CANONICAL_WIRINGS == ("rola-arm1-densread-sparsewrite", "rola-arm2-union", "rola-arm3-levelsplit",
                                       "rola-d1-dense")
    assert set(cells.WIRINGS) - set(cells.CANONICAL_WIRINGS) == {
        "rola-arm3-levelsplit-a2w",                  # the alpha axis
        "rola-arm1-tied", "rola-d1-dense-tied",      # the coupling-independence pair
        "rola-hybrid", "rola-hybrid-tiedtop",        # the depth hybrid and its tied-top ablation
    }


@pytest.mark.parametrize("wiring", ["rola-arm1-densread-sparsewrite", "rola-arm2-union", "rola-arm3-levelsplit",
                                    "rola-arm3-levelsplit-a2w", "rola-d1-dense", "rola-arm1-tied", "rola-d1-dense-tied",
                                    "rola-hybrid", "rola-hybrid-tiedtop"])
def test_every_wiring_constructs(wiring):
    from rola_bench.mqar import MQAR_GEOM

    mixer = _mixer(wiring)
    assert mixer.layer.topology.N == _N and mixer.layer.head_v_dim == MQAR_GEOM["d_v"]


@pytest.mark.parametrize("wiring", ["rola-arm1-densread-sparsewrite", "rola-arm2-union", "rola-arm3-levelsplit",
                                    "rola-d1-dense"])
def test_canonical_wirings_train_on_the_kernel(wiring):
    """Forward and backward on the CUDA op at the grid's own geometry."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    mixer = _mixer(wiring).cuda().train()
    d_model = mixer.layer.hidden_size
    x = torch.randn(2, 64, d_model, device="cuda", requires_grad=True)
    out = mixer(x)
    out.float().pow(2).mean().backward()
    assert out.shape == (2, 64, d_model) and bool(torch.isfinite(out).all())
    assert x.grad is not None and float(mixer.layer.routes.route_W.grad.abs().sum()) > 0


@pytest.mark.parametrize("wiring", ["rola-arm1-densread-sparsewrite", "rola-arm2-union", "rola-arm3-levelsplit",
                                    "rola-d1-dense"])
def test_routing_resolves_to_the_intended_structure(wiring):
    """Union is the one wiring whose untied sparse/sparse level resolves to rola's UnionRouting (one shared support
    split between the duties); every other level is IndependentRouting (each duty its own exact activation)."""
    import rola

    union_levels = {"rola-arm2-union": (0,)}.get(wiring, ())
    for index, level in enumerate(_mixer(wiring).layer.routes.levels):
        want = rola.UnionRouting if index in union_levels else rola.IndependentRouting
        assert isinstance(level, want), f"{wiring}: level {index} is {type(level).__name__}, expected {want.__name__}"


def test_d1_dense_control_is_a_single_dense_level():
    import rola

    layer = _mixer("rola-d1-dense").layer
    assert len(layer.routes.levels) == 1 and layer.topology.N == _N
    level = layer.routes.levels[0]
    assert isinstance(level.read, rola.SoftmaxActivation) and isinstance(level.write, rola.SoftmaxActivation)


def test_a2w_variant_differs_from_arm3_in_alpha_alone():
    from rola_bench.models import rola as cells

    base, a2w = cells.WIRINGS["rola-arm3-levelsplit"], cells.WIRINGS["rola-arm3-levelsplit-a2w"]
    assert [(r["tied"], r["read"], r["write"]) for r in base] == [(r["tied"], r["read"], r["write"]) for r in a2w]
    assert [r.get("alpha") for r in a2w] == [2.0, 1.5]      # sparsemax on the sparse write, entmax-1.5 on the read


def test_a_wiring_without_a_uniform_spelling_is_refused_by_name():
    from rola_bench.models import rola as cells

    assert cells.cell("rola-arm1-densread-sparsewrite", 256).widths == (16, 16)
    with pytest.raises(cells.NoSpelling):
        cells.cell("rola-arm1-densread-sparsewrite", 128)
    assert cells.cell("rola-arm1-densread-sparsewrite", 128, widths=[8, 16]).widths == (8, 16)
    with pytest.raises(ValueError, match="multiply"):
        cells.cell("rola-arm1-densread-sparsewrite", 128, widths=[8, 8])


# =============================================================== 2. the capacity coordinate
def test_realized_state_is_read_back_not_computed():
    """H * N * d_v content and the H * N mass column, measured off the constructed mixer."""
    from rola_bench.models import rola as cells
    from rola_bench.mqar import MQAR_GEOM

    H, dv = MQAR_GEOM["n_heads"], MQAR_GEOM["d_v"]
    for n in (64, 256):
        content, overhead = cells.state_floats(_mixer("rola-arm1-densread-sparsewrite", n))
        assert (content, overhead) == (H * n * dv, H * n)


def test_ref_is_derived_from_the_pinned_geometry():
    import rola_bench.mqar.experiments.canonical_baselines as cb
    from rola_bench.mqar import MQAR_GEOM

    for n in (2, 16, 256):
        assert cb.REF(n) == MQAR_GEOM["n_heads"] * n * MQAR_GEOM["d_v"]


def test_readback_covers_every_arm_the_grid_builds():
    import rola_bench.mqar.experiments.canonical_baselines as cb
    from rola_bench.models import rola as cells
    from rola_bench.mqar import MQAR_GEOM
    from rola_bench.mqar import common as C
    from rola_bench.mqar.realized_state import state_floats

    for method in cb.METHODS:
        for shape in ("wide", "square"):
            cell = cb.baseline_cell(method, shape, 64)
            if cell is None:
                continue
            content, overhead = state_floats(cell[0], cb.DMODEL)
            assert isinstance(content, int) and content > 0 and overhead >= 0
    rola_kernel = cells.mixer_config(cells.cell("rola-d1-dense", 64), **MQAR_GEOM)
    assert state_floats(rola_kernel, cb.DMODEL)[0] == MQAR_GEOM["n_heads"] * 64 * MQAR_GEOM["d_v"]
    assert state_floats(C.mha, cb.DMODEL) == (None, 0)      # attention: unbounded, no rung


def test_baseline_readback_is_within_a_rounding_step_of_the_target():
    import rola_bench.mqar.experiments.canonical_baselines as cb
    from rola_bench.mqar.realized_state import state_floats

    for method in cb.METHODS:
        for shape in cb.SHAPES:
            for n in (16, 64, 256):
                cell = cb.baseline_cell(method, shape, n)
                if cell is None:
                    continue
                kernel, target = cell
                content, _ = state_floats(kernel, cb.DMODEL)
                assert abs(content - target) / target <= 0.12, f"{method}/{shape}/nc{n}: {content} vs {target}"


# =============================================================== 3. the contender ladder
def test_contender_ladder_is_one_occupant_per_rung():
    import rola_bench.mqar.experiments.canonical_baselines as cb

    assert cb.METHODS == ["rla", "gla", "gdn"]
    assert cb.baseline_cell("hedgehog", "wide", 16) is None


def test_gdn_structural_exit_is_reported_not_hidden():
    import rola_bench.mqar.experiments.canonical_baselines as cb

    assert cb.baseline_cell("gdn", "wide", 256) is not None
    assert cb.baseline_cell("gdn", "wide", 512) is None       # head_dim = N > 256
    assert cb.baseline_cell("gdn", "square", 1024) is not None
    assert cb.baseline_cell("gdn", "square", 2048) is None    # head_dim = 8*sqrt(N) > 256
    assert cb.baseline_cell("gla", "wide", 2048) is not None
    assert cb.baseline_cell("rla", "wide", 2048) is not None


# =============================================================== 4. the specs
def _load(name):
    from rola_bench.mqar.build_configs import load_spec

    return load_spec(os.path.join(_EXP_DIR, f"{name}.yaml"))


@pytest.mark.parametrize("name", _SPECS)
def test_every_spec_expands(name):
    from rola_bench.mqar.build_configs import build_configs

    configs, envs = build_configs(_load(name))
    assert configs and len(configs) == len(envs)
    ids = [c.run_id for c in configs]
    assert len(ids) == len(set(ids)), f"{name}: {len(ids) - len(set(ids))} duplicate run_id(s)"


@pytest.mark.parametrize("name", _SPECS)
def test_every_spec_carries_at_least_three_seeds(name):
    spec = _load(name)
    seeds = spec["train"].get("seeds") or [spec["train"]["seed"]]
    assert len(seeds) >= 3 or spec.get("single_seed_rationale"), f"{name}: {len(seeds)} seed(s) and no rationale"


def test_seed_floor_actually_bites():
    from rola_bench.mqar.build_configs import build_configs

    spec = _load("law_grid")
    spec["train"]["seeds"] = [0]
    with pytest.raises(ValueError, match="seed"):
        build_configs(spec)


def test_a_routed_arm_with_a_field_no_cell_has_is_refused():
    from rola_bench.mqar.build_configs import build_configs

    spec = _load("canonical")
    spec["tiers"] = {"rola": [{"build": "routed", "instance": "rola-d1-dense", "tag": "d1", "state_norm": "raw"}]}
    with pytest.raises(ValueError, match="state_norm"):
        build_configs(spec)


def test_law_grid_walks_alpha_cap_through_one_in_both_directions():
    spec = _load("law_grid")
    ratios = [nc / L for nc in spec["ncs"] for L in spec["seq_lens"]]
    assert min(ratios) < 1 < max(ratios)
    diagonal = sorted(set(spec["ncs"]) & set(spec["seq_lens"]))
    assert len(diagonal) >= 3, f"the N = L diagonal has only {diagonal}"


def test_law_grid_unconfounds_items_from_length():
    proto = _load("law_grid")["data"]["protocol"]
    ladders = {L: set(kvs) for L, kvs in proto["test_kv_by_len"].items()}
    longest = max(ladders)
    for L, kvs in ladders.items():
        assert kvs == {kv for kv in ladders[longest] if 4 * kv <= L}, f"L={L} ladder is {sorted(kvs)}"


def test_law_grid_has_extrapolation_cells():
    spec = _load("law_grid")
    assert max(spec["data"]["protocol"]["test_seq_lens"]) > max(spec["seq_lens"])
    assert "input_seq_len" in spec["train"]["slice_keys"]


def test_separation_cells_sit_on_the_diagonal():
    spec = _load("separation")
    for tier, arms in spec["tiers"].items():
        if not tier.startswith("diag"):
            continue
        for arm in arms:
            if arm["build"] == "routed":
                assert arm["ncs"] == arm["seq_lens"], f"{tier}: {arm['tag']} is off the N = L diagonal"
    rungs = sorted({arm["ncs"][0] for t, arms in spec["tiers"].items() if t.startswith("diag")
                    for arm in arms if arm["build"] == "routed"})
    assert rungs == [16384, 32768, 65536]


def test_separation_gdn_is_capped_at_its_reachable_rungs():
    import rola_bench.mqar.experiments.canonical_baselines as cb

    for arm in _load("separation")["tiers"]["gdn_ceiling"]:
        if "gdn" not in arm["methods"]:
            continue
        for shape in arm["shapes"]:
            for nc in arm["ncs"]:
                assert cb.baseline_cell("gdn", shape, nc) is not None, f"gdn/{shape}/nc{nc} is above the head_dim cap"


def test_ood_recall_holds_the_ratio_and_sweeps_absolute_length():
    spec = _load("ood_recall")
    proto = spec["data"]["protocol"]
    train_L, train_kv = spec["seq_lens"][0], proto["train_kv"][0]
    ratio = train_kv / train_L
    for L, kvs in proto["test_kv_by_len"].items():
        assert kvs[0] / L == ratio, f"L={L}: the fixed-ratio ladder point is {kvs[0]}, not {L * ratio:g}"
    fixed_item = [kvs[1] for L, kvs in proto["test_kv_by_len"].items() if len(kvs) > 1]
    assert fixed_item and len(set(fixed_item)) == 1
    assert max(proto["test_seq_lens"]) >= 16 * train_L


def test_attention_dqk_ladder_expands():
    from rola_bench.mqar.build_configs import build_configs

    spec = _load("canonical")
    spec["tiers"] = {"attn": [{"build": "attn", "head_dims": [2, 4, 8, 16, 32]}]}
    configs, _ = build_configs(spec)
    assert {c.run_id.split("_")[0] for c in configs} == {f"grid-mha-dqk{d}" for d in (2, 4, 8, 16, 32)}
