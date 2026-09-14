"""Generic MQAR config generator: a declarative spec -> zoology (configs, configs_envs).

One generator owns all the grid LOGIC (cell builders, the baseline shape/state-match solve, run_id
naming); an experiment is just a YAML spec declaring the axes (backbone, data, lrs/seeds, and an
`arms` list grouped into tiers).

The grid's axes:

  * the `routed` arm builds a RoLA cell (`rola_bench.models.rola`): a wiring at each state count, as zoology's mixer
    over fla.
  * **`seq_lens` is a first-class axis, crossed with `ncs`.** `L` used to be baked into the data
    preset, so `alpha_cap = N/L` was derivable post-hoc and sweepable never. Crossing them is what
    lets a spec walk `alpha_cap` through 1 from BOTH directions at fixed everything-else, which is
    the form the N = L boundary claim is stated in (audit §4.1 item 1).
  * **L-extrapolation is an axis, not a by-product**: `data.protocol.test_seq_lens` may contain
    lengths greater than the cell's train length, and those slices are reported separately.
  * **the addressing-entropy ladder `S = sum_l b_l`** is expressible: a routed arm may carry an
    explicit `states_per_level` list (mixed radix), which must multiply to `nc` (audit §4.1 item 3).
  * **the attention `d_qk` ladder** is expressible: `{build: attn, head_dims: [...]}` sweeps the
    per-head key width, which for zoology's MHA is `d_model / num_heads` — the one place `d_qk`
    legitimately survives the V3 migration, on the attention baseline only (audit §4.1 item 4).

Spec schema (see experiments/*.yaml):
  project_name: str
  backbone: {d_model, n_layers, vocab, l_max}        # any omitted -> common.DEFAULTS
  data:
    # either a fixed named preset ...
    {preset: ext, train_batch, test_batch, cache_dir}
    # ... or the law-grid protocol, which makes L an axis:
    {train_batch, test_batch, cache_dir,
     protocol: {train_kv: [..], test_seq_lens: [..], test_kv_by_len: {L: [..]},
                train_examples: int, test_examples: int}}
  train:    {max_epochs, weight_decay, seed | seeds:[..], early_stopping_threshold,
             early_stopping_metric, slice_keys, eval_every}
  lrs:  [..]                  # default LR sweep (an arm may override with its own `lrs`)
  ncs:  [..]                  # default state-rung ladder (an arm may override with `ncs`)
  seq_lens: [..]              # OPTIONAL train-length ladder; requires data.protocol. Crossed with ncs.
  conditions: [{tag, <generator knob overrides>}, ...]
                              # OPTIONAL DATA-CONTRACT axis (requires a knob-carrying
                              # data.protocol.kind: `cue_consistency` / `graph_recall` /
                              # `presence_recall`). Crossed with everything; each condition is
                              # trained separately. See `_conditions`, `common.make_cue_data` /
                              # `common.make_graph_data` / `common.make_presence_data`.
  tiers: {"<id>": [arm, ...]} # selected by the GRID_TIERS env (default all)
  arm kinds:
    {build: routed, instance, tag, n_heads?, d_v?, states_per_level?, decay?, router_bias?, lrs?, ncs?,
     seq_lens?, conditions?, run_id?}      # instance: a wiring; decay: null or a rola decay source dict
        run_id template vars: {tag} {nc} {st} {lr:.0e} {seed} {d_model} {L} {alpha_cap} {cond}
        default: "grid-routed-{tag}-nc{nc}_L{L}_st{st}_lr{lr:.0e}_s{seed}"
    {build: baselines, methods: [..], shapes: [..]}
    {build: attn, head_dims: [..]?}
"""
import os
from itertools import product

from zoology.config import LoggerConfig, ModelConfig, ModuleConfig, TrainConfig

from rola_bench.models import rola as cells
from rola_bench.mqar import MQAR_GEOM, realized_state
from rola_bench.mqar import common as C
from rola_bench.mqar.experiments import canonical_baselines as _cb
from rola_bench.mqar.experiments.canonical_baselines import baseline_cell

_ROUTED_RUNID = "grid-routed-{tag}-nc{nc}_L{L}_st{st}_lr{lr:.0e}_s{seed}"
_BASELINE_RUNID = "grid-{method}-{shape}-nc{nc}_L{L}_st{st}_lr{lr:.0e}_s{seed}"
_ATTN_RUNID = "grid-mha-dqk{d_qk}_L{L}_lr{lr:.0e}_s{seed}"

#: Minimum seeds for a published cell — see the check in `build_configs`.
MIN_SEEDS = 3

#: The fields a routed arm may carry; any other is refused rather than silently ignored.
ROUTED_FIELDS = frozenset({"build", "instance", "tag", "n_heads", "d_v", "states_per_level", "decay", "router_bias",
                           "lrs", "ncs", "seq_lens", "conditions", "run_id"})


def _conditions(spec):
    """The CUE-CONDITION axis: `[(tag, knobs), ...]`, or `[(None, {})]` when a spec declares none.

    A condition is one setting of the cue-consistency knobs (`common.CUE_KNOBS`). It is a DATA axis
    like `seq_lens`, crossed with everything else, and it exists because the paper's second boundary
    (address agreement, Sec. 7.1 / Eq. (7)) is a property of the retrieval CONTRACT, which no
    capacity axis can move.
    """
    conds = spec.get("conditions")
    if not conds:
        return [(None, {})]
    out, seen = [], set()
    for c in conds:
        c = dict(c)
        tag = c.pop("tag", None)
        if not tag:
            raise ValueError("every entry of `conditions:` needs a `tag:` — it is the only thing "
                             "that separates the cells in the result rows.")
        if tag in seen:
            raise ValueError(f"duplicate condition tag {tag!r}; tags key the results.")
        seen.add(tag)
        out.append((tag, c))
    return out


def _data_for(spec, bb, L, knobs):
    """The DataConfig for train length `L` (or the spec's fixed preset when L is None)."""
    dat = spec["data"]
    proto0 = dat.get("protocol") or {}
    # The KNOB-CARRYING protocols: a `conditions:` axis is one setting of the named generator's
    # knobs, and each condition is TRAINED separately (a model trained on a mixture of contracts
    # answers a different question than "is this contract servable").
    builder = {"cue_consistency": C.make_cue_data, "graph_recall": C.make_graph_data,
               "presence_recall": C.make_presence_data}.get(proto0.get("kind"))
    if builder is not None:
        if L is None:
            raise ValueError(
                f"the {proto0['kind']} protocol makes L explicit; declare `seq_lens:`.")
        merged = {k: v for k, v in proto0.items()
                  if k not in ("kind", "train_examples", "test_examples", "test_seq_lens")}
        merged.update(knobs)
        # An OMITTED `test_seq_lens` means "evaluate in-distribution, at this cell's own train
        # length". It exists because these protocols cross conditions whose sequence geometry
        # differs (a `complete(64)` graph episode needs L >= 256 while `complete(16)` needs 64), so
        # one global evaluation-length list would hand some conditions a length their generator
        # structurally refuses. Extrapolation is `ood_recall`'s axis, not this mechanism's.
        test_lens = [int(x) for x in proto0["test_seq_lens"]] if proto0.get("test_seq_lens") else [L]
        return builder(
            bb["vocab"], dat["train_batch"], dat["test_batch"], dat["cache_dir"],
            train_seq_len=L, test_seq_lens=test_lens,
            train_examples=proto0["train_examples"], test_examples=proto0["test_examples"],
            **merged)
    if knobs:
        raise ValueError(
            f"spec declares conditions {sorted(knobs)} but its data protocol declares no knob-"
            "carrying `kind:` (`cue_consistency` / `graph_recall` / `presence_recall`); the knobs "
            "would be silently ignored.")
    if L is None:
        if "preset" not in dat:
            raise ValueError(
                "spec declares no `seq_lens` axis and no `data.preset`: one of the two must say "
                "what sequence lengths the cell trains on.")
        return C.make_data(dat["preset"], bb["vocab"], dat["train_batch"], dat["test_batch"],
                           dat["cache_dir"])
    proto = dat.get("protocol")
    if proto is None:
        raise ValueError(
            "`seq_lens` makes L an axis, which requires `data.protocol` (train_kv / test_seq_lens / "
            "test_kv_by_len / train_examples / test_examples). A named `data.preset` bakes its own "
            "fixed L ladder in and cannot be crossed with `ncs` — that is the axis the pre-V3 grid "
            "was missing, so silently falling back to the preset here would reproduce the gap.")
    # `by_train_len` lets one spec carry protocols that differ PER TRAIN LENGTH. It exists for the
    # separation cells, where a rung's evaluation ladder must scale with the rung (holding a fixed
    # item/length ratio across three orders of magnitude) and where evaluating a 16K cell at 64K
    # would be pure waste. Top-level protocol keys are the defaults; the per-length block overrides.
    per_len = (proto.get("by_train_len") or {})
    override = per_len.get(L, per_len.get(str(L)))
    if override is not None:
        proto = {**proto, **override}
    elif per_len:
        raise ValueError(
            f"data.protocol.by_train_len is declared but has no entry for train length {L}. Every "
            "length in `seq_lens` must be answered explicitly — an unstated rung would silently "
            "inherit another rung's evaluation ladder.")
    test_lens = [int(x) for x in proto["test_seq_lens"]]
    kv_by_len = {int(k): list(v) for k, v in proto["test_kv_by_len"].items()}
    missing = [x for x in test_lens if x not in kv_by_len]
    if missing:
        raise ValueError(f"data.protocol.test_kv_by_len has no entry for test length(s) {missing}")
    return C.make_law_data(
        bb["vocab"], dat["train_batch"], dat["test_batch"], dat["cache_dir"],
        train_seq_len=L, train_kv=proto["train_kv"], test_seq_lens=test_lens,
        test_kv_by_len=kv_by_len, train_examples=proto["train_examples"],
        test_examples=proto["test_examples"])


def build_configs(spec):
    """Expand a spec dict into (configs, configs_envs)."""
    bb = {**C.DEFAULTS, **spec.get("backbone", {})}   # spec overrides the library defaults
    _cb.DMODEL = bb["d_model"]                         # baseline projection solve reads this
    tr = spec["train"]
    def_lrs, def_ncs = spec["lrs"], spec["ncs"]
    def_seq_lens = spec.get("seq_lens") or [None]      # [None] => the fixed data preset
    seeds = tr.get("seeds") or [tr["seed"]]
    # SEED FLOOR (ruling 2026-08-02, from the MQAR retrospective): the pre-V3 grid was single-seed
    # EVERYWHERE, so no cell in it carries a variance estimate — and the law claim is a claim about
    # where a boundary sits, which is not readable off one draw. Three seeds is the floor for any
    # published cell. The escape is deliberately a WRITTEN REASON, not a flag: a spec that runs
    # single-seed must say in the spec why that is defensible, and the reason travels with the file.
    if len(seeds) < MIN_SEEDS and not spec.get("single_seed_rationale"):
        raise ValueError(
            f"spec {spec.get('project_name')!r} declares {len(seeds)} seed(s); published MQAR cells "
            f"require >= {MIN_SEEDS} (variance is part of the law claim, and the pre-V3 grid's "
            "single-seed cells are exactly why). Either widen `train.seeds`, or state a "
            "`single_seed_rationale:` in the spec explaining why this experiment does not need one.")
    l_max = bb["l_max"]
    env = {"EVAL_EVERY_N": str(tr["eval_every"])} if tr.get("eval_every") else {}

    conds = _conditions(spec)
    if len(conds) > 1 or conds[0][0] is not None:
        # With a condition axis every default run_id template becomes ambiguous. Refuse rather than
        # emit colliding ids: two conditions writing one result row is indistinguishable from noise.
        for tier in spec["tiers"].values():
            for arm in tier:
                rid = arm.get("run_id")
                if rid is not None and "{cond}" not in rid:
                    raise ValueError(
                        f"spec declares a `conditions:` axis, so arm run_id {rid!r} must contain "
                        "{cond} — otherwise every condition writes the same result row.")

    configs, envs = [], []
    _data_cache = {}

    def data_for(L, cond_tag, knobs):
        if (L, cond_tag) not in _data_cache:
            _data_cache[(L, cond_tag)] = _data_for(spec, bb, L, knobs)
        return _data_cache[(L, cond_tag)]

    def add(kernel, run_id, lr, seed, L, cond_tag, knobs):
        configs.append(TrainConfig(
            data=data_for(L, cond_tag, knobs),
            model=ModelConfig(block_type="TransformerBlock",
                              sequence_mixer=C.wrap_hybrid(kernel, l_max),
                              state_mixer=ModuleConfig(name="torch.nn.Identity", kwargs={}),
                              d_model=bb["d_model"], n_layers=bb["n_layers"],
                              max_position_embeddings=0, vocab_size=bb["vocab"]),
            logger=LoggerConfig(project_name=spec["project_name"], entity=""),
            max_epochs=tr["max_epochs"], learning_rate=lr, weight_decay=tr["weight_decay"],
            seed=seed, run_id=run_id,
            early_stopping_threshold=tr["early_stopping_threshold"],
            early_stopping_metric=tr["early_stopping_metric"],
            slice_keys=list(tr["slice_keys"])))
        envs.append(dict(env))

    def _axes(arm):
        """(lrs, ncs, seq_lens) for one arm — arm-level overrides beat the spec defaults."""
        return (arm.get("lrs", def_lrs), arm.get("ncs", def_ncs),
                arm.get("seq_lens", def_seq_lens))

    def _arm_conds(arm):
        """The cue conditions this arm runs: all of them, or the named subset in `arm['conditions']`.

        The subset exists so an expensive reference tier (the contender ladder) can run the SPINE of
        the knob ladder while the arms the theory makes claims about run all of it. Naming an unknown
        tag refuses — a typo would silently shrink the grid."""
        want = arm.get("conditions")
        if want is None:
            return conds
        want = list(want)
        unknown = [t for t in want if t not in {c[0] for c in conds}]
        if unknown:
            raise ValueError(f"arm names condition tag(s) {unknown} that the spec does not declare; "
                             f"declared: {[c[0] for c in conds]}")
        return [c for c in conds if c[0] in want]

    def _fmt(rid, cond_tag=None, **kw):
        # `alpha_cap = N/L` is the capacity ratio the boundary claim is stated in; it is available to
        # every run_id template so a cell names its own position relative to the N = L diagonal.
        L, nc = kw.get("L"), kw.get("nc")
        kw.setdefault("alpha_cap", (f"{nc / L:g}" if (L and nc) else "na"))
        kw.setdefault("L", "preset")
        kw["cond"] = cond_tag
        if cond_tag is not None and "{cond}" not in rid:
            rid = rid + "-{cond}"        # DEFAULT templates predate the axis; explicit ones are checked
        return rid.format(**kw)

    def realized_st(kernel):
        """The cell's CAPACITY COORDINATE — realized content floats, read back off the constructed
        module (`realized_state`). Never the nominal target, never a formula: the retrospective's
        ruling, and the same fix as finding F1. `unbounded` for attention, which has no rung."""
        content, _overhead = realized_state.state_floats(kernel, bb["d_model"])
        return "unbounded" if content is None else content

    # --- arm builders (run_id formats are load-bearing: they key the result rows) ---
    def routed(arm):
        rid = arm.get("run_id", _ROUTED_RUNID)
        unknown = sorted(set(arm) - ROUTED_FIELDS)
        if unknown:
            raise ValueError(f"routed arm {arm.get('tag')!r} carries field(s) {unknown} that are not a RoLA cell's; "
                             f"expected a subset of {sorted(ROUTED_FIELDS)}")
        n_heads, d_v = arm.get("n_heads", MQAR_GEOM["n_heads"]), arm.get("d_v", MQAR_GEOM["d_v"])
        a_lrs, a_ncs, a_lens = _axes(arm)
        a_conds = _arm_conds(arm)
        dropped = []
        for seed, (ctag, cknobs) in product(seeds, a_conds):
            for L in a_lens:
                for nc in a_ncs:
                    try:
                        cell = cells.cell(arm["instance"], nc, widths=arm.get("states_per_level"),
                                          decay=arm.get("decay"), router_bias=arm.get("router_bias", True))
                    except cells.NoSpelling as e:
                        # Structural, not a failure: a D=2 wiring has no spelling at N=2. Reported
                        # exactly like a baseline's kernel ceiling — an omitted cell with a reason.
                        if seed == seeds[0] and a_lens[0] == L and ctag == a_conds[0][0]:
                            dropped.append(f"{arm['tag']}/nc{nc}: {e}")
                        continue
                    kernel = cells.mixer_config(cell, n_heads=n_heads, d_v=d_v)
                    st = realized_st(kernel)
                    for lr in a_lrs:
                        add(kernel, _fmt(rid, ctag, tag=arm["tag"], nc=nc, st=st, lr=lr, seed=seed,
                                         d_model=bb["d_model"], L=L), lr, seed, L, ctag, cknobs)
        if dropped:
            print(f"[build_configs] {len(dropped)} routed cell(s) omitted -- the preset has no "
                  f"spelling at that rung (NOT a run failure): {dropped}", flush=True)

    def baselines(arm):
        rid = arm.get("run_id", _BASELINE_RUNID)
        a_lrs, a_ncs, a_lens = _axes(arm)
        a_conds = _arm_conds(arm)
        dropped = []
        for seed, (ctag, cknobs) in product(seeds, a_conds):
            for L in a_lens:
                for method in arm["methods"]:
                    for shape in arm["shapes"]:
                        for nc in a_ncs:
                            cell = baseline_cell(method, shape, nc)
                            if cell is None:
                                # NOT a run failure: an undefined shape or a STRUCTURAL EXIT (the
                                # method's own kernel/memory bound). Logged once, and it is a
                                # reportable cost column for that method, never missing data.
                                if seed == seeds[0] and a_lens[0] == L and ctag == a_conds[0][0]:
                                    dropped.append(f"{method}/{shape}/nc{nc}")
                                continue
                            kernel, _nominal = cell
                            st = realized_st(kernel)
                            for lr in a_lrs:
                                add(kernel, _fmt(rid, ctag, method=method, shape=shape, nc=nc, st=st,
                                                 lr=lr, seed=seed, d_model=bb["d_model"], L=L),
                                    lr, seed, L, ctag, cknobs)
        if dropped:
            print(f"[build_configs] {len(dropped)} baseline cell(s) omitted — matched-state "
                  f"undefined / structural exit (NOT a run failure): {dropped}", flush=True)

    def attn(arm):
        """The MHA oracle. `head_dims` sweeps the per-head KEY width — the attention `d_qk` ladder.

        zoology's MHA derives `head_dim = d_model // num_heads`, so a head width `d_qk` is requested
        by asking for `d_model // d_qk` heads. This is the ONE place `d_qk` survives the V3
        migration: it is attention's own capacity dimension, and the fitted exchange rate between it
        and RoLA's `N` is a paper row (audit §4.1 item 4). Omit `head_dims` for the single canonical
        oracle cell.
        """
        rid = arm.get("run_id", _ATTN_RUNID)
        a_lrs, _, a_lens = _axes(arm)
        head_dims = arm.get("head_dims")
        if head_dims is None:
            cells = [(C.mha, C.mha["kwargs"]["num_heads"] and bb["d_model"] // C.mha["kwargs"]["num_heads"])]
        else:
            cells = []
            for d_qk in head_dims:
                if bb["d_model"] % d_qk:
                    raise ValueError(
                        f"attention head width d_qk={d_qk} does not divide d_model="
                        f"{bb['d_model']}; zoology's MHA derives head_dim = d_model // num_heads, "
                        "so the ladder must be built from divisors.")
                cells.append(({"name": "zoology.mixers.attention.MHA",
                                   "kwargs": {"num_heads": bb["d_model"] // d_qk, "dropout": 0.1}}, d_qk))
        for seed, (ctag, cknobs) in product(seeds, _arm_conds(arm)):
            for L in a_lens:
                for kernel, d_qk in cells:
                    for lr in a_lrs:
                        add(kernel, _fmt(rid, ctag, d_qk=d_qk, lr=lr, seed=seed,
                                         d_model=bb["d_model"], L=L), lr, seed, L, ctag, cknobs)

    builders = {"routed": routed, "baselines": baselines, "attn": attn}

    tiers = spec["tiers"]
    sel = os.environ.get("GRID_TIERS", "all")
    chosen = list(tiers) if sel == "all" else [t for t in tiers if t in sel]
    for t in chosen:                       # preserve declaration order (== run order)
        for arm in tiers[t]:
            kind = arm["build"]
            if kind not in builders:
                raise ValueError(f"unknown arm builder {kind!r}; expected one of {sorted(builders)}")
            builders[kind](arm)
    if not configs:
        raise ValueError(
            f"spec {spec.get('project_name')!r} expanded to ZERO cells (GRID_TIERS={sel!r}). An "
            "empty grid is always a spec or tier-selection bug — refusing rather than launching a "
            "no-op fleet job.")
    return configs, envs


def load_spec(path):
    import yaml
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_configs_and_envs(spec_path):
    return build_configs(load_spec(spec_path))
