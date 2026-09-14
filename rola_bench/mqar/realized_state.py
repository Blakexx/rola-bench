"""REALIZED recurrent state, read back off a constructed module. The MQAR grid's capacity axis.

**The rule this module exists to enforce (ruling, 2026-08-02, from the MQAR retrospective).** The
capacity coordinate of every cell is a READBACK, never a formula and never the trainer's logged
`model_state_total` (measured unreliable). It is obtained by constructing the cell's mixer on CPU
and asking the realized projections how wide they are.

Two independent incidents motivate the rule, and they are the same bug:
  * finding F1 (LM tier): `H * N * d_qk * d_v`, the pre-V3 matrix-state formula, published a state
    16x too large for a year because nothing read the number back.
  * the MQAR `REF(nc) = 624*nc` constant (Phase 1.2): `624 = H*d_qk*(d_v+1)` — again written in the
    deleted `d_qk`, over-sizing every matched baseline by 12x at the same rung.

A formula cannot notice an architecture change. A readback fails loudly at construction.

`state_floats(kernel_cfg, d_model)` returns `(content, overhead)` for any arm the grid builds:

  * content  = the recurrent floats that hold retrievable content. This is the matched axis.
  * overhead = normalization/denominator sidecars and short-conv caches, reported separately and
    never placed on the matched axis (#126).

Attention returns `content = None`: its state is unbounded (a growing KV cache), so it has no
coordinate on this axis at all. That is a property of the arm, not a missing measurement, and the
caller must render it as such rather than substituting a number.
"""
from __future__ import annotations

import importlib

_CACHE: dict[tuple, tuple] = {}


def _instantiate(kernel_cfg, d_model):
    module_path, cls_name = kernel_cfg["name"].rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), cls_name)
    kwargs = dict(kernel_cfg["kwargs"])
    if "hidden_size" not in kwargs and "d_model" not in kwargs:
        kwargs["d_model"] = d_model
    kwargs.setdefault("layer_idx", 0)
    return cls(**kwargs)


def _readback(name: str, module) -> tuple[int | None, int]:
    """(content, overhead) from the CONSTRUCTED module's own realized dimensions."""
    if name.endswith("RoLAMixer"):
        from rola_bench.models.rola import state_floats

        return state_floats(module)
    if name.endswith("LinearAttention"):        # vanilla-LA rung: H * k_head * v_head
        return module.key_dim * module.value_dim // module.num_heads, 0
    if name.endswith("GatedLinearAttention"):
        return module.key_dim * module.value_dim // module.num_heads, 0
    if name.endswith("GatedDeltaNet"):
        v_head = module.value_dim // module.num_heads
        return module.num_heads * module.head_dim * v_head, 0
    if name.endswith("MHA"):
        return None, 0                          # unbounded: a growing KV cache has no fixed rung
    raise ValueError(
        f"no realized-state readback is defined for {name!r}. Add one that reads the CONSTRUCTED "
        "module's own dimensions — do not add a formula, and do not fall back to the nominal "
        "target: that substitution is exactly the F1 / REF bug class this module closes.")


def state_floats(kernel_cfg, d_model: int) -> tuple[int | None, int]:
    """(content, overhead) recurrent floats per layer for one arm's kernel config.

    Memoized on the config, because a grid asks the same question for every (lr, seed) of a cell.
    Construction failures propagate: a cell that cannot be built has no capacity coordinate, and
    guessing one would publish a comparison against a model that does not exist.
    """
    key = (kernel_cfg["name"], repr(sorted(kernel_cfg["kwargs"].items(), key=lambda kv: kv[0])),
           d_model)
    if key not in _CACHE:
        _CACHE[key] = _readback(kernel_cfg["name"], _instantiate(kernel_cfg, d_model))
    return _CACHE[key]
