"""THE ROLA ARMS: every RoLA cell rola-bench measures, as plain data, built only through fla.

A WIRING names the duties each routing level carries, outermost level first: a row per level with each duty's density
(`read`/`write` in {'dense', 'sparse'}), whether the duties share a projection (`tied`), and the entmax order of a
sparse duty (`alpha`). A CELL is a wiring at a state count N: every level's width (the uniform factorization b**D = N
unless widths are given), the recurrence's decay source, and the router pins. Every bench builds a cell one way --
`mixer_config` for zoology's MQAR model, `layer` for a bare fla layer, `config_kwargs` for fla's RoLAConfig --
and reads a built cell's state back off the layer (`state_floats`), never from a formula.

What rola refuses (a configuration outside its built envelope, prefill while its kernel is rebuilt, training before its
native backward) refuses through every one of these builds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

_DENSE = {"tied": False, "read": "dense", "write": "dense"}

#: The wirings. The first four are the canonical MQAR slate; the rest are the variants a grid axis or a named cell asks
#: for (the alpha-2 write split, the tied spellings, the depth hybrids).
WIRINGS: dict[str, tuple[dict, ...]] = {
    "rola-arm1-densread-sparsewrite": ({"tied": False, "read": "dense", "write": "sparse"}, _DENSE),
    "rola-arm2-union": ({"tied": False, "read": "sparse", "write": "sparse"}, _DENSE),
    "rola-arm3-levelsplit": ({"tied": False, "read": "dense", "write": "sparse"},
                             {"tied": False, "read": "sparse", "write": "dense"}),
    "rola-d1-dense": (_DENSE,),
    "rola-arm3-levelsplit-a2w": ({"tied": False, "read": "dense", "write": "sparse", "alpha": 2.0},
                                 {"tied": False, "read": "sparse", "write": "dense", "alpha": 1.5}),
    "rola-arm1-tied": ({"tied": True, "read": "dense", "write": "sparse"}, {"tied": True, "read": "dense", "write": "dense"}),
    "rola-d1-dense-tied": ({"tied": True, "read": "dense", "write": "dense"},),
    "rola-hybrid": ({"tied": False, "read": "sparse", "write": "sparse"}, {"tied": False, "read": "dense", "write": "sparse"}),
    "rola-hybrid-tiedtop": ({"tied": True, "read": "sparse", "write": "sparse"},
                            {"tied": False, "read": "dense", "write": "sparse"}),
}

CANONICAL_WIRINGS = ("rola-arm1-densread-sparsewrite", "rola-arm2-union", "rola-arm3-levelsplit", "rola-d1-dense")


class NoSpelling(ValueError):
    """A wiring has no spelling at this state count (a D-level wiring at an N with no uniform b**D = N)."""


def uniform_widths(depth: int, n: int) -> tuple[int, ...]:
    """The widths (b,) * depth with b**depth == n, or NoSpelling."""
    b = round(n ** (1.0 / depth))
    for candidate in (b - 1, b, b + 1):
        if candidate >= 2 and candidate ** depth == n:
            return (candidate,) * depth
    raise NoSpelling(f"N={n} has no uniform {depth}-level factorization b**{depth}")


@dataclass(frozen=True)
class Cell:
    """A wiring at concrete widths, with its decay and router pins."""

    wiring: str
    widths: tuple[int, ...]
    decay: dict | None = None
    router_bias: bool = True
    gain_bias_init: float | None = None
    levels: tuple[dict, ...] = field(init=False)

    def __post_init__(self) -> None:
        if self.wiring not in WIRINGS:
            raise ValueError(f"unknown RoLA wiring {self.wiring!r}; expected one of {sorted(WIRINGS)}")
        rows = WIRINGS[self.wiring]
        if len(self.widths) != len(rows):
            raise ValueError(f"{self.wiring} has {len(rows)} level(s); widths {self.widths} have {len(self.widths)}")
        object.__setattr__(self, "levels", tuple({"width": int(w), **row} for w, row in zip(self.widths, rows, strict=True)))

    @property
    def n(self) -> int:
        return math.prod(self.widths)

    def config_kwargs(self) -> dict:
        """fla.layers.RoLA's (and RoLAConfig's) routing arguments."""
        return {"levels": [dict(level) for level in self.levels], "decay": self.decay, "router_bias": self.router_bias,
                "gain_bias_init": self.gain_bias_init}


def cell(wiring: str, n: int, *, widths=None, decay: dict | None = None, router_bias: bool = True) -> Cell:
    """`wiring` at state count `n`: the given widths (which must multiply to n) or the uniform factorization."""
    depth = len(WIRINGS[wiring]) if wiring in WIRINGS else 0
    if widths is None:
        widths = (int(n),) if depth == 1 else uniform_widths(depth, int(n))
    elif math.prod(widths) != int(n):
        raise ValueError(f"widths {list(widths)} multiply to {math.prod(widths)}, not N={n}")
    return Cell(wiring, tuple(int(w) for w in widths), decay=decay, router_bias=router_bias)


def layer(c: Cell, *, hidden_size: int, num_heads: int, head_v_dim: int, layer_idx: int = 0):
    """The cell as a bare fla RoLA layer."""
    from fla.layers.rola import RoLA

    return RoLA(hidden_size=hidden_size, num_heads=num_heads, head_v_dim=head_v_dim, layer_idx=layer_idx,
                **c.config_kwargs())


def mixer_config(c: Cell, *, n_heads: int, d_v: int) -> dict:
    """The cell as zoology's sequence-mixer config (`zoology.mixers.rola.RoLAMixer` over fla)."""
    kwargs = c.config_kwargs()
    return {"name": "zoology.mixers.rola.RoLAMixer", "kwargs": {"n_heads": n_heads, "d_v": d_v, **kwargs}}


def state_floats(module) -> tuple[int, int]:
    """(content, overhead) recurrent floats per layer, read off a built RoLA layer or mixer: content is H * N * d_v,
    the matched capacity axis; overhead is the rest (the mass column), reported beside it."""
    stats = module.get_stats() if hasattr(module, "get_stats") else module.layer.get_stats()
    content = stats["n_heads"] * stats["num_chunks"] * stats["d_v"]
    return content, stats["state_floats"] - content
