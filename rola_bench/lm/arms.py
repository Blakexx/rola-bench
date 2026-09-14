"""LM arms: the architectures of the LM comparison on one shared backbone, so only the token mixer differs.

Every arm is an fla HF model -- RoLA (fla's RoLA over rola), GatedDeltaNet, GLA and the transformer -- with
identical backbone fields (hidden size, depth, SwiGLU MLP ratio, RMSNorm, untied embeddings). The bounded-state arms
are matched on recurrent state at the textbook head shape, d_k = d_v = 128 at H = 8 on hidden 512:

    RoLA  H=8, N=128 (widths 8, 16), d_v=128   ->  8 * 128 * 128 = 131,072
    GDN   H=8, head_dim=128, expand_v=1        ->  8 * 128 * 128 = 131,072
    GLA   H=8, expand_k=2, expand_v=2          ->  1024 * 1024 / 8 = 131,072

Softmax attention is the unbounded-state ceiling. RoLA's state is read back off a layer built at the arm's geometry
(`rola_bench.models.rola.state_floats`), never computed here. The geometry comes from the box environment (`LM_*`),
which `rola_bench.lm.job` flattens from the experiment spec.
"""
from __future__ import annotations

import math
import os

import fla.models  # noqa: F401 -- registers fla's architectures with the HF auto-classes
from fla.models import GatedDeltaNetConfig, GLAConfig, RoLAConfig, TransformerConfig
from transformers import AutoModelForCausalLM

from rola_bench.models import rola as cells

ROLA_WIDTHS = (8, 16)


def geometry(env=os.environ) -> dict:
    """The arms' geometry from a box environment (`LM_*`, what `rola_bench.lm.job` flattens from a spec)."""
    def get(name, default, kind=int):
        return kind(env.get(name, default))

    hidden = get("LM_HIDDEN", 512)
    return {
        "backbone": {"hidden_size": hidden, "num_hidden_layers": get("LM_LAYERS", 12), "hidden_ratio": 4, "norm_eps": 1e-6,
                      "max_position_embeddings": get("LM_MAX_POS", 2048), "tie_word_embeddings": False,
                      "fuse_cross_entropy": True},
        "rola": {"num_heads": get("LM_ROLA_NH", 8), "head_v_dim": get("LM_ROLA_DV", 128)},
        "gdn": {"num_heads": get("LM_GDN_NH", 8), "head_dim": get("LM_GDN_HEAD_DIM", 128),
                 "expand_v": get("LM_GDN_EXPAND_V", 1.0, float)},
        "gla": {"num_heads": get("LM_GLA_NH", 8), "expand_k": get("LM_GLA_EK", 2.0, float),
                 "expand_v": get("LM_GLA_EV", 2.0, float), "gate_logit_normalizer": 16},
        "attn_heads": get("LM_ATTN_HEADS", 8),
        "qk_norm": str(env.get("LM_QK_NORM", "0")).lower() in ("1", "true", "yes"),
    }


GEOMETRY = geometry()
ROLA_ARMS = tuple(cells.NAMED)
ARMS = ROLA_ARMS + ("gdn", "gla", "attn")


def rola_cell(arm: str) -> cells.Cell:
    return cells.named(arm, math.prod(ROLA_WIDTHS), widths=ROLA_WIDTHS)


def arm_config(arm: str, vocab_size: int, bos_token_id: int = 50256, eos_token_id: int = 50256, g: dict = GEOMETRY):
    """The HF config of `arm` on the shared backbone."""
    common = dict(vocab_size=vocab_size, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **g["backbone"])
    if arm in ROLA_ARMS:
        return RoLAConfig(**g["rola"], **rola_cell(arm).config_kwargs(), **common)
    if arm == "gdn":
        return GatedDeltaNetConfig(use_short_conv=False, use_gate=True, **g["gdn"], **common)
    if arm == "gla":
        return GLAConfig(use_short_conv=False, **g["gla"], **common)
    if arm == "attn":
        return TransformerConfig(num_heads=g["attn_heads"], window_size=None, qk_norm=g["qk_norm"], **common)
    raise ValueError(f"unknown LM arm {arm!r}; expected one of {ARMS}")


def build_arm(arm: str, vocab_size: int, **kw):
    """A fresh model for `arm`."""
    return AutoModelForCausalLM.from_config(arm_config(arm, vocab_size, **kw))


def recurrent_state_floats(arm: str, g: dict = GEOMETRY) -> tuple[int | None, int | None]:
    """(content, overhead) recurrent floats per layer at the arm's geometry; (None, None) for attention (unbounded).
    RoLA's is read off a layer built at that geometry; GDN's and GLA's are their own published definitions."""
    hidden = g["backbone"]["hidden_size"]
    if arm in ROLA_ARMS:
        return cells.state_floats(cells.layer(rola_cell(arm), hidden_size=hidden, **g["rola"]))
    if arm == "gdn":
        gdn = g["gdn"]
        return gdn["num_heads"] * gdn["head_dim"] * int(gdn["head_dim"] * gdn["expand_v"]), 0
    if arm == "gla":
        gla = g["gla"]
        return int(hidden * gla["expand_k"]) * int(hidden * gla["expand_v"]) // gla["num_heads"], 0
    if arm == "attn":
        return None, None
    raise ValueError(f"unknown LM arm {arm!r}; expected one of {ARMS}")


def realized_state_floats_from_model(model) -> tuple[int | None, int | None]:
    """(content, overhead) per layer read off a built model's first RoLA layer; (None, None) for a model without one."""
    from fla.layers.rola import RoLA

    layer = next((m for m in model.modules() if isinstance(m, RoLA)), None)
    return (None, None) if layer is None else cells.state_floats(layer)


def decay_parameter_floats(model) -> int:
    """The decay source's own parameters per layer, read off a built model's first RoLA layer (0 without decay)."""
    from fla.layers.rola import RoLA

    layer = next((m for m in model.modules() if isinstance(m, RoLA)), None)
    decay = None if layer is None else layer.layer.decay
    return 0 if decay is None else sum(p.numel() for p in decay.parameters())
