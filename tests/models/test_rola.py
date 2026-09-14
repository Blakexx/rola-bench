"""The RoLA arms registry (`rola_bench.models.rola`): every build of a cell is the same layer, built only through
fla, and its state is read back off that layer. CPU only."""
from __future__ import annotations

import pytest
import torch

from rola_bench.models import rola as cells


def _params(module):
    return dict(module.named_parameters())


@pytest.mark.parametrize("wiring", sorted(cells.WIRINGS))
def test_the_mixer_and_the_bare_layer_are_the_same_layer(wiring):
    """zoology's mixer config and the bare fla layer build bit-identical parameters from one cell."""
    from zoology.mixers.rola import RoLAMixer

    c = cells.cell(wiring, 64)
    torch.manual_seed(0)
    layer = cells.layer(c, hidden_size=128, num_heads=2, head_v_dim=64)
    torch.manual_seed(0)
    config = cells.mixer_config(c, n_heads=2, d_v=64)
    mixer = RoLAMixer(d_model=128, layer_idx=0, **config["kwargs"])
    a, b = _params(layer), _params(mixer.layer)
    assert sorted(a) == sorted(b) and all(torch.equal(a[k], b[k]) for k in a)
    assert config["name"] == "zoology.mixers.rola.RoLAMixer"


def test_a_cell_is_rolas_own_layer_with_the_wirings_levels():
    import rola

    c = cells.cell("rola-hybrid", 256)
    layer = cells.layer(c, hidden_size=128, num_heads=2, head_v_dim=64)
    assert type(layer.layer) is rola.RoLA
    assert layer.topology.widths == (16, 16)
    assert [type(level) for level in layer.routes.levels] == [rola.UnionRouting, rola.IndependentRouting]


def test_state_is_read_off_the_built_layer():
    for n in (16, 64, 256):
        layer = cells.layer(cells.cell("rola-arm2-union", n), hidden_size=128, num_heads=4, head_v_dim=64)
        assert cells.state_floats(layer) == (4 * n * 64, 4 * n)


def test_an_unknown_wiring_or_name_is_refused():
    with pytest.raises(ValueError, match="wiring"):
        cells.Cell("rola-arm9", (16,))
    with pytest.raises(ValueError, match="level"):
        cells.Cell("rola-d1-dense", (8, 8))
