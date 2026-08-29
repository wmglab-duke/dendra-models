from __future__ import annotations

import dendra as dn  # noqa: F401 - configure Dendra before importing torch
import torch
from dendra.models.mechanisms._state import _materialize_derived_buffers

from dendra_models.models.cells.peripheral import Sweeney1987
from dendra_models.models.cells.peripheral.mech import ka14, km


DTYPE = torch.float64
SHAPE = (1, 3)


def _new_mechanism(mechanism_cls):
    mechanism = mechanism_cls(
        mechanism_cls.__qualname__,
        torch.tensor(37.0, dtype=DTYPE),
        torch.ones(SHAPE, dtype=DTYPE),
        SHAPE,
        SHAPE,
    ).to(dtype=DTYPE)
    for state in mechanism.DE.values():
        _materialize_derived_buffers(state)
    return mechanism


def test_sweeney_reduced_geometry_matches_resistive_internode_equivalent():
    with dn.ctx(DTYPE="float64", DEVICE="cpu", JIT=0):
        model = Sweeney1987(diameters=[5.0, 10.0], n_node=3)
        model.eval()
        model.initialize()

    fiber_diameter = torch.tensor([5.0, 10.0], dtype=model.dtype())
    node_diameter = 0.6 * fiber_diameter
    spacing = 100.0 * fiber_diameter

    torch.testing.assert_close(
        model.diam,
        node_diameter[:, None].expand_as(model.diam),
    )
    torch.testing.assert_close(model.dx, torch.full_like(model.dx, 1.5))
    torch.testing.assert_close(model.cm, torch.full_like(model.cm, 2.5))
    torch.testing.assert_close(
        model.rhoa,
        (54.7 * spacing / 1.5)[:, None].expand_as(model.rhoa),
    )
    expected_x = torch.stack(
        (-spacing, torch.zeros_like(spacing), spacing),
        dim=1,
    )
    torch.testing.assert_close(model.x, expected_x)


def test_thio_km_gate_time_constants_match_mod_translation():
    mechanism = _new_mechanism(km)
    voltage = torch.tensor([[-80.0, -58.5, -20.0]], dtype=DTYPE)
    q10 = 3.0 ** ((37.0 - 22.0) / 10.0)

    expected_taum = (
        104.0
        / (torch.exp((voltage - 7.0) / 20.0) + torch.exp(-(voltage + 32.0) / 20.0))
        + 30.0 / (1.0 + torch.exp(-(voltage + 40.0) / 80.0))
    ) / q10 / 2.0
    expected_taun = (
        15.0
        + 62.0
        / (torch.exp((voltage - 13.0) / 20.0) + torch.exp(-(voltage + 90.0) / 20.0))
        + 50.0 / (1.0 + torch.exp(-(voltage + 50.0) / 8.0))
    ) / q10 / 2.0

    torch.testing.assert_close(mechanism.DE["m"].taum(voltage), expected_taum)
    torch.testing.assert_close(mechanism.DE["n"].taun(voltage), expected_taun)


def test_thio_ka14_inactivation_matches_mod_translation():
    mechanism = _new_mechanism(ka14)
    voltage = torch.tensor([[-80.0, -58.5, -20.0]], dtype=DTYPE)
    expected = 0.073 + 0.924 / (1.0 + torch.exp((voltage + 47.0) / 4.75))

    torch.testing.assert_close(mechanism.DE["h"].hinf(voltage), expected)
