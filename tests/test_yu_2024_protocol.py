"""Protocol regressions for the Yu 2024 Esser spike mechanisms."""

from __future__ import annotations

from copy import deepcopy

import dendra as dn
import torch

from dendra_models.models.networks.yu_2024.mechanisms.esser import (
    esser_mech_h,
    esser_mech_s,
)
from dendra_models.models.networks.yu_2024.net import yu_2024
from dendra_models.models.networks.yu_2024.params import params as default_params


DTYPE = torch.float64
SHAPE = (3,)
VOLTAGE = torch.full(SHAPE, -65.0, dtype=DTYPE)
DT = 0.1


def _initialized(mechanism_type):
    mechanism = mechanism_type(
        mechanism_type.__name__,
        torch.tensor(37.0, dtype=DTYPE),
        torch.ones(SHAPE, dtype=DTYPE),
        SHAPE,
        SHAPE,
    ).to(dtype=DTYPE)
    mechanism._configure_timestep(DT)
    mechanism._init_buffers_s(VOLTAGE)
    return mechanism


def test_esser_runtime_storage_has_canonical_roles_and_reinitializes():
    hard = _initialized(esser_mech_h)
    soft = _initialized(esser_mech_s)

    assert hard._derived_buffers == {"factor"}
    assert hard._carry == ("spikes", "h_prev", "time_left")
    assert soft._derived_buffers == {"factor"}
    assert soft._carry == ("spikes", "g_prev")
    assert not hasattr(soft, "time_left")

    for mechanism in (hard, soft):
        state = mechanism.DE["esser_states"]
        assert state._carry == ("gspike",)
        assert "gspike" not in state.all_parameter_names()
        torch.testing.assert_close(
            mechanism.factor,
            torch.full_like(VOLTAGE, 4.0),
        )
        torch.testing.assert_close(state.gspike, torch.zeros_like(VOLTAGE))

    hard.spikes.fill_(2.0)
    hard.h_prev.fill_(0.75)
    hard.time_left.fill_(1.0)
    hard.DE["esser_states"].gspike.fill_(1.0)
    hard._init_buffers_s(VOLTAGE)

    torch.testing.assert_close(hard.spikes, torch.zeros_like(VOLTAGE))
    torch.testing.assert_close(hard.h_prev, torch.zeros_like(VOLTAGE))
    torch.testing.assert_close(hard.time_left, torch.zeros_like(VOLTAGE))
    torch.testing.assert_close(
        hard.DE["esser_states"].gspike,
        torch.zeros_like(VOLTAGE),
    )


def test_hard_esser_event_preserves_crossing_and_timer_semantics():
    mechanism = _initialized(esser_mech_h)
    state = mechanism.DE["esser_states"]
    mechanism.v_iaf.fill_(-40.0)
    mechanism.theta.fill_(-53.0)

    mechanism.net_receive(torch.zeros_like(VOLTAGE), None)

    torch.testing.assert_close(mechanism.spikes, torch.ones_like(VOLTAGE))
    torch.testing.assert_close(
        mechanism.time_left,
        mechanism.tspike.expand_as(VOLTAGE) - DT,
    )
    torch.testing.assert_close(state.gspike, torch.ones_like(VOLTAGE))
    torch.testing.assert_close(mechanism.v_iaf, state.ena_iaf.expand_as(VOLTAGE))
    torch.testing.assert_close(mechanism.theta, state.ena_iaf.expand_as(VOLTAGE))

    previous_time = mechanism.time_left.clone()
    mechanism.net_receive(torch.zeros_like(VOLTAGE), None)

    torch.testing.assert_close(mechanism.spikes, torch.zeros_like(VOLTAGE))
    torch.testing.assert_close(mechanism.time_left, previous_time - DT)
    torch.testing.assert_close(state.gspike, torch.ones_like(VOLTAGE))


def test_soft_esser_event_preserves_gate_and_conductance_semantics():
    mechanism = _initialized(esser_mech_s)
    state = mechanism.DE["esser_states"]
    mechanism.v_iaf.fill_(-40.0)
    mechanism.theta.fill_(-53.0)

    expected_gate = torch.sigmoid(
        (mechanism.v_iaf - mechanism.theta) / mechanism.tau_gate
    )
    expected_rise = expected_gate
    expected_v = mechanism.v_iaf + mechanism.alpha_peak * expected_rise * (
        state.ena_iaf - mechanism.v_iaf
    )

    mechanism.net_receive(torch.zeros_like(VOLTAGE), None)

    torch.testing.assert_close(mechanism.g_prev, expected_gate)
    torch.testing.assert_close(mechanism.spikes, expected_rise)
    torch.testing.assert_close(mechanism.v_iaf, expected_v)
    torch.testing.assert_close(state.gspike, torch.ones_like(VOLTAGE))


def test_small_yu_network_initializes_and_steps_with_declared_carry():
    params = deepcopy(default_params)
    params["diameter_column"] = 100
    params["spacing_column"] = 100
    params["cells"]["n_per_column"] = {name: 1 for name in params["cells"]["celltypes"]}
    params["netstim"]["n_per_group"] = {
        name: 1 for name in params["cells"]["celltypes"]
    }
    for post_probabilities in params["con"]["p"].values():
        for post_name in post_probabilities:
            post_probabilities[post_name] = 0.0

    with dn.ctx(JIT=0, DTYPE="float64", DEVICE="cpu"):
        network = yu_2024(params=params)
        network.initialize(DT)

        mechanism = network.cells.mech.esser_mech
        state = mechanism.DE["esser_states"]
        assert tuple(network.cells.v.shape) == (1, 6)
        torch.testing.assert_close(state.gspike, torch.zeros_like(network.cells.v))

        network.step()

        assert network.t.item() == DT
        assert torch.isfinite(network.cells.v).all()
