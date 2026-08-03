from __future__ import annotations

import importlib

import dendra  # noqa: F401 - configure Dendra before importing torch
import pytest
import torch
from dendra.const import FARADAY, R
from dendra.models.mod import pas as dendra_pas

from dendra_models.models.cells.cortical.mech import (
    ca_hva,
    ca_lva,
    cadynamics,
    ih,
    im,
    k_t,
    kdshu2007,
    nap_et2,
    nata_t,
    nats2_t,
)
from dendra_models.models.cells.cortical.mech import pas as cortical_pas
from dendra_models.models.cells.cortical.mech import (
    sk_e2,
    skv3_1,
)
from dendra_models.models.cells.cortical.mech.pas import pas as compatibility_pas

SHAPE = (1, 3)
DTYPE = torch.float64


def _new_mechanism(mechanism_cls, temperature=37.0):
    return mechanism_cls(
        mechanism_cls.__qualname__,
        torch.tensor(temperature, dtype=DTYPE),
        torch.ones(SHAPE, dtype=DTYPE),
        SHAPE,
        SHAPE,
    ).to(dtype=DTYPE)


def _voltage(values):
    voltage = torch.as_tensor(values, dtype=DTYPE)
    if voltage.ndim == 0:
        return voltage.expand(SHAPE).clone()
    return voltage.reshape(SHAPE)


def _rates(mechanism_cls, state_name, values, temperature=37.0):
    mechanism = _new_mechanism(mechanism_cls, temperature)
    return mechanism.DE[state_name].breakpoint(_voltage(values), None)


def test_cortical_pas_reexports_dendras_canonical_mechanism():
    builder = importlib.import_module(
        "dendra_models.models.cells.cortical.L4.L4_NBC_dNAC"
    )

    assert cortical_pas is dendra_pas
    assert compatibility_pas is dendra_pas
    assert builder.pas is dendra_pas
    assert dendra_pas._range["e"] == -70.0
    assert "e" not in dendra_pas._global
    assert "e" not in dendra_pas._global_n


@pytest.mark.parametrize(
    ("mechanism_cls", "expected"),
    [
        (ca_hva, 1.0e-5),
        (ca_lva, 1.0e-5),
        (ih, 1.0e-5),
        (im, 1.0e-5),
        (nap_et2, 1.0e-5),
        (sk_e2, 1.0e-6),
        (skv3_1, 1.0e-5),
    ],
)
def test_conductance_defaults_match_the_mod_files(mechanism_cls, expected):
    mechanism = _new_mechanism(mechanism_cls)
    torch.testing.assert_close(
        mechanism.gbar,
        torch.full(SHAPE, expected, dtype=DTYPE),
    )


def test_mod_parameter_scopes_are_preserved():
    calcium = _new_mechanism(cadynamics).DE["cai"]
    for name in ("gamma", "decay", "depth", "minCai"):
        assert getattr(calcium, name).shape == SHAPE
    assert calcium.FARADAY.shape == torch.Size([])
    assert float(calcium.FARADAY) == pytest.approx(FARADAY, rel=5.0e-8)

    kd = _new_mechanism(kdshu2007)
    assert kd.ek.shape == SHAPE

    nata = _new_mechanism(nata_t)
    assert nata.DE["m"].mtau_scale.shape == torch.Size([])
    assert nata.DE["h"].htau_scale.shape == torch.Size([])

    nats = _new_mechanism(nats2_t)
    assert nats.DE["m"].mtau_scale.shape == torch.Size([])
    assert nats.DE["h"].htau_scale.shape == torch.Size([])


TEMPERATURE_CASES = [
    (ca_lva, "mh", "taum", -40.0),
    (ca_lva, "mh", "tauh", -40.0),
    (im, "m", "taum", -35.0),
    (k_t, "mh", "taum", -45.0),
    (k_t, "mh", "tauh", -45.0),
    (nap_et2, "m", "taum", -38.0),
    (nap_et2, "h", "tauh", -50.0),
    (nata_t, "m", "taum", -45.0),
    (nata_t, "h", "tauh", -70.0),
    (nats2_t, "m", "taum", -45.0),
    (nats2_t, "h", "tauh", -70.0),
]


@pytest.mark.parametrize(
    ("mechanism_cls", "state_name", "tau_name", "voltage"),
    TEMPERATURE_CASES,
)
def test_q10_mechanisms_use_the_model_temperature(
    mechanism_cls,
    state_name,
    tau_name,
    voltage,
):
    tau_34 = _rates(mechanism_cls, state_name, voltage, temperature=34.0)[tau_name]
    tau_37 = _rates(mechanism_cls, state_name, voltage, temperature=37.0)[tau_name]
    expected_ratio = 2.3 ** ((37.0 - 34.0) / 10.0)

    torch.testing.assert_close(
        tau_34 / tau_37,
        torch.full_like(tau_34, expected_ratio),
        rtol=2.0e-7,
        atol=0.0,
    )


def test_im_applies_q10_once():
    rates = _rates(im, "m", -35.0, temperature=34.0)
    q10 = 2.3 ** ((34.0 - 21.0) / 10.0)
    expected_tau = 1.0 / (2.0 * 3.3e-3 * q10)

    torch.testing.assert_close(
        rates["taum"],
        torch.full_like(rates["taum"], expected_tau),
        rtol=2.0e-7,
        atol=0.0,
    )
    torch.testing.assert_close(
        rates["minf"],
        torch.full_like(rates["minf"], 0.5),
    )


@pytest.mark.parametrize("mechanism_cls", [nata_t, nats2_t])
def test_fast_sodium_scales_only_change_tau(mechanism_cls):
    mechanism = _new_mechanism(mechanism_cls, temperature=34.0)
    voltage = _voltage(-45.0)

    for state_name, tau_name, inf_name in (
        ("m", "taum", "minf"),
        ("h", "tauh", "hinf"),
    ):
        state = mechanism.DE[state_name]
        alpha = state.alpha(voltage)
        beta = state.beta(voltage)
        rates = state.breakpoint(voltage, None)

        torch.testing.assert_close(
            rates[tau_name],
            0.4 / (alpha + beta),
        )
        torch.testing.assert_close(
            rates[inf_name],
            alpha / (alpha + beta),
        )


@pytest.mark.parametrize(
    ("mechanism_cls", "state_name", "voltages"),
    [
        (ca_hva, "mh", [-27.0, -27.0 + 1.0e-9, -40.0]),
        (nap_et2, "m", [-38.0, -38.0 + 1.0e-9, -50.0]),
        (nap_et2, "h", [-17.0, -64.4, -50.0]),
        (nata_t, "m", [-38.0, -38.0 + 1.0e-4, -50.0]),
        (nata_t, "h", [-66.0, -66.0 + 1.0e-4, -50.0]),
        (nats2_t, "m", [-32.0, -32.0 + 1.0e-9, -50.0]),
        (nats2_t, "h", [-60.0, -60.0 + 1.0e-9, -50.0]),
    ],
)
def test_rate_equations_are_finite_at_singular_voltages(
    mechanism_cls,
    state_name,
    voltages,
):
    rates = _rates(mechanism_cls, state_name, voltages, temperature=34.0)
    assert all(torch.isfinite(value).all() for value in rates.values())


def _calcium_coupled_model():
    with dendra.ctx(DTYPE=DTYPE):
        model = dendra.Population(
            N=1,
            C=3,
            v_init=torch.tensor([-75.0, -60.0, -45.0], dtype=DTYPE),
            dtype=DTYPE,
        )
        model.insert(ca_hva, gbar=0.01)
        model.insert(
            cadynamics,
            gamma=0.05,
            decay=80.0,
            depth=0.1,
            minCai=1.0e-4,
        )
        model.insert(sk_e2, gbar=1.0e-4)
        model.initialize()
    return model


def test_cadynamics_uses_step_current_not_diagnostic_current_cache():
    baseline = _calcium_coupled_model()
    with_diagnostics = _calcium_coupled_model()

    for step_index in range(3):
        distractor_v = torch.full_like(
            with_diagnostics.v,
            40.0 + 10.0 * step_index,
        )
        with_diagnostics.mech.iexp(distractor_v)
        baseline.step(dt=0.025)
        with_diagnostics.step(dt=0.025)

        torch.testing.assert_close(with_diagnostics.v, baseline.v)
        torch.testing.assert_close(
            with_diagnostics.mech.ions["ca"].cai,
            baseline.mech.ions["ca"].cai,
        )
        torch.testing.assert_close(
            with_diagnostics.mech.sk_e2.z,
            baseline.mech.sk_e2.z,
        )


def test_cortical_calcium_coupling_uses_the_implicit_accepted_flux():
    model = _calcium_coupled_model()
    ca = model.mech.ca_hva
    dynamics = model.mech.cadynamics
    calcium_state = dynamics.DE["cai"]
    sk = model.mech.sk_e2
    calcium = model.mech.ions["ca"]
    potassium = model.mech.ions["k"]

    # Move voltage and calcium away from their initialization points so every
    # gate/state update is observable in this one-step oracle.
    model.v = _voltage([-55.0, -40.0, -25.0])
    custom_cai = _voltage([2.0e-4, 5.0e-4, 9.0e-4])
    calcium._buffers["cai"] = custom_cai.clone()
    dynamics.cai = custom_cai.clone()
    calcium.advance(model.celsius)
    model.mech.read_from_ions()

    dt = torch.as_tensor(0.025, dtype=DTYPE)
    v0 = model.v.detach().clone()
    cai0 = calcium.cai.detach().clone()
    eca0 = calcium.eca.detach().clone()
    ek0 = potassium.ek.detach().clone()
    m0 = ca.m.detach().clone()
    h0 = ca.h.detach().clone()
    z0 = sk.z.detach().clone()

    m_alpha = 0.055 * (-(v0 + 27.0)) / (torch.exp((-(v0 + 27.0)) / 3.8) - 1.0)
    m_beta = 0.94 * torch.exp((-75.0 - v0) / 17.0)
    m_inf = m_alpha / (m_alpha + m_beta)
    m_tau = 1.0 / (m_alpha + m_beta)
    expected_m = m_inf + (m0 - m_inf) * torch.exp(-dt / m_tau)

    h_alpha = 0.000457 * torch.exp((-13.0 - v0) / 50.0)
    h_beta = 0.0065 / (torch.exp((-v0 - 15.0) / 28.0) + 1.0)
    h_inf = h_alpha / (h_alpha + h_beta)
    h_tau = 1.0 / (h_alpha + h_beta)
    expected_h = h_inf + (h0 - h_inf) * torch.exp(-dt / h_tau)

    safe_cai = torch.where(cai0 < 1.0e-7, cai0 + 1.0e-7, cai0)
    z_inf = 1.0 / (1.0 + (0.00043 / safe_cai) ** 4.8)
    expected_z = z_inf + (z0 - z_inf) * torch.exp(-dt / sk.DE["z"].ztau)

    gca = ca.gbar * expected_m**2 * expected_h
    gsk = sk.gbar * expected_z
    cmdt = 1.0e-3 * model.cm * model.cm_scale / dt
    expected_v = (cmdt * v0 + gca * eca0 + gsk * ek0) / (cmdt + gca + gsk)
    expected_ica = gca * (expected_v - eca0)
    expected_ik = gsk * (expected_v - ek0)

    shell = -10_000.0 * (
        calcium_state.gamma / (2.0 * calcium_state.FARADAY * calcium_state.depth)
    )
    cai_inf = calcium_state.minCai + calcium_state.decay * shell * expected_ica
    expected_cai = cai_inf + (cai0 - cai_inf) * torch.exp(-dt / calcium_state.decay)
    expected_eca = (
        torch.log(calcium.cao / expected_cai)
        * (R / (2.0 * FARADAY))
        * (273.15 + model.celsius)
    )

    model.step(dt=float(dt))

    torch.testing.assert_close(ca.m, expected_m)
    torch.testing.assert_close(ca.h, expected_h)
    torch.testing.assert_close(sk.z, expected_z)
    torch.testing.assert_close(model.v, expected_v)
    torch.testing.assert_close(calcium.ica, expected_ica)
    torch.testing.assert_close(dynamics.ica, expected_ica)
    torch.testing.assert_close(calcium_state.ica, expected_ica)
    torch.testing.assert_close(potassium.ik, expected_ik)
    torch.testing.assert_close(calcium.cai, expected_cai)
    torch.testing.assert_close(dynamics.cai, expected_cai)
    torch.testing.assert_close(calcium.eca, expected_eca)
    torch.testing.assert_close(ca.eca, expected_eca)
