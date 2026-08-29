"""Regression tests for the hand-written Kumaravelu fused transition."""

from copy import deepcopy

import dendra as dn
import numpy as np
import pytest
import torch

from dendra_models.models.networks.kumaravelu_2016 import (
    Kumaravelu2016,
    default_params,
)


def _small_params():
    params = default_params()
    params["n"] = 3
    params["cells"]["n_by_type"] = {name: 3 for name in params["cells"]["n_by_type"]}
    return params


def _build(
    params,
    *,
    dt=0.01,
    synapse_discretization="euler",
    synapse_update_mode="uncoalesced",
    delay_mode="shift",
    batch=None,
):
    dn.set_jit_enabled(False)
    with dn.ctx(DTYPE="float64", DEVICE="cpu", JIT=0, TF32=0):
        model = Kumaravelu2016(
            params,
            rng=np.random.default_rng(271828),
            N=1,
            dt=dt,
            differentiable_spikes=False,
            synapse_discretization=synapse_discretization,
            synapse_update_mode=synapse_update_mode,
            delay_mode=delay_mode,
        )
    if batch is not None:
        model.batch(batch)
    model.eval()
    model.initialize()
    return model


def _configure_timestep(model, dt, *, requires_grad=False):
    dt = torch.tensor(
        dt,
        device=model.device(),
        dtype=model.dtype(),
        requires_grad=requires_grad,
    )
    model.mech.set_dt(dt)
    return dt


def test_fused_hidden_state_advances_one_step():
    """Dendra's proxy fast path must not replace the custom fused transition."""
    model = _build(_small_params())
    before = model.mech.kumaravelu.v_all.detach().clone()

    model.run(tstop=0.01, dt=0.01, progressbar=False)

    after = model.mech.kumaravelu.v_all.detach()
    assert not np.array_equal(after.cpu().numpy(), before.cpu().numpy())


def test_fused_protocol_has_explicit_frozen_schemas_and_pure_builders():
    model = _build(_small_params())
    mechanism = model.mech.kumaravelu

    assert mechanism._assigned == ("i_inj",)
    assert "i_inj" not in mechanism._carry
    assert set(mechanism.initial_values(model.v, {})) == set(mechanism._carry)
    assert set(mechanism.derive_buffers()) == set(mechanism._derived_buffers)
    assert all(shape is not None for shape in mechanism._carry_resolved_shapes.values())
    assert all(
        shape is not None
        for shape in mechanism._derived_resolved_shapes.values()
    )

    assert mechanism.buf_pathway_delays.ndim == model.v.ndim + 2
    assert mechanism.buf_pathway_delays_ptr.shape == ()
    assert mechanism.buf_pathway_delays_ptr.dtype == torch.long
    assert mechanism.pathway_delays_delay_steps.shape == (12,)
    assert mechanism.pathway_delays_delay_steps.dtype == torch.long

    registered = {
        name: value
        for name, value in mechanism._buffers.items()
        if name in {*mechanism._carry, *mechanism._derived_buffers}
    }
    mechanism.initial_values(model.v, {})
    mechanism.derive_buffers()
    assert all(mechanism._buffers[name] is value for name, value in registered.items())


def test_fused_public_advance_matches_framework_commit_without_shift_side_effects():
    model = _build(_small_params())
    mechanism = model.mech.kumaravelu
    mechanism._evaluate_assigned(mechanism.v_all)
    values = mechanism._runtime_value_frame()
    before = {name: mechanism._buffers[name].clone() for name in mechanism._carry}

    expected = mechanism.advance(mechanism.v_all, mechanism.dt, values)

    for name, value in before.items():
        torch.testing.assert_close(
            mechanism._buffers[name], value, rtol=0.0, atol=0.0
        )

    mechanism._advance_states(mechanism.v_all, mechanism.dt)
    for name, value in expected.items():
        torch.testing.assert_close(
            mechanism._buffers[name], value, rtol=0.0, atol=0.0
        )


@pytest.mark.parametrize("delay_mode", ["shift", "circular_eager"])
def test_fused_declared_carry_round_trips_through_handler_checkpoint(delay_mode):
    model = _build(_small_params(), delay_mode=delay_mode)
    model.run(tstop=0.02, dt=0.01, progressbar=False)
    mechanism = model.mech.kumaravelu
    checkpoint = model.mech.mutable_state_dict()

    for name in mechanism._carry:
        value = mechanism._buffers[name]
        if value.is_floating_point():
            mechanism._buffers[name] = value + 1.0
        else:
            mechanism._buffers[name] = value + 1

    model.mech.restore_mutable_state_dict(checkpoint)
    for name in mechanism._carry:
        torch.testing.assert_close(
            mechanism._buffers[name],
            checkpoint[f"kumaravelu.{name}"],
            rtol=0.0,
            atol=0.0,
        )


def test_exact_sampled_cortical_current_uses_matlab_step_index():
    """MATLAB sample 2 drives the transition from t=0 to t=dt."""
    base = _small_params()
    base["stim_samples"] = {
        "Idbs": np.zeros(2, dtype=float),
        "Iappco": np.zeros(2, dtype=float),
    }
    stimulated = deepcopy(base)
    stimulated["stim_samples"]["Iappco"][1] = 10.0

    control_model = _build(base)
    stimulated_model = _build(stimulated)
    control_model.run(tstop=0.01, dt=0.01, progressbar=False)
    stimulated_model.run(tstop=0.01, dt=0.01, progressbar=False)

    delta = (
        (stimulated_model.mech.kumaravelu.v_all - control_model.mech.kumaravelu.v_all)
        .detach()
        .cpu()
        .numpy()[0]
    )
    n = 3
    np.testing.assert_allclose(delta[: 6 * n], 0.0, atol=1.0e-13, rtol=0.0)
    np.testing.assert_allclose(delta[6 * n :], 0.1, atol=1.0e-13, rtol=0.0)


@pytest.mark.parametrize("dt", [0.01, 0.025])
def test_synapse_timestep_buffers_have_exact_shapes_values_and_gradients(dt):
    model = _build(
        _small_params(),
        dt=dt,
        synapse_discretization="exact",
        synapse_update_mode="coalesced",
    )
    dt_tensor = _configure_timestep(model, dt, requires_grad=True)
    mechanism = model.mech.kumaravelu

    assert mechanism._alpha_dt.shape == ()
    assert mechanism._alpha_h.shape == ()
    assert mechanism._alpha_decay.shape == ()
    assert mechanism._alpha_dt_over_tau2.shape == ()
    assert mechanism.alpha_const_streams.shape == (1, 11, 1)
    for name in (
        "exp2_tau1_streams",
        "exp2_tau2_streams",
        "exp2_inc_streams",
        "exp2_euler_decay1_streams",
        "exp2_euler_decay2_streams",
        "exp2_be_decay1_streams",
        "exp2_be_decay2_streams",
        "exp2_exact_decay1_streams",
        "exp2_exact_decay2_streams",
    ):
        value = getattr(mechanism, name)
        assert value.shape == (1, 5, 1)
        assert value.dtype == model.dtype()
        assert value.device == model.device()

    tau_alpha = float(mechanism.cfg["syn"]["tau_alpha"])
    torch.testing.assert_close(mechanism._alpha_dt, dt_tensor)
    torch.testing.assert_close(mechanism._alpha_h, dt_tensor / tau_alpha)
    torch.testing.assert_close(
        mechanism._alpha_decay,
        torch.exp(-dt_tensor / tau_alpha),
    )
    torch.testing.assert_close(
        mechanism._alpha_dt_over_tau2,
        dt_tensor / (tau_alpha * tau_alpha),
    )
    torch.testing.assert_close(
        mechanism.exp2_exact_decay1_streams,
        torch.exp(-dt_tensor / mechanism.exp2_tau1_streams),
    )
    torch.testing.assert_close(
        mechanism.exp2_exact_decay2_streams,
        torch.exp(-dt_tensor / mechanism.exp2_tau2_streams),
    )

    loss = (
        mechanism._alpha_decay
        + mechanism.exp2_euler_decay1_streams.sum()
        + mechanism.exp2_be_decay2_streams.sum()
        + mechanism.exp2_exact_decay1_streams.sum()
    )
    (gradient,) = torch.autograd.grad(loss, dt_tensor)
    assert torch.isfinite(gradient)
    assert gradient.abs() > 0.0


def test_runtime_timestep_must_match_fixed_delay_queue_timestep():
    model = _build(
        _small_params(),
        synapse_discretization="exact",
        synapse_update_mode="coalesced",
    )
    mechanism = model.mech.kumaravelu
    _configure_timestep(model, 0.01)
    original_dt = mechanism.dt.clone()
    original_decay = mechanism.exp2_exact_decay1_streams.clone()

    with pytest.raises(ValueError, match="fixed timestep.*delay queues.*Rebuild"):
        _configure_timestep(model, 0.025)

    torch.testing.assert_close(mechanism.dt, original_dt, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        mechanism.exp2_exact_decay1_streams,
        original_decay,
        rtol=0.0,
        atol=0.0,
    )


def test_structural_timestep_buffers_survive_batch_and_strict_state_dict_load():
    source = _build(
        _small_params(),
        synapse_discretization="exact",
        synapse_update_mode="coalesced",
        batch=2,
    )
    _configure_timestep(source, 0.01)
    mechanism = source.mech.kumaravelu
    assert source.shape[0] == 2
    assert mechanism.alpha_const_streams.shape == (1, 11, 1)
    assert mechanism.exp2_exact_decay1_streams.shape == (1, 5, 1)

    target = _build(
        _small_params(),
        synapse_discretization="exact",
        synapse_update_mode="coalesced",
        batch=2,
    )
    target.load_state_dict(source.state_dict(), strict=True)
    target_mechanism = target.mech.kumaravelu
    for name in mechanism._timestep_buffers:
        torch.testing.assert_close(
            getattr(target_mechanism, name),
            getattr(mechanism, name),
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize("discretization", ["euler", "backward_euler", "exact"])
def test_coalesced_and_uncoalesced_synapse_updates_match(discretization):
    params = _small_params()
    uncoalesced = _build(
        deepcopy(params),
        synapse_discretization=discretization,
        synapse_update_mode="uncoalesced",
    )
    coalesced = _build(
        deepcopy(params),
        synapse_discretization=discretization,
        synapse_update_mode="coalesced",
    )

    uncoalesced.run(tstop=0.1, dt=0.01, progressbar=False)
    coalesced.run(tstop=0.1, dt=0.01, progressbar=False)

    for name in (
        "v_all",
        "S1a",
        "Z1a",
        "S2a",
        "A_stn_gpe_a",
        "B_stn_gpe_a",
    ):
        torch.testing.assert_close(
            getattr(coalesced.mech.kumaravelu, name),
            getattr(uncoalesced.mech.kumaravelu, name),
            rtol=1.0e-13,
            atol=1.0e-13,
        )


def test_exact_uncoalesced_filter_helpers_use_only_prepared_coefficients(monkeypatch):
    model = _build(
        _small_params(),
        synapse_discretization="exact",
        synapse_update_mode="uncoalesced",
    )
    _configure_timestep(model, 0.01)
    mechanism = model.mech.kumaravelu
    ref = mechanism.S1a
    zeros = torch.zeros_like(ref)
    ones = torch.ones_like(ref)

    def unexpected_transcendental(*_args, **_kwargs):
        raise AssertionError("exact filter helper recomputed a prepared coefficient")

    monkeypatch.setattr(torch, "exp", unexpected_transcendental)
    monkeypatch.setattr(torch, "log", unexpected_transcendental)
    mechanism._alpha_step(
        zeros,
        zeros,
        ones,
        float(mechanism.cfg["syn"]["gpeak"]),
        float(mechanism.cfg["syn"]["tau_alpha"]),
        mechanism.dt,
    )
    mechanism._exp2_step(
        zeros,
        zeros,
        ones,
        float(mechanism.cfg["syn"]["gpeak"]),
        0.4,
        2.5,
        mechanism.dt,
    )
