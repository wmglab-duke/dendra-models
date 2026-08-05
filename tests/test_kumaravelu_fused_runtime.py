"""Regression tests for the hand-written Kumaravelu fused transition."""

from copy import deepcopy

import dendra as dn
import numpy as np

from dendra_models.models.networks.kumaravelu_2016 import (
    Kumaravelu2016,
    default_params,
)


def _small_params():
    params = default_params()
    params["n"] = 3
    params["cells"]["n_by_type"] = {
        name: 3 for name in params["cells"]["n_by_type"]
    }
    return params


def _build(params):
    dn.set_jit_enabled(False)
    with dn.ctx(DTYPE="float64", DEVICE="cpu", JIT=0, TF32=0):
        model = Kumaravelu2016(
            params,
            rng=np.random.default_rng(271828),
            N=1,
            dt=0.01,
            differentiable_spikes=False,
            synapse_discretization="euler",
            delay_mode="shift",
        )
    model.eval()
    model.initialize()
    return model


def test_fused_hidden_state_advances_one_step():
    """Dendra's proxy fast path must not replace the custom fused transition."""
    model = _build(_small_params())
    before = model.mech.kumaravelu.v_all.detach().clone()

    model.run(tstop=0.01, dt=0.01, progressbar=False)

    after = model.mech.kumaravelu.v_all.detach()
    assert not np.array_equal(after.cpu().numpy(), before.cpu().numpy())


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
        stimulated_model.mech.kumaravelu.v_all
        - control_model.mech.kumaravelu.v_all
    ).detach().cpu().numpy()[0]
    n = 3
    np.testing.assert_allclose(delta[: 6 * n], 0.0, atol=1.0e-13, rtol=0.0)
    np.testing.assert_allclose(delta[6 * n :], 0.1, atol=1.0e-13, rtol=0.0)

