"""The STN logistic rate remains differentiable at resting potentials."""

import math

import pytest
import torch

from dendra_models.models.networks.kumaravelu_2016.mechanisms import fused, hh


@pytest.mark.parametrize("rate", (fused.stn_rinf, hh.stn_rinf))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_stn_rinf_matches_original_equation_with_finite_saturated_gradients(
    rate, dtype
):
    voltage = torch.tensor(
        [-1000.0, -80.0, -62.0, -1.0, 0.0, 0.17, 1.0, 1000.0],
        dtype=dtype,
        requires_grad=True,
    )
    actual = rate(voltage)

    def logistic(v):
        x = (v - 0.17) / 0.08
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        numerator = math.exp(x)
        return numerator / (1.0 + numerator)

    expected = voltage.new_tensor([logistic(float(v)) for v in voltage.detach()])
    torch.testing.assert_close(actual, expected)
    gradient = torch.autograd.grad(actual.sum(), voltage)[0]
    assert torch.isfinite(gradient).all()
    torch.testing.assert_close(gradient, expected * (1.0 - expected) / 0.08)
