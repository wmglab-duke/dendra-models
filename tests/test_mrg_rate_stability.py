"""Regression coverage for the AXNODE_myel.mod voltage branches."""

import math

import dendra as dn
import pytest
import torch
from dendra.models.mechanisms._state import _materialize_derived_buffers

from dendra_models.models.cells.peripheral import bigMRG
from dendra_models.models.cells.peripheral.MRG.axnode_myel import axnode_myel


def _mechanism(voltage):
    with dn.ctx(DTYPE=str(voltage.dtype).split(".")[-1], DEVICE="cpu", JIT=0):
        mechanism = axnode_myel(
            "axnode_myel",
            voltage.new_tensor(37.0),
            torch.ones_like(voltage),
            voltage.shape,
            voltage.shape,
        )
        for state in mechanism.DE.values():
            _materialize_derived_buffers(state)
        return mechanism


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize(
    "gate,rate,amplitude,offset,scale,sign,extreme_rate",
    (
        ("p", "vtrap1", 0.01, 27.0, 10.2, -1, 0.00086725),
        ("p", "vtrap2", 0.00025, 34.0, 10.0, 1, 1.5855e-05),
        ("m", "vtrap6", 1.86, 21.4, 10.3, -1, 0.15733),
        ("m", "vtrap7", 0.086, 25.7, 9.16, 1, 0.0057268),
        ("h", "vtrap8", 0.062, 114.0, 11.0, 1, 0.0032594),
    ),
)
def test_mrg_vtrap_singularity_and_extreme_rate(
    dtype, gate, rate, amplitude, offset, scale, sign, extreme_rate
):
    voltage = torch.tensor(
        [
            [
                sign * 1000.0,
                sign * 150.001,
                sign * 150.0,
                -offset - 1e-3,
                -offset,
                -offset + 1e-3,
                -80.0,
                100.0,
            ]
        ],
        dtype=dtype,
        requires_grad=True,
    )
    result = getattr(_mechanism(voltage).DE[gate], rate)(voltage)

    def reference(v):
        displacement = sign * (v + offset)
        if abs(displacement / scale) < 1e-6:
            return amplitude * scale
        if sign * v > 150.0:
            return extreme_rate
        return amplitude * displacement / math.expm1(displacement / scale)

    expected = voltage.new_tensor(
        [[reference(v) for v in voltage.detach()[0].tolist()]]
    )
    torch.testing.assert_close(result, expected)
    gradient = torch.autograd.grad(result.sum(), voltage, create_graph=True)[0]
    assert torch.isfinite(gradient).all()
    curvature = torch.autograd.grad(gradient.sum(), voltage)[0]
    assert torch.isfinite(curvature).all()
    torch.testing.assert_close(
        gradient[:, [0, 1, 4]], torch.zeros_like(gradient[:, [0, 1, 4]])
    )
    assert gradient[0, 2] != 0
    assert gradient[0, 3] != 0
    assert gradient[0, 5] != 0


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_mrg_vtrap_parameter_gradients_follow_selected_branch(dtype):
    voltage = torch.tensor([[-27.0, -669.0]], dtype=dtype)
    gate = _mechanism(voltage).DE["p"]
    parameters = tuple(
        parameter.requires_grad_() for parameter in (gate.ampA, gate.ampB, gate.ampC)
    )
    singular_rate, extreme_rate = gate.vtrap1(voltage)[0]
    singular_gradient = torch.autograd.grad(
        singular_rate, parameters, retain_graph=True
    )
    expected = (gate.ampC, gate.ampB.new_zeros(()), gate.ampA)
    for actual, value in zip(singular_gradient, expected, strict=True):
        torch.testing.assert_close(actual, value)
    for actual in torch.autograd.grad(extreme_rate, parameters):
        torch.testing.assert_close(actual, torch.zeros_like(actual))


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_mrg_logistic_limits_keep_gating_finite_under_strong_fields(dtype):
    voltage = torch.tensor(
        [[-1000.0, -669.0, -491.2, -150.001, -150.0, -80.0, 150.0]],
        dtype=dtype,
        requires_grad=True,
    )
    mechanism = _mechanism(voltage)
    for gate, rate, amplitude, intercept, scale, extreme_rate in (
        ("h", "vtrap9", 2.3, 31.8, 13.4, 0.0014054),
        ("s", "vtrap10", 0.3, 53.0, 5.0, 3.3484e-05),
        ("s", "vtrap11", 0.03, 90.0, 1.0, 3.3484e-06),
    ):
        actual = getattr(mechanism.DE[gate], rate)(voltage)
        expected = voltage.new_tensor(
            [
                [
                    (
                        extreme_rate
                        if v < -150.0
                        else amplitude / (1 + math.exp(-(v + intercept) / scale))
                    )
                    for v in voltage.detach()[0].tolist()
                ]
            ]
        )
        torch.testing.assert_close(actual, expected)
        gradient = torch.autograd.grad(actual.sum(), voltage, create_graph=True)[0]
        assert torch.isfinite(gradient).all()
        curvature = torch.autograd.grad(gradient.sum(), voltage)[0]
        assert torch.isfinite(curvature).all()
        torch.testing.assert_close(gradient[:, :4], torch.zeros_like(gradient[:, :4]))

    assigned = mechanism.DE["s"].assigned_values(voltage, None)
    q10 = 3.0**0.1
    slow_alpha, slow_beta = 3.3484e-05, 3.3484e-06
    torch.testing.assert_close(
        assigned["sinf"][:, :4],
        voltage.new_full((1, 4), slow_alpha / (slow_alpha + slow_beta)),
    )
    torch.testing.assert_close(
        assigned["stau"][:, :4],
        voltage.new_full((1, 4), 1 / (q10 * (slow_alpha + slow_beta))),
    )
    gradient = torch.autograd.grad(
        sum(value.sum() for value in assigned.values()), voltage
    )[0]
    assert torch.isfinite(gradient).all()


@pytest.mark.parametrize("initial_voltage", (-27.0, -491.2))
def test_mrg_initializes_and_steps_at_problem_voltages(initial_voltage):
    with dn.ctx(DTYPE="float32", DEVICE="cpu", JIT=0):
        fiber = bigMRG(diameters=[5.7], n_node=3, v_init=initial_voltage)
        fiber.eval()
        fiber.initialize()
        assert torch.isfinite(fiber.v).all()
        fiber.longrun(tstop=0.004, dt=0.001, chunklength=4, progressbar=False)
        assert torch.isfinite(fiber.v).all()
        assert fiber.t.item() == pytest.approx(0.004)
