"""Isolated Esser voltage exposure and subthreshold dynamics stay functional.

Yu network event delivery still calls ``net_receive`` imperatively; the pure
voltage hook exposes the integrated voltage without delivering another event.
"""

from __future__ import annotations

import dendra as dn
import pytest
import torch
from torch.utils import _pytree as pytree

from dendra._bootstrap import reset_torch_compiler, torch_compiler_warning_context
from dendra_models.models.networks.yu_2024.mechanisms.esser import (
    esser_mech_h,
    esser_mech_s,
)


DTYPE = torch.float64
DT = 0.025
STEPS = 4
TYPES = (esser_mech_h, esser_mech_s)
OBSERVATIONS = ("v", "esser.v_iaf", "esser.theta", "esser.A", "esser.B")


@pytest.fixture(autouse=True)
def _reset_compiler():
    reset_torch_compiler()
    try:
        yield
    finally:
        reset_torch_compiler()


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _model(mechanism_type, batch=None):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DEVICE="cpu", DTYPE=DTYPE):
        model = dn.Population(N=1, C=3, dtype=DTYPE, integrator=dn.scnv())
        model.insert(mechanism_type.rename("esser"))
        if batch is not None:
            model.batch(batch)
        model.train()
        model.initialize()
        mechanism = model.mech.esser
        shape = model.v.shape
        mechanism._buffers["v_iaf"] = (
            torch.linspace(-76, -50, model.v.numel()).to(dtype=DTYPE).reshape(shape)
        )
        mechanism._buffers["theta"] = torch.full_like(model.v, -52.0)
        mechanism._buffers["A"] = torch.full_like(model.v, 0.02)
        mechanism._buffers["B"] = torch.full_like(model.v, 0.05)
        mechanism.DE["esser_states"]._buffers["gspike"] = torch.linspace(
            0.0, 0.4, model.v.numel(), dtype=DTYPE
        ).reshape(shape)
    return model


def _observe(model):
    return {
        "v": model.v.clone(),
        **{
            f"esser.{name}": getattr(model.mech.esser, name).clone()
            for name in ("v_iaf", "theta", "A", "B")
        },
    }


def _close(actual, expected):
    assert actual.keys() == expected.keys()
    for name in actual:
        torch.testing.assert_close(actual[name], expected[name], rtol=2e-9, atol=2e-10)


@pytest.mark.parametrize("mechanism_type", TYPES)
def test_esser_voltage_hook_uses_explicit_owned_voltage_without_mutation(
    mechanism_type,
):
    model = _model(mechanism_type)
    mechanism = model.mech.esser
    source = _clone(dict(mechanism.named_buffers()))
    exposed = torch.full_like(model.v, -43.0, requires_grad=True)
    frame = {"v_iaf": exposed}
    voltage, updates = mechanism.voltage_values(model.v, model.v.new_tensor(DT), frame)
    assert voltage is exposed
    assert updates == {}
    assert frame == {"v_iaf": exposed}
    _close(dict(mechanism.named_buffers()), source)
    torch.testing.assert_close(
        torch.autograd.grad(voltage.sum(), exposed)[0], torch.ones_like(exposed)
    )


@pytest.mark.parametrize("mechanism_type", TYPES)
def test_esser_assigned_values_preserve_bound_currents_and_use_explicit_carry(
    mechanism_type,
):
    model = _model(mechanism_type)
    mechanism = model.mech.esser
    state = mechanism.DE["esser_states"]
    voltage = torch.full_like(model.v, -62.0)
    carry = torch.full_like(voltage, 0.7, requires_grad=True)
    values = {"A": mechanism.A, "B": mechanism.B, "gspike": carry}
    absent = state.assigned_values(voltage, values)
    torch.testing.assert_close(absent["i_spike"], carry * (voltage - state.ek_iaf))
    torch.testing.assert_close(
        torch.autograd.grad(absent["i_spike"].sum(), carry)[0], voltage - state.ek_iaf
    )
    before = _clone(dict(state.named_buffers()))
    currents = {
        name: torch.full_like(voltage, index + 1.0)
        for index, name in enumerate(("i_ampa", "i_nmda", "i_gaba_a", "i_gaba_b"))
    }
    for name, current in currents.items():
        state.setreference(name, lambda current=current: current)
    bound = state.assigned_values(voltage, values)
    torch.testing.assert_close(
        bound["i_total"], absent["i_total"] + sum(currents.values())
    )
    explicit = state.assigned_values(voltage, {**values, "i_ampa": -currents["i_ampa"]})
    torch.testing.assert_close(
        explicit["i_total"], bound["i_total"] - 2 * currents["i_ampa"]
    )
    _close(dict(state.named_buffers()), before)


@pytest.mark.parametrize("mechanism_type", TYPES)
@pytest.mark.parametrize("binding", ("closure", "tensor"))
def test_esser_functionalization_rejects_imperative_current_reference(
    mechanism_type, binding
):
    model = _model(mechanism_type)
    mechanism = model.mech.esser
    state = mechanism.DE["esser_states"]
    values = {"A": mechanism.A, "B": mechanism.B, "gspike": state.gspike}
    absent = state.assigned_values(model.v, values)
    external = torch.ones_like(model.v, requires_grad=True)
    if binding == "closure":
        state.setreference("i_ampa", lambda: external)
    else:
        state.i_ampa = external
    bound = state.assigned_values(model.v, values)
    torch.testing.assert_close(bound["i_total"], absent["i_total"] + external)
    torch.testing.assert_close(
        torch.autograd.grad(bound["i_total"].sum(), external)[0],
        torch.ones_like(external),
    )
    with torch.no_grad():
        external.add_(1.0)
    rebound = state.assigned_values(model.v, values)
    torch.testing.assert_close(rebound["i_total"], absent["i_total"] + external)
    with pytest.raises(
        dn.func.FunctionalizationError, match="reference|binding|i_ampa"
    ):
        dn.func.make_functional(model, dt=DT)


def _reference(parameters, constants, state):
    """Independent cnexp recurrence with voltage frozen for each accepted step."""

    def parameter(name):
        matches = [
            value
            for key, value in parameters.items()
            if key.endswith(f".esser_states.{name}_param")
        ]
        assert len(matches) == 1
        return matches[0]

    dt = constants["dt"]
    owned = state["mechanisms"]["esser"]
    frame = {
        "v": state["integrator"]["v"],
        **{f"esser.{name}": owned[name] for name in ("v_iaf", "theta", "A", "B")},
    }
    frames = [frame]
    conductance = state["state_buffers"]["esser"]["esser_states"]["gspike"]
    for _ in range(STEPS):
        voltage, theta = frame["esser.v_iaf"], frame["esser.theta"]
        a, b = frame["esser.A"], frame["esser.B"]
        total = (
            parameter("gNa_leak") * (voltage - parameter("ena_iaf"))
            + parameter("gK_leak") * (voltage - parameter("ek_iaf"))
            - parameter("i_stim")
            + (b - a) * (voltage - parameter("e"))
        )
        spike = conductance * (voltage - parameter("ek_iaf"))
        equilibrium = parameter("theta_eq") + parameter("C") * (
            voltage - parameter("theta_eq")
        )
        frame = {
            "v": voltage,
            "esser.v_iaf": voltage
            - dt * (total / parameter("tau_m") + spike / parameter("tau_spike")),
            "esser.theta": equilibrium
            + (theta - equilibrium) * torch.exp(-dt / parameter("tau_theta")),
            "esser.A": a * torch.exp(-dt / parameter("tau1")),
            "esser.B": b * torch.exp(-dt / parameter("tau2")),
        }
        frames.append(frame)
    return {name: torch.stack([frame[name] for frame in frames]) for name in frame}


def _loss(trace):
    return sum(value.square().mean() for value in trace.values())


@pytest.mark.parametrize("mechanism_type", TYPES)
def test_esser_fresh_initialization_resets_owned_state_and_spike_carry(mechanism_type):
    model = _model(mechanism_type, batch=2)
    source = _clone(dict(model.named_buffers()))
    with torch_compiler_warning_context():
        runner = dn.compile(
            model,
            dt=DT,
            steps=STEPS,
            backend="eager",
            inference=False,
            initialization="fresh",
            callbacks={"trace": dn.func.Recorder(OBSERVATIONS)},
        )
        initialized = runner.functional.initialize(
            runner.parameters, runner.constants, runner.initialization_inputs
        )
        state, auxiliary = runner()
        reference = _reference(
            initialized.parameters, initialized.constants, initialized.state
        )
        _close(dict(auxiliary["callbacks"]["trace"]), reference)
    state_parameters = model.mech.esser.DE["esser_states"]
    # Preserve the original model's deliberately detached initial defaults.
    defaults = initialized.state["mechanisms"]["esser"]
    assert not defaults["v_iaf"].requires_grad
    assert not defaults["theta"].requires_grad
    torch.testing.assert_close(
        defaults["v_iaf"], state_parameters.v_iaf0.expand_as(model.v)
    )
    torch.testing.assert_close(
        state["state_buffers"]["esser"]["esser_states"]["gspike"],
        torch.zeros_like(model.v),
    )
    _close(dict(model.named_buffers()), source)


@pytest.mark.parametrize("mechanism_type", TYPES)
@pytest.mark.parametrize("batch", (None, 2))
def test_esser_scnv_imperative_compiled_scan_and_gradient_parity(mechanism_type, batch):
    model = _model(mechanism_type, batch)
    before = _clone(dict(model.named_buffers()))
    with torch_compiler_warning_context():
        runner = dn.compile(
            model,
            dt=DT,
            steps=STEPS,
            backend="eager",
            execution="scan",
            inference=False,
            initialization="snapshot",
            callbacks={"trace": dn.func.Recorder(OBSERVATIONS)},
        )
        parameters = _clone(dict(runner.parameters))
        state = _clone(runner.initial_state)
        targets = [
            state["mechanisms"]["esser"]["v_iaf"].requires_grad_(),
            state["state_buffers"]["esser"]["esser_states"]["gspike"].requires_grad_(),
        ]
        for name in ("gNa_leak", "tau_m", "tau_theta", "i_stim"):
            key = next(
                key for key in parameters if key.endswith(f".esser_states.{name}_param")
            )
            targets.append(parameters[key].requires_grad_())
        actual, auxiliary = runner(state, parameters=parameters)
        trace = dict(auxiliary["callbacks"]["trace"])
        reference = _reference(parameters, runner.constants, state)
        _close(trace, reference)
        for gradient, expected in zip(
            torch.autograd.grad(_loss(trace), targets),
            torch.autograd.grad(_loss(reference), targets),
            strict=True,
        ):
            torch.testing.assert_close(gradient, expected, rtol=2e-9, atol=2e-10)
    _close(dict(model.named_buffers()), before)
    dt = model.v.new_tensor(DT)
    model.integrator._initialize(model, dt, force=True, compile_scope="population")
    frames = [_observe(model)]
    for _ in range(STEPS):
        model.integrator.step(model, dt)
        model.t = model.t + dt
        frames.append(_observe(model))
    _close(
        trace,
        {name: torch.stack([frame[name] for frame in frames]) for name in OBSERVATIONS},
    )
    torch.testing.assert_close(
        actual["mechanisms"]["esser"]["v_iaf"], model.mech.esser.v_iaf
    )
    torch.testing.assert_close(
        actual["state_buffers"]["esser"]["esser_states"]["gspike"],
        model.mech.esser.DE["esser_states"].gspike,
    )
    for name, value in state["mechanism_buffers"]["esser"].items():
        torch.testing.assert_close(actual["mechanism_buffers"]["esser"][name], value)
    assert not torch.equal(
        actual["integrator"]["v"], actual["mechanisms"]["esser"]["v_iaf"]
    )
