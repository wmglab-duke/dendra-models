"""Isolated Izhikevich voltage resets preserve the published event equations."""

from __future__ import annotations

import dendra as dn
import pytest
import torch
from dendra._bootstrap import reset_torch_compiler
from torch.utils import _pytree as pytree

from dendra_models.models.networks.kumaravelu_2016.mechanisms.izhikevich import (
    fast_spiking_interneuron,
    regular_spiking_cortex,
)

KINDS = (regular_spiking_cortex, fast_spiking_interneuron)
DT = 0.01
STEPS = 5


@pytest.fixture(autouse=True)
def _reset_compiler():
    reset_torch_compiler()
    try:
        yield
    finally:
        reset_torch_compiler()


def _model(kind, *, batch=None, ic=None, **parameters):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, IMEM=False):
        model = dn.Population(
            N=1, C=6, dtype=torch.float64, integrator=dn.scnv(imem=False)
        )
        model.insert(kind.rename("izh"), i_stim=8.0, ic=ic, **parameters)
        if batch is not None:
            model.batch(batch)
        model.train()
        model.initialize()
    return model


def _clone(tree):
    return pytree.tree_map_only(torch.Tensor, lambda x: x.detach().clone(), tree)


def _assert_close(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for name in actual:
            _assert_close(actual[name], expected[name])
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-10, atol=2e-11)


def _buffers(model):
    return {name: value.detach().clone() for name, value in model.named_buffers()}


def _assert_buffers(model, before):
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0, msg=name)


def _original_reset(v, u, previous, parameters):
    """Original hard reset/crossing and intentional surrogate backward rules."""
    reset = v >= parameters["v_peak"]
    crossed = (previous < parameters["syn_threshold"]) & (
        v >= parameters["syn_threshold"]
    )
    tau = parameters["tau_gate"].clamp_min(1e-6)
    reset_gate = torch.sigmoid((v - parameters["v_peak"]) / tau)
    current_gate = torch.sigmoid((v - parameters["syn_threshold"]) / tau)
    previous_gate = torch.sigmoid((previous - parameters["syn_threshold"]) / tau)
    crossing_gate = torch.relu(current_gate - previous_gate)
    voltage = torch.where(reset, torch.zeros_like(v) + parameters["c"], v)
    return voltage, {
        "v_izh": voltage,
        "u": torch.where(reset, u + parameters["d"], u),
        "v_prev": voltage,
        "spikes": reset.to(v.dtype)
        + parameters["ste_scale"] * (reset_gate - reset_gate.detach()),
        "syn_spikes": crossed.to(v.dtype)
        + parameters["ste_scale"] * (crossing_gate - crossing_gate.detach()),
    }


def _loss(values):
    return (
        0.001 * values["v_izh"].square().sum()
        + 0.01 * values["u"].square().sum()
        + 0.002 * values["v_prev"].sum()
        + 0.3 * values["spikes"].sum()
        + 0.7 * values["syn_spikes"].sum()
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("tau_gate", (0.0, 0.5))
def test_voltage_values_matches_original_boundary_values_gradients_and_is_pure(
    kind, tau_gate
):
    model = _model(kind, tau_gate=tau_gate)
    mechanism = model.mech.izh
    voltage = model.v.new_tensor([[-10.001, -10.0, -9.999, 29.999, 30.0, 30.001]])
    previous = model.v.new_tensor([[-10.0, -10.001, -11.0, -10.0, -11.0, -11.0]])
    initial = {
        "v_izh": voltage,
        "u": model.v.new_tensor([[-13.0, -12.0, -11.0, -10.0, -9.0, -8.0]]),
        "v_prev": previous,
    }
    # The transition must take its State and carry from the explicit frame.
    mechanism.v_izh = torch.full_like(voltage, -70.0)
    mechanism.u = torch.ones_like(voltage)
    mechanism.v_prev = torch.full_like(voltage, 80.0)
    live = _buffers(model)
    values = {name: value.clone().requires_grad_() for name, value in initial.items()}
    source = _clone(values)
    expected_values = {
        name: value.clone().requires_grad_() for name, value in initial.items()
    }
    names = ("v_peak", "syn_threshold", "c", "d", "tau_gate", "ste_scale")
    parameters = {name: getattr(mechanism, name) for name in names}
    raw = tuple(
        getattr(
            mechanism, name if name in {"tau_gate", "ste_scale"} else name + "_param"
        )
        for name in names
    )
    actual_v, actual = mechanism.voltage_values(model.v, DT, values)
    expected_v, expected = _original_reset(
        expected_values["v_izh"],
        expected_values["u"],
        expected_values["v_prev"],
        parameters,
    )
    torch.testing.assert_close(actual_v, expected_v, rtol=0, atol=0)
    _assert_close(actual, expected)
    torch.testing.assert_close(
        actual["spikes"], voltage.new_tensor([[0, 0, 0, 0, 1, 1]]), rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual["syn_spikes"], voltage.new_tensor([[0, 1, 1, 0, 1, 1]]), rtol=0, atol=0
    )
    actual_gradients = torch.autograd.grad(
        _loss(actual), (*values.values(), *raw), allow_unused=True, retain_graph=True
    )
    expected_gradients = torch.autograd.grad(
        _loss(expected), (*expected_values.values(), *raw), allow_unused=True
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        assert (actual_gradient is None) == (expected_gradient is None)
        if actual_gradient is not None:
            torch.testing.assert_close(actual_gradient, expected_gradient)
            assert torch.isfinite(actual_gradient).all()
    _assert_close(values, source)
    _assert_buffers(model, live)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("binding", ("reference", "tensor"))
def test_state_currents_prefer_explicit_values_and_preserve_live_network_bindings(
    kind, binding
):
    model = _model(kind)
    state = model.mech.izh.DE[model.mech.izh._state_name]
    currents = ("i_ie", "i_thcor") if kind is regular_spiking_cortex else ("i_ei",)
    live = {
        name: model.v.new_full(model.shape, 2.0 + index, requires_grad=True)
        for index, name in enumerate(currents)
    }
    for name in currents:
        if binding == "reference":
            state.setreference(name, lambda name=name: live[name])
        else:
            setattr(state, name, live[name])
    values = {"v_izh": model.mech.izh.v_izh, "u": model.mech.izh.u}
    bound = state.assigned_values(model.v, values)
    torch.testing.assert_close(bound["i_syn"], sum(live.values()))
    # Both legacy binding styles remain available to imperative network execution.
    live[currents[0]] = live[currents[0]] + 1.0
    if binding == "tensor":
        setattr(state, currents[0], live[currents[0]])
    rebound = state.assigned_values(model.v, values)
    torch.testing.assert_close(rebound["i_syn"], sum(live.values()))
    explicit = {name: torch.full_like(model.v, 0.25) for name in currents}
    assigned = state.assigned_values(model.v, {**values, **explicit})
    torch.testing.assert_close(assigned["i_syn"], sum(explicit.values()))
    with pytest.raises(dn.func.FunctionalizationError):
        dn.func.make_functional(model, dt=DT)


def _seed(model):
    mechanism = model.mech.izh
    voltage = model.v.new_tensor([[-10.001, -10.0, -9.999, 29.999, 30.0, 30.001]])
    previous = model.v.new_tensor([[-10.0, -10.001, -11.0, -10.0, -11.0, -11.0]])
    mechanism.v_izh = voltage.expand_as(model.v).clone()
    mechanism.v_prev = previous.expand_as(model.v).clone()
    mechanism.u = torch.linspace(-13, -8, model.v.numel(), dtype=model.dtype()).reshape(
        model.shape
    )


def _parameter(parameters, suffix):
    names = [name for name in parameters if name.endswith(suffix)]
    assert len(names) == 1
    return parameters[names[0]]


def _recurrence(state, parameters, dt, steps):
    state = pytree.tree_map(lambda x: x, state)
    reset_parameters = {
        name: _parameter(
            parameters,
            ".izh." + (name if name in {"tau_gate", "ste_scale"} else name + "_param"),
        )
        for name in ("v_peak", "syn_threshold", "c", "d", "tau_gate", "ste_scale")
    }
    a = _parameter(parameters, ".a_param")
    b = _parameter(parameters, ".b_param")
    drive = _parameter(parameters, ".i_stim_param")
    for _ in range(steps):
        owned = state["mechanisms"]["izh"]
        carry = state["mechanism_buffers"]["izh"]
        voltage, update = _original_reset(
            owned["v_izh"], owned["u"], carry["v_prev"], reset_parameters
        )
        u = update["u"]
        rhs = 0.04 * voltage.square() + 5 * voltage + 140 - u + drive
        state = {
            **state,
            "integrator": {"v": voltage},
            "mechanisms": {
                "izh": {
                    "v_izh": voltage + dt * rhs,
                    "u": u - torch.expm1(-a * dt) * (b * voltage - u),
                }
            },
            "mechanism_buffers": {
                "izh": {
                    name: update[name] for name in ("spikes", "syn_spikes", "v_prev")
                }
            },
            "clock": {"t": state["clock"]["t"] + dt},
        }
    return state


def _state_loss(state):
    return _loss({**state["mechanisms"]["izh"], **state["mechanism_buffers"]["izh"]})


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("batch", (None, 2))
def test_scnv_trace_matches_original_recurrence_imperative_and_commit_resume(
    kind, batch
):
    model = _model(kind, batch=batch)
    _seed(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    before = _buffers(model)
    state = tensors.state
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    reference = _model(kind, batch=batch)
    _seed(reference)
    dt = reference.v.new_tensor(DT)
    reference.integrator._initialize(
        reference, dt, force=True, compile_scope="population"
    )
    for index in range(STEPS):
        state, _ = functional.step(tensors.parameters, prepared, state)
        expected = _recurrence(
            tensors.state, tensors.parameters, tensors.constants["dt"], index + 1
        )
        _assert_close(state, expected)
        reference.integrator.step(reference, dt)
        reference.t = reference.t + dt
        _assert_close(state, functional.extract(reference).state)
    whole, _ = functional.rollout(
        tensors.parameters, prepared, tensors.state, steps=STEPS
    )
    _assert_close(whole, state)
    first, _ = functional.rollout(tensors.parameters, prepared, tensors.state, steps=2)
    continued, _ = functional.rollout(
        tensors.parameters, prepared, first, steps=STEPS - 2
    )
    _assert_close(continued, state)
    _assert_buffers(model, before)
    functional.commit_state_(model, state)
    _assert_close(functional.extract(model).state, state)
    expected, _ = functional.step(tensors.parameters, prepared, state)
    model.integrator._initialize(model, dt, force=True, compile_scope="population")
    model.integrator.step(model, dt)
    model.t = model.t + dt
    _assert_close(functional.extract(model).state, expected)


@pytest.mark.parametrize("kind", KINDS)
def test_scnv_gradients_match_original_recurrence(kind):
    model = _model(kind, batch=2)
    _seed(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters = _clone(tensors.parameters)
    raw_drive = _parameter(parameters, ".i_stim_param").requires_grad_()
    raw_a = _parameter(parameters, ".a_param").requires_grad_()
    raw_tau = _parameter(parameters, ".izh.tau_gate").requires_grad_()
    state = _clone(tensors.state)
    v = state["mechanisms"]["izh"]["v_izh"].requires_grad_()
    u = state["mechanisms"]["izh"]["u"].requires_grad_()
    previous = state["mechanism_buffers"]["izh"]["v_prev"].requires_grad_()
    targets = (raw_drive, raw_a, raw_tau, v, u, previous)
    first, _ = functional.prepare_and_step(parameters, tensors.constants, state)
    actual, _ = functional.prepare_and_step(parameters, tensors.constants, first)
    expected_first = _recurrence(state, parameters, tensors.constants["dt"], 1)
    expected = _recurrence(state, parameters, tensors.constants["dt"], 2)
    _assert_close(actual, expected)
    actual_gradients = torch.autograd.grad(
        _state_loss(first) + _state_loss(actual), targets, retain_graph=True
    )
    expected_gradients = torch.autograd.grad(
        _state_loss(expected_first) + _state_loss(expected), targets
    )
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=2e-9, atol=2e-10
        )
        assert torch.isfinite(actual_gradient).all()


@pytest.mark.parametrize("kind", KINDS)
def test_fresh_initialization_preserves_parameter_v_prev_with_state_override(kind):
    model = _model(kind, batch=2, v0=-62.0, ic={"v_izh": -9.0})
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters = _clone(tensors.parameters)
    raw_v0 = _parameter(parameters, ".v0_param")
    raw_v0.fill_(-61.0).requires_grad_()
    initialized = functional.initialize(
        parameters, tensors.constants, tensors.initialization
    )
    state = initialized.state
    torch.testing.assert_close(
        state["mechanisms"]["izh"]["v_izh"], torch.full_like(model.v, -9.0)
    )
    torch.testing.assert_close(
        state["mechanisms"]["izh"]["u"], torch.full_like(model.v, -12.2)
    )
    previous = state["mechanism_buffers"]["izh"]["v_prev"]
    torch.testing.assert_close(previous, torch.full_like(model.v, -61.0))
    gradient = torch.autograd.grad(previous.sum(), raw_v0)[0]
    torch.testing.assert_close(gradient, torch.full_like(raw_v0, model.v.numel()))
    after, _ = functional.prepare_and_step(
        initialized.parameters, initialized.constants, state
    )
    expected = _recurrence(
        state, initialized.parameters, initialized.constants["dt"], 1
    )
    _assert_close(after, expected)


@pytest.mark.parametrize("kind", KINDS)
def test_functional_transforms_and_compiled_scan_match_eager(kind):
    model = _model(kind)
    _seed(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    drive_name = next(
        name for name in tensors.parameters if name.endswith(".i_stim_param")
    )

    def run(drive):
        parameters = {**tensors.parameters, drive_name: drive}
        return functional.prepare_and_rollout(
            parameters, tensors.constants, tensors.state, steps=2
        )[0]

    drive = tensors.parameters[drive_name]
    lanes = torch.stack((drive - 1.0, drive + 2.0))
    actual = torch.vmap(run)(lanes)
    expected = pytree.tree_map(
        lambda *values: torch.stack(values), *(run(lane) for lane in lanes)
    )
    _assert_close(actual, expected)

    def voltage(raw_drive):
        return run(raw_drive)["mechanisms"]["izh"]["v_izh"]

    reverse = torch.func.jacrev(voltage)(drive)
    forward = torch.func.jacfwd(voltage)(drive)
    torch.testing.assert_close(reverse, forward, rtol=2e-9, atol=2e-10)
    assert torch.count_nonzero(reverse) == model.v.numel()

    prepared = functional.prepare(tensors.parameters, tensors.constants)
    chunk = functional.compile_rollout_chunk(2, backend="aot_eager")
    actual, _ = chunk(tensors.parameters, prepared, tensors.state)
    expected, _ = functional.rollout(
        tensors.parameters, prepared, tensors.state, steps=2
    )
    _assert_close(actual, expected)
    raw = tensors.parameters[drive_name]
    actual_gradient = torch.autograd.grad(_state_loss(actual), raw, retain_graph=True)[
        0
    ]
    expected_gradient = torch.autograd.grad(_state_loss(expected), raw)[0]
    torch.testing.assert_close(
        actual_gradient, expected_gradient, rtol=2e-9, atol=2e-10
    )
