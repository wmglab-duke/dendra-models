"""Pure builtin synaptic delivery composes with a companion HH cable model."""

from __future__ import annotations

import dendra as dn
import pytest
import torch
from dendra._bootstrap import reset_torch_compiler
from dendra.models.mod import exp2syn, expsyn
from torch.utils import _pytree as pytree

from dendra_models.models import Rattay1993

DT = 0.01
KINDS = (expsyn, exp2syn)


@pytest.fixture(autouse=True)
def _reset_compiler():
    reset_torch_compiler()
    try:
        yield
    finally:
        reset_torch_compiler()


def _model(kind, *, batch=None):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, IMEM=False):
        model = Rattay1993(
            diameters=[1.0],
            L=40.0,
            dx=10.0,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        ).double()
        parameters = {"tau": 0.7} if kind is expsyn else {"tau1": 0.3, "tau2": 2.0}
        model[:, 1:3].insert(kind.rename("event"), **parameters)
        if batch is not None:
            model.batch(batch)
        model.train()
        model.initialize()
    mechanism = model.mech.event
    if kind is expsyn:
        mechanism.g = torch.full_like(mechanism.g, 0.001)
    else:
        mechanism.A = torch.full_like(mechanism.A, 0.0004)
        mechanism.B = torch.full_like(mechanism.B, 0.001)
    return model


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _assert_close(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for name in actual:
            _assert_close(actual[name], expected[name])
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-10)


def _parameter_key(parameters, field, state=None):
    suffix = ".event." + (f"DE.{state}." if state is not None else "")
    suffix += field + "_param"
    overridden = [name for name in parameters if name.endswith(suffix + "_0")]
    matches = overridden or [name for name in parameters if name.endswith(suffix)]
    assert len(matches) == 1, suffix
    return matches[0]


def _parameter(parameters, field, state=None):
    return parameters[_parameter_key(parameters, field, state)]


def _original_increment(kind, values, payload, parameters):
    """Original conductance increments, independent of either receive adapter."""
    if kind is expsyn:
        return {"g": values["g"] + payload}
    tau1 = _parameter(parameters, "tau1", "A")
    tau2 = _parameter(parameters, "tau2", "B")
    peak_time = tau1 * tau2 / (tau2 - tau1) * torch.log(tau2 / tau1)
    factor = 1.0 / (torch.exp(-peak_time / tau2) - torch.exp(-peak_time / tau1))
    increment = payload * factor
    return {"A": values["A"] + increment, "B": values["B"] + increment}


def _original_decay(kind, values, parameters, dt):
    if kind is expsyn:
        return {"g": values["g"] * torch.exp(-dt / _parameter(parameters, "tau", "g"))}
    return {
        "A": values["A"] * torch.exp(-dt / _parameter(parameters, "tau1", "A")),
        "B": values["B"] * torch.exp(-dt / _parameter(parameters, "tau2", "B")),
    }


def _payloads(model):
    shape = next(
        iter(model.mech.event._buffers[name] for name in model.mech.event._all_states)
    ).shape
    count = torch.tensor(shape).prod().item()
    first = torch.linspace(0.0005, 0.0015, int(count), dtype=model.dtype()).reshape(
        shape
    )
    return first, torch.zeros_like(first), 0.4 * first


def _initialize_integrator(model):
    dt = model.v.new_tensor(DT)
    model.integrator._initialize(model, dt, force=True, compile_scope="population")
    return dt


def _imperative_original_receive(model, kind, payload):
    mechanism = model.mech.event
    values = {name: getattr(mechanism, name) for name in mechanism._all_states}
    updates = _original_increment(kind, values, payload, dict(model.named_parameters()))
    for name, value in updates.items():
        setattr(mechanism, name, value)


def _imperative_step(model, dt):
    model.integrator.step(model, dt)
    model.t = model.t + dt


def _loss(state):
    synaptic = sum(
        value.square().sum() for value in state["mechanisms"]["event"].values()
    )
    return 0.001 * state["integrator"]["v"].square().sum() + 100.0 * synaptic


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("batch", (None, 2))
def test_receive_then_step_matches_original_synaptic_events_on_rattay_cable(
    kind, batch
):
    model, reference = _model(kind, batch=batch), _model(kind, batch=batch)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    dt = _initialize_integrator(reference)
    before = {name: value.detach().clone() for name, value in model.named_buffers()}
    payloads = _payloads(model)
    original_payloads = _clone(payloads)
    state = tensors.state
    synaptic = state["mechanisms"]["event"]
    for payload in payloads:
        expected_event = _original_increment(
            kind, synaptic, payload, tensors.parameters
        )
        received = functional.receive(
            tensors.parameters, prepared, state, target="event", payload=payload
        )
        expected = {
            **state,
            "mechanisms": {**state["mechanisms"], "event": expected_event},
        }
        _assert_close(received, expected)
        _imperative_original_receive(reference, kind, payload)
        state, _ = functional.step(tensors.parameters, prepared, received)
        _imperative_step(reference, dt)
        _assert_close(state, functional.extract(reference).state)
        synaptic = _original_decay(kind, expected_event, tensors.parameters, dt)
        _assert_close(state["mechanisms"]["event"], synaptic)
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0, msg=name)
    for actual, original in zip(payloads, original_payloads, strict=True):
        torch.testing.assert_close(actual, original, rtol=0, atol=0)
    functional.commit_state_(model, state)
    _assert_close(functional.extract(model).state, state)
    continued = functional.prepare_and_receive(
        tensors.parameters,
        tensors.constants,
        state,
        target="event",
        payload=payloads[0],
    )
    continued, _ = functional.step(tensors.parameters, prepared, continued)
    # The canonical imperative net_receive adapter must remain usable after
    # committing a functional state; it applies the same physical payload.
    model.mech.event.net_receive(payloads[0], None)
    _imperative_step(model, _initialize_integrator(model))
    _assert_close(functional.extract(model).state, continued)


@pytest.mark.parametrize("kind", KINDS)
def test_event_and_membrane_gradients_match_original_imperative_delivery(kind):
    model, reference = _model(kind, batch=2), _model(kind, batch=2)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    state = _clone(tensors.state)
    payload = _payloads(model)[0].requires_grad_()
    reference_payload = payload.detach().clone().requires_grad_()
    state_targets, reference_state_targets = [], []
    for name in state["mechanisms"]["event"]:
        state_targets.append(state["mechanisms"]["event"][name].requires_grad_())
        reference_value = state_targets[-1].detach().clone().requires_grad_()
        setattr(reference.mech.event, name, reference_value)
        reference_state_targets.append(reference_value)
    names = (
        (_parameter_key(tensors.parameters, "tau", "g"),)
        if kind is expsyn
        else (
            _parameter_key(tensors.parameters, "tau1", "A"),
            _parameter_key(tensors.parameters, "tau2", "B"),
        )
    ) + (_parameter_key(tensors.parameters, "e"),)
    parameters = tensors.parameters
    reference_parameters = dict(reference.named_parameters())
    prepared = functional.prepare(parameters, tensors.constants)
    dt = _initialize_integrator(reference)
    for scale in (1.0, 0.0, 0.4):
        state = functional.receive(
            parameters, prepared, state, target="event", payload=scale * payload
        )
        state, _ = functional.step(parameters, prepared, state)
        _imperative_original_receive(reference, kind, scale * reference_payload)
        _imperative_step(reference, dt)
    expected = functional.extract(reference).state
    _assert_close(state, expected)
    targets = (payload, *state_targets, *(parameters[name] for name in names))
    reference_targets = (
        reference_payload,
        *reference_state_targets,
        *(reference_parameters[name] for name in names),
    )
    actual_gradients = torch.autograd.grad(_loss(state), targets)
    # extract() intentionally snapshots detached leaves. Use the live
    # imperative tensors when comparing the original delivery's backward pass.
    live_reference = {
        "integrator": {"v": reference.v},
        "mechanisms": {
            "event": {
                name: getattr(reference.mech.event, name)
                for name in reference.mech.event._all_states
            }
        },
    }
    expected_gradients = torch.autograd.grad(_loss(live_reference), reference_targets)
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)
        assert torch.isfinite(actual).all()
        assert torch.count_nonzero(actual)


@pytest.mark.parametrize("kind", KINDS)
def test_receive_uses_explicit_synaptic_state_and_fresh_preparation(kind):
    model = _model(kind)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters, state = _clone(tensors.parameters), _clone(tensors.state)
    for name in state["mechanisms"]["event"]:
        state["mechanisms"]["event"][name] = state["mechanisms"]["event"][name] + 0.002
        setattr(
            model.mech.event,
            name,
            torch.full_like(getattr(model.mech.event, name), 100.0),
        )
    tau_name, tau_state = ("tau", "g") if kind is expsyn else ("tau1", "A")
    _parameter(parameters, tau_name, tau_state).mul_(1.2)
    payload = _payloads(model)[0]
    source = _clone(state)
    actual = functional.prepare_and_receive(
        parameters, tensors.constants, state, target="event", payload=payload
    )
    expected = {
        **state,
        "mechanisms": {
            **state["mechanisms"],
            "event": _original_increment(
                kind, state["mechanisms"]["event"], payload, parameters
            ),
        },
    }
    _assert_close(actual, expected)
    _assert_close(state, source)
    for name in state["mechanisms"]["event"]:
        torch.testing.assert_close(
            getattr(model.mech.event, name),
            torch.full_like(getattr(model.mech.event, name), 100.0),
        )


@pytest.mark.parametrize("kind", KINDS)
def test_compiled_receive_and_cable_step_preserve_event_gradient(kind):
    model = _model(kind)
    functional, tensors = dn.func.make_functional(model, dt=DT)

    def receive_and_step(parameters, state, payload):
        received = functional.prepare_and_receive(
            parameters, tensors.constants, state, target="event", payload=payload
        )
        return functional.prepare_and_step(parameters, tensors.constants, received)[0]

    payload = _payloads(model)[0].requires_grad_()
    compiled = torch.compile(receive_and_step, backend="aot_eager", fullgraph=True)
    actual = compiled(tensors.parameters, tensors.state, payload)
    expected = receive_and_step(tensors.parameters, tensors.state, payload)
    _assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(_loss(actual), payload, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(_loss(expected), payload)[0]
    torch.testing.assert_close(actual_gradient, expected_gradient, rtol=2e-8, atol=2e-9)
