"""Custom event receivers share native and functional event algebra."""

import dendra as dn
import pytest
import torch
from dendra._bootstrap import torch_compiler_warning_context
from torch.utils import _pytree as pytree

from dendra_models.models.networks.kumaravelu_2016.mechanisms.synapses import (
    alpha_syn_current,
    alpha_syn_event,
    exp2syn_current,
)
from dendra_models.models.networks.yu_2024.mechanisms.esser import (
    esser_mech_h,
    esser_mech_s,
)


KINDS = (
    alpha_syn_current,
    alpha_syn_event,
    exp2syn_current,
    esser_mech_h,
    esser_mech_s,
)
DT = 0.025


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _close(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for name in actual:
            _close(actual[name], expected[name])
        return
    assert pytree.tree_structure(actual) == pytree.tree_structure(expected)
    for value, reference in zip(
        pytree.tree_leaves(actual), pytree.tree_leaves(expected), strict=True
    ):
        torch.testing.assert_close(value, reference, rtol=2e-9, atol=2e-10)


def _model(kind, batch=2):
    with dn.ctx(JIT=0, DTYPE="float64", DEVICE="cpu", REQUIRE_GRAD=1):
        model = dn.Population(N=1, C=3, integrator=dn.scnv())
        model.insert(kind.rename("event"))
        if batch is not None:
            model.batch(batch)
        model.train()
        model.initialize()
        mechanism = model.mech.event
        mechanism._configure_timestep(model.v.new_tensor(DT))
        for name in mechanism._all_states:
            mechanism._buffers[name] = torch.full_like(getattr(mechanism, name), 0.03)
        if kind in (esser_mech_h, esser_mech_s):
            mechanism.v_iaf = (
                model.v.new_tensor((-53.4, -52.8, -40.0)).expand_as(model.v).clone()
            )
            mechanism.theta = torch.full_like(model.v, -53.0)
            mechanism.DE["esser_states"].gspike = torch.full_like(model.v, 0.4)
            if kind is esser_mech_h:
                mechanism.h_prev = torch.full_like(model.v, 0.1)
                mechanism.time_left = (
                    model.v.new_tensor((0.0, 0.08, 0.0)).expand_as(model.v).clone()
                )
            else:
                mechanism.g_prev = torch.full_like(model.v, 0.1)
    return model


def _original(kind, mechanism, weights, values):
    """The original imperative equations, evaluated without committing writes."""
    increment = weights * values["factor"]
    if kind in (alpha_syn_current, alpha_syn_event):
        return {"states": {"z": values["z"] + increment}}
    states = {"A": values["A"] + increment, "B": values["B"] + increment}
    if kind is exp2syn_current:
        return {"states": states}
    voltage, threshold = values["v_iaf"], values["theta"]
    ena = mechanism.DE["esser_states"].ena_iaf
    if kind is esser_mech_h:
        tau = mechanism.tau_gate.clamp_min(1e-3)
        gate = torch.sigmoid((voltage - threshold) / tau)
        eligible = values["time_left"] <= 0
        rising = (gate > 0.5) & (values["h_prev"] <= 0.5) & eligible
        probability = (
            torch.sigmoid((gate - 0.5) / tau)
            * torch.sigmoid((0.5 - values["h_prev"]) / tau)
            * eligible.to(gate.dtype)
        )
        spikes = rising.to(voltage.dtype) + mechanism.ste_scale * (
            probability - probability.detach()
        )
        time_left = torch.clamp(
            torch.where(
                rising,
                mechanism.tspike.expand_as(values["time_left"]),
                values["time_left"],
            )
            - DT,
            min=0.0,
        )
        states.update(
            v_iaf=torch.where(rising, ena.expand_as(voltage), voltage),
            theta=torch.where(rising, ena.expand_as(threshold), threshold),
        )
        carry = {"spikes": spikes, "h_prev": gate, "time_left": time_left}
        conductance = (time_left > 0).to(voltage.dtype)
    else:
        gate = torch.sigmoid((voltage - threshold) / mechanism.tau_gate.clamp_min(1e-6))
        rise = torch.relu(gate - values["g_prev"])
        conductance = values["state_carry"]["esser_states"]["gspike"]
        conductance = torch.clamp(
            conductance
            + DT
            * (
                -conductance / mechanism.DE["esser_states"].tau_spike + 10.0 * rise / DT
            ),
            0.0,
            1.0,
        )
        states["v_iaf"] = voltage + mechanism.alpha_peak * rise * (ena - voltage)
        carry = {"spikes": rise.to(voltage.dtype), "g_prev": gate}
    return {
        "states": states,
        "carry": carry,
        "state_carry": {"esser_states": {"gspike": conductance}},
    }


def _commit(mechanism, updates):
    for section in ("states", "carry"):
        for name, value in updates.get(section, {}).items():
            setattr(mechanism, name, value)
    for state, values in updates.get("state_carry", {}).items():
        for name, value in values.items():
            setattr(mechanism.DE[state], name, value)


def _loss(updates):
    return sum(value.square().mean() for value in pytree.tree_leaves(updates))


@pytest.mark.parametrize("kind", KINDS)
def test_receiver_uses_explicit_frames_without_mutation_and_preserves_gradients(kind):
    model = _model(kind)
    mechanism = model.mech.event
    original_buffers = _clone(dict(mechanism.named_buffers()))
    values = _clone(mechanism._net_receive_value_frame())
    values["factor"] = (2.0 * values["factor"]).requires_grad_()
    payload = torch.full_like(model.v, 0.002, requires_grad=True)
    before = _clone(values)
    actual = mechanism.net_receive_values(payload, model.v.new_tensor(DT), values)
    expected = _original(kind, mechanism, payload, values)
    _close(actual, expected)
    _close(values, before)
    _close(dict(mechanism.named_buffers()), original_buffers)
    targets = (payload, values["factor"])
    _close(
        torch.autograd.grad(_loss(actual), targets),
        torch.autograd.grad(_loss(expected), targets),
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("batch", (None, 2))
def test_native_functional_and_checkpointed_receives_match_original_repeated_events(
    kind, batch
):
    model, reference = _model(kind, batch), _model(kind, batch)
    with torch_compiler_warning_context(), dn.ctx(CHECK_PURITY=1):
        functional, tensors = dn.func.make_functional(model, dt=DT)
    state = tensors.state
    for weight in (0.002, 0.0, 0.0007):
        payload = torch.full_like(model.v, weight)
        state = functional.prepare_and_receive(
            tensors.parameters,
            tensors.constants,
            state,
            target="event",
            payload=payload,
        )
        expected = _original(
            kind,
            reference.mech.event,
            payload,
            reference.mech.event._net_receive_value_frame(),
        )
        _commit(reference.mech.event, expected)
        _close(state, functional.extract(reference).state)
    resumed = _model(kind, batch)
    functional.commit_state_(resumed, state)
    payload = torch.full_like(model.v, 0.001)
    resumed.mech.event.net_receive(payload, None)
    state = functional.prepare_and_receive(
        tensors.parameters, tensors.constants, state, target="event", payload=payload
    )
    _close(functional.extract(resumed).state, state)


@pytest.mark.parametrize("kind", KINDS)
def test_compiled_custom_receive_and_step_preserve_payload_gradients(kind):
    model = _model(kind)
    functional, tensors = dn.func.make_functional(model, dt=DT)

    def receive_and_step(parameters, state, payload):
        received = functional.prepare_and_receive(
            parameters, tensors.constants, state, target="event", payload=payload
        )
        return functional.prepare_and_step(parameters, tensors.constants, received)[0]

    payload = torch.full_like(model.v, 0.002, requires_grad=True)
    compiled = torch.compile(receive_and_step, backend="aot_eager", fullgraph=True)
    actual = compiled(tensors.parameters, tensors.state, payload)
    expected = receive_and_step(tensors.parameters, tensors.state, payload)
    _close(actual, expected)
    _close(
        torch.autograd.grad(_loss(actual), payload, retain_graph=True),
        torch.autograd.grad(_loss(expected), payload),
    )


@pytest.mark.parametrize("kind", (esser_mech_h, esser_mech_s))
def test_esser_event_workspaces_follow_raw_sources_and_preserve_parameter_gradients(
    kind,
):
    model = _model(kind)
    mechanism = model.mech.event
    state = mechanism.DE["esser_states"]
    with torch.no_grad():
        state.ena_iaf_param.add_(7.0)
        state.tau_spike_param.mul_(2.0)
    model.initialize()
    torch.testing.assert_close(mechanism.event_ena, state.ena_iaf.expand_as(model.v))
    torch.testing.assert_close(
        mechanism.event_tau_spike, state.tau_spike.expand_as(model.v)
    )

    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters = dict(tensors.parameters)
    ena_key = next(
        name for name in parameters if name.endswith(".esser_states.ena_iaf_param")
    )
    tau_key = next(
        name for name in parameters if name.endswith(".esser_states.tau_spike_param")
    )
    ena = parameters[ena_key] = (parameters[ena_key].detach() + 5.0).requires_grad_()
    tau = parameters[tau_key] = (parameters[tau_key].detach() * 1.5).requires_grad_()
    carry = _clone(tensors.state)
    voltage = carry["mechanisms"]["event"]["v_iaf"] = torch.full_like(model.v, -40.0)
    threshold = carry["mechanisms"]["event"]["theta"] = torch.full_like(model.v, -53.0)
    if kind is esser_mech_s:
        gate = torch.sigmoid((voltage - threshold) / mechanism.tau_gate.clamp_min(1e-6))
        carry["mechanism_buffers"]["event"]["g_prev"] = gate - 0.02
        carry["state_buffers"]["event"]["esser_states"]["gspike"] = torch.full_like(
            model.v, 0.4
        )
    received = functional.prepare_and_receive(
        parameters,
        tensors.constants,
        carry,
        target="event",
        payload=torch.zeros_like(model.v),
    )
    if kind is esser_mech_h:
        expected_voltage = ena.expand_as(voltage)
    else:
        rise = torch.relu(gate - carry["mechanism_buffers"]["event"]["g_prev"])
        expected_voltage = voltage + mechanism.alpha_peak * rise * (ena - voltage)
        expected_conductance = 0.4 + DT * (-0.4 / tau + 10.0 * rise / DT)
        actual_conductance = received["state_buffers"]["event"]["esser_states"][
            "gspike"
        ]
        _close(actual_conductance, expected_conductance)
        _close(
            torch.autograd.grad(actual_conductance.sum(), tau, retain_graph=True),
            torch.autograd.grad(expected_conductance.sum(), tau),
        )
    actual_voltage = received["mechanisms"]["event"]["v_iaf"]
    _close(actual_voltage, expected_voltage)
    _close(
        torch.autograd.grad(actual_voltage.sum(), ena),
        torch.autograd.grad(expected_voltage.sum(), ena),
    )
    torch.testing.assert_close(mechanism.event_ena, state.ena_iaf.expand_as(model.v))
    torch.testing.assert_close(
        mechanism.event_tau_spike, state.tau_spike.expand_as(model.v)
    )
