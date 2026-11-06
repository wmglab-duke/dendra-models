"""Zhang STP recovery is an accepted-step transition shared by all receivers."""

import dendra as dn
import pytest
import torch
from torch.utils import _pytree as pytree

from dendra_models.models.networks.zhang_2014.mechanisms import (
    AMPA_DynSyn,
    GABAa_DynSyn,
    GABAb_DynSyn,
    Glycine_DynSyn,
    NK1_DynSyn,
    NMDA_DynSyn,
)

DT = 0.1
KINDS = (
    AMPA_DynSyn,
    GABAa_DynSyn,
    GABAb_DynSyn,
    Glycine_DynSyn,
    NMDA_DynSyn,
    NK1_DynSyn,
)


def _model(kind, batch=None):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, DEVICE="cpu"):
        model = dn.SingleCompartment(N=1, C=3, v_init=-65.0)
        model.insert(kind.rename("event"), U1=0.3, tau_rec=2.0, tau_fac=3.0)
        if batch is not None:
            model.batch(batch)
        model.train()
        model.initialize()
        model.integrator._initialize(
            model, model.v.new_tensor(DT), force=True, compile_scope="population"
        )
        model.mech.event.P = torch.full_like(model.mech.event.P, 0.6)
        model.mech.event.Use = torch.full_like(model.mech.event.Use, 0.2)
    return model


def _clone(value):
    return pytree.tree_map_only(torch.Tensor, lambda item: item.clone(), value)


def _close(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for name in actual:
            _close(actual[name], expected[name])
    elif torch.is_tensor(actual):
        torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-11)
    else:
        assert actual == expected


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("batch", (None, 2))
def test_multiple_receivers_recover_before_delivery_once_per_step_and_resume(
    kind, batch
):
    source, native = _model(kind, batch), _model(kind, batch)
    with dn.ctx(CHECK_PURITY=1):
        functional, tensors = dn.func.make_functional(source, dt=DT)
    before = _clone(dict(source.named_buffers()))
    state = tensors.state
    expected = {
        name: getattr(native.mech.event, name).clone()
        for name in ("A", "B", "P", "Use")
    }
    synapse = native.mech.event
    tau_rise, tau_decay = synapse._tau_rise_decay()
    peak = (
        tau_rise * tau_decay / (tau_decay - tau_rise) * torch.log(tau_decay / tau_rise)
    )
    factor = 1.0 / (-torch.exp(-peak / tau_rise) + torch.exp(-peak / tau_decay))
    for step in range(3):
        expected["Use"] = expected["Use"] * torch.exp(-synapse.dt / synapse.tau_fac)
        expected["P"] = 1.0 - (1.0 - expected["P"]) * torch.exp(
            -synapse.dt / synapse.tau_rec
        )
        for weight in (0.0, 0.002, 0.0, 0.0007):
            payload = torch.full_like(synapse.A, weight)
            use_event = expected["Use"] + synapse.U1 * (1.0 - expected["Use"])
            released = use_event * expected["P"]
            increment = payload * factor * released
            expected["A"] = expected["A"] + increment
            expected["B"] = expected["B"] + increment
            expected["Use"] = torch.where(payload != 0, use_event, expected["Use"])
            expected["P"] = torch.where(
                payload != 0, expected["P"] - released, expected["P"]
            )
            state = functional.prepare_and_receive(
                tensors.parameters,
                tensors.constants,
                state,
                target="event",
                payload=payload,
            )
            synapse.net_receive(payload, None)
            for name, value in expected.items():
                torch.testing.assert_close(
                    getattr(synapse, name), value, rtol=2e-12, atol=2e-14
                )
            assert synapse.zhang2014_last_stp_step.tolist() == [step]
            _close(state, functional.extract(native).state)
        state, _ = functional.prepare_and_step(
            tensors.parameters, tensors.constants, state
        )
        native.step(dt=DT)
        expected["A"] = expected["A"] * torch.exp(-synapse.dt / tau_rise)
        expected["B"] = expected["B"] * torch.exp(-synapse.dt / tau_decay)
        _close(state, functional.extract(native).state)
        assert synapse.zhang2014_stp_step.tolist() == [step + 1]
        if step == 1:
            resumed = _model(kind, batch)
            resumed.restore_dict_from_checkpoint(native.state_dict_for_checkpoint())
            native = resumed
            synapse = native.mech.event
            state = _clone(state)
    for name, value in source.named_buffers():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_integer_counter_distinguishes_steps_even_when_float32_time_cannot():
    model = _model(AMPA_DynSyn).float()
    synapse = model.mech.event
    synapse.zhang2014_stp_step.fill_(2**24 + 1)
    synapse.zhang2014_last_stp_step.fill_(2**24)
    payload = torch.zeros_like(synapse.A)
    previous_time = model.t.clone()
    for step in (2**24 + 1, 2**24 + 2):
        before_use, before_p = synapse.Use.clone(), synapse.P.clone()
        expected_use = before_use * torch.exp(-synapse.dt / synapse.tau_fac)
        expected_p = 1.0 - (1.0 - before_p) * torch.exp(-synapse.dt / synapse.tau_rec)
        synapse.net_receive(payload, None)
        torch.testing.assert_close(synapse.Use, expected_use)
        torch.testing.assert_close(synapse.P, expected_p)
        assert synapse.zhang2014_last_stp_step.tolist() == [step]
        synapse.net_receive(payload, None)
        torch.testing.assert_close(synapse.Use, expected_use, rtol=0, atol=0)
        updates = synapse.advance(model.v, synapse.dt, synapse._runtime_value_frame())
        for name, value in updates.items():
            setattr(synapse, name, value)
    torch.testing.assert_close(model.t, previous_time, rtol=0, atol=0)


def test_compiled_receive_and_step_preserve_payload_gradients():
    model = _model(AMPA_DynSyn)
    functional, tensors = dn.func.make_functional(model, dt=DT)

    def receive_and_step(parameters, state, payload):
        for _ in range(2):
            state = functional.prepare_and_receive(
                parameters, tensors.constants, state, target="event", payload=payload
            )
        return functional.prepare_and_step(parameters, tensors.constants, state)[0]

    payload = torch.full_like(model.mech.event.A, 0.002, requires_grad=True)
    compiled = torch.compile(receive_and_step, backend="aot_eager", fullgraph=True)
    actual = compiled(tensors.parameters, tensors.state, payload)
    expected = receive_and_step(tensors.parameters, tensors.state, payload)
    _close(actual, expected)
    actual_loss = sum(
        value.square().sum() for value in actual["mechanisms"]["event"].values()
    )
    expected_loss = sum(
        value.square().sum() for value in expected["mechanisms"]["event"].values()
    )
    actual_grad = torch.autograd.grad(actual_loss, payload, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected_loss, payload)[0]
    assert torch.isfinite(actual_grad).all()
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-9, atol=2e-11)


@pytest.mark.parametrize("kind", KINDS)
def test_fresh_initialization_resets_plasticity_and_both_step_counters(kind):
    with dn.ctx(CHECK_PURITY=1):
        model = _model(kind)
        functional, tensors = dn.func.make_functional(model, dt=DT)
        continuation = dn.func.initialize(functional, tensors)
    carry = continuation.state["mechanism_buffers"]["event"]
    torch.testing.assert_close(carry["P"], torch.ones_like(carry["P"]))
    torch.testing.assert_close(carry["Use"], torch.zeros_like(carry["Use"]))
    assert carry["zhang2014_last_stp_step"].tolist() == [-1]
    assert carry["zhang2014_stp_step"].tolist() == [0]


def test_legacy_state_dict_restores_the_next_accepted_step_without_mutating_input():
    model = _model(AMPA_DynSyn)
    synapse = model.mech.event
    payload = torch.full_like(synapse.A, 0.002)
    synapse.net_receive(payload, None)
    updates = synapse.advance(model.v, synapse.dt, synapse._runtime_value_frame())
    for name, value in updates.items():
        setattr(synapse, name, value)
    legacy = {name: value.clone() for name, value in synapse.state_dict().items()}
    del legacy["zhang2014_stp_step"]
    keys = tuple(legacy)
    restored = _model(AMPA_DynSyn).mech.event
    restored.load_state_dict(legacy, strict=True)
    assert tuple(legacy) == keys
    assert restored.zhang2014_last_stp_step.tolist() == [0]
    assert restored.zhang2014_stp_step.tolist() == [1]
    synapse.net_receive(torch.zeros_like(payload), None)
    restored.net_receive(torch.zeros_like(payload), None)
    for name in ("A", "B", "P", "Use", "zhang2014_last_stp_step", "zhang2014_stp_step"):
        torch.testing.assert_close(getattr(restored, name), getattr(synapse, name))
