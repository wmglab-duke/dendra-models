"""Shared threshold detectors drive real companion cable synapses."""

from __future__ import annotations

import math

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


def _network(kind, *, batch=None, surrogate=True):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, IMEM=False):
        source = dn.SingleCompartment(
            N=1, C=3, v_init=-0.2, integrator=dn.scnv(imem=False)
        ).double()
        post = Rattay1993(
            diameters=[1.0],
            L=40.0,
            dx=10.0,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        ).double()
        parameters = {"tau": 0.7} if kind is expsyn else {"tau1": 0.3, "tau2": 2.0}
        post[:, torch.tensor([3, 1])].insert(kind.rename("event"), **parameters)
        network = dn.Network(
            {"source": source, "post": post},
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        # One source has two different finite thresholds; another has both a
        # finite threshold and a raw gate. Targets include repeated locations.
        network.connect_one_to_one(
            source[:, torch.tensor([0, 1, 1, 2, 0])],
            post[:, torch.tensor([1, 3, 1, 3, 1])],
            post.mech.event,
            threshold=torch.tensor([0.0, 0.2, float("nan"), -0.1, 0.5]),
            weight=torch.tensor([0.0002, 0.0003, 0.0004, 0.0005, 0.0006]),
            delay=torch.tensor([0.0, DT, 3 * DT, DT, 0.0]),
            allow_multapses=True,
        )
        if batch is not None:
            network.batch(batch)
        network.train()
        network.initialize(DT)
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=surrogate,
            diff_scheduled_times=False,
            train_delay_backend="dense",
            tau=0.25,
        )
    return network, next(iter(network.synapses.values()))


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


def _voltages(connection):
    # Equality triggers; remaining above does not retrigger. Later drops reset
    # each detector independently, including different thresholds on source 0.
    tape = torch.tensor(
        [
            [-0.2, -0.3, -0.3],
            [0.0, 0.1, -0.1],
            [0.5, 0.2, 0.2],
            [0.8, 0.6, 0.4],
            [0.1, 0.0, -0.2],
            [-0.1, 0.4, -0.1],
            [-0.2, 0.5, 0.0],
            [0.6, 0.1, -0.3],
            [0.7, 0.2, -0.1],
            [0.4, 0.4, 0.2],
            [-0.4, -0.2, -0.4],
            [0.0, 0.2, -0.1],
        ],
        dtype=connection.pre_dtype,
        device=connection.pre_device,
    )
    shape = connection.get_pre_var(connection.pre).shape
    if len(shape) == 3:
        delayed = torch.cat((tape[:1], tape[:-1]), dim=0)
        tape = torch.stack((tape, delayed), dim=1)
    return tape.reshape((len(tape), *shape))


def _expected_gates(connection, voltages, index):
    """Hand-specified event times, independent of either detector path."""
    times = {
        (0, 0.0): {1, 7, 11},
        (0, 0.5): {2, 7},
        (1, 0.2): {2, 5, 8, 11},
        (2, -0.1): {1, 5, 8, 11},
    }
    values = []
    for edge, source in enumerate(connection.pre_idx.tolist()):
        if bool(connection.thresh_is_nan[edge]):
            values.append(voltages.reshape(-1)[source])
        else:
            threshold = round(float(connection.threshold[edge]), 1)
            active = index - source // 3 in times[(source % 3, threshold)]
            values.append(voltages.new_tensor(float(active)))
    return torch.stack(values)


def _ledger_step(connection, pending, index, weights, gates):
    """A chronological due-time ledger rather than another delay ring."""
    template = weights.new_zeros(math.prod(connection.syn.shape_f))
    payload = pending.pop(index, template)
    for delay in torch.unique(connection.inference_delay_steps).tolist():
        edges = connection.inference_delay_steps == delay
        contribution = template.index_add(
            0, connection.post_idx[edges], (weights * gates)[edges]
        )
        due = index + delay
        pending[due] = pending.get(due, template) + contribution
    return payload.reshape(connection.syn.shape_f)


def _receive(functional, population, prepared, state, payload):
    return functional.receive(
        population.parameters, prepared, state, target="event", payload=payload
    )


def _loss(state, queue):
    synaptic = sum(
        value.square().sum() for value in state["mechanisms"]["event"].values()
    )
    return (
        0.001 * state["integrator"]["v"].square().sum()
        + 100.0 * synaptic
        + 100.0 * queue["delivery_buffer"].square().sum()
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("batch", (None, 2))
@pytest.mark.parametrize("surrogate", (False, True))
def test_split_threshold_projection_matches_event_timeline_network_and_gradients(
    kind, batch, surrogate
):
    network, connection = _network(kind, batch=batch, surrogate=surrogate)
    native, native_connection = _network(kind, batch=batch, surrogate=surrogate)
    projection, delivery = dn.func.make_functional_netcon(connection)
    functional, population = dn.func.make_functional(network.post, dt=DT)
    prepared = functional.prepare(population.parameters, population.constants)
    weights = delivery.weights.detach().clone().requires_grad_()
    native_weights = weights.detach().clone().requires_grad_()
    native_connection.weight.w = native_weights
    voltages = _voltages(connection).requires_grad_()
    native_voltages = voltages.detach().clone().requires_grad_()
    original = _clone(dict(network.named_buffers()))
    queue, state, pending = delivery.state, population.state, {}
    ledger_state = _clone(state)
    assert queue["has_spiked"].dtype == torch.bool
    assert connection.post_idx.unique().numel() < connection.post_idx.numel()

    for index, voltage in enumerate(voltages):
        queue, payload = projection.read_due(queue)
        # Keep Network's per-projection order: due -> receive -> observe the
        # source -> enqueue. Population advancement follows all projections.
        state = _receive(functional, population, prepared, state, payload)
        queue = projection.advance(weights, queue, voltage)
        gates = _expected_gates(connection, voltage, index)
        expected = _ledger_step(connection, pending, index, weights, gates)
        _assert_close(payload, expected)
        ledger_state = _receive(
            functional, population, prepared, ledger_state, expected
        )
        state = functional.step(population.parameters, prepared, state)[0]
        ledger_state = functional.step(population.parameters, prepared, ledger_state)[0]
        _assert_close(state, ledger_state)
        native.source.v = native_voltages[index]
        native.step()
        _assert_close(queue, native_connection.state_dict_for_checkpoint())
        _assert_close(state, functional.extract(native.post).state)

    for name, value in network.named_buffers():
        _assert_close(value, original[name])
    actual = torch.autograd.grad(_loss(state, queue), (weights, voltages))
    native_state = {
        "integrator": {"v": native.post.v},
        "mechanisms": {
            "event": {
                name: getattr(native.post.mech.event, name)
                for name in native.post.mech.event._all_states
            }
        },
    }
    expected = torch.autograd.grad(
        _loss(native_state, native_connection.state_dict_for_checkpoint()),
        (native_weights, native_voltages),
    )
    for observed, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, reference, rtol=2e-8, atol=2e-9)
        assert torch.isfinite(observed).all()
        assert torch.count_nonzero(observed)
    # Hard finite thresholds have no source derivative; the mixed raw lane
    # remains differentiable. Surrogate mode also reaches finite source lanes.
    finite_only_sources = actual[1][..., [0, 2]]
    assert bool(torch.count_nonzero(finite_only_sources)) is surrogate


@pytest.mark.parametrize("kind", KINDS)
def test_threshold_latches_and_pending_deliveries_resume_while_above(kind):
    network, connection = _network(kind, batch=2)
    projection, delivery = dn.func.make_functional_netcon(connection)
    functional, population = dn.func.make_functional(network.post, dt=DT)
    prepared = functional.prepare(population.parameters, population.constants)
    queue, state = delivery.state, population.state
    voltages = _voltages(connection)
    split = 4
    for voltage in voltages[:split]:
        queue, payload = projection.read_due(queue)
        state = _receive(functional, population, prepared, state, payload)
        queue = projection.advance(delivery.weights, queue, voltage)
        state = functional.step(population.parameters, prepared, state)[0]
        network.source.v = voltage
        network.step()
    checkpoint, population_checkpoint = _clone(queue), _clone(state)
    assert checkpoint["has_spiked"].all()
    assert torch.count_nonzero(checkpoint["delivery_buffer"])
    projection.commit_state_(connection, checkpoint)
    functional.commit_state_(network.post, population_checkpoint)
    _assert_close(projection.extract(connection).state, checkpoint)
    resumed_queue, resumed_state = _clone(queue), _clone(state)

    # Repeat the above-threshold observation at the boundary: restoring the
    # queue without restoring the latch would spuriously enqueue new events.
    tail = torch.cat((voltages[split - 1 : split], voltages[split:]), dim=0)
    for voltage in tail:
        queue, payload = projection.read_due(queue)
        state = _receive(functional, population, prepared, state, payload)
        queue = projection.advance(delivery.weights, queue, voltage)
        state = functional.step(population.parameters, prepared, state)[0]
        resumed_queue, resumed_payload = projection.read_due(resumed_queue)
        resumed_state = _receive(
            functional, population, prepared, resumed_state, resumed_payload
        )
        resumed_queue = projection.advance(delivery.weights, resumed_queue, voltage)
        resumed_state = functional.step(population.parameters, prepared, resumed_state)[
            0
        ]
        network.source.v = voltage
        network.step()
        _assert_close(queue, resumed_queue)
        _assert_close(state, resumed_state)
        _assert_close(queue, connection.state_dict_for_checkpoint())
        _assert_close(state, functional.extract(network.post).state)


@pytest.mark.parametrize("kind", KINDS)
def test_compiled_batched_split_detector_receive_and_cable_step(kind):
    network, connection = _network(kind, batch=2)
    projection, delivery = dn.func.make_functional_netcon(connection)
    functional, population = dn.func.make_functional(network.post, dt=DT)

    def advance(weights, queue, state, voltage):
        queue, payload = projection.read_due(queue)
        state = functional.prepare_and_receive(
            population.parameters,
            population.constants,
            state,
            target="event",
            payload=payload,
        )
        queue = projection.advance(weights, queue, voltage)
        state, _ = functional.prepare_and_step(
            population.parameters, population.constants, state
        )
        return queue, state

    weights = delivery.weights.detach().clone().requires_grad_()
    voltages = _voltages(connection)[:6].requires_grad_()
    compiled = torch.compile(advance, backend="aot_eager", fullgraph=True)
    queue, state = delivery.state, population.state
    expected_queue, expected_state = _clone(queue), _clone(state)
    for voltage in voltages:
        queue, state = compiled(weights, queue, state, voltage)
        expected_queue, expected_state = advance(
            weights, expected_queue, expected_state, voltage
        )
    _assert_close(queue, expected_queue)
    _assert_close(state, expected_state)
    actual = torch.autograd.grad(
        _loss(state, queue), (weights, voltages), retain_graph=True
    )
    expected = torch.autograd.grad(
        _loss(expected_state, expected_queue), (weights, voltages)
    )
    for observed, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, reference, rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize("surrogate", (False, True))
def test_recurrent_conductance_is_observed_after_receiving_due_event(surrogate):
    """Receiving into expsyn can itself cause the observed threshold crossing."""

    def recurrent_network():
        with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, IMEM=False):
            post = Rattay1993(
                diameters=[1.0],
                L=20.0,
                dx=10.0,
                integrator=dn.bwd_euler_ub(method="pcr", imem=False),
            ).double()
            post.insert(expsyn.rename("event"), tau=0.7)
            network = dn.Network(
                {"post": post},
                netcon_delay_backend="dense",
                netcon_train_backend="dense",
            )
            network.connect_one_to_one(
                post[:],
                post[:],
                post.mech.event,
                pre_var="mech.event.g",
                threshold=0.0002,
                weight=0.0001,
                delay=DT,
                allow_autapses=True,
            )
            network.train()
            network.initialize(DT)
            network.set_synaptic_diff_config(
                diff_weights=True,
                diff_delays=False,
                diff_spiking=surrogate,
                diff_scheduled_times=False,
                train_delay_backend="dense",
                tau=0.0001,
            )
        connection = next(iter(network.synapses.values()))
        # A previously queued event arrives in the first call. Conductance is
        # initially below threshold, so observing before receive loses a spike.
        connection.delivery_buffer[0, 0] = 0.0003
        return network, connection

    network, connection = recurrent_network()
    native, native_connection = recurrent_network()
    projection, delivery = dn.func.make_functional_netcon(connection)
    functional, population = dn.func.make_functional(network.post, dt=DT)
    prepared = functional.prepare(population.parameters, population.constants)
    weights = delivery.weights.detach().clone().requires_grad_()
    native_weights = weights.detach().clone().requires_grad_()
    native_connection.weight.w = native_weights
    queue, state = delivery.state, population.state

    for index in range(5):
        queue, payload = projection.read_due(queue)
        state = _receive(functional, population, prepared, state, payload)
        observed = state["mechanisms"]["event"]["g"]
        queue = projection.advance(weights, queue, observed)
        state = functional.step(population.parameters, prepared, state)[0]
        native.step()
        _assert_close(queue, native_connection.state_dict_for_checkpoint())
        _assert_close(state, functional.extract(native.post).state)
        if index == 0:
            assert queue["has_spiked"][0]
            assert not queue["has_spiked"][1:].any()
        if index == 1:
            expected_payload = torch.zeros_like(payload)
            expected_payload.reshape(-1)[0] = 0.0001
            _assert_close(payload, expected_payload)

    native_state = {
        "integrator": {"v": native.post.v},
        "mechanisms": {"event": {"g": native.post.mech.event.g}},
    }
    actual = torch.autograd.grad(_loss(state, queue), weights)[0]
    expected = torch.autograd.grad(
        _loss(native_state, native_connection.state_dict_for_checkpoint()),
        native_weights,
    )[0]
    torch.testing.assert_close(actual, expected, rtol=2e-8, atol=2e-9)
    assert torch.count_nonzero(actual)
