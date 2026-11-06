"""Fixed-delay projections deliver real companion cable synapses explicitly."""

from __future__ import annotations

import math

import dendra as dn
import pytest
import torch
from dendra._bootstrap import reset_torch_compiler
from dendra.models.mechanisms import Mechanism
from dendra.models.mod import exp2syn, expsyn
from torch.utils import _pytree as pytree

from dendra_models.models import Rattay1993

DT = 0.01
KINDS = (expsyn, exp2syn)


class GateSource(Mechanism):
    """An explicit numeric source, independent of threshold/spike detection."""

    Mechanism.CARRY("gate")


@pytest.fixture(autouse=True)
def _reset_compiler():
    reset_torch_compiler()
    try:
        yield
    finally:
        reset_torch_compiler()


def _network(kind, *, batch=None):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, IMEM=False):
        source = dn.Population(N=1, C=3, integrator=dn.scnv(imem=False))
        source.insert(GateSource.rename("source"))
        post = Rattay1993(
            diameters=[1.0],
            L=40.0,
            dx=10.0,
            integrator=dn.bwd_euler_ub(method="pcr", imem=False),
        ).double()
        parameters = {"tau": 0.7} if kind is expsyn else {"tau1": 0.3, "tau2": 2.0}
        # The physical regional order is compiled by Dendra. Neither the
        # ledger nor the projection assumes that insertion order is retained.
        post[:, torch.tensor([3, 1])].insert(kind.rename("event"), **parameters)
        network = dn.Network(
            {"source": source, "post": post},
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        network.connect_one_to_one(
            source[:, torch.tensor([0, 1, 1, 2, 0])],
            post[:, torch.tensor([1, 3, 1, 3, 1])],
            post.mech.event,
            pre_var="mech.source.gate",
            threshold=None,
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
            diff_spiking=True,
            diff_scheduled_times=False,
            train_delay_backend="dense",
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


def _gates(connection):
    tape = torch.tensor(
        [
            [1.0, 0.5, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.25, 0.8],
            [0.3, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.7, 0.4],
            [0.0, 0.0, 0.0],
            [0.6, 0.2, 0.0],
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.9],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=connection.weight().dtype,
        device=connection.weight().device,
    )
    shape = connection.get_pre_var(connection.pre).shape
    if len(shape) == 3:
        tape = torch.stack((tape, 0.6 * tape.roll(1, dims=0)), dim=1)
    return tape.reshape((len(tape), *shape))


def _ledger_step(connection, pending, index, weights, gates):
    """Chronological due-time ledger, with no circular-buffer implementation."""
    template = weights.new_zeros(math.prod(connection.syn.shape_f))
    payload = pending.pop(index, template)
    edge_values = weights * gates.reshape(-1).index_select(0, connection.pre_idx)
    delays = connection.inference_delay_steps
    for delay in torch.unique(delays).tolist():
        edges = delays == delay
        contribution = template.index_add(
            0, connection.post_idx[edges], edge_values[edges]
        )
        due = index + delay
        pending[due] = pending.get(due, template) + contribution
    return payload.reshape(connection.syn.shape_f)


def _assert_pending(state, pending, next_index):
    buffer = state["delivery_buffer"]
    pointer = int(state["current_time_step"].item())
    chronological = buffer.roll(-pointer, dims=0)
    expected = torch.stack(
        [
            pending.get(next_index + age, torch.zeros_like(buffer[0]))
            for age in range(buffer.shape[0])
        ]
    )
    _assert_close(chronological, expected)
    assert int(state["global_step"].item()) == next_index


def _advance(functional, tensors, prepared, state, payload):
    received = functional.receive(
        tensors.parameters, prepared, state, target="event", payload=payload
    )
    return functional.step(tensors.parameters, prepared, received)[0]


def _loss(state, queue_state):
    synaptic = sum(
        value.square().sum() for value in state["mechanisms"]["event"].values()
    )
    return (
        0.001 * state["integrator"]["v"].square().sum()
        + 100.0 * synaptic
        + 100.0 * queue_state["delivery_buffer"].square().sum()
    )


def _native_loss(network, connection):
    state = {
        "integrator": {"v": network.post.v},
        "mechanisms": {
            "event": {
                name: getattr(network.post.mech.event, name)
                for name in network.post.mech.event._all_states
            }
        },
    }
    return _loss(state, connection.state_dict_for_checkpoint())


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("batch", (None, 2))
def test_projection_cable_trajectory_and_gradients_match_ledger_and_network(
    kind, batch
):
    network, connection = _network(kind, batch=batch)
    native, native_connection = _network(kind, batch=batch)
    projection, delivery = dn.func.make_functional_netcon(connection)
    functional, population = dn.func.make_functional(network.post, dt=DT)
    prepared = functional.prepare(population.parameters, population.constants)
    weights = delivery.weights.detach().clone().requires_grad_()
    native_weights = weights.detach().clone().requires_grad_()
    native_connection.weight.w = native_weights
    gates = _gates(connection).requires_grad_()
    native_gates = gates.detach().clone().requires_grad_()
    original_buffers = dict(network.named_buffers())
    snapshots = _clone(original_buffers)
    queue, state, pending = delivery.state, population.state, {}
    ledger_state = _clone(population.state)
    assert connection.inference_delay_steps.min().item() == 1
    assert connection.post_idx.unique().numel() < connection.post_idx.numel()

    for index, gate in enumerate(gates):
        queue, payload = projection.step(weights, queue, gate)
        expected = _ledger_step(connection, pending, index, weights, gate)
        _assert_close(payload, expected)
        if index == 0:
            assert not torch.count_nonzero(payload)
        _assert_pending(queue, pending, index + 1)
        state = _advance(functional, population, prepared, state, payload)
        ledger_state = _advance(
            functional, population, prepared, ledger_state, expected
        )
        _assert_close(state, ledger_state)
        native.source.mech.source.gate = native_gates[index]
        native.step()
        _assert_close(queue, native_connection.state_dict_for_checkpoint())
        _assert_close(state, functional.extract(native.post).state)

    # The pure run does not depend on live source activity or mutate the model
    # from which either plan was lowered.
    for name, value in network.named_buffers():
        _assert_close(value, snapshots[name])
    actual = torch.autograd.grad(
        _loss(state, queue), (weights, gates), retain_graph=True
    )
    ledger_queue = {"delivery_buffer": torch.stack(tuple(pending.values()))}
    independent = torch.autograd.grad(
        _loss(ledger_state, ledger_queue), (weights, gates)
    )
    expected = torch.autograd.grad(
        _native_loss(native, native_connection), (native_weights, native_gates)
    )
    for observed, reference, original in zip(
        actual, expected, independent, strict=True
    ):
        torch.testing.assert_close(observed, reference, rtol=2e-8, atol=2e-9)
        torch.testing.assert_close(observed, original, rtol=2e-8, atol=2e-9)
        assert torch.isfinite(observed).all()
        assert torch.count_nonzero(observed)


def test_pending_deliveries_keep_enqueue_weights_across_checkpoint_and_commit():
    network, connection = _network(exp2syn, batch=2)
    projection, delivery = dn.func.make_functional_netcon(connection)
    functional, population = dn.func.make_functional(network.post, dt=DT)
    prepared = functional.prepare(population.parameters, population.constants)
    old_weights = delivery.weights.detach().clone().requires_grad_()
    new_weights = (2.3 * delivery.weights).detach().clone().requires_grad_()
    gates = _gates(connection)
    queue, state, pending = delivery.state, population.state, {}
    split = 6
    for index, gate in enumerate(gates[:split]):
        queue, payload = projection.step(old_weights, queue, gate)
        _assert_close(
            payload, _ledger_step(connection, pending, index, old_weights, gate)
        )
        state = _advance(functional, population, prepared, state, payload)
        # Synchronize native clocks before committing the independent snapshot.
        network.source.mech.source.gate = torch.zeros_like(gate)
        network.step()
    checkpoint = _clone(queue)
    checkpoint_state = _clone(state)
    assert torch.count_nonzero(checkpoint["delivery_buffer"])
    projection.commit_state_(connection, checkpoint)
    functional.commit_state_(network.post, checkpoint_state)
    _assert_close(projection.extract(connection).state, checkpoint)
    native_weights = new_weights.detach().clone().requires_grad_()
    connection.weight.w = native_weights
    resumed_queue, resumed_state = _clone(queue), _clone(state)

    for index in range(split, len(gates)):
        queue, payload = projection.step(new_weights, queue, gates[index])
        expected = _ledger_step(connection, pending, index, new_weights, gates[index])
        _assert_close(payload, expected)
        if index == split:
            # This due row was already weighted before the checkpoint.
            pointer = int(checkpoint["current_time_step"].item())
            _assert_close(payload.reshape(-1), checkpoint["delivery_buffer"][pointer])
        state = _advance(functional, population, prepared, state, payload)
        resumed_queue, resumed_payload = projection.step(
            new_weights, resumed_queue, gates[index]
        )
        resumed_state = _advance(
            functional, population, prepared, resumed_state, resumed_payload
        )
        network.source.mech.source.gate = gates[index]
        network.step()
        _assert_close(queue, resumed_queue)
        _assert_close(state, resumed_state)
        _assert_close(queue, connection.state_dict_for_checkpoint())
        _assert_close(state, functional.extract(network.post).state)
        _assert_pending(queue, pending, index + 1)
    # A connected graph through pending deliveries reaches both generations of
    # explicit weights; detached checkpoint replay above reproduces its values.
    gradients = torch.autograd.grad(_loss(state, queue), (old_weights, new_weights))
    assert all(torch.count_nonzero(gradient) for gradient in gradients)
    native_gradient = torch.autograd.grad(
        _native_loss(network, connection), native_weights
    )[0]
    torch.testing.assert_close(gradients[1], native_gradient, rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize("kind", KINDS)
def test_compiled_projection_receive_and_cable_step_preserve_gradients(kind):
    network, connection = _network(kind)
    projection, delivery = dn.func.make_functional_netcon(connection)
    functional, population = dn.func.make_functional(network.post, dt=DT)

    def advance(weights, queue, state, gates):
        queue, payload = projection.step(weights, queue, gates)
        state = functional.prepare_and_receive(
            population.parameters,
            population.constants,
            state,
            target="event",
            payload=payload,
        )
        state, _ = functional.prepare_and_step(
            population.parameters, population.constants, state
        )
        return queue, state

    weights = delivery.weights.detach().clone().requires_grad_()
    gates = _gates(connection)[:5].requires_grad_()
    compiled = torch.compile(advance, backend="aot_eager", fullgraph=True)
    actual_queue, actual_state = delivery.state, population.state
    expected_queue, expected_state = _clone(delivery.state), _clone(population.state)
    for gate in gates:
        actual_queue, actual_state = compiled(weights, actual_queue, actual_state, gate)
        expected_queue, expected_state = advance(
            weights, expected_queue, expected_state, gate
        )
    _assert_close(actual_queue, expected_queue)
    _assert_close(actual_state, expected_state)
    actual = torch.autograd.grad(
        _loss(actual_state, actual_queue), (weights, gates), retain_graph=True
    )
    expected = torch.autograd.grad(
        _loss(expected_state, expected_queue), (weights, gates)
    )
    for observed, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, reference, rtol=2e-8, atol=2e-9)
