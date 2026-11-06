"""Consumed legacy events share the pure renewal engine on companion cables."""

from __future__ import annotations

import dendra as dn
import pytest
import torch
from dendra._bootstrap import reset_torch_compiler
from dendra.models.mod import exp2syn, expsyn
from torch.utils import _pytree as pytree

from dendra_models.models import Rattay1993

DT = 0.01
STEPS = 20


@pytest.fixture(autouse=True)
def _compiler_cache():
    reset_torch_compiler()
    try:
        yield
    finally:
        reset_torch_compiler()


def _network(*, batch=None, mixed=False, capped=False, bptt=True):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, IMEM=False):
        stimulus = dn.NetStim(
            N=2,
            interval=[0.035, 0.055],
            noise=[0.4, 0.8],
            start=0.0,
            max_spikes=[4, 5] if capped else 100,
            tau=0.01,
            seed=719,
        )
        # The active fixed-seed stream and the checkpointed future seeder are
        # independent. Align both when comparing separately built networks.
        stimulus._seeder.manual_seed(97)
        populations = {}
        for name, mechanism in (("fast", expsyn), ("slow", exp2syn)):
            cable = Rattay1993(
                diameters=[1.0],
                L=40.0,
                dx=10.0,
                integrator=dn.bwd_euler_ub(method="pcr", imem=False),
            ).double()
            kinetics = (
                {"tau": 0.7} if mechanism is expsyn else {"tau1": 0.3, "tau2": 2.0}
            )
            cable[:, torch.tensor([3, 1])].insert(mechanism.rename("event"), **kinetics)
            populations[name] = cable
        network = dn.Network(
            populations,
            netstim=stimulus,
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        for population in populations.values():
            network.connect_one_to_one(
                stimulus[torch.tensor([0, 1, 1])],
                population[:, torch.tensor([1, 3, 1])],
                population.mech.event,
                weight=torch.tensor([0.0002, 0.0003, 0.0004]),
                delay=torch.tensor([0.0, DT, 3 * DT]),
                allow_multapses=True,
            )
        if batch is not None:
            network.batch(batch)
        network.train()
        network.initialize(DT)
        network.netstim.train(bptt)
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=True,
            diff_scheduled_times=mixed,
            sched_width=1.3,
            train_delay_backend="dense",
        )
        stimulus = network.netstim
        for row, times in enumerate(
            (
                [0.015, 0.015, 0.045, 0.075, 0.11, 0.31],
                [0.025, 0.025, 0.055, 0.095, 0.15, 0.32],
            )
        ):
            stimulus.schedule([row] * len(times), times)
        if batch is not None:
            stimulus.schedule([(0, 0), (1, 1)], [0.037, 0.048])
        if mixed:
            stimulus.schedule(
                [0, 1], [0.063, 0.087], weight=[0.2, -0.1], differentiable=True
            )
            for connection in network.synapses.values():
                repeats = connection._n_conn // 3
                connection.schedule(
                    con_indices=torch.arange(connection._n_conn),
                    times_ms=torch.tensor([0.074, 0.105, 0.13]).repeat(repeats),
                    weight=torch.tensor([0.3, -0.1, 0.2]).repeat(repeats),
                )
    return network


def _clone(value):
    return pytree.tree_map_only(torch.Tensor, lambda item: item.detach().clone(), value)


def _close(actual, expected):
    values, spec = pytree.tree_flatten(actual)
    references, reference_spec = pytree.tree_flatten(expected)
    assert spec == reference_spec
    for value, reference in zip(values, references, strict=True):
        torch.testing.assert_close(value, reference, rtol=2e-9, atol=2e-10)


def _lower(network):
    renewal, source = dn.func.make_functional_netstim(network.netstim)
    projections, deliveries = zip(
        *(dn.func.make_functional_netcon(nc) for nc in network.synapses.values()),
        strict=True,
    )
    functions, populations = zip(
        *(dn.func.make_functional(pop, dt=DT) for pop in network.populations.values()),
        strict=True,
    )
    initial = (
        source.state,
        tuple(item.state for item in deliveries),
        tuple(item.state for item in populations),
    )
    weights = tuple(item.weights for item in deliveries)

    def step(weights, carry, time, draw):
        source_state, queues, states = carry
        source_state, outputs = renewal.step(
            source.parameters,
            source.constants,
            source_state,
            time,
            draw,
            bptt=network.netstim.training,
            schedule=source.schedule,
        )
        next_queues, received = [], []
        for projection, function, pop, queue, state, weight, delivery in zip(
            projections,
            functions,
            populations,
            queues,
            states,
            weights,
            deliveries,
            strict=True,
        ):
            queue, payload = projection.read_due(queue)
            state = function.prepare_and_receive(
                pop.parameters, pop.constants, state, target="event", payload=payload
            )
            queue = projection.advance(
                weight, queue, outputs.spike_gate, schedule=delivery.schedule
            )
            next_queues.append(queue)
            received.append(state)
        next_states = tuple(
            function.prepare_and_step(pop.parameters, pop.constants, state)[0]
            for function, pop, state in zip(
                functions, populations, received, strict=True
            )
        )
        return (source_state, tuple(next_queues), next_states), outputs

    return renewal, source, projections, functions, initial, weights, step


def _assert_native(network, carry, renewal, projections, functions):
    _close(carry[0], renewal.extract(network.netstim).state)
    for queue, projection, native in zip(
        carry[1], projections, network.synapses.values(), strict=True
    ):
        _close(queue, projection.extract(native).state)
    for state, function, native in zip(
        carry[2], functions, network.populations.values(), strict=True
    ):
        _close(state, function.extract(native).state)


def _loss(carry):
    return sum(state["integrator"]["v"].square().sum() for state in carry[2]) + sum(
        queue["delivery_buffer"].square().sum() for queue in carry[1]
    )


@pytest.mark.parametrize("batch", (None, 2))
@pytest.mark.parametrize("mixed,capped", ((False, False), (True, False), (True, True)))
@pytest.mark.parametrize("bptt", (False, True))
def test_heap_sources_drive_two_companion_populations(batch, mixed, capped, bptt):
    network = _network(batch=batch, mixed=mixed, capped=capped, bptt=bptt)
    renewal, _, projections, functions, carry, weights, step = _lower(network)
    initial, saved = carry, _clone(carry)
    origin = network.t.detach().clone()
    for index in range(STEPS):
        time = origin + origin.new_tensor(index) * DT
        carry, outputs = step(weights, carry, time, renewal.sample_step_draw(carry[0]))
        network.step()
        _assert_native(network, carry, renewal, projections, functions)
        _close(outputs.spikes, network.netstim.spikes)
        _close(outputs.spike_gate, network.netstim.spike_gate)
    _close(initial, saved)
    assert torch.any(carry[0]["heap_cursor"] > saved[0]["heap_cursor"])
    if capped:
        assert (carry[0]["spike_counts"] == network.netstim.max_spikes).all()


@pytest.mark.parametrize("batch", (None, 2))
@pytest.mark.parametrize("bptt", (False, True))
def test_heap_renewal_interval_and_projection_weight_gradients_match_native(
    batch, bptt
):
    network = _network(batch=batch, mixed=True, bptt=bptt)
    renewal, source, _, _, carry, weights, step = _lower(network)
    origin = network.t.detach().clone()
    for index in range(STEPS):
        time = origin + origin.new_tensor(index) * DT
        carry, _ = step(weights, carry, time, renewal.sample_step_draw(carry[0]))
        network.step()
    actual = torch.autograd.grad(
        _loss(carry), (source.parameters["interval"], *weights), allow_unused=True
    )
    expected = torch.autograd.grad(
        sum(pop.v.square().sum() for pop in network.populations.values())
        + sum(nc.delivery_buffer.square().sum() for nc in network.synapses.values()),
        (
            network.netstim.interval.rho,
            *(nc.weight.w for nc in network.synapses.values()),
        ),
        allow_unused=True,
    )
    for index, (gradient, reference) in enumerate(zip(actual, expected, strict=True)):
        if reference is None:
            assert gradient is None
            continue
        if index == 0:
            gradient = gradient * torch.sigmoid(network.netstim.interval.rho)
        torch.testing.assert_close(gradient, reference, rtol=2e-8, atol=2e-9)
        assert torch.isfinite(gradient).all()
        if bptt or index != 0:
            assert torch.count_nonzero(gradient)


def test_compiled_heap_composition_matches_native_and_eager_gradients():
    network = _network(batch=2, mixed=True)
    renewal, source, projections, functions, carry, weights, step = _lower(network)
    compiled = torch.compile(step, backend="aot_eager", fullgraph=True)
    expected = _clone(carry)
    origin = network.t.detach().clone()
    for index in range(12):
        time = origin + origin.new_tensor(index) * DT
        packet = renewal.sample_step_draw(carry[0])
        carry, output = compiled(weights, carry, time, packet)
        expected, reference = step(weights, expected, time, packet)
        network.step()
        _close(carry, expected)
        _close(output, reference)
        _assert_native(network, carry, renewal, projections, functions)
    targets = (source.parameters["interval"], *weights)
    actual = torch.autograd.grad(_loss(carry), targets, retain_graph=True)
    expected_gradients = torch.autograd.grad(_loss(expected), targets)
    for gradient, reference in zip(actual, expected_gradients, strict=True):
        torch.testing.assert_close(gradient, reference, rtol=2e-8, atol=2e-9)


def test_heap_checkpoint_continues_in_fresh_native_network():
    network = _network(batch=2, mixed=True)
    renewal, _, projections, functions, carry, weights, step = _lower(network)
    origin = network.t.detach().clone()
    for index in range(8):
        time = origin + origin.new_tensor(index) * DT
        carry, _ = step(weights, carry, time, renewal.sample_step_draw(carry[0]))
        network.step()
    checkpoint, frozen = _clone(carry), _clone(carry)
    resumed = _network(batch=2, mixed=True)
    resumed.restore_dict_from_checkpoint(network.state_dict_for_checkpoint())
    # Deliberately consume the remaining calendar and RNG before committing
    # component carry, so continuation verifies an actual cursor/RNG restore.
    resumed.netstim(1000.0, bptt=True)
    renewal.commit_state_(resumed.netstim, checkpoint[0])
    for projection, nc, queue in zip(
        projections, resumed.synapses.values(), checkpoint[1], strict=True
    ):
        projection.commit_state_(nc, queue)
    for function, pop, state in zip(
        functions, resumed.populations.values(), checkpoint[2], strict=True
    ):
        function.commit_state_(pop, state)
    carry = checkpoint
    for index in range(8, STEPS):
        time = origin + origin.new_tensor(index) * DT
        carry, _ = step(weights, carry, time, renewal.sample_step_draw(carry[0]))
        network.step()
        resumed.step()
        _assert_native(network, carry, renewal, projections, functions)
        _assert_native(resumed, carry, renewal, projections, functions)
    _close(checkpoint, frozen)


def test_source_reinitialization_preserves_remaining_events_inside_network():
    model = _network(batch=2, capped=True)
    native = _network(batch=2, capped=True)
    renewal, source, projections, functions, carry, weights, step = _lower(model)
    origin = native.t.detach().clone()
    for index in range(8):
        time = origin + origin.new_tensor(index) * DT
        carry, _ = step(weights, carry, time, renewal.sample_step_draw(carry[0]))
        native.step()
    cursor = carry[0]["heap_cursor"].clone()
    reset = renewal.initial_state(
        source.parameters,
        source.constants,
        carry[0],
        renewal.sample_initialization_draw(source.parameters, carry[0]),
    )
    native.netstim.initialize()
    _close(reset["heap_cursor"], cursor)
    assert not reset["spike_counts"].any()
    carry = (reset, carry[1], carry[2])
    _assert_native(native, carry, renewal, projections, functions)
    for index in range(8, STEPS):
        time = origin + origin.new_tensor(index) * DT
        carry, _ = step(weights, carry, time, renewal.sample_step_draw(carry[0]))
        native.step()
        _assert_native(native, carry, renewal, projections, functions)
