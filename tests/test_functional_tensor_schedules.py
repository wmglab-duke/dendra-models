"""Explicit scheduled sources compose with companion cable populations."""

from __future__ import annotations

import dendra as dn
import pytest
import torch
from dendra._bootstrap import reset_torch_compiler
from dendra.models.mod import exp2syn, expsyn
from torch.utils import _pytree as pytree

from dendra_models.models import Rattay1993

DT = 0.01
STEPS = 18


@pytest.fixture(autouse=True)
def _compiler_cache():
    reset_torch_compiler()
    try:
        yield
    finally:
        reset_torch_compiler()


def _leaf(values):
    return torch.tensor(values, dtype=torch.float64, requires_grad=True)


def _network(*, mode="triangular", batch=None, shared_sources=False):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, IMEM=False):
        stimulus = dn.NetStim(N=2, start=1000.0, seed=719)
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
            track_netcon_events=mode != "triangular",
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
        network.train(mode != "eval")
        network.initialize(DT)
        network.set_synaptic_diff_config(
            diff_weights=True,
            diff_delays=False,
            diff_spiking=mode == "triangular",
            diff_scheduled_times=mode == "triangular",
            sched_width=1.3,
            train_delay_backend="dense",
        )
        stimulus = network.netstim
        source_amplitudes = _leaf([0.6, -0.2])
        source_times = _leaf([0.073, 0.096])
        stimulus.bind_weight_source(source_amplitudes)
        stimulus.bind_time_source(source_times)
        stimulus.schedule(
            [0, 1, 0],
            _leaf([0.016, 0.027, 0.016]),
            weight=_leaf([0.3, 0.0, -0.1]),
            differentiable=True,
        )
        stimulus.schedule_ref([1, 0], times=_leaf([0.043, 0.062]), weight_idx=[1, 0])
        stimulus.schedule_time_ref([0, 1], time_idx=[0, 1], weight=_leaf([0.2, 0.4]))
        for connection in network.synapses.values():
            indices = torch.arange(connection._n_conn)
            repeats = connection._n_conn // 3
            connection.bind_weight_source(
                source_amplitudes if shared_sources else _leaf([0.4, -0.15])
            )
            connection.bind_time_source(
                source_times if shared_sources else _leaf([0.084, 0.107])
            )
            connection.schedule(
                con_indices=indices,
                times_ms=_leaf([0.024, 0.035, 0.024]).repeat(repeats),
                weight=_leaf([0.25, 0.0, -0.1]).repeat(repeats),
            )
            connection.schedule_ref(
                con_indices=indices,
                times_ms=_leaf([0.046, 0.057, 0.068]).repeat(repeats),
                weight_idx=torch.tensor([0, 1, 0]).repeat(repeats),
            )
            connection.schedule_time_ref(
                con_indices=indices,
                time_idx=torch.tensor([0, 1, 0]).repeat(repeats),
                weight=_leaf([0.2, -0.1, 0.3]).repeat(repeats),
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

    def step(weights, schedules, carry, time, draw):
        source_state, queues, states = carry
        source_state, outputs = renewal.step(
            source.parameters,
            source.constants,
            source_state,
            time,
            draw,
            bptt=network.netstim.training,
            schedule=schedules[0],
        )
        next_queues, received = [], []
        for projection, function, pop, queue, state, weight, schedule in zip(
            projections,
            functions,
            populations,
            queues,
            states,
            weights,
            schedules[1:],
            strict=True,
        ):
            queue, payload = projection.read_due(queue)
            state = function.prepare_and_receive(
                pop.parameters, pop.constants, state, target="event", payload=payload
            )
            gates = (
                outputs.spike_gate
                if network.training and network.netstim.diff_scheduled_times
                else outputs.spikes.to(projection.source_dtype)
            )
            queue = projection.advance(weight, queue, gates, schedule=schedule)
            next_queues.append(queue)
            received.append(state)
        next_states = tuple(
            function.prepare_and_step(pop.parameters, pop.constants, state)[0]
            for function, pop, state in zip(
                functions, populations, received, strict=True
            )
        )
        return (source_state, tuple(next_queues), next_states), outputs

    schedules = (source.schedule, *(item.schedule for item in deliveries))
    weights = tuple(item.weights for item in deliveries)
    return renewal, source, projections, functions, initial, schedules, weights, step


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


@pytest.mark.parametrize("mode", ("eval", "hard", "triangular"))
@pytest.mark.parametrize("batch", (None, 2))
def test_tensor_schedules_drive_two_companion_populations(mode, batch):
    network = _network(mode=mode, batch=batch)
    renewal, source, projections, functions, carry, schedules, weights, step = _lower(
        network
    )
    initial, saved = carry, _clone(carry)
    origin = network.t.detach().clone()
    for index in range(STEPS):
        time = origin + index * DT
        draw = renewal.sample_step_draw(carry[0])
        carry, outputs = step(weights, schedules, carry, time, draw)
        network.step()
        _assert_native(network, carry, renewal, projections, functions)
        _close(outputs.spikes, network.netstim.spikes)
        _close(outputs.spike_gate, network.netstim.spike_gate)
    _close(initial, saved)
    assert carry[0]["spike_counts"].min() > 0


@pytest.mark.parametrize("batch", (None, 2))
@pytest.mark.parametrize("shared_sources", (False, True))
def test_schedule_time_amplitude_and_bound_source_gradients_match_native(
    batch, shared_sources
):
    network = _network(batch=batch, shared_sources=shared_sources)
    renewal, source, projections, functions, carry, schedules, weights, step = _lower(
        network
    )
    owners = (network.netstim, *network.synapses.values())
    if shared_sources:
        shared_times = schedules[0].time_source
        shared_amplitudes = schedules[0].amplitude_source
        assert shared_times is not network.netstim._sched_t_source
        assert shared_amplitudes is not network.netstim._sched_w_source
        for owner, packet in zip(owners[1:], schedules[1:], strict=True):
            assert owner._sched_t_source is network.netstim._sched_t_source
            assert owner._sched_w_source is network.netstim._sched_w_source
            assert packet.time_source is not shared_times
            assert packet.amplitude_source is not shared_amplitudes
        schedules = tuple(
            packet._replace(
                time_source=shared_times, amplitude_source=shared_amplitudes
            )
            for packet in schedules
        )
        assert all(packet.time_source is shared_times for packet in schedules)
        assert all(packet.amplitude_source is shared_amplitudes for packet in schedules)
    origin = network.t.detach().clone()
    for index in range(STEPS):
        time = origin + index * DT
        carry, _ = step(
            weights, schedules, carry, time, renewal.sample_step_draw(carry[0])
        )
        network.step()
    if shared_sources:
        # Request shared leaves once so their gradients sum every owner's use.
        explicit = (
            *(value for packet in schedules for value in packet[:2]),
            shared_times,
            shared_amplitudes,
        )
        native_schedules = (
            *(
                value
                for owner in owners
                for value in (owner.sched_time_ms, owner.sched_weight)
            ),
            network.netstim._sched_t_source,
            network.netstim._sched_w_source,
        )
    else:
        explicit = tuple(value for packet in schedules for value in packet)
        native_schedules = tuple(
            value
            for owner in owners
            for value in (
                owner.sched_time_ms,
                owner.sched_weight,
                owner._sched_t_source,
                owner._sched_w_source,
            )
        )
    native_weights = tuple(nc.weight.w for nc in network.synapses.values())
    native_loss = sum(
        pop.v.square().sum() for pop in network.populations.values()
    ) + sum(nc.delivery_buffer.square().sum() for nc in network.synapses.values())
    actual = torch.autograd.grad(_loss(carry), (*explicit, *weights))
    expected = torch.autograd.grad(native_loss, (*native_schedules, *native_weights))
    for gradient, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(gradient, reference, rtol=2e-8, atol=2e-9)
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient)
    assert all(parameter.grad is None for parameter in network.parameters())


def test_compiled_schedule_composition_uses_explicit_bindings():
    network = _network(batch=2)
    renewal, _, _, _, carry, schedules, weights, step = _lower(network)
    compiled = torch.compile(step, backend="aot_eager", fullgraph=True)
    expected = _clone(carry)
    origin = network.t.detach().clone()
    for index in range(12):
        time = origin + index * DT
        packet = renewal.sample_step_draw(carry[0])
        carry, output = compiled(weights, schedules, carry, time, packet)
        expected, reference = step(weights, schedules, expected, time, packet)
        _close(carry, expected)
        _close(output, reference)
    targets = tuple(value for schedule in schedules for value in schedule) + weights
    actual = torch.autograd.grad(_loss(carry), targets, retain_graph=True)
    expected_grad = torch.autograd.grad(_loss(expected), targets)
    for gradient, reference in zip(actual, expected_grad, strict=True):
        torch.testing.assert_close(gradient, reference, rtol=2e-8, atol=2e-9)


def test_schedule_checkpoint_continues_in_fresh_native_network():
    network = _network(batch=2)
    renewal, _, projections, functions, carry, schedules, weights, step = _lower(
        network
    )
    origin = network.t.detach().clone()
    for index in range(7):
        time = origin + index * DT
        carry, _ = step(
            weights, schedules, carry, time, renewal.sample_step_draw(carry[0])
        )
        network.step()
    checkpoint = _clone(carry)
    frozen = _clone(checkpoint)
    resumed = _network(batch=2)
    resumed.restore_dict_from_checkpoint(network.state_dict_for_checkpoint())
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
    for index in range(7, STEPS):
        time = origin + index * DT
        carry, _ = step(
            weights, schedules, carry, time, renewal.sample_step_draw(carry[0])
        )
        network.step()
        resumed.step()
        _assert_native(network, carry, renewal, projections, functions)
        _assert_native(resumed, carry, renewal, projections, functions)
    _close(checkpoint, frozen)
