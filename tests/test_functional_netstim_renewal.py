"""One shared renewal source drives several functional companion projections."""

from __future__ import annotations

import dendra as dn
import pytest
import torch
from dendra._bootstrap import reset_torch_compiler
from dendra.models.mod import exp2syn, expsyn
from torch.nn import functional as F
from torch.utils import _pytree as pytree

from dendra_models.models import Rattay1993

DT = 0.01
STEPS = 32
RNG_KEYS = ("rng_state", "seeder_state")


@pytest.fixture(autouse=True)
def _reset_compiler():
    reset_torch_compiler()
    try:
        yield
    finally:
        reset_torch_compiler()


def _network(*, batch=None, bptt=True, surrogate=True):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, IMEM=False):
        source = dn.NetStim(
            N=2,
            interval=[0.025, 0.035],
            start=[0.0, 0.01],
            noise=[0.4, 0.8],
            max_spikes=100,
            tau=0.01,
            seed=719,
        )
        # A fixed NetStim seed governs its renewal stream. Standardize the
        # separate seed=None seeder too, so complete explicit snapshots can be
        # compared across independently constructed fixture networks.
        source._seeder.manual_seed(97)
        populations = {}
        for name, kind in (("fast", expsyn), ("slow", exp2syn)):
            post = Rattay1993(
                diameters=[1.0],
                L=40.0,
                dx=10.0,
                integrator=dn.bwd_euler_ub(method="pcr", imem=False),
            ).double()
            parameters = {"tau": 0.7} if kind is expsyn else {"tau1": 0.3, "tau2": 2.0}
            post[:, torch.tensor([3, 1])].insert(kind.rename("event"), **parameters)
            populations[name] = post
        network = dn.Network(
            populations,
            netstim=source,
            netcon_delay_backend="dense",
            netcon_train_backend="dense",
        )
        for name in populations:
            network.connect_one_to_one(
                source[torch.tensor([0, 1, 1])],
                populations[name][:, torch.tensor([1, 3, 1])],
                populations[name].mech.event,
                weight=torch.tensor([0.0002, 0.0003, 0.0004]),
                delay=torch.tensor([0.0, DT, 3 * DT]),
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
        )
        # Network training controls NetCons independently. NetStim's own mode
        # selects whether renewal clocks retain their temporal autograd graph.
        network.netstim.train(bptt)
    return network


def _clone(tree):
    return pytree.tree_map_only(
        torch.Tensor, lambda value: value.detach().clone(), tree
    )


def _assert_close(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for name in actual:
            _assert_close(actual[name], expected[name])
    elif isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for observed, reference in zip(actual, expected, strict=True):
            _assert_close(observed, reference)
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-10)


def _lower(network):
    source, renewal = dn.func.make_functional_netstim(network.netstim)
    connections = tuple(network.synapses.values())
    projections, deliveries = zip(
        *(dn.func.make_functional_netcon(connection) for connection in connections),
        strict=True,
    )
    populations = tuple(network.populations.values())
    functions, tensors = zip(
        *(dn.func.make_functional(population, dt=DT) for population in populations),
        strict=True,
    )
    return source, renewal, connections, projections, deliveries, functions, tensors


def _advance(
    source,
    projections,
    functions,
    tensors,
    parameters,
    constants,
    weights,
    renewal_state,
    queues,
    population_states,
    time,
    draw,
    *,
    bptt,
    surrogate,
):
    # Generate once per network step, before any connection. Both projections
    # consume this same event; neither projection resamples its own source.
    renewal_state, outputs = source.step(
        parameters, constants, renewal_state, time, draw, bptt=bptt
    )
    queues_out, received = [], []
    for projection, functional, population, weight, queue, state in zip(
        projections,
        functions,
        tensors,
        weights,
        queues,
        population_states,
        strict=True,
    ):
        queue, payload = projection.read_due(queue)
        state = functional.prepare_and_receive(
            population.parameters,
            population.constants,
            state,
            target="event",
            payload=payload,
        )
        gates = outputs.spike_gate if surrogate else outputs.spikes.to(weight.dtype)
        queue = projection.advance(weight, queue, gates)
        queues_out.append(queue)
        received.append(state)
    # Native Network advances populations only after all connections finish.
    states_out = tuple(
        functional.prepare_and_step(population.parameters, population.constants, state)[
            0
        ]
        for functional, population, state in zip(
            functions, tensors, received, strict=True
        )
    )
    return renewal_state, tuple(queues_out), states_out, outputs


def _assert_native(
    network, source, renewal_state, connections, queues, functions, states
):
    native_state = source.extract(network.netstim).state
    _assert_close(renewal_state, native_state)
    for name in RNG_KEYS:
        assert torch.equal(renewal_state[name], native_state[name])
    for connection, queue in zip(connections, queues, strict=True):
        _assert_close(queue, connection.state_dict_for_checkpoint())
    for functional, population, state in zip(
        functions, network.populations.values(), states, strict=True
    ):
        _assert_close(state, functional.extract(population).state)


def _loss(states, queues):
    result = 0
    for state, queue in zip(states, queues, strict=True):
        result = result + 0.001 * state["integrator"]["v"].square().sum()
        result = result + 100.0 * sum(
            value.square().sum() for value in state["mechanisms"]["event"].values()
        )
        result = result + 100.0 * queue["delivery_buffer"].square().sum()
    return result


@pytest.mark.parametrize("batch", (None, 2))
@pytest.mark.parametrize("bptt", (False, True))
@pytest.mark.parametrize("surrogate", (False, True))
def test_seeded_renewal_drives_multiple_cables_with_native_gradients(
    batch, bptt, surrogate
):
    network = _network(batch=batch, bptt=bptt, surrogate=surrogate)
    native = _network(batch=batch, bptt=bptt, surrogate=surrogate)
    source, renewal, _, projections, deliveries, functions, tensors = _lower(network)
    native_connections = tuple(native.synapses.values())
    raw_interval = network.netstim.interval.rho.detach().clone().requires_grad_()
    parameters = dict(renewal.parameters, interval=F.softplus(raw_interval))
    weights = tuple(
        delivery.weights.detach().clone().requires_grad_() for delivery in deliveries
    )
    native_weights = tuple(
        weight.detach().clone().requires_grad_() for weight in weights
    )
    for connection, weight in zip(native_connections, native_weights, strict=True):
        connection.weight.w = weight
    renewal_state = renewal.state
    queues = tuple(delivery.state for delivery in deliveries)
    states = tuple(population.state for population in tensors)
    original = _clone(dict(network.named_buffers()))
    original_rng = network.netstim._rng.get_state().clone()
    spikes = []

    for index in range(STEPS):
        draw = source.sample_step_draw(renewal_state)
        if batch is not None:
            assert not torch.equal(draw.exp_rand[0], draw.exp_rand[1])
        time = parameters["interval"].new_tensor(index * DT)
        renewal_state, queues, states, outputs = _advance(
            source,
            projections,
            functions,
            tensors,
            parameters,
            renewal.constants,
            weights,
            renewal_state,
            queues,
            states,
            time,
            draw,
            bptt=bptt,
            surrogate=surrogate,
        )
        native.step()
        _assert_native(
            native, source, renewal_state, native_connections, queues, functions, states
        )
        _assert_close(outputs.spikes, native.netstim.spikes)
        _assert_close(outputs.spike_gate, native.netstim.spike_gate)
        spikes.append(outputs.spikes)

    assert torch.stack(spikes).sum(dim=0).min() >= 5
    for name, value in network.named_buffers():
        _assert_close(value, original[name])
    assert torch.equal(network.netstim._rng.get_state(), original_rng)
    native_states = tuple(
        {
            "integrator": {"v": population.v},
            "mechanisms": {
                "event": {
                    name: getattr(population.mech.event, name)
                    for name in population.mech.event._all_states
                }
            },
        }
        for population in native.populations.values()
    )
    native_queues = tuple(
        connection.state_dict_for_checkpoint() for connection in native_connections
    )
    actual = torch.autograd.grad(
        _loss(states, queues), (raw_interval, *weights), allow_unused=True
    )
    expected = torch.autograd.grad(
        _loss(native_states, native_queues),
        (native.netstim.interval.rho, *native_weights),
        allow_unused=True,
    )
    for observed, reference in zip(actual, expected, strict=True):
        if reference is None:
            assert observed is None
        else:
            torch.testing.assert_close(observed, reference, rtol=2e-8, atol=2e-9)
            assert torch.isfinite(observed).all()
            assert torch.count_nonzero(observed)
    assert (actual[0] is not None) is (bptt and surrogate)


@pytest.mark.parametrize("bptt", (False, True))
def test_compiled_batched_renewal_receive_and_cable_step(bptt):
    network = _network(batch=2, bptt=bptt)
    source, renewal, _, projections, deliveries, functions, tensors = _lower(network)

    def advance(parameters, weights, renewal_state, queues, states, time, draw):
        return _advance(
            source,
            projections,
            functions,
            tensors,
            parameters,
            renewal.constants,
            weights,
            renewal_state,
            queues,
            states,
            time,
            draw,
            bptt=bptt,
            surrogate=True,
        )

    compiled = torch.compile(advance, backend="aot_eager", fullgraph=True)
    weights = tuple(
        delivery.weights.detach().clone().requires_grad_() for delivery in deliveries
    )
    parameters = dict(renewal.parameters)
    parameters["interval"] = parameters["interval"].detach().clone().requires_grad_()
    current = (
        renewal.state,
        tuple(delivery.state for delivery in deliveries),
        tuple(population.state for population in tensors),
    )
    expected = _clone(current)
    for index in range(12):
        draw = source.sample_step_draw(current[0])
        time = parameters["interval"].new_tensor(index * DT)
        *current, outputs = compiled(parameters, weights, *current, time, draw)
        *expected, reference = advance(parameters, weights, *expected, time, draw)
        _assert_close(current, expected)
        _assert_close(outputs.spikes, reference.spikes)
        _assert_close(outputs.spike_gate, reference.spike_gate)
    actual_grad = torch.autograd.grad(
        _loss(current[2], current[1]),
        (parameters["interval"], *weights),
        allow_unused=True,
        retain_graph=True,
    )
    expected_grad = torch.autograd.grad(
        _loss(expected[2], expected[1]),
        (parameters["interval"], *weights),
        allow_unused=True,
    )
    for observed, reference in zip(actual_grad, expected_grad, strict=True):
        if reference is None:
            # AOTAutograd may materialize a zero for an unused tensor input.
            assert observed is None or not torch.count_nonzero(observed)
        else:
            torch.testing.assert_close(observed, reference, rtol=2e-8, atol=2e-9)


@pytest.mark.parametrize("bptt", (False, True))
def test_clock_and_rng_checkpoint_resume_into_native_network(bptt):
    network = _network(batch=2, bptt=bptt)
    source, renewal, _, projections, deliveries, functions, tensors = _lower(network)
    weights = tuple(delivery.weights for delivery in deliveries)
    renewal_state = renewal.state
    queues = tuple(delivery.state for delivery in deliveries)
    states = tuple(population.state for population in tensors)
    split = 9
    for index in range(split):
        draw = source.sample_step_draw(renewal_state)
        time = renewal.parameters["interval"].new_tensor(index * DT)
        renewal_state, queues, states, _ = _advance(
            source,
            projections,
            functions,
            tensors,
            renewal.parameters,
            renewal.constants,
            weights,
            renewal_state,
            queues,
            states,
            time,
            draw,
            bptt=bptt,
            surrogate=True,
        )
        network.step()
    checkpoint = _clone((renewal_state, queues, states))
    assert checkpoint[0]["spike_counts"].min() > 0
    assert any(torch.count_nonzero(queue["delivery_buffer"]) for queue in queues)
    # Continue from the saved state in a fresh, identically parameterized
    # Network. Its own initialized RNG and clocks are deliberately discarded.
    resumed = _network(batch=2, bptt=bptt)
    network_checkpoint = network.state_dict_for_checkpoint()
    resumed.restore_dict_from_checkpoint(network_checkpoint)
    # Discard an actual divergent clock/stream, rather than recommitting the
    # same source state that the whole-Network checkpoint just installed.
    resumed.netstim(1000.0, bptt=True)
    assert not torch.equal(
        resumed.netstim.state_dict_for_checkpoint()["rng_state"],
        checkpoint[0]["rng_state"],
    )
    source.commit_state_(resumed.netstim, checkpoint[0])
    for projection, connection, queue in zip(
        projections, resumed.synapses.values(), checkpoint[1], strict=True
    ):
        projection.commit_state_(connection, queue)
    for functional, population, state in zip(
        functions, resumed.populations.values(), checkpoint[2], strict=True
    ):
        functional.commit_state_(population, state)
    _assert_native(
        resumed,
        source,
        checkpoint[0],
        tuple(resumed.synapses.values()),
        checkpoint[1],
        functions,
        checkpoint[2],
    )
    renewal_state, queues, states = checkpoint
    counts_at_split = renewal_state["spike_counts"].clone()
    saved_rng = renewal_state["rng_state"].clone()
    for index in range(split, STEPS):
        # New draws come from the checkpoint RNG; no saved noise tape is used.
        draw = source.sample_step_draw(renewal_state)
        time = renewal.parameters["interval"].new_tensor(index * DT)
        renewal_state, queues, states, outputs = _advance(
            source,
            projections,
            functions,
            tensors,
            renewal.parameters,
            renewal.constants,
            weights,
            renewal_state,
            queues,
            states,
            time,
            draw,
            bptt=bptt,
            surrogate=True,
        )
        resumed.step()
        _assert_native(
            resumed,
            source,
            renewal_state,
            tuple(resumed.synapses.values()),
            queues,
            functions,
            states,
        )
        _assert_close(outputs.spikes, resumed.netstim.spikes)
    assert (renewal_state["spike_counts"] > counts_at_split).all()
    assert not torch.equal(renewal_state["rng_state"], saved_rng)
    _assert_close(checkpoint[0]["spike_counts"], counts_at_split)
    assert torch.equal(checkpoint[0]["rng_state"], saved_rng)


def test_pure_initialization_resets_seeded_source_without_mutation():
    network = _network(batch=2)
    source, renewal, *_ = _lower(network)
    original = _clone(renewal.state)
    state = renewal.state
    for index in range(7):
        draw = source.sample_step_draw(state)
        state, _ = source.step(
            renewal.parameters,
            renewal.constants,
            state,
            renewal.parameters["interval"].new_tensor(index * DT),
            draw,
            bptt=True,
        )
    draw = source.sample_initialization_draw(renewal.parameters, state)
    initialized = source.initial_state(
        renewal.parameters, renewal.constants, state, draw
    )
    _assert_close(initialized, original)
    _assert_close(source.extract(network.netstim).state, original)
    assert initialized["spike_counts"].count_nonzero() == 0
    assert torch.isneginf(initialized["t_last"]).all()
