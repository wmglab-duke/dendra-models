"""Connected model inputs preserve native timing and functional continuation."""

from copy import deepcopy

import dendra as dn
import pytest
import torch

from dendra_models.models.networks.yu_2024.net import yu_2024
from dendra_models.models.networks.yu_2024.params import params as yu_parameters


DT = 0.025
YU_CURRENTS = (
    ("i_ampa", "AMPA"),
    ("i_nmda", "NMDA"),
    ("i_gaba_a", "GABAA"),
    ("i_gaba_b", "GABAB"),
)


def _assert_tree(actual, expected):
    if torch.is_tensor(actual):
        torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    elif isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for name in actual:
            _assert_tree(actual[name], expected[name])
    elif isinstance(actual, (tuple, list)):
        assert type(actual) is type(expected)
        for value, reference in zip(actual, expected, strict=True):
            _assert_tree(value, reference)
    else:
        assert actual == expected


def _yu_model(*, training=False, batch=None):
    params = deepcopy(yu_parameters)
    params["diameter_column"] = 100
    params["spacing_column"] = 100
    params["cells"]["n_per_column"] = {
        name: 1 for name in params["cells"]["celltypes"]
    }
    params["netstim"]["n_per_group"] = {
        name: 1 for name in params["cells"]["celltypes"]
    }
    params["depth"] = {
        name: (10.0 * index, 10.0 * index)
        for index, name in enumerate(params["cells"]["celltypes"])
    }
    params["syn_delay"] = DT
    # Keep actual recurrent connections and their parametrized distance delays.
    for probabilities in params["con"]["p"].values():
        for post in probabilities:
            probabilities[post] = 1.0
    torch.manual_seed(2718)
    with dn.ctx(JIT=0, DTYPE=torch.float64, REQUIRE_GRAD=1, DEVICE="cpu"):
        network = yu_2024(params=params)
        if batch is not None:
            network.batch(batch)
        network.train(training)
        for source in range(network.netstim.N):
            network.netstim.schedule(source, [DT])
        network.initialize(DT)
        if training:
            network.set_synaptic_diff_config(
                diff_weights=True,
                diff_delays=True,
                diff_spiking=True,
                train_delay_backend="dense",
            )
    return network


def _seed_yu(network):
    mechanism = network.cells.mech.esser_mech
    mechanism.v_iaf = torch.full_like(mechanism.v_iaf, -46.0)
    mechanism.theta = torch.full_like(mechanism.theta, -53.0)
    for index, (_, name) in enumerate(YU_CURRENTS, 1):
        synapse = getattr(network.cells.mech, name)
        synapse.A = torch.full_like(synapse.A, index * 0.001)
        synapse.B = torch.full_like(synapse.B, index * 0.003)
    # State INPUT reads the saved current, not a recomputation at reset voltage.
    network.cells.mech.i(network.cells.v)


def _legacy_yu(network):
    population = network.cells
    population.mech.configure_input_bindings(())
    population._input_bindings = ()
    owner = population.mech
    state = owner.esser_mech.DE["esser_states"]
    for name, source in YU_CURRENTS:
        state.setreference(
            name, lambda source=source: getattr(owner, source).i_
        )
    return network


@pytest.mark.parametrize("training", (False, True))
@pytest.mark.parametrize("batch", (None, 2))
def test_yu_bound_currents_match_native_and_original_saved_current_references(
    training, batch
):
    network = _yu_model(training=training, batch=batch)
    _seed_yu(network)
    source_checkpoint = network.state_dict_for_checkpoint()
    native = deepcopy(network)
    legacy = _legacy_yu(deepcopy(network))
    functional, tensors = dn.func.make_functional(network, dt=DT)
    bound = functional.bind(tensors)
    state = tensors.state
    for _ in range(4):
        draws = functional.sample_runtime_draws(state)
        state, _ = bound(state, inputs={}, draws=draws)
        native.step(draws=draws)
        legacy.step(draws=draws)
        torch.testing.assert_close(
            state, functional.extract(native).state, rtol=2e-11, atol=2e-12
        )
        _assert_tree(
            native.state_dict_for_checkpoint(),
            legacy.state_dict_for_checkpoint(),
        )
    _assert_tree(network.state_dict_for_checkpoint(), source_checkpoint)


def test_yu_current_binding_declarations_survive_rebuild_and_fresh_initialization():
    network = _yu_model(training=True)
    before = tuple(network.cells._input_bindings)
    network.batch(2)
    network.initialize(DT)
    network.set_synaptic_diff_config(
        diff_weights=True,
        diff_delays=True,
        diff_spiking=True,
        train_delay_backend="dense",
    )
    assert tuple(network.cells._input_bindings) == before
    assert len(before) == 4
    functional, tensors = dn.func.make_functional(network, dt=DT)
    fresh = dn.func.initialize(functional, tensors)
    native = deepcopy(network)
    native.initialize(DT)
    torch.testing.assert_close(
        fresh.state, functional.extract(native).state, rtol=2e-11, atol=2e-12
    )


def test_yu_compiled_current_gradient_and_portable_checkpoint_continuation():
    network = _yu_model(training=True)
    _seed_yu(network)
    network.detach()
    source_checkpoint = network.state_dict_for_checkpoint()
    native = deepcopy(network)
    functional, tensors = dn.func.make_functional(network, dt=DT)
    compiled = torch.compile(
        functional.prepare_and_step, backend="aot_eager", fullgraph=True
    )
    state = tensors.state
    restored = None
    for index in range(12):
        draws = functional.sample_runtime_draws(state)
        state, _ = compiled(
            tensors.parameters, tensors.constants, state, inputs={}, draws=draws
        )
        native.step(draws=draws)
        if restored is not None:
            resumed_state, _ = restored_bound(
                restored.state, inputs={}, draws=draws
            )
            restored = dn.func.Continuation(
                resumed_state, restored.parameters, restored.constants
            )
        if index == 1:
            continuation = dn.func.Continuation(
                state, tensors.parameters, tensors.constants
            )
            packet = dn.func.export_continuation(functional, continuation)
            restored = dn.func.restore_continuation(functional, packet)
            restored_bound = functional.bind(restored)
    torch.testing.assert_close(
        state, functional.extract(native).state, rtol=2e-11, atol=2e-12
    )
    torch.testing.assert_close(state, restored.state, rtol=2e-11, atol=2e-12)
    parameter = next(
        value
        for name, value in tensors.named_parameters.items()
        if name.endswith(".AMPA.e_param")
    )
    native_parameter = native.cells.mech.AMPA.e_param
    owned_voltage = state["populations"]["cells"]["mechanisms"]["esser_mech"]["v_iaf"]
    gradient = torch.autograd.grad(owned_voltage.sum(), parameter)[0]
    native_gradient = torch.autograd.grad(
        native.cells.mech.esser_mech.v_iaf.sum(), native_parameter
    )[0]
    assert gradient.abs().sum() > 0
    torch.testing.assert_close(gradient, native_gradient, rtol=2e-9, atol=2e-11)
    _assert_tree(network.state_dict_for_checkpoint(), source_checkpoint)
