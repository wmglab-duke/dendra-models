"""Cortical current inputs preserve the native Izhikevich update phase."""

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
from dendra_models.models.networks.kumaravelu_2016.mechanisms.synapses import (
    alpha_syn_event,
)

DT = 0.01
STEPS = 4
TARGETS = {
    "ctx_rs": ("regular_spiking_cortex_states", ("FS_RS", "TH_CTX")),
    "ctx_fs": ("fast_spiking_interneuron_states", ("RS_FS",)),
}
BINDINGS = (
    ("ctx_rs", "regular_spiking_cortex_states", "ie", "FS_RS"),
    ("ctx_rs", "regular_spiking_cortex_states", "thcor", "TH_CTX"),
    ("ctx_fs", "fast_spiking_interneuron_states", "ei", "RS_FS"),
)


@pytest.fixture(autouse=True)
def _reset_compiler():
    reset_torch_compiler()
    try:
        yield
    finally:
        reset_torch_compiler()


def _model(*, batch=None, legacy=False):
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, IMEM=False):
        model = dn.Population(
            N=1, C=6, dtype=torch.float64, integrator=dn.scnv(imem=False)
        )
        rs, fs = model[0, :3], model[0, 3:]
        rs.insert(regular_spiking_cortex.rename("ctx_rs"), i_stim=8.0)
        fs.insert(fast_spiking_interneuron.rename("ctx_fs"), i_stim=7.0)
        for target, name, tau, reversal in (
            (rs, "TH_CTX", 4.0, 0.0),
            (rs, "FS_RS", 5.0, -85.0),
            (fs, "RS_FS", 6.0, 0.0),
        ):
            # The original cortical closures ignore current_scale. Keeping a
            # conspicuous value here detects accidentally changing that rule.
            target.insert(
                alpha_syn_event.rename(name),
                tau=tau,
                e=reversal,
                current_scale=7.0,
            )
        if not legacy:
            for target, state, suffix, source in BINDINGS:
                for name in ("s", "e"):
                    model.bind_input(
                        f"{target}.DE.{state}.{name}_{suffix}", f"{source}.{name}"
                    )
        if batch is not None:
            model.batch(batch)
        model.train()
        model.initialize()
    if legacy:
        for target, state, suffix, source in BINDINGS:
            consumer = getattr(model.mech, target)
            provider = getattr(model.mech, source)
            reversal = {"FS_RS": -85.0, "TH_CTX": 0.0, "RS_FS": 0.0}[source]
            consumer.DE[state].setreference(
                "i_" + suffix,
                lambda consumer=consumer, provider=provider, reversal=reversal: (
                    provider.s * (consumer.v_izh - reversal)
                ),
            )
    _seed(model)
    return model


def _seed(model):
    for index, name in enumerate(TARGETS):
        mechanism = getattr(model.mech, name)
        voltage = model.v.new_tensor(
            [[-9.999, 30.0, 31.0]] if index == 0 else [[-10.001, 29.999, 30.0]]
        )
        previous = model.v.new_tensor([[-11.0, -11.0, -9.0]])
        mechanism.v_izh = voltage.expand_as(mechanism.v_izh).clone()
        mechanism.v_prev = previous.expand_as(mechanism.v_prev).clone()
        mechanism.u = (
            torch.linspace(-13.0, -10.0, mechanism.u.numel(), dtype=model.dtype())
            .reshape_as(mechanism.u)
            .to(model.device())
        )
    for index, name in enumerate(("TH_CTX", "FS_RS", "RS_FS"), 1):
        mechanism = getattr(model.mech, name)
        offsets = torch.arange(
            mechanism.s.numel(), device=model.device(), dtype=model.dtype()
        ).reshape_as(mechanism.s)
        mechanism.s = index * 0.03 + offsets * 0.002
        mechanism.z = index * 0.01 + offsets * 0.001


def _clone(tree):
    return pytree.tree_map_only(torch.Tensor, lambda x: x.detach().clone(), tree)


def _assert_close(actual, expected):
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for name in actual:
            _assert_close(actual[name], expected[name])
    else:
        torch.testing.assert_close(actual, expected, rtol=2e-9, atol=2e-10)


def _parameter_key(parameters, mechanism, name, state=None):
    suffix = f".{mechanism}."
    if state is not None:
        suffix += f"DE.{state}."
    suffix += name if name in {"tau_gate", "ste_scale"} else name + "_param"
    # Slice insertions retain a whole-population base parameter and encode
    # this fixture's one regional override as ``_0``. The override owns all
    # entries in the mechanism's local frame.
    overridden = [key for key in parameters if key.endswith(suffix + "_0")]
    matches = overridden or [key for key in parameters if key.endswith(suffix)]
    assert len(matches) == 1, suffix
    return matches[0]


def _parameter(parameters, mechanism, name, state=None):
    return parameters[_parameter_key(parameters, mechanism, name, state)]


def _reset(voltage, u, previous, parameters, name):
    """Original hard forward and surrogate backward equations, independently."""

    def p(field):
        return _parameter(parameters, name, field)

    reset = voltage >= p("v_peak")
    crossed = (previous < p("syn_threshold")) & (voltage >= p("syn_threshold"))
    tau = p("tau_gate").clamp_min(1e-6)
    gate = torch.sigmoid((voltage - p("v_peak")) / tau)
    now = torch.sigmoid((voltage - p("syn_threshold")) / tau)
    before = torch.sigmoid((previous - p("syn_threshold")) / tau)
    crossing_gate = torch.relu(now - before)
    exposed = torch.where(reset, torch.zeros_like(voltage) + p("c"), voltage)
    return {
        "v_izh": exposed,
        "u": torch.where(reset, u + p("d"), u),
        "v_prev": exposed,
        "spikes": reset.to(voltage.dtype) + p("ste_scale") * (gate - gate.detach()),
        "syn_spikes": crossed.to(voltage.dtype)
        + p("ste_scale") * (crossing_gate - crossing_gate.detach()),
    }


def _recurrence(state, parameters, dt, steps):
    """Reset, consume old conductance at reset voltage, then advance states."""
    for _ in range(steps):
        owned, carry, exposed = {}, {}, []
        for name, (state_name, providers) in TARGETS.items():
            old = state["mechanisms"][name]
            previous = state["mechanism_buffers"][name]["v_prev"]
            reset = _reset(old["v_izh"], old["u"], previous, parameters, name)
            voltage, u = reset["v_izh"], reset["u"]
            current = sum(
                state["mechanisms"][provider]["s"]
                * (voltage - _parameter(parameters, provider, "e"))
                for provider in providers
            )
            drive = _parameter(parameters, name, "i_stim", state_name)
            a = _parameter(parameters, name, "a", state_name)
            b = _parameter(parameters, name, "b", state_name)
            rhs = 0.04 * voltage.square() + 5.0 * voltage + 140.0 - u - current + drive
            owned[name] = {
                "v_izh": voltage + dt * rhs,
                "u": u - torch.expm1(-a * dt) * (b * voltage - u),
            }
            carry[name] = {
                field: reset[field] for field in ("spikes", "syn_spikes", "v_prev")
            }
            exposed.append(voltage)
        for name in ("TH_CTX", "FS_RS", "RS_FS"):
            old = state["mechanisms"][name]
            tau = _parameter(parameters, name, "tau", "alpha_state")
            increment = -torch.expm1(-2.0 * dt / tau)
            owned[name] = {
                "s": old["s"] + dt * old["z"],
                "z": old["z"] - increment * (old["z"] + old["s"] / (2.0 * tau)),
            }
        state = {
            **state,
            "integrator": {"v": torch.cat(exposed, dim=-1)},
            "mechanisms": owned,
            "mechanism_buffers": carry,
            "clock": {"t": state["clock"]["t"] + dt},
        }
    return state


def _loss(state):
    terms = []
    for name in TARGETS:
        owned = state["mechanisms"][name]
        carry = state["mechanism_buffers"][name]
        terms.extend(
            (
                0.001 * owned["v_izh"].square().sum(),
                0.01 * owned["u"].square().sum(),
                0.3 * carry["spikes"].sum(),
                0.7 * carry["syn_spikes"].sum(),
            )
        )
    for name in ("TH_CTX", "FS_RS", "RS_FS"):
        terms.extend(
            value.square().sum() for value in state["mechanisms"][name].values()
        )
    return sum(terms)


def _initialize_integrator(model):
    dt = model.v.new_tensor(DT)
    model.integrator._initialize(model, dt, force=True, compile_scope="population")
    return dt


def _imperative_step(model, dt):
    model.integrator.step(model, dt)
    model.t = model.t + dt


def _assert_legacy_state(model, expected):
    # A closure-based population deliberately has a different binding schema
    # and cannot be extracted using the bound population's functional plan.
    _assert_close(model.v, expected["integrator"]["v"])
    _assert_close(model.t, expected["clock"]["t"])
    for family in ("mechanisms", "mechanism_buffers"):
        for name, values in expected[family].items():
            mechanism = getattr(model.mech, name)
            for field, value in values.items():
                _assert_close(getattr(mechanism, field), value)


@pytest.mark.parametrize("batch", (None, 2))
def test_bound_cortical_trace_matches_original_currents_and_legacy_closures(batch):
    model = _model(batch=batch)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    prepared = functional.prepare(tensors.parameters, tensors.constants)
    snapshot = {name: value.detach().clone() for name, value in model.named_buffers()}
    imperative = _model(batch=batch)
    legacy = _model(batch=batch, legacy=True)
    dt = _initialize_integrator(imperative)
    _initialize_integrator(legacy)
    state = tensors.state
    for index in range(STEPS):
        state, _ = functional.step(tensors.parameters, prepared, state)
        expected = _recurrence(
            tensors.state, tensors.parameters, tensors.constants["dt"], index + 1
        )
        _assert_close(state, expected)
        _imperative_step(imperative, dt)
        _imperative_step(legacy, dt)
        _assert_close(functional.extract(imperative).state, state)
        _assert_legacy_state(legacy, state)
    # Source and consumer live tensors remain untouched by functional execution.
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, snapshot[name], rtol=0, atol=0, msg=name)
    first, _ = functional.rollout(tensors.parameters, prepared, tensors.state, steps=2)
    continued, _ = functional.rollout(
        tensors.parameters, prepared, first, steps=STEPS - 2
    )
    _assert_close(continued, state)
    functional.commit_state_(model, state)
    _assert_close(functional.extract(model).state, state)
    next_state, _ = functional.step(tensors.parameters, prepared, state)
    _imperative_step(model, _initialize_integrator(model))
    _assert_close(functional.extract(model).state, next_state)


def test_bindings_follow_explicit_source_state_and_reversal_parameters():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    state = _clone(tensors.state)
    parameters = _clone(tensors.parameters)
    for name in ("TH_CTX", "FS_RS", "RS_FS"):
        state["mechanisms"][name]["s"] = state["mechanisms"][name]["s"] + 0.04
        reversal = _parameter(parameters, name, "e")
        reversal.add_(3.0)
        # Poison every live source after extraction. Pure execution must use
        # the supplied values, including the changed reversal parameters.
        provider = getattr(model.mech, name)
        provider.s = torch.full_like(provider.s, 1000.0)
    actual, _ = functional.prepare_and_step(parameters, tensors.constants, state)
    expected = _recurrence(state, parameters, tensors.constants["dt"], 1)
    _assert_close(actual, expected)
    initialized = functional.initialize(
        parameters, tensors.constants, tensors.initialization
    )
    fresh, _ = functional.prepare_and_step(
        initialized.parameters, initialized.constants, initialized.state
    )
    _assert_close(
        fresh,
        _recurrence(
            initialized.state, initialized.parameters, initialized.constants["dt"], 1
        ),
    )


def test_all_three_bindings_have_original_state_and_parameter_gradients():
    model = _model(batch=2)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    parameters, state = _clone(tensors.parameters), _clone(tensors.state)
    targets = []
    for name in ("TH_CTX", "FS_RS", "RS_FS"):
        for field in ("s", "z"):
            targets.append(state["mechanisms"][name][field].requires_grad_())
        targets.append(_parameter(parameters, name, "e").requires_grad_())
        targets.append(
            _parameter(parameters, name, "tau", "alpha_state").requires_grad_()
        )
    for name, (state_name, _) in TARGETS.items():
        targets.append(state["mechanisms"][name]["v_izh"].requires_grad_())
        targets.append(
            _parameter(parameters, name, "i_stim", state_name).requires_grad_()
        )
    actual, _ = functional.prepare_and_rollout(
        parameters, tensors.constants, state, steps=3
    )
    expected = _recurrence(state, parameters, tensors.constants["dt"], 3)
    _assert_close(actual, expected)
    actual_gradients = torch.autograd.grad(_loss(actual), targets, retain_graph=True)
    expected_gradients = torch.autograd.grad(_loss(expected), targets)
    for actual_gradient, expected_gradient in zip(
        actual_gradients, expected_gradients, strict=True
    ):
        torch.testing.assert_close(
            actual_gradient, expected_gradient, rtol=2e-8, atol=2e-10
        )
        assert torch.isfinite(actual_gradient).all()
        assert torch.count_nonzero(actual_gradient)


def test_binding_transforms_and_compiled_scan_match_independent_recurrence():
    model = _model()
    functional, tensors = dn.func.make_functional(model, dt=DT)
    reversal_name = _parameter_key(tensors.parameters, "FS_RS", "e")

    def run(reversal):
        parameters = {**tensors.parameters, reversal_name: reversal}
        return functional.prepare_and_rollout(
            parameters, tensors.constants, tensors.state, steps=2
        )[0]

    reversal = tensors.parameters[reversal_name]
    lanes = torch.stack((reversal - 1.0, reversal + 2.0))
    actual = torch.vmap(run)(lanes)
    expected = pytree.tree_map(
        lambda *values: torch.stack(values), *(run(lane) for lane in lanes)
    )
    _assert_close(actual, expected)

    def voltage(value):
        return run(value)["mechanisms"]["ctx_rs"]["v_izh"]

    reverse = torch.func.jacrev(voltage)(reversal)
    forward = torch.func.jacfwd(voltage)(reversal)
    torch.testing.assert_close(reverse, forward, rtol=2e-9, atol=2e-10)
    assert torch.count_nonzero(reverse) == 3

    prepared = functional.prepare(tensors.parameters, tensors.constants)
    chunk = functional.compile_rollout_chunk(2, backend="aot_eager")
    actual, _ = chunk(tensors.parameters, prepared, tensors.state)
    expected = _recurrence(
        tensors.state, tensors.parameters, tensors.constants["dt"], 2
    )
    _assert_close(actual, expected)
    actual_gradient = torch.autograd.grad(_loss(actual), reversal, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(_loss(expected), reversal)[0]
    torch.testing.assert_close(
        actual_gradient, expected_gradient, rtol=2e-9, atol=2e-10
    )


def test_binding_checkpointed_inductor_scan_matches_original_values_and_gradients():
    model = _model()
    _, tensors = dn.func.make_functional(model, dt=DT)
    runner = dn.compile(
        model,
        dt=DT,
        steps=STEPS,
        initialization="snapshot",
        backend="inductor",
        execution="scan",
        inference=False,
        checkpoint_every=2,
        trainable=(_parameter_key(tensors.parameters, "FS_RS", "e"),),
        fullgraph=True,
        options={"cpp_wrapper": True, "compile_threads": 1},
    )
    actual, _ = runner()
    expected = _recurrence(
        tensors.state, runner.parameters, runner.constants["dt"], STEPS
    )
    _assert_close(actual, expected)
    reversal = _parameter(runner.parameters, "FS_RS", "e")
    actual_gradient = torch.autograd.grad(_loss(actual), reversal, retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(_loss(expected), reversal)[0]
    torch.testing.assert_close(
        actual_gradient, expected_gradient, rtol=2e-8, atol=2e-10
    )


def test_bindings_survive_population_and_network_batch_rebuilds():
    model = _model()
    model.build(force_rebuild=True)
    model.initialize()
    _seed(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    actual, _ = functional.prepare_and_step(
        tensors.parameters, tensors.constants, tensors.state
    )
    _assert_close(
        actual,
        _recurrence(tensors.state, tensors.parameters, tensors.constants["dt"], 1),
    )
    with dn.ctx(JIT=0, REQUIRE_GRAD=1, DTYPE=torch.float64, IMEM=False):
        network = dn.Network({"ctx": model})
        network.batch(2)
        network.train()
        network.initialize(DT)
    model = network.ctx
    _seed(model)
    functional, tensors = dn.func.make_functional(model, dt=DT)
    expected = _recurrence(
        tensors.state, tensors.parameters, tensors.constants["dt"], 1
    )
    network.run(tstop=DT, progressbar=False)
    _assert_close(functional.extract(model).state, expected)
