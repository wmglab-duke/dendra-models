from __future__ import annotations

import dendra as dn
import pytest
import torch

from dendra_models.models.cells.peripheral.cfiber.thio import (
    ThioAutonomic2024,
    ThioCutaneous2024,
    ThioCutaneousAugmented2024,
    ThioCutaneousReduced,
)
from dendra_models.models.cells.peripheral.cfiber.tigerholm import Tigerholm2014

MODEL_CASES = (
    pytest.param(Tigerholm2014, ("na", "k"), id="tigerholm"),
    pytest.param(ThioAutonomic2024, ("na", "k", "ca"), id="thio-autonomic"),
    pytest.param(ThioCutaneous2024, ("na", "k", "ca"), id="thio-cutaneous"),
    pytest.param(
        ThioCutaneousAugmented2024,
        ("na", "k", "ca"),
        id="thio-cutaneous-augmented",
    ),
    pytest.param(
        ThioCutaneousReduced,
        ("na", "k", "ca"),
        id="thio-cutaneous-reduced",
    ),
)

VALIDATION_MODEL_CASES = (
    pytest.param(Tigerholm2014, id="tigerholm"),
    pytest.param(ThioAutonomic2024, id="thio-autonomic"),
    pytest.param(ThioCutaneous2024, id="thio-cutaneous"),
)

ION_FIELDS = {
    "na": ("ina", "ena", "gnaleak", "pumpina"),
    "k": ("ik", "ek", "gkleak", "pumpik"),
    "ca": ("ica", "eca", "gcaleak", "pumpica"),
}


def _first(value):
    return value[(0,) * value.ndim]


def _assert_raw_and_effective_parameters_match(model):
    for mechanism_name, parameter_names in (
        ("leak", ("gnaleak", "gkleak", "gcaleak")),
        ("extrapump", ("pumpina", "pumpik", "pumpica")),
    ):
        mechanism = getattr(model.mech, mechanism_name)
        for name in parameter_names:
            torch.testing.assert_close(
                getattr(mechanism, f"{name}_param"),
                getattr(mechanism, name),
                rtol=0.0,
                atol=0.0,
            )


def _assert_balanced_resting_current(model, ion_name):
    current_name, reversal_name, leak_name, pump_name = ION_FIELDS[ion_name]
    ion = getattr(model.mech, f"{ion_name}_ion")
    current = _first(getattr(ion, current_name))
    reversal = _first(getattr(ion, reversal_name))
    voltage = _first(model.v)
    leak = getattr(model.mech.leak, leak_name)
    pump = getattr(model.mech.extrapump, pump_name)

    required_conductance = -current / (voltage - reversal)
    use_pump = required_conductance < 0.0
    zero = torch.zeros_like(required_conductance)
    torch.testing.assert_close(
        leak,
        torch.where(use_pump, zero, required_conductance),
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        pump,
        torch.where(use_pump, -current, zero),
        rtol=0.0,
        atol=0.0,
    )

    residual = current + leak * (voltage - reversal) + pump
    scale = max(1.0, current.abs().item())
    torch.testing.assert_close(
        residual,
        torch.zeros_like(residual),
        rtol=0.0,
        atol=10.0 * torch.finfo(residual.dtype).eps * scale,
    )


@pytest.mark.parametrize(("constructor", "balanced_ions"), MODEL_CASES)
def test_cfiber_resting_balance_is_pure_exact_and_idempotent(
    constructor,
    balanced_ions,
):
    with dn.ctx(JIT=0, REQUIRE_GRAD=0, DTYPE="float64"):
        model = constructor(diameters=[1.0], L=10.0, dx=10)
        model.initialize()

    # The augmented and reduced Thio variants historically omitted the leak
    # mechanism even though their initialization balance required it.
    assert model.mech.leak is not None
    assert model.mech.extrapump is not None
    assert [
        (hook.phase, hook.action_name)
        for hook in (*model.pre_initialize_hooks, *model.post_initialize_hooks)
    ] == [
        ("pre", "reset_resting_balance"),
        ("post", "balance_resting_currents"),
    ]

    _assert_raw_and_effective_parameters_match(model)
    for ion_name in balanced_ions:
        _assert_balanced_resting_current(model, ion_name)

    expected = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    leak_names = [ION_FIELDS[name][2] for name in balanced_ions]
    with torch.no_grad():
        for name in leak_names:
            getattr(model.mech.leak, f"{name}_param").fill_(3.14159)
        for name in ("pumpina", "pumpik", "pumpica"):
            getattr(model.mech.extrapump, f"{name}_param").fill_(-2.71828)

    with dn.ctx(JIT=0, REQUIRE_GRAD=0, DTYPE="float64"):
        model.initialize()

    actual = model.state_dict()
    assert actual.keys() == expected.keys()
    for name, expected_value in expected.items():
        torch.testing.assert_close(
            actual[name],
            expected_value,
            rtol=0.0,
            atol=0.0,
            msg=lambda message: f"state_dict entry {name}: {message}",
        )
    _assert_raw_and_effective_parameters_match(model)


@pytest.mark.parametrize("constructor", VALIDATION_MODEL_CASES)
def test_validation_cfiber_steady_state_restore_is_exact(constructor):
    with dn.ctx(JIT=0, REQUIRE_GRAD=0, DTYPE="float64"):
        model = constructor(diameters=[1.0], L=10.0, dx=10)
        # At least one timestep is essential: it replaces the initialization
        # current snapshot with the accepted, balanced runtime current that
        # exposed structured post-transform replay during cache restoration.
        model.steady_state(dt=0.2, tstop=0.2)

    expected = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    model.v.fill_(20.0)

    with dn.ctx(JIT=0, REQUIRE_GRAD=0, DTYPE="float64"):
        model.initialize()

    assert model.initializing_from_state_cache
    actual = model.state_dict()
    assert actual.keys() == expected.keys()
    for name, expected_value in expected.items():
        torch.testing.assert_close(
            actual[name],
            expected_value,
            rtol=0.0,
            atol=0.0,
            msg=lambda message: f"state_dict entry {name}: {message}",
        )
    _assert_raw_and_effective_parameters_match(model)
