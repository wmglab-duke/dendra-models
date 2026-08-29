from __future__ import annotations

import importlib
import inspect
import pkgutil

import dendra  # noqa: F401 - configure Dendra before importing torch
import pytest
import torch
from dendra.models.mechanisms import Mechanism
from dendra.models.mechanisms.compilers.ast import linear_conductance_in_v

import dendra_models
from dendra_models.models.cells.peripheral.cfiber.thio import ThioAutonomic2024


SHAPE = (1, 3)
DTYPE = torch.float64
VOLTAGE = torch.tensor([[-80.0, 0.0, 10.0]], dtype=DTYPE)
NUMERICAL_VOLTAGE = torch.tensor([[-80.0, -40.0, 10.0]], dtype=DTYPE)
ION_DEFAULTS = {
    "ena": 55.0,
    "ek": -82.0,
    "ecl": -70.0,
    "nai": 12.0,
    "nao": 145.0,
    "ki": 140.0,
    "ko": 5.0,
    "cai": 1.0e-4,
    "cao": 2.0,
}


def _mechanism_classes():
    classes = {}
    for module_info in pkgutil.walk_packages(
        dendra_models.__path__, f"{dendra_models.__name__}."
    ):
        if {"examples", "tests"}.intersection(module_info.name.split(".")):
            continue
        module = importlib.import_module(module_info.name)
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if (
                cls.__module__ == module_info.name
                and issubclass(cls, Mechanism)
                and cls is not Mechanism
            ):
                classes[(module_info.name, cls.__qualname__)] = cls
    return tuple(classes[key] for key in sorted(classes))


def _current_names(mechanism_cls):
    names = []
    for currents in mechanism_cls._currents.values():
        names.extend(currents)
    for currents in mechanism_cls._write_ion.values():
        names.extend(currents)
    return tuple(dict.fromkeys(names))


MECHANISM_CLASSES = _mechanism_classes()
ANALYTIC_CURRENT_CASES = tuple(
    (mechanism_cls, current)
    for mechanism_cls in MECHANISM_CLASSES
    for current in _current_names(mechanism_cls)
    if current not in mechanism_cls._explicit
    and hasattr(mechanism_cls, f"{current}_with_conductance")
)
NUMERICAL_CURRENT_CASES = tuple(
    (mechanism_cls, current)
    for mechanism_cls in MECHANISM_CLASSES
    for current in _current_names(mechanism_cls)
    if current in mechanism_cls._numerical
)


def _case_id(case):
    mechanism_cls, current = case
    return f"{mechanism_cls.__module__}.{mechanism_cls.__qualname__}.{current}"


def _new_mechanism(mechanism_cls):
    mechanism = mechanism_cls(
        mechanism_cls.__qualname__,
        torch.tensor(37.0, dtype=DTYPE),
        torch.ones(SHAPE, dtype=DTYPE),
        SHAPE,
        SHAPE,
    ).to(dtype=DTYPE)

    for reads in mechanism_cls._read_ion.values():
        for name in reads:
            if not hasattr(mechanism, name):
                mechanism.register_buffer(
                    name,
                    torch.full(SHAPE, ION_DEFAULTS.get(name, 1.0), dtype=DTYPE),
                )

    with torch.no_grad():
        for index, name in enumerate(mechanism._all_states):
            state = mechanism._buffers.get(name)
            if state is not None and state.is_floating_point():
                state.fill_(0.15 + 0.07 * (index % 7))
        if hasattr(mechanism, "A"):
            mechanism.A.fill_(0.2)
        if hasattr(mechanism, "B"):
            mechanism.B.fill_(0.65)

    if mechanism_cls.__qualname__ == "striatal_recurrent_gaba":
        refs = mechanism.DE["striatal_recurrent_gaba_refs"]
        refs.s0 = torch.full(SHAPE, 0.2, dtype=DTYPE)
        refs.s3 = torch.full(SHAPE, 0.4, dtype=DTYPE)

    return mechanism


def _jacobian_matrix(mechanism, current, voltage):
    jacobian = torch.autograd.functional.jacobian(
        lambda at_v: getattr(mechanism, current)(at_v), voltage
    )
    return jacobian.reshape(voltage.numel(), voltage.numel())


def test_every_mechanism_class_constructs_with_a_safe_conductance_path():
    failures = []
    for mechanism_cls in MECHANISM_CLASSES:
        try:
            _new_mechanism(mechanism_cls)
        except Exception as exc:  # report the complete inventory in one failure
            failures.append(
                f"{mechanism_cls.__module__}.{mechanism_cls.__qualname__}: "
                f"{type(exc).__name__}: {exc}"
            )

    assert not failures, "\n" + "\n".join(failures)


def test_every_undeclared_current_is_symbolically_affine():
    failures = []
    for mechanism_cls in MECHANISM_CLASSES:
        for current in _current_names(mechanism_cls):
            if (
                current in mechanism_cls._explicit
                or current in mechanism_cls._numerical
                or hasattr(mechanism_cls, f"{current}_with_conductance")
            ):
                continue
            try:
                linear_conductance_in_v(mechanism_cls, method=current)
            except Exception as exc:  # report every current, not just the first
                failures.append(
                    f"{mechanism_cls.__module__}.{mechanism_cls.__qualname__}."
                    f"{current}: {type(exc).__name__}: {exc}"
                )

    assert not failures, "\n" + "\n".join(failures)


@pytest.mark.parametrize(
    ("mechanism_cls", "current"),
    ANALYTIC_CURRENT_CASES,
    ids=[_case_id(case) for case in ANALYTIC_CURRENT_CASES],
)
def test_analytic_conductance_matches_the_pointwise_jacobian(
    mechanism_cls, current
):
    mechanism = _new_mechanism(mechanism_cls)
    voltage = VOLTAGE.clone().requires_grad_()

    authored_current = getattr(mechanism, current)(voltage)
    paired_current, conductance = getattr(mechanism, f"{current}_with_g")(voltage)
    jacobian = _jacobian_matrix(mechanism, current, voltage)
    diagonal = jacobian.diagonal().reshape_as(voltage)
    conductance = torch.as_tensor(
        conductance, device=voltage.device, dtype=voltage.dtype
    ).expand_as(voltage)

    assert mechanism._current_conductance_mode[current] == "analytic"
    torch.testing.assert_close(
        paired_current, authored_current, rtol=2.0e-12, atol=2.0e-12
    )
    torch.testing.assert_close(
        conductance, diagonal, rtol=2.0e-10, atol=2.0e-12
    )
    off_diagonal = jacobian - torch.diag(jacobian.diagonal())
    torch.testing.assert_close(
        off_diagonal, torch.zeros_like(off_diagonal), rtol=0.0, atol=0.0
    )


def test_only_reviewed_currents_use_numerical_differentiation():
    actual = {
        (mechanism_cls.__module__, mechanism_cls.__qualname__, current)
        for mechanism_cls, current in NUMERICAL_CURRENT_CASES
    }
    assert actual == {
        ("dendra_models.models.cells.peripheral.mech.fh", "fh", "ina"),
        ("dendra_models.models.cells.peripheral.mech.fh", "fh", "ik"),
    }


@pytest.mark.parametrize(
    ("mechanism_cls", "current"),
    NUMERICAL_CURRENT_CASES,
    ids=[_case_id(case) for case in NUMERICAL_CURRENT_CASES],
)
def test_reviewed_numerical_conductance_is_deterministic_and_pointwise(
    mechanism_cls, current
):
    mechanism = _new_mechanism(mechanism_cls)
    voltage = NUMERICAL_VOLTAGE.clone().requires_grad_()

    authored_current = getattr(mechanism, current)(voltage)
    repeated_current = getattr(mechanism, current)(voltage)
    paired_current, conductance = getattr(mechanism, f"{current}_with_g")(voltage)
    jacobian = _jacobian_matrix(mechanism, current, voltage)
    diagonal = jacobian.diagonal().reshape_as(voltage)

    assert mechanism._current_conductance_mode[current] == "numerical-declared"
    torch.testing.assert_close(repeated_current, authored_current, rtol=0.0, atol=0.0)
    torch.testing.assert_close(paired_current, authored_current, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        conductance, diagonal, rtol=2.0e-8, atol=2.0e-10
    )
    off_diagonal = jacobian - torch.diag(jacobian.diagonal())
    torch.testing.assert_close(
        off_diagonal, torch.zeros_like(off_diagonal), rtol=0.0, atol=0.0
    )


def test_thio_autonomic_initializes_with_exact_hcn_conductances():
    model = ThioAutonomic2024(diameters=[1.0], L=10.0, dx=10)
    model.initialize()

    assert tuple(model.shape) == (1, 1)
    assert model.mech.hcn._current_conductance_mode == {
        "ik": "analytic",
        "ina": "analytic",
    }


def test_modular_kumaravelu_network_initializes_with_safe_current_paths():
    from dendra_models.models.networks.kumaravelu_2016.net import kumaravelu_2016

    network = kumaravelu_2016()
    network.initialize(0.01)

    assert network.hh.mech.thalamic._current_conductance_mode["ina"] == "analytic"
    assert network.hh.mech.stn._current_conductance_mode["ilca"] == "analytic"
    assert network.hh.mech.gpe._current_conductance_mode["ica"] == "analytic"
    assert network.hh.mech.str_d1._current_conductance_mode["istim"] == "explicit"
    assert (
        network.hh.mech.StrD2_recurrent_gaba._current_conductance_mode["i"]
        == "analytic"
    )


def test_esser_soft_spike_initializes_its_declared_previous_gate_carry():
    from dendra_models.models.networks.yu_2024.mechanisms.esser import esser_mech_s

    mechanism = _new_mechanism(esser_mech_s)
    mechanism._configure_timestep(0.1)
    mechanism.g_prev.fill_(1.0)
    mechanism._init_buffers_s(VOLTAGE)

    torch.testing.assert_close(mechanism.g_prev, torch.zeros_like(VOLTAGE))
    assert not hasattr(mechanism, "h_prev")
