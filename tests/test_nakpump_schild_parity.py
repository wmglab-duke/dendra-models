import math

import dendra as dn
import pytest
import torch

from dendra_models.models.cells.peripheral.mech.nakpumpSchild import (
    nakpumpSchild,
    nakpumpSchild_augmented,
)


# Reference produced with NEURON 9.0.2 nrnivmodl and the untouched PyFibers 0.8.5
# wheel member pyfibers/MOD/NakpumpSchild.mod, SHA-256
# 96b7831d33b0b44b19be3440d245fbf9537316566f88ed7d3e78174d0f749171.
# Insert the pump into two sections, use ion_style(name, 1, 1, 0, 0, 0), and
# set concentrations and RANGE values from _model (gbar_INaKmax22 is INaKmax22
# in NEURON). Initialize at 34 °C, then set segment voltages and call fcurrent.
# Slopes are centered differences at v ± 0.001 mV. At 37 °C, fcurrent gave the
# same currents until finitialize ran again. The augmented variant applies its
# Dendra multipliers independently to the base reference currents and slopes.
INITIAL_V = (-70.0, -80.0)
INITIAL_I = (
    (0.003950968583991407, 0.00411061118195312),
    (-0.0026339790559942716, -0.0027404074546354133),
)
CROSSING_V = (-150.0, -110.0)
CROSSING_I = ((0.0, 0.0031318942338690436), (0.0, -0.002087929489246029))
CROSSING_SLOPE = (
    (0.00012840647903169646, 4.34985310316955e-05),
    (-8.56043193544643e-05, -2.8999020687797e-05),
)
PROBE_V = (-163.0, -85.0)
PROBE_I = (
    (-0.0022557894955896893, 0.003982952449594327),
    (0.0015038596637264594, -0.0026553016330628848),
)
PROBE_SLOPE = (
    (0.0002344895527286485, 2.6641822407913462e-05),
    (-0.00015632636848569338, -1.7761214938681255e-05),
)
REINITIALIZED_I = (
    (-0.002157551511097519, 0.003809497780290615),
    (0.0014383676740650129, -0.00253966518686041),
)
ION_CURRENTS = (("na", "ina"), ("k", "ik"))
VARIANTS = (
    pytest.param(nakpumpSchild, (1.0, 1.0), id="base"),
    pytest.param(nakpumpSchild_augmented, (1.5, 0.25), id="augmented"),
)


def _row(values):
    return torch.tensor([values], dtype=torch.float64)


def _model(mechanism, factors, voltage):
    model = dn.SingleCompartment(N=1, C=2, celsius=34.0, integrator=dn.scnv(imem=True))
    for ion, _ in ION_CURRENTS:
        model.ion_style(ion, 0, 0)
    model.concentrations(
        nai0=_row((11.4, 14.0)),
        nao0=_row((140.0, 145.0)),
        ki0=_row((120.0, 130.0)),
        ko0=_row((5.6, 7.6)),
    )
    augmentation = (
        dict(zip(("ina_aug", "ik_aug"), factors, strict=True))
        if mechanism is nakpumpSchild_augmented
        else {}
    )
    model.insert(
        mechanism,
        Kmnai=_row((4.25, 7.75)),
        Kmko=_row((0.4, 0.9)),
        gbar_INaKmax22=_row((0.0075, 0.013)),
        **augmentation,
    )
    model.set_v_init(voltage)
    model.train()
    model.initialize()
    return model


def _assert_current(actual, expected, factor):
    torch.testing.assert_close(actual, _row(expected) * factor, rtol=1e-12, atol=1e-17)


@pytest.mark.parametrize(("mechanism", "factors"), VARIANTS)
def test_schild_pump_independent_hooks_and_assembled_slopes_match_neuron(
    mechanism, factors
):
    with dn.ctx(DEVICE="cpu", DTYPE="float64", JIT=0, REQUIRE_GRAD=1):
        model = _model(mechanism, factors, _row(INITIAL_V))
        pump = getattr(model.mech, mechanism.__name__)
        for (ion, name), current, factor in zip(
            ION_CURRENTS, INITIAL_I, factors, strict=True
        ):
            _assert_current(getattr(model.mech.ions[ion], name), current, factor)

        for points, currents, slopes in (
            (CROSSING_V, CROSSING_I, CROSSING_SLOPE),
            (PROBE_V, PROBE_I, PROBE_SLOPE),
        ):
            voltage = _row(points).requires_grad_()
            pairs = []
            for (_, name), current, slope, factor in zip(
                ION_CURRENTS, currents, slopes, factors, strict=True
            ):
                plain = getattr(pump, name)(voltage)
                pair_current, pair_slope = getattr(pump, f"{name}_with_conductance")(
                    voltage
                )
                _assert_current(plain, current, factor)
                _assert_current(pair_current, current, factor)
                (analytic,) = torch.autograd.grad(plain.sum(), voltage)
                torch.testing.assert_close(pair_slope, analytic, rtol=1e-12, atol=0)
                torch.testing.assert_close(
                    pair_slope, _row(slope) * factor, rtol=2e-9, atol=1e-17
                )
                pairs.append((pair_current, pair_slope))

            torch.testing.assert_close(
                model.mech.itot(voltage), sum(current for current, _ in pairs)
            )
            assembled = model.mech.i(voltage)
            for actual, components in zip(
                assembled, zip(*pairs, strict=True), strict=True
            ):
                torch.testing.assert_close(actual, sum(components), rtol=0, atol=0)


@pytest.mark.parametrize(("mechanism", "factors"), VARIANTS)
def test_schild_pump_temperature_changes_at_reinitialization(mechanism, factors):
    with dn.ctx(DEVICE="cpu", DTYPE="float64", JIT=0, REQUIRE_GRAD=1):
        voltage = _row(PROBE_V)
        model = _model(mechanism, factors, voltage)
        pump = getattr(model.mech, mechanism.__name__)

        # Change the live inputs while retaining the initialized scale.
        pump.celsius = torch.full_like(pump.celsius, 37.0)
        pump.gbar_INaKmax22 = 2 * pump.gbar_INaKmax22
        for (_, name), current, factor in zip(
            ION_CURRENTS, PROBE_I, factors, strict=True
        ):
            _assert_current(getattr(pump, name)(voltage), current, factor)

        model.parameter_set_(celsius_param=37.0)
        model.initialize()
        for (ion, name), current, factor in zip(
            ION_CURRENTS, REINITIALIZED_I, factors, strict=True
        ):
            _assert_current(getattr(model.mech.ions[ion], name), current, factor)


@pytest.mark.parametrize(("mechanism", "factors"), VARIANTS)
def test_schild_pump_functional_reinitialization_matches_neuron(mechanism, factors):
    with dn.ctx(DEVICE="cpu", DTYPE="float64", JIT=0, REQUIRE_GRAD=1):
        model = _model(mechanism, factors, _row(PROBE_V))
        runner = dn.compile(
            model,
            dt=0.025,
            steps=2,
            backend="eager",
            inference=False,
            trainable=("celsius_param",),
        )
        parameter = runner.parameter_name("celsius_param")
        for temperature, currents in ((34.0, PROBE_I), (37.0, REINITIALIZED_I)):
            celsius = torch.tensor(temperature, dtype=torch.float64, requires_grad=True)
            initial = dn.func.initialize(
                runner, parameters={**runner.parameters, parameter: celsius}
            )
            state, _ = dn.func.run(runner, initial)
            for (ion, name), current, factor in zip(
                ION_CURRENTS, currents, factors, strict=True
            ):
                actual = state["ions"][ion][name]
                _assert_current(actual, current, factor)
                (gradient,) = torch.autograd.grad(
                    actual.sum(), celsius, retain_graph=True
                )
                expected = -_row(current).sum() * factor * math.log(1.16) / 10
                torch.testing.assert_close(gradient, expected, rtol=1e-12, atol=0)
