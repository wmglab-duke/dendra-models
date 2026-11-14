import dendra as dn
import torch

from dendra_models.models.cells.peripheral.mech.nakpump import nakpump


# NEURON 9.0.2 nrnivmodl, untouched PyFibers 0.8.5 pyfibers/MOD/Nakpump.mod
# (SHA-256 29d92940ddda2b0fd35c26740b3d4fcdae9fa18cffb52f32810b65cfd34aeb14).
# Insert into two sections with the concentrations and overrides in _model,
# use ion_style(name, 1, 1, 0, 0, 0), and sample ina and ik with fcurrent
# after finitialize(-65). For REFRESHED, change v to (0, -150) mV and
# set nai = (16, 20) and ko = (2, 8) mM before calling fcurrent again.
INITIAL = (
    (0.011342968091863175, 0.011795729783261547),
    (-0.007561978727908783, -0.007863819855507699),
)
REFRESHED = (
    (0.011697411013639218, 0.009658255578967915),
    (-0.007798274009092812, -0.006438837052645276),
)
CURRENTS = (("na", "ina"), ("k", "ik"))


def _row(values):
    return torch.tensor([values], dtype=torch.float64)


def _assert_current(actual, reference):
    torch.testing.assert_close(actual, _row(reference), rtol=1e-12, atol=1e-17)


def _model():
    model = dn.SingleCompartment(N=1, C=2, celsius=37.0, integrator=dn.scnv(imem=True))
    for ion, _ in CURRENTS:
        model.ion_style(ion, 0, 0)
    model.concentrations(nai0=_row((11.4, 18.0)), ko0=_row((5.6, 3.2)))
    model.insert(nakpump, smalla=_row((-0.0047891, -0.0034)), b1=_row((0.0, -0.5)))
    model.set_v_init(_row((-65.0, -65.0)))
    model.initialize()
    return model


def test_nakpump_matches_neuron_and_hooks_use_live_concentrations():
    with dn.ctx(DEVICE="cpu", DTYPE="float64", JIT=0):
        model = _model()
        pump = model.mech.nakpump
        voltage = _row((0.0, -150.0))
        for (ion, name), reference in zip(CURRENTS, INITIAL, strict=True):
            _assert_current(getattr(model.mech.ions[ion], name), reference)
            _assert_current(getattr(pump, name)(voltage), reference)

        model.mech.ions["na"].nai.copy_(_row((16.0, 20.0)))
        model.mech.ions["k"].ko.copy_(_row((2.0, 8.0)))
        model.mech.read_from_ions()
        for (_, name), reference in zip(CURRENTS, REFRESHED, strict=True):
            _assert_current(getattr(pump, name)(voltage), reference)

        expected_total = _row(REFRESHED[0]) + _row(REFRESHED[1])
        torch.testing.assert_close(model.mech.itot(voltage), expected_total)
        total, slope = model.mech.i(voltage)
        torch.testing.assert_close(total, expected_total)
        torch.testing.assert_close(slope, torch.zeros_like(voltage), rtol=0, atol=0)
        model.run(steps=1, dt=0.025)
        for (ion, name), reference in zip(CURRENTS, REFRESHED, strict=True):
            _assert_current(getattr(model.mech.ions[ion], name), reference)


def test_nakpump_functional_currents_match_neuron():
    with dn.ctx(DEVICE="cpu", DTYPE="float64", JIT=0):
        runner = dn.compile(_model(), dt=0.025, steps=2, backend="eager")
        initial = dn.func.initialize(runner)
        state, _ = dn.func.run(runner, initial)
        for (ion, name), reference in zip(CURRENTS, INITIAL, strict=True):
            _assert_current(state["ions"][ion][name], reference)
