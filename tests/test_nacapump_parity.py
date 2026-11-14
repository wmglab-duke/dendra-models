import dendra as dn
import torch

from dendra_models.models.cells.peripheral.mech.nacapump import nacapump


# Reference produced with NEURON 9.0.2 nrnivmodl and the untouched PyFibers 0.8.5
# wheel member pyfibers/MOD/NaCaPump.mod, SHA-256
# febeff01fb245afc9a7ff5095e27dbbefd3dd960cf0d3f662f325a417eec2607.
# Insert the sole pump into two independent sections and match the inputs from
# _model, with ion_style(name, 1, 1, 0, 0, 0). Initialize at 34 °C, sample each
# voltage with fcurrent, then sample at 37 °C both before and after finitialize.
# Slopes are centered differences from fcurrent at v ± 0.001 mV.
INITIAL = (
    (0.0003496442099999634, 2.5401231179296643e-05),
    (-0.0002330961399999756, -1.6934154119531096e-05),
)
VOLTAGE_PROBE = (
    (8.241133284875732e-06, 0.00010790244092250293),
    (-5.4940888565838216e-06, -7.193496061500195e-05),
)
VOLTAGE_SLOPES = (
    (4.047139172405388e-06, 2.6543705229211474e-06),
    (-2.69809278160331e-06, -1.7695803486140983e-06),
)
LIVE_TEMPERATURE = (
    (1.0786324508390696e-05, 0.0001088548914486642),
    (-7.190883005593797e-06, -7.25699276324428e-05),
)
REINITIALIZED = (
    (8.514255483609032e-06, 8.59253173510107e-05),
    (-5.6761703224060215e-06, -5.7283544900673804e-05),
)
ION_CURRENTS = (("na", "ina"), ("ca", "ica"))


def _row(values):
    return torch.tensor([values], dtype=torch.float64)


def _model(initial_voltage):
    model = dn.SingleCompartment(N=1, C=2, celsius=34.0, integrator=dn.scnv(imem=True))
    for ion in ("na", "ca"):
        model.ion_style(ion, 0, 0)
    model.concentrations(
        nai0=_row((12.0, 14.0)),
        nao0=_row((140.0, 145.0)),
        cai0=_row((1e-4, 7e-5)),
        cao0=_row((2.0, 2.4)),
    )
    model.insert(nacapump, DNaCa=_row((0.0036, 0.006)))
    model.set_v_init(initial_voltage)
    model.train()
    model.initialize()
    return model


def _assert_current(actual, expected):
    torch.testing.assert_close(actual, _row(expected), rtol=1e-12, atol=1e-17)


def test_nacapump_currents_and_slopes_match_neuron():
    with dn.ctx(DEVICE="cpu", DTYPE="float64", JIT=0, REQUIRE_GRAD=1):
        initial_voltage = _row((0.0, -76.0))
        model = _model(initial_voltage)
        pump = model.mech.nacapump
        voltage = _row((-65.0, -37.0)).requires_grad_()
        slopes = []
        for (ion, name), initial, expected, reference_slope in zip(
            ION_CURRENTS, INITIAL, VOLTAGE_PROBE, VOLTAGE_SLOPES, strict=True
        ):
            _assert_current(getattr(model.mech.ions[ion], name), initial)
            plain = getattr(pump, name)
            pair = getattr(pump, f"{name}_with_conductance")
            with_zero, at_zero_slope = pair(initial_voltage)
            _assert_current(with_zero, initial)
            assert torch.all(at_zero_slope != 0)

            # Probe a different voltage before the handler refreshes its frame.
            _assert_current(plain(voltage), expected)
            current, slope = pair(voltage)
            _assert_current(current, expected)
            (analytic,) = torch.autograd.grad(plain(voltage).sum(), voltage)
            torch.testing.assert_close(slope, analytic, rtol=1e-12, atol=1e-17)
            torch.testing.assert_close(
                slope, _row(reference_slope), rtol=1e-8, atol=1e-16
            )
            slopes.append(slope)

        _, assembled_slope = model.mech.i(voltage)
        torch.testing.assert_close(assembled_slope, sum(slopes), rtol=0, atol=0)

        for _ in range(2):
            model.run(steps=1, dt=0.025)
            for (ion, name), expected in zip(ION_CURRENTS, INITIAL, strict=True):
                _assert_current(getattr(model.mech.ions[ion], name), expected)


def test_nacapump_live_temperature_and_functional_reinitialization_match_neuron():
    with dn.ctx(DEVICE="cpu", DTYPE="float64", JIT=0, REQUIRE_GRAD=1):
        voltage = _row((-65.0, -37.0))
        model = _model(voltage)
        pump = model.mech.nacapump
        runner = dn.compile(
            model,
            dt=0.025,
            steps=2,
            backend="eager",
            inference=False,
            trainable=("celsius_param",),
        )

        # Direct evaluation at the new temperature keeps the old INITIAL scale.
        pump.celsius = torch.full_like(pump.celsius, 37.0)
        for (_, name), expected in zip(ION_CURRENTS, LIVE_TEMPERATURE, strict=True):
            _assert_current(getattr(pump, name)(voltage), expected)

        # Fresh initialization of the same runner refreshes the INITIAL scale.
        name = runner.parameter_name("celsius_param")
        for temperature, expected in ((34.0, VOLTAGE_PROBE), (37.0, REINITIALIZED)):
            celsius = torch.tensor(temperature, dtype=torch.float64, requires_grad=True)
            initial = dn.func.initialize(
                runner, parameters={**runner.parameters, name: celsius}
            )
            state, _ = dn.func.run(runner, initial)
            currents = []
            for (ion, current_name), reference in zip(
                ION_CURRENTS, expected, strict=True
            ):
                current = state["ions"][ion][current_name]
                _assert_current(current, reference)
                currents.append(current)
            (gradient,) = torch.autograd.grad(
                (currents[0] - currents[1]).sum(), celsius
            )
            assert torch.isfinite(gradient) and gradient != 0
