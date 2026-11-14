import dendra as dn
import pytest
import torch
from torch.nn.functional import softplus

from dendra_models.models.cells.peripheral.mech.nakpump import nakpump
from dendra_models.models.cells.peripheral.mech.nakpumpSchild import (
    nakpumpSchild,
    nakpumpSchild_augmented,
)


@pytest.mark.parametrize(
    ("mechanism", "smalla", "rhos"),
    [
        (nakpump, -0.002, {"b1": 0.8}),
        (nakpumpSchild, None, {"Kmnai": 6.0, "Kmko": -0.5, "gbar_INaKmax22": -4.0}),
        (
            nakpumpSchild_augmented,
            None,
            {"Kmnai": 6.0, "Kmko": -0.5, "gbar_INaKmax22": -4.0},
        ),
    ],
)
def test_strict_load_migrates_legacy_scalar_pump_parameters(mechanism, smalla, rhos):
    with dn.ctx(DEVICE="cpu", DTYPE="float64", JIT=0):
        model = dn.SingleCompartment(N=1, C=2)
        rho_values = {
            name: torch.tensor(rho, dtype=torch.float64) for name, rho in rhos.items()
        }
        raw_values = {name: softplus(rho) for name, rho in rho_values.items()}
        if smalla is not None:
            raw_values["smalla"] = torch.tensor(smalla, dtype=torch.float64)
        # A new model may have a vector parameter for the legacy scalar.
        model.insert(
            mechanism,
            **{name: value.repeat(1, 2) * 0.8 for name, value in raw_values.items()},
        )
        model.initialize()
        pump = getattr(model.mech, mechanism.__name__)

        legacy = {key: value.clone() for key, value in model.state_dict().items()}
        effective_values = {name: value * 0.9 for name, value in raw_values.items()}
        buffer_keys = {key for key in legacy if f".{mechanism.__name__}." in key}
        for key in buffer_keys:
            name = key.rsplit(".", 1)[-1]
            if name in effective_values:
                legacy[key] = effective_values[name].clone()
            elif name.endswith("_param"):
                name = name.removesuffix("_param")
                if name in rhos:
                    del legacy[key]
                    legacy[key + ".rho"] = rho_values[name].clone()
                elif name == "smalla":
                    legacy[key] = raw_values[name].clone()
            elif name == "INaKmax":
                legacy[key].fill_(0.003)

        original_keys = tuple(legacy)
        model.load_state_dict(legacy, strict=True)
        assert tuple(legacy) == original_keys
        for name, raw in raw_values.items():
            torch.testing.assert_close(getattr(pump, f"{name}_param"), raw.expand(1, 2))
            torch.testing.assert_close(
                getattr(pump, name), effective_values[name].expand(1, 2)
            )
        if mechanism is not nakpump:
            torch.testing.assert_close(
                pump.INaKmax, torch.full_like(pump.INaKmax, 0.003)
            )
