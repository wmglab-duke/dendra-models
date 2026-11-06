"""Public model families initialize and step through the shared pure API."""

from copy import deepcopy
import inspect

import dendra as dn
import numpy as np
import pytest
import torch

from dendra_models.models.cells import cortical, peripheral
from dendra_models.models.networks import Kumaravelu2016
from dendra_models.models.networks.kumaravelu_2016 import default_params


DT = 0.001


def _unique_exports(module):
    result = {}
    for name in module.__all__:
        factory = getattr(module, name)
        result.setdefault(factory, name)
    return tuple(result.values())


PERIPHERAL_MODELS = _unique_exports(peripheral)
CORTICAL_MODELS = _unique_exports(cortical)
PUBLIC_MODELS = (
    *(f"peripheral.{name}" for name in PERIPHERAL_MODELS),
    *(f"cortical.{name}" for name in CORTICAL_MODELS),
    "networks.Kumaravelu2016",
)
TRAINING_MODELS = (
    "peripheral.Sundt2015",
    "peripheral.ThioCutaneousAugmented2024",
    "peripheral.SMF",
    "peripheral.exactMRG",
    "cortical.L4_LBC_cACint",
)


def _model(name):
    family, name = name.split(".")
    if family == "peripheral":
        factory = getattr(peripheral, name)
        signature = inspect.signature(factory)
        if "n_node" in signature.parameters:
            diameters = [2.0, 3.0] if name == "smolMRG" else [5.7, 10.0]
            return factory(diameters=diameters, n_node=3)
        return factory(diameters=[1.0, 1.5], L=30.0, dx=10.0)
    if family == "cortical":
        factory = getattr(cortical, name)
        return factory(ID=factory.valid_ids()[0], N=1)
    params = default_params()
    params["n"] = 2
    params["cells"]["n_by_type"] = {
        name: 2 for name in params["cells"]["n_by_type"]
    }
    return Kumaravelu2016(
        params,
        rng=np.random.default_rng(2718),
        dt=DT,
        delay_mode="shift",
    )


@pytest.mark.parametrize("name", PUBLIC_MODELS)
def test_public_model_fresh_snapshot_inputs_and_checkpoint_continuation(name):
    with dn.ctx(DTYPE="float64", DEVICE="cpu", JIT=0, REQUIRE_GRAD=1):
        model = _model(name)
        model.eval()
        model.initialize(DT)
        functional, tensors = dn.func.make_functional(model, dt=DT)
        fresh = dn.func.initialize(functional, tensors)
        torch.testing.assert_close(
            fresh.state, tensors.state, rtol=2e-10, atol=2e-11
        )
        native = deepcopy(model)
        bound = functional.bind(tensors)
        state = tensors.state
        spatial = torch.linspace(
            -0.01, 0.01, model.v.numel(), dtype=model.v.dtype
        ).reshape_as(model.v)
        # Nonconstant explicit inputs exercise geometry without registered clocks.
        voltage_tape = torch.stack((spatial, -0.5 * spatial))
        current_tape = torch.stack(
            (torch.full_like(model.v, 1e-7), torch.zeros_like(model.v))
        )
        for index in range(2):
            inputs = dn.func.Inputs(
                ve=voltage_tape[index], intra=current_tape[index]
            )
            state, _ = bound(state, inputs=inputs)
            native.step(DT, inputs=inputs)
            torch.testing.assert_close(
                state, functional.extract(native).state, rtol=2e-10, atol=2e-11
            )
            assert all(
                torch.isfinite(value).all()
                for value in torch.utils._pytree.tree_leaves(state)
            )
            if index == 0:
                packet = dn.func.export_continuation(
                    functional,
                    dn.func.Continuation(
                        state, tensors.parameters, tensors.constants
                    ),
                )
                restored = dn.func.restore_continuation(functional, packet)
                restored_bound = functional.bind(restored)
            else:
                resumed, _ = restored_bound(restored.state, inputs=inputs)
                torch.testing.assert_close(state, resumed, rtol=2e-10, atol=2e-11)


@pytest.mark.parametrize("name", TRAINING_MODELS)
def test_public_model_input_gradients_match_native_execution(name):
    with dn.ctx(DTYPE="float64", DEVICE="cpu", JIT=0, REQUIRE_GRAD=1):
        model = _model(name)
        model.train()
        model.initialize(DT)
        model.detach()
        native = deepcopy(model)
        functional, tensors = dn.func.make_functional(model, dt=DT)
        bound = functional.bind(tensors)
        current = torch.full_like(model.v, 1e-7, requires_grad=True)
        native_current = current.detach().clone().requires_grad_()
        state = tensors.state
        for _ in range(2):
            state, _ = bound(state, inputs=dn.func.Inputs(intra=current))
            native.step(DT, inputs=dn.func.Inputs(intra=native_current))
        voltage = state["integrator"]["v"]
        torch.testing.assert_close(voltage, native.v, rtol=2e-10, atol=2e-11)
        gradient = torch.autograd.grad(voltage.sum(), current)[0]
        native_gradient = torch.autograd.grad(native.v.sum(), native_current)[0]
        assert gradient.abs().sum() > 0
        torch.testing.assert_close(
            gradient, native_gradient, rtol=2e-9, atol=2e-10
        )
