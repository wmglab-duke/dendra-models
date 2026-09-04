"""Pure initialization transforms for C-fibre resting-current balance."""

from __future__ import annotations

import torch


_PARAMETER_PREFIX = "parameters.integrator.mech.mechanisms"


def _parameter(mechanism: str, name: str) -> str:
    return f"{_PARAMETER_PREFIX}.{mechanism}.{name}_param"


_GNA = _parameter("leak", "gnaleak")
_GK = _parameter("leak", "gkleak")
_GCA = _parameter("leak", "gcaleak")
_PUMP_NA = _parameter("extrapump", "pumpina")
_PUMP_K = _parameter("extrapump", "pumpik")
_PUMP_CA = _parameter("extrapump", "pumpica")


def _balanced_leak_and_pump(voltage, current, reversal):
    """Return scalar leak/pump values that cancel one accepted ionic current."""

    # Retain singleton visible axes through the arithmetic, then reduce them.
    # This preserves the historical first-element balance rule without
    # evaluating ignored spatial points, and lets transformed values compose
    # with an invariant reversal potential under a zero-lane vmap. Selecting a
    # visible scalar first exposes a PyTorch mixed-BatchedTensor corner case.
    voltage = voltage[(slice(0, 1),) * voltage.ndim]
    current = current[(slice(0, 1),) * current.ndim]
    reversal = reversal[(slice(0, 1),) * reversal.ndim]
    conductance = -current / (voltage - reversal)
    use_pump = conductance < 0.0
    zero = torch.zeros_like(conductance)
    leak = torch.where(use_pump, zero, conductance)
    pump = torch.where(use_pump, -current, zero)
    return leak.sum(), pump.sum()


class _ResetTigerholmBalance(torch.nn.Module):
    def forward(self, gna, gk, pump_na, pump_k, pump_ca):
        return (
            torch.zeros_like(gna),
            torch.zeros_like(gk),
            torch.zeros_like(pump_na),
            torch.zeros_like(pump_k),
            torch.zeros_like(pump_ca),
        )


class _BalanceTigerholm(torch.nn.Module):
    def forward(self, voltage, ina, ena, ik, ek):
        gna, pump_na = _balanced_leak_and_pump(voltage, ina, ena)
        gk, pump_k = _balanced_leak_and_pump(voltage, ik, ek)
        return gna, pump_na, gk, pump_k


class _ResetThioBalance(torch.nn.Module):
    def forward(self, gna, gk, gca, pump_na, pump_k, pump_ca):
        return (
            torch.zeros_like(gna),
            torch.zeros_like(gk),
            torch.zeros_like(gca),
            torch.zeros_like(pump_na),
            torch.zeros_like(pump_k),
            torch.zeros_like(pump_ca),
        )


class _BalanceThio(torch.nn.Module):
    def forward(self, voltage, ina, ena, ik, ek, ica, eca):
        gna, pump_na = _balanced_leak_and_pump(voltage, ina, ena)
        gk, pump_k = _balanced_leak_and_pump(voltage, ik, ek)
        gca, pump_ca = _balanced_leak_and_pump(voltage, ica, eca)
        return gna, pump_na, gk, pump_k, gca, pump_ca


def register_tigerholm_balance(model) -> None:
    """Register Tigerholm's historical Na/K resting-balance transaction."""

    reset = (_GNA, _GK, _PUMP_NA, _PUMP_K, _PUMP_CA)
    model.register_pre_initialize_transform(
        "reset_resting_balance",
        _ResetTigerholmBalance(),
        reads=reset,
        writes=reset,
    )
    model.register_post_initialize_transform(
        "balance_resting_currents",
        _BalanceTigerholm(),
        reads=(
            "state.integrator.v",
            "state.ions.na.ina",
            "state.ions.na.ena",
            "state.ions.k.ik",
            "state.ions.k.ek",
        ),
        writes=(_GNA, _PUMP_NA, _GK, _PUMP_K),
    )


def register_thio_balance(model) -> None:
    """Register Thio's Na/K/Ca resting-balance transaction."""

    reset = (_GNA, _GK, _GCA, _PUMP_NA, _PUMP_K, _PUMP_CA)
    model.register_pre_initialize_transform(
        "reset_resting_balance",
        _ResetThioBalance(),
        reads=reset,
        writes=reset,
    )
    model.register_post_initialize_transform(
        "balance_resting_currents",
        _BalanceThio(),
        reads=(
            "state.integrator.v",
            "state.ions.na.ina",
            "state.ions.na.ena",
            "state.ions.k.ik",
            "state.ions.k.ek",
            "state.ions.ca.ica",
            "state.ions.ca.eca",
        ),
        writes=(_GNA, _PUMP_NA, _GK, _PUMP_K, _GCA, _PUMP_CA),
    )
