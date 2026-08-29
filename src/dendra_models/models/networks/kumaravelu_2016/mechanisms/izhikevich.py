"""
Izhikevich cortical mechanisms for the Kumaravelu et al. CTX-BG-TH model.

These are Dendra VoltageProcess mechanisms intended for use in a Population
constructed with dn.scnv(). The physical membrane voltage is stored in the
mechanism state ``v_izh`` and returned by update_v(), matching the Esser-style
pattern in the supplied yu_2024 implementation.
"""

import torch

from dendra.models.mechanisms._mechanism import Mechanism as M, VoltageProcess as V, Synapse as Syn
from dendra.models.mechanisms._state import State as S


def _maybe_current(obj, name, like):
    value = getattr(obj, name, None)
    if value is None:
        return torch.zeros_like(like)
    return value


class regular_spiking_cortex_states(S):
    S.STATE("v_izh", "u")
    S.ASSIGNED("i_syn", "rhs_v")
    S.DERIVATIVE("v_izh' = rhs_v", "u' = a * (b * v_izh - u)")
    S.RANGE(a=0.02, b=0.2, v0=-65.0, i_stim=0.0)

    def assigned_values(self, v, values):
        v_izh = values["v_izh"]
        # MATLAB RS equation: v' = ... - Iie - Ithcor + Iappco.
        i_syn = _maybe_current(self, "i_ie", v_izh) + _maybe_current(self, "i_thcor", v_izh)
        rhs_v = 0.04 * v_izh**2 + 5.0 * v_izh + 140.0 - values["u"] - i_syn + self.i_stim
        return {"i_syn": i_syn, "rhs_v": rhs_v}

    def state_defaults(self, v, values):
        v0 = torch.zeros_like(v) + self.v0
        return {"v_izh": v0, "u": self.b * v0}


class fast_spiking_interneuron_states(S):
    S.STATE("v_izh", "u")
    S.ASSIGNED("i_syn", "rhs_v")
    S.DERIVATIVE("v_izh' = rhs_v", "u' = a * (b * v_izh - u)")
    S.RANGE(a=0.1, b=0.2, v0=-65.0, i_stim=0.0)

    def assigned_values(self, v, values):
        v_izh = values["v_izh"]
        # MATLAB FS equation: v' = ... - Iei + Iappco.
        i_syn = _maybe_current(self, "i_ei", v_izh)
        rhs_v = 0.04 * v_izh**2 + 5.0 * v_izh + 140.0 - values["u"] - i_syn + self.i_stim
        return {"i_syn": i_syn, "rhs_v": rhs_v}

    def state_defaults(self, v, values):
        v0 = torch.zeros_like(v) + self.v0
        return {"v_izh": v0, "u": self.b * v0}


def _reshape_like(x, template):
    if x is None:
        return torch.zeros_like(template)
    if tuple(x.shape) == tuple(template.shape):
        return x
    if x.numel() == template.numel():
        return x.reshape_as(template)
    return x


class cortical_spike_router(M):
    """Expose cortical spike flags in full ctx-population coordinates.

    The RS and FS VoltageProcess mechanisms are inserted on disjoint slices of
    the same cortical population, so their assigned variables are mechanism-local
    tensors.  Dendra Network pre indices are population-flat; this router creates
    full-population tensors so ``pre_var`` indexing remains in the same coordinate
    frame as ``pre_idx``.

    Bind the local references after build with::

        router.setreference("rs_spikes_local", lambda: mc.ctx_rs.spikes)
        router.setreference("rs_syn_spikes_local", lambda: mc.ctx_rs.syn_spikes)
        router.setreference("fs_spikes_local", lambda: mc.ctx_fs.spikes)
        router.setreference("fs_syn_spikes_local", lambda: mc.ctx_fs.syn_spikes)
    """

    M.ASSIGNED("rs_spikes", "rs_syn_spikes", "fs_spikes", "fs_syn_spikes")

    def assigned_values(self, v, values):
        del values
        n_rs = v.shape[-1] // 2
        left = torch.zeros_like(v[..., :n_rs])
        right = torch.zeros_like(v[..., n_rs:])

        rs_spikes = _reshape_like(getattr(self, "rs_spikes_local", None), left)
        rs_syn_spikes = _reshape_like(getattr(self, "rs_syn_spikes_local", None), left)
        fs_spikes = _reshape_like(getattr(self, "fs_spikes_local", None), right)
        fs_syn_spikes = _reshape_like(getattr(self, "fs_syn_spikes_local", None), right)

        return {
            "rs_spikes": torch.cat((rs_spikes, right), dim=-1),
            "rs_syn_spikes": torch.cat((rs_syn_spikes, right), dim=-1),
            "fs_spikes": torch.cat((left, fs_spikes), dim=-1),
            "fs_syn_spikes": torch.cat((left, fs_syn_spikes), dim=-1),
        }


class _IzhikevichResetMixin:
    """Hard reset plus differentiable spike/crossing indicators."""

    def initial_values(self, v, values):
        del values
        return {
            "spikes": torch.zeros_like(v),
            "syn_spikes": torch.zeros_like(v),
            "v_prev": torch.zeros_like(v) + self.DE[self._state_name].v0,
        }

    def update_v(self, v):
        v_now = self.v_izh

        # The MATLAB code uses two event notions for cortex:
        #   1. v >= 30 mV for the reset and corticofugal spike list.
        #   2. crossing -10 mV for local E/I alpha synapses S1a/S1b.
        reset_bool = v_now >= self.v_peak
        cross_bool = (self.v_prev < self.syn_threshold) & (v_now >= self.syn_threshold)

        hard_reset = reset_bool.to(v_now.dtype)
        soft_reset = torch.sigmoid((v_now - self.v_peak) / self.tau_gate.clamp_min(1.0e-6))
        self.spikes = hard_reset + self.ste_scale * (soft_reset - soft_reset.detach())

        hard_cross = cross_bool.to(v_now.dtype)
        soft_cross_now = torch.sigmoid((v_now - self.syn_threshold) / self.tau_gate.clamp_min(1.0e-6))
        soft_cross_prev = torch.sigmoid((self.v_prev - self.syn_threshold) / self.tau_gate.clamp_min(1.0e-6))
        soft_cross = torch.relu(soft_cross_now - soft_cross_prev)
        self.syn_spikes = hard_cross + self.ste_scale * (soft_cross - soft_cross.detach())

        self.u = torch.where(reset_bool, self.u + self.d, self.u)
        self.v_izh = torch.where(reset_bool, torch.zeros_like(v_now) + self.c, v_now)
        self.v_prev = self.v_izh
        return self.v_izh


class regular_spiking_cortex(_IzhikevichResetMixin, V, Syn):
    V.STATE_BUNDLE(regular_spiking_cortex_states)
    V.RANGE(v_peak=30.0, c=-65.0, d=8.0, syn_threshold=-10.0)
    V.PARAMETER(tau_gate=0.5, ste_scale=1.0)
    V.CARRY("spikes", "syn_spikes", "v_prev")
    _state_name = "regular_spiking_cortex_states"


class fast_spiking_interneuron(_IzhikevichResetMixin, V, Syn):
    V.STATE_BUNDLE(fast_spiking_interneuron_states)
    V.RANGE(v_peak=30.0, c=-65.0, d=2.0, syn_threshold=-10.0)
    V.PARAMETER(tau_gate=0.5, ste_scale=1.0)
    V.CARRY("spikes", "syn_spikes", "v_prev")
    _state_name = "fast_spiking_interneuron_states"
