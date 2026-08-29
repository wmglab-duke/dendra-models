"""Fused fast-path mechanism for the Kumaravelu et al. CTX-BG-TH model.

This module implements the entire single-compartment network as one Dendra
``VoltageProcess``.  It deliberately bypasses ``NetCon`` and current-map
bookkeeping; all intrinsic currents, synaptic filters, fixed delays, routing
permutations, cortical Izhikevich reset logic, and spike surrogates are updated
inside a single PyTorch tensor program.

The equations follow the MATLAB reference implementation, but spike-triggered
lookup-table synapses are represented as equivalent alpha or bi-exponential
filter states with fixed delay queues.  The filter discretization is selectable
(``euler``, ``backward_euler``, or ``exact``), and the filter update layout is
selectable (``uncoalesced`` or ``coalesced``).  Fixed delays use Dendra's
batched mechanism-level delayed-state helper; ``delay_mode="auto"`` uses fast
circular buffers in eval/no-grad mode and graph-safe shifted queues in training
mode.  The local cortical alpha synapses use the same second-order alpha ODE
form used in the MATLAB ``S1a/Z1a`` and ``S1b/Z1b`` updates.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Mapping

import torch

from dendra.models.networks.spiking import (
    crossing_spike as _dendra_crossing_spike,
    level_spike as _dendra_level_spike,
)

from dendra.models.networks.spiking import (
    crossing_spikes as _dendra_crossing_spikes,
    level_spikes as _dendra_level_spikes,
)

from dendra.models.mechanisms._mechanism import VoltageProcess as V


# ---------------------------------------------------------------------------
# Small tensor helpers
# ---------------------------------------------------------------------------


def _exp(x):
    return torch.exp(x)


def _log(x):
    return torch.log(x)


def _positive(x, eps: float = 1.0e-12):
    return torch.clamp(x, min=eps)


def _x_over_1_minus_exp_neg_x_over_s(x, s: float):
    """Stable x/(1-exp(-x/s))."""
    y = x / s
    return torch.where(
        torch.abs(y) < 1.0e-6,
        s * (1.0 + y / 2.0 + y * y / 12.0),
        x / (1.0 - _exp(-y)),
    )


def _x_over_exp_x_over_s_minus_1(x, s: float):
    """Stable x/(exp(x/s)-1)."""
    y = x / s
    return torch.where(
        torch.abs(y) < 1.0e-6,
        s * (1.0 - y / 2.0 + y * y / 12.0),
        x / (_exp(y) - 1.0),
    )


def _gather_index(x, index):
    """Gather a fixed last-axis index vector/matrix without hot-path casting."""
    return torch.gather(x, dim=-1, index=index.expand_as(x))


def _sum_gather_index(x, index):
    """Sum several fixed ring-shift gathers.

    ``index`` has shape ``(k, n)`` and is precomputed during initialization.
    """
    flat = index.reshape(-1)
    y = x.index_select(-1, flat)
    return y.reshape(*x.shape[:-1], index.shape[0], index.shape[1]).sum(dim=-2)


def _gather_perm(x, perm):
    """Gather ``x`` along the neuron axis with initialized long permutations."""
    while perm.ndim < x.ndim:
        perm = perm.unsqueeze(0)
    return torch.gather(x, dim=-1, index=perm.expand_as(x))


# ---------------------------------------------------------------------------
# Thalamic relay kinetics
# ---------------------------------------------------------------------------


def th_minf(v):
    return 1.0 / (1.0 + _exp(-(v + 37.0) / 7.0))


def th_hinf(v):
    return 1.0 / (1.0 + _exp((v + 41.0) / 4.0))


def th_pinf(v):
    return 1.0 / (1.0 + _exp(-(v + 60.0) / 6.2))


def th_rinf(v):
    return 1.0 / (1.0 + _exp((v + 84.0) / 4.0))


def th_tauh(v):
    ah = 0.128 * _exp(-(v + 46.0) / 18.0)
    bh = 4.0 / (1.0 + _exp(-(v + 23.0) / 5.0))
    return 1.0 / (ah + bh)


def th_taur(v):
    return 0.15 * (28.0 + _exp(-(v + 25.0) / 10.5))


# ---------------------------------------------------------------------------
# STN kinetics
# ---------------------------------------------------------------------------


def stn_ainf(v):
    return 1.0 / (1.0 + _exp(-(v + 45.0) / 14.7))


def stn_binf(v):
    return 1.0 / (1.0 + _exp((v + 90.0) / 7.5))


def stn_cinf(v):
    return 1.0 / (1.0 + _exp(-(v + 30.6) / 5.0))


def stn_d1inf(v):
    return 1.0 / (1.0 + _exp((v + 60.0) / 7.5))


def stn_d2inf(v):
    return 1.0 / (1.0 + _exp((v - 0.1) / 0.02))


def stn_hinf(v):
    return 1.0 / (1.0 + _exp((v + 45.5) / 6.4))


def stn_minf(v):
    return 1.0 / (1.0 + _exp(-(v + 40.0) / 8.0))


def stn_ninf(v):
    return 1.0 / (1.0 + _exp(-(v + 41.0) / 14.0))


def stn_pinf(v):
    return 1.0 / (1.0 + _exp(-(v + 56.0) / 6.7))


def stn_qinf(v):
    return 1.0 / (1.0 + _exp((v + 85.0) / 5.8))


def stn_rinf(v):
    return 1.0 / (1.0 + _exp(-(v - 0.17) / 0.08))


def stn_taua(v):
    return 1.0 + 1.0 / (1.0 + _exp(-(v + 40.0) / -0.5))


def stn_taub(v):
    return 200.0 / (_exp(-(v + 60.0) / -30.0) + _exp(-(v + 40.0) / 10.0))


def stn_tauc(v):
    return 45.0 + 10.0 / (_exp(-(v + 27.0) / -20.0) + _exp(-(v + 50.0) / 15.0))


def stn_taud1(v):
    return 400.0 + 500.0 / (_exp(-(v + 40.0) / -15.0) + _exp(-(v + 20.0) / 20.0))


def stn_tauh(v):
    return 24.5 / (_exp(-(v + 50.0) / -15.0) + _exp(-(v + 50.0) / 16.0))


def stn_taum(v):
    return 0.2 + 3.0 / (1.0 + _exp(-(v + 53.0) / -0.7))


def stn_taun(v):
    return 11.0 / (_exp(-(v + 40.0) / -40.0) + _exp(-(v + 40.0) / 50.0))


def stn_taup(v):
    return 5.0 + 0.33 / (_exp(-(v + 27.0) / -10.0) + _exp(-(v + 102.0) / 15.0))


def stn_tauq(v):
    return 400.0 / (_exp(-(v + 50.0) / -15.0) + _exp(-(v + 50.0) / 16.0))


# ---------------------------------------------------------------------------
# GPe / GPi kinetics
# ---------------------------------------------------------------------------


def gpe_ainf(v):
    return 1.0 / (1.0 + _exp(-(v + 57.0) / 2.0))


def gpe_hinf(v):
    return 1.0 / (1.0 + _exp((v + 58.0) / 12.0))


def gpe_minf(v):
    return 1.0 / (1.0 + _exp(-(v + 37.0) / 10.0))


def gpe_ninf(v):
    return 1.0 / (1.0 + _exp(-(v + 50.0) / 14.0))


def gpe_rinf(v):
    return 1.0 / (1.0 + _exp((v + 70.0) / 2.0))


def gpe_sinf(v):
    return 1.0 / (1.0 + _exp(-(v + 35.0) / 2.0))


def gpe_tauh(v):
    return 0.05 + 0.27 / (1.0 + _exp(-(v + 40.0) / -12.0))


def gpe_taun(v):
    return 0.05 + 0.27 / (1.0 + _exp(-(v + 40.0) / -12.0))


# ---------------------------------------------------------------------------
# Striatal kinetics
# ---------------------------------------------------------------------------


def alphah(v):
    return 0.128 * _exp((-50.0 - v) / 18.0)


def alpham(v):
    return 0.32 * _x_over_1_minus_exp_neg_x_over_s(54.0 + v, 4.0)


def alphan(v):
    return 0.032 * _x_over_1_minus_exp_neg_x_over_s(52.0 + v, 5.0)


def alphap(v):
    return 3.209e-4 * _x_over_1_minus_exp_neg_x_over_s(30.0 + v, 9.0)


def betah(v):
    return 4.0 / (1.0 + _exp((-27.0 - v) / 5.0))


def betan(v):
    return 0.5 * _exp((-57.0 - v) / 40.0)


def betam(v):
    return 0.28 * _x_over_exp_x_over_s_minus_1(27.0 + v, 5.0)


def betap(v):
    return 3.209e-4 * _x_over_exp_x_over_s_minus_1(30.0 + v, 9.0)


def Ggaba(v):
    return 2.0 * (1.0 + torch.tanh(v / 4.0))


# Pathway delay stream layout used by the batched delayed-state API.  The order
# here is also the unpack order used before synaptic filter updates.
_DELAY_PATHWAYS = (
    "th_ctx",
    "stn_gpe",
    "stn_gpi",
    "gpe_stn",
    "gpe_gpi",
    "gpe_gpe",
    "gpi_th",
    "d2_gpe",
    "d1_gpi",
    "ctx_d2",
    "ctx_d1",
    "ctx_stn",
)

_EXP2_SCALAR_COEFFICIENTS = {
    (0.4, 2.5): ("_exp2_exact_decay1_0", "_exp2_exact_decay2_0", "_exp2_inc_0"),
    (2.0, 67.0): ("_exp2_exact_decay1_1", "_exp2_exact_decay2_1", "_exp2_inc_1"),
    (0.4, 7.7): ("_exp2_exact_decay1_2", "_exp2_exact_decay2_2", "_exp2_inc_2"),
    (0.5, 2.49): ("_exp2_exact_decay1_3", "_exp2_exact_decay2_3", "_exp2_inc_3"),
    (2.0, 90.0): ("_exp2_exact_decay1_4", "_exp2_exact_decay2_4", "_exp2_inc_4"),
}


# ---------------------------------------------------------------------------
# Fused mechanism
# ---------------------------------------------------------------------------


class _Kumaravelu2016FusedBase(V):
    """Base fused mechanism; use ``make_kumaravelu_2016_fused`` to bind config."""

    CONFIG: Dict[str, Any] = {}

    # Accepted-step state is explicit CARRY so checkpointing and functional
    # schemas can distinguish it from repeatable algebra and static workspaces.
    # These layouts depend on the configured network size, optional Population
    # batch prefix, and delay depths, so pure initialization resolves and freezes
    # every deferred shape exactly once.
    V.CARRY(
        # exposed voltage/state summaries
        "v_all", "spikes", "syn_spikes", "ap_spikes",
        # voltages
        "v_th", "v_stn", "v_gpe", "v_gpi", "v_d2", "v_d1", "v_rs", "v_fs",
        # TH states
        "H1", "R1",
        # STN states
        "N2", "H2", "M2", "A2", "B2", "C2", "D1", "D2", "P2", "Q2", "R2", "CAsn2",
        # GPe / GPi states
        "N3", "H3", "R3", "CA3", "N4", "H4", "R4", "CA4",
        # Striatal states
        "m5", "h5", "n5", "p5", "m6", "h6", "n6", "p6",
        # Cortex recovery states
        "u_rs", "u_fs",
        # Synaptic conductance/filter states
        "S2a", "S2an", "S2b", "S3a", "S3b", "S3c", "S4", "S5", "S6a", "S6b", "S6bn",
        "S7", "S8", "S9", "S1a", "Z1a", "S1b", "Z1b", "S1c",
        # Double-exponential hidden states
        "A_stn_gpe_a", "B_stn_gpe_a", "A_stn_gpe_n", "B_stn_gpe_n",
        "A_gpe_stn", "B_gpe_stn", "A_ctx_stn_a", "B_ctx_stn_a", "A_ctx_stn_n", "B_ctx_stn_n",
        # Alpha hidden Z states for non-cortical pathways
        "Z_th_ctx", "Z_stn_gpi", "Z_gpe_gpi", "Z_gpe_gpe", "Z_gpi_th",
        "Z_d2_gpe", "Z_d1_gpi", "Z_ctx_d2", "Z_ctx_d1",
        # Coalesced pathway-delay queue.
        "buf_pathway_delays",
        # Optional per-pathway delay queues used by delay_mode="circular_eager".
        "buf_th_ctx", "buf_stn_gpe", "buf_stn_gpi", "buf_gpe_stn", "buf_gpe_gpi", "buf_gpe_gpe",
        "buf_gpi_th", "buf_d2_gpe", "buf_d1_gpi", "buf_ctx_d2", "buf_ctx_d1", "buf_ctx_stn",
        shape="deferred",
    )
    V.CARRY(
        "buf_pathway_delays_ptr",
        "buf_th_ctx_ptr", "buf_stn_gpe_ptr", "buf_stn_gpi_ptr",
        "buf_gpe_stn_ptr", "buf_gpe_gpi_ptr", "buf_gpe_gpe_ptr",
        "buf_gpi_th_ptr", "buf_d2_gpe_ptr", "buf_d1_gpi_ptr",
        "buf_ctx_d2_ptr", "buf_ctx_d1_ptr", "buf_ctx_stn_ptr",
        dtype=torch.long,
        shape="deferred",
    )

    # Intracellular waveform evaluation is repeatable algebra at the current
    # model time; it must not become checkpoint carry.
    V.ASSIGNED("i_inj")

    V.DERIVED_BUFFER(
        "spike_crossing_thresholds", "ctx_reset_thresholds",
        "stim_idbs", "stim_iappco",
        "gcorsna", "gcorsnn", "gcordrstr", "ggege", "gsngen", "gsngea", "gsngi",
        shape="deferred",
    )
    V.DERIVED_BUFFER(
        "idx_roll_p1", "idx_roll_m1", "idx_roll_p2", "idx_sum_10",
        "pathway_delays_delay_steps",
        "perm_d2_0", "perm_d2_1", "perm_d2_2", "perm_d2_3",
        "perm_d1_0", "perm_d1_1", "perm_d1_2",
        "perm_fsrs_0", "perm_fsrs_1", "perm_fsrs_2", "perm_fsrs_3",
        "perm_rsfs_0", "perm_rsfs_1", "perm_rsfs_2", "perm_rsfs_3",
        dtype=torch.long,
        shape="deferred",
    )

    # Exact-discretization scalar coefficients retain their true scalar layout.
    V.TIMESTEP_BUFFER(
        "_alpha_dt", "_alpha_h", "_alpha_decay", "_alpha_dt_over_tau2",
        "_alpha_const_peak", "_alpha_const_peak1",
        "_exp2_exact_decay1_0", "_exp2_exact_decay1_1", "_exp2_exact_decay1_2",
        "_exp2_exact_decay1_3", "_exp2_exact_decay1_4",
        "_exp2_exact_decay2_0", "_exp2_exact_decay2_1", "_exp2_exact_decay2_2",
        "_exp2_exact_decay2_3", "_exp2_exact_decay2_4",
        "_exp2_inc_0", "_exp2_inc_1", "_exp2_inc_2", "_exp2_inc_3", "_exp2_inc_4",
        shape=(),
    )

    # Coalesced pathway coefficients are structural axes, independent of the
    # Population morphology and any explicit batch prefix.
    V.TIMESTEP_BUFFER("alpha_const_streams", shape=(1, 11, 1))
    V.TIMESTEP_BUFFER(
        "exp2_tau1_streams", "exp2_tau2_streams", "exp2_inc_streams",
        "exp2_euler_decay1_streams", "exp2_euler_decay2_streams",
        "exp2_be_decay1_streams", "exp2_be_decay2_streams",
        "exp2_exact_decay1_streams", "exp2_exact_decay2_streams",
        shape=(1, 5, 1),
    )

    # Spike-surrogate settings are trainable/tunable Dendra parameters.
    V.PARAMETER(tau_gate=0.5, ste_scale=1.0)

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    @property
    def cfg(self) -> Mapping[str, Any]:
        return self.__class__.CONFIG

    def _n(self) -> int:
        return int(self.cfg["n"])

    def _group(self, v, k: int):
        n = self._n()
        return v[..., k * n : (k + 1) * n]

    def _cat_v(self, v_th, v_stn, v_gpe, v_gpi, v_d2, v_d1, v_rs, v_fs):
        return torch.cat((v_th, v_stn, v_gpe, v_gpi, v_d2, v_d1, v_rs, v_fs), dim=-1)

    def _zero_group(self, ref):
        return torch.zeros_like(ref)

    def _as_vector(self, values, ref, *, dtype=None):
        n = ref.shape[-1]
        dtype = ref.dtype if dtype is None else dtype
        x = torch.as_tensor(values, device=ref.device, dtype=dtype)
        if x.ndim == 0:
            x = x.reshape(()).expand_as(ref)
            return x.clone()
        if x.numel() == n:
            x = x.reshape(*([1] * (ref.ndim - 1)), n)
            return x.expand_as(ref).clone()
        if x.numel() == ref.numel():
            return x.reshape_as(ref).clone()
        if x.shape[-1] == n:
            while x.ndim < ref.ndim:
                x = x.unsqueeze(0)
            return x.expand_as(ref).clone()
        raise ValueError(
            f"Cannot broadcast vector with shape {tuple(x.shape)} to fused group shape {tuple(ref.shape)}."
        )

    def _as_index(self, values, ref):
        n = ref.shape[-1]
        x = torch.as_tensor(values, device=ref.device, dtype=torch.long)
        if x.numel() == n:
            return x.reshape(n)
        if x.numel() == ref.numel():
            return x.reshape_as(ref)
        if x.shape[-1] == n:
            while x.ndim < ref.ndim:
                x = x.unsqueeze(0)
            return x.expand_as(ref).clone()
        raise ValueError(
            f"Cannot broadcast permutation with shape {tuple(x.shape)} to fused group shape {tuple(ref.shape)}."
        )

    def _routing_buffers(self, ref):
        n = self._n()
        base = torch.arange(n, device=ref.device, dtype=torch.long)
        shifts = torch.arange(min(10, n), device=ref.device, dtype=torch.long)
        return {
            "idx_roll_p1": ((base - 1) % n).reshape(1, n),
            "idx_roll_m1": ((base + 1) % n).reshape(1, n),
            "idx_roll_p2": ((base - 2) % n).reshape(1, n),
            "idx_sum_10": (base.unsqueeze(0) - shifts.unsqueeze(1)) % n,
        }

    def _spike_threshold_buffers(self, ref):
        """Precompute static threshold vectors used by coalesced spike detection."""
        return {
            "spike_crossing_thresholds": ref.new_tensor((-10.0, -20.0)),
            "ctx_reset_thresholds": ref.new_tensor(
                (
                    float(self.cfg["ctx_rs"].get("v_peak", 30.0)),
                    float(self.cfg["ctx_fs"].get("v_peak", 30.0)),
                )
            ),
        }

    def _stimulus_buffers(self, ref):
        samples = self.cfg.get("stim_samples", {})
        outputs = {}
        for buffer_name, key in (
            ("stim_idbs", "Idbs"),
            ("stim_iappco", "Iappco"),
        ):
            values = samples.get(key)
            if values is None:
                value = torch.empty(0, device=ref.device, dtype=ref.dtype)
            else:
                value = torch.as_tensor(values, device=ref.device, dtype=ref.dtype)
                if value.ndim != 2:
                    raise ValueError(
                        f"Configured {key} samples must have shape (network, time); "
                        f"got {tuple(value.shape)}."
                    )
                if value.shape[0] != ref.shape[-2]:
                    raise ValueError(
                        f"Configured {key} samples have {value.shape[0]} networks, "
                        f"but the fused state has {ref.shape[-2]}."
                    )
            outputs[buffer_name] = value
        return outputs

    def derive_buffers(self):
        """Purely derive routing, stimulus, realization, and delay metadata."""
        ref = self._group(self.diam, 0)
        outputs = {
            **self._routing_buffers(ref),
            **self._spike_threshold_buffers(ref),
            **self._stimulus_buffers(ref),
            "pathway_delays_delay_steps": torch.tensor(
                [
                    int(self.cfg["delay_steps"].get(name, 0))
                    for name in _DELAY_PATHWAYS
                ],
                device=ref.device,
                dtype=torch.long,
            ),
        }

        realization = self.cfg["realization"]
        for name in (
            "gcorsna",
            "gcorsnn",
            "gcordrstr",
            "ggege",
            "gsngen",
            "gsngea",
            "gsngi",
        ):
            outputs[name] = self._as_vector(realization[name], ref)
        for index in range(4):
            outputs[f"perm_d2_{index}"] = self._as_index(
                realization["str_d2_perms"][index], ref
            )
            outputs[f"perm_fsrs_{index}"] = self._as_index(
                realization["fs_to_rs_perms"][index], ref
            )
            outputs[f"perm_rsfs_{index}"] = self._as_index(
                realization["rs_to_fs_perms"][index], ref
            )
        for index in range(3):
            outputs[f"perm_d1_{index}"] = self._as_index(
                realization["str_d1_perms"][index], ref
            )
        return outputs

    def derive_timestep_buffers(self, dt):
        """Build all synaptic filter coefficients as pure tensor workspaces."""
        # Pathway delays are discretized into fixed-size queues when the fused
        # model class is constructed. Updating only the synaptic coefficients
        # for another runtime timestep would produce inconsistent dynamics.
        dt_ref = float(self.cfg["dt_ref"])
        dt_value = float(dt.detach())
        finfo = torch.finfo(dt.dtype)
        tolerance = 4.0 * finfo.eps * max(abs(dt_ref), abs(dt_value), finfo.tiny)
        if not math.isclose(dt_value, dt_ref, rel_tol=0.0, abs_tol=tolerance):
            raise ValueError(
                "Kumaravelu2016 has a fixed timestep because its pathway delay "
                f"queues were constructed for dt={dt_ref}; got runtime dt={dt_value}. "
                "Rebuild the model with the desired dt."
            )

        syn = self.cfg.get("syn", {})
        peak = float(syn.get("gpeak", 0.43))
        peak1 = float(syn.get("gpeak1", 0.3))
        tau_alpha = float(syn.get("tau_alpha", 5.0))

        def stream_tensor(values):
            return dt.new_tensor(values).reshape(1, -1, 1)

        # Stream order for coalesced alpha updates:
        #   S7, S2b, S3b, S3c, S4, S5, S9, S6a(ctx_d2),
        #   S6a(ctx_d1), S1a, S1b.
        alpha_peaks = stream_tensor(
            [peak, peak, peak1, peak1, peak1, peak1, peak1, peak, peak, peak, peak]
        )
        alpha_consts = alpha_peaks / (tau_alpha * math.exp(-1.0))

        # Stream order for coalesced exp2 updates:
        #   STN->GPe AMPA, STN->GPe NMDA, GPe->STN,
        #   CTX->STN AMPA, CTX->STN NMDA.
        tau1s = stream_tensor([0.4, 2.0, 0.4, 0.5, 2.0])
        tau2s = stream_tensor([2.5, 67.0, 7.7, 2.49, 90.0])
        exp2_peaks = stream_tensor([peak, peak, peak1, peak, peak])
        tp = tau1s * tau2s / (tau2s - tau1s) * torch.log(tau2s / tau1s)
        incs = exp2_peaks / (-torch.exp(-tp / tau1s) + torch.exp(-tp / tau2s))
        alpha_h = dt / tau_alpha
        exact_decay1 = torch.exp(-dt / tau1s)
        exact_decay2 = torch.exp(-dt / tau2s)

        workspaces = {
            "_alpha_dt": dt.clone(),
            "_alpha_h": alpha_h,
            "_alpha_decay": torch.exp(-alpha_h),
            "_alpha_dt_over_tau2": dt / (tau_alpha * tau_alpha),
            "_alpha_const_peak": alpha_consts[0, 0, 0],
            "_alpha_const_peak1": alpha_consts[0, 2, 0],
            "alpha_const_streams": alpha_consts,
            "exp2_tau1_streams": tau1s,
            "exp2_tau2_streams": tau2s,
            "exp2_inc_streams": incs,
            "exp2_euler_decay1_streams": 1.0 - dt / tau1s,
            "exp2_euler_decay2_streams": 1.0 - dt / tau2s,
            "exp2_be_decay1_streams": 1.0 / (1.0 + dt / tau1s),
            "exp2_be_decay2_streams": 1.0 / (1.0 + dt / tau2s),
            "exp2_exact_decay1_streams": exact_decay1,
            "exp2_exact_decay2_streams": exact_decay2,
        }
        for index in range(5):
            workspaces[f"_exp2_exact_decay1_{index}"] = exact_decay1[0, index, 0]
            workspaces[f"_exp2_exact_decay2_{index}"] = exact_decay2[0, index, 0]
            workspaces[f"_exp2_inc_{index}"] = incs[0, index, 0]
        return workspaces

    def _delay_mode(self) -> str:
        return str(self.cfg.get("delay_mode", "auto")).lower()

    def _synapse_update_mode(self) -> str:
        return str(self.cfg.get("synapse_update_mode", "uncoalesced")).lower()

    def _spike_update_mode(self) -> str:
        return str(self.cfg.get("spike_update_mode", "uncoalesced")).lower()

    def _initial_delay_values(self, ref):
        """Return every delay queue/pointer in its configured frozen layout."""
        pointer = torch.zeros((), device=ref.device, dtype=torch.long)
        empty = torch.empty(0, device=ref.device, dtype=ref.dtype)
        outputs = {"buf_pathway_delays_ptr": pointer.clone()}
        outputs.update(
            {f"buf_{name}_ptr": pointer.clone() for name in _DELAY_PATHWAYS}
        )

        if self._delay_mode() == "circular_eager":
            outputs["buf_pathway_delays"] = empty.clone()
            for name in _DELAY_PATHWAYS:
                steps = int(self.cfg["delay_steps"].get(name, 0))
                depth = max(1, steps + 1)
                outputs[f"buf_{name}"] = torch.zeros(
                    (*ref.shape[:-1], depth, ref.shape[-1]),
                    device=ref.device,
                    dtype=ref.dtype,
                )
            return outputs

        values_shape = (
            *ref.shape[:-1],
            len(_DELAY_PATHWAYS),
            ref.shape[-1],
        )
        max_steps = max(
            int(self.cfg["delay_steps"].get(name, 0))
            for name in _DELAY_PATHWAYS
        )
        outputs["buf_pathway_delays"] = torch.zeros(
            (max(1, max_steps + 1), *values_shape),
            device=ref.device,
            dtype=ref.dtype,
        )
        outputs.update({f"buf_{name}": empty.clone() for name in _DELAY_PATHWAYS})
        return outputs

    def _pathway_delay_spec(self, values):
        """Build static Python metadata for the coalesced delay helper."""
        max_steps = max(
            int(self.cfg["delay_steps"].get(name, 0))
            for name in _DELAY_PATHWAYS
        )
        return {
            "buffer": "buf_pathway_delays",
            "pointer": "buf_pathway_delays_ptr",
            "steps_buffer": "pathway_delays_delay_steps",
            "steps": max_steps,
            "depth": max(1, max_steps + 1),
            "axis": 0,
            "stream_axis": values.ndim - 1,
            "value_stream_axis": values.ndim - 2,
            "value_shape": tuple(values.shape),
            "n_streams": len(_DELAY_PATHWAYS),
            "batched": True,
            "has_zero_delay": any(
                int(self.cfg["delay_steps"].get(name, 0)) == 0
                for name in _DELAY_PATHWAYS
            ),
            "all_zero_delay": all(
                int(self.cfg["delay_steps"].get(name, 0)) == 0
                for name in _DELAY_PATHWAYS
            ),
        }

    def _delay_pathways_batched(
        self,
        spk_th,
        spk_stn,
        spk_gpe,
        spk_gpi,
        spk_d2,
        spk_d1,
        spk_rs_reset,
    ):
        """Delay all pathways and return both values and explicit carry updates."""
        values = torch.stack(
            (
                spk_th,
                spk_stn,
                spk_stn,
                spk_gpe,
                spk_gpe,
                spk_gpe,
                spk_gpi,
                spk_d2,
                spk_d1,
                spk_rs_reset,
                spk_rs_reset,
                spk_rs_reset,
            ),
            dim=-2,
        )
        spec = self._pathway_delay_spec(values)
        if spec["all_zero_delay"]:
            return tuple(values.unbind(dim=-2)), {}

        steps = self.pathway_delays_delay_steps
        mode = self._delay_mode()
        if mode == "auto":
            mode = "shift" if (self.training or torch.is_grad_enabled()) else "circular"

        if mode == "shift":
            queue = self.buf_pathway_delays
            queue_new = torch.cat((values.unsqueeze(0), queue[:-1]), dim=0)
            delayed = self._delayed_states_gather(queue_new, spec, steps)
            return tuple(delayed.unbind(dim=-2)), {
                "buf_pathway_delays": queue_new
            }

        # Deliberate framework-owned performance exception: eval circular
        # backends update the declared queue/pointer in place under no_grad.
        # Copying the full queue every accepted step defeats the purpose of the
        # fused fast path. Both tensors remain explicit CARRY and are returned
        # for normal schema/checkpoint accounting; functional lowering may fail
        # closed on this helper until it has a dedicated pure circular operator.
        delayed = self._delayed_states_circular(spec, values, steps)
        return tuple(delayed.unbind(dim=-2)), {
            "buf_pathway_delays": self.buf_pathway_delays,
            "buf_pathway_delays_ptr": self.buf_pathway_delays_ptr,
        }

    def _delay_one_circular_eager(self, name: str, spike):
        steps = int(self.cfg["delay_steps"].get(name, 0))
        if steps <= 0:
            return spike, {}
        spec = {
            "buffer": f"buf_{name}",
            "pointer": f"buf_{name}_ptr",
            "axis": spike.ndim - 1,
        }
        delayed = self._delayed_state_circular(spec, spike, steps)
        return delayed, {
            f"buf_{name}": getattr(self, f"buf_{name}"),
            f"buf_{name}_ptr": getattr(self, f"buf_{name}_ptr"),
        }

    @torch._dynamo.disable
    def _delay_pathways_circular_eager(
        self,
        spk_th,
        spk_stn,
        spk_gpe,
        spk_gpi,
        spk_d2,
        spk_d1,
        spk_rs_reset,
    ):
        """Update per-pathway rings in one eager island and expose their carry."""
        pairs = (
            self._delay_one_circular_eager("th_ctx", spk_th),
            self._delay_one_circular_eager("stn_gpe", spk_stn),
            self._delay_one_circular_eager("stn_gpi", spk_stn),
            self._delay_one_circular_eager("gpe_stn", spk_gpe),
            self._delay_one_circular_eager("gpe_gpi", spk_gpe),
            self._delay_one_circular_eager("gpe_gpe", spk_gpe),
            self._delay_one_circular_eager("gpi_th", spk_gpi),
            self._delay_one_circular_eager("d2_gpe", spk_d2),
            self._delay_one_circular_eager("d1_gpi", spk_d1),
            self._delay_one_circular_eager("ctx_d2", spk_rs_reset),
            self._delay_one_circular_eager("ctx_d1", spk_rs_reset),
            self._delay_one_circular_eager("ctx_stn", spk_rs_reset),
        )
        updates = {}
        for _, local_updates in pairs:
            updates.update(local_updates)
        return tuple(value for value, _ in pairs), updates

    def _delay_pathways(
        self,
        spk_th,
        spk_stn,
        spk_gpe,
        spk_gpi,
        spk_d2,
        spk_d1,
        spk_rs_reset,
    ):
        """Delay fixed-delay pathway spikes using the selected backend."""
        if self._delay_mode() == "circular_eager":
            return self._delay_pathways_circular_eager(
                spk_th, spk_stn, spk_gpe, spk_gpi, spk_d2, spk_d1, spk_rs_reset
            )
        return self._delay_pathways_batched(
            spk_th, spk_stn, spk_gpe, spk_gpi, spk_d2, spk_d1, spk_rs_reset
        )

    # ------------------------------------------------------------------
    # Mechanism-level waveform injection
    # ------------------------------------------------------------------

    def inject(self, waveform, *, index=None, shape=None, model_shape=None, model=None, **kwargs):
        """Accept ``model[idx].inject(waveform)`` stimuli for the fused model.

        The waveform is evaluated each timestep in :meth:`assigned_values` and exposed
        as ``i_inj`` with the same full ``8*n`` voltage layout as ``v_all``.
        Only compartments selected by ``idx`` receive nonzero current; all other
        entries are padded with zeros by the base helper.
        """
        return self.register_waveform_injection(
            waveform,
            index=index,
            model_shape=model_shape,
            model=model,
            current_name="i_inj",
        )

    # ------------------------------------------------------------------
    # Dendra hooks
    # ------------------------------------------------------------------

    def initial_values(self, v, values):
        """Return the complete pure initial CARRY mapping."""
        del values
        cfg = self.cfg
        n = self._n()
        if v.shape[-1] != 8 * n:
            raise ValueError(
                f"Kumaravelu fused mechanism expected last voltage dimension 8*n={8*n}; got {v.shape[-1]}."
            )

        # Voltages from the Population v_init vector.
        v_th = self._group(v, 0).clone()
        v_stn = self._group(v, 1).clone()
        v_gpe = self._group(v, 2).clone()
        v_gpi = self._group(v, 3).clone()
        v_d2 = self._group(v, 4).clone()
        v_d1 = self._group(v, 5).clone()
        v_rs = self._group(v, 6).clone()
        v_fs = self._group(v, 7).clone()

        z = torch.zeros_like(v_th)
        one = torch.ones_like(v_th)
        outputs = {
            "v_th": v_th,
            "v_stn": v_stn,
            "v_gpe": v_gpe,
            "v_gpi": v_gpi,
            "v_d2": v_d2,
            "v_d1": v_d1,
            "v_rs": v_rs,
            "v_fs": v_fs,
        }

        # Intrinsic states at steady-state/in MATLAB initial values.
        outputs.update(
            {
                "H1": th_hinf(v_th),
                "R1": th_rinf(v_th),
                "N2": stn_ninf(v_stn),
                "H2": stn_hinf(v_stn),
                "M2": stn_minf(v_stn),
                "A2": stn_ainf(v_stn),
                "B2": stn_binf(v_stn),
                "C2": stn_cinf(v_stn),
                "D2": stn_d2inf(v_stn),
                "D1": stn_d1inf(v_stn),
                "P2": stn_pinf(v_stn),
                "Q2": stn_qinf(v_stn),
                "R2": stn_rinf(v_stn),
                "CAsn2": 0.005 * one,
                "N3": gpe_ninf(v_gpe),
                "H3": gpe_hinf(v_gpe),
                "R3": gpe_rinf(v_gpe),
                "CA3": 0.1 * one,
                "N4": gpe_ninf(v_gpi),
                "H4": gpe_hinf(v_gpi),
                "R4": gpe_rinf(v_gpi),
                "CA4": 0.1 * one,
            }
        )

        am5, ah5, an5, ap5 = alpham(v_d2), alphah(v_d2), alphan(v_d2), alphap(v_d2)
        bm5, bh5, bn5, bp5 = betam(v_d2), betah(v_d2), betan(v_d2), betap(v_d2)
        outputs.update(
            {
                "m5": am5 / (am5 + bm5),
                "h5": ah5 / (ah5 + bh5),
                "n5": an5 / (an5 + bn5),
                "p5": ap5 / (ap5 + bp5),
            }
        )

        am6, ah6, an6, ap6 = alpham(v_d1), alphah(v_d1), alphan(v_d1), alphap(v_d1)
        bm6, bh6, bn6, bp6 = betam(v_d1), betah(v_d1), betan(v_d1), betap(v_d1)
        outputs.update(
            {
                "m6": am6 / (am6 + bm6),
                "h6": ah6 / (ah6 + bh6),
                "n6": an6 / (an6 + bn6),
                "p6": ap6 / (ap6 + bp6),
            }
        )

        # Cortical recovery variables.
        outputs["u_rs"] = float(cfg["ctx_rs"]["b"]) * v_rs
        outputs["u_fs"] = float(cfg["ctx_fs"]["b"]) * v_fs

        # Synaptic filter states.
        for name in (
            "S2a", "S2an", "S2b", "S3a", "S3b", "S3c", "S4", "S5", "S6a", "S6b", "S6bn",
            "S7", "S8", "S9", "S1a", "Z1a", "S1b", "Z1b", "S1c",
            "A_stn_gpe_a", "B_stn_gpe_a", "A_stn_gpe_n", "B_stn_gpe_n",
            "A_gpe_stn", "B_gpe_stn", "A_ctx_stn_a", "B_ctx_stn_a", "A_ctx_stn_n", "B_ctx_stn_n",
            "Z_th_ctx", "Z_stn_gpi", "Z_gpe_gpi", "Z_gpe_gpe", "Z_gpi_th",
            "Z_d2_gpe", "Z_d1_gpi", "Z_ctx_d2", "Z_ctx_d1",
        ):
            outputs[name] = z.clone()

        outputs.update(self._initial_delay_values(v_th))
        outputs["spikes"] = torch.zeros_like(v)
        outputs["syn_spikes"] = torch.zeros_like(v)
        outputs["ap_spikes"] = torch.zeros_like(v)
        outputs["v_all"] = self._cat_v(
            v_th, v_stn, v_gpe, v_gpi, v_d2, v_d1, v_rs, v_fs
        )
        return outputs

    def assigned_values(self, v, values):
        """Evaluate the current-time intracellular waveform without carry."""
        del v
        return {
            "i_inj": self.evaluate_injections(
                values["v_all"], current_name="i_inj"
            )
        }

    def update_v(self, v):
        # scnv calls update_v() before advance(); return the current exposed
        # voltage vector. advance() will return v_all for the accepted step.
        return self.v_all

    # ------------------------------------------------------------------
    # Spike/event helpers
    # ------------------------------------------------------------------

    def _surrogate_tau(self):
        # Match Dendra's spike-detector convention: keep the surrogate
        # temperature positive and avoid extremely sharp, numerically fragile
        # logistic gates.
        return self.tau_gate.clamp_min(1.0e-3)

    def _crossing_spike(self, v_old, v_new, threshold: float):
        """Upward crossing event using Dendra's canonical STE surrogate.

        This base implementation is intentionally always differentiable.  The
        factory below selects a hard-spike subclass when
        ``CONFIG['differentiable_spikes']`` is false, so forward-only runs do not
        pay for sigmoid/ReLU surrogate calculations or tensor-valued branches.
        """
        return _dendra_crossing_spike(
            v_old,
            v_new,
            threshold,
            tau=self._surrogate_tau(),
            ste_scale=self.ste_scale,
        )

    def _level_spike(self, v, threshold: float):
        """Above-threshold event using Dendra's canonical STE surrogate."""
        return _dendra_level_spike(
            v,
            threshold,
            tau=self._surrogate_tau(),
            ste_scale=self.ste_scale,
        )

    def _crossing_spikes(self, v_old, v_new, thresholds):
        """Broadcasted upward crossing events using Dendra's STE surrogate."""
        return _dendra_crossing_spikes(
            v_old,
            v_new,
            thresholds,
            tau=self._surrogate_tau(),
            ste_scale=self.ste_scale,
        )

    def _level_spikes(self, v, thresholds):
        """Broadcasted above-threshold events using Dendra's STE surrogate."""
        return _dendra_level_spikes(
            v,
            thresholds,
            tau=self._surrogate_tau(),
            ste_scale=self.ste_scale,
        )

    # ------------------------------------------------------------------
    # Synaptic filter helpers
    # ------------------------------------------------------------------

    def _alpha_step(self, s, z, spike, peak: float, tau: float, dt):
        const = peak / (tau * math.exp(-1.0))
        s_new = s + dt * z
        z_new = z + const * spike - dt * ((2.0 / tau) * z + (1.0 / (tau * tau)) * s_new)
        return s_new, z_new

    def _exp2_step(self, a, b, spike, peak: float, tau1: float, tau2: float, dt):
        tp = (tau1 * tau2) / (tau2 - tau1) * math.log(tau2 / tau1)
        factor = 1.0 / (-math.exp(-tp / tau1) + math.exp(-tp / tau2))
        inc = peak * factor * spike
        a_new = a + inc - dt * (a / tau1)
        b_new = b + inc - dt * (b / tau2)
        return a_new, b_new, b_new - a_new

    def _dbs_current(self, ref, dt):
        sampled = self._sampled_current(self.stim_idbs, ref, dt)
        if sampled is not None:
            return sampled
        dbs = self.cfg.get("dbs", {})
        freq = float(dbs.get("freq_hz", 0.0))
        amp = float(dbs.get("amplitude", 0.0))
        pw = float(dbs.get("pulse_width_ms", 0.3))
        if freq <= 0.0 or amp == 0.0:
            return torch.zeros_like(ref)
        t_ms = self.t + dt
        isi = 1000.0 / freq
        in_pulse = (torch.remainder(t_ms, isi) < pw).to(ref.dtype)
        return torch.zeros_like(ref) + amp * in_pulse

    def _ctx_stim_current(self, ref, dt):
        sampled = self._sampled_current(self.stim_iappco, ref, dt)
        if sampled is not None:
            return sampled
        stim = self.cfg.get("ctx_stim", {})
        enabled = bool(stim.get("enabled", False))
        amp = float(stim.get("amplitude", 0.0))
        if (not enabled) or amp == 0.0:
            return torch.zeros_like(ref)
        t_ms = self.t + dt
        start = float(stim.get("start_ms", 1000.0))
        stop = start + float(stim.get("duration_ms", 0.3))
        on = ((t_ms >= start) & (t_ms <= stop)).to(ref.dtype)
        return torch.zeros_like(ref) + amp * on

    def _sampled_current(self, samples, ref, dt):
        """Return the MATLAB sample used for the pending Euler transition."""
        if samples.numel() == 0:
            return None
        # MATLAB advances from column i-1 to i using stimulus sample i.  Dendra
        # enters this method at the old time, so select (t + dt) / dt.
        index = torch.round((self.t + dt) / dt).to(torch.long)
        index = index.clamp(0, samples.shape[-1] - 1)
        value = samples[..., index].unsqueeze(-1)
        while value.ndim < ref.ndim:
            value = value.unsqueeze(0)
        return value.expand_as(ref)

    # ------------------------------------------------------------------
    # Fused explicit Euler step
    # ------------------------------------------------------------------

    def advance(self, v, dt, values):  # noqa: C901, PLR0915 - intentionally fused
        del v
        cfg = self.cfg
        p = cfg["constants"]
        c = cfg["coupling"]
        syn = cfg["syn"]

        # Old voltages/states.
        V1, V2, V3, V4 = self.v_th, self.v_stn, self.v_gpe, self.v_gpi
        V5, V6, V7, V8 = self.v_d2, self.v_d1, self.v_rs, self.v_fs

        # Optional user-supplied waveform stimulation.  The exposed vector uses
        # the same layout as v_all: TH, STN, GPe, GPi, StrD2, StrD1, CTX_RS, CTX_FS.
        Iinj_all = values["i_inj"]
        Iinj1 = self._group(Iinj_all, 0)
        Iinj2 = self._group(Iinj_all, 1)
        Iinj3 = self._group(Iinj_all, 2)
        Iinj4 = self._group(Iinj_all, 3)
        Iinj5 = self._group(Iinj_all, 4)
        Iinj6 = self._group(Iinj_all, 5)
        Iinj7 = self._group(Iinj_all, 6)
        Iinj8 = self._group(Iinj_all, 7)

        # Routing aliases using precomputed gather indices.  This avoids
        # repeated torch.roll kernels in the timestep hot path.
        S21a = _gather_index(self.S2a, self.idx_roll_p1)
        S21an = _gather_index(self.S2an, self.idx_roll_p1)
        S21b = _gather_index(self.S2b, self.idx_roll_p1)
        S31a = _gather_index(self.S3a, self.idx_roll_m1)
        S31b = _gather_index(self.S3b, self.idx_roll_m1)
        S31c = _gather_index(self.S3c, self.idx_roll_m1)
        S32b = _gather_index(self.S3b, self.idx_roll_p2)
        S32c = _gather_index(self.S3c, self.idx_roll_p2)
        S61b = _gather_index(self.S6b, self.idx_roll_m1)
        S61bn = _gather_index(self.S6bn, self.idx_roll_m1)
        S5sum = _sum_gather_index(self.S5, self.idx_sum_10)
        S9sum = _sum_gather_index(self.S9, self.idx_sum_10)

        S11cr = _gather_perm(self.S1c, self.perm_d2_0)
        S12cr = _gather_perm(self.S1c, self.perm_d2_1)
        S13cr = _gather_perm(self.S1c, self.perm_d2_2)
        S14cr = _gather_perm(self.S1c, self.perm_d2_3)
        S81r = _gather_perm(self.S8, self.perm_d1_0)
        S82r = _gather_perm(self.S8, self.perm_d1_1)
        S83r = _gather_perm(self.S8, self.perm_d1_2)

        S11br = _gather_perm(self.S1b, self.perm_fsrs_0)
        S12br = _gather_perm(self.S1b, self.perm_fsrs_1)
        S13br = _gather_perm(self.S1b, self.perm_fsrs_2)
        S14br = _gather_perm(self.S1b, self.perm_fsrs_3)
        S11ar = _gather_perm(self.S1a, self.perm_rsfs_0)
        S12ar = _gather_perm(self.S1a, self.perm_rsfs_1)
        S13ar = _gather_perm(self.S1a, self.perm_rsfs_2)
        S14ar = _gather_perm(self.S1a, self.perm_rsfs_3)

        # ---------------- Intrinsic and synaptic currents ----------------
        # TH
        Il1 = p["gl"][0] * (V1 - p["El"][0])
        Ina1 = p["gna"][0] * th_minf(V1) ** 3 * self.H1 * (V1 - p["Ena"][0])
        Ik1 = p["gk"][0] * (0.75 * (1.0 - self.H1)) ** 4 * (V1 - p["Ek"][0])
        It1 = p["gt"][0] * th_pinf(V1) ** 2 * self.R1 * (V1 - p["Et"])
        Igith = c["ggith"] * (V1 - p["Esyn"][5]) * self.S4

        # STN
        Ecasn = p["con"] * _log(p["Cao"] / _positive(self.CAsn2))
        Ina2 = p["gna"][1] * self.M2 ** 3 * self.H2 * (V2 - p["Ena"][1])
        Ik2 = p["gk"][1] * self.N2 ** 4 * (V2 - p["Ek"][1])
        Ia2 = p["ga"] * self.A2 ** 2 * self.B2 * (V2 - p["Ek"][1])
        IL2 = p["gL"] * self.C2 ** 2 * self.D1 * self.D2 * (V2 - Ecasn)
        It2 = p["gt"][1] * self.P2 ** 2 * self.Q2 * (V2 - Ecasn)
        Icak2 = p["gcak"] * self.R2 ** 2 * (V2 - p["Ek"][1])
        Il2 = p["gl"][1] * (V2 - p["El"][1])
        Igesn = c["ggesn"] * (V2 - p["Esyn"][0]) * (self.S3a + S31a)
        Icorsnampa = self.gcorsna * (V2 - p["Esyn"][1]) * (self.S6b + S61b)
        Icorsnnmda = self.gcorsnn * (V2 - p["Esyn"][1]) * (self.S6bn + S61bn)

        # GPe
        m3, n3, h3 = gpe_minf(V3), gpe_ninf(V3), gpe_hinf(V3)
        a3, s3, r3 = gpe_ainf(V3), gpe_sinf(V3), gpe_rinf(V3)
        Il3 = p["gl"][2] * (V3 - p["El"][2])
        Ik3 = p["gk"][2] * self.N3 ** 4 * (V3 - p["Ek"][2])
        Ina3 = p["gna"][2] * m3 ** 3 * self.H3 * (V3 - p["Ena"][2])
        It3 = p["gt"][2] * a3 ** 3 * self.R3 * (V3 - p["Eca"][2])
        Ica3 = p["gca"][2] * s3 ** 2 * (V3 - p["Eca"][2])
        Iahp3 = p["gahp"][2] * (V3 - p["Ek"][2]) * (self.CA3 / (self.CA3 + p["k1"][2]))
        Isngeampa = self.gsngea * (V3 - p["Esyn"][1]) * (self.S2a + S21a)
        Isngenmda = self.gsngen * (V3 - p["Esyn"][1]) * (self.S2an + S21an)
        Igege = 0.25 * (float(cfg["pd"]) * 3.0 + 1.0) * self.ggege * (V3 - p["Esyn"][2]) * (S31c + S32c)
        Istrgpe = c["gstrgpe"] * (V3 - p["Esyn"][5]) * S5sum
        Iappgpe = 3.0 - 2.0 * float(cfg["corstim"]) * (1.0 - float(cfg["pd"]))

        # GPi
        m4, n4, h4 = gpe_minf(V4), gpe_ninf(V4), gpe_hinf(V4)
        a4, s4, r4 = gpe_ainf(V4), gpe_sinf(V4), gpe_rinf(V4)
        Il4 = p["gl"][2] * (V4 - p["El"][2])
        Ik4 = p["gk"][2] * self.N4 ** 4 * (V4 - p["Ek"][2])
        Ina4 = p["gna"][2] * m4 ** 3 * self.H4 * (V4 - p["Ena"][2])
        It4 = p["gt"][2] * a4 ** 3 * self.R4 * (V4 - p["Eca"][2])
        Ica4 = p["gca"][2] * s4 ** 2 * (V4 - p["Eca"][2])
        Iahp4 = p["gahp"][2] * (V4 - p["Ek"][2]) * (self.CA4 / (self.CA4 + p["k1"][2]))
        Isngi = self.gsngi * (V4 - p["Esyn"][3]) * (self.S2b + S21b)
        Igigi = c["ggigi"] * (V4 - p["Esyn"][4]) * (S31b + S32b)
        Istrgpi = c["gstrgpi"] * (V4 - p["Esyn"][5]) * S9sum

        # Striatum
        Ina5 = p["gna"][3] * self.m5 ** 3 * self.h5 * (V5 - p["Ena"][3])
        Ik5 = p["gk"][3] * self.n5 ** 4 * (V5 - p["Ek"][3])
        Il5 = p["gl"][3] * (V5 - p["El"][3])
        Im5 = (2.6 - 1.1 * float(cfg["pd"])) * c["gm"] * self.p5 * (V5 - p["Em"])
        Igaba5 = (c["ggaba"] / 4.0) * (V5 - p["Esyn"][6]) * (S11cr + S12cr + S13cr + S14cr)
        Icorstr5 = c["gcorindrstr"] * (V5 - p["Esyn"][1]) * self.S6a

        Ina6 = p["gna"][3] * self.m6 ** 3 * self.h6 * (V6 - p["Ena"][3])
        Ik6 = p["gk"][3] * self.n6 ** 4 * (V6 - p["Ek"][3])
        Il6 = p["gl"][3] * (V6 - p["El"][3])
        Im6 = (2.6 - 1.1 * float(cfg["pd"])) * c["gm"] * self.p6 * (V6 - p["Em"])
        Igaba6 = (c["ggaba"] / 3.0) * (V6 - p["Esyn"][6]) * (S81r + S82r + S83r)
        Icorstr6 = self.gcordrstr * (V6 - p["Esyn"][1]) * self.S6a

        # Cortex
        Iie = c["gie"] * (V7 - p["Esyn"][0]) * (S11br + S12br + S13br + S14br)
        Ithcor = c["gthcor"] * (V7 - p["Esyn"][1]) * self.S7
        Iei = c["gei"] * (V8 - p["Esyn"][1]) * (S11ar + S12ar + S13ar + S14ar)

        # ---------------- Explicit Euler voltage/state updates ----------------
        Idbs = self._dbs_current(V2, dt)
        Iappco = self._ctx_stim_current(V7, dt)

        v_th_new = V1 + dt * (-Il1 - Ik1 - Ina1 - It1 - Igith + 1.2 + Iinj1)
        H1_new = self.H1 + dt * ((th_hinf(V1) - self.H1) / th_tauh(V1))
        R1_new = self.R1 + dt * ((th_rinf(V1) - self.R1) / th_taur(V1))

        v_stn_new = V2 + dt * (-Ina2 - Ik2 - Ia2 - IL2 - It2 - Icak2 - Il2 - Igesn - Icorsnampa - Icorsnnmda + Idbs + Iinj2)
        N2_new = self.N2 + dt * ((stn_ninf(V2) - self.N2) / stn_taun(V2))
        H2_new = self.H2 + dt * ((stn_hinf(V2) - self.H2) / stn_tauh(V2))
        M2_new = self.M2 + dt * ((stn_minf(V2) - self.M2) / stn_taum(V2))
        A2_new = self.A2 + dt * ((stn_ainf(V2) - self.A2) / stn_taua(V2))
        B2_new = self.B2 + dt * ((stn_binf(V2) - self.B2) / stn_taub(V2))
        C2_new = self.C2 + dt * ((stn_cinf(V2) - self.C2) / stn_tauc(V2))
        D2_new = self.D2 + dt * ((stn_d2inf(V2) - self.D2) / 130.0)
        D1_new = self.D1 + dt * ((stn_d1inf(V2) - self.D1) / stn_taud1(V2))
        P2_new = self.P2 + dt * ((stn_pinf(V2) - self.P2) / stn_taup(V2))
        Q2_new = self.Q2 + dt * ((stn_qinf(V2) - self.Q2) / stn_tauq(V2))
        R2_new = self.R2 + dt * ((stn_rinf(V2) - self.R2) / 2.0)
        CAsn2_new = self.CAsn2 + dt * ((-p["alp"] * (IL2 + It2)) - (p["Kca"] * self.CAsn2))

        v_gpe_new = V3 + dt * (-Il3 - Ik3 - Ina3 - It3 - Ica3 - Iahp3 - Isngeampa - Isngenmda - Igege - Istrgpe + Iappgpe + Iinj3)
        N3_new = self.N3 + dt * (0.1 * (n3 - self.N3) / gpe_taun(V3))
        H3_new = self.H3 + dt * (0.05 * (h3 - self.H3) / gpe_tauh(V3))
        R3_new = self.R3 + dt * ((r3 - self.R3) / 30.0)
        CA3_new = self.CA3 + dt * (1.0e-4 * (-Ica3 - It3 - p["kca"][2] * self.CA3))

        v_gpi_new = V4 + dt * (-Il4 - Ik4 - Ina4 - It4 - Ica4 - Iahp4 - Isngi - Igigi - Istrgpi + 3.0 + Iinj4)
        N4_new = self.N4 + dt * (0.1 * (n4 - self.N4) / gpe_taun(V4))
        H4_new = self.H4 + dt * (0.05 * (h4 - self.H4) / gpe_tauh(V4))
        R4_new = self.R4 + dt * ((r4 - self.R4) / 30.0)
        CA4_new = self.CA4 + dt * (1.0e-4 * (-Ica4 - It4 - p["kca"][2] * self.CA4))

        v_d2_new = V5 + dt * (-Ina5 - Ik5 - Il5 - Im5 - Igaba5 - Icorstr5 + Iinj5)
        m5_new = self.m5 + dt * (alpham(V5) * (1.0 - self.m5) - betam(V5) * self.m5)
        h5_new = self.h5 + dt * (alphah(V5) * (1.0 - self.h5) - betah(V5) * self.h5)
        n5_new = self.n5 + dt * (alphan(V5) * (1.0 - self.n5) - betan(V5) * self.n5)
        p5_new = self.p5 + dt * (alphap(V5) * (1.0 - self.p5) - betap(V5) * self.p5)
        S1c_new = self.S1c + dt * (Ggaba(V5) * (1.0 - self.S1c) - self.S1c / float(syn["tau_i_striatum"]))

        v_d1_new = V6 + dt * (-Ina6 - Ik6 - Il6 - Im6 - Igaba6 - Icorstr6 + Iinj6)
        m6_new = self.m6 + dt * (alpham(V6) * (1.0 - self.m6) - betam(V6) * self.m6)
        h6_new = self.h6 + dt * (alphah(V6) * (1.0 - self.h6) - betah(V6) * self.h6)
        n6_new = self.n6 + dt * (alphan(V6) * (1.0 - self.n6) - betan(V6) * self.n6)
        p6_new = self.p6 + dt * (alphap(V6) * (1.0 - self.p6) - betap(V6) * self.p6)
        S8_new = self.S8 + dt * (Ggaba(V6) * (1.0 - self.S8) - self.S8 / float(syn["tau_i_striatum"]))

        rs = cfg["ctx_rs"]
        fs = cfg["ctx_fs"]
        v_rs_euler = V7 + dt * (0.04 * V7 ** 2 + 5.0 * V7 + 140.0 - self.u_rs - Iie - Ithcor + Iappco + Iinj7)
        u_rs_euler = self.u_rs + dt * (float(rs["a"]) * (float(rs["b"]) * V7 - self.u_rs))
        v_fs_euler = V8 + dt * (0.04 * V8 ** 2 + 5.0 * V8 + 140.0 - self.u_fs - Iei + Iappco + Iinj8)
        u_fs_euler = self.u_fs + dt * (float(fs["a"]) * (float(fs["b"]) * V8 - self.u_fs))

        rs_reset_hard, fs_reset_hard, spk_rs_reset, spk_fs_reset = self._compute_reset_events(V7, V8)
        v_rs_new = torch.where(rs_reset_hard, torch.zeros_like(V7) + float(rs["c"]), v_rs_euler)
        u_rs_new = torch.where(rs_reset_hard, self.u_rs + float(rs["d"]), u_rs_euler)
        v_fs_new = torch.where(fs_reset_hard, torch.zeros_like(V8) + float(fs["c"]), v_fs_euler)
        u_fs_new = torch.where(fs_reset_hard, self.u_fs + float(fs["d"]), u_fs_euler)

        # ---------------- Spike events ----------------
        (
            spk_th, spk_stn, spk_gpe, spk_gpi, spk_d2, spk_d1, spk_rs_syn, spk_fs_syn,
            ap_th, ap_stn, ap_gpe, ap_gpi, ap_d2, ap_d1, ap_rs, ap_fs,
        ) = self._compute_crossing_events(
            V1, V2, V3, V4, V5, V6, V7, V8,
            v_th_new, v_stn_new, v_gpe_new, v_gpi_new, v_d2_new, v_d1_new, v_rs_new, v_fs_new,
        )

        # ---------------- Delays and synaptic filter updates ----------------
        delayed_pathways, delay_updates = self._delay_pathways(
            spk_th,
            spk_stn,
            spk_gpe,
            spk_gpi,
            spk_d2,
            spk_d1,
            spk_rs_reset,
        )
        (
            d_th_ctx,
            d_stn_gpe,
            d_stn_gpi,
            d_gpe_stn,
            d_gpe_gpi,
            d_gpe_gpe,
            d_gpi_th,
            d_d2_gpe,
            d_d1_gpi,
            d_ctx_d2,
            d_ctx_d1,
            d_ctx_stn,
        ) = delayed_pathways

        (
            S7_new, Z_th_ctx_new,
            S2b_new, Z_stn_gpi_new,
            S3b_new, Z_gpe_gpi_new,
            S3c_new, Z_gpe_gpe_new,
            S4_new, Z_gpi_th_new,
            S5_new, Z_d2_gpe_new,
            S9_new, Z_d1_gpi_new,
            S6a_new, Z_ctx_d2_new, Z_ctx_d1_new,
            S1a_new, Z1a_new,
            S1b_new, Z1b_new,
            A_stn_gpe_a_new, B_stn_gpe_a_new, S2a_new,
            A_stn_gpe_n_new, B_stn_gpe_n_new, S2an_new,
            A_gpe_stn_new, B_gpe_stn_new, S3a_new,
            A_ctx_stn_a_new, B_ctx_stn_a_new, S6b_new,
            A_ctx_stn_n_new, B_ctx_stn_n_new, S6bn_new,
        ) = self._update_synaptic_filters(
            d_th_ctx, d_stn_gpe, d_stn_gpi, d_gpe_stn, d_gpe_gpi, d_gpe_gpe,
            d_gpi_th, d_d2_gpe, d_d1_gpi, d_ctx_d2, d_ctx_d1, d_ctx_stn,
            spk_rs_syn, spk_fs_syn, dt, syn,
        )

        # ---------------- Return accepted-step state ----------------
        outputs = {
            **delay_updates,
            "v_th": v_th_new,
            "v_stn": v_stn_new,
            "v_gpe": v_gpe_new,
            "v_gpi": v_gpi_new,
            "v_d2": v_d2_new,
            "v_d1": v_d1_new,
            "v_rs": v_rs_new,
            "v_fs": v_fs_new,
            "H1": H1_new,
            "R1": R1_new,
            "N2": N2_new,
            "H2": H2_new,
            "M2": M2_new,
            "A2": A2_new,
            "B2": B2_new,
            "C2": C2_new,
            "D1": D1_new,
            "D2": D2_new,
            "P2": P2_new,
            "Q2": Q2_new,
            "R2": R2_new,
            "CAsn2": CAsn2_new,
            "N3": N3_new,
            "H3": H3_new,
            "R3": R3_new,
            "CA3": CA3_new,
            "N4": N4_new,
            "H4": H4_new,
            "R4": R4_new,
            "CA4": CA4_new,
            "m5": m5_new,
            "h5": h5_new,
            "n5": n5_new,
            "p5": p5_new,
            "m6": m6_new,
            "h6": h6_new,
            "n6": n6_new,
            "p6": p6_new,
            "u_rs": u_rs_new,
            "u_fs": u_fs_new,
            "S2a": S2a_new,
            "S2an": S2an_new,
            "S2b": S2b_new,
            "S3a": S3a_new,
            "S3b": S3b_new,
            "S3c": S3c_new,
            "S4": S4_new,
            "S5": S5_new,
            "S6a": S6a_new,
            "S6b": S6b_new,
            "S6bn": S6bn_new,
            "S7": S7_new,
            "S8": S8_new,
            "S9": S9_new,
            "S1a": S1a_new,
            "Z1a": Z1a_new,
            "S1b": S1b_new,
            "Z1b": Z1b_new,
            "S1c": S1c_new,
            "A_stn_gpe_a": A_stn_gpe_a_new,
            "B_stn_gpe_a": B_stn_gpe_a_new,
            "A_stn_gpe_n": A_stn_gpe_n_new,
            "B_stn_gpe_n": B_stn_gpe_n_new,
            "A_gpe_stn": A_gpe_stn_new,
            "B_gpe_stn": B_gpe_stn_new,
            "A_ctx_stn_a": A_ctx_stn_a_new,
            "B_ctx_stn_a": B_ctx_stn_a_new,
            "A_ctx_stn_n": A_ctx_stn_n_new,
            "B_ctx_stn_n": B_ctx_stn_n_new,
            "Z_th_ctx": Z_th_ctx_new,
            "Z_stn_gpi": Z_stn_gpi_new,
            "Z_gpe_gpi": Z_gpe_gpi_new,
            "Z_gpe_gpe": Z_gpe_gpe_new,
            "Z_gpi_th": Z_gpi_th_new,
            "Z_d2_gpe": Z_d2_gpe_new,
            "Z_d1_gpi": Z_d1_gpi_new,
            "Z_ctx_d2": Z_ctx_d2_new,
            "Z_ctx_d1": Z_ctx_d1_new,
            "spikes": self._cat_v(
                spk_th,
                spk_stn,
                spk_gpe,
                spk_gpi,
                spk_d2,
                spk_d1,
                spk_rs_reset,
                spk_fs_reset,
            ),
            "syn_spikes": self._cat_v(
                spk_th,
                spk_stn,
                spk_gpe,
                spk_gpi,
                spk_d2,
                spk_d1,
                spk_rs_syn,
                spk_fs_syn,
            ),
            "ap_spikes": self._cat_v(
                ap_th,
                ap_stn,
                ap_gpe,
                ap_gpi,
                ap_d2,
                ap_d1,
                ap_rs,
                ap_fs,
            ),
        }
        outputs["v_all"] = self._cat_v(
            v_th_new,
            v_stn_new,
            v_gpe_new,
            v_gpi_new,
            v_d2_new,
            v_d1_new,
            v_rs_new,
            v_fs_new,
        )
        return outputs


class _UncoalescedSpikeEventUpdates:
    """Compute reset and crossing events as separate per-population operations."""

    def _compute_reset_events(self, V7, V8):
        rs_peak = float(self.cfg["ctx_rs"].get("v_peak", 30.0))
        fs_peak = float(self.cfg["ctx_fs"].get("v_peak", 30.0))
        rs_reset_hard = V7 >= rs_peak
        fs_reset_hard = V8 >= fs_peak
        spk_rs_reset = self._level_spike(V7, rs_peak)
        spk_fs_reset = self._level_spike(V8, fs_peak)
        return rs_reset_hard, fs_reset_hard, spk_rs_reset, spk_fs_reset

    def _compute_crossing_events(
        self,
        V1, V2, V3, V4, V5, V6, V7, V8,
        v_th_new, v_stn_new, v_gpe_new, v_gpi_new, v_d2_new, v_d1_new, v_rs_new, v_fs_new,
    ):
        spk_th = self._crossing_spike(V1, v_th_new, -10.0)
        spk_stn = self._crossing_spike(V2, v_stn_new, -10.0)
        spk_gpe = self._crossing_spike(V3, v_gpe_new, -10.0)
        spk_gpi = self._crossing_spike(V4, v_gpi_new, -10.0)
        spk_d2 = self._crossing_spike(V5, v_d2_new, -10.0)
        spk_d1 = self._crossing_spike(V6, v_d1_new, -10.0)
        spk_rs_syn = self._crossing_spike(V7, v_rs_new, -10.0)
        spk_fs_syn = self._crossing_spike(V8, v_fs_new, -10.0)

        ap_th = self._crossing_spike(V1, v_th_new, -20.0)
        ap_stn = self._crossing_spike(V2, v_stn_new, -20.0)
        ap_gpe = self._crossing_spike(V3, v_gpe_new, -20.0)
        ap_gpi = self._crossing_spike(V4, v_gpi_new, -20.0)
        ap_d2 = self._crossing_spike(V5, v_d2_new, -20.0)
        ap_d1 = self._crossing_spike(V6, v_d1_new, -20.0)
        ap_rs = self._crossing_spike(V7, v_rs_new, -20.0)
        ap_fs = self._crossing_spike(V8, v_fs_new, -20.0)

        return (
            spk_th, spk_stn, spk_gpe, spk_gpi, spk_d2, spk_d1, spk_rs_syn, spk_fs_syn,
            ap_th, ap_stn, ap_gpe, ap_gpi, ap_d2, ap_d1, ap_rs, ap_fs,
        )


class _CoalescedSpikeEventUpdates:
    """Compute reset and threshold-crossing events with stacked population axes."""

    def _thresholds_for(self, thresholds, stacked):
        # ``stacked`` is shaped (..., n_streams, n).  Thresholds are static
        # length-n_stream tensors registered during initialization.
        return thresholds.reshape(*([1] * (stacked.ndim - 2)), thresholds.numel(), 1)

    def _compute_reset_events(self, V7, V8):
        v_ctx = torch.stack((V7, V8), dim=-2)
        thresholds = self._thresholds_for(self.ctx_reset_thresholds, v_ctx)
        reset_hard = v_ctx >= thresholds
        reset_spikes = self._level_spikes(v_ctx, thresholds)
        rs_reset_hard, fs_reset_hard = reset_hard.unbind(dim=-2)
        spk_rs_reset, spk_fs_reset = reset_spikes.unbind(dim=-2)
        return rs_reset_hard, fs_reset_hard, spk_rs_reset, spk_fs_reset

    def _compute_crossing_events(
        self,
        V1, V2, V3, V4, V5, V6, V7, V8,
        v_th_new, v_stn_new, v_gpe_new, v_gpi_new, v_d2_new, v_d1_new, v_rs_new, v_fs_new,
    ):
        v_old = torch.stack((V1, V2, V3, V4, V5, V6, V7, V8), dim=-2)
        v_new = torch.stack((v_th_new, v_stn_new, v_gpe_new, v_gpi_new, v_d2_new, v_d1_new, v_rs_new, v_fs_new), dim=-2)
        thresholds = self.spike_crossing_thresholds.reshape(
            *([1] * (v_old.ndim - 2)), self.spike_crossing_thresholds.numel(), 1, 1
        )
        spikes_by_threshold = self._crossing_spikes(
            v_old.unsqueeze(-3),
            v_new.unsqueeze(-3),
            thresholds,
        )
        syn_spikes = spikes_by_threshold.select(-3, 0)
        ap_spikes = spikes_by_threshold.select(-3, 1)

        (
            spk_th, spk_stn, spk_gpe, spk_gpi,
            spk_d2, spk_d1, spk_rs_syn, spk_fs_syn,
        ) = syn_spikes.unbind(dim=-2)
        ap_th, ap_stn, ap_gpe, ap_gpi, ap_d2, ap_d1, ap_rs, ap_fs = ap_spikes.unbind(dim=-2)
        return (
            spk_th, spk_stn, spk_gpe, spk_gpi, spk_d2, spk_d1, spk_rs_syn, spk_fs_syn,
            ap_th, ap_stn, ap_gpe, ap_gpi, ap_d2, ap_d1, ap_rs, ap_fs,
        )


class _UncoalescedSynapseUpdates:
    """Update each synaptic filter pathway separately.

    This preserves the original fused implementation's call structure and can be
    faster for some CPU/batch-size regimes where stacking/unbinding overhead is
    not amortized.
    """

    def _update_synaptic_filters(
        self,
        d_th_ctx, d_stn_gpe, d_stn_gpi, d_gpe_stn, d_gpe_gpi, d_gpe_gpe,
        d_gpi_th, d_d2_gpe, d_d1_gpi, d_ctx_d2, d_ctx_d1, d_ctx_stn,
        spk_rs_syn, spk_fs_syn, dt, syn,
    ):
        peak = float(syn["gpeak"])
        peak1 = float(syn["gpeak1"])
        tau_alpha = float(syn["tau_alpha"])

        S7_new, Z_th_ctx_new = self._alpha_step(self.S7, self.Z_th_ctx, d_th_ctx, peak, tau_alpha, dt)
        S2b_new, Z_stn_gpi_new = self._alpha_step(self.S2b, self.Z_stn_gpi, d_stn_gpi, peak, tau_alpha, dt)
        S3b_new, Z_gpe_gpi_new = self._alpha_step(self.S3b, self.Z_gpe_gpi, d_gpe_gpi, peak1, tau_alpha, dt)
        S3c_new, Z_gpe_gpe_new = self._alpha_step(self.S3c, self.Z_gpe_gpe, d_gpe_gpe, peak1, tau_alpha, dt)
        S4_new, Z_gpi_th_new = self._alpha_step(self.S4, self.Z_gpi_th, d_gpi_th, peak1, tau_alpha, dt)
        S5_new, Z_d2_gpe_new = self._alpha_step(self.S5, self.Z_d2_gpe, d_d2_gpe, peak1, tau_alpha, dt)
        S9_new, Z_d1_gpi_new = self._alpha_step(self.S9, self.Z_d1_gpi, d_d1_gpi, peak1, tau_alpha, dt)
        S6a_d2_new, Z_ctx_d2_new = self._alpha_step(self.S6a, self.Z_ctx_d2, d_ctx_d2, peak, tau_alpha, dt)
        # CTX_D1 has identical waveform/input but separate Z state for optional future divergence.
        S6a_d1_new, Z_ctx_d1_new = self._alpha_step(self.S6a, self.Z_ctx_d1, d_ctx_d1, peak, tau_alpha, dt)
        S6a_new = 0.5 * (S6a_d2_new + S6a_d1_new)

        # Local cortical alpha ODEs: no axonal delay, threshold at -10 mV.
        S1a_new, Z1a_new = self._alpha_step(self.S1a, self.Z1a, spk_rs_syn, peak, tau_alpha, dt)
        S1b_new, Z1b_new = self._alpha_step(self.S1b, self.Z1b, spk_fs_syn, peak, tau_alpha, dt)

        A_stn_gpe_a_new, B_stn_gpe_a_new, S2a_new = self._exp2_step(
            self.A_stn_gpe_a, self.B_stn_gpe_a, d_stn_gpe, peak, 0.4, 2.5, dt
        )
        A_stn_gpe_n_new, B_stn_gpe_n_new, S2an_new = self._exp2_step(
            self.A_stn_gpe_n, self.B_stn_gpe_n, d_stn_gpe, peak, 2.0, 67.0, dt
        )
        A_gpe_stn_new, B_gpe_stn_new, S3a_new = self._exp2_step(
            self.A_gpe_stn, self.B_gpe_stn, d_gpe_stn, peak1, 0.4, 7.7, dt
        )
        A_ctx_stn_a_new, B_ctx_stn_a_new, S6b_new = self._exp2_step(
            self.A_ctx_stn_a, self.B_ctx_stn_a, d_ctx_stn, peak, 0.5, 2.49, dt
        )
        A_ctx_stn_n_new, B_ctx_stn_n_new, S6bn_new = self._exp2_step(
            self.A_ctx_stn_n, self.B_ctx_stn_n, d_ctx_stn, peak, 2.0, 90.0, dt
        )

        return (
            S7_new, Z_th_ctx_new,
            S2b_new, Z_stn_gpi_new,
            S3b_new, Z_gpe_gpi_new,
            S3c_new, Z_gpe_gpe_new,
            S4_new, Z_gpi_th_new,
            S5_new, Z_d2_gpe_new,
            S9_new, Z_d1_gpi_new,
            S6a_new, Z_ctx_d2_new, Z_ctx_d1_new,
            S1a_new, Z1a_new,
            S1b_new, Z1b_new,
            A_stn_gpe_a_new, B_stn_gpe_a_new, S2a_new,
            A_stn_gpe_n_new, B_stn_gpe_n_new, S2an_new,
            A_gpe_stn_new, B_gpe_stn_new, S3a_new,
            A_ctx_stn_a_new, B_ctx_stn_a_new, S6b_new,
            A_ctx_stn_n_new, B_ctx_stn_n_new, S6bn_new,
        )


class _CoalescedSynapseUpdates:
    """Update alpha and bi-exponential filters as pathway batches.

    This reduces Python/torch dispatch overhead and often reduces GPU kernel
    launch count by stacking the same filter family into stream dimensions.
    The state layout is still committed back to the original named buffers so
    recording and validation code remains unchanged.
    """

    def _update_synaptic_filters(
        self,
        d_th_ctx, d_stn_gpe, d_stn_gpi, d_gpe_stn, d_gpe_gpi, d_gpe_gpe,
        d_gpi_th, d_d2_gpe, d_d1_gpi, d_ctx_d2, d_ctx_d1, d_ctx_stn,
        spk_rs_syn, spk_fs_syn, dt, syn,
    ):
        # Alpha stream order:
        #   S7, S2b, S3b, S3c, S4, S5, S9, S6a(ctx_d2),
        #   S6a(ctx_d1), S1a, S1b.
        s_alpha = torch.stack(
            (
                self.S7, self.S2b, self.S3b, self.S3c, self.S4, self.S5, self.S9,
                self.S6a, self.S6a, self.S1a, self.S1b,
            ),
            dim=-2,
        )
        z_alpha = torch.stack(
            (
                self.Z_th_ctx, self.Z_stn_gpi, self.Z_gpe_gpi, self.Z_gpe_gpe,
                self.Z_gpi_th, self.Z_d2_gpe, self.Z_d1_gpi, self.Z_ctx_d2,
                self.Z_ctx_d1, self.Z1a, self.Z1b,
            ),
            dim=-2,
        )
        u_alpha = torch.stack(
            (
                d_th_ctx, d_stn_gpi, d_gpe_gpi, d_gpe_gpe, d_gpi_th, d_d2_gpe,
                d_d1_gpi, d_ctx_d2, d_ctx_d1, spk_rs_syn, spk_fs_syn,
            ),
            dim=-2,
        )
        s_alpha_new, z_alpha_new = self._alpha_steps(s_alpha, z_alpha, u_alpha, dt)
        (
            S7_new, S2b_new, S3b_new, S3c_new, S4_new, S5_new, S9_new,
            S6a_d2_new, S6a_d1_new, S1a_new, S1b_new,
        ) = s_alpha_new.unbind(dim=-2)
        (
            Z_th_ctx_new, Z_stn_gpi_new, Z_gpe_gpi_new, Z_gpe_gpe_new,
            Z_gpi_th_new, Z_d2_gpe_new, Z_d1_gpi_new, Z_ctx_d2_new,
            Z_ctx_d1_new, Z1a_new, Z1b_new,
        ) = z_alpha_new.unbind(dim=-2)
        S6a_new = 0.5 * (S6a_d2_new + S6a_d1_new)

        # Exp2 stream order:
        #   STN->GPe AMPA, STN->GPe NMDA, GPe->STN,
        #   CTX->STN AMPA, CTX->STN NMDA.
        a_exp2 = torch.stack(
            (
                self.A_stn_gpe_a, self.A_stn_gpe_n, self.A_gpe_stn,
                self.A_ctx_stn_a, self.A_ctx_stn_n,
            ),
            dim=-2,
        )
        b_exp2 = torch.stack(
            (
                self.B_stn_gpe_a, self.B_stn_gpe_n, self.B_gpe_stn,
                self.B_ctx_stn_a, self.B_ctx_stn_n,
            ),
            dim=-2,
        )
        u_exp2 = torch.stack(
            (d_stn_gpe, d_stn_gpe, d_gpe_stn, d_ctx_stn, d_ctx_stn),
            dim=-2,
        )
        a_exp2_new, b_exp2_new, s_exp2_new = self._exp2_steps(a_exp2, b_exp2, u_exp2, dt)
        (
            A_stn_gpe_a_new, A_stn_gpe_n_new, A_gpe_stn_new,
            A_ctx_stn_a_new, A_ctx_stn_n_new,
        ) = a_exp2_new.unbind(dim=-2)
        (
            B_stn_gpe_a_new, B_stn_gpe_n_new, B_gpe_stn_new,
            B_ctx_stn_a_new, B_ctx_stn_n_new,
        ) = b_exp2_new.unbind(dim=-2)
        S2a_new, S2an_new, S3a_new, S6b_new, S6bn_new = s_exp2_new.unbind(dim=-2)

        return (
            S7_new, Z_th_ctx_new,
            S2b_new, Z_stn_gpi_new,
            S3b_new, Z_gpe_gpi_new,
            S3c_new, Z_gpe_gpe_new,
            S4_new, Z_gpi_th_new,
            S5_new, Z_d2_gpe_new,
            S9_new, Z_d1_gpi_new,
            S6a_new, Z_ctx_d2_new, Z_ctx_d1_new,
            S1a_new, Z1a_new,
            S1b_new, Z1b_new,
            A_stn_gpe_a_new, B_stn_gpe_a_new, S2a_new,
            A_stn_gpe_n_new, B_stn_gpe_n_new, S2an_new,
            A_gpe_stn_new, B_gpe_stn_new, S3a_new,
            A_ctx_stn_a_new, B_ctx_stn_a_new, S6b_new,
            A_ctx_stn_n_new, B_ctx_stn_n_new, S6bn_new,
        )


class _EulerSynapseDiscretization:
    """Original explicit-Euler recursive synaptic filter update."""

    def _alpha_step(self, s, z, spike, peak: float, tau: float, dt):
        const = peak / (tau * math.exp(-1.0))
        s_new = s + dt * z
        z_new = z + const * spike - dt * ((2.0 / tau) * z + (1.0 / (tau * tau)) * s_new)
        return s_new, z_new

    def _alpha_steps(self, s, z, spike, dt):
        tau = float(self.cfg["syn"].get("tau_alpha", 5.0))
        s_new = s + dt * z
        z_new = z + self.alpha_const_streams * spike - dt * ((2.0 / tau) * z + (1.0 / (tau * tau)) * s_new)
        return s_new, z_new

    def _exp2_step(self, a, b, spike, peak: float, tau1: float, tau2: float, dt):
        tp = (tau1 * tau2) / (tau2 - tau1) * math.log(tau2 / tau1)
        factor = 1.0 / (-math.exp(-tp / tau1) + math.exp(-tp / tau2))
        inc = peak * factor * spike
        a_new = a + inc - dt * (a / tau1)
        b_new = b + inc - dt * (b / tau2)
        return a_new, b_new, b_new - a_new

    def _exp2_steps(self, a, b, spike, dt):
        inc = self.exp2_inc_streams * spike
        a_new = a * self.exp2_euler_decay1_streams + inc
        b_new = b * self.exp2_euler_decay2_streams + inc
        return a_new, b_new, b_new - a_new


class _BackwardEulerSynapseDiscretization:
    """Backward-Euler homogeneous decay with MATLAB-like end-of-step event injection."""

    def _alpha_step(self, s, z, spike, peak: float, tau: float, dt):
        const = peak / (tau * math.exp(-1.0))
        h = dt / tau
        denom = (1.0 + h) * (1.0 + h)
        z_h = (z - (dt / (tau * tau)) * s) / denom
        s_h = s + dt * z_h
        z_new = z_h + const * spike
        return s_h, z_new

    def _alpha_steps(self, s, z, spike, dt):
        tau = float(self.cfg["syn"].get("tau_alpha", 5.0))
        h = dt / tau
        denom = (1.0 + h) * (1.0 + h)
        z_h = (z - (dt / (tau * tau)) * s) / denom
        s_h = s + dt * z_h
        z_new = z_h + self.alpha_const_streams * spike
        return s_h, z_new

    def _exp2_step(self, a, b, spike, peak: float, tau1: float, tau2: float, dt):
        tp = (tau1 * tau2) / (tau2 - tau1) * math.log(tau2 / tau1)
        factor = 1.0 / (-math.exp(-tp / tau1) + math.exp(-tp / tau2))
        inc = peak * factor * spike
        a_new = a / (1.0 + dt / tau1) + inc
        b_new = b / (1.0 + dt / tau2) + inc
        return a_new, b_new, b_new - a_new

    def _exp2_steps(self, a, b, spike, dt):
        inc = self.exp2_inc_streams * spike
        a_new = a * self.exp2_be_decay1_streams + inc
        b_new = b * self.exp2_be_decay2_streams + inc
        return a_new, b_new, b_new - a_new


class _ExactSynapseDiscretization:
    """Exact homogeneous transition for linear alpha/bi-exponential filters.

    All dt-dependent coefficients are prepared once per configured timestep. This keeps
    the timestep hot path free of scalar ``torch.exp`` calls.
    """

    def _alpha_step(self, s, z, spike, peak: float, tau: float, dt):
        peak1 = float(self.cfg["syn"].get("gpeak1", 0.3))
        const = self._alpha_const_peak1 if peak == peak1 else self._alpha_const_peak
        s_h = self._alpha_decay * ((1.0 + self._alpha_h) * s + self._alpha_dt * z)
        z_h = self._alpha_decay * (-self._alpha_dt_over_tau2 * s + (1.0 - self._alpha_h) * z)
        z_new = z_h + const * spike
        return s_h, z_new

    def _alpha_steps(self, s, z, spike, dt):
        s_h = self._alpha_decay * ((1.0 + self._alpha_h) * s + self._alpha_dt * z)
        z_h = self._alpha_decay * (-self._alpha_dt_over_tau2 * s + (1.0 - self._alpha_h) * z)
        z_new = z_h + self.alpha_const_streams * spike
        return s_h, z_new

    def _exp2_step(self, a, b, spike, peak: float, tau1: float, tau2: float, dt):
        del peak, dt
        decay1_name, decay2_name, inc_name = _EXP2_SCALAR_COEFFICIENTS[(tau1, tau2)]
        decay1 = getattr(self, decay1_name)
        decay2 = getattr(self, decay2_name)
        inc_factor = getattr(self, inc_name)
        inc = inc_factor * spike
        a_new = a * decay1 + inc
        b_new = b * decay2 + inc
        return a_new, b_new, b_new - a_new

    def _exp2_steps(self, a, b, spike, dt):
        inc = self.exp2_inc_streams * spike
        a_new = a * self.exp2_exact_decay1_streams + inc
        b_new = b * self.exp2_exact_decay2_streams + inc
        return a_new, b_new, b_new - a_new


class _Kumaravelu2016FusedSurrogateSpikes(_Kumaravelu2016FusedBase):
    """Fused model variant with Dendra STE spike surrogates."""


class _Kumaravelu2016FusedHardSpikes(_Kumaravelu2016FusedBase):
    """Fused model variant with hard, non-differentiable spike events.

    This is the fast forward-simulation path.  It preserves the same hard
    threshold/reset semantics used by the surrogate helper's forward pass, but
    avoids all sigmoid/ReLU surrogate work when gradients through spike events
    are not requested.
    """

    def _crossing_spike(self, v_old, v_new, threshold: float):
        return ((v_old < threshold) & (v_new > threshold)).to(v_new.dtype)

    def _level_spike(self, v, threshold: float):
        return (v >= threshold).to(v.dtype)

    def _crossing_spikes(self, v_old, v_new, thresholds):
        return ((v_old < thresholds) & (v_new > thresholds)).to(v_new.dtype)

    def _level_spikes(self, v, thresholds):
        return (v >= thresholds).to(v.dtype)


def make_kumaravelu_2016_fused(config: Mapping[str, Any]):
    """Return a configured fused VoltageProcess class.

    The Dendra mechanism constructor validates keyword parameters fairly
    strictly.  Binding the realization and network constants as class-level
    configuration avoids passing large integer permutation arrays through the
    parameter system while still keeping the runtime state in Dendra buffers.
    """

    cfg = copy.deepcopy(dict(config))
    cfg.setdefault("differentiable_spikes", True)
    cfg.setdefault("synapse_discretization", "euler")
    cfg.setdefault("synapse_update_mode", "uncoalesced")
    cfg.setdefault("spike_update_mode", "uncoalesced")
    cfg.setdefault("delay_mode", "auto")

    spike_cls = (
        _Kumaravelu2016FusedSurrogateSpikes
        if bool(cfg.get("differentiable_spikes", True))
        else _Kumaravelu2016FusedHardSpikes
    )

    mode = str(cfg.get("synapse_discretization", "euler")).lower()
    synapse_cls_by_mode = {
        "euler": _EulerSynapseDiscretization,
        "explicit_euler": _EulerSynapseDiscretization,
        "backward_euler": _BackwardEulerSynapseDiscretization,
        "implicit": _BackwardEulerSynapseDiscretization,
        "exact": _ExactSynapseDiscretization,
    }
    if mode not in synapse_cls_by_mode:
        raise ValueError(
            "synapse_discretization must be one of "
            "'euler', 'backward_euler', or 'exact'; "
            f"got {mode!r}."
        )
    cfg["synapse_discretization"] = mode
    synapse_cls = synapse_cls_by_mode[mode]

    update_mode = str(cfg.get("synapse_update_mode", "uncoalesced")).lower()
    update_aliases = {
        "separate": "uncoalesced",
        "individual": "uncoalesced",
        "batched": "coalesced",
        "batch": "coalesced",
        "stacked": "coalesced",
    }
    update_mode = update_aliases.get(update_mode, update_mode)
    update_cls_by_mode = {
        "uncoalesced": _UncoalescedSynapseUpdates,
        "coalesced": _CoalescedSynapseUpdates,
    }
    if update_mode not in update_cls_by_mode:
        raise ValueError(
            "synapse_update_mode must be one of 'uncoalesced' or 'coalesced'; "
            f"got {update_mode!r}."
        )
    cfg["synapse_update_mode"] = update_mode
    update_cls = update_cls_by_mode[update_mode]

    spike_update_mode = str(cfg.get("spike_update_mode", "uncoalesced")).lower()
    spike_update_aliases = {
        "separate": "uncoalesced",
        "individual": "uncoalesced",
        "batched": "coalesced",
        "batch": "coalesced",
        "stacked": "coalesced",
    }
    spike_update_mode = spike_update_aliases.get(spike_update_mode, spike_update_mode)
    spike_update_cls_by_mode = {
        "uncoalesced": _UncoalescedSpikeEventUpdates,
        "coalesced": _CoalescedSpikeEventUpdates,
    }
    if spike_update_mode not in spike_update_cls_by_mode:
        raise ValueError(
            "spike_update_mode must be one of 'uncoalesced' or 'coalesced'; "
            f"got {spike_update_mode!r}."
        )
    cfg["spike_update_mode"] = spike_update_mode
    spike_update_cls = spike_update_cls_by_mode[spike_update_mode]

    delay_mode = str(cfg.get("delay_mode", "auto")).lower()
    delay_aliases = {
        "eval_circular": "auto",
        "fast_eval": "auto",
        "functional": "shift",
        "queue": "shift",
        "shift_queue": "shift",
    }
    delay_mode = delay_aliases.get(delay_mode, delay_mode)
    if delay_mode not in {"auto", "shift", "circular", "circular_eager"}:
        raise ValueError(
            "delay_mode must be one of 'auto', 'shift', 'circular', or 'circular_eager'; "
            f"got {delay_mode!r}."
        )
    cfg["delay_mode"] = delay_mode

    class kumaravelu_2016_fused(spike_update_cls, update_cls, synapse_cls, spike_cls):
        CONFIG = cfg
        # Keep Dendra's rename cache local to this generated class.  The default
        # cache lives on the base mechanism class and is keyed only by alias,
        # which is unsafe for generated classes whose CONFIG differs by n, seed,
        # connectivity realization, DBS settings, etc.
        _renamed_aliases = {}

    # Population.insert currently uses mechanism.__name__ for mechanisms inserted
    # everywhere, so give the generated class the user-facing alias directly
    # instead of going through Mechanism.rename().
    kumaravelu_2016_fused.__name__ = "kumaravelu"
    kumaravelu_2016_fused.__qualname__ = "kumaravelu"
    return kumaravelu_2016_fused
