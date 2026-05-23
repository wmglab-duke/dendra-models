"""Fused fast-path mechanism for the Kumaravelu et al. CTX-BG-TH model.

This module implements the entire single-compartment network as one Dendra
``VoltageProcess``.  It deliberately bypasses ``NetCon`` and current-map
bookkeeping; all intrinsic currents, synaptic filters, fixed delays, routing
permutations, cortical Izhikevich reset logic, and spike surrogates are updated
inside a single PyTorch tensor program.

The equations follow the MATLAB reference implementation, but spike-triggered
lookup-table synapses are represented as equivalent alpha or bi-exponential
filter states with fixed delay queues.  The filter discretization is selectable
(``euler``, ``backward_euler``, or ``exact``).  Fixed delays use Dendra's
batched mechanism-level delayed-state helper; ``delay_mode="auto"`` uses fast
circular buffers in eval/no-grad mode and graph-safe shifted queues in training
mode.  The local cortical alpha synapses use the same second-order alpha ODE
form used in the MATLAB ``S1a/Z1a`` and ``S1b/Z1b`` updates.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, Iterable, Mapping

import torch

from dendra.models.networks.spiking import (
    crossing_spike as _dendra_crossing_spike,
    level_spike as _dendra_level_spike,
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


def _roll(x, shift: int):
    return torch.roll(x, shifts=int(shift), dims=-1)


def _sum_rolls(x, shifts: Iterable[int]):
    out = torch.zeros_like(x)
    for shift in shifts:
        out = out + _roll(x, int(shift))
    return out


def _gather_perm(x, perm):
    """Gather ``x`` along the neuron axis with optional per-network permutations.

    ``perm`` may be a single length-n permutation shared by every independent
    network, or a tensor with the same leading dimensions as ``x`` and length n
    along the last axis.  The latter is required when ``Population(N, 8*n)``
    carries N independently randomized networks.
    """
    perm = perm.to(device=x.device, dtype=torch.long)
    while perm.ndim < x.ndim:
        perm = perm.unsqueeze(0)
    if perm.shape != x.shape:
        perm = perm.expand_as(x)
    return torch.gather(x, dim=-1, index=perm)


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


# ---------------------------------------------------------------------------
# Fused mechanism
# ---------------------------------------------------------------------------


class _Kumaravelu2016FusedBase(V):
    """Base fused mechanism; use ``make_kumaravelu_2016_fused`` to bind config."""

    CONFIG: Dict[str, Any] = {}

    # All mutable simulation state is assigned at the mechanism level so Dendra's
    # checkpointing path can save/restore it through MechanismHandler.mutable_state_dict().
    V.ASSIGNED(
        # exposed voltage/state summaries
        "v_all", "spikes", "syn_spikes", "ap_spikes", "i_inj",
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
        # Coalesced pathway-delay group used by register_delayed_states(...).
        "buf_pathway_delays", "buf_pathway_delays_ptr",
        # Optional per-pathway delay queues used by delay_mode="circular_eager".
        "buf_th_ctx", "buf_stn_gpe", "buf_stn_gpi", "buf_gpe_stn", "buf_gpe_gpi", "buf_gpe_gpe",
        "buf_gpi_th", "buf_d2_gpe", "buf_d1_gpi", "buf_ctx_d2", "buf_ctx_d1", "buf_ctx_stn",
        # Realization buffers
        "gcorsna", "gcorsnn", "gcordrstr", "ggege", "gsngen", "gsngea", "gsngi",
        "perm_d2_0", "perm_d2_1", "perm_d2_2", "perm_d2_3",
        "perm_d1_0", "perm_d1_1", "perm_d1_2",
        "perm_fsrs_0", "perm_fsrs_1", "perm_fsrs_2", "perm_fsrs_3",
        "perm_rsfs_0", "perm_rsfs_1", "perm_rsfs_2", "perm_rsfs_3",
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

    def _delay_mode(self) -> str:
        return str(self.cfg.get("delay_mode", "auto")).lower()

    def _delay_register_mode(self) -> str:
        """Return the backend used when registering delay buffers."""
        mode = self._delay_mode()
        # ``circular_eager`` is an execution policy: update delay buffers in one
        # torch._dynamo-disabled island while keeping the rest of the fused step
        # compiled.  The underlying per-pathway buffers are ordinary circular
        # delayed states.
        return "circular" if mode == "circular_eager" else mode

    def _new_pathway_delay_buffer(self, ref):
        """Register one coalesced delay buffer for all delayed pathways."""
        like = ref.unsqueeze(-2).expand(
            *ref.shape[:-1], len(_DELAY_PATHWAYS), ref.shape[-1]
        )
        delay_steps = [
            int(self.cfg["delay_steps"].get(name, 0)) for name in _DELAY_PATHWAYS
        ]
        return self.register_delayed_states(
            "pathway_delays",
            like,
            delay_steps,
            mode=self._delay_register_mode(),
            buffer_name="buf_pathway_delays",
            pointer_name="buf_pathway_delays_ptr",
            stream_axis=-2,
            clear=True,
        )

    def _new_delay_buffer(self, name: str, ref):
        """Register a single per-pathway circular delay buffer.

        This is used only by ``delay_mode='circular_eager'``.  For large CPU
        batches the per-pathway layout ``(..., depth, n)`` is often faster than
        a coalesced ``(..., depth, n_pathways, n)`` gather because each pathway
        reads/writes a contiguous neuron slice and avoids a cross-stream gather.
        """
        steps = int(self.cfg["delay_steps"].get(name, 0))
        return self.register_delayed_state(
            name,
            ref,
            steps,
            mode="circular",
            buffer_name=f"buf_{name}",
            pointer_name=f"buf_{name}_ptr",
            insert_axis=-2,
            clear=True,
        )

    def _init_delay_buffers(self, ref):
        """Initialize the delay backend selected by ``delay_mode``."""
        mode = self._delay_mode()
        if mode == "circular_eager":
            # Keep these assigned buffers small and unused in this backend.
            self.buf_pathway_delays = torch.empty(0, device=ref.device, dtype=ref.dtype)
            self.buf_pathway_delays_ptr = torch.zeros((), device=ref.device, dtype=torch.long)
            for name in _DELAY_PATHWAYS:
                setattr(self, f"buf_{name}", self._new_delay_buffer(name, ref))
        else:
            self.buf_pathway_delays = self._new_pathway_delay_buffer(ref)

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
        """Delay all fixed-delay pathway spike streams with one batched call."""
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
        return tuple(self.delayed_states("pathway_delays", values).unbind(dim=-2))

    def _delay_one_circular_eager(self, name: str, spike):
        steps = int(self.cfg["delay_steps"].get(name, 0))
        if steps <= 0:
            return spike
        return self._delayed_state_circular(self._delayed_state_specs[name], spike, steps)

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
        """Update all per-pathway circular delay lines in one eager island."""
        return (
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

        The waveform is evaluated each timestep in :meth:`_advance` and exposed
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

    def initial(self, v):
        cfg = self.cfg
        n = self._n()
        if v.shape[-1] != 8 * n:
            raise ValueError(
                f"Kumaravelu fused mechanism expected last voltage dimension 8*n={8*n}; got {v.shape[-1]}."
            )

        # Voltages from the Population v_init vector.
        self.v_th = self._group(v, 0).clone()
        self.v_stn = self._group(v, 1).clone()
        self.v_gpe = self._group(v, 2).clone()
        self.v_gpi = self._group(v, 3).clone()
        self.v_d2 = self._group(v, 4).clone()
        self.v_d1 = self._group(v, 5).clone()
        self.v_rs = self._group(v, 6).clone()
        self.v_fs = self._group(v, 7).clone()

        z = torch.zeros_like(self.v_th)
        one = torch.ones_like(self.v_th)

        # Intrinsic states at steady-state/in MATLAB initial values.
        self.H1 = th_hinf(self.v_th)
        self.R1 = th_rinf(self.v_th)

        self.N2 = stn_ninf(self.v_stn)
        self.H2 = stn_hinf(self.v_stn)
        self.M2 = stn_minf(self.v_stn)
        self.A2 = stn_ainf(self.v_stn)
        self.B2 = stn_binf(self.v_stn)
        self.C2 = stn_cinf(self.v_stn)
        self.D2 = stn_d2inf(self.v_stn)
        self.D1 = stn_d1inf(self.v_stn)
        self.P2 = stn_pinf(self.v_stn)
        self.Q2 = stn_qinf(self.v_stn)
        self.R2 = stn_rinf(self.v_stn)
        self.CAsn2 = 0.005 * one

        self.N3 = gpe_ninf(self.v_gpe)
        self.H3 = gpe_hinf(self.v_gpe)
        self.R3 = gpe_rinf(self.v_gpe)
        self.CA3 = 0.1 * one
        self.N4 = gpe_ninf(self.v_gpi)
        self.H4 = gpe_hinf(self.v_gpi)
        self.R4 = gpe_rinf(self.v_gpi)
        self.CA4 = 0.1 * one

        am5, ah5, an5, ap5 = alpham(self.v_d2), alphah(self.v_d2), alphan(self.v_d2), alphap(self.v_d2)
        bm5, bh5, bn5, bp5 = betam(self.v_d2), betah(self.v_d2), betan(self.v_d2), betap(self.v_d2)
        self.m5 = am5 / (am5 + bm5)
        self.h5 = ah5 / (ah5 + bh5)
        self.n5 = an5 / (an5 + bn5)
        self.p5 = ap5 / (ap5 + bp5)

        am6, ah6, an6, ap6 = alpham(self.v_d1), alphah(self.v_d1), alphan(self.v_d1), alphap(self.v_d1)
        bm6, bh6, bn6, bp6 = betam(self.v_d1), betah(self.v_d1), betan(self.v_d1), betap(self.v_d1)
        self.m6 = am6 / (am6 + bm6)
        self.h6 = ah6 / (ah6 + bh6)
        self.n6 = an6 / (an6 + bn6)
        self.p6 = ap6 / (ap6 + bp6)

        # Cortical recovery variables.
        self.u_rs = float(cfg["ctx_rs"]["b"]) * self.v_rs
        self.u_fs = float(cfg["ctx_fs"]["b"]) * self.v_fs

        # Synaptic filter states.
        for name in (
            "S2a", "S2an", "S2b", "S3a", "S3b", "S3c", "S4", "S5", "S6a", "S6b", "S6bn",
            "S7", "S8", "S9", "S1a", "Z1a", "S1b", "Z1b", "S1c",
            "A_stn_gpe_a", "B_stn_gpe_a", "A_stn_gpe_n", "B_stn_gpe_n",
            "A_gpe_stn", "B_gpe_stn", "A_ctx_stn_a", "B_ctx_stn_a", "A_ctx_stn_n", "B_ctx_stn_n",
            "Z_th_ctx", "Z_stn_gpi", "Z_gpe_gpi", "Z_gpe_gpe", "Z_gpi_th",
            "Z_d2_gpe", "Z_d1_gpi", "Z_ctx_d2", "Z_ctx_d1",
        ):
            setattr(self, name, z.clone())

        # Fixed-delay queues.
        self._init_delay_buffers(self.v_th)

        # Realization arrays.
        r = cfg["realization"]
        self.gcorsna = self._as_vector(r["gcorsna"], self.v_th)
        self.gcorsnn = self._as_vector(r["gcorsnn"], self.v_th)
        self.gcordrstr = self._as_vector(r["gcordrstr"], self.v_th)
        self.ggege = self._as_vector(r["ggege"], self.v_th)
        self.gsngen = self._as_vector(r["gsngen"], self.v_th)
        self.gsngea = self._as_vector(r["gsngea"], self.v_th)
        self.gsngi = self._as_vector(r["gsngi"], self.v_th)

        for k in range(4):
            setattr(self, f"perm_d2_{k}", self._as_index(r["str_d2_perms"][k], self.v_th))
            setattr(self, f"perm_fsrs_{k}", self._as_index(r["fs_to_rs_perms"][k], self.v_th))
            setattr(self, f"perm_rsfs_{k}", self._as_index(r["rs_to_fs_perms"][k], self.v_th))
        for k in range(3):
            setattr(self, f"perm_d1_{k}", self._as_index(r["str_d1_perms"][k], self.v_th))

        self.spikes = torch.zeros_like(v)
        self.syn_spikes = torch.zeros_like(v)
        self.ap_spikes = torch.zeros_like(v)
        self.i_inj = torch.zeros_like(v)
        self.v_all = self._cat_v(
            self.v_th, self.v_stn, self.v_gpe, self.v_gpi, self.v_d2, self.v_d1, self.v_rs, self.v_fs
        )

    def update_v(self, v):
        # scnv calls update_v() before _advance(); return the current exposed
        # voltage vector. _advance() will update self.v_all for the next call.
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

    # ------------------------------------------------------------------
    # Fused explicit Euler step
    # ------------------------------------------------------------------

    def _advance(self, v, dt):  # noqa: C901, PLR0915 - intentionally fused
        cfg = self.cfg
        p = cfg["constants"]
        c = cfg["coupling"]
        syn = cfg["syn"]
        n = self._n()
        dt = dt.to(dtype=self.v_th.dtype, device=self.v_th.device) if torch.is_tensor(dt) else torch.as_tensor(dt, device=self.v_th.device, dtype=self.v_th.dtype)

        # Old voltages/states.
        V1, V2, V3, V4 = self.v_th, self.v_stn, self.v_gpe, self.v_gpi
        V5, V6, V7, V8 = self.v_d2, self.v_d1, self.v_rs, self.v_fs

        # Optional user-supplied waveform stimulation.  The exposed vector uses
        # the same layout as v_all: TH, STN, GPe, GPi, StrD2, StrD1, CTX_RS, CTX_FS.
        Iinj_all = self.evaluate_injections(self.v_all, current_name="i_inj")
        Iinj1 = self._group(Iinj_all, 0)
        Iinj2 = self._group(Iinj_all, 1)
        Iinj3 = self._group(Iinj_all, 2)
        Iinj4 = self._group(Iinj_all, 3)
        Iinj5 = self._group(Iinj_all, 4)
        Iinj6 = self._group(Iinj_all, 5)
        Iinj7 = self._group(Iinj_all, 6)
        Iinj8 = self._group(Iinj_all, 7)

        # Routing aliases using current synaptic conductance states.
        S21a = _roll(self.S2a, +1)
        S21an = _roll(self.S2an, +1)
        S21b = _roll(self.S2b, +1)
        S31a = _roll(self.S3a, -1)
        S31b = _roll(self.S3b, -1)
        S31c = _roll(self.S3c, -1)
        S32b = _roll(self.S3b, +2)
        S32c = _roll(self.S3c, +2)
        S61b = _roll(self.S6b, -1)
        S61bn = _roll(self.S6bn, -1)
        S5sum = _sum_rolls(self.S5, range(0, min(10, n)))
        S9sum = _sum_rolls(self.S9, range(0, min(10, n)))

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
        rs_reset_hard = V7 >= float(rs["v_peak"])
        spk_rs_reset = self._level_spike(V7, float(rs["v_peak"]))
        v_rs_new = torch.where(rs_reset_hard, torch.zeros_like(V7) + float(rs["c"]), v_rs_euler)
        u_rs_new = torch.where(rs_reset_hard, self.u_rs + float(rs["d"]), u_rs_euler)

        v_fs_euler = V8 + dt * (0.04 * V8 ** 2 + 5.0 * V8 + 140.0 - self.u_fs - Iei + Iappco + Iinj8)
        u_fs_euler = self.u_fs + dt * (float(fs["a"]) * (float(fs["b"]) * V8 - self.u_fs))
        fs_reset_hard = V8 >= float(fs["v_peak"])
        spk_fs_reset = self._level_spike(V8, float(fs["v_peak"]))
        v_fs_new = torch.where(fs_reset_hard, torch.zeros_like(V8) + float(fs["c"]), v_fs_euler)
        u_fs_new = torch.where(fs_reset_hard, self.u_fs + float(fs["d"]), u_fs_euler)

        # ---------------- Spike events ----------------
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

        # ---------------- Delays and synaptic filter updates ----------------
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
        ) = self._delay_pathways(
            spk_th,
            spk_stn,
            spk_gpe,
            spk_gpi,
            spk_d2,
            spk_d1,
            spk_rs_reset,
        )

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

        # ---------------- Commit state updates ----------------
        self.v_th, self.v_stn, self.v_gpe, self.v_gpi = v_th_new, v_stn_new, v_gpe_new, v_gpi_new
        self.v_d2, self.v_d1, self.v_rs, self.v_fs = v_d2_new, v_d1_new, v_rs_new, v_fs_new

        self.H1, self.R1 = H1_new, R1_new
        self.N2, self.H2, self.M2, self.A2, self.B2, self.C2 = N2_new, H2_new, M2_new, A2_new, B2_new, C2_new
        self.D1, self.D2, self.P2, self.Q2, self.R2, self.CAsn2 = D1_new, D2_new, P2_new, Q2_new, R2_new, CAsn2_new
        self.N3, self.H3, self.R3, self.CA3 = N3_new, H3_new, R3_new, CA3_new
        self.N4, self.H4, self.R4, self.CA4 = N4_new, H4_new, R4_new, CA4_new
        self.m5, self.h5, self.n5, self.p5, self.S1c = m5_new, h5_new, n5_new, p5_new, S1c_new
        self.m6, self.h6, self.n6, self.p6, self.S8 = m6_new, h6_new, n6_new, p6_new, S8_new
        self.u_rs, self.u_fs = u_rs_new, u_fs_new

        self.S7, self.Z_th_ctx = S7_new, Z_th_ctx_new
        self.S2b, self.Z_stn_gpi = S2b_new, Z_stn_gpi_new
        self.S3b, self.Z_gpe_gpi = S3b_new, Z_gpe_gpi_new
        self.S3c, self.Z_gpe_gpe = S3c_new, Z_gpe_gpe_new
        self.S4, self.Z_gpi_th = S4_new, Z_gpi_th_new
        self.S5, self.Z_d2_gpe = S5_new, Z_d2_gpe_new
        self.S9, self.Z_d1_gpi = S9_new, Z_d1_gpi_new
        self.S6a, self.Z_ctx_d2, self.Z_ctx_d1 = S6a_new, Z_ctx_d2_new, Z_ctx_d1_new
        self.S1a, self.Z1a, self.S1b, self.Z1b = S1a_new, Z1a_new, S1b_new, Z1b_new

        self.A_stn_gpe_a, self.B_stn_gpe_a, self.S2a = A_stn_gpe_a_new, B_stn_gpe_a_new, S2a_new
        self.A_stn_gpe_n, self.B_stn_gpe_n, self.S2an = A_stn_gpe_n_new, B_stn_gpe_n_new, S2an_new
        self.A_gpe_stn, self.B_gpe_stn, self.S3a = A_gpe_stn_new, B_gpe_stn_new, S3a_new
        self.A_ctx_stn_a, self.B_ctx_stn_a, self.S6b = A_ctx_stn_a_new, B_ctx_stn_a_new, S6b_new
        self.A_ctx_stn_n, self.B_ctx_stn_n, self.S6bn = A_ctx_stn_n_new, B_ctx_stn_n_new, S6bn_new

        self.spikes = self._cat_v(spk_th, spk_stn, spk_gpe, spk_gpi, spk_d2, spk_d1, spk_rs_reset, spk_fs_reset)
        self.syn_spikes = self._cat_v(spk_th, spk_stn, spk_gpe, spk_gpi, spk_d2, spk_d1, spk_rs_syn, spk_fs_syn)
        self.ap_spikes = self._cat_v(ap_th, ap_stn, ap_gpe, ap_gpi, ap_d2, ap_d1, ap_rs, ap_fs)
        self.v_all = self._cat_v(
            self.v_th, self.v_stn, self.v_gpe, self.v_gpi, self.v_d2, self.v_d1, self.v_rs, self.v_fs
        )


class _EulerSynapseDiscretization:
    """Original explicit-Euler recursive synaptic filter update."""

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

    def _exp2_step(self, a, b, spike, peak: float, tau1: float, tau2: float, dt):
        tp = (tau1 * tau2) / (tau2 - tau1) * math.log(tau2 / tau1)
        factor = 1.0 / (-math.exp(-tp / tau1) + math.exp(-tp / tau2))
        inc = peak * factor * spike
        a_new = a / (1.0 + dt / tau1) + inc
        b_new = b / (1.0 + dt / tau2) + inc
        return a_new, b_new, b_new - a_new


class _ExactSynapseDiscretization:
    """Exact homogeneous transition for linear alpha/bi-exponential filters.

    Spike increments are applied after the homogeneous transition, matching the
    current explicit-Euler event-order convention: an event changes the hidden
    filter state at the end of the current step, and conductance appears on
    subsequent steps rather than instantaneously.
    """

    def _alpha_step(self, s, z, spike, peak: float, tau: float, dt):
        const = peak / (tau * math.exp(-1.0))
        h = dt / tau
        decay = torch.exp(-h)
        s_h = decay * ((1.0 + h) * s + dt * z)
        z_h = decay * (-(dt / (tau * tau)) * s + (1.0 - h) * z)
        z_new = z_h + const * spike
        return s_h, z_new

    def _exp2_step(self, a, b, spike, peak: float, tau1: float, tau2: float, dt):
        tp = (tau1 * tau2) / (tau2 - tau1) * math.log(tau2 / tau1)
        factor = 1.0 / (-math.exp(-tp / tau1) + math.exp(-tp / tau2))
        inc = peak * factor * spike
        a_new = a * torch.exp(-dt / tau1) + inc
        b_new = b * torch.exp(-dt / tau2) + inc
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

    class kumaravelu_2016_fused(synapse_cls, spike_cls):
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
