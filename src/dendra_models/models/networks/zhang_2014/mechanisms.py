"""Dendra mechanism translations for Zhang et al. 2014 / ModelDB 168414.

The classes in this module are direct, first-pass translations of the NEURON
MOD mechanisms used by the Zhang dorsal horn network.  Intrinsic mechanisms are
implemented as distributed currents.  Synaptic mechanisms are implemented as
Dendra PointProcess/Synapse classes with double-exponential conductance and the
Fuhrmann-style short-term-plasticity update used by the original `*_DynSyn.mod`
files.

Units follow the original MOD files as closely as Dendra's mechanism interface
allows: membrane potentials in mV, time in ms, conductance densities in
mho/cm^2-style NEURON values, and point-process weights in the same conductance
scale used by Dendra's stock exp2syn.
"""

from __future__ import annotations

from dendra.models.mechanisms import Mechanism as M
from dendra.models.mechanisms import PointProcess as PP
from dendra.models.mechanisms import State as S
from dendra.models.mechanisms import Synapse as Syn
from dendra.models.mechanisms.ops import exp, log, vtrap

import torch


FARADAY = 96485.33212331001
R_GAS = 8.314


def _zeros_like(v):
    return torch.zeros_like(v)


def _ones_like(v):
    return torch.ones_like(v)


def _clamp_tau(x, minimum=1e-9):
    return torch.clamp(x, min=minimum)


def _vtrap_1mexp(x, c):
    """Return x / (1 - exp(-x/c)) with the MOD vtrap Taylor fallback."""
    q = x / c
    val = x / (1.0 - exp(-q))
    approx = c + x / 2.0
    return torch.where(torch.abs(q) < 1e-6, approx, val)


# ---------------------------------------------------------------------------
# Fast Traub-style Na/K current, HH2.mod
# ---------------------------------------------------------------------------


class HH2_mhn(S):
    has_q10 = True

    S.STATE("m", "h", "n")
    S.RANGE(vtraub=-55.0)
    S.ASSIGNED("m_inf", "h_inf", "n_inf", "tau_m", "tau_h", "tau_n")
    S.DERIVATIVE(
        "m' = (m_inf - m) / tau_m",
        "h' = (h_inf - h) / tau_h",
        "n' = (n_inf - n) / tau_n",
    )

    def calc_q10(self):
        return 3.0 ** ((self.celsius - 36.0) / 10.0)

    def breakpoint(self, v, states):
        q10 = self.q10()
        v2 = v - self.vtraub
        vh = 5.0

        a_m = 0.32 * vtrap(13.0 - v2, 4.0)
        b_m = 0.28 * vtrap(v2 - 40.0, 5.0)
        tau_m = 1.0 / _clamp_tau(a_m + b_m) / q10
        m_inf = a_m / _clamp_tau(a_m + b_m)

        a_h = 0.128 * exp((17.0 - v2 - vh) / 18.0)
        b_h = 4.0 / (1.0 + exp((40.0 - v2 - vh) / 5.0))
        tau_h = 1.0 / _clamp_tau(a_h + b_h) / q10
        h_inf = a_h / _clamp_tau(a_h + b_h)

        a_n = 0.032 * vtrap(15.0 - v2, 5.0)
        b_n = 0.5 * exp((10.0 - v2) / 40.0)
        tau_n = 1.0 / _clamp_tau(a_n + b_n) / q10
        n_inf = a_n / _clamp_tau(a_n + b_n)

        return {
            "m_inf": m_inf,
            "h_inf": h_inf,
            "n_inf": n_inf,
            "tau_m": tau_m,
            "tau_h": tau_h,
            "tau_n": tau_n,
        }

    def inf(self, v):
        states = self.breakpoint(v, None)
        return {"m": states["m_inf"], "h": states["h_inf"], "n": states["n_inf"]}


class HH2(M):
    M.STATE(HH2_mhn)
    M.RANGE(gnabar=0.1, gkbar=0.06, ena=50.0, ek=-77.0)
    M.NONSPECIFIC_CURRENT("ina", "ik")

    def ina(self, v):
        return self.gnabar * self.m**3 * self.h * (v - self.ena)

    def ik(self, v):
        return self.gkbar * self.n**4 * (v - self.ek)


# ---------------------------------------------------------------------------
# Melnick / Safronov Na and K currents: B_NA, B_A, B_DR, KDR, KDRI, SS
# ---------------------------------------------------------------------------


class B_Na_mh(S):
    has_q10 = True

    S.STATE("m", "h")
    S.RANGE(alpha_shift=0.0, beta_shift=0.0)
    S.ASSIGNED("m_inf", "h_inf", "tau_m", "tau_h")
    S.DERIVATIVE("m' = (m_inf - m) / tau_m", "h' = (h_inf - h) / tau_h")

    def calc_q10(self):
        return 3.0 ** ((self.celsius - 23.0) / 10.0)

    def breakpoint(self, v, states):
        q10 = self.q10()
        a_m = q10 * 0.182 * vtrap(-v + 7.0 - 35.0 + self.alpha_shift, 9.0)
        b_m = q10 * 0.124 * vtrap(v - 7.0 + 35.0 + self.beta_shift, 9.0)
        tau_m = 1.0 / _clamp_tau(a_m + b_m)
        m_inf = a_m / _clamp_tau(a_m + b_m)

        a_h = q10 * (0.061 * vtrap(-v + 13.0 - 48.0 + self.alpha_shift, 3.0) + 0.0166)
        b_h = q10 * 0.0018 * vtrap(v - 13.0 + 84.0 + self.beta_shift, 18.0)
        tau_h = 1.0 / _clamp_tau(a_h + b_h)
        h_inf = 1.0 / (1.0 + exp((v + 75.0 - 11.0) / 9.0))
        return {"m_inf": m_inf, "h_inf": h_inf, "tau_m": tau_m, "tau_h": tau_h}

    def inf(self, v):
        states = self.breakpoint(v, None)
        return {"m": states["m_inf"], "h": states["h_inf"]}


class B_Na(M):
    M.STATE(B_Na_mh)
    M.RANGE(gnabar=0.0, ena=53.0)
    M.NONSPECIFIC_CURRENT("ina")

    def ina(self, v):
        return self.gnabar * self.m**3 * self.h * (v - self.ena)


class SS_mh(S):
    has_q10 = True

    S.STATE("m", "h")
    S.ASSIGNED("m_inf", "h_inf", "tau_m", "tau_h")
    S.DERIVATIVE("m' = (m_inf - m) / tau_m", "h' = (h_inf - h) / tau_h")

    def calc_q10(self):
        return 3.0 ** ((self.celsius - 23.0) / 10.0)

    def breakpoint(self, v, states):
        q10 = self.q10()
        a_m = q10 * 0.182 * vtrap(-v - 45.0, 9.0)
        b_m = q10 * 0.124 * vtrap(v + 45.0, 9.0)
        tau_m = 1.0 / _clamp_tau(a_m + b_m)
        m_inf = a_m / _clamp_tau(a_m + b_m)

        a_h = q10 * 0.024 * vtrap(-v - 50.0, 5.0)
        b_h = q10 * 0.0091 * vtrap(v + 75.0, 5.0)
        tau_h = 1.0 / _clamp_tau(a_h + b_h)
        h_inf = 1.0 / (1.0 + exp((v + 75.0) / 9.0))
        return {"m_inf": m_inf, "h_inf": h_inf, "tau_m": tau_m, "tau_h": tau_h}

    def inf(self, v):
        states = self.breakpoint(v, None)
        return {"m": states["m_inf"], "h": states["h_inf"]}


class SS(M):
    M.STATE(SS_mh)
    M.RANGE(gnabar=0.0, ena=53.0)
    M.NONSPECIFIC_CURRENT("ina")

    def ina(self, v):
        return self.gnabar * self.m**3 * (v - self.ena)


class KDR_n(S):
    has_q10 = True

    S.STATE("n")
    S.ASSIGNED("n_inf", "tau_n")
    S.DERIVATIVE("n' = (n_inf - n) / tau_n")

    def calc_q10(self):
        return 3.0 ** ((self.celsius - 23.0) / 10.0)

    def breakpoint(self, v, states):
        q10 = self.q10()
        a = q10 * 0.069 * vtrap(-v - 5.0, 10.0)
        b = q10 * 0.024 * exp((-v - 1.0) / 30.0)
        tau = 1.0 / _clamp_tau(a + b)
        inf = a / _clamp_tau(a + b)
        return {"n_inf": inf, "tau_n": tau}

    def inf(self, v):
        states = self.breakpoint(v, None)
        return {"n": states["n_inf"]}


class KDR(M):
    M.STATE(KDR_n)
    M.RANGE(gkbar=0.0, ek=-80.0)
    M.NONSPECIFIC_CURRENT("ik")

    def ik(self, v):
        return self.gkbar * self.n**4 * (v - self.ek)


class B_DR_n(S):
    has_q10 = True

    S.STATE("n")
    S.ASSIGNED("n_inf", "tau_n")
    S.DERIVATIVE("n' = (n_inf - n) / tau_n")

    def calc_q10(self):
        return 3.0 ** ((self.celsius - 23.0) / 10.0)

    def breakpoint(self, v, states):
        q10 = self.q10()
        a = q10 * 0.0075 * vtrap(-v - 30.0, 10.0)
        b = q10 * 0.1 * exp((-v - 46.0) / 31.0)
        tau = 1.0 / _clamp_tau(a + b)
        inf = a / _clamp_tau(a + b)
        return {"n_inf": inf, "tau_n": tau}

    def inf(self, v):
        states = self.breakpoint(v, None)
        return {"n": states["n_inf"]}


class B_DR(M):
    M.STATE(B_DR_n)
    M.RANGE(gkbar=0.0, ek=-80.0)
    M.NONSPECIFIC_CURRENT("ik")

    def ik(self, v):
        return self.gkbar * self.n**4 * (v - self.ek)


class KDRI_nh(S):
    has_q10 = True

    S.STATE("n", "h")
    S.ASSIGNED("n_inf", "h_inf", "tau_n", "tau_h")
    S.DERIVATIVE("n' = (n_inf - n) / tau_n", "h' = (h_inf - h) / tau_h")

    def calc_q10(self):
        return 3.0 ** ((self.celsius - 23.0) / 10.0)

    def breakpoint(self, v, states):
        q10 = self.q10()
        a_n = q10 * 0.035 * vtrap(-v - 15.0, 9.0)
        b_n = q10 * 0.014 * exp((-v + 12.0) / 46.0)
        tau_n = 1.0 / _clamp_tau(a_n + b_n)
        n_inf = a_n / _clamp_tau(a_n + b_n)

        a_h = q10 * 0.0083 * (1.0 / (exp((v + 20.0) / 10.0) + 1.0) + 1.0)
        b_h = q10 * 0.0083 / (exp((-v - 20.0) / 10.0) + 1.0)
        tau_h = 1.0 / _clamp_tau(a_h + b_h)
        h_inf = a_h / _clamp_tau(a_h + b_h)
        return {"n_inf": n_inf, "h_inf": h_inf, "tau_n": tau_n, "tau_h": tau_h}

    def inf(self, v):
        states = self.breakpoint(v, None)
        return {"n": states["n_inf"], "h": states["h_inf"]}


class KDRI(M):
    M.STATE(KDRI_nh)
    M.RANGE(gkbar=0.0, ek=-80.0)
    M.NONSPECIFIC_CURRENT("ik")

    def ik(self, v):
        return self.gkbar * self.n**4 * self.h * (v - self.ek)


class B_A_nh(S):
    has_q10 = True

    S.STATE("n", "h")
    S.ASSIGNED("n_inf", "h_inf", "tau_n", "tau_h")
    S.DERIVATIVE("n' = (n_inf - n) / tau_n", "h' = (h_inf - h) / tau_h")

    def calc_q10(self):
        return 3.0 ** ((self.celsius - 23.0) / 10.0)

    def breakpoint(self, v, states):
        q10 = self.q10()
        a_n = q10 * 0.032 * vtrap(-v - 64.0, 6.0)
        b_n = q10 * 0.203 * exp((-v - 40.0) / 24.0)
        tau_n = 1.0 / _clamp_tau(a_n + b_n)
        n_inf = a_n / _clamp_tau(a_n + b_n)

        a_h = q10 * 0.05 / (exp((v + 86.0) / 10.0) + 1.0)
        b_h = q10 * 0.05 / (exp((-v - 86.0) / 10.0) + 1.0)
        tau_h = 1.0 / _clamp_tau(a_h + b_h)
        h_inf = a_h / _clamp_tau(a_h + b_h)
        return {"n_inf": n_inf, "h_inf": h_inf, "tau_n": tau_n, "tau_h": tau_h}

    def inf(self, v):
        states = self.breakpoint(v, None)
        return {"n": states["n_inf"], "h": states["h_inf"]}


class B_A(M):
    M.STATE(B_A_nh)
    M.RANGE(gkbar=0.0, ek=-80.0)
    M.NONSPECIFIC_CURRENT("ik")

    def ik(self, v):
        return self.gkbar * self.n**4 * self.h * (v - self.ek)


# ---------------------------------------------------------------------------
# Calcium dynamics and calcium-dependent currents
# ---------------------------------------------------------------------------


class ca_dynamics(S):
    S.STATE("cai_new")
    S.RANGE(depth=0.1, cai_inf=50.0e-6, cai_tau=2.0)
    S.ASSIGNED("drive_channel")
    S.DERIVATIVE("cai_new' = drive_channel + (cai_inf - cai_new) / cai_tau")

    def breakpoint(self, v, states):
        drive = -(10000.0) * self.ica / (2.0 * FARADAY * self.depth)
        drive = torch.clamp(drive, min=0.0)
        return {"drive_channel": drive}

    def inf(self, v):
        return {"cai_new": _zeros_like(v) + self.cai_inf}


class CaIntraCellDyn(M):
    M.STATE(ca_dynamics)
    M.USEION("ca", read=["ica"], write=["cai"])

    def initial(self, v):
        # The ion write buffer is local to the inserted slice; initialize it to
        # the same value as the internal state used by the calcium ODE.
        if "cai" in self._buffers:
            self._buffers["cai"] = self.cai_new.clone()

    def _advance(self, v, dt):  # noqa: N802 - follows Dendra internal hook name
        super()._advance(v, dt)
        if "cai" in self._buffers:
            self._buffers["cai"] = self.cai_new


class iKCa_m(S):
    has_q10 = True

    S.STATE("m")
    S.RANGE(beta=0.03, cac=0.001, taumin=0.1)
    S.ASSIGNED("m_inf", "tau_m")
    S.DERIVATIVE("m' = (m_inf - m) / tau_m")

    def calc_q10(self):
        return 3.0 ** ((self.celsius - 22.0) / 10.0)

    def breakpoint(self, v, states):
        car = (self.cai / self.cac) ** 2
        m_inf = car / (1.0 + car)
        tau_m = 1.0 / self.beta / (1.0 + car) / self.q10()
        tau_m = torch.clamp(tau_m, min=self.taumin)
        return {"m_inf": m_inf, "tau_m": tau_m}

    def inf(self, v):
        states = self.breakpoint(v, None)
        return {"m": states["m_inf"]}


class iKCa(M):
    M.STATE(iKCa_m)
    M.USEION("ca", read=["cai"])
    M.RANGE(gbar=0.01, ek=-80.0)
    M.NONSPECIFIC_CURRENT("ik")

    def ik(self, v):
        return self.gbar * self.m**3 * (v - self.ek)


class iCaAN_m(S):
    has_q10 = True

    S.STATE("m")
    S.RANGE(beta=2.0e-3, tau_factor=40.0, cac=5.0e-4, taumin=0.1)
    S.ASSIGNED("m_inf", "tau_m")
    S.DERIVATIVE("m' = (m_inf - m) / tau_m")

    def calc_q10(self):
        return 3.0 ** ((self.celsius - 22.0) / 10.0)

    def breakpoint(self, v, states):
        alpha2 = self.beta * (self.cai / self.cac) ** 2
        denom = _clamp_tau(alpha2 + self.beta)
        tau_m = self.tau_factor / denom / self.q10()
        tau_m = torch.clamp(tau_m, min=self.taumin)
        m_inf = alpha2 / denom
        return {"m_inf": m_inf, "tau_m": tau_m}

    def inf(self, v):
        states = self.breakpoint(v, None)
        return {"m": states["m_inf"]}


class iCaAN(M):
    M.STATE(iCaAN_m)
    M.USEION("ca", read=["cai"])
    M.RANGE(gbar=0.00025, ecan=-20.0)
    M.NONSPECIFIC_CURRENT("ican")

    def ican(self, v):
        return self.gbar * self.m**2 * (v - self.ecan)


class iCaL_m(S):
    has_q10 = True

    S.STATE("m")
    S.ASSIGNED("m_inf", "tau_m")
    S.DERIVATIVE("m' = (m_inf - m) / tau_m")

    def calc_q10(self):
        return 3.0 ** ((self.celsius - 23.5) / 10.0)

    def breakpoint(self, v, states):
        a = 1.6 / (1.0 + exp(-0.072 * (v - 5.0)))
        b = 0.02 * _vtrap_1mexp(-(v - 1.31), 5.36)
        tau_m = 1.0 / _clamp_tau(a + b) / self.q10()
        m_inf = 1.0 / (1.0 + exp((v + 10.0) / -10.0))
        return {"m_inf": m_inf, "tau_m": tau_m}

    def inf(self, v):
        states = self.breakpoint(v, None)
        return {"m": states["m_inf"]}


class iCaL(M):
    M.STATE(iCaL_m)
    M.USEION("ca", read=["cai", "cao"], write=["ica"])
    M.RANGE(pcabar=0.000276)

    def _ghk(self, v):
        z = 2.0
        w = v * 0.001 * z * FARADAY / (R_GAS * (self.celsius + 273.16))
        e = torch.where(torch.abs(w) > 1.0e-4, w / (exp(w) - 1.0), 1.0 - w / 2.0)
        return -0.001 * z * FARADAY * (self.cao - self.cai * exp(w)) * e

    def ica(self, v):
        return self.pcabar * self.m**2 * self._ghk(v)


class iNaP_mh(S):
    has_q10 = True

    S.STATE("m", "h")
    S.RANGE(vtraub=-55.0, vsm=-2.0, vsh=-5.0, gamma=0.5)
    S.ASSIGNED("m_inf", "h_inf", "tau_m", "tau_h")
    S.DERIVATIVE("m' = (m_inf - m) / tau_m", "h' = (h_inf - h) / tau_h")

    def calc_q10(self):
        return 3.0 ** ((self.celsius - 36.0) / 10.0)

    def breakpoint(self, v, states):
        q10 = self.q10()
        v2 = v - self.vtraub
        a_m = 0.32 * vtrap(self.vsm + 13.0 - v2, 4.0)
        b_m = 0.28 * vtrap(self.vsm + v2 - 40.0, 5.0)
        tau_m = 1.0 / _clamp_tau(a_m + b_m) / q10
        m_inf = a_m / _clamp_tau(a_m + b_m)

        a_h = 0.128 * exp((self.vsh + 17.0 - v2) / 18.0)
        b_h = 4.0 / (1.0 + exp(((self.vsh + 40.0 - v2) / 5.0) * self.gamma))
        tau_h = 1.0 / _clamp_tau(a_h + b_h) / q10
        h_inf = a_h / _clamp_tau(a_h + b_h)
        return {"m_inf": m_inf, "h_inf": h_inf, "tau_m": tau_m, "tau_h": tau_h}

    def inf(self, v):
        states = self.breakpoint(v, None)
        return {"m": states["m_inf"], "h": states["h_inf"]}


class iNaP(M):
    M.STATE(iNaP_mh)
    M.RANGE(gnabar=0.00029, ena=50.0)
    M.NONSPECIFIC_CURRENT("ina")

    def ina(self, v):
        return self.gnabar * self.m * self.h * (v - self.ena)


# ---------------------------------------------------------------------------
# Dynamic synapses from AMPA/GABA/Glycine/NMDA/NK1 *_DynSyn.mod files
# ---------------------------------------------------------------------------


class _DynSynMixin:
    """Fuhrmann-style STP + double exponential conductance mixin."""

    def _tau_rise_decay(self):
        tau_rise = None
        tau_decay = None
        for state in self.DE.values():
            if hasattr(state, "tau_rise"):
                tau_rise = state.tau_rise
            if hasattr(state, "tau_decay"):
                tau_decay = state.tau_decay
        if tau_rise is None or tau_decay is None:  # defensive; should not happen
            raise AttributeError("Dynamic synapse states must define tau_rise and tau_decay.")
        return tau_rise, tau_decay

    def initial(self, v):
        tau_rise, tau_decay = self._tau_rise_decay()
        denom = tau_decay - tau_rise
        safe_denom = torch.where(torch.abs(denom) < 1.0e-12, torch.ones_like(denom), denom)
        tp = (tau_rise * tau_decay) / safe_denom * log(tau_decay / tau_rise)
        factor = -exp(-tp / tau_rise) + exp(-tp / tau_decay)
        factor = torch.where(torch.abs(factor) < 1.0e-12, torch.ones_like(factor), factor)
        self.factor = 1.0 / factor
        self.P = _ones_like(v)
        self.Use = _zeros_like(v)
        if not hasattr(self, "zhang2014_last_stp_step"):
            self.register_buffer(
                "zhang2014_last_stp_step",
                torch.full((1,), -1, device=v.device, dtype=torch.long),
            )
        else:
            self.zhang2014_last_stp_step.fill_(-1)

    def _g(self):
        return self.B - self.A

    def _netcon_global_step(self, netcon):
        if netcon is None or not hasattr(netcon, "global_step"):
            return None
        try:
            return int(netcon.global_step.detach().cpu().reshape(-1)[0].item())
        except Exception:
            return None

    def _recover_stp_once_per_step(self, netcon):
        tau_fac = torch.clamp(self.tau_fac, min=1.0e-12)
        tau_rec = torch.clamp(self.tau_rec, min=1.0e-12)

        step = self._netcon_global_step(netcon)
        if not hasattr(self, "zhang2014_last_stp_step"):
            self.register_buffer(
                "zhang2014_last_stp_step",
                torch.full((1,), -1, device=self.P.device, dtype=torch.long),
            )

        should_recover = True
        if step is not None:
            last = int(self.zhang2014_last_stp_step.detach().cpu().reshape(-1)[0].item())
            should_recover = step != last

        if should_recover:
            self.Use = self.Use * exp(-self.dt / tau_fac)
            self.P = 1.0 - (1.0 - self.P) * exp(-self.dt / tau_rec)
            if step is not None:
                self.zhang2014_last_stp_step.fill_(int(step))

    def _advance_stp(self, weights, netcon):
        # A banked mechanism may be targeted by several NetCons in one network
        # timestep.  Recovery/facilitation decay is a property of the synapse
        # state and must happen once per timestep, not once per incoming NetCon.
        self._recover_stp_once_per_step(netcon)

        has_event = weights != 0
        use_event = self.Use + self.U1 * (1.0 - self.Use)
        p_event = self.P - use_event * self.P
        pv = use_event * self.P

        self.Use = torch.where(has_event, use_event, self.Use)
        self.P = torch.where(has_event, p_event, self.P)

        inc = weights * self.factor * pv
        self.A = self.A + inc
        self.B = self.B + inc

    def net_receive(self, weights, netcon):
        self._advance_stp(weights, netcon)


def _make_A_state(name: str, tau_rise_default: float):
    class A_state(S):
        S.STATE("A")
        S.RANGE(tau_rise=tau_rise_default)
        S.DERIVATIVE("A' = -A / tau_rise")

        def inf(self, v):
            return {"A": _zeros_like(v)}

    A_state.__name__ = f"{name}_A"
    return A_state


def _make_B_state(name: str, tau_decay_default: float):
    class B_state(S):
        S.STATE("B")
        S.RANGE(tau_decay=tau_decay_default)
        S.DERIVATIVE("B' = -B / tau_decay")

        def inf(self, v):
            return {"B": _zeros_like(v)}

    B_state.__name__ = f"{name}_B"
    return B_state


AMPA_A = _make_A_state("AMPA", 0.1)
AMPA_B = _make_B_state("AMPA", 5.0)
GABAa_A = _make_A_state("GABAa", 0.1)
GABAa_B = _make_B_state("GABAa", 20.0)
GABAb_A = _make_A_state("GABAb", 3.5)
GABAb_B = _make_B_state("GABAb", 260.9)
Glycine_A = _make_A_state("Glycine", 0.1)
Glycine_B = _make_B_state("Glycine", 10.0)
NMDA_A = _make_A_state("NMDA", 2.0)
NMDA_B = _make_B_state("NMDA", 100.0)
NK1_A = _make_A_state("NK1", 100.0)
NK1_B = _make_B_state("NK1", 3000.0)


class AMPA_DynSyn(_DynSynMixin, PP, Syn):
    PP.STATE(AMPA_A, AMPA_B)
    PP.RANGE(e=0.0, U1=1.0, tau_rec=0.1, tau_fac=0.1)
    PP.ASSIGNED("factor", "P", "Use")
    PP.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self._g() * (v - self.e)


class GABAa_DynSyn(_DynSynMixin, PP, Syn):
    PP.STATE(GABAa_A, GABAa_B)
    PP.RANGE(e=-70.0, U1=1.0, tau_rec=0.1, tau_fac=0.1)
    PP.ASSIGNED("factor", "P", "Use")
    PP.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self._g() * (v - self.e)


class GABAb_DynSyn(_DynSynMixin, PP, Syn):
    PP.STATE(GABAb_A, GABAb_B)
    PP.RANGE(e=-90.0, U1=1.0, tau_rec=0.1, tau_fac=0.1)
    PP.ASSIGNED("factor", "P", "Use")
    PP.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self._g() * (v - self.e)


class Glycine_DynSyn(_DynSynMixin, PP, Syn):
    PP.STATE(Glycine_A, Glycine_B)
    PP.RANGE(e=-70.0, U1=1.0, tau_rec=0.1, tau_fac=0.1)
    PP.ASSIGNED("factor", "P", "Use")
    PP.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self._g() * (v - self.e)


class NMDA_DynSyn(_DynSynMixin, PP, Syn):
    PP.STATE(NMDA_A, NMDA_B)
    PP.USEION("ca", write=["ica"])
    PP.RANGE(e=0.0, mgo=1.0, ca_ratio=0.1, U1=1.0, tau_rec=0.1, tau_fac=0.1)
    PP.ASSIGNED("factor", "P", "Use")
    PP.NONSPECIFIC_CURRENT("inon")

    def mgblock(self, v):
        return 1.0 / (1.0 + exp(0.062 * -v) * (self.mgo / 3.57))

    def _itotal(self, v):
        return self._g() * self.mgblock(v) * (v - self.e)

    def ica(self, v):
        return self.ca_ratio * self._itotal(v)

    def inon(self, v):
        return (1.0 - self.ca_ratio) * self._itotal(v)


class NK1_DynSyn(_DynSynMixin, PP, Syn):
    PP.STATE(NK1_A, NK1_B)
    PP.USEION("ca", write=["ica"])
    PP.RANGE(e=0.0, ca_ratio=0.1, U1=1.0, tau_rec=0.1, tau_fac=0.1)
    PP.ASSIGNED("factor", "P", "Use")
    PP.NONSPECIFIC_CURRENT("iNK1R")

    def _itotal(self, v):
        return self._g() * (v - self.e)

    def ica(self, v):
        return self.ca_ratio * self._itotal(v)

    def iNK1R(self, v):
        return (1.0 - self.ca_ratio) * self._itotal(v)


INTRINSIC_MECHANISMS = (
    HH2,
    B_Na,
    SS,
    KDR,
    B_DR,
    KDRI,
    B_A,
    CaIntraCellDyn,
    iKCa,
    iCaAN,
    iCaL,
    iNaP,
)

SYNAPTIC_MECHANISMS = (
    AMPA_DynSyn,
    GABAa_DynSyn,
    GABAb_DynSyn,
    Glycine_DynSyn,
    NMDA_DynSyn,
    NK1_DynSyn,
)

__all__ = [cls.__name__ for cls in INTRINSIC_MECHANISMS + SYNAPTIC_MECHANISMS]
