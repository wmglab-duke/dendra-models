"""
Dendra-style conductance-based mechanisms for the Kumaravelu, Brocker & Grill
(2016) CTX-BG-TH network model.

These classes implement the intrinsic Hodgkin-Huxley-style biophysics in the
MATLAB model. Synaptic currents and network connectivity should be attached as
separate synapse/current mechanisms. Current sign follows the MATLAB/Dendra
convention used in hh.py: currents returned here are outward, so positive
injected current is represented as a negative nonspecific current.
"""

import torch

from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import exp, log, vtrap


def _positive(x, eps=1.0e-12):
    """Clamp concentrations away from zero while preserving tensor dtype/device."""
    return torch.clamp(x, min=eps)


# MATLAB current densities are numerically in uA/cm^2 (conductance in mS/cm^2,
# voltage in mV). The single-compartment Dendra solver uses SI-scaled
# capacitance and expects mechanism currents in mA/cm^2. For HH-population
# voltage currents, multiply MATLAB-density currents by 1e-3. Calcium state
# updates below intentionally continue to use the unscaled MATLAB currents.
DENDRA_CURRENT_SCALE = 1.0e-3


# ---------------------------------------------------------------------------
# Thalamic relay cell
# ---------------------------------------------------------------------------


def th_minf(v):
    return 1.0 / (1.0 + exp(-(v + 37.0) / 7.0))


def th_hinf(v):
    return 1.0 / (1.0 + exp((v + 41.0) / 4.0))


def th_pinf(v):
    return 1.0 / (1.0 + exp(-(v + 60.0) / 6.2))


def th_rinf(v):
    return 1.0 / (1.0 + exp((v + 84.0) / 4.0))


def th_tauh(v):
    ah = 0.128 * exp(-(v + 46.0) / 18.0)
    bh = 4.0 / (1.0 + exp(-(v + 23.0) / 5.0))
    return 1.0 / (ah + bh)


def th_taur(v):
    return 0.15 * (28.0 + exp(-(v + 25.0) / 10.5))


class thalamic_states(S):
    S.STATE("h", "r")
    S.ASSIGNED("hinf", "htau", "rinf", "rtau")
    S.DERIVATIVE("h' = (hinf - h) / htau", "r' = (rinf - r) / rtau")
    S.RANGE(
        cm=1.0,
        current_scale=DENDRA_CURRENT_SCALE,
        gl=0.05,
        el=-70.0,
        gna=3.0,
        ena=50.0,
        gk=5.0,
        ek=-75.0,
        gt=5.0,
        et=0.0,
        i_stim=1.2,
    )

    def assigned_values(self, v, values):
        return {
            "hinf": th_hinf(v),
            "htau": th_tauh(v),
            "rinf": th_rinf(v),
            "rtau": th_taur(v),
        }

    def state_defaults(self, v, values):
        return {"h": th_hinf(v), "r": th_rinf(v)}


class thalamic(M):
    M.STATE_BUNDLE(thalamic_states)
    M.NONSPECIFIC_CURRENT("il", "ina", "ik", "it", "istim")
    M.EXPLICIT("istim")

    def _p(self):
        return self.DE["thalamic_states"]

    def il(self, v):
        p = self._p()
        return p.current_scale * p.gl * (v - p.el)

    def il_with_conductance(self, v):
        p = self._p()
        conductance = p.current_scale * p.gl
        return conductance * (v - p.el), conductance

    def ina(self, v):
        p = self._p()
        return p.current_scale * p.gna * th_minf(v) ** 3 * self.h * (v - p.ena)

    def ina_with_conductance(self, v):
        p = self._p()
        activation = th_minf(v)
        activation_derivative = activation * (1.0 - activation) / 7.0
        prefactor = p.current_scale * p.gna * self.h
        current = prefactor * activation**3 * (v - p.ena)
        conductance = prefactor * (
            activation**3
            + 3.0 * activation**2 * activation_derivative * (v - p.ena)
        )
        return current, conductance

    def ik(self, v):
        p = self._p()
        return p.current_scale * p.gk * (0.75 * (1.0 - self.h)) ** 4 * (v - p.ek)

    def ik_with_conductance(self, v):
        p = self._p()
        conductance = (
            p.current_scale * p.gk * (0.75 * (1.0 - self.h)) ** 4
        )
        return conductance * (v - p.ek), conductance

    def it(self, v):
        p = self._p()
        return p.current_scale * p.gt * th_pinf(v) ** 2 * self.r * (v - p.et)

    def it_with_conductance(self, v):
        p = self._p()
        activation = th_pinf(v)
        activation_derivative = activation * (1.0 - activation) / 6.2
        prefactor = p.current_scale * p.gt * self.r
        current = prefactor * activation**2 * (v - p.et)
        conductance = prefactor * (
            activation**2
            + 2.0 * activation * activation_derivative * (v - p.et)
        )
        return current, conductance

    def istim(self, v):
        p = self._p()
        return torch.zeros_like(v) - p.current_scale * p.i_stim


# ---------------------------------------------------------------------------
# Subthalamic nucleus cell
# ---------------------------------------------------------------------------


def stn_ainf(v):
    return 1.0 / (1.0 + exp(-(v + 45.0) / 14.7))


def stn_binf(v):
    return 1.0 / (1.0 + exp((v + 90.0) / 7.5))


def stn_cinf(v):
    return 1.0 / (1.0 + exp(-(v + 30.6) / 5.0))


def stn_d1inf(v):
    return 1.0 / (1.0 + exp((v + 60.0) / 7.5))


def stn_d2inf(v):
    return 1.0 / (1.0 + exp((v - 0.1) / 0.02))


def stn_hinf(v):
    return 1.0 / (1.0 + exp((v + 45.5) / 6.4))


def stn_minf(v):
    return 1.0 / (1.0 + exp(-(v + 40.0) / 8.0))


def stn_ninf(v):
    return 1.0 / (1.0 + exp(-(v + 41.0) / 14.0))


def stn_pinf(v):
    return 1.0 / (1.0 + exp(-(v + 56.0) / 6.7))


def stn_qinf(v):
    return 1.0 / (1.0 + exp((v + 85.0) / 5.8))


def stn_rinf(v):
    return 1.0 / (1.0 + exp(-(v - 0.17) / 0.08))


def stn_taua(v):
    return 1.0 + 1.0 / (1.0 + exp(-(v + 40.0) / -0.5))


def stn_taub(v):
    return 200.0 / (exp(-(v + 60.0) / -30.0) + exp(-(v + 40.0) / 10.0))


def stn_tauc(v):
    return 45.0 + 10.0 / (exp(-(v + 27.0) / -20.0) + exp(-(v + 50.0) / 15.0))


def stn_taud1(v):
    return 400.0 + 500.0 / (exp(-(v + 40.0) / -15.0) + exp(-(v + 20.0) / 20.0))


def stn_tauh(v):
    return 24.5 / (exp(-(v + 50.0) / -15.0) + exp(-(v + 50.0) / 16.0))


def stn_taum(v):
    return 0.2 + 3.0 / (1.0 + exp(-(v + 53.0) / -0.7))


def stn_taun(v):
    return 11.0 / (exp(-(v + 40.0) / -40.0) + exp(-(v + 40.0) / 50.0))


def stn_taup(v):
    return 5.0 + 0.33 / (exp(-(v + 27.0) / -10.0) + exp(-(v + 102.0) / 15.0))


def stn_tauq(v):
    return 400.0 / (exp(-(v + 50.0) / -15.0) + exp(-(v + 50.0) / 16.0))


class stn_states(S):
    S.STATE("n", "h", "m", "a", "b", "c", "d1", "d2", "p", "q", "r", "ca")
    S.ASSIGNED(
        "ninf", "ntau", "hinf", "htau", "minf", "mtau", "ainf", "atau",
        "binf", "btau", "cinf", "ctau", "d1inf", "d1tau", "d2inf", "d2tau",
        "pinf", "ptau", "qinf", "qtau", "rinf", "rtau", "eca", "ilca_for_ca",
        "it_for_ca",
    )
    S.DERIVATIVE(
        "n' = (ninf - n) / ntau",
        "h' = (hinf - h) / htau",
        "m' = (minf - m) / mtau",
        "a' = (ainf - a) / atau",
        "b' = (binf - b) / btau",
        "c' = (cinf - c) / ctau",
        "d1' = (d1inf - d1) / d1tau",
        "d2' = (d2inf - d2) / d2tau",
        "p' = (pinf - p) / ptau",
        "q' = (qinf - q) / qtau",
        "r' = (rinf - r) / rtau",
        "ca' = -alpha_ca * (ilca_for_ca + it_for_ca) - k_ca_decay * ca",
    )
    S.RANGE(
        cm=1.0,
        current_scale=DENDRA_CURRENT_SCALE,
        gl=0.35,
        el=-60.0,
        gna=49.0,
        ena=60.0,
        gk=57.0,
        ek=-90.0,
        ga=5.0,
        gL_Ca=15.0,
        gt=5.0,
        gcak=1.0,
        cao=2000.0,
        con=(8314.0 * 298.0) / (2.0 * 96485.0),
        alpha_ca=1.0 / (2.0 * 96485.0),
        k_ca_decay=2.0e-3,
        i_stim=0.0,
    )

    def assigned_values(self, v, values):
        ca = _positive(values["ca"])
        eca = self.con * log(self.cao / ca)
        ilca = self.gL_Ca * values["c"] ** 2 * values["d1"] * values["d2"] * (v - eca)
        it = self.gt * values["p"] ** 2 * values["q"] * (v - eca)
        return {
            "ninf": stn_ninf(v), "ntau": stn_taun(v),
            "hinf": stn_hinf(v), "htau": stn_tauh(v),
            "minf": stn_minf(v), "mtau": stn_taum(v),
            "ainf": stn_ainf(v), "atau": stn_taua(v),
            "binf": stn_binf(v), "btau": stn_taub(v),
            "cinf": stn_cinf(v), "ctau": stn_tauc(v),
            "d1inf": stn_d1inf(v), "d1tau": stn_taud1(v),
            "d2inf": stn_d2inf(v), "d2tau": torch.zeros_like(v) + 130.0,
            "pinf": stn_pinf(v), "ptau": stn_taup(v),
            "qinf": stn_qinf(v), "qtau": stn_tauq(v),
            "rinf": stn_rinf(v), "rtau": torch.zeros_like(v) + 2.0,
            "eca": eca,
            "ilca_for_ca": ilca,
            "it_for_ca": it,
        }

    def state_defaults(self, v, values):
        return {
            "n": stn_ninf(v), "h": stn_hinf(v), "m": stn_minf(v),
            "a": stn_ainf(v), "b": stn_binf(v), "c": stn_cinf(v),
            "d1": stn_d1inf(v), "d2": stn_d2inf(v), "p": stn_pinf(v),
            "q": stn_qinf(v), "r": stn_rinf(v), "ca": torch.zeros_like(v) + 0.005,
        }


class stn(M):
    M.STATE_BUNDLE(stn_states)
    M.NONSPECIFIC_CURRENT("ina", "ik", "ia", "ilca", "it", "icak", "il", "istim")
    M.EXPLICIT("istim")

    def _p(self):
        return self.DE["stn_states"]

    def _eca(self):
        p = self._p()
        return p.con * log(p.cao / _positive(self.ca))

    def ina(self, v):
        p = self._p()
        return p.current_scale * p.gna * self.m ** 3 * self.h * (v - p.ena)

    def ina_with_conductance(self, v):
        p = self._p()
        conductance = p.current_scale * p.gna * self.m**3 * self.h
        return conductance * (v - p.ena), conductance

    def ik(self, v):
        p = self._p()
        return p.current_scale * p.gk * self.n ** 4 * (v - p.ek)

    def ik_with_conductance(self, v):
        p = self._p()
        conductance = p.current_scale * p.gk * self.n**4
        return conductance * (v - p.ek), conductance

    def ia(self, v):
        p = self._p()
        return p.current_scale * p.ga * self.a ** 2 * self.b * (v - p.ek)

    def ia_with_conductance(self, v):
        p = self._p()
        conductance = p.current_scale * p.ga * self.a**2 * self.b
        return conductance * (v - p.ek), conductance

    def ilca(self, v):
        p = self._p()
        return p.current_scale * p.gL_Ca * self.c ** 2 * self.d1 * self.d2 * (v - self._eca())

    def ilca_with_conductance(self, v):
        p = self._p()
        reversal = self._eca()
        conductance = (
            p.current_scale * p.gL_Ca * self.c**2 * self.d1 * self.d2
        )
        return conductance * (v - reversal), conductance

    def it(self, v):
        p = self._p()
        return p.current_scale * p.gt * self.p ** 2 * self.q * (v - self._eca())

    def it_with_conductance(self, v):
        p = self._p()
        reversal = self._eca()
        conductance = p.current_scale * p.gt * self.p**2 * self.q
        return conductance * (v - reversal), conductance

    def icak(self, v):
        p = self._p()
        return p.current_scale * p.gcak * self.r ** 2 * (v - p.ek)

    def icak_with_conductance(self, v):
        p = self._p()
        conductance = p.current_scale * p.gcak * self.r**2
        return conductance * (v - p.ek), conductance

    def il(self, v):
        p = self._p()
        return p.current_scale * p.gl * (v - p.el)

    def il_with_conductance(self, v):
        p = self._p()
        conductance = p.current_scale * p.gl
        return conductance * (v - p.el), conductance

    def istim(self, v):
        p = self._p()
        return torch.zeros_like(v) - p.current_scale * p.i_stim


# ---------------------------------------------------------------------------
# GPe and GPi pallidal cells. Intrinsic dynamics are identical in the MATLAB
# implementation; different network inputs distinguish GPe from GPi.
# ---------------------------------------------------------------------------


def gpe_ainf(v):
    return 1.0 / (1.0 + exp(-(v + 57.0) / 2.0))


def gpe_hinf(v):
    return 1.0 / (1.0 + exp((v + 58.0) / 12.0))


def gpe_minf(v):
    return 1.0 / (1.0 + exp(-(v + 37.0) / 10.0))


def gpe_ninf(v):
    return 1.0 / (1.0 + exp(-(v + 50.0) / 14.0))


def gpe_rinf(v):
    return 1.0 / (1.0 + exp((v + 70.0) / 2.0))


def gpe_sinf(v):
    return 1.0 / (1.0 + exp(-(v + 35.0) / 2.0))


def gpe_tauh(v):
    return 0.05 + 0.27 / (1.0 + exp(-(v + 40.0) / -12.0))


def gpe_taun(v):
    return 0.05 + 0.27 / (1.0 + exp(-(v + 40.0) / -12.0))


class pallidal_states(S):
    S.STATE("n", "h", "r", "ca")
    S.ASSIGNED("ninf", "ntau", "hinf", "htau", "rinf", "rtau", "ica_for_ca", "it_for_ca")
    S.DERIVATIVE(
        "n' = n_scale * (ninf - n) / ntau",
        "h' = h_scale * (hinf - h) / htau",
        "r' = (rinf - r) / rtau",
        "ca' = ca_scale * (-ica_for_ca - it_for_ca - kca * ca)",
    )
    S.RANGE(
        cm=1.0,
        current_scale=DENDRA_CURRENT_SCALE,
        gl=0.1,
        el=-65.0,
        gna=120.0,
        ena=55.0,
        gk=30.0,
        ek=-80.0,
        gt=0.5,
        gca=0.15,
        eca=120.0,
        gahp=10.0,
        k1=10.0,
        kca=15.0,
        n_scale=0.1,
        h_scale=0.05,
        rtau=30.0,
        ca_scale=1.0e-4,
        i_stim=3.0,
    )

    def assigned_values(self, v, values):
        it = self.gt * gpe_ainf(v) ** 3 * values["r"] * (v - self.eca)
        ica = self.gca * gpe_sinf(v) ** 2 * (v - self.eca)
        return {
            "ninf": gpe_ninf(v),
            "ntau": gpe_taun(v),
            "hinf": gpe_hinf(v),
            "htau": gpe_tauh(v),
            "rinf": gpe_rinf(v),
            "rtau": torch.zeros_like(v) + self.rtau,
            "ica_for_ca": ica,
            "it_for_ca": it,
        }

    def state_defaults(self, v, values):
        return {"n": gpe_ninf(v), "h": gpe_hinf(v), "r": gpe_rinf(v), "ca": torch.zeros_like(v) + 0.1}


class gpe(M):
    M.STATE_BUNDLE(pallidal_states)
    M.NONSPECIFIC_CURRENT("il", "ik", "ina", "it", "ica", "iahp", "istim")
    M.EXPLICIT("istim")

    def _p(self):
        return self.DE["pallidal_states"]

    def il(self, v):
        p = self._p()
        return p.current_scale * p.gl * (v - p.el)

    def il_with_conductance(self, v):
        p = self._p()
        conductance = p.current_scale * p.gl
        return conductance * (v - p.el), conductance

    def ik(self, v):
        p = self._p()
        return p.current_scale * p.gk * self.n ** 4 * (v - p.ek)

    def ik_with_conductance(self, v):
        p = self._p()
        conductance = p.current_scale * p.gk * self.n**4
        return conductance * (v - p.ek), conductance

    def ina(self, v):
        p = self._p()
        return p.current_scale * p.gna * gpe_minf(v) ** 3 * self.h * (v - p.ena)

    def ina_with_conductance(self, v):
        p = self._p()
        activation = gpe_minf(v)
        activation_derivative = activation * (1.0 - activation) / 10.0
        prefactor = p.current_scale * p.gna * self.h
        current = prefactor * activation**3 * (v - p.ena)
        conductance = prefactor * (
            activation**3
            + 3.0 * activation**2 * activation_derivative * (v - p.ena)
        )
        return current, conductance

    def it(self, v):
        p = self._p()
        return p.current_scale * p.gt * gpe_ainf(v) ** 3 * self.r * (v - p.eca)

    def it_with_conductance(self, v):
        p = self._p()
        activation = gpe_ainf(v)
        activation_derivative = activation * (1.0 - activation) / 2.0
        prefactor = p.current_scale * p.gt * self.r
        current = prefactor * activation**3 * (v - p.eca)
        conductance = prefactor * (
            activation**3
            + 3.0 * activation**2 * activation_derivative * (v - p.eca)
        )
        return current, conductance

    def ica(self, v):
        p = self._p()
        return p.current_scale * p.gca * gpe_sinf(v) ** 2 * (v - p.eca)

    def ica_with_conductance(self, v):
        p = self._p()
        activation = gpe_sinf(v)
        activation_derivative = activation * (1.0 - activation) / 2.0
        prefactor = p.current_scale * p.gca
        current = prefactor * activation**2 * (v - p.eca)
        conductance = prefactor * (
            activation**2
            + 2.0 * activation * activation_derivative * (v - p.eca)
        )
        return current, conductance

    def iahp(self, v):
        p = self._p()
        return p.current_scale * p.gahp * (v - p.ek) * (self.ca / (self.ca + p.k1))

    def iahp_with_conductance(self, v):
        p = self._p()
        conductance = (
            p.current_scale * p.gahp * (self.ca / (self.ca + p.k1))
        )
        return conductance * (v - p.ek), conductance

    def istim(self, v):
        p = self._p()
        return torch.zeros_like(v) - p.current_scale * p.i_stim


class gpi(gpe):
    """Intrinsic GPi cell. Network inputs differ from GPe; intrinsic currents do not."""


# ---------------------------------------------------------------------------
# Striatal D1/D2 medium-spiny neuron intrinsic currents
# ---------------------------------------------------------------------------


def str_alpha_m(v):
    return 0.32 * vtrap(-(v + 54.0), 4.0)


def str_beta_m(v):
    return 0.28 * vtrap(v + 27.0, 5.0)


def str_alpha_h(v):
    return 0.128 * exp((-50.0 - v) / 18.0)


def str_beta_h(v):
    return 4.0 / (1.0 + exp((-27.0 - v) / 5.0))


def str_alpha_n(v):
    return 0.032 * vtrap(-(v + 52.0), 5.0)


def str_beta_n(v):
    return 0.5 * exp((-57.0 - v) / 40.0)


def str_alpha_p(v):
    return 3.209e-4 * vtrap(-(v + 30.0), 9.0)


def str_beta_p(v):
    return 3.209e-4 * vtrap(v + 30.0, 9.0)


def str_gaba_drive(v):
    return 2.0 * (1.0 + torch.tanh(v / 4.0))


class striatal_msn_states(S):
    S.STATE("m", "h", "n", "p")
    S.ASSIGNED("am", "bm", "ah", "bh", "an", "bn", "ap", "bp")
    S.DERIVATIVE(
        "m' = am * (1 - m) - bm * m",
        "h' = ah * (1 - h) - bh * h",
        "n' = an * (1 - n) - bn * n",
        "p' = ap * (1 - p) - bp * p",
    )
    S.RANGE(
        cm=1.0,
        current_scale=DENDRA_CURRENT_SCALE,
        gl=0.1,
        el=-67.0,
        gna=100.0,
        ena=50.0,
        gk=80.0,
        ek=-100.0,
        gm=1.0,
        gm_base=2.6,
        gm_pd_reduction=1.1,
        pd=0.0,
        em=-100.0,
        i_stim=0.0,
    )

    def assigned_values(self, v, values):
        return {
            "am": str_alpha_m(v), "bm": str_beta_m(v),
            "ah": str_alpha_h(v), "bh": str_beta_h(v),
            "an": str_alpha_n(v), "bn": str_beta_n(v),
            "ap": str_alpha_p(v), "bp": str_beta_p(v),
        }

    def state_defaults(self, v, values):
        am, bm = str_alpha_m(v), str_beta_m(v)
        ah, bh = str_alpha_h(v), str_beta_h(v)
        an, bn = str_alpha_n(v), str_beta_n(v)
        ap, bp = str_alpha_p(v), str_beta_p(v)
        return {
            "m": am / (am + bm),
            "h": ah / (ah + bh),
            "n": an / (an + bn),
            "p": ap / (ap + bp),
        }


class striatal_msn(M):
    M.STATE_BUNDLE(striatal_msn_states)
    M.NONSPECIFIC_CURRENT("ina", "ik", "il", "im", "istim")
    M.EXPLICIT("istim")

    def _p(self):
        return self.DE["striatal_msn_states"]

    def ina(self, v):
        p = self._p()
        return p.current_scale * p.gna * self.m ** 3 * self.h * (v - p.ena)

    def ina_with_conductance(self, v):
        p = self._p()
        conductance = p.current_scale * p.gna * self.m**3 * self.h
        return conductance * (v - p.ena), conductance

    def ik(self, v):
        p = self._p()
        return p.current_scale * p.gk * self.n ** 4 * (v - p.ek)

    def ik_with_conductance(self, v):
        p = self._p()
        conductance = p.current_scale * p.gk * self.n**4
        return conductance * (v - p.ek), conductance

    def il(self, v):
        p = self._p()
        return p.current_scale * p.gl * (v - p.el)

    def il_with_conductance(self, v):
        p = self._p()
        conductance = p.current_scale * p.gl
        return conductance * (v - p.el), conductance

    def im(self, v):
        p = self._p()
        gm_eff = (p.gm_base - p.gm_pd_reduction * p.pd) * p.gm
        return p.current_scale * gm_eff * self.p * (v - p.em)

    def im_with_conductance(self, v):
        p = self._p()
        gm_eff = (p.gm_base - p.gm_pd_reduction * p.pd) * p.gm
        conductance = p.current_scale * gm_eff * self.p
        return conductance * (v - p.em), conductance

    def istim(self, v):
        p = self._p()
        return torch.zeros_like(v) - p.current_scale * p.i_stim


class striatal_d2(striatal_msn):
    """Indirect-pathway striatal MSN. Same intrinsic currents as D1 in this model."""


class striatal_d1(striatal_msn):
    """Direct-pathway striatal MSN. Same intrinsic currents as D2 in this model."""
