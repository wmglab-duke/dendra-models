# h channel
# from Tom Andersson Sensitivity studies of voltage-dpendent conductance in neurons
# Tom has build his model on Kouranova 2008 Hyoerpolarization -activated
# cyclic nuleotide-gated channel mRNA and protein expression in large
# versus small diameter dorsal root ganglion neurons: correlation with
# hyperpolarization-activated current

import torch

from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S
from axonml.models.mechanisms.ops import sigmoid, exp


class s(S):
    has_q10 = True

    S.STATE("s")
    S.GLOBAL(aq10=3.0, bq10=22.0, cq10=10.0)
    S.DERIVATIVE("s' = (sinf - s) / taus")
    S.ASSIGNED("sinf", "taus")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.celsius - self.bq10) / self.cq10))

    def breakpoint(self, v, states):
        sinf = sigmoid(-(v + 87.2) / 9.7)
        taus_a = 300.0 + 542.0 * exp((v + 25.0) / -20.0)
        taus_b = 2500.0 + 100.0 * exp((v + 240.0) / 50.0)
        taus = self.q10() * torch.where(v < -70.0, taus_b, taus_a)
        return {"sinf": sinf, "taus": taus}

    def inf(self, v):
        return {"s": sigmoid(-(v + 87.2) / 9.7)}


class f(S):
    has_q10 = True

    S.STATE("f")
    S.GLOBAL(aq10=3.0, bq10=22.0, cq10=10.0)
    S.DERIVATIVE("f' = (finf - f) / tauf")
    S.ASSIGNED("finf", "tauf")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.celsius - self.bq10) / self.cq10))

    def breakpoint(self, v, states):
        finf = sigmoid(-(v + 87.2) / 9.7)
        tauf_a = 140.0 + 50.0 * exp((v + 25.0) / -20.0)
        tauf_b = 250.0 + 12.0 * exp((v + 240.0) / 50.0)
        tauf = self.q10() * torch.where(v < -70.0, tauf_b, tauf_a)
        return {"finf": finf, "tauf": tauf}

    def inf(self, v):
        return {"f": sigmoid(-(v + 87.2) / 9.7)}


class h(M):
    M.STATE(s, f)
    M.GLOBAL(gbar=0.0001)
    M.USEION("k", read=["ek"], write=["ik"])
    M.USEION("na", read=["ena"], write=["ina"])
    M.ASSIGNED("g")

    def initial(self, v):
        self.breakpoint(v, None)

    def breakpoint(self, v, states):
        self.g = self.gbar * (0.5 * self.s + 0.5 * self.f)

    def ina(self, v):
        return 0.5 * self.g * (v - self.ena)

    def ik(self, v):
        return 0.5 * self.g * (v - self.ek)
