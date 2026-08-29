# h channel
# from Tom Andersson Sensitivity studies of voltage-dpendent conductance in neurons
# Tom has build his model on Kouranova 2008 Hyoerpolarization -activated
# cyclic nuleotide-gated channel mRNA and protein expression in large
# versus small diameter dorsal root ganglion neurons: correlation with
# hyperpolarization-activated current

import torch

from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import sigmoid, exp


class s(S):
    S.DERIVED_BUFFER("q10")

    S.STATE("s")
    S.GLOBAL_SIGNED(aq10=3.0, bq10=22.0, cq10=10.0)
    S.DERIVATIVE("s' = (sinf - s) / taus")
    S.ASSIGNED("sinf", "taus")

    def derive_buffers(self):
        return {"q10": 1 / (self.aq10 ** ((self.celsius - self.bq10) / self.cq10))}

    def assigned_values(self, v, values):
        sinf = sigmoid(-(v + 87.2) / 9.7)
        taus_a = 300.0 + 542.0 * exp((v + 25.0) / -20.0)
        taus_b = 2500.0 + 100.0 * exp((v + 240.0) / 50.0)
        taus = self.q10 * torch.where(v < -70.0, taus_b, taus_a)
        return {"sinf": sinf, "taus": taus}

    def state_defaults(self, v, values):
        return {"s": sigmoid(-(v + 87.2) / 9.7)}


class f(S):
    S.DERIVED_BUFFER("q10")

    S.STATE("f")
    S.GLOBAL_SIGNED(aq10=3.0, bq10=22.0, cq10=10.0)
    S.DERIVATIVE("f' = (finf - f) / tauf")
    S.ASSIGNED("finf", "tauf")

    def derive_buffers(self):
        return {"q10": 1 / (self.aq10 ** ((self.celsius - self.bq10) / self.cq10))}

    def assigned_values(self, v, values):
        finf = sigmoid(-(v + 87.2) / 9.7)
        tauf_a = 140.0 + 50.0 * exp((v + 25.0) / -20.0)
        tauf_b = 250.0 + 12.0 * exp((v + 240.0) / 50.0)
        tauf = self.q10 * torch.where(v < -70.0, tauf_b, tauf_a)
        return {"finf": finf, "tauf": tauf}

    def state_defaults(self, v, values):
        return {"f": sigmoid(-(v + 87.2) / 9.7)}


class mh(M):
    M.STATE_BUNDLE(s, f)
    M.GLOBAL_SIGNED(gbar=0.0001)
    M.USEION("k", read=["ek"], write=["ik"])
    M.USEION("na", read=["ena"], write=["ina"])
    M.ASSIGNED("g")

    def assigned_values(self, v, values):
        del v
        return {"g": self.gbar * (0.5 * values["s"] + 0.5 * values["f"])}

    def ina(self, v):
        return 0.5 * self.g * (v - self.ena)

    def ik(self, v):
        return 0.5 * self.g * (v - self.ek)
