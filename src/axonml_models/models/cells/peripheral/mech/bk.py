from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S
from axonml.models.mechanisms.ops import *


class m(S):
    has_q10 = True
    S.STATE("m")
    S.DERIVATIVE("m' = (minf - m) / taum")
    S.ASSIGNED("minf", "taum")
    S.GLOBAL(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - 22.0) / 10.0)

    def breakpoint(self, v):
        q10 = self.q10()
        pca = log10(self.cai) - 3
        v12 = -50.0 * pca - 232.0
        minf = 1 / (1 + exp(-1 * (v - v12) / 24))
        taum = (
            1
            / (
                exp((v + (58 * pca) + 303) / (3.2 * pca))
                + exp(-1 * (v + (107 * pca) + 453) / (6.8 * pca))
            )
            + 0.4
        )
        taum = taum / q10
        return {"taum": taum, "minf": minf}

    def inf(self, v):
        pca = log10(self.cai) - 3
        v12 = -50.0 * pca - 232.0
        return {"m": 1 / (1 + exp(-1 * (v - v12) / 24))}


class h(S):
    has_q10 = True
    S.STATE("h")
    S.DERIVATIVE("h' = (hinf - h) / tauh")
    S.ASSIGNED("hinf", "tauh")
    S.GLOBAL(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - 22.0) / 10.0)

    def breakpoint(self, v):
        q10 = self.q10()
        pca = log10(self.cai) - 3
        vh12 = -8 * pca + 35
        hinf = 1 / (1 + exp((v - vh12) / 47))
        tauh = 1 / (
            exp((v + (3 * pca) + 100) / (3 * pca))
            + exp(-1 * (v + (191 * pca) + 600) / (17 * pca))
        )
        tauh = tauh / q10
        return {"tauh": tauh, "hinf": hinf}

    def inf(self, v):
        pca = log10(self.cai) - 3
        vh12 = -8 * pca + 35
        return {"h": 1 / (1 + exp((v - vh12) / 47))}


class bk(M):
    M.STATE(m, h)
    M.GLOBAL(gbar=0.0001)

    M.USEION("k", read=["ek"], write=["ik"])
    M.USEION("ca", read=["cai"])

    def ik(self, v):
        return self.gbar * self.m * self.h * (v - self.ek)
