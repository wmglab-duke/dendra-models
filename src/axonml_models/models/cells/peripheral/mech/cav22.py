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

    def breakpoint(self, v, states):
        minf = (1 / (1 + exp(-1 * (v + 4) / 7.5))) ** (1 / 3)
        taum = 0.1 + 0.5 / (exp((v - 3) / 6.7) + exp(-1 * (v + 37) / 13.5))
        taum = taum / self.q10()
        return {"taum": taum, "minf": minf}

    def inf(self, v):
        return {"m": self.breakpoint(v, None)["minf"]}


class h(S):
    has_q10 = True
    S.STATE("h")
    S.DERIVATIVE("h' = (hinf - h) / tauh")
    S.ASSIGNED("hinf", "tauh")
    S.GLOBAL(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - 22.0) / 10.0)

    def breakpoint(self, v, states):
        hinf = 1 / (1 + exp((v + 48) / 7))
        tauh = 10 / (exp((v - 54) / 23) + exp(-1 * (v + 150) / 35))
        tauh = tauh / self.q10()
        return {"tauh": tauh, "hinf": hinf}

    def inf(self, v):
        return {"h": self.breakpoint(v, None)["hinf"]}


class s(S):
    has_q10 = True
    S.STATE("s")
    S.DERIVATIVE("s' = (sinf - s) / taus")
    S.ASSIGNED("sinf", "taus")
    S.GLOBAL(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - 22.0) / 10.0)

    def breakpoint(self, v, states):
        sinf = 1 / (1 + exp((v + 81) / 8.6))
        taus = 50 + 30 / (exp((v - 50) / 26) + exp(-1 * (v + 150) / 26))
        taus = taus / self.q10()
        return {"taus": taus, "sinf": sinf}

    def inf(self, v):
        return {"s": self.breakpoint(v, None)["sinf"]}


class cav22(M):
    M.STATE(m, h, s)
    M.GLOBAL(gbar=0.0001)

    M.USEION("ca", read=["eca"], write=["ica"])

    def ica(self, v):
        return self.gbar * self.m**3 * self.h * self.s * (v - self.eca)


class cav22_augmented(cav22):
    cav22.GLOBAL(aug=1.0)

    def ica(self, v):
        return self.gbar * self.m**3 * self.h * self.s * (v - self.eca) * self.aug