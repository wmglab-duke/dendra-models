from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class m(S):
    has_q10 = True
    S.STATE("m")
    S.DERIVATIVE("m' = (minf - m) / taum")
    S.ASSIGNED("minf", "taum")
    S.GLOBAL(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - 22.0) / 10.0)

    def minf(self, v):
        return (1 / (1 + exp(-1 * (v + 4) / 7.5))) ** (1 / 3)
    
    def taum(self, v):
        taum = 0.1 + 0.5 / (exp((v - 3) / 6.7) + exp(-1 * (v + 37) / 13.5))
        return taum / self.q10()

    def breakpoint(self, v, states):
        return {"taum": self.taum(v), "minf": self.minf(v)}

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
    
    def hinf(self, v):
        return 1 / (1 + exp((v + 48) / 7))
    
    def tauh(self, v):
        tauh = 10 / (exp((v - 54) / 23) + exp(-1 * (v + 150) / 35))
        return tauh / self.q10()

    def breakpoint(self, v, states):
        return {"tauh": self.tauh(v), "hinf": self.hinf(v)}

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
    
    def sinf(self, v):
        return 1 / (1 + exp((v + 81) / 8.6))
    
    def taus(self, v):
        taus = 50 + 30 / (exp((v - 50) / 26) + exp(-1 * (v + 150) / 26))
        return taus / self.q10()

    def breakpoint(self, v, states):
        return {"taus": self.taus(v), "sinf": self.sinf(v)}

    def inf(self, v):
        return {"s": self.breakpoint(v, None)["sinf"]}


class cav22(M):
    M.STATE(m, h, s)
    M.GLOBALP(gbar=0.0001)

    M.USEION("ca", read=["eca"], write=["ica"])

    def ica(self, v):
        return self.gbar * self.m**3 * self.h * self.s * (v - self.eca)


class cav22_augmented(cav22):
    cav22.GLOBAL(aug=1.0)

    def ica(self, v):
        return self.gbar * self.m**3 * self.h * self.s * (v - self.eca) * self.aug