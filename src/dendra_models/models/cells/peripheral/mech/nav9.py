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
        return (1 / (1 + exp(-1 * (v + 51) / 8.4))) ** (1 / 3)
    
    def taum(self, v):
        taum = (0.3 + 1 / (exp((v + 1.3) / 22) + exp(-1 * (v + 82) / 10))) / 2
        return taum / self.q10()

    def breakpoint(self, v, states):
        return {"taum": self.taum(v), "minf": self.minf(v)}

    def inf(self, v):
        return {"m": (1 / (1 + exp(-1 * (v + 51) / 8.4))) ** (1 / 3)}


class h(S):
    has_q10 = True
    S.STATE("h")
    S.DERIVATIVE("h' = (hinf - h) / tauh")
    S.ASSIGNED("hinf", "tauh")
    S.GLOBAL(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - 22.0) / 10.0)
    
    def hinf(self, v):
        return 1 / (1 + exp((v + 55) / 11.5))
    
    def tauh(self, v):
        tauh = 2 + 1 / (exp((v - 32) / 14) + exp(-1 * (v + 157) / 12))
        return tauh / self.q10()

    def breakpoint(self, v, states):
        return {"tauh": self.tauh(v), "hinf": self.hinf(v)}

    def inf(self, v):
        return {"h": 1 / (1 + exp((v + 55) / 11.5))}


class s(S):
    has_q10 = True
    S.STATE("s")
    S.DERIVATIVE("s' = (sinf - s) / taus")
    S.ASSIGNED("sinf", "taus")
    S.GLOBAL(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - 22.0) / 10.0)
    
    def taus(self, v):
        taus = 1 / (exp((v - 195) / 29) + exp(-1 * (v + 193) / 12.4)) + 1165 / (
            1 + exp(-1 * (v + 40) / 30)
        )
        return taus / self.q10()
    
    def sinf(self, v):
        return 0.03 + 0.97 / (1 + exp((v + 79) / 6.87))

    def breakpoint(self, v, states):
        return {"taus": self.taus(v), "sinf": self.sinf(v)}

    def inf(self, v):
        return {"s": 0.03 + 0.97 / (1 + exp((v + 79) / 6.87))}


class nav9(M):
    M.STATE(m, h, s)
    M.GLOBALP(gbar=0.0)
    M.USEION("na", read=["ena"], write=["ina"])

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * self.s * (v - self.ena)


class nav9_augmented(nav9):
    nav9.GLOBAL(aug=1.0)

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * self.s * (v - self.ena) * self.aug