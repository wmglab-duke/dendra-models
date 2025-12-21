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
    
    def taum(self, v):
        taum = 0.03 + 0.5 / (exp((v) / 12) + exp(-1 * (v + 29) / 18))
        return taum / self.q10() / 2
    
    def minf(self, v):
        return (1 / (1 + exp(-1 * (v + 4) / 7))) ** (1 / 3)

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
        return 1 / (1 + exp((v + 28) / 4.56))
    
    def tauh(self, v):
        tauh = (
            2.63
            + 250 / (exp((v + 36) / 7.7) + exp(-1 * (v) / 16.3))
            + 1.4 / (1 + exp(-1 * (v + 0.6) / 2.95))
        )
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
        return 1 / (1 + exp((v + 50) / 7.5))
    
    def taus(self, v):
        taus = 34 / (exp((v + 2) / 16) + exp(-1 * (v + 108) / 8)) + 160 / (
            1 + exp(-1 * (v + 110) / 75)
        )
        return taus / self.q10()

    def breakpoint(self, v, states):
        return {"taus": self.taus(v), "sinf": self.sinf(v)}

    def inf(self, v):
        return {"s": self.breakpoint(v, None)["sinf"]}


class newnav8(M):
    M.STATE(m, h, s)
    M.GLOBAL(gbar=0.0)
    M.USEION("na", read=["ena"], write=["ina"])

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * self.s * (v - self.ena)


class newnav8_augmented(newnav8):
    newnav8.GLOBAL(aug=1.0)
    def ina(self, v):
        return self.gbar * self.m**3 * self.h * self.s * (v - self.ena) * self.aug
