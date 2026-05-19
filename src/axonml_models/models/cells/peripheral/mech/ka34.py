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
    
    def minf(self, v):
        return (1 / (1 + exp(-1 * (v - 24) / 17))) ** (1 / 3)
    
    def taum(self, v):
        taum = 1 / (exp((v - 39) / 13) + exp(-1 * (v + 134) / 47)) / 2
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
        return 1 / (1 + exp((v + 31) / 12))
    
    def tauh(self, v):
        tauh = 15 + 1 / (exp((v - 60) / 15) + exp(-1 * (v + 300) / 27))
        return tauh / self.q10()

    def breakpoint(self, v, states):
        return {"tauh": self.tauh(v), "hinf": self.hinf(v)}

    def inf(self, v):
        return {"h": self.breakpoint(v, None)["hinf"]}


class ka34(M):
    M.STATE(m, h)
    M.GLOBALP(gbar=0.0001)

    M.USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gbar * self.m**3 * self.h * (v - self.ek)


class ka34_augmented(ka34):
    ka34.GLOBAL(aug=1.0)

    def ik(self, v):
        return self.gbar * self.m**3 * self.h * (v - self.ek) * self.aug