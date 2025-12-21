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
        taum = 0.25 + 0.5 / (exp((v - 15) / 5) + exp(-1 * (v + 27) / 12))
        return taum / self.q10()
    
    def minf(self, v):
        return (1 / (1 + exp(-1 * (v - 1.75) / 10))) ** (1 / 3)

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
    
    def tauh(self, v):
        tauh = 1.4 / (exp((v - 145) / 30) + exp(-1 * (v + 150) / 16.4))
        return tauh / self.q10()
    
    def hinf(self, v):
        return 1 / (1 + exp((v - 10) / 8))

    def breakpoint(self, v, states):
        return {"tauh": self.tauh(v), "hinf": self.hinf(v)}

    def inf(self, v):
        return {"h": self.breakpoint(v, None)["hinf"]}


class cav12(M):
    M.STATE(m, h)
    M.GLOBAL(gbar=0.0001)

    M.USEION("ca", read=["eca"], write=["ica"])

    def ica(self, v):
        return self.gbar * self.m**3 * self.h * (v - self.eca)


class cav12_augmented(cav12):
    cav12.GLOBAL(aug=1.0)

    def ica(self, v):
        return self.gbar * self.m**3 * self.h * (v - self.eca) * self.aug
    