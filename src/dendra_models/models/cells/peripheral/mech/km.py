from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class m(S):
    S.DERIVED_BUFFER("q10")
    S.STATE("m")
    S.DERIVATIVE("m' = (minf - m) / taum")
    S.ASSIGNED("minf", "taum")
    S.GLOBAL_SIGNED(aq10=3.0)

    def derive_buffers(self):
        return {"q10": self.aq10 ** ((self.celsius - 22.0) / 10.0)}
    
    def taum(self, v):
        taum = 104 / (exp((v - 7) / 20) + exp(-1 * (v + 32) / 20)) + 30 / (
            1 + exp((-1 * (v + 40) / 80))
        )
        return taum / self.q10 / 2
    
    def minf(self, v):
        return (1 / (1 + exp((-1 * (v + 44) / 6.4)))) ** (1 / 3)

    def assigned_values(self, v, values):
        return {"taum": self.taum(v), "minf": self.minf(v)}

    def state_defaults(self, v, values):
        return {"m": (1 / (1 + exp((-1 * (v + 44) / 6.4)))) ** (1 / 3)}


class n(S):
    S.DERIVED_BUFFER("q10")
    S.STATE("n")
    S.DERIVATIVE("n' = (ninf - n) / taun")
    S.ASSIGNED("ninf", "taun")
    S.GLOBAL_SIGNED(aq10=3.0)

    def derive_buffers(self):
        return {"q10": self.aq10 ** ((self.celsius - 22.0) / 10.0)}
    
    def taun(self, v):
        taun = (
            15
            + 62 / (exp((v - 13) / 20) + exp(-1 * (v + 90) / 20))
            + 50 / (1 + exp((-1 * (v + 50) / 8)))
        )
        return taun / self.q10 / 2
    
    def ninf(self, v):
        return (1 / (1 + exp((-1 * (v + 32) / 9.2)))) ** (1 / 3)

    def assigned_values(self, v, values):
        return {"taun": self.taun(v), "ninf": self.ninf(v)}

    def state_defaults(self, v, values):
        return {"n": (1 / (1 + exp((-1 * (v + 32) / 9.2)))) ** (1 / 3)}


class km(M):
    M.STATE_BUNDLE(m, n)
    M.GLOBAL_SIGNED(gbar=0.0001)

    M.USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gbar * (0.25 * self.m**3 + 0.75 * self.n**3) * (v - self.ek)
    

class km_augmented(km):
    km.GLOBAL_SIGNED(aug=1.0)

    def ik(self, v):
        return self.gbar * (0.25 * self.m**3 + 0.75 * self.n**3) * (v - self.ek) * self.aug
