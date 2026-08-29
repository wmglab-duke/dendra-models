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

    def minf(self, v):
        return (1 / (1 + exp(-1 * (v + 4) / 7.5))) ** (1 / 3)
    
    def taum(self, v):
        taum = 0.1 + 0.5 / (exp((v - 3) / 6.7) + exp(-1 * (v + 37) / 13.5))
        return taum / self.q10

    def assigned_values(self, v, values):
        return {"taum": self.taum(v), "minf": self.minf(v)}

    def state_defaults(self, v, values):
        return {"m": self.assigned_values(v, values)["minf"]}


class h(S):
    S.DERIVED_BUFFER("q10")
    S.STATE("h")
    S.DERIVATIVE("h' = (hinf - h) / tauh")
    S.ASSIGNED("hinf", "tauh")
    S.GLOBAL_SIGNED(aq10=3.0)

    def derive_buffers(self):
        return {"q10": self.aq10 ** ((self.celsius - 22.0) / 10.0)}
    
    def hinf(self, v):
        return 1 / (1 + exp((v + 48) / 7))
    
    def tauh(self, v):
        tauh = 10 / (exp((v - 54) / 23) + exp(-1 * (v + 150) / 35))
        return tauh / self.q10

    def assigned_values(self, v, values):
        return {"tauh": self.tauh(v), "hinf": self.hinf(v)}

    def state_defaults(self, v, values):
        return {"h": self.assigned_values(v, values)["hinf"]}


class s(S):
    S.DERIVED_BUFFER("q10")
    S.STATE("s")
    S.DERIVATIVE("s' = (sinf - s) / taus")
    S.ASSIGNED("sinf", "taus")
    S.GLOBAL_SIGNED(aq10=3.0)

    def derive_buffers(self):
        return {"q10": self.aq10 ** ((self.celsius - 22.0) / 10.0)}
    
    def sinf(self, v):
        return 1 / (1 + exp((v + 81) / 8.6))
    
    def taus(self, v):
        taus = 50 + 30 / (exp((v - 50) / 26) + exp(-1 * (v + 150) / 26))
        return taus / self.q10

    def assigned_values(self, v, values):
        return {"taus": self.taus(v), "sinf": self.sinf(v)}

    def state_defaults(self, v, values):
        return {"s": self.assigned_values(v, values)["sinf"]}


class cav22(M):
    M.STATE_BUNDLE(m, h, s)
    M.GLOBAL_SIGNED(gbar=0.0001)

    M.USEION("ca", read=["eca"], write=["ica"])

    def ica(self, v):
        return self.gbar * self.m**3 * self.h * self.s * (v - self.eca)


class cav22_augmented(cav22):
    cav22.GLOBAL_SIGNED(aug=1.0)

    def ica(self, v):
        return self.gbar * self.m**3 * self.h * self.s * (v - self.eca) * self.aug
