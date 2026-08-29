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
        taum = 0.25 + 0.5 / (exp((v - 15) / 5) + exp(-1 * (v + 27) / 12))
        return taum / self.q10
    
    def minf(self, v):
        return (1 / (1 + exp(-1 * (v - 1.75) / 10))) ** (1 / 3)

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
    
    def tauh(self, v):
        tauh = 1.4 / (exp((v - 145) / 30) + exp(-1 * (v + 150) / 16.4))
        return tauh / self.q10
    
    def hinf(self, v):
        return 1 / (1 + exp((v - 10) / 8))

    def assigned_values(self, v, values):
        return {"tauh": self.tauh(v), "hinf": self.hinf(v)}

    def state_defaults(self, v, values):
        return {"h": self.assigned_values(v, values)["hinf"]}


class cav12(M):
    M.STATE_BUNDLE(m, h)
    M.GLOBAL_SIGNED(gbar=0.0001)

    M.USEION("ca", read=["eca"], write=["ica"])

    def ica(self, v):
        return self.gbar * self.m**3 * self.h * (v - self.eca)


class cav12_augmented(cav12):
    cav12.GLOBAL_SIGNED(aug=1.0)

    def ica(self, v):
        return self.gbar * self.m**3 * self.h * (v - self.eca) * self.aug
