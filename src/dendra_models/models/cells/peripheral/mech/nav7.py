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
        return {"q10": self.aq10 ** ((self.celsius - 21.0) / 10.0)}
    
    def taum(self, v):
        taum = (
            0.07 / (exp((v - 11.4) / 14) + exp(-1 * (v + 61) / 9.4))
            + 0.1 / (1 + exp(-1 * (v + 5.5) / 8))
        )
        return taum / self.q10 / 2
    
    def minf(self, v):
        return (1 / (1 + exp(-1 * (v + 25) / 7))) ** (1 / 3)

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
        return {"q10": self.aq10 ** ((self.celsius - 21.0) / 10.0)}
    
    def hinf(self, v):
        return 1 / (1 + exp((v + 79) / 7))
    
    def tauh(self, v):
        tauh = 1 / (exp(-1 * (v + 131) / 9.5) + exp((v + 5.7) / 12.4))
        return tauh / self.q10

    def assigned_values(self, v, values):
        return {"tauh": self.tauh(v), "hinf": self.hinf(v)}

    def state_defaults(self, v, values):
        return {"h": self.assigned_values(v, values)["hinf"]}


class nav7(M):
    M.STATE_BUNDLE(m, h)
    M.GLOBAL_SIGNED(gbar=0.12)
    M.USEION("na", read=["ena"], write=["ina"])

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * (v - self.ena)

class nav7_augmented(nav7):
    nav7.GLOBAL_SIGNED(aug=1.0)
    def ina(self, v):
        return self.gbar * self.m**3 * self.h * (v - self.ena) * self.aug
