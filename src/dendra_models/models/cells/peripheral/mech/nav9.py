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
        return (1 / (1 + exp(-1 * (v + 51) / 8.4))) ** (1 / 3)
    
    def taum(self, v):
        taum = (0.3 + 1 / (exp((v + 1.3) / 22) + exp(-1 * (v + 82) / 10))) / 2
        return taum / self.q10

    def assigned_values(self, v, values):
        return {"taum": self.taum(v), "minf": self.minf(v)}

    def state_defaults(self, v, values):
        return {"m": (1 / (1 + exp(-1 * (v + 51) / 8.4))) ** (1 / 3)}


class h(S):
    S.DERIVED_BUFFER("q10")
    S.STATE("h")
    S.DERIVATIVE("h' = (hinf - h) / tauh")
    S.ASSIGNED("hinf", "tauh")
    S.GLOBAL_SIGNED(aq10=3.0)

    def derive_buffers(self):
        return {"q10": self.aq10 ** ((self.celsius - 22.0) / 10.0)}
    
    def hinf(self, v):
        return 1 / (1 + exp((v + 55) / 11.5))
    
    def tauh(self, v):
        tauh = 2 + 1 / (exp((v - 32) / 14) + exp(-1 * (v + 157) / 12))
        return tauh / self.q10

    def assigned_values(self, v, values):
        return {"tauh": self.tauh(v), "hinf": self.hinf(v)}

    def state_defaults(self, v, values):
        return {"h": 1 / (1 + exp((v + 55) / 11.5))}


class s(S):
    S.DERIVED_BUFFER("q10")
    S.STATE("s")
    S.DERIVATIVE("s' = (sinf - s) / taus")
    S.ASSIGNED("sinf", "taus")
    S.GLOBAL_SIGNED(aq10=3.0)

    def derive_buffers(self):
        return {"q10": self.aq10 ** ((self.celsius - 22.0) / 10.0)}
    
    def taus(self, v):
        taus = 1 / (exp((v - 195) / 29) + exp(-1 * (v + 193) / 12.4)) + 1165 / (
            1 + exp(-1 * (v + 40) / 30)
        )
        return taus / self.q10
    
    def sinf(self, v):
        return 0.03 + 0.97 / (1 + exp((v + 79) / 6.87))

    def assigned_values(self, v, values):
        return {"taus": self.taus(v), "sinf": self.sinf(v)}

    def state_defaults(self, v, values):
        return {"s": 0.03 + 0.97 / (1 + exp((v + 79) / 6.87))}


class nav9(M):
    M.STATE_BUNDLE(m, h, s)
    M.GLOBAL_SIGNED(gbar=0.001)
    M.USEION("na", read=["ena"], write=["ina"])

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * self.s * (v - self.ena)


class nav9_augmented(nav9):
    nav9.GLOBAL_SIGNED(aug=1.0)

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * self.s * (v - self.ena) * self.aug
