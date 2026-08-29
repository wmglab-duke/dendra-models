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
        return (1 / (1 + exp(-1 * (v + 25) / 12))) ** (1 / 3)
    
    def taum(self, v):
        taum = (
            0.6
            + 2748 / (exp((v + 128) / 14.5) + exp(-1 * (v + 10) / 8))
            + 1.7 / (1 + exp((v + 8.5) / 10.65))
        ) / 2
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
        return 0.073 + 0.924 / (1 + exp((v + 47) / 4.75))
    
    def tauh(self, v):
        tauh = 35 + 11.22 / (exp((v + 21.4) / 9.48) + exp(-1 * (v + 155.3) / 16.4))
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
        return (
            0.415
            + 0.576 / (1 + exp((v + 44.5) / 5.93))
            + 0.103 / (1 + exp(-1 * (v) / 18.37))
        )
    
    def taus(self, v):
        taus = 1000 * (
            3.97 / (exp((v + 35.1) / 9.97) + exp(-1 * (v + 83.3) / 18))
            + 2.5 / (1 + exp(-1 * (v + 27.3) / 7.11))
        )
        return taus / self.q10

    def assigned_values(self, v, values):
        return {"taus": self.taus(v), "sinf": self.sinf(v)}

    def state_defaults(self, v, values):
        return {"s": self.assigned_values(v, values)["sinf"]}


class ka14(M):
    M.STATE_BUNDLE(m, h, s)
    M.GLOBAL_SIGNED(gbar=0.0001)

    M.USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gbar * self.m**3 * self.h * self.s * (v - self.ek)
    

class ka14_augmented(ka14):
    ka14.GLOBAL_SIGNED(aug=1.0)

    def ik(self, v):
        return self.gbar * self.m**3 * self.h * self.s * (v - self.ek) * self.aug
