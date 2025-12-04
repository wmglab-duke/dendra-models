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

    def breakpoint(self, v, states):
        minf = (1 / (1 + exp((v + 97) / 7.35))) ** (1 / 3)
        taum = 0.5 / (exp((v - 42) / 11.8) + exp(-1 * (v + 498) / 66.6))
        taum = taum / self.q10()
        return {"taum": taum, "minf": minf}

    def inf(self, v):
        return {"m": self.breakpoint(v, None)["minf"]}


class n(S):
    has_q10 = True
    S.STATE("n")
    S.DERIVATIVE("n' = (ninf - n) / taun")
    S.ASSIGNED("ninf", "taun")
    S.GLOBAL(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - 22.0) / 10.0)

    def breakpoint(self, v, states):
        ninf = (1 / (1 + exp((v + 94) / 8.9))) ** (1 / 3)
        taun = 0.5 / (exp((v + 25) / 4.1) + exp(-1 * (v + 356) / 32))
        taun = taun / self.q10()
        return {"taun": taun, "ninf": ninf}

    def inf(self, v):
        return {"n": self.breakpoint(v, None)["ninf"]}


class hcn(M):
    M.STATE(m, n)
    M.GLOBAL(gbar=0.0001, ekna=-30.0)

    M.USEION("k", read=["ek"], write=["ik"])
    M.USEION("na", read=["ena"], write=["ina"])

    def ik(self, v):
        g = self.gbar * (0.25 * self.n + 0.75 * self.m)
        return (self.ekna - self.ena) * g * (v - self.ek) / (self.ek - self.ena)

    def ina(self, v):
        g = self.gbar * (0.25 * self.n + 0.75 * self.m)
        return (self.ekna - self.ek) * g * (v - self.ena) / (self.ena - self.ek)
    

class hcn_augmented(hcn):
    hcn.GLOBAL(ik_aug=1.0, ina_aug=1.0)

    def ik(self, v):
        g = self.gbar * (0.25 * self.n + 0.75 * self.m)
        return (self.ekna - self.ena) * g * self.ik_aug * (v - self.ek) / (self.ek - self.ena)
    
    def ina(self, v):
        g = self.gbar * (0.25 * self.n + 0.75 * self.m)
        return (self.ekna - self.ek) * g * self.ina_aug * (v - self.ena) / (self.ena - self.ek)
    