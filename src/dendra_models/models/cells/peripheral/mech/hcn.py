from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class m(S):
    has_q10 = True
    S.STATE("m")
    S.DERIVATIVE("m' = (minf - m) / taum")
    S.ASSIGNED("minf", "taum")
    S.GLOBAL_SIGNED(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - 22.0) / 10.0)
    
    def minf(self, v):
        return (1 / (1 + exp((v + 97) / 7.35))) ** (1 / 3)
    
    def taum(self, v):
        taum = 0.5 / (exp((v - 42) / 11.8) + exp(-1 * (v + 498) / 66.6))
        return taum / self.q10()

    def breakpoint(self, v, states):
        return {"taum": self.taum(v), "minf": self.minf(v)}
    
    def inf(self, v):
        return {"m": self.breakpoint(v, None)["minf"]}


class n(S):
    has_q10 = True
    S.STATE("n")
    S.DERIVATIVE("n' = (ninf - n) / taun")
    S.ASSIGNED("ninf", "taun")
    S.GLOBAL_SIGNED(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - 22.0) / 10.0)
    
    def ninf(self, v):
        return (1 / (1 + exp((v + 94) / 8.9))) ** (1 / 3)
    
    def taun(self, v):
        taun = 0.5 / (exp((v + 25) / 4.1) + exp(-1 * (v + 356) / 32))
        return taun / self.q10()

    def breakpoint(self, v, states):
        return {"taun": self.taun(v), "ninf": self.ninf(v)}

    def inf(self, v):
        return {"n": self.breakpoint(v, None)["ninf"]}


class hcn(M):
    M.STATE(m, n)
    M.GLOBAL_SIGNED(gbar=0.0001)
    M.GLOBAL_SIGNED(ekna=-30.0)

    M.USEION("k", read=["ek"], write=["ik"])
    M.USEION("na", read=["ena"], write=["ina"])

    def _g(self):
        return self.gbar * (0.25 * self.n**3 + 0.75 * self.m**3)

    def ik(self, v):
        g = self._g()
        return (self.ekna - self.ena) * g * (v - self.ek) / (self.ek - self.ena)

    def ik_with_conductance(self, v):
        g = self._g()
        conductance = (self.ekna - self.ena) * g / (self.ek - self.ena)
        return conductance * (v - self.ek), conductance

    def ina(self, v):
        g = self._g()
        return (self.ekna - self.ek) * g * (v - self.ena) / (self.ena - self.ek)

    def ina_with_conductance(self, v):
        g = self._g()
        conductance = (self.ekna - self.ek) * g / (self.ena - self.ek)
        return conductance * (v - self.ena), conductance


class hcn_augmented(hcn):
    hcn.GLOBAL_SIGNED(ik_aug=1.0, ina_aug=1.0)

    def _g(self):
        return self.gbar * (0.25 * self.n**3 + 0.75 * self.m**3)

    def ik(self, v):
        g = self._g()
        return (
            (self.ekna - self.ena)
            * g
            * self.ik_aug
            * (v - self.ek)
            / (self.ek - self.ena)
        )

    def ik_with_conductance(self, v):
        g = self._g()
        conductance = (
            (self.ekna - self.ena)
            * g
            * self.ik_aug
            / (self.ek - self.ena)
        )
        return conductance * (v - self.ek), conductance

    def ina(self, v):
        g = self._g()
        return (
            (self.ekna - self.ek)
            * g
            * self.ina_aug
            * (v - self.ena)
            / (self.ena - self.ek)
        )

    def ina_with_conductance(self, v):
        g = self._g()
        conductance = (
            (self.ekna - self.ek)
            * g
            * self.ina_aug
            / (self.ena - self.ek)
        )
        return conductance * (v - self.ena), conductance
