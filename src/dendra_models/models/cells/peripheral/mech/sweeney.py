# Sweeney Channel
# Sweeney channel fast sodium channel for myelinated axon
# set for a resting potential of -80 mV


from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class m(S):
    S.STATE("m")
    S.GLOBAL_SIGNED(amA=49.0, amB=126.0, amC=0.363, amD=5.3, bmA=56.2, bmB=4.17)
    S.DERIVATIVE("m' = (minf - m) / taum")
    S.ASSIGNED("minf", "taum")

    def alpha(self, v):
        return (self.amB + self.amC * v) / (1 + exp(-(self.amA + v) / self.amD))

    def beta(self, v):
        return (
            (1 / exp((v + self.bmA) / self.bmB))
            * (self.amB + self.amC * v)
            / (1 + exp(-(self.amA + v) / self.amD))
        )

    def assigned_values(self, v, values):
        a = self.alpha(v)
        b = self.beta(v)
        taum = 1 / (a + b)
        minf = a * taum
        return {"taum": taum, "minf": minf}

    def state_defaults(self, v, values):
        return {"m": self.assigned_values(v, values)["minf"]}


class h(S):
    S.STATE("h")
    S.GLOBAL_SIGNED(ahA=56.0, ahB=15.6, bhA=10.0, bhB=74.5, bhC=5.0)
    S.DERIVATIVE("h' = (hinf - h) / tauh")
    S.ASSIGNED("hinf", "tauh")

    def alpha(self, v):
        return (
            (1 / exp((v + self.bhB) / self.bhC))
            * self.ahB
            / (1 + exp(-(v + self.ahA) / self.bhA))
        )

    def beta(self, v):
        return self.ahB / (1 + exp(-(v + self.ahA) / self.bhA))

    def assigned_values(self, v, values):
        a = self.alpha(v)
        b = self.beta(v)
        tauh = 1 / (a + b)
        hinf = a * tauh
        return {"tauh": tauh, "hinf": hinf}

    def state_defaults(self, v, values):
        return {"h": self.assigned_values(v, values)["hinf"]}


class sweeney(M):
    M.STATE_BUNDLE(m, h)
    M.GLOBAL_SIGNED(gnabar=1.445, gl=0.128, el=-80.01, ena=35.64)

    M.USEION("na", write=["ina"])
    M.NONSPECIFIC_CURRENT("il")

    def ina(self, v):
        return self.gnabar * self.m**2 * self.h * (v - self.ena)

    def il(self, v):
        return self.gl * (v - self.el)
