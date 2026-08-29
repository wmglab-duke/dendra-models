from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class m(S):
    S.STATE("m")
    S.DERIVATIVE("m' = (minf - m) / taum")
    S.ASSIGNED("minf", "taum")

    def assigned_values(self, v, values):
        alpha = 0.001 * 6.43 * vtrap(v + 154.9, 11.9)
        beta = 0.001 * 193 * exp(v / 33.1)
        taum = 1 / (alpha + beta)
        minf = alpha * taum
        return {"taum": taum, "minf": minf}

    def state_defaults(self, v, values):
        return {"m": self.assigned_values(v, values)["minf"]}


class ih(M):
    M.STATE_BUNDLE(m)

    M.RANGEP(gbar=0.00001)
    M.GLOBAL_SIGNED(ehcn=-45.0)
    M.NONSPECIFIC_CURRENT("ihcn")

    def ihcn(self, v):
        return self.gbar * self.m * (v - self.ehcn)
