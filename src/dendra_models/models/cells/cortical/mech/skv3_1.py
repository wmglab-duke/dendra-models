from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class m(S):
    S.STATE("m")
    S.DERIVATIVE("m' = (minf - m) / taum")
    S.ASSIGNED("minf", "taum")

    def assigned_values(self, v, values):
        taum = 0.2 * 20.000 / (1 + exp(((v - (-46.560)) / (-44.140))))
        minf = 1 / (1 + exp(((v - (18.700)) / (-9.700))))
        return {"taum": taum, "minf": minf}

    def state_defaults(self, v, values):
        return {"m": self.assigned_values(v, values)["minf"]}


class skv3_1(M):
    M.STATE_BUNDLE(m)
    M.RANGEP(gbar=0.00001)
    M.USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gbar * self.m * (v - self.ek)
