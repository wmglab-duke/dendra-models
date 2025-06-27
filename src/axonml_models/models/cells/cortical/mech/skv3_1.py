from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S
from axonml.models.mechanisms.ops import *


class m(S):
    S.STATE("m")
    S.DERIVATIVE("m' = (minf - m) / taum")
    S.ASSIGNED("minf", "taum")

    def breakpoint(self, v):
        taum = 0.2 * 20.000 / (1 + exp(((v - (-46.560)) / (-44.140))))
        minf = 1 / (1 + exp(((v - (18.700)) / (-9.700))))
        return {'taum': taum, 'minf': minf}

    def inf(self, v):
        return {'m': self.breakpoint(v)['minf']}


class skv3_1(M):
    M.STATE(m)
    M.PARAMETER(gbar=0.0001)
    M.USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gbar * self.m * (v - self.ek)
