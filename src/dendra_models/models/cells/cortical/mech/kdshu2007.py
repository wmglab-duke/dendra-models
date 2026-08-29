# K-D current for prefrontal cortical neuron ------Yuguo Yu  2007

from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class mh(S):
    S.STATE("m", "h")
    S.GLOBAL_SIGNED(aq10=2.3, bq10=22.0, cq10=10.0, vhalfm=-43.0, vhalfh=-67.0, km=8.0, kh=7.3)
    S.PARAMETER(taum=0.6, tauh=1500.0)

    S.DERIVATIVE("m' = (minf - m) / taum", "h' = (hinf - h) / tauh")
    S.ASSIGNED("minf", "hinf")

    def assigned_values(self, v, values):
        minf = 1 - 1 / (1 + exp((v - self.vhalfm) / self.km))
        hinf = 1 / (1 + exp((v - self.vhalfh) / self.kh))
        return {"minf": minf, "hinf": hinf}

    def state_defaults(self, v, values):
        states = self.assigned_values(v, values)
        return {"m": states["minf"], "h": states["hinf"]}


class kdshu2007(M):
    M.STATE_BUNDLE(mh)
    M.USEION("k", write=["ik"])
    M.RANGE(ek=-100.0)
    M.RANGE(gbar=0.1)

    def ik(self, v):
        return self.gbar * self.m * self.h * (v - self.ek)
