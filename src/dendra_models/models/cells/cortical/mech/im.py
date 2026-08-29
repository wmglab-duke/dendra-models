from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class m(S):
    S.DERIVED_BUFFER("q10")
    S.STATE("m")
    S.DERIVATIVE("m' = (minf - m) / taum")
    S.ASSIGNED("minf", "taum")

    def derive_buffers(self):
        return {"q10": 2.3 ** ((self.celsius - 21.0) / 10.0)}

    def alpha(self, v):
        return self.q10 * 3.3e-3 * exp(2.5 * 0.04 * (v - -35))

    def beta(self, v):
        return self.q10 * 3.3e-3 * exp(-2.5 * 0.04 * (v - -35))

    def assigned_values(self, v, values):
        a = self.alpha(v)
        b = self.beta(v)
        taum = 1 / (a + b)
        minf = a * taum
        return {"taum": taum, "minf": minf}

    def state_defaults(self, v, values):
        states = self.assigned_values(v, values)
        return {"m": states["minf"]}


class im(M):
    M.STATE_BUNDLE(m)
    M.USEION("k", read=["ek"], write=["ik"])

    M.RANGEP(gbar=0.00001)

    def ik(self, v):
        return self.gbar * self.m * (v - self.ek)
