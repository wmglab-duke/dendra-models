from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S
from axonml.models.mechanisms.ops import *


class m(S):
    has_q10 = True
    S.STATE("m")
    S.DERIVATIVE("m' = (minf - m) / taum")
    S.ASSIGNED("minf", "taum")

    def calc_q10(self):
        return 2.3 ** ((self.celsius - 21.0) / 10.0)

    def alpha(self, v):
        return self.q10() * 3.3e-3 * exp(2.5 * 0.04 * (v - -35))

    def beta(self, v):
        return self.q10() * 3.3e-3 * exp(-2.5 * 0.04 * (v - -35))

    def breakpoint(self, v):
        q10 = self.q10()
        a = q10 * self.alpha(v)
        b = q10 * self.beta(v)
        taum = 1 / (a + b)
        minf = a * taum
        return {"taum": taum, "minf": minf}

    def inf(self, v):
        states = self.breakpoint(v)
        return {"m": states["minf"]}


class im(M):
    M.STATE(m)
    M.USEION("k", read=["ek"], write=["ik"])

    M.PARAMETER(gbar=0.0001)

    def ik(self, v):
        return self.gbar * self.m * (v - self.ek)
