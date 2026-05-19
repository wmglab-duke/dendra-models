from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class mh(S):
    has_q10 = True
    S.STATE("m", "h")
    S.DERIVATIVE(
        "m' = (minf - m) / taum",
        "h' = (hinf - h) / tauh",
    )
    S.ASSIGNED("minf", "taum", "hinf", "tauh")

    def calc_q10(self):
        return 2.3 ** ((self.celsius - 21.0) / 10.0)

    def breakpoint(self, v, states):
        v = v + 10.0
        q10 = self.q10()
        minf = 1.0000 / (1 + exp((v - -30.000) / -6))
        taum = (5.0000 + 20.0000 / (1 + exp((v - -25.000) / 5))) / q10
        hinf = 1.0000 / (1 + exp((v - -80.000) / 6.4))
        tauh = (20.0000 + 50.0000 / (1 + exp((v - -40.000) / 7))) / q10
        return {
            "taum": taum,
            "minf": minf,
            "tauh": tauh,
            "hinf": hinf,
        }

    def inf(self, v):
        states = self.breakpoint(v, None)
        return {
            "m": states["minf"],
            "h": states["hinf"],
        }


class ca_lva(M):
    M.STATE(mh)
    M.USEION("ca", read=["eca"], write=["ica"])

    M.RANGEP(gbar=0.0001)

    def ica(self, v):
        return self.gbar * self.m**2 * self.h * (v - self.eca)
