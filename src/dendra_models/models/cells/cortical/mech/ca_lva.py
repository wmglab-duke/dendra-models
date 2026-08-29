from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class mh(S):
    S.DERIVED_BUFFER("q10")
    S.STATE("m", "h")
    S.DERIVATIVE(
        "m' = (minf - m) / taum",
        "h' = (hinf - h) / tauh",
    )
    S.ASSIGNED("minf", "taum", "hinf", "tauh")

    def derive_buffers(self):
        return {"q10": 2.3 ** ((self.celsius - 21.0) / 10.0)}

    def assigned_values(self, v, values):
        v = v + 10.0
        q10 = self.q10
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

    def state_defaults(self, v, values):
        states = self.assigned_values(v, values)
        return {
            "m": states["minf"],
            "h": states["hinf"],
        }


class ca_lva(M):
    M.STATE_BUNDLE(mh)
    M.USEION("ca", read=["eca"], write=["ica"])

    M.RANGEP(gbar=0.00001)

    def ica(self, v):
        return self.gbar * self.m**2 * self.h * (v - self.eca)
