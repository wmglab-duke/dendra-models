# calcium-activated potassium current in Schild 1994

from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class c(S):
    S.DERIVED_BUFFER("q10")

    S.STATE("c")
    S.GLOBAL_SIGNED(
        Q10kcac=2.30,
        Q10TempA=22.85,
        Q10TempB=10.0,
        A_alphac=750.0,
        B_alphac=-10.0,
        C_alphac=12.0,
        A_betac=0.05,
        B_betac=-10.0,
        C_betac=-60.0,
    )

    S.DERIVATIVE("c' = (cinf - c) / tauc")
    S.ASSIGNED("cinf", "tauc")

    def derive_buffers(self):
        return {"q10": self.Q10kcac ** ((self.Q10TempA - self.celsius) / self.Q10TempB)}

    def alpha(self, v):
        return self.A_alphac * self.cai * safe_exp((v + self.B_alphac) / self.C_alphac)

    def beta(self, v):
        return self.A_betac * safe_exp((v + self.B_betac) / self.C_betac)

    def assigned_values(self, v, values):
        a = self.alpha(v)
        b = self.beta(v)
        s = 1 / (a + b)
        cinf = a * s
        tauc = self.q10 * 4.5 * s
        return {"tauc": tauc, "cinf": cinf}

    def state_defaults(self, v, values):
        return {"c": self.assigned_values(v, values)["cinf"]}


class kca(M):
    M.STATE_BUNDLE(c)
    M.GLOBAL_SIGNED(gbar=0.000141471)
    M.USEION("k", read=["ek"], write=["ik"])
    M.USEION("ca", read=["cai"])

    def ik(self, v):
        return self.gbar * self.c * (v - self.ek)
