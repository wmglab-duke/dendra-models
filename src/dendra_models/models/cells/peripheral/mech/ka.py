from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class p(S):
    S.DERIVED_BUFFER("q10")

    S.STATE("p")
    S.GLOBAL_SIGNED(
        shiftka=3.0,
        V0p5p=28.0,
        S0p5p=-28.0,
        A_taup=5.0,
        B_taup=0.022,
        C_taup=2.5,
        Vpp=-65.0,
        q10ka=1.93,
        q10TempA=22.85,
        q10TempB=10.0,
    )
    S.DERIVATIVE("p' = (pinf - p) / taup")
    S.ASSIGNED("pinf", "taup")

    def derive_buffers(self):
        return {"q10": self.q10ka ** ((self.q10TempA - self.celsius) / self.q10TempB)}

    def assigned_values(self, v, values):
        taup = self.q10 * (
            self.A_taup * exp(-((self.B_taup) ** 2) * (v - self.Vpp) ** 2) + self.C_taup
        )
        pinf = 1.0 / (1.0 + exp((v + self.V0p5p + self.shiftka) / self.S0p5p))
        return {"taup": taup, "pinf": pinf}

    def state_defaults(self, v, values):
        return {"p": self.assigned_values(v, values)["pinf"]}


class q(S):
    S.DERIVED_BUFFER("q10")
    S.STATE("q")
    S.GLOBAL_SIGNED(
        shiftka=3.0,
        V0p5q=58.0,
        S0p5q=7.0,
        A_tauq=100.0,
        B_tauq=0.035,
        C_tauq=10.5,
        Vpq=-30.0,
        q10ka=1.93,
        q10TempA=22.85,
        q10TempB=10.0,
    )

    S.DERIVATIVE("q' = (qinf - q) / tauq")
    S.ASSIGNED("qinf", "tauq")

    def derive_buffers(self):
        return {"q10": self.q10ka ** ((self.q10TempA - self.celsius) / self.q10TempB)}

    def assigned_values(self, v, values):
        tauq = self.q10 * (
            self.A_tauq * exp(-((self.B_tauq) ** 2) * (v - self.Vpq) ** 2) + self.C_tauq
        )
        qinf = 1.0 / (1.0 + exp((v + self.V0p5q + self.shiftka) / self.S0p5q))
        return {"tauq": tauq, "qinf": qinf}

    def state_defaults(self, v, values):
        return {"q": self.assigned_values(v, values)["qinf"]}


class ka(M):
    M.STATE_BUNDLE(p, q)
    M.GLOBAL_SIGNED(gbar=0.000141471)
    M.USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gbar * self.p**3 * self.q * (v - self.ek)
