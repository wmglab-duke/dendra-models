from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S
from axonml.models.mechanisms.ops import *


class p(S):
    has_q10 = True

    S.STATE("p")
    S.GLOBAL(
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

    def calc_q10(self):
        return self.q10ka ** ((self.q10TempA - self.celsius) / self.q10TempB)

    def breakpoint(self, v):
        taup = self.q10() * (
            self.A_taup * exp(-((self.B_taup) ** 2) * (v - self.Vpp) ** 2) + self.C_taup
        )
        pinf = 1.0 / (1.0 + exp((v + self.V0p5p + self.shiftka) / self.S0p5p))
        return {"taup": taup, "pinf": pinf}

    def inf(self, v):
        return {"p": self.breakpoint(v)["pinf"]}


class q(S):
    has_q10 = True
    S.STATE("q")
    S.GLOBAL(
        shiftka=3.0,
        V0p5q=58.0,
        S0p5q=7.9,
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

    def calc_q10(self):
        return self.q10ka ** ((self.q10TempA - self.celsius) / self.q10TempB)

    def breakpoint(self, v):
        tauq = self.q10() * (
            self.A_tauq * exp(-((self.B_tauq) ** 2) * (v - self.Vpq) ** 2) + self.C_tauq
        )
        qinf = 1.0 / (1.0 + exp((v + self.V0p5q + self.shiftka) / self.S0p5q))
        return {"tauq": tauq, "qinf": qinf}

    def inf(self, v):
        return {"q": self.breakpoint(v)["qinf"]}


class ka(M):
    M.STATE(p, q)
    M.GLOBAL(gbar=0.000141471)
    M.USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gbar * self.p**3 * self.q * (v - self.ek)
