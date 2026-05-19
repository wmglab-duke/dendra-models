from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class n(S):
    has_q10 = True

    S.STATE("n")
    S.GLOBAL(
        V0p5n=14.62,
        S0p5n=-18.38,
        A_alphan=0.001265,
        B_alphan=14.273,
        C_alphan=-10.0,
        A_betan=0.125,
        B_betan=55.0,
        C_betan=-2.5,
        q10kdn=1.40,
        q10TempA=22.85,
        q10TempB=10.0,
        shiftkd=3.0,
    )

    S.DERIVATIVE("n' = (ninf - n) / ntau")
    S.ASSIGNED("ninf", "ntau")

    def calc_q10(self):
        return self.q10kdn ** ((self.q10TempA - self.celsius) / self.q10TempB)

    def alpha(self, v):
        x = v + self.B_alphan
        return -self.A_alphan * exprelr(x, self.C_alphan)

    def beta(self, v):
        return self.A_betan * exp((v + self.B_betan) / self.C_betan)

    def breakpoint(self, v, states):
        a = self.alpha(v)
        b = self.beta(v)
        ntau = self.q10() * (1.0 + (1.0 / (a + b)))
        ninf = sigmoid((-v - self.V0p5n - self.shiftkd) / self.S0p5n)
        return {"ntau": ntau, "ninf": ninf}

    def inf(self, v):
        return {"n": self.breakpoint(v, None)["ninf"]}


class kd(M):
    M.STATE(n)
    M.GLOBAL(gbar=0.000180376)
    M.USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gbar * self.n * (v - self.ek)
