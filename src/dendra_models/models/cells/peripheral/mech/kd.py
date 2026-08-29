from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class n(S):
    S.DERIVED_BUFFER("q10")

    S.STATE("n")
    S.GLOBAL_SIGNED(
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

    def derive_buffers(self):
        return {"q10": self.q10kdn ** ((self.q10TempA - self.celsius) / self.q10TempB)}

    def alpha(self, v):
        x = v + self.B_alphan
        return -self.A_alphan * exprelr(x, self.C_alphan)

    def beta(self, v):
        return self.A_betan * exp((v + self.B_betan) / self.C_betan)

    def assigned_values(self, v, values):
        a = self.alpha(v)
        b = self.beta(v)
        ntau = self.q10 * (1.0 + (1.0 / (a + b)))
        ninf = sigmoid((-v - self.V0p5n - self.shiftkd) / self.S0p5n)
        return {"ntau": ntau, "ninf": ninf}

    def state_defaults(self, v, values):
        return {"n": self.assigned_values(v, values)["ninf"]}


class kd(M):
    M.STATE_BUNDLE(n)
    M.GLOBAL_SIGNED(gbar=0.000180376)
    M.USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gbar * self.n * (v - self.ek)
