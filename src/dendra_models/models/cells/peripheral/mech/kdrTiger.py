# Sheets 2007


from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import sigmoid, exp, torch


class n(S):
    S.DERIVED_BUFFER("q10")

    S.STATE("n")
    S.GLOBAL_SIGNED(aq10=3.3, bq10=22.0, cq10=10.0, k1=15.4, vh=35.0)
    S.DERIVATIVE("n' = (ninf - n) / ntau")
    S.ASSIGNED("ninf", "ntau")

    def derive_buffers(self):
        return {"q10": 1.0 / (self.aq10 ** ((self.celsius - self.bq10) / self.cq10))}

    def assigned_values(self, v, values):
        ninf = sigmoid((v + self.vh - 10.0) / self.k1)
        ntau_a = 0.16 + 0.8 * exp(-0.0267 * (v + 11.0))
        ntau_b = 1000 * (
            0.000688 + 1 / (exp((v + 75.2) / 6.5) + exp((v - 131.5) / -34.8))
        )
        ntau = self.q10 * torch.where(v < -31.0, ntau_b, ntau_a)
        return {"ninf": ninf, "ntau": ntau}

    def state_defaults(self, v, values):
        return {"n": sigmoid((v + self.vh - 10.0) / self.k1)}


class kdrTiger(M):
    M.STATE_BUNDLE(n)
    M.GLOBAL_SIGNED(gbar=0.0001)
    M.USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gbar * self.n**4 * (v - self.ek)
