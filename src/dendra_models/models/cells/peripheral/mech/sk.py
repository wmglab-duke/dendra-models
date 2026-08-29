from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class n(S):
    S.DERIVED_BUFFER("q10")
    S.STATE("n")
    S.DERIVATIVE("n' = (ninf - n) / taun")
    S.ASSIGNED("ninf", "taun")
    S.GLOBAL_SIGNED(aq10=3.0)

    def derive_buffers(self):
        return {"q10": self.aq10 ** ((self.celsius - 22.0) / 10.0)}

    def assigned_values(self, v, values):
        pca = log10(self.cai) - 3
        ninf = 1 / (1 + exp(-1 * (pca + 6.4) / 0.12))
        taun = -1 * pca / self.q10
        return {"taun": taun, "ninf": ninf}

    def state_defaults(self, v, values):
        pca = log10(self.cai) - 3
        return {"n": 1 / (1 + exp(-1 * (pca + 6.4) / 0.12))}


class sk(M):
    M.STATE_BUNDLE(n)
    M.GLOBAL_SIGNED(gbar=0.0001)

    M.USEION("k", read=["ek"], write=["ik"])
    M.USEION("ca", read=["cai"])

    def ik(self, v):
        return self.gbar * self.n * (v - self.ek)


class sk_augmented(sk):
    sk.GLOBAL_SIGNED(aug=1.0)

    def ik(self, v):
        return self.gbar * self.n * (v - self.ek) * self.aug
