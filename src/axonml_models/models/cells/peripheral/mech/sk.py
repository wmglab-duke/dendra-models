from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S
from axonml.models.mechanisms.ops import *


class n(S):
    has_q10 = True
    S.STATE("n")
    S.DERIVATIVE("n' = (ninf - n) / taun")
    S.ASSIGNED("ninf", "taun")
    S.GLOBAL(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - 22.0) / 10.0)

    def breakpoint(self, v, states):
        pca = log10(self.cai) - 3
        ninf = 1 / (1 + exp(-1 * (pca + 6.4) / 0.12))
        taun = -1 * pca / self.q10()
        return {"taun": taun, "ninf": ninf}

    def inf(self, v):
        pca = log10(self.cai) - 3
        return {"n": 1 / (1 + exp(-1 * (pca + 6.4) / 0.12))}


class sk(M):
    M.STATE(n)
    M.GLOBAL(gbar=0.0001)

    M.USEION("k", read=["ek"], write=["ik"])
    M.USEION("ca", read=["cai"])

    def ik(self, v):
        return self.gbar * self.n * (v - self.ek)
