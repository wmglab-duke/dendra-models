# Borg-Graham type KDR channel; Borg-Graham 1987

from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S
from axonml.models.mechanisms.ops import *


class l(S):
    has_q10 = True

    S.STATE("l")

    S.GLOBAL(
        zetal=2.0,
        gml=1.0,
        vhalfl=-61.0,
        a0l=0.001,
        aq10=3.0,
        bq10=30.0,
        cq10=10.0,
    )

    S.DERIVATIVE("l' = (linf - l) / taul")
    S.ASSIGNED("linf", "taul")

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - self.bq10) / self.cq10)

    def alpha(self, v):
        return exp(
            1e-3
            * self.zetal
            * (v - self.vhalfl)
            * 9.648e4
            / (8.315 * (273.16 + self.celsius))
        )

    def beta(self, v):
        return exp(
            1e-3
            * self.zetal
            * self.gml
            * (v - self.vhalfl)
            * 9.648e4
            / (8.315 * (273.16 + self.celsius))
        )

    def breakpoint(self, v, states):
        a = self.alpha(v)
        b = self.beta(v)
        al = 1 + a
        linf = 1 / al
        taul = b / (self.q10() * self.a0l * al)
        return {"taul": taul, "linf": linf}

    def inf(self, v):
        return {"l": 1 / (1 + self.alpha(v))}


class n(S):
    has_q10 = True

    S.STATE("n")

    S.GLOBAL(
        zetan=-5.0,
        gmn=0.4,
        vhalfn=-32.0,
        a0n=0.03,
        aq10=3.0,
        bq10=30.0,
        cq10=10.0,
    )

    S.DERIVATIVE("n' = (ninf - n) / taun")
    S.ASSIGNED("ninf", "taun")

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - self.bq10) / self.cq10)

    def alpha(self, v):
        return exp(
            1e-3
            * self.zetan
            * (v - self.vhalfn)
            * 9.648e4
            / (8.315 * (273.16 + self.celsius))
        )

    def beta(self, v):
        return exp(
            1e-3
            * self.zetan
            * self.gmn
            * (v - self.vhalfn)
            * 9.648e4
            / (8.315 * (273.16 + self.celsius))
        )

    def breakpoint(self, v, states):
        a = self.alpha(v)
        b = self.beta(v)
        an = 1 + a
        ninf = 1 / an
        taun = b / (self.q10() * self.a0n * an)
        return {"taun": taun, "ninf": ninf}

    def inf(self, v):
        return {"n": 1 / (1 + self.alpha(v))}


class kdr(M):
    M.STATE(l, n)

    M.GLOBAL(gkbar=0.003)

    M.USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gkbar * self.n**3 * self.l * (v - self.ek)
