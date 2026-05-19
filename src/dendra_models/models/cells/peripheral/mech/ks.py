# This is mainly the 7.3 channel.
# It is an inactivation potassium current
# The inactivation long time constant is based on the article Passmore 2003.
# The steady state inactivation and short time constant
# is from Maingret 2008 (which is based on Passmore 2003)


from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class s(S):
    has_q10 = True

    S.STATE("s")
    S.GLOBAL(aq10=3.3, bq10=21.0, cq10=10.0)
    S.DERIVATIVE("s' = (sinf - s) / taus")
    S.ASSIGNED("sinf", "taus")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.celsius - self.bq10) / self.cq10))

    def breakpoint(self, v, states):
        sinf = sigmoid((v + 30.0) / 6.0)
        taus = self.q10() * torch.where(v < -60.0, 219.0, (13.0 * v + 1000.0))
        return {"sinf": sinf, "taus": taus}

    def inf(self, v):
        return {"s": sigmoid((v + 30.0) / 6.0)}


class f(S):
    has_q10 = True

    S.STATE("f")
    S.GLOBAL(aq10=3.3, bq10=21.0, cq10=10.0)
    S.DERIVATIVE("f' = (finf - f) / tauf")
    S.ASSIGNED("finf", "tauf")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.celsius - self.bq10) / self.cq10))

    def alpha(self, v):
        return 0.00395 * exp((v + 30.0) / 40.0)

    def beta(self, v):
        return 0.00395 * exp(-(v + 30.0) / 20.0)

    def breakpoint(self, v, states):
        a = self.alpha(v)
        b = self.beta(v)
        finf = sigmoid((v + 30.0) / 6.0)
        tauf = self.q10() / (a + b)
        return {"finf": finf, "tauf": tauf}

    def inf(self, v):
        return {"f": sigmoid((v + 30.0) / 6.0)}


class ks(M):
    M.STATE(s, f)
    M.GLOBAL(gbar=0.0001)
    M.USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gbar * (0.25 * self.s + 0.75 * self.f) * (v - self.ek)
