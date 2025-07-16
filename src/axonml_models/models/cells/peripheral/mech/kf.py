# The steady state curves are collected from Winkelman 2005
# The time constant is from Gold 1996 and Safron 1996


from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S
from axonml.models.mechanisms.ops import *


class h(S):
    has_q10 = True

    S.STATE("h")
    S.GLOBAL(aq10=3.3, bq10=23, cq10=10, vhh=-49.9, kh=4.6, shift=-15.0)
    S.DERIVATIVE("h' = (hinf - h) / tauh")
    S.ASSIGNED("hinf", "tauh")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.celsius - self.bq10) / self.cq10))

    def breakpoint(self, v):
        hinf = sigmoid((v - self.vhh + self.shift) / -self.kh)
        tauh = 20 + 50 * exp(-((v + 40) ** 2) / (2 * 40**2))
        tauh = self.q10() * torch.where(tauh < 5, 5.0, tauh)
        return {"hinf": hinf, "tauh": tauh}

    def inf(self, v):
        return {"h": sigmoid((v - self.vhh + self.shift) / -self.kh)}


class m(S):
    has_q10 = True

    S.STATE("m")
    S.GLOBAL(aq10=3.3, bq10=23.0, cq10=10.0, vhm=-5.4, km=16.4, shift=-15.0)
    S.DERIVATIVE("m' = (minf - m) / taum")
    S.ASSIGNED("minf", "taum")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.celsius - self.bq10) / self.cq10))

    def breakpoint(self, v):
        minf = sigmoid((v - self.vhm + self.shift) / self.km) ** 4
        taum = self.q10() * (0.25 + 10.04 * exp(-((v + 24.67) ** 2) / (2 * 34.8**2)))
        return {"minf": minf, "taum": taum}

    def inf(self, v):
        return {"m": sigmoid((v - self.vhm + self.shift) / self.km) ** 4}


class kf(M):
    M.STATE(m, h)
    M.GLOBAL(gbar=0.0001)
    M.USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gbar * self.m * self.h * (v - self.ek)
