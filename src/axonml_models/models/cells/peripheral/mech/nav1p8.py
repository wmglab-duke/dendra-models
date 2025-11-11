# m and h are from Sheets, 2007
# s and u are from Delmas


from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S
from axonml.models.mechanisms.ops import *


class m(S):
    has_q10 = True

    S.STATE("m")
    S.GLOBAL(aq10=2.5, bq10=22.0, cq10=10.0)
    S.DERIVATIVE("m' = (minf - m) / taum")
    S.ASSIGNED("minf", "taum")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.celsius - self.bq10) / self.cq10))

    def breakpoint(self, v, states):
        a = 2.85 - 2.839 * sigmoid((1.159 - v) / 13.95)
        b = 7.6205 * sigmoid((-46.463 - v) / 8.8289)
        s = 1 / (a + b)
        taum = self.q10() * s
        minf = a * s
        return {"minf": minf, "taum": taum}

    def inf(self, v):
        a = 2.85 - 2.839 * sigmoid((1.159 - v) / 13.95)
        b = 7.6205 * sigmoid((-46.463 - v) / 8.8289)
        return {"m": a / (a + b)}


class h(S):
    has_q10 = True

    S.STATE("h")
    S.GLOBAL(aq10=2.5, bq10=22.0, cq10=10.0)
    S.DERIVATIVE("h' = (hinf - h) / tauh")
    S.ASSIGNED("hinf", "tauh")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.celsius - self.bq10) / self.cq10))

    def breakpoint(self, v, states):
        hinf = sigmoid((-32.2 - v) / 4.0)
        tauh = self.q10() * (1.218 + 42.043 * exp(-((v + 38.1) ** 2) / (2 * 15.19**2)))
        return {"hinf": hinf, "tauh": tauh}

    def inf(self, v):
        return {"h": sigmoid((-32.2 - v) / 4.0)}


class s(S):
    has_q10 = True

    S.STATE("s")
    S.GLOBAL(aq10=2.5, bq10=22.0, cq10=10.0)
    S.DERIVATIVE("s' = (sinf - s) / taus")
    S.ASSIGNED("sinf", "taus")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.celsius - self.bq10) / self.cq10))

    def alpha(self, v):
        return 0.001 * 5.4203 * sigmoid((-79.816 - v) / 16.269)

    def beta(self, v):
        return 0.001 * 5.0757 * sigmoid((v + 15.968) / 11.542)

    def breakpoint(self, v, states):
        a = self.alpha(v)
        b = self.beta(v)
        taus = self.q10() / (a + b)
        sinf = sigmoid((-45.0 - v) / 8.0)
        return {"sinf": sinf, "taus": taus}

    def inf(self, v):
        return {"s": sigmoid((-45.0 - v) / 8.0)}


class u(S):
    has_q10 = True

    S.STATE("u")
    S.GLOBAL(aq10=2.5, bq10=22.0, cq10=10.0)
    S.DERIVATIVE("u' = (uinf - u) / tauu")
    S.ASSIGNED("uinf", "tauu")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.celsius - self.bq10) / self.cq10))

    def alpha(self, v):
        return 0.0002 * 2.0434 * sigmoid((-67.499 - v) / 19.51)

    def beta(self, v):
        return 0.0002 * 1.9952 * sigmoid((v + 30.963) / 14.792)

    def breakpoint(self, v, states):
        a = self.alpha(v)
        b = self.beta(v)
        tauu = self.q10() / (a + b)
        uinf = sigmoid((-51.0 - v) / 8.0)
        return {"uinf": uinf, "tauu": tauu}

    def inf(self, v):
        return {"u": sigmoid((-51.0 - v) / 8.0)}


class nav1p8(M):
    M.STATE(m, h, s, u)
    M.GLOBAL(gbar=0.0)
    M.USEION("na", read=["ena"], write=["ina"])

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * self.s * self.u * (v - self.ena)
