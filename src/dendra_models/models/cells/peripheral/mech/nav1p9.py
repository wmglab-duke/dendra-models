# NaV1.9 Na+ current from Herzog, Cummins, and Waxman 2001 p1353
# This current is also called the ttx-rp current (the tetrodotoxin resistent persistant current)


from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class m(S):
    has_q10 = True
    S.STATE("m")
    S.GLOBAL(
        aq10=2.5,
        bq10=21.0,
        cq10=10.0,
        A_am=1.032,
        B_am=6.99,
        C_am=-14.87115,
        A_bm=5.79,
        B_bm=130.4,
        C_bm=22.9,
    )
    S.DERIVATIVE("m' = (minf - m) / taum")
    S.ASSIGNED("minf", "taum")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.celsius - self.bq10) / self.cq10))

    def alpha(self, v):
        return self.A_am * sigmoid((-self.B_am - v) / self.C_am)

    def beta(self, v):
        return self.A_bm * sigmoid((-self.B_bm - v) / self.C_bm)

    def breakpoint(self, v, states):
        a = self.alpha(v)
        b = self.beta(v)
        s = 1 / (a + b)
        minf = a * s
        taum = self.q10() * s
        return {"minf": minf, "taum": taum}

    def inf(self, v):
        states = self.breakpoint(v, None)
        return {"m": states["minf"]}


class h(S):
    has_q10 = True
    S.STATE("h")
    S.GLOBAL(
        aq10=2.5,
        bq10=21.0,
        cq10=10.0,
        A_ah=0.06435,
        B_ah=73.26415,
        C_ah=3.71928,
        A_bh=0.13496,
        B_bh=10.27853,
        C_bh=-9.09334,
    )
    S.DERIVATIVE("h' = (hinf - h) / tauh")
    S.ASSIGNED("hinf", "tauh")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.celsius - self.bq10) / self.cq10))

    def alpha(self, v):
        return self.A_ah * sigmoid((-self.B_ah - v) / self.C_ah)

    def beta(self, v):
        return self.A_bh * sigmoid((-self.B_bh - v) / self.C_bh)

    def breakpoint(self, v, states):
        a = self.alpha(v)
        b = self.beta(v)
        s = 1 / (a + b)
        hinf = a * s
        tauh = self.q10() * s
        return {"hinf": hinf, "tauh": tauh}

    def inf(self, v):
        states = self.breakpoint(v, None)
        return {"h": states["hinf"]}


class s(S):
    has_q10 = True
    S.STATE("s")
    S.GLOBAL(
        aq10=2.5,
        bq10=21.0,
        cq10=10.0,
        A_as=0.00000016,
        B_as=0.0,
        C_as=12.0,
        A_bs=0.0005,
        B_bs=32.0,
        C_bs=23.0,
    )
    S.DERIVATIVE("s' = (sinf - s) / taus")
    S.ASSIGNED("sinf", "taus")

    def calc_q10(self):
        return 1 / (self.aq10 ** ((self.celsius - self.bq10) / self.cq10))

    def alpha(self, v):
        return self.A_as * sigmoid((v + self.B_as) / self.C_as)

    def beta(self, v):
        return self.A_bs * sigmoid((v + self.B_bs) / self.C_bs)

    def breakpoint(self, v, states):
        a = self.alpha(v)
        b = self.beta(v)
        s = 1 / (a + b)
        sinf = a * s
        taus = self.q10() * s
        return {"sinf": sinf, "taus": taus}

    def inf(self, v):
        states = self.breakpoint(v, None)
        return {"s": states["sinf"]}


class nav1p9(M):
    M.STATE(m, h, s)
    M.GLOBAL(gbar=0.0)
    M.USEION("na", read=["ena"], write=["ina"])

    def ina(self, v):
        return self.gbar * self.m * self.h * self.s * (v - self.ena)
