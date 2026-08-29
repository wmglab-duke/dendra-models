# transient ttx-sensitive Na+ current from Sheets et al. 2007

from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class m(S):
    S.DERIVED_BUFFER("q10")

    S.STATE("m")
    S.GLOBAL_SIGNED(
        aq10=2.5,
        bq10=21.0,
        cq10=10.0,
        A_am=15.5,
        B_am=-5.0,
        C_am=-12.08,
        A_bm=35.2,
        B_bm=72.7,
        C_bm=16.7,
    )

    S.DERIVATIVE("m' = (minf - m) / taum")
    S.ASSIGNED("minf", "taum")

    def derive_buffers(self):
        return {"q10": 1 / (self.aq10 ** ((self.celsius - self.bq10) / self.cq10))}

    def alpha(self, v):
        return self.A_am * sigmoid((-self.B_am - v) / self.C_am)

    def beta(self, v):
        return self.A_bm * sigmoid((-self.B_bm - v) / self.C_bm)

    def assigned_values(self, v, values):
        a = self.alpha(v)
        b = self.beta(v)
        s = 1 / (a + b)
        taum = self.q10 * s
        minf = a * s
        return {"minf": minf, "taum": taum}

    def state_defaults(self, v, values):
        states = self.assigned_values(v, values)
        return {"m": states["minf"]}


class h(S):
    S.DERIVED_BUFFER("q10")

    S.STATE("h")
    S.GLOBAL_SIGNED(
        aq10=2.5,
        bq10=21.0,
        cq10=10.0,
        A_ah=0.38685,
        B_ah=122.35,
        C_ah=15.29,
        A_bh=2.00283,
        B_bh=5.5266,
        C_bh=-12.70195,
    )

    S.DERIVATIVE("h' = (hinf - h) / tauh")
    S.ASSIGNED("hinf", "tauh")

    def derive_buffers(self):
        return {"q10": 1 / (self.aq10 ** ((self.celsius - self.bq10) / self.cq10))}

    def alpha(self, v):
        return self.A_ah * sigmoid((-self.B_ah - v) / self.C_ah)

    def beta(self, v):
        return -0.00283 + self.A_bh * sigmoid((-self.B_bh - v) / self.C_bh)

    def assigned_values(self, v, values):
        a = self.alpha(v)
        b = self.beta(v)
        s = 1 / (a + b)
        tauh = self.q10 * s
        hinf = a * s
        return {"hinf": hinf, "tauh": tauh}

    def state_defaults(self, v, values):
        states = self.assigned_values(v, values)
        return {"h": states["hinf"]}


class s(S):
    S.DERIVED_BUFFER("q10")

    S.STATE("s")
    S.GLOBAL_SIGNED(
        aq10=2.5,
        bq10=21.0,
        cq10=10.0,
        A_as=0.00092,
        B_as=93.9,
        C_as=16.6,
        A_bs=-132.05,
        B_bs=-384.9,
        C_bs=28.5,
    )

    S.DERIVATIVE("s' = (sinf - s) / taus")
    S.ASSIGNED("sinf", "taus")

    def derive_buffers(self):
        return {"q10": 1 / (self.aq10 ** ((self.celsius - self.bq10) / self.cq10))}

    def alpha(self, v):
        return 0.00003 + self.A_as * sigmoid((-self.B_as - v) / self.C_as)

    def beta(self, v):
        return 132.05 + self.A_bs * sigmoid((-self.B_bs - v) / self.C_bs)

    def assigned_values(self, v, values):
        a = self.alpha(v)
        b = self.beta(v)
        s = 1 / (a + b)
        taus = self.q10 * s
        sinf = a * s
        return {"sinf": sinf, "taus": taus}

    def state_defaults(self, v, values):
        states = self.assigned_values(v, values)
        return {"s": states["sinf"]}


class nattxs(M):
    M.STATE_BUNDLE(m, h, s)
    M.GLOBAL_SIGNED(gbar=0.001)
    M.USEION("na", read=["ena"], write=["ina"])

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * self.s * (v - self.ena)
