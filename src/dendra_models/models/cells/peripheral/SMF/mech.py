# McIntyre, Richardson, Grill 2002

from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class m(S):
    S.DERIVED_BUFFER("q10")
    S.STATE("m")
    S.GLOBAL(
        amA=1.86,
        amB=21.4,
        amC=10.3,
        bmA=0.086,
        bmB=25.7,
        bmC=9.16,
        aq10_1=2.2,
        bq10=20.0,
        cq10=10.0,
    )
    S.DERIVATIVE("m' = (minf - m) / mtau")
    S.ASSIGNED("minf", "mtau")

    def derive_buffers(self):
        return {"q10": self.aq10_1 ** ((self.celsius - self.bq10) / self.cq10)}

    def alpha(self, v):
        x = -(v + self.amB)
        return self.q10 * self.amA * exprelr(x, self.amC)

    def beta(self, v):
        x = v + self.bmB
        return self.q10 * self.bmA * exprelr(x, self.bmC)

    def assigned_values(self, v, values):
        am = self.alpha(v)
        bm = self.beta(v)
        mtau = 1 / (am + bm)
        minf = am * mtau
        return {"mtau": mtau, "minf": minf}

    def state_defaults(self, v, values):
        return {"m": self.assigned_values(v, values)["minf"]}


class p(S):
    S.DERIVED_BUFFER("q10")
    S.STATE("p")
    S.GLOBAL(
        ampA=0.01,
        ampB=27.0,
        ampC=10.2,
        bmpA=0.00025,
        bmpB=34.0,
        bmpC=10.0,
        pq10_1=2.2,
        bq10=20.0,
        cq10=10.0,
    )
    S.DERIVATIVE("p' = (pinf - p) / ptau")
    S.ASSIGNED("pinf", "ptau")

    def derive_buffers(self):
        return {"q10": self.pq10_1 ** ((self.celsius - self.bq10) / self.cq10)}

    def alpha(self, v):
        x = -(v + self.ampB)
        return self.q10 * self.ampA * exprelr(x, self.ampC)

    def beta(self, v):
        x = v + self.bmpB
        return self.q10 * self.bmpA * exprelr(x, self.bmpC)

    def assigned_values(self, v, values):
        amp = self.alpha(v)
        bmp = self.beta(v)
        ptau = 1 / (amp + bmp)
        pinf = amp * ptau
        return {"ptau": ptau, "pinf": pinf}

    def state_defaults(self, v, values):
        return {"p": self.assigned_values(v, values)["pinf"]}


class h(S):
    S.DERIVED_BUFFER("q10")
    S.STATE("h")
    S.GLOBAL(
        ahA=0.062,
        ahB=114.0,
        ahC=11.0,
        bhA=2.3,
        bhB=31.8,
        bhC=13.4,
        aq10_2=2.9,
        bq10=20.0,
        cq10=10.0,
    )
    S.DERIVATIVE("h' = (hinf - h) / htau")
    S.ASSIGNED("hinf", "htau")

    def derive_buffers(self):
        return {"q10": self.aq10_2 ** ((self.celsius - self.bq10) / self.cq10)}

    def alpha(self, v):
        x = v + self.ahB
        return self.q10 * self.ahA * exprelr(x, self.ahC)

    def beta(self, v):
        return self.q10 * self.bhA * expit((v + self.bhB) / self.bhC)

    def assigned_values(self, v, values):
        ah = self.alpha(v)
        bh = self.beta(v)
        htau = 1 / (ah + bh)
        hinf = ah * htau
        return {"htau": htau, "hinf": hinf}

    def state_defaults(self, v, values):
        return {"h": self.assigned_values(v, values)["hinf"]}


class s(S):
    S.DERIVED_BUFFER("q10")
    S.STATE("s")
    S.GLOBAL(
        asA=0.3,
        asB=-27.0,
        asC=-5.0,
        bsA=0.03,
        bsB=10.0,
        bsC=-1.0,
        aq10_3=3.0,
        bq10=36.0,
        cq10=10.0,
        vtraub=-80.0,
    )
    S.DERIVATIVE("s' = (sinf - s) / stau")
    S.ASSIGNED("sinf", "stau")

    def derive_buffers(self):
        return {"q10": self.aq10_3 ** ((self.celsius - self.bq10) / self.cq10)}

    def alpha(self, v):
        b = self.q10 * self.asA * expit((self.vtraub - v - self.asB) / self.asC)
        return b

    def beta(self, v):
        b = self.q10 * self.bsA * expit((self.vtraub - v - self.bsB) / self.bsC)
        return b

    def assigned_values(self, v, values):
        as_ = self.alpha(v)
        bs = self.beta(v)
        stau = 1 / (as_ + bs)
        sinf = as_ * stau
        return {"stau": stau, "sinf": sinf}

    def state_defaults(self, v, values):
        return {"s": self.assigned_values(v, values)["sinf"]}


class axnode_myel(M):
    M.STATE_BUNDLE(m, p, h, s)
    M.GLOBAL(
        gnabar=3.0, gnapbar=0.01, gkbar=0.08, gl=0.007, ena=50.0, ek=-90.0, el=-90.0
    )

    M.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return (
            self.gnabar * self.m**3 * self.h * (v - self.ena)
            + self.gnapbar * self.p**3 * (v - self.ena)
            + self.gkbar * self.s * (v - self.ek)
            + self.gl * (v - self.el)
        )
