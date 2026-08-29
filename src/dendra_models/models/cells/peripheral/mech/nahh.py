# From Traub & Miles "Neuronal networks of the hippocampus" (1991)
# Cummins et al. (2007), Sheets et al. (2007)

from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import exprelr, exp, expit


class m(S):
    S.DERIVED_BUFFER("q10")

    S.STATE("m")
    S.GLOBAL_SIGNED(
        am1=0.32,
        am2=13.1,
        am3=4.0,
        bm1=0.28,
        bm2=40.1,
        bm3=5.0,
        aq10=3.0,
        bq10=30.0,
        cq10=10.0,
        mshift=-6.0,
    )

    S.DERIVATIVE("m' = (minf - m) / taum")
    S.ASSIGNED("minf", "taum")

    def derive_buffers(self):
        return {"q10": self.aq10 ** ((self.celsius - self.bq10) / self.cq10)}

    def alpha(self, v):
        return self.q10 * self.am1 * exprelr(self.am2 - v, self.am3)

    def beta(self, v):
        return self.q10 * self.bm1 * exprelr(v - self.bm2, self.bm3)

    def assigned_values(self, v, values):
        v = v + 65.0 + self.mshift
        a = self.alpha(v)
        b = self.beta(v)
        taum = 1 / (a + b)
        minf = a * taum
        return {"taum": taum, "minf": minf}

    def state_defaults(self, v, values):
        states = self.assigned_values(v, values)
        return {"m": states["minf"]}


class h(S):
    S.DERIVED_BUFFER("q10")

    S.STATE("h")
    S.GLOBAL_SIGNED(
        ah1=0.128,
        ah2=17.0,
        ah3=18.0,
        bh1=4.0,
        bh2=40.0,
        bh3=5.0,
        aq10=3.0,
        bq10=30.0,
        cq10=10.0,
        hshift=6.0,
    )

    S.DERIVATIVE("h' = (hinf - h) / tauh")
    S.ASSIGNED("hinf", "tauh")

    def derive_buffers(self):
        return {"q10": self.aq10 ** ((self.celsius - self.bq10) / self.cq10)}

    def alpha(self, v):
        return self.q10 * self.ah1 * exp((self.ah2 - v) / self.ah3)

    def beta(self, v):
        return self.q10 * self.bh1 * expit((v - self.bh2) / self.bh3)

    def assigned_values(self, v, values):
        v = v + 65.0 + self.hshift
        a = self.alpha(v)
        b = self.beta(v)
        tauh = 1 / (a + b)
        hinf = a * tauh
        return {"tauh": tauh, "hinf": hinf}

    def state_defaults(self, v, values):
        states = self.assigned_values(v, values)
        return {"h": states["hinf"]}


class nahh(M):
    M.STATE_BUNDLE(m, h)
    M.GLOBAL_SIGNED(gnabar=0.3)
    M.USEION("na", read=["ena"], write=["ina"])

    def ina(self, v):
        return self.gnabar * self.m**3 * self.h * (v - self.ena)
