from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class m(S):
    has_q10 = True

    S.STATE("m")
    S.GLOBAL(amA=1.0, aq10=2.24659524757)
    S.DERIVATIVE("m' = (minf - m) / mtau")
    S.ASSIGNED("minf", "mtau")

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - 6.3) / 10)

    def alpha(self, v):
        return exprelr(2.5 - 0.1 * (v + 70), self.amA)

    def beta(self, v):
        return 4.0 * exp(-(v + 70.0) / 18.0)

    def breakpoint(self, v, states):
        a = self.alpha(v)
        b = self.beta(v)
        s = a + b
        minf = a / s
        mtau = 1.0 / (self.q10() * s)
        return {"mtau": mtau, "minf": minf}

    def inf(self, v):
        states = self.breakpoint(v, None)
        return {"m": states["minf"]}


class h(S):
    has_q10 = True
    S.STATE("h")
    S.GLOBAL(aq10=2.24659524757)
    S.DERIVATIVE("h' = (hinf - h) / htau")
    S.ASSIGNED("hinf", "htau")

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - 6.3) / 10)

    def alpha(self, v):
        return 0.07 * exp(-(v + 70) / 20)

    def beta(self, v):
        return expit((-3.0) + 0.1 * (v + 70))

    def breakpoint(self, v, states):
        a = self.alpha(v)
        b = self.beta(v)
        s = a + b
        hinf = a / s
        htau = 1.0 / (self.q10() * s)
        return {"htau": htau, "hinf": hinf}

    def inf(self, v):
        states = self.breakpoint(v, None)
        return {"h": states["hinf"]}


class n(S):
    has_q10 = True
    S.STATE("n")
    S.GLOBAL(anA=1.0, aq10=2.24659524757)
    S.DERIVATIVE("n' = (ninf - n) / ntau")
    S.ASSIGNED("ninf", "ntau")

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - 6.3) / 10)

    def alpha(self, v):
        return 0.1 * exprelr(1.0 - 0.1 * (v + 70.0), self.anA)

    def beta(self, v):
        return 0.125 * exp(-(v + 70.0) / 80.0)

    def breakpoint(self, v, states):
        a = self.alpha(v)
        b = self.beta(v)
        s = a + b
        ninf = a / s
        ntau = 1.0 / (self.q10() * s)
        return {"ntau": ntau, "ninf": ninf}

    def inf(self, v):
        states = self.breakpoint(v, None)
        return {"n": states["ninf"]}


class rattay_aberham(M):
    M.STATE(m, h, n)
    M.GLOBAL(gnabar=0.12, gkbar=0.036, gl=0.0003, el=-59.4)

    M.USEION("na", read=["ena"], write=["ina"])
    M.USEION("k", read=["ek"], write=["ik"])
    M.NONSPECIFIC_CURRENT("il")

    def ina(self, v):
        return self.gnabar * self.m**3 * self.h * (v - self.ena)

    def ik(self, v):
        return self.gkbar * self.n**4 * (v - self.ek)

    def il(self, v):
        return self.gl * (v - self.el)
