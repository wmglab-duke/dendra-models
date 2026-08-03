from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class m(S):
    has_q10 = True
    S.STATE("m")
    S.DERIVATIVE("m' = (minf - m) / taum")
    S.ASSIGNED("minf", "taum")

    def calc_q10(self):
        return 2.3 ** ((self.celsius - 21.0) / 10.0)

    def breakpoint(self, v, states):
        qt = self.q10()
        minf = 1.0 / (1 + exp((v - -52.6) / -4.6))
        mAlpha = 0.182 * vtrap(-(v + 38), 6)
        mBeta = 0.124 * vtrap(v + 38, 6)
        mTau = 6 * (1 / (mAlpha + mBeta)) / qt
        return {"taum": mTau, "minf": minf}

    def inf(self, v):
        return {"m": self.breakpoint(v, None)["minf"]}


class h(S):
    has_q10 = True
    S.STATE("h")
    S.DERIVATIVE("h' = (hinf - h) / tauh")
    S.ASSIGNED("hinf", "tauh")

    def calc_q10(self):
        return 2.3 ** ((self.celsius - 21.0) / 10.0)

    def breakpoint(self, v, states):
        qt = self.q10()
        hInf = 1.0 / (1 + exp((v - -48.8) / 10))
        hAlpha = 2.88e-6 * vtrap(v + 17, 4.63)
        hBeta = 6.94e-6 * vtrap(-(v + 64.4), 2.63)
        hTau = (1 / (hAlpha + hBeta)) / qt
        return {"tauh": hTau, "hinf": hInf}

    def inf(self, v):
        return {"h": self.breakpoint(v, None)["hinf"]}


class nap_et2(M):
    M.STATE(m, h)
    M.USEION("na", read=["ena"], write=["ina"])
    M.RANGEP(gbar=0.00001)

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * (v - self.ena)
