from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class m(S):
    S.DERIVED_BUFFER("q10")
    S.STATE("m")
    S.DERIVATIVE("m' = (minf - m) / taum")
    S.ASSIGNED("minf", "taum")

    def derive_buffers(self):
        return {"q10": 2.3 ** ((self.celsius - 21.0) / 10.0)}

    def assigned_values(self, v, values):
        qt = self.q10
        minf = 1.0 / (1 + exp((v - -52.6) / -4.6))
        mAlpha = 0.182 * vtrap(-(v + 38), 6)
        mBeta = 0.124 * vtrap(v + 38, 6)
        mTau = 6 * (1 / (mAlpha + mBeta)) / qt
        return {"taum": mTau, "minf": minf}

    def state_defaults(self, v, values):
        return {"m": self.assigned_values(v, values)["minf"]}


class h(S):
    S.DERIVED_BUFFER("q10")
    S.STATE("h")
    S.DERIVATIVE("h' = (hinf - h) / tauh")
    S.ASSIGNED("hinf", "tauh")

    def derive_buffers(self):
        return {"q10": 2.3 ** ((self.celsius - 21.0) / 10.0)}

    def assigned_values(self, v, values):
        qt = self.q10
        hInf = 1.0 / (1 + exp((v - -48.8) / 10))
        hAlpha = 2.88e-6 * vtrap(v + 17, 4.63)
        hBeta = 6.94e-6 * vtrap(-(v + 64.4), 2.63)
        hTau = (1 / (hAlpha + hBeta)) / qt
        return {"tauh": hTau, "hinf": hInf}

    def state_defaults(self, v, values):
        return {"h": self.assigned_values(v, values)["hinf"]}


class nap_et2(M):
    M.STATE_BUNDLE(m, h)
    M.USEION("na", read=["ena"], write=["ina"])
    M.RANGEP(gbar=0.00001)

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * (v - self.ena)
