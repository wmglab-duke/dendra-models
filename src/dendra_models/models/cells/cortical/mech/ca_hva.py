from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class mh(S):
    S.STATE("m", "h")
    S.DERIVATIVE(
        "m' = (minf - m) / taum",
        "h' = (hinf - h) / tauh",
    )
    S.ASSIGNED("minf", "taum", "hinf", "tauh")

    def breakpoint(self, v, states):
        mAlpha = 0.055 * vtrap(-(v + 27), 3.8)
        mBeta = 0.94 * exp((-75 - v) / 17)
        mInf = mAlpha / (mAlpha + mBeta)
        mTau = 1 / (mAlpha + mBeta)
        hAlpha = 0.000457 * exp((-13 - v) / 50)
        hBeta = 0.0065 / (exp((-v - 15) / 28) + 1)
        hInf = hAlpha / (hAlpha + hBeta)
        hTau = 1 / (hAlpha + hBeta)
        return {
            "taum": mTau,
            "minf": mInf,
            "tauh": hTau,
            "hinf": hInf,
        }

    def inf(self, v):
        states = self.breakpoint(v, None)
        return {
            "m": states["minf"],
            "h": states["hinf"],
        }


class ca_hva(M):
    M.STATE(mh)
    M.RANGEP(gbar=0.00001)
    M.USEION("ca", read=["eca"], write=["ica"])

    def ica(self, v):
        return self.gbar * self.m**2 * self.h * (v - self.eca)
