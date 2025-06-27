from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S
from axonml.models.mechanisms.ops import *


class mh(S):
    S.STATE("m", "h")
    S.DERIVATIVE(
        "m' = (minf - m) / taum",
        "h' = (hinf - h) / tauh",
    )
    S.ASSIGNED("minf", "taum", "hinf", "tauh")

    def breakpoint(self, v):
        v = torch.where(v==-27.0, v+0.0001, v)
        mAlpha = (0.055*(-27-v))/(exp((-27-v)/3.8) - 1)
        mBeta = (0.94*exp((-75-v)/17))
        mInf = mAlpha/(mAlpha + mBeta)
        mTau = 1/(mAlpha + mBeta)
        hAlpha = (0.000457*exp((-13-v)/50))
        hBeta = (0.0065/(exp((-v-15)/28)+1))
        hInf = hAlpha/(hAlpha + hBeta)
        hTau = 1/(hAlpha + hBeta)
        return {
            "taum": mTau,
            "minf": mInf,
            "tauh": hTau,
            "hinf": hInf,
        }
    
    def inf(self, v):
        states = self.breakpoint(v)
        return {
            "m": states["minf"],
            "h": states["hinf"],
        }


class ca_hva(M):
    M.STATE(mh)
    M.PARAMETER(gbar=0.0001)
    M.USEION("ca", read=["eca"], write=["ica"])

    def ica(self, v):
        return self.gbar * self.m**2 * self.h * (v - self.eca)