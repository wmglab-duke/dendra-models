# Colbert and Pan 2002

from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class m(S):
    S.DERIVED_BUFFER("q10")
    S.STATE("m")
    S.GLOBALP(mtau_scale=0.4)
    S.GLOBAL_SIGNED(
        aq10=2.3,
        bq10=21.0,
        cq10=10.0,
        ma1=0.182,
        ma2=6.0,
        mb1=0.124,
        mb2=6.0,
    )
    S.GLOBAL_SIGNED(mshift=-38.0)

    S.DERIVATIVE("m' = (minf - m) / taum")
    S.ASSIGNED("minf", "taum")

    def derive_buffers(self):
        return {"q10": self.aq10 ** ((self.celsius - self.bq10) / self.cq10)}

    def alpha(self, v):
        return (
            self.q10
            * self.ma1
            * (v - self.mshift)
            / (1 - exp(-(v - self.mshift) / self.ma2))
        )

    def beta(self, v):
        return (
            self.q10
            * self.mb1
            * (-v + self.mshift)
            / (1 - exp(-(-v + self.mshift) / self.mb2))
        )

    def assigned_values(self, v, values):
        v = torch.where(torch.isin(v, self.mshift), v + 0.0001, v)
        a = self.alpha(v)
        b = self.beta(v)
        taum = self.mtau_scale / (a + b)
        minf = a / (a + b)
        return {"taum": taum, "minf": minf}

    def state_defaults(self, v, values):
        return {"m": self.assigned_values(v, values)["minf"]}


class h(S):
    S.DERIVED_BUFFER("q10")
    S.STATE("h")
    S.GLOBALP(htau_scale=0.4)
    S.GLOBAL_SIGNED(
        aq10=2.3,
        bq10=21.0,
        cq10=10.0,
        ha2=6.0,
        hb2=6.0,
    )
    S.GLOBAL_SIGNED(
        hshift=-66.0,
        ha1=-0.015,
        hb1=-0.015,
    )

    S.DERIVATIVE("h' = (hinf - h) / tauh")
    S.ASSIGNED("hinf", "tauh")

    def derive_buffers(self):
        return {"q10": self.aq10 ** ((self.celsius - self.bq10) / self.cq10)}

    def alpha(self, v):
        return (
            self.q10
            * self.ha1
            * (v - self.hshift)
            / (1 - exp((v - self.hshift) / self.ha2))
        )

    def beta(self, v):
        return (
            self.q10
            * self.hb1
            * (-v + self.hshift)
            / (1 - exp((-v + self.hshift) / self.hb2))
        )

    def assigned_values(self, v, values):
        v = torch.where(torch.isin(v, self.hshift), v + 0.0001, v)
        a = self.alpha(v)
        b = self.beta(v)
        tauh = self.htau_scale / (a + b)
        hinf = a / (a + b)
        return {"tauh": tauh, "hinf": hinf}

    def state_defaults(self, v, values):
        return {"h": self.assigned_values(v, values)["hinf"]}


class nata_t(M):
    M.STATE_BUNDLE(m, h)
    M.USEION("na", read=["ena"], write=["ina"])
    M.RANGEP(gbar=0.00001)

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * (v - self.ena)
