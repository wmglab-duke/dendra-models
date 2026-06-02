# The persistent component of the K current
# Voltage-gated K+ channels in layer 5 neocortical pyramidal
# neurons from young rats: subtypes and gradients
# Korngreen and Sakmann, J. Physiology, 2000

from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class mh(S):
    has_q10 = True
    S.STATE("m", "h")
    S.GLOBAL_SIGNED(aq10=2.3, bq10=21.0, cq10=10.0)

    S.DERIVATIVE(
        "m' = (minf - m) / taum",
        "h' = (hinf - h) / tauh",
    )
    S.ASSIGNED("minf", "taum", "hinf", "tauh")

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - self.bq10) / self.cq10)

    def breakpoint(self, v, states):
        q10 = self.q10()
        v = v + 10.0
        minf = 1.0 / (1.0 + exp(-(v + 1.0) / 12.0))
        mtau_o = 1.25 + 13 * exp(-v * 0.026)
        mtau_e = 1.25 + 175.03 * exp(-v * -0.026)
        taum = torch.where(v < -50.0, mtau_e, mtau_o) / q10
        hinf = 1.0 / (1.0 + exp(-(v + 54.0) / -11.0))
        tauh = (360 + (1010 + 24 * (v + 55)) * exp(-(((v + 75) / 48) ** 2))) / q10
        return {"taum": taum, "minf": minf, "tauh": tauh, "hinf": hinf}

    def inf(self, v):
        states = self.breakpoint(v, None)
        return {
            "m": states["minf"],
            "h": states["hinf"],
        }


class k_p(M):
    M.STATE(mh)
    M.USEION("k", read=["ek"], write=["ik"])

    M.RANGEP(gbar=0.00001)

    def ik(self, v):
        return self.gbar * self.m**2 * self.h * (v - self.ek)
