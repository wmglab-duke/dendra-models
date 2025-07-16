# The transient component of the K current
# Voltage-gated K+ channels in layer 5 neocortical pyramidal
# neurons from young rats:subtypes and gradients
# Korngreen and Sakmann, J. Physiology, 2000


from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S
from axonml.models.mechanisms.ops import *


class mh(S):
    has_q10 = True
    S.STATE("m", "h")
    S.GLOBAL(aq10=2.3, bq10=21.0, cq10=10.0, celsius_const=34.0)

    S.DERIVATIVE(
        "m' = (minf - m) / taum",
        "h' = (hinf - h) / tauh",
    )
    S.ASSIGNED("minf", "taum", "hinf", "tauh")

    def calc_q10(self):
        return self.aq10 ** ((self.celsius_const - self.bq10) / self.cq10)

    def breakpoint(self, v):
        v = v + 10.0
        q10 = self.q10()
        minf = 1.0 / (1.0 + exp(-v / 19.0))
        taum = (0.34 + 0.92 * exp(-(((v + 71) / 59) ** 2))) / q10
        hinf = 1.0 / (1.0 + exp(-(v + 66.0) / -10.0))
        tauh = (8.0 + 49.0 * exp(-(((v + 73.0) / 23.0) ** 2))) / q10
        return {"taum": taum, "minf": minf, "tauh": tauh, "hinf": hinf}

    def inf(self, v):
        states = self.breakpoint(v)
        return {
            "m": states["minf"],
            "h": states["hinf"],
        }


class k_t(M):
    M.STATE(mh)
    M.USEION("k", read=["ek"], write=["ik"])

    M.RANGE(gbar=0.00001)

    def ik(self, v):
        return self.gbar * self.m**4 * self.h * (v - self.ek)
