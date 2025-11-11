from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S
from axonml.models.mechanisms.ops import *


class m(S):
    has_q10 = True
    S.STATE("m")
    S.DERIVATIVE("m' = (minf - m) / taum")
    S.ASSIGNED("minf", "taum")
    S.GLOBAL(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - 22.0) / 10.0)

    def breakpoint(self, v, states):
        q10 = self.q10()
        minf = (1 / (1 + exp(-1 * (v + 51) / 8.4))) ** (1 / 3)
        taum = (0.3 + 1 / (exp((v + 1.3) / 22) + exp(-1 * (v + 82) / 10))) / 2
        taum = taum / q10
        return {"taum": taum, "minf": minf}

    def inf(self, v):
        return {"m": (1 / (1 + exp(-1 * (v + 51) / 8.4))) ** (1 / 3)}


class h(S):
    has_q10 = True
    S.STATE("h")
    S.DERIVATIVE("h' = (hinf - h) / tauh")
    S.ASSIGNED("hinf", "tauh")
    S.GLOBAL(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - 22.0) / 10.0)

    def breakpoint(self, v, states):
        q10 = self.q10()
        hinf = 1 / (1 + exp((v + 55) / 11.5))
        tauh = 2 + 1 / (exp((v - 32) / 14) + exp(-1 * (v + 157) / 12))
        tauh = tauh / q10
        return {"tauh": tauh, "hinf": hinf}

    def inf(self, v):
        return {"h": 1 / (1 + exp((v + 55) / 11.5))}


class s(S):
    has_q10 = True
    S.STATE("s")
    S.DERIVATIVE("s' = (sinf - s) / taus")
    S.ASSIGNED("sinf", "taus")
    S.GLOBAL(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - 22.0) / 10.0)

    def breakpoint(self, v, states):
        q10 = self.q10()
        sinf = 0.03 + 0.97 / (1 + exp((v + 79) / 6.87))
        taus = 1 / (exp((v - 195) / 29) + exp(-1 * (v + 193) / 12.4)) + 1165 / (
            1 + exp(-1 * (v + 40) / 30)
        )
        taus = taus / q10
        return {"taus": taus, "sinf": sinf}

    def inf(self, v):
        return {"s": 0.03 + 0.97 / (1 + exp((v + 79) / 6.87))}


class nav9(M):
    M.STATE(m, h, s)
    M.GLOBAL(gbar=0.0)
    M.USEION("na", read=["ena"], write=["ina"])

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * self.s * (v - self.ena)
