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
    
    def breakpoint(self, v):
        q10 = self.q10()
        minf = (1/(1+exp(-1*(v+4)/7)))**(1/3)
        taum = 0.03 + .5/(exp((v)/12)+exp(-1*(v+29)/18))
        taum = taum / q10 / 2
        return {"taum": taum, "minf": minf}

    def inf(self, v):
        return {"m": self.breakpoint(v)["minf"]}
    

class h(S):
    has_q10 = True
    S.STATE("h")
    S.DERIVATIVE("h' = (hinf - h) / tauh")
    S.ASSIGNED("hinf", "tauh")
    S.GLOBAL(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - 22.0) / 10.0)

    def breakpoint(self, v):
        q10 = self.q10()
        hinf = 1/(1+exp((v+28)/4.56))
        tauh = 2.63+250/(exp((v+36)/7.7)+exp(-1*(v)/16.3)) + 1.4/(1+exp(-1*(v+0.6)/2.95))
        tauh = tauh / q10
        return {"tauh": tauh, "hinf": hinf}

    def inf(self, v):
        return {"h": 1/(1+exp((v+28)/4.56))}


class s(S):
    has_q10 = True
    S.STATE("s")
    S.DERIVATIVE("s' = (sinf - s) / taus")
    S.ASSIGNED("sinf", "taus")
    S.GLOBAL(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - 22.0) / 10.0)

    def breakpoint(self, v):
        q10 = self.q10()
        sinf = 1/(1+exp((v+50)/7.5))
        taus = 34/(exp((v+2)/16)+exp(-1*(v+108)/8)) + 160/(1+exp(-1*(v+110)/75))
        taus = taus / q10
        return {"taus": taus, "sinf": sinf}

    def inf(self, v):
        return {"s": 1/(1+exp((v+50)/7.5))}


class newnav8(M):
    M.STATE(m, h, s)
    M.RANGE(gbar=0.0)
    M.USEION("na", read=["ena"], write=["ina"])

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * self.s * (v - self.ena)