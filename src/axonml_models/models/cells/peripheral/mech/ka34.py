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
        minf = (1/(1+exp(-1*(v-24)/17)))**(1/3)
        taum = (1/(exp((v-39)/13)+exp(-1*(v+134)/47)))/2
        taum = taum / self.q10()
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
        hinf = 1/(1+exp((v+31)/12))
        tauh = 15 + 1/(exp((v-60)/15)+exp(-1*(v+300)/27))
        tauh = tauh / self.q10()
        return {"tauh": tauh, "hinf": hinf}

    def inf(self, v):
        return {"h": self.breakpoint(v)["hinf"]}
    

class ka34(M):
    M.STATE(m, h)
    M.GLOBAL(gbar=0.0001)

    M.USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gbar * self.m**3 * self.h * (v - self.ek)
