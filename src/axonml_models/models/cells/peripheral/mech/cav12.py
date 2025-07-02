from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S
from axonml.models.mechanisms.ops import *


class m(S):
    has_q10 = True
    S.STATE("m")
    S.DERIVATIVE("m' = (minf - m) / taum")
    S.ASSIGNED("minf", "taum")
    S.PARAMETER(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - 22.0) / 10.0)
    
    def breakpoint(self, v):
        minf = (1/(1 + exp(-1*(v - 1.75)/10)))**(1/3)
        taum = 0.25 + 0.5/(exp((v - 15)/5) + exp(-1*(v + 27)/12))
        taum = taum / self.q10()
        return {"taum": taum, "minf": minf}

    def inf(self, v):
        return {"m": self.breakpoint(v)["minf"]}
    

class h(S):
    has_q10 = True
    S.STATE("h")
    S.DERIVATIVE("h' = (hinf - h) / tauh")
    S.ASSIGNED("hinf", "tauh")
    S.PARAMETER(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - 22.0) / 10.0)
    
    def breakpoint(self, v):
        hinf = 1/(1 + exp((v - 10)/8))
        tauh = 1.4/(exp((v - 145)/30) + exp(-1*(v + 150)/16.4))
        tauh = tauh / self.q10()
        return {"tauh": tauh, "hinf": hinf}

    def inf(self, v):
        return {"h": self.breakpoint(v)["hinf"]}
    

class cav12(M):
    M.STATE(m, h)
    M.PARAMETER(gbar=0.0001)

    M.USEION("ca", read=["eca"], write=["ica"])

    def ica(self, v):
        return self.gbar * self.m**3 * self.h * (v - self.eca)
