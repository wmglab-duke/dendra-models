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
        return self.aq10 ** ((self.celsius - 21.0) / 10.0)
    
    def breakpoint(self, v):
        minf = (1/(1+exp(-1*(v+25)/7)))**(1/3)
        taum = (0.07/(exp((v-11.4)/14)+exp(-1*(v+61)/9.4)) + .1/(1+exp(-1*(v+5.5)/8))) / self.q10() / 2
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
        return self.aq10 ** ((self.celsius - 21.0) / 10.0)
    
    def breakpoint(self, v):
        hinf = 1/(1+exp((v+79)/7))
        tauh = (1/(exp(-1*(v+131)/9.5) + exp((v+5.7)/12.4))) / self.q10()
        return {"tauh": tauh, "hinf": hinf}

    def inf(self, v):
        return {"h": self.breakpoint(v)["hinf"]}
    

class nav7(M):
    M.STATE(m, h)
    M.PARAMETER(gbar=0.12)
    M.USEION("na", read=["ena"], write=["ina"])

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * (v - self.ena)