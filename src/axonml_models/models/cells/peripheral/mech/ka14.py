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
        minf = (1/(1+exp(-1*(v+25)/12)))**(1/3)
        taum = (.6 + 2748/(exp((v+128)/14.5)+exp(-1*(v+10)/8))+1.7/(1+exp((v+8.5)/10.65)))/2
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
        hinf = .073 + 0.924/(1+exp((v+47)/4.75)) 
        tauh = 35 + 11.22/(exp((v+21.4)/9.48)+exp(-1*(v+155.3)/16.4))
        tauh = tauh / self.q10()
        return {"tauh": tauh, "hinf": hinf}

    def inf(self, v):
        return {"h": self.breakpoint(v)["hinf"]}
    

class s(S):
    has_q10 = True
    S.STATE("s")
    S.DERIVATIVE("s' = (sinf - s) / taus")
    S.ASSIGNED("sinf", "taus")
    S.GLOBAL(aq10=3.0)

    def calc_q10(self):
        return self.aq10 ** ((self.celsius - 22.0) / 10.0)
    
    def breakpoint(self, v):
        sinf = .415 + .576/(1+exp((v+44.5)/5.93)) + .103/(1+exp(-1*(v)/18.37))
        taus = 1000*(3.97/(exp((v+35.1)/9.97)+exp(-1*(v+83.3)/18))+2.5/(1+exp(-1*(v+27.3)/7.11)))
        taus = taus / self.q10()
        return {"taus": taus, "sinf": sinf}

    def inf(self, v):
        return {"s": self.breakpoint(v)["sinf"]}
    

class ka14(M):
    M.STATE(m, h, s)
    M.GLOBAL(gbar=0.0001)

    M.USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gbar * self.m**3 * self.h * self.s * (v - self.ek)
