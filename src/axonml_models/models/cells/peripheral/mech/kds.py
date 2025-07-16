# slowly inactivating delay current in Schild 1994

from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S
from axonml.models.mechanisms.ops import *


class x(S):
    has_q10 = True

    S.STATE("x")
    S.GLOBAL(
        Q10kds=1.93,
        Q10TempA=22.85,
        Q10TempB=10,
        V0p5x=39.59,
        S0p5x=-14.68,
        A_taux=5.0,
        B_taux=0.022,
        C_taux=2.5,
        Vpx=-65.0,
        shiftkds=3.0,
    )

    S.DERIVATIVE("x' = (xinf - x) / taux")
    S.ASSIGNED("xinf", "taux")

    def calc_q10(self):
        return self.Q10kds ** ((self.Q10TempA - self.celsius) / self.Q10TempB)

    def breakpoint(self, v):
        taux = (
            self.A_taux * exp(-((self.B_taux) ** 2) * (v - self.Vpx) ** 2) + self.C_taux
        )
        taux = self.q10() * taux
        xinf = sigmoid(-(v + self.V0p5x + self.shiftkds) / self.S0p5x)
        return {"taux": taux, "xinf": xinf}

    def inf(self, v):
        return {"x": sigmoid(-(v + self.V0p5x + self.shiftkds) / self.S0p5x)}


class y(S):
    has_q10 = True

    S.STATE("y")
    S.GLOBAL(
        Q10kds=1.93,
        Q10TempA=22.85,
        Q10TempB=10,
        V0p5y=48.0,
        S0p5y=7.0,
        tau_y22=7500,
        shiftkds=3.0,
    )

    S.DERIVATIVE("y' = (yinf - y) / tauy")
    S.ASSIGNED("yinf", "tauy")

    def calc_q10(self):
        return self.Q10kds ** ((self.Q10TempA - self.celsius) / self.Q10TempB)

    def breakpoint(self, v):
        tauy = self.tau_y22 * self.q10()
        yinf = sigmoid(-(v + self.V0p5y + self.shiftkds) / self.S0p5y)
        return {"tauy": tauy, "yinf": yinf}

    def inf(self, v):
        return {"y": sigmoid(-(v + self.V0p5y + self.shiftkds) / self.S0p5y)}


class kds(M):
    M.STATE(x, y)
    M.GLOBAL(gbar=0.000106103)
    M.USEION("k", read=["ek"], write=["ik"])

    def ik(self, v):
        return self.gbar * self.x**3 * self.y * (v - self.ek)
