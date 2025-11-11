# can is the high threshold, long-lasting calcium current in Schild 1994

from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S
from axonml.models.mechanisms.ops import *


class d(S):
    has_q10 = True

    S.STATE("d")
    S.GLOBAL(
        Q10can=4.30,
        Q10TempA=22.85,
        Q10TempB=10.0,
        shiftcan=-7.0,
        V0p5d=20.0,
        S0p5d=-4.5,
        A_taud=3.25,
        B_taud=0.042,
        C_taud=0.395,
        Vpd=-31.0,
    )

    S.DERIVATIVE("d' = (dinf - d) / taud")
    S.ASSIGNED("dinf", "taud")

    def calc_q10(self):
        return self.Q10can ** ((self.Q10TempA - self.celsius) / self.Q10TempB)

    def breakpoint(self, v, states):
        taud = self.q10() * (
            self.A_taud * exp(-((self.B_taud) ** 2) * (v - self.Vpd) ** 2) + self.C_taud
        )
        dinf = 1.0 / (1.0 + exp((v + self.V0p5d + self.shiftcan) / self.S0p5d))
        return {"taud": taud, "dinf": dinf}

    def inf(self, v):
        return {"d": self.breakpoint(v, None)["dinf"]}


class f1(S):
    has_q10 = True

    S.STATE("f1")
    S.GLOBAL(
        Q10can=4.30,
        Q10TempA=22.85,
        Q10TempB=10.0,
        shiftcan=-7.0,
        V0p5f1=20.0,
        S0p5f1=25.0,
        A_tauf1=33.5,
        B_tauf1=0.0395,
        C_tauf1=5.0,
        Vpf1=-30.0,
    )

    S.DERIVATIVE("f1' = (f1inf - f1) / tauf1")
    S.ASSIGNED("f1inf", "tauf1")

    def calc_q10(self):
        return self.Q10can ** ((self.Q10TempA - self.celsius) / self.Q10TempB)

    def breakpoint(self, v, states):
        tauf1 = self.q10() * (
            self.A_tauf1 * exp(-((self.B_tauf1) ** 2) * (v - self.Vpf1) ** 2)
            + self.C_tauf1
        )
        f1inf = 1.0 / (1.0 + exp((v + self.V0p5f1 + self.shiftcan) / self.S0p5f1))
        return {"tauf1": tauf1, "f1inf": f1inf}

    def inf(self, v):
        return {"f1": self.breakpoint(v, None)["f1inf"]}


class f2(S):
    has_q10 = True

    S.STATE("f2")
    S.GLOBAL(
        Q10can=4.30,
        Q10TempA=22.85,
        Q10TempB=10.0,
        shiftcan=-7.0,
        V0p5f2=40.0,
        S0p5f2=10.0,
        A_tauf2=225.0,
        B_tauf2=0.0275,
        C_tauf2=75.0,
        Vpf2=-40.0,
        A_rn=5.0,
        B_rn=-10.0,
    )

    S.DERIVATIVE("f2' = (f2inf - f2) / tauf2")
    S.ASSIGNED("f2inf", "tauf2")

    def calc_q10(self):
        return self.Q10can ** ((self.Q10TempA - self.celsius) / self.Q10TempB)

    def breakpoint(self, v, states):
        tauf2 = self.q10() * (
            self.A_tauf2 * exp(-((self.B_tauf2) ** 2) * (v - self.Vpf2) ** 2)
            + self.C_tauf2
        )
        rn = 0.2 / (1.0 + exp((v + self.A_rn + self.shiftcan) / self.B_rn))
        f2inf = rn + (
            1.0 / (1.0 + exp((v + self.V0p5f2 + self.shiftcan) / self.S0p5f2))
        )
        return {"tauf2": tauf2, "f2inf": f2inf}

    def inf(self, v):
        return {"f2": self.breakpoint(v, None)["f2inf"]}


class can(M):
    M.STATE(d, f1, f2)
    M.GLOBAL(gbar=0.000106103, R=8314.0, z=2, ecaoffset=78.7, F=96500)
    M.USEION("ca", read=["cao", "cai"], write=["ica"])

    @property
    def ecan(self):
        return (
            self.R
            * (self.celsius + 273.15)
            / self.z
            / self.F
            * log(self.cao / self.cai)
        ) - self.ecaoffset

    def ica(self, v):
        return self.gbar * self.d * (0.55 * self.f1 + 0.45 * self.f2) * (v - self.ecan)
