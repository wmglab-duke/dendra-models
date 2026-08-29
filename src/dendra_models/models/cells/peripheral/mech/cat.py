# Cat is the Low threshold, transient Ca current in Schild 1994

from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class d(S):
    S.DERIVED_BUFFER("q10")

    S.STATE("d")
    S.GLOBAL_SIGNED(
        Q10catd=1.90,
        Q10TempA=22.85,
        Q10TempB=10.0,
        V0p5d=54.00,
        S0p5d=-5.75,
        A_taud=22.0,
        B_taud=0.052,
        C_taud=2.5,
        Vpd=-68.0,
        shiftcat=-7.0,
    )

    S.DERIVATIVE("d' = (dinf - d) / taud")
    S.ASSIGNED("taud", "dinf")

    def derive_buffers(self):
        return {"q10": self.Q10catd ** ((self.Q10TempA - self.celsius) / self.Q10TempB)}

    def assigned_values(self, v, values):
        taud = (
            self.A_taud * exp(-((self.B_taud) ** 2) * (v - self.Vpd) ** 2) + self.C_taud
        )
        taud = self.q10 * taud
        dinf = sigmoid(-(v + self.V0p5d + self.shiftcat) / self.S0p5d)
        return {"taud": taud, "dinf": dinf}

    def state_defaults(self, v, values):
        return {"d": sigmoid(-(v + self.V0p5d + self.shiftcat) / self.S0p5d)}


class f(S):
    S.DERIVED_BUFFER("q10")

    S.STATE("f")
    S.GLOBAL_SIGNED(
        Q10catf=2.20,
        Q10TempA=22.85,
        Q10TempB=10.0,
        V0p5f=68.00,
        S0p5f=6.0,
        A_tauf=103.0,
        B_tauf=0.050,
        C_tauf=12.5,
        Vpf=-58.0,
        shiftcat=-7.0,
    )

    S.DERIVATIVE("f' = (finf - f) / tauf")
    S.ASSIGNED("tauf", "finf")

    def derive_buffers(self):
        return {"q10": self.Q10catf ** ((self.Q10TempA - self.celsius) / self.Q10TempB)}

    def assigned_values(self, v, values):
        tauf = (
            self.A_tauf * exp(-((self.B_tauf) ** 2) * (v - self.Vpf) ** 2) + self.C_tauf
        )
        tauf = self.q10 * tauf
        finf = sigmoid(-(v + self.V0p5f + self.shiftcat) / self.S0p5f)
        return {"tauf": tauf, "finf": finf}

    def state_defaults(self, v, values):
        return {"f": sigmoid(-(v + self.V0p5f + self.shiftcat) / self.S0p5f)}


class cat(M):
    M.STATE_BUNDLE(d, f)
    M.GLOBAL_SIGNED(gbar=1.23787e-05, R=8314.0, z=2.0, ecaoffset=78.7, F=96500.0)
    M.USEION("ca", read=["cao", "cai"], write=["ica"])

    @property
    def ecat(self):
        return (
            self.R
            * (self.celsius + 273.15)
            / self.z
            / self.F
            * log(self.cao / self.cai)
        ) - self.ecaoffset

    def ica(self, v):
        return self.gbar * self.d * self.f * (v - self.ecat)
