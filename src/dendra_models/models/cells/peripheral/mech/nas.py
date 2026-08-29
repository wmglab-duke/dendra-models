# slower, TTX-insensitive current in Schild 1994

from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class m(S):
    S.DERIVED_BUFFER("q10")

    S.STATE("m")
    S.GLOBAL_SIGNED(
        Q10nasm=2.30,
        Q10TempA=22.85,
        Q10TempB=10.0,
        shiftnas=-20.0,
        V0p5m=20.35,
        S0p5m=-4.45,
        A_taum=1.50,
        B_taum=0.0595,
        C_taum=0.15,
        Vpm=-20.35,
    )

    S.DERIVATIVE("m' = (minf - m) / taum")
    S.ASSIGNED("minf", "taum")

    def derive_buffers(self):
        return {"q10": self.Q10nasm ** ((self.Q10TempA - self.celsius) / self.Q10TempB)}

    def assigned_values(self, v, values):
        taum = self.q10 * (
            self.A_taum * exp(-((self.B_taum) ** 2) * (v - self.Vpm) ** 2) + self.C_taum
        )
        minf = sigmoid(-(v + self.V0p5m + self.shiftnas) / self.S0p5m)
        return {"taum": taum, "minf": minf}

    def state_defaults(self, v, values):
        return {"m": sigmoid(-(v + self.V0p5m + self.shiftnas) / self.S0p5m)}


class h(S):
    S.DERIVED_BUFFER("q10")

    S.STATE("h")
    S.GLOBAL_SIGNED(
        Q10nash=1.50,
        Q10TempA=22.85,
        Q10TempB=10.0,
        shiftnas=-20.0,
        V0p5h=18.00,
        S0p5h=4.50,
        A_tauh=4.95,
        B_tauh=0.0335,
        C_tauh=0.75,
        Vph=-20.00,
    )

    S.DERIVATIVE("h' = (hinf - h) / tauh")
    S.ASSIGNED("hinf", "tauh")

    def derive_buffers(self):
        return {"q10": self.Q10nash ** ((self.Q10TempA - self.celsius) / self.Q10TempB)}

    def assigned_values(self, v, values):
        tauh = self.q10 * (
            self.A_tauh * exp(-((self.B_tauh) ** 2) * (v - self.Vph) ** 2) + self.C_tauh
        )
        hinf = sigmoid(-(v + self.V0p5h + self.shiftnas) / self.S0p5h)
        return {"tauh": tauh, "hinf": hinf}

    def state_defaults(self, v, values):
        return {"h": sigmoid(-(v + self.V0p5h + self.shiftnas) / self.S0p5h)}


class nas(M):
    M.STATE_BUNDLE(m, h)
    M.GLOBAL_SIGNED(gbar=0.001043349)
    M.USEION("na", read=["ena"], write=["ina"])

    def ina(self, v):
        return self.gbar * self.m**3 * self.h * (v - self.ena)
