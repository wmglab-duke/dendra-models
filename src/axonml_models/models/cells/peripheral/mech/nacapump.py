# NaCaPump is the Sodium-Calcium Exchanger in Schild 1994

from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S
from axonml.models.mechanisms.ops import *


class nacapump(M):
    M.GLOBAL(
        KNaCa22=1.27324e-06,
        Q10NaCa=2.20,
        Q10TempA=22.85,
        Q10TempB=10,
        r=3,
        gamma=0.5,
        DNaCa=0.0036,
        F=96500,
        R=8314,
    )

    M.USEION("ca", read=["cao", "cai"], write=["ica"])
    M.USEION("na", read=["nai", "nao"], write=["ina"])

    M.ASSIGNED("inca", "KNaCa", "DFin", "DFout")
    M.EXPLICIT("ina", "ica")

    def initial(self, v):
        self.KNaCa = self.KNaCa22 * self.Q10NaCa ** (
            (self.Q10TempA - self.celsius) / self.Q10TempB
        )
        temp = self.celsius + 273.15
        self.DFin = ((self.r - 2) * self.gamma * self.F) / (self.R * temp)
        self.DFout = ((self.r - 2) * (self.gamma - 1) * self.F) / (self.R * temp)
        self.breakpoint(v, None)

    def breakpoint(self, v, states):
        S = 1.0 + self.DNaCa * (self.cai * self.nao**3 + self.cao * self.nai**3)
        DFin = self.nai**3 * self.cao * exp(self.DFin * v)
        DFout = self.nao**3 * self.cai * exp(self.DFout * v)
        self.inca = self.KNaCa * ((DFin - DFout) / S)

    def ina(self, v):
        return 3 * self.inca

    def ica(self, v):
        return -2 * self.inca
