# NaCaPump is the Sodium-Calcium Exchanger in Schild 1994

from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class nacapump(M):
    M.GLOBAL_SIGNED(
        KNaCa22=1.27324e-06,
        Q10NaCa=2.20,
        Q10TempA=22.85,
        Q10TempB=10.0,
        r=3.0,
        gamma=0.5,
        DNaCa=0.0036,
        F=96500.0,
        R=8314.0,
    )

    M.USEION("ca", read=["cao", "cai"], write=["ica"])
    M.USEION("na", read=["nai", "nao"], write=["ina"])

    M.BUFFER("inca")
    M.DERIVED_BUFFER("KNaCa", "DFin", "DFout")
    M.EXPLICIT("ina", "ica")

    def derive_buffers(self):
        KNaCa = self.KNaCa22 * self.Q10NaCa ** (
            (self.Q10TempA - self.celsius) / self.Q10TempB
        )
        temp = self.celsius + 273.15
        DFin = ((self.r - 2) * self.gamma * self.F) / (self.R * temp)
        DFout = ((self.r - 2) * (self.gamma - 1) * self.F) / (self.R * temp)
        return {"KNaCa": KNaCa, "DFin": DFin, "DFout": DFout}

    def initial(self, v):
        self.breakpoint(v)

    def breakpoint(self, v):
        S = 1.0 + self.DNaCa * (self.cai * self.nao**3 + self.cao * self.nai**3)
        DFin = self.nai**3 * self.cao * exp(self.DFin * v)
        DFout = self.nao**3 * self.cai * exp(self.DFout * v)
        self.inca = self.KNaCa * ((DFin - DFout) / S)

    def ina(self, v):
        return 3 * self.inca

    def ica(self, v):
        return -2 * self.inca
