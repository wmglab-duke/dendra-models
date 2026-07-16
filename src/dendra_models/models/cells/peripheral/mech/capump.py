# Calcium Pump in Schild 1994

from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class capump(M):
    M.GLOBAL_SIGNED(
        ICaPmax22=0.000859437, KmCa=0.0005, Q10CaP=2.30, Q10TempA=22.0, Q10TempB=10.0
    )
    M.USEION("ca", read=["cai"], write=["ica"])
    M.BUFFER("ICaPmax")
    M.EXPLICIT("ica")

    def initial(self, v):
        self.ICaPmax = self.ICaPmax22 * self.Q10CaP ** (
            (self.Q10TempA - self.celsius) / self.Q10TempB
        )

    def ica(self, v):
        return self.ICaPmax * self.cai / (self.KmCa + self.cai)
