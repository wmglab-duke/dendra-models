from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S
from axonml.models.mechanisms.ops import *


class leakSchild(M):
    M.PARAMETER(
        gbna=1.85681e-05, gbca=3.00626e-06, R=8314, z=2, ecaoffset=78.7, F=96500
    )
    M.USEION("na", read=["ena"], write=["ina"])
    M.USEION("ca", read=["cao", "cai"], write=["ica"])

    def ina(self, v):
        return self.gbna * (v - self.ena)

    @property
    def ecaleak(self):
        return (
            self.R
            * (self.celsius + 273.15)
            / self.z
            / self.F
            * log(self.cao / self.cai)
        ) - self.ecaoffset

    def ica(self, v):
        return self.gbca * (v - self.ecaleak)
