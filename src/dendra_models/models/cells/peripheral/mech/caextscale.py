# Extracellular calcium ion accumulation

import math

from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class cao(S):
    S.STATE("cao")
    S.GLOBAL_SIGNED(
        lseg=1.0,
        txfer=4511.0,
        FARADAY=96500.0,
        cabath=2.0,
        fhspace=1.0,
    )

    S.DERIVED_BUFFER("SA", "Vol_peri")
    S.DERIVATIVE("cao' = (ica*SA/(2*Vol_peri*FARADAY) + (cabath - cao)/txfer)")

    def derive_buffers(self):
        SA = math.pi * (1e-4) * self.diam * self.lseg
        Vol = math.pi * ((1e-4) * (self.diam / 2)) ** 2 * self.lseg
        Vol_peri = (
            math.pi * ((1e-4) * ((self.diam + self.fhspace) / 2)) ** 2 * self.lseg
        ) - Vol
        return {"SA": SA, "Vol_peri": Vol_peri}


class cao_augmented(S):
    S.STATE("cao")
    S.GLOBAL_SIGNED(
        lseg=1.0,
        txfer=4511.0,
        FARADAY=96500.0,
        cabath=2.0,
        fhspace=1.0,
        raug=1.0,
    )

    S.DERIVED_BUFFER("SA", "Vol_peri")
    S.DERIVATIVE("cao' = raug * (ica*SA/(2*Vol_peri*FARADAY) + (cabath - cao)/txfer)")

    def derive_buffers(self):
        SA = math.pi * (1e-4) * self.diam * self.lseg
        Vol = math.pi * ((1e-4) * (self.diam / 2)) ** 2 * self.lseg
        Vol_peri = (
            math.pi * ((1e-4) * ((self.diam + self.fhspace) / 2)) ** 2 * self.lseg
        ) - Vol
        return {"SA": SA, "Vol_peri": Vol_peri}


class caextscale(M):
    M.STATE(cao)
    M.USEION("ca", read=["ica"], write=["cao"])


class caextscale_augmented(M):
    M.STATE(cao_augmented)
    M.USEION("ca", read=["ica"], write=["cao"])
