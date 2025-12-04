# Extracellular calcium ion accumulation

import math

from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S
from axonml.models.mechanisms.ops import *


class cao(S):
    S.STATE("cao")
    S.GLOBAL(
        lseg=1.0,
        txfer=4511.0,
        FARADAY=96500.0,
        cabath=2.0,
        fhspace=1.0,
    )

    S.BUFFER("SA", "Vol_peri")
    S.DERIVATIVE("cao' = (ica*SA/(2*Vol_peri*FARADAY) + (cabath - cao)/txfer)")
    def initial(self, v):
        self.SA = math.pi * (1e-4) * self.diam * self.lseg
        Vol = math.pi * ((1e-4) * (self.diam / 2)) ** 2 * self.lseg
        self.Vol_peri = (
            math.pi * ((1e-4) * ((self.diam + self.fhspace) / 2)) ** 2 * self.lseg
        ) - Vol


class cao_augmented(S):
    S.STATE("cao")
    S.GLOBAL(
        lseg=1.0,
        txfer=4511.0,
        FARADAY=96500.0,
        cabath=2.0,
        fhspace=1.0,
        raug=1.0,
    )

    S.BUFFER("SA", "Vol_peri")
    S.DERIVATIVE("cao' = raug * (ica*SA/(2*Vol_peri*FARADAY) + (cabath - cao)/txfer)")
    def initial(self, v):
        self.SA = math.pi * (1e-4) * self.diam * self.lseg
        Vol = math.pi * ((1e-4) * (self.diam / 2)) ** 2 * self.lseg
        self.Vol_peri = (
            math.pi * ((1e-4) * ((self.diam + self.fhspace) / 2)) ** 2 * self.lseg
        ) - Vol

class caextscale(M):
    M.STATE(cao)
    M.USEION("ca", read=["ica"], write=["cao"])


class caextscale_augmented(M):
    M.STATE(cao_augmented)
    M.USEION("ca", read=["ica"], write=["cao"])
