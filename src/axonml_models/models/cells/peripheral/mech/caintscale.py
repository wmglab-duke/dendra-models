# Intracellular calcium ion accumulation

import math

from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S
from axonml.models.mechanisms.ops import *


class oc_cai(S):
    S.STATE("oc", "cai")
    S.GLOBAL(lseg=1e-3, ku=100, kr=0.238, nb=4.0, Bi=0.001, FARADAY=96500)
    S.BUFFER("SA", "Vol")

    S.DERIVATIVE(
        "oc' = ku * cai * (1-oc) - kr * oc",
        "cai' = -ica * (SA) / Vol / (2*FARADAY) - (nb * Bi * (ku * cai * (1 - oc) - kr * oc))",
    )

    def initial(self, v):
        self.SA = math.pi * (1e-4) * self.diam * self.lseg
        self.Vol = math.pi * ((1e-4) * (self.diam / 2)) ** 2 * self.lseg


class caintscale(M):
    M.STATE(oc_cai)
    M.USEION("ca", read=["ica"], write=["cai"])
    M.INIT(oc=0.05)
