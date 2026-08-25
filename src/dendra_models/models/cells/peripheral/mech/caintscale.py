# Intracellular calcium ion accumulation

import math

from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class oc_cai(S):
    S.STATE("oc", "cai")
    S.METHOD("bufferimplicit", bound="oc", free="cai")
    S.GLOBAL_SIGNED(lseg=1e-3, ku=100.0, kr=0.238, nb=4.0, Bi=0.001, FARADAY=96500.0)
    S.DERIVED_BUFFER("SA", "Vol")

    S.DERIVATIVE(
        "oc' = ku * cai * (1-oc) - kr * oc",
        "cai' = (-ica * (SA) / Vol / (2*FARADAY) - (nb * Bi * (ku * cai * (1 - oc) - kr * oc)))",
    )

    def derive_buffers(self):
        return {
            "SA": math.pi * (1e-4) * self.diam * self.lseg,
            "Vol": math.pi * ((1e-4) * (self.diam / 2)) ** 2 * self.lseg,
        }


class oc_cai_augmented(S):
    S.STATE("oc", "cai")
    S.METHOD("bufferimplicit", bound="oc", free="cai")
    S.GLOBAL_SIGNED(
        lseg=1e-3,
        ku=100.0,
        kr=0.238,
        nb=4.0,
        Bi=0.001,
        FARADAY=96500.0,
        raug=1.0,
    )
    S.DERIVED_BUFFER("SA", "Vol")

    S.DERIVATIVE(
        "oc' = (ku * cai * (1-oc) - kr * oc)",
        "cai' = raug * (-ica * (SA) / Vol / (2*FARADAY) - (nb * Bi * (ku * cai * (1 - oc) - kr * oc)))",
    )

    def derive_buffers(self):
        return {
            "SA": math.pi * (1e-4) * self.diam * self.lseg,
            "Vol": math.pi * ((1e-4) * (self.diam / 2)) ** 2 * self.lseg,
        }


class caintscale(M):
    M.STATE(oc_cai)
    M.USEION("ca", read=["ica"], write=["cai"])
    M.INIT(oc=0.05)


class caintscale_augmented(M):
    M.STATE(oc_cai_augmented)
    M.USEION("ca", read=["ica"], write=["cai"])
    M.INIT(oc=0.05)
