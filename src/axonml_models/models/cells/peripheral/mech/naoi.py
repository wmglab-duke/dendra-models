from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S
from axonml.models.mechanisms.ops import *


class nai(S):
    S.STATE("nai")
    S.GLOBAL(FARADAY=96520)
    S.DERIVATIVE("nai' = -ina*4/FARADAY/diam*(1e4)")


class nao(S):
    S.STATE("nao")
    S.GLOBAL(FARADAY=96520, theta=30e-3, D=0.1e-6, naoinf=154.0)
    S.DERIVATIVE("nao' = (ina/FARADAY - 0.1*D*(nao-naoinf)) / theta*(1e4)")


class naoi(M):
    M.STATE(nai, nao)
    M.USEION("na", read=["ina"], write=["nao", "nai"])
