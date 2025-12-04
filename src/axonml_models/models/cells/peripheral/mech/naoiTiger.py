from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S


class nai(S):
    S.STATE("nai")
    S.GLOBAL(FARADAY=96520.0)
    S.DERIVATIVE("nai' = -ina*4/FARADAY/diam*(1e4)")


class nao(S):
    S.STATE("nao")
    S.GLOBAL(FARADAY=96520.0, theta=0.029, D=0.1e-6, naoinf=154.0)
    S.DERIVATIVE("nao' = (ina/FARADAY - 0.1*D*(nao-naoinf)) / theta*(1e4)")


class naoiTiger(M):
    M.STATE(nai, nao)
    M.USEION("na", read=["ina"], write=["nao", "nai"])
