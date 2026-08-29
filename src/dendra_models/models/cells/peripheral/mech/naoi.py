from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class nai_augmented(S):
    S.STATE("nai")
    S.GLOBAL_SIGNED(FARADAY=96520.0, raug=1.0)
    S.DERIVATIVE("nai' = -raug*ina*4/FARADAY/diam*(1e4)")



class nai(S):
    S.STATE("nai")
    S.GLOBAL_SIGNED(FARADAY=96520.0)
    S.DERIVATIVE("nai' = -ina*4/FARADAY/diam*(1e4)")


class nao(S):
    S.STATE("nao")
    S.GLOBAL_SIGNED(FARADAY=96520.0, theta=30e-3, D=0.1e-6, naoinf=154.0)
    S.DERIVATIVE("nao' = (ina/FARADAY - 0.1*D*(nao-naoinf)) / theta*(1e4)")


class nao_augmented(S):
    S.STATE("nao")
    S.GLOBAL_SIGNED(FARADAY=96520.0, theta=30e-3, D=0.1e-6, naoinf=154.0, raug=1.0)
    S.DERIVATIVE("nao' = raug * (ina/FARADAY - 0.1*D*(nao-naoinf)) / theta*(1e4)")


class naoi(M):
    M.STATE_BUNDLE(nai, nao)
    M.USEION("na", read=["ina"], write=["nao", "nai"])


class naoi_augmented(M):
    M.STATE_BUNDLE(nai_augmented, nao_augmented)
    M.USEION("na", read=["ina"], write=["nao", "nai"])
