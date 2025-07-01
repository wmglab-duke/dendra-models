from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S


class ki(S):
    S.STATE("ki")
    S.PARAMETER(FARADAY=96520)
    S.DERIVATIVE("ki' = -ik*4/FARADAY/diam*(1e4)")


class ko(S):
    S.STATE("ko")
    S.PARAMETER(FARADAY=96520, theta=0.029, D=0.1e-6, koinf=5.6)
    S.DERIVATIVE("ko' = (ik/FARADAY - 0.1*D*(ko-koinf)) / theta*(1e4)")


class koiTiger(M):
    M.STATE(ki, ko)
    M.USEION("k", read=["ik"], write=["ko", "ki"])
