from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S


class ki(S):
    S.STATE("ki")
    S.GLOBAL_SIGNED(FARADAY=96520.0)
    S.DERIVATIVE("ki' = -ik*4/FARADAY/diam*(1e4)")


class ko(S):
    S.STATE("ko")
    S.GLOBAL_SIGNED(FARADAY=96520.0, theta=0.029, D=0.1e-6, koinf=5.6)
    S.DERIVATIVE("ko' = (ik/FARADAY - 0.1*D*(ko-koinf)) / theta*(1e4)")


class koiTiger(M):
    M.STATE_BUNDLE(ki, ko)
    M.USEION("k", read=["ik"], write=["ko", "ki"])
