from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class ki(S):
    S.STATE("ki")
    S.GLOBAL_SIGNED(FARADAY=96520.0)
    S.DERIVATIVE("ki' = -ik*4/FARADAY/diam*(1e4)")


class ko(S):
    S.STATE("ko")
    S.GLOBAL_SIGNED(FARADAY=96520.0, theta=0.03, D=0.1e-6, koinf=5.6)
    S.DERIVATIVE("ko' = (ik/FARADAY - 0.1*D*(ko-koinf)) / theta*(1e4)")


class ko_augmented(S):
    S.STATE("ko")
    S.GLOBAL_SIGNED(FARADAY=96520.0, theta=0.03, D=0.1e-6, koinf=5.6, raug=1.0)
    S.DERIVATIVE("ko' = raug * (ik/FARADAY - 0.1*D*(ko-koinf)) / theta*(1e4)")


class ki_augmented(S):
    S.STATE("ki")
    S.GLOBAL_SIGNED(FARADAY=96520.0, raug=1.0)
    S.DERIVATIVE("ki' = -raug*ik*4/FARADAY/diam*(1e4)")


class koi(M):
    M.STATE_BUNDLE(ki, ko)
    M.USEION("k", read=["ik"], write=["ko", "ki"])


class koi_augmented(M):
    M.STATE_BUNDLE(ki_augmented, ko_augmented)
    M.USEION("k", read=["ik"], write=["ko", "ki"])
