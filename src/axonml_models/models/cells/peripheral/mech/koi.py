from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S
from axonml.models.mechanisms.ops import *


class ki(S):
    S.STATE("ki")
    S.GLOBAL(FARADAY=96520)
    S.DERIVATIVE("ki' = -ik*4/FARADAY/diam*(1e4)")


class ko(S):
    S.STATE("ko")
    S.GLOBAL(FARADAY=96520, theta=0.03, D=0.1e-6, koinf=5.6)
    S.DERIVATIVE("ko' = (ik/FARADAY - 0.1*D*(ko-koinf)) / theta*(1e4)")


class koi(M):
    M.STATE(ki, ko)
    M.USEION("k", read=["ik"], write=["ko", "ki"])
