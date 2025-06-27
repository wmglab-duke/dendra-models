from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S
from axonml.models.mechanisms.ops import *


class cai(S):
    S.STATE("cai")
    S.PARAMETER(FARADAY=96500, gamma=0.05, decay=80, depth=0.1, minCai=1e-4)
    S.DERIVATIVE("cai' = -(10000)*(ica*gamma/(2*FARADAY*depth)) - (cai - minCai)/decay")


class cadynamics(M):
    M.STATE(cai)
    M.USEION("ca", read=["ica"], write=["cai"])