from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class cai(S):
    S.STATE("cai")
    S.RANGEP(gamma=0.05, decay=80.0)
    S.GLOBALP(FARADAY=96500, depth=0.1, minCai=1e-4)
    S.ASSIGNED("shell_ica")
    S.DERIVATIVE("cai' = shell_ica - (cai - minCai)/decay")
    S.BUFFER("shell")

    def initial(self, v):
        self.shell = -10_000 * (self.gamma / (2 * self.FARADAY * self.depth))

    def breakpoint(self, v, states):
        return {"shell_ica": self.shell * self.ica}


class cadynamics(M):
    M.STATE(cai)
    M.USEION("ca", read=["ica"], write=["cai"])
