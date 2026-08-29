from dendra.const import FARADAY as DENDRA_FARADAY
from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class cai(S):
    S.STATE("cai")
    S.RANGEP(gamma=0.05, decay=80.0, depth=0.1, minCai=1e-4)
    S.GLOBAL_SIGNED(FARADAY=DENDRA_FARADAY)
    S.ASSIGNED("shell_ica")
    S.DERIVATIVE("cai' = shell_ica - (cai - minCai)/decay")
    S.DERIVED_BUFFER("shell")

    def derive_buffers(self):
        return {"shell": -10_000 * (self.gamma / (2 * self.FARADAY * self.depth))}

    def assigned_values(self, v, values):
        return {"shell_ica": self.shell * self.ica}


class cadynamics(M):
    M.STATE_BUNDLE(cai)
    M.USEION("ca", read=["ica"], write=["cai"])
