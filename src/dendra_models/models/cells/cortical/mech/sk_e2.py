from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class z(S):
    S.GLOBAL_SIGNED(ztau=1.0)
    S.STATE("z")
    S.DERIVATIVE("z' = (zinf - z) / ztau")
    S.ASSIGNED("zinf")

    def breakpoint(self, v, states):
        cai = torch.where(self.cai < 1e-7, self.cai + 1e-7, self.cai)
        zinf = 1 / (1 + (0.00043 / cai) ** 4.8)
        return {"zinf": zinf}

    def inf(self, v):
        return {"z": self.breakpoint(v, None)["zinf"]}


class sk_e2(M):
    M.STATE(z)

    M.USEION("k", read=["ek"], write=["ik"])
    M.USEION("ca", read=["cai"])

    M.RANGEP(gbar=0.0001)

    def ik(self, v):
        return self.gbar * self.z * (v - self.ek)
