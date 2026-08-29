from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class z(S):
    S.GLOBAL_SIGNED(ztau=1.0)
    S.STATE("z")
    S.DERIVATIVE("z' = (zinf - z) / ztau")
    S.ASSIGNED("zinf")

    def assigned_values(self, v, values):
        cai = torch.where(self.cai < 1e-7, self.cai + 1e-7, self.cai)
        zinf = 1 / (1 + (0.00043 / cai) ** 4.8)
        return {"zinf": zinf}

    def state_defaults(self, v, values):
        return {"z": self.assigned_values(v, values)["zinf"]}


class sk_e2(M):
    M.STATE_BUNDLE(z)

    M.USEION("k", read=["ek"], write=["ik"])
    M.USEION("ca", read=["cai"])

    M.RANGEP(gbar=0.000001)

    def ik(self, v):
        return self.gbar * self.z * (v - self.ek)
