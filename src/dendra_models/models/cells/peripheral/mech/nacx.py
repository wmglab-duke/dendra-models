from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class nacx(M):
    M.GLOBAL_SIGNED(gbar=316.0)
    M.GLOBAL_SIGNED(F=96500.0, R=8314.0, knaca=36e-9, dnaca=0.0036)
    M.USEION("na", read=["nai", "nao"], write=["ina"])
    M.USEION("ca", read=["cai", "cao"], write=["ica"])

    M.ASSIGNED("inaca", "q10", "FRT")
    M.EXPLICIT("ina", "ica")

    def breakpoint(self, v):
        dfcain = self.nai**3 * self.cao * exp(0.5 * v * self.FRT)
        dfcaout = self.nao**3 * self.cai * exp(-0.5 * v * self.FRT)
        s = 1 + self.dnaca * (self.cai * self.nao**3 + self.cao * self.nai**3)
        self.inaca = self.gbar * self.q10 * self.knaca * (dfcain - dfcaout) / s

    def initial(self, v):
        T = 273 + self.celsius
        self.q10.copy_((2.2 * (T - 296.0) + (310.0 - T)) / 14.0)
        self.FRT = self.F / (self.R * T)
        self.breakpoint(v)

    def ina(self, v):
        return 3 * self.inaca

    def ica(self, v):
        return -2 * self.inaca


class nacx_augmented(nacx):
    nacx.GLOBAL_SIGNED(ina_aug=1.0, ica_aug=1.0)

    def ina(self, v):
        return 3 * self.inaca * self.ina_aug

    def ica(self, v):
        return -2 * self.inaca * self.ica_aug