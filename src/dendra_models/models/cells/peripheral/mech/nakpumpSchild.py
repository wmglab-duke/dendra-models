# Sodium-Potassium Pump in Schild 1994

from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import *


class nakpumpSchild(M):
    M.GLOBAL_SIGNED(
        Kmnai=5.46,
        Kmko=0.621,
        Q10NaK=1.16,
        Q10TempA=22.85,
        Q10TempB=10.0,
    )
    M.GLOBAL_SIGNED(gbar_INaKmax22=0.009726135)

    M.USEION("k", read=["ko"], write=["ik"])
    M.USEION("na", read=["nai"], write=["ina"])

    M.ASSIGNED("ink", "INaKmax")
    M.EXPLICIT("ina", "ik")

    def initial(self, v):
        self.INaKmax = self.gbar_INaKmax22 * self.Q10NaK ** (
            (self.Q10TempA - self.celsius) / self.Q10TempB
        )
        self.breakpoint(v)

    def breakpoint(self, v):
        fnk = (v + 150.0) / (v + 200.0)
        self.ink = (
            self.INaKmax
            * fnk
            * ((self.nai / (self.nai + self.Kmnai)) ** 3)
            * ((self.ko / (self.ko + self.Kmko)) ** 2)
        )

    def ina(self, v):
        return 3 * self.ink

    def ik(self, v):
        return -2 * self.ink


class nakpumpSchild_augmented(nakpumpSchild):
    nakpumpSchild.GLOBAL_SIGNED(ina_aug=1.0, ik_aug=1.0)

    def ina(self, v):
        return 3 * self.ink * self.ina_aug

    def ik(self, v):
        return -2 * self.ink * self.ik_aug