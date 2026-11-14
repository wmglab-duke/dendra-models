# Sodium-Potassium Pump in Schild 1994

from dendra.models.mechanisms import Mechanism as M

from .nakpump import _LegacyPumpParameters


class nakpumpSchild(_LegacyPumpParameters, M):
    _legacy_positive_parameters = ("Kmnai", "Kmko", "gbar_INaKmax22")

    M.GLOBAL_SIGNED(
        Q10NaK=1.16,
        Q10TempA=22.85,
        Q10TempB=10.0,
    )
    M.RANGE(Kmnai=5.46, Kmko=0.621, gbar_INaKmax22=0.009726135)

    M.USEION("k", read=["ko"], write=["ik"])
    M.USEION("na", read=["nai"], write=["ina"])

    M.DERIVED_BUFFER("INaKmax")

    def derive_buffers(self):
        return {
            "INaKmax": self.gbar_INaKmax22
            * self.Q10NaK ** ((self.Q10TempA - self.celsius) / self.Q10TempB)
        }

    def _pump_current(self, v, *, with_slope=False):
        denominator = v + 200.0
        voltage_factor = (v + 150.0) / denominator
        sodium_factor = (self.nai / (self.nai + self.Kmnai)) ** 3
        potassium_factor = (self.ko / (self.ko + self.Kmko)) ** 2
        current = self.INaKmax * voltage_factor * sodium_factor * potassium_factor
        if not with_slope:
            return current
        voltage_slope = 50.0 / denominator.square()
        slope = self.INaKmax * voltage_slope * sodium_factor * potassium_factor
        return current, slope

    def ina(self, v):
        return 3.0 * self._pump_current(v)

    def ina_with_conductance(self, v):
        current, slope = self._pump_current(v, with_slope=True)
        return 3.0 * current, 3.0 * slope

    def ik(self, v):
        return -2.0 * self._pump_current(v)

    def ik_with_conductance(self, v):
        current, slope = self._pump_current(v, with_slope=True)
        return -2.0 * current, -2.0 * slope


class nakpumpSchild_augmented(nakpumpSchild):
    nakpumpSchild.GLOBAL_SIGNED(ina_aug=1.0, ik_aug=1.0)

    def ina(self, v):
        return super().ina(v) * self.ina_aug

    def ina_with_conductance(self, v):
        current, slope = super().ina_with_conductance(v)
        return current * self.ina_aug, slope * self.ina_aug

    def ik(self, v):
        return super().ik(v) * self.ik_aug

    def ik_with_conductance(self, v):
        current, slope = super().ik_with_conductance(v)
        return current * self.ik_aug, slope * self.ik_aug
