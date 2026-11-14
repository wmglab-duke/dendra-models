# NaCaPump is the Sodium-Calcium Exchanger in Schild 1994

from dendra.models.mechanisms import Mechanism as M
from dendra.models.mechanisms.ops import exp


class nacapump(M):
    M.GLOBAL_SIGNED(
        KNaCa22=1.27324e-06,
        Q10NaCa=2.20,
        Q10TempA=22.85,
        Q10TempB=10.0,
        r=3.0,
        gamma=0.5,
        F=96500.0,
        R=8314.0,  # The source uses 1000 * R with R = 8.314.
    )
    M.RANGE(DNaCa=0.0036)

    M.USEION("ca", read=["cao", "cai"], write=["ica"])
    M.USEION("na", read=["nai", "nao"], write=["ina"])

    M.DERIVED_BUFFER("KNaCa")

    def derive_buffers(self):
        KNaCa = self.KNaCa22 * self.Q10NaCa ** (
            (self.Q10TempA - self.celsius) / self.Q10TempB
        )
        return {"KNaCa": KNaCa}

    def _exchange(self, v, *, with_slope=False):
        thermal = self.R * (self.celsius + 273.15)
        saturation = 1.0 + self.DNaCa * (
            self.cai * self.nao * self.nao * self.nao
            + self.cao * self.nai * self.nai * self.nai
        )
        forward_gain = (self.r - 2.0) * self.gamma
        reverse_gain = (self.r - 2.0) * (self.gamma - 1.0)
        forward = (
            self.nai
            * self.nai
            * self.nai
            * self.cao
            * exp((forward_gain * v * self.F) / thermal)
        )
        reverse = (
            self.nao
            * self.nao
            * self.nao
            * self.cai
            * exp((reverse_gain * v * self.F) / thermal)
        )
        current = self.KNaCa * ((forward - reverse) / saturation)
        if not with_slope:
            return current
        forward_slope = forward * ((forward_gain * self.F) / thermal)
        reverse_slope = reverse * ((reverse_gain * self.F) / thermal)
        slope = self.KNaCa * ((forward_slope - reverse_slope) / saturation)
        return current, slope

    def ina(self, v):
        return 3.0 * self._exchange(v)

    def ina_with_conductance(self, v):
        current, slope = self._exchange(v, with_slope=True)
        return 3.0 * current, 3.0 * slope

    def ica(self, v):
        return -2.0 * self._exchange(v)

    def ica_with_conductance(self, v):
        current, slope = self._exchange(v, with_slope=True)
        return -2.0 * current, -2.0 * slope
