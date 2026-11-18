# McIntyre, Richardson, Grill 2002

import torch

from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import expit, exprelr


def _vtrap(displacement, scale):
    # AXNODE_myel.mod uses a constant inside this window. exprelr also keeps
    # the discarded branch finite at zero so it cannot poison gradients.
    near_zero = torch.abs(displacement / scale) < 1e-6
    return torch.where(near_zero, scale, exprelr(displacement, scale))


class m(S):
    S.DERIVED_BUFFER("q10")
    S.STATE("m")
    S.GLOBAL_SIGNED(
        amA=1.86,
        amB=21.4,
        amC=10.3,
        bmA=0.086,
        bmB=25.7,
        bmC=9.16,
        aq10_1=2.2,
        bq10=20.0,
        cq10=10.0,
    )
    S.DERIVATIVE("m' = (minf - m) / mtau")
    S.ASSIGNED("minf", "mtau")

    def derive_buffers(self):
        return {"q10": self.aq10_1 ** ((self.celsius - self.bq10) / self.cq10)}

    def vtrap6(self, v):
        limited = v < -150.0
        voltage = torch.where(limited, -150.0, v)
        rate = self.amA * _vtrap(-(voltage + self.amB), self.amC)
        return torch.where(limited, 0.15733, rate)

    def vtrap7(self, v):
        limited = v > 150.0
        voltage = torch.where(limited, 150.0, v)
        rate = self.bmA * _vtrap(voltage + self.bmB, self.bmC)
        return torch.where(limited, 0.0057268, rate)

    def alpha(self, v):
        return self.q10 * self.vtrap6(v)

    def beta(self, v):
        return self.q10 * self.vtrap7(v)

    def assigned_values(self, v, values):
        am = self.alpha(v)
        bm = self.beta(v)
        mtau = 1 / (am + bm)
        minf = am * mtau
        return {"mtau": mtau, "minf": minf}

    def state_defaults(self, v, values):
        return {"m": self.assigned_values(v, values)["minf"]}


class p(S):
    S.DERIVED_BUFFER("q10")
    S.STATE("p")
    S.GLOBAL_SIGNED(
        ampA=0.01,
        ampB=27.0,
        ampC=10.2,
        bmpA=0.00025,
        bmpB=34.0,
        bmpC=10.0,
        pq10_1=2.2,
        bq10=20.0,
        cq10=10.0,
    )
    S.DERIVATIVE("p' = (pinf - p) / ptau")
    S.ASSIGNED("pinf", "ptau")

    def derive_buffers(self):
        return {"q10": self.pq10_1 ** ((self.celsius - self.bq10) / self.cq10)}

    def vtrap1(self, v):
        limited = v < -150.0
        voltage = torch.where(limited, -150.0, v)
        rate = self.ampA * _vtrap(-(voltage + self.ampB), self.ampC)
        return torch.where(limited, 0.00086725, rate)

    def vtrap2(self, v):
        limited = v > 150.0
        voltage = torch.where(limited, 150.0, v)
        rate = self.bmpA * _vtrap(voltage + self.bmpB, self.bmpC)
        return torch.where(limited, 1.5855e-05, rate)

    def alpha(self, v):
        return self.q10 * self.vtrap1(v)

    def beta(self, v):
        return self.q10 * self.vtrap2(v)

    def assigned_values(self, v, values):
        amp = self.alpha(v)
        bmp = self.beta(v)
        ptau = 1 / (amp + bmp)
        pinf = amp * ptau
        return {"ptau": ptau, "pinf": pinf}

    def state_defaults(self, v, values):
        return {"p": self.assigned_values(v, values)["pinf"]}


class h(S):
    S.DERIVED_BUFFER("q10")
    S.STATE("h")
    S.GLOBAL_SIGNED(
        ahA=0.062,
        ahB=114.0,
        ahC=11.0,
        bhA=2.3,
        bhB=31.8,
        bhC=13.4,
        aq10_2=2.9,
        bq10=20.0,
        cq10=10.0,
    )
    S.DERIVATIVE("h' = (hinf - h) / htau")
    S.ASSIGNED("hinf", "htau")

    def derive_buffers(self):
        return {"q10": self.aq10_2 ** ((self.celsius - self.bq10) / self.cq10)}

    def vtrap8(self, v):
        limited = v > 150.0
        voltage = torch.where(limited, 150.0, v)
        rate = self.ahA * _vtrap(voltage + self.ahB, self.ahC)
        return torch.where(limited, 0.0032594, rate)

    def vtrap9(self, v):
        rate = self.bhA * expit((v + self.bhB) / self.bhC)
        return torch.where(v < -150.0, 0.0014054, rate)

    def alpha(self, v):
        return self.q10 * self.vtrap8(v)

    def beta(self, v):
        return self.q10 * self.vtrap9(v)

    def assigned_values(self, v, values):
        ah = self.alpha(v)
        bh = self.beta(v)
        htau = 1 / (ah + bh)
        hinf = ah * htau
        return {"htau": htau, "hinf": hinf}

    def state_defaults(self, v, values):
        return {"h": self.assigned_values(v, values)["hinf"]}


class s(S):
    S.DERIVED_BUFFER("q10")
    S.STATE("s")
    S.GLOBAL_SIGNED(
        asA=0.3,
        asB=-27.0,
        asC=-5.0,
        bsA=0.03,
        bsB=10.0,
        bsC=-1.0,
        aq10_3=3.0,
        bq10=36.0,
        cq10=10.0,
        vtraub=-80.0,
    )
    S.DERIVATIVE("s' = (sinf - s) / stau")
    S.ASSIGNED("sinf", "stau")

    def derive_buffers(self):
        return {"q10": self.aq10_3 ** ((self.celsius - self.bq10) / self.cq10)}

    def vtrap10(self, v):
        rate = self.asA * expit((self.vtraub - v - self.asB) / self.asC)
        return torch.where(v < -150.0, 3.3484e-05, rate)

    def vtrap11(self, v):
        rate = self.bsA * expit((self.vtraub - v - self.bsB) / self.bsC)
        return torch.where(v < -150.0, 3.3484e-06, rate)

    def alpha(self, v):
        return self.q10 * self.vtrap10(v)

    def beta(self, v):
        return self.q10 * self.vtrap11(v)

    def assigned_values(self, v, values):
        as_ = self.alpha(v)
        bs = self.beta(v)
        stau = 1 / (as_ + bs)
        sinf = as_ * stau
        return {"stau": stau, "sinf": sinf}

    def state_defaults(self, v, values):
        return {"s": self.assigned_values(v, values)["sinf"]}


class axnode_myel(M):
    M.STATE_BUNDLE(m, p, h, s)
    M.GLOBAL_SIGNED(
        gnabar=3.0, gnapbar=0.01, gkbar=0.08, gl=0.007, ena=50.0, ek=-90.0, el=-90.0
    )

    M.NONSPECIFIC_CURRENT("inap", "ina", "ik", "il")

    def ina(self, v):
        return self.gnabar * self.m**3 * self.h * (v - self.ena)

    def inap(self, v):
        return self.gnapbar * self.p**3 * (v - self.ena)

    def ik(self, v):
        return self.gkbar * self.s * (v - self.ek)

    def il(self, v):
        return self.gl * (v - self.el)
