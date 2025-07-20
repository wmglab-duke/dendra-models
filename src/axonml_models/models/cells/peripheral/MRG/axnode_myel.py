# McIntyre, Richardson, Grill 2002

from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S
from axonml.models.mechanisms.ops import *


def Exp(x):
    return torch.where(x < -100.0, torch.tensor(0.0, dtype=x.dtype, device=x.device), torch.exp(x))


class m(S):
    has_q10 = True
    S.STATE("m")
    S.GLOBAL(
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

    def calc_q10(self):
        return self.aq10_1 ** ((self.celsius - self.bq10) / self.cq10)

    def vtrap6(self, v):
        cond = (v + self.amB) / self.amC
        out = (self.amA * (v + self.amB)) / (1.0 - Exp(-cond))
        out = torch.where(
            torch.abs(cond) < 1e-6, self.amA * self.amC, out
        )
        guard = torch.tensor(0.15733, dtype=out.dtype, device=out.device)
        out = torch.where(v < -150.0, guard, out)
        return out

    def vtrap7(self, v):
        cond = (v + self.bmB) / self.bmC
        out = (self.bmA * -(v + self.bmB)) / (1.0 - Exp(cond))
        out = torch.where(
            torch.abs(cond) < 1e-6, self.bmA * self.bmC, out
        )
        guard = torch.tensor(0.0057268, dtype=out.dtype, device=out.device)
        out = torch.where(v > 150.0, guard, out)
        return out

    def alpha(self, v):
        return self.q10() * self.vtrap6(v)

    def beta(self, v):
        return self.q10() * self.vtrap7(v)

    def breakpoint(self, v):
        am = self.alpha(v)
        bm = self.beta(v)
        mtau = 1 / (am + bm)
        minf = am * mtau
        return {"mtau": mtau, "minf": minf}
    
    def inf(self, v):
        return {"m": self.breakpoint(v)["minf"]}


class p(S):
    has_q10 = True
    S.STATE("p")
    S.GLOBAL(
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

    def calc_q10(self):
        return self.pq10_1 ** ((self.celsius - self.bq10) / self.cq10)

    def vtrap1(self, v):
        cond = (v + self.ampB) / self.ampC
        out = (self.ampA * (v + self.ampB)) / (1.0 - Exp(-cond))
        out = torch.where(
            torch.abs(cond) < 1e-6, self.ampA * self.ampC, out
        )
        guard = torch.tensor(0.00086725, dtype=out.dtype, device=out.device)
        out = torch.where(v < -150.0, guard, out)
        return out

    def vtrap2(self, v):
        cond = (v + self.bmpB) / self.bmpC
        out = (self.bmpA * -(v + self.bmpB)) / (1.0 - Exp(cond))
        out = torch.where(
            torch.abs(cond) < 1e-6, self.bmpA * self.bmpC, out
        )
        guard = torch.tensor(1.5855e-05, dtype=out.dtype, device=out.device)
        out = torch.where(v > 150.0, guard, out)
        return out
    
    def alpha(self, v):
        return self.q10() * self.vtrap1(v)

    def beta(self, v):
        return self.q10() * self.vtrap2(v)

    def breakpoint(self, v):
        amp = self.alpha(v)
        bmp = self.beta(v)
        ptau = 1 / (amp + bmp)
        pinf = amp * ptau
        return {"ptau": ptau, "pinf": pinf}
    
    def inf(self, v):
        return {"p": self.breakpoint(v)["pinf"]}


class h(S):
    has_q10 = True
    S.STATE("h")
    S.GLOBAL(
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

    def calc_q10(self):
        return self.aq10_2 ** ((self.celsius - self.bq10) / self.cq10)

    def vtrap8(self, v):
        cond = (v + self.ahB) / self.ahC
        out = (self.ahA * -(v + self.ahB)) / (1.0 - Exp(cond))
        out = torch.where(
            torch.abs(cond) < 1e-6, self.ahA * self.ahC, out
        )
        guard = torch.tensor(0.0032594, dtype=out.dtype, device=out.device)
        out = torch.where(v > 150.0, guard, out)
        return out

    def vtrap9(self, v):
        out = self.bhA / (1.0 + Exp(-(v + self.bhB) / self.bhC))
        guard = torch.tensor(0.0014054, dtype=out.dtype, device=out.device)
        return torch.where(v < -150.0, guard, out)

    def alpha(self, v):
        return self.q10() * self.vtrap8(v)

    def beta(self, v):
        return self.q10() * self.vtrap9(v)

    def breakpoint(self, v):
        ah = self.alpha(v)
        bh = self.beta(v)
        htau = 1 / (ah + bh)
        hinf = ah * htau
        return {"htau": htau, "hinf": hinf}
    
    def inf(self, v):
        return {"h": self.breakpoint(v)["hinf"]}


class s(S):
    has_q10 = True
    S.STATE("s")
    S.GLOBAL(
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

    def calc_q10(self):
        return self.aq10_3 ** ((self.celsius - self.bq10) / self.cq10)

    def vtrap10(self, v):
        guard = torch.tensor(3.3484e-05, dtype=v.dtype, device=v.device)
        out = self.asA / (1.0 + Exp((v - self.vtraub + self.asB) / self.asC))
        return torch.where(v < -150.0, guard, out)

    def vtrap11(self, v):
        guard = torch.tensor(3.3484e-06, dtype=v.dtype, device=v.device)
        out = self.bsA / (1.0 + Exp((v - self.vtraub + self.bsB) / self.bsC))
        return torch.where(v < -150.0, guard, out)
    
    def alpha(self, v):
        return self.q10() * self.vtrap10(v)

    def beta(self, v):
        return self.q10() * self.vtrap11(v)

    def breakpoint(self, v):
        as_ = self.alpha(v)
        bs = self.beta(v)
        stau = 1 / (as_ + bs)
        sinf = as_ * stau
        return {"stau": stau, "sinf": sinf}
    
    def inf(self, v):
        return {"s": self.breakpoint(v)["sinf"]}


class axnode_myel(M):
    M.STATE(m, p, h, s)
    M.GLOBAL(
        gnabar=3.0, gnapbar=0.01, gkbar=0.08, 
        gl=0.007, ena=50.0, ek=-90.0, el=-90.0
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
