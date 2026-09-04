from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms._state import State as S
from dendra.models.mechanisms.ops import exp, log10


class m(S):
    S.DERIVED_BUFFER("q10")
    S.STATE("m")
    S.DERIVATIVE("m' = (minf - m) / taum")
    S.ASSIGNED("minf", "taum")
    S.GLOBAL_SIGNED(aq10=3.0)

    def derive_buffers(self):
        return {"q10": self.aq10 ** ((self.celsius - 22.0) / 10.0)}

    def assigned_values(self, v, values):
        q10 = self.q10
        pca = log10(self.cai) - 3.0
        v12 = -50.0 * pca - 232.0
        minf = 1.0 / (1.0 + exp(-1.0 * (v - v12) / 24.0))
        taum = (
            1.0
            / (
                exp((v + (58.0 * pca) + 303.0) / (3.2 * pca))
                + exp(-1.0 * (v + (107.0 * pca) + 453.0) / (6.8 * pca))
            )
            + 0.4
        )
        taum = taum / q10
        return {"taum": taum, "minf": minf}

    def initial_values(self, v, values):
        pca = log10(values["cai"]) - 3.0
        v12 = -50.0 * pca - 232.0
        return {"m": 1.0 / (1.0 + exp(-1.0 * (v - v12) / 24.0))}


class h(S):
    S.DERIVED_BUFFER("q10")
    S.STATE("h")
    S.DERIVATIVE("h' = (hinf - h) / tauh")
    S.ASSIGNED("hinf", "tauh")
    S.GLOBAL_SIGNED(aq10=3.0)

    def derive_buffers(self):
        return {"q10": self.aq10 ** ((self.celsius - 22.0) / 10.0)}

    def assigned_values(self, v, values):
        q10 = self.q10
        pca = log10(self.cai) - 3.0
        vh12 = -8.0 * pca + 35.0
        hinf = 1.0 / (1.0 + exp((v - vh12) / 47.0))
        tauh = 1.0 / (
            exp((v + (3.0 * pca) + 100.0) / (3.0 * pca))
            + exp(-1.0 * (v + (191.0 * pca) + 600.0) / (17.0 * pca))
        )
        tauh = tauh / q10
        return {"tauh": tauh, "hinf": hinf}

    def initial_values(self, v, values):
        pca = log10(values["cai"]) - 3.0
        vh12 = -8.0 * pca + 35.0
        return {"h": 1.0 / (1.0 + exp((v - vh12) / 47.0))}


class bk(M):
    M.STATE_BUNDLE(m, h)
    M.GLOBAL_SIGNED(gbar=0.0001)

    M.USEION("k", read=["ek"], write=["ik"])
    M.USEION("ca", read=["cai"])

    def ik(self, v):
        return self.gbar * self.m * self.h * (v - self.ek)


class bk_augmented(bk):
    bk.GLOBAL_SIGNED(aug=1.0)

    def ik(self, v):
        return self.gbar * self.m * self.h * (v - self.ek) * self.aug
