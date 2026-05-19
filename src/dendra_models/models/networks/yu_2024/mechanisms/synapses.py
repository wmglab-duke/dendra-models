from dendra.models.mod import exp2syn
from dendra.models.mechanisms.ops import *


class exp2NMDA(exp2syn):
    exp2syn.GLOBAL(eta=0.2801, gamma=0.062, tau1=0.6, tau2=55.0, Mg=1.0)
    exp2syn.EXPLICIT("i")
    exp2syn.SAVE("i")

    def mgblock(self, v):
        return 1.0 / (1.0 + self.eta * self.Mg * exp(-self.gamma * v))

    def i(self, v):
        return (self.B - self.A) * self.mgblock(v) * (v - self.e)
