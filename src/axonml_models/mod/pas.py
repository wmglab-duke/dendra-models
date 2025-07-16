from ..mechanisms import *


class pas(Mechanism):
    GLOBAL(g=0.001, e=-70.0)

    NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g * (v - self.e)
