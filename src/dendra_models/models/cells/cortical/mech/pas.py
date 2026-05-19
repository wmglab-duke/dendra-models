from dendra.models.mechanisms._mechanism import Mechanism as M
from dendra.models.mechanisms.ops import *


class pas(M):
    M.GLOBAL(e=-70.0)
    M.RANGEP(g=0.001)
    M.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g * (v - self.e)
