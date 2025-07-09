from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms.ops import *


class pas(M):
    M.PARAMETER(e=-70.0)
    M.RANGE(g=0.001)
    M.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g * (v - self.e)
