from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms.ops import *


class pas(M):
    M.PARAMETER(g=0.001, e=-70.0)
    M.NONSPECIFIC_CURRENT("i")

    def i(self, v):
        return self.g * (v - self.e)
