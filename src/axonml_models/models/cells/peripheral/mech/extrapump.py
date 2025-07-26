from axonml.models.mechanisms._mechanism import Mechanism as M
from axonml.models.mechanisms._state import State as S
from axonml.models.mechanisms.ops import *


class extrapump(M):
    M.GLOBAL(pumpik=0.0, pumpina=0.0, pumpica=0.0)
    M.USEION("k", write=["ik"])
    M.USEION("na", write=["ina"])
    M.USEION("ca", write=["ica"])
    M.EXPLICIT("ik", "ina", "ica")

    def ik(self, v):
        return self.pumpik

    def ina(self, v):
        return self.pumpina

    def ica(self, v):
        return self.pumpica
