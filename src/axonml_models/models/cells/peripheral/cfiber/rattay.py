from axonml.models.core import Unmyelinated
from axonml.units import mm

from ..mech import rattay_aberham


class Rattay1993(Unmyelinated):
    Unmyelinated.RANGE(rhoa=100.0)

    def __init__(
        self,
        diameters=[1.0],
        L=5.0 * mm,
        dx=10,
        temp=37.0,
        v_init=-70.0,
        integrator=None,
    ):
        super().__init__(diameters, L, dx, temp, v_init, integrator)
        self.insert(rattay_aberham)
        self.equilibria(ena=45.0, ek=-82.0)
