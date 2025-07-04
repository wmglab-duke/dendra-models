from axonml.models.core import Unmyelinated
from axonml.models.mechanisms import equilibria
from axonml.units import mm
from axonml.models.integrators import bwd_euler_ub

from ..mech import rattay_aberham


class Rattay1993(Unmyelinated):
    Unmyelinated.PARAMETER(rhoa=100.0)

    def __init__(
        self,
        diameters=[1.0],
        L=5.0*mm,
        dx=10,
        temp=37.0,
        v_init=-70.0,
        integrator=None,
    ):
        if integrator is None:
            integrator = bwd_euler_ub()
        super().__init__(diameters, L, dx, temp, v_init, integrator)

        self.insert(rattay_aberham)

        with equilibria(ena=45.0, ek=-82.0):
            self.build()

