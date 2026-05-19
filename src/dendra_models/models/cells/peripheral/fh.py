from dendra.models.core import Unmyelinated, Myelinated
from dendra.units import mm

from .mech import fh


class FHM(Myelinated):
    Myelinated.RANGE(cm=2.0, rhoa=110.0)

    def __init__(
        self,
        diameters=[10.0],
        n_node=101,
        node_length=2.5,
        celsius=20.0,
        v_init=-70.0,
        integrator=None,
    ):
        super().__init__(diameters, n_node, node_length, celsius, v_init, integrator)
        self.insert(fh)
        self.concentrations(nai0=13.74, nao0=114.5, ki0=120.0, ko0=2.5)


SENN = FHM


class FHUM(Unmyelinated):
    Unmyelinated.RANGE(cm=2.0, rhoa=110.0)

    def __init__(
        self,
        diameters=[2.0],
        L=5.0 * mm,
        dx=25.0,
        celsius=20.0,
        v_init=-70.0,
        integrator=None,
    ):
        super().__init__(diameters, L, dx, celsius, v_init, integrator)
        self.insert(fh)
        self.concentrations(nai0=13.74, nao0=114.5, ki0=120.0, ko0=2.5)
