from dendra.models.core import Unmyelinated
from dendra.units import mm

from ..mech import (
    ks,
    kf,
    mh,
    nattxs,
    nav1p8,
    nav1p9,
    nakpump,
    kdrTiger,
    kna,
    naoiTiger,
    koiTiger,
    leak,
    extrapump,
)
from ._balance import register_tigerholm_balance


class Tigerholm2014(Unmyelinated):
    """
    Computational model of unmyelinated C-fiber nociceptors based on Tigerholm et al. (2014).

    This model implements the biophysically detailed C-fiber axon described in
    "Modeling activity-dependent changes of axonal spike conduction in primary
    afferent C-nociceptors" (Tigerholm et al., 2014). The model includes multiple
    voltage-gated ion channels, ion accumulation/diffusion mechanisms, and pump
    dynamics that govern the excitability and conduction properties of unmyelinated
    nociceptive axons.

    Parameters
    ----------
    diameters : list of float, optional
        Axon diameter(s) in μm. Default is [1.0].
    L : float, optional
        Length of axon in mm. Default is 5.0.
    dx : float, optional
        Spatial discretization step in μm. Default is 10.
    celsius : float, optional
        Temperature in °C. Default is 37.0.
    v_init : float, optional
        Initial membrane potential in mV. Default is -55.0.
    method : str, optional
        Numerical integration method. Default is "dufort-frankel".

    Attributes
    ----------
    cm : float
        Specific membrane capacitance in μF/cm². Default is 1.0.
    rhoa : float
        Axoplasmic resistivity in Ω·cm. Default is 35.4.

    Notes
    -----
    The model includes the following ion channels and mechanisms:

    - ks: Slow potassium channel
    - kf: Fast potassium channel
    - h: Hyperpolarization-activated cyclic nucleotide-gated (HCN) channel
    - nattxs: TTX-sensitive sodium channel
    - nav1p8: Voltage-gated sodium channel Nav1.8
    - nav1p9_slow_inact: Voltage-gated sodium channel Nav1.9 with slow inactivation
    - nakpump: Na⁺/K⁺ ATPase pump
    - kdrTiger: Delayed rectifier potassium channel
    - kna: Sodium-activated potassium channel
    - naoiTiger: Sodium ion accumulation/diffusion mechanism
    - koiTiger: Potassium ion accumulation/diffusion mechanism
    - leak: Background leak conductances for Na⁺ and K⁺

    This model is particularly useful for studying:
    - Activity-dependent slowing (ADS) of conduction velocity
    - Effects of repetitive stimulation on nociceptor excitability
    - Pain signaling in unmyelinated C-fibers

    References
    ----------
    .. [1] Tigerholm J, Petersson ME, Obreja O, Lampert A, Carr R, Schmelz M,
           Fransén E (2014). Modeling activity-dependent changes of axonal spike
           conduction in primary afferent C-nociceptors. J Neurophysiol 111(9):
           1721-35. doi:10.1152/jn.00777.2012
    """

    Unmyelinated.RANGE(cm=1.0, rhoa=35.4)

    def __init__(
        self,
        diameters=[1.0],
        L=5.0 * mm,
        dx=10.0,
        celsius=37.0,
        v_init=-55.0,
        integrator=None,
    ):
        super().__init__(diameters, L, dx, celsius, v_init, integrator)

        self.insert(ks, gbar=0.0069733)  # KM
        self.insert(kf, gbar=0.012756)  # KA
        self.insert(mh, gbar=0.0025377)
        self.insert(nattxs, gbar=0.10664)  # Nav1.7
        self.insert(nav1p8, gbar=0.24271)
        self.insert(nav1p9, gbar=9.4779e-05)
        self.insert(nakpump, smalla=-0.0047891)
        self.insert(kdrTiger, gbar=0.018002)
        self.insert(kna, gbar=0.00042)
        self.insert(naoiTiger)
        self.insert(koiTiger)
        self.insert(leak)
        self.insert(extrapump)
        register_tigerholm_balance(self)

        self.concentrations(nai0=11.4, nao0=154.0, ki0=121.7, ko0=5.6)
