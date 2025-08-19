from importlib.resources import files, as_file

import torch

from axonml.models.core import Myelinated
from axonml.models.integrators import eulerv1

from ..mech import axnode_myel


ic = {"m": 0.0732093, "h": 0.62069505, "p": 0.20260409, "s": 0.04302994}


class SMF(Myelinated):
    """
    Surrogate Myelinated Fiber (S-MF) model from Hussain et al. (2024).

    This model implements a highly efficient GPU-based surrogate for the McIntyre-Richardson-Grill (MRG)
    myelinated axon model, designed to accurately replicate neural responses to electrical stimulation
    while achieving exceptional computational efficiency. The model uses simplified myelinated cable
    geometry with reparameterized non-linear ionic conductances to reproduce the behavior of the
    detailed NEURON implementation with orders-of-magnitude faster performance.

    Notes
    -----
    The S-MF model is designed to enable rapid exploration of electrical stimulation parameters
    while maintaining high accuracy compared to the gold-standard MRG model. It provides
    up to 1,000,000x speedup over single-core simulations in NEURON while accurately predicting:

    - Full spatiotemporal responses of nerve fibers to electrical stimulation
    - Action potential generation, propagation, and conduction velocity
    - Response to complex waveforms and electrode configurations
    - Nonlinear phenomena such as conduction block and subthreshold modulation
    - Dependence on prior excitation history

    Key features:
    - GPU-accelerated implementation under the AxonML framework
    - Compatible with monopolar and multipolar stimulation configurations
    - Handles various stimulus waveforms including conventional and kilohertz frequency signals
    - Supports different nerve morphologies and fiber diameters
    - Enables both gradient-free and gradient-based optimization approaches

    Applications:
    - Design of selective neural stimulation protocols
    - Optimization of electrode configurations and waveform parameters
    - Large-scale simulations of fiber populations
    - Development of neuromodulation therapies for various conditions
    - Studying complex nonlinear neural phenomena

    Limitations:
    - Minimum fiber diameter of 5.7 μm (use smolMRG for smaller fibers)
    - Limited to 'euler' or 'rk1' integration methods for computational efficiency
    - Simplified geometry compared to the full MRG model

    References
    ----------
    .. [1] M. A. Hussain, W. M. Grill, and N. A. Pelot, “Highly efficient modeling
           and optimization of neural fiber responses to electrical stimulation,”
           Nat Commun, vol. 15, no. 1, p. 7597, Aug. 2024, doi: 10.1038/s41467-024-51709-8.

    .. [2] McIntyre CC, Richardson AG, Grill WM (2002). Modeling the excitability of
           mammalian nerve fibers: influence of afterpotentials on the recovery cycle.
           Journal of Neurophysiology, 87(2), 995-1006.
    """

    Myelinated.RANGE(
        cm=9.352452121675014,
        rhoa=69.99446868896484,
    )
    Myelinated.GLOBAL(
        axon_d={
            "axond1": 0.0187623,
            "axond2": 4.787487e-01,
            "axond3": 1.203613e-01,
        },
        node_d={
            "noded1": 6.303781e-03,
            "noded2": 2.070544e-01,
            "noded3": 5.339006e-01,
        },
        delta_x={
            "deltax1": -8.215284e00,
            "deltax2": 2.724201e02,
            "deltax3": -7.802411e02,
        },
    )

    def __init__(
        self,
        diameters=[8.0],
        n_node=101,
        node_length=1.0,
        celsius=37.0,
        v_init=-80.0,
    ):
        if torch.any(torch.as_tensor(diameters) < 5.7):
            raise ValueError(
                "Fiber diameter should not be less than 5.7 um for SMF. Use smolMRG instead."
            )
        super().__init__(diameters, n_node, node_length, celsius, v_init, eulerv1())
        self.insert(axnode_myel, ic=ic)
        self.build()
        with as_file(files(__package__) / "SMF.pt") as p:
            self.load(str(p))
