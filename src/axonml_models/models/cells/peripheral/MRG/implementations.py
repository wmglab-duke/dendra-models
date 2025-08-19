from typing import Callable, Sequence, Union, Optional
import warnings

import torch

from axonml.models.heterogeneous.compartments import CompartmentID
from axonml.models.extcell import ExtCellAxon
from axonml.models.mod import pas

from .axnode_myel import axnode_myel


ns = ["node", "mysa", "flut", "stin * 6", "flut", "mysa"]


class mrg_cm(torch.nn.Module):
    def __init__(self, secd, fd):
        super().__init__()
        self.register_buffer("secd", secd)
        self.register_buffer("fd", fd)

    def forward(self, cm):
        return cm * self.secd / self.fd


class mrg_rhoa(torch.nn.Module):
    def __init__(self, secd, fd):
        super().__init__()
        self.register_buffer("secd", secd)
        self.register_buffer("fd", fd)

    def forward(self, rhoa):
        return rhoa * (1 / (self.secd / self.fd)) ** 2


class MRG(ExtCellAxon):
    """
    A model of a large myelinated axon with multiple nodes of Ranvier.
    This model is based on the MRG model by McIntyre et al. (2002).
    """

    ExtCellAxon.PARAMETER(cm=2.0, rhoa=70.0)

    def __init__(
        self,
        diameters=[8.0],
        n_node=101,
        celsius=37.0,
        v_init=-80.0,
        integrator=None,
    ):
        cid = CompartmentID(ns, n_node - 1)
        n_ax = len(diameters)
        n_c = cid.nc()

        super().__init__(diameters, n_c, celsius, v_init, integrator=integrator)
        self.register_cid(cid)

        fd = self.diameters.unsqueeze(1)

        nl = self.__class__.nl(fd)
        axonD = self.__class__.axonD(fd)
        nodeD = self.__class__.nodeD(fd)

        deltax = self.__class__.deltax(fd)
        nodelength0 = self.__class__.nodelength0(fd)
        paralength1 = self.__class__.paralength1(fd)
        paralength2 = self.__class__.paralength2(fd)

        interlength = (deltax - nodelength0 - (2 * paralength1) - (2 * paralength2)) / 6

        rhoa = 0.7e6
        space_p1 = 0.002
        space_p2 = 0.004
        space_i = 0.004

        rpn0 = (rhoa * 0.01) / (
            torch.pi * ((((nodeD / 2) + space_p1) ** 2) - ((nodeD / 2) ** 2))
        )
        rpn1 = (rhoa * 0.01) / (
            torch.pi * ((((nodeD / 2) + space_p1) ** 2) - ((nodeD / 2) ** 2))
        )
        rpn2 = (rhoa * 0.01) / (
            torch.pi * ((((axonD / 2) + space_p2) ** 2) - ((axonD / 2) ** 2))
        )
        rpx = (rhoa * 0.01) / (
            torch.pi * ((((axonD / 2) + space_i) ** 2) - ((axonD / 2) ** 2))
        )

        diam = fd.expand(n_ax, n_c).clone()
        diam[:, self.find("node")] = nodeD
        self.diam[:] = diam

        dx = torch.zeros(n_ax, n_c)
        dx[:, self.find("node")] = nodelength0
        dx[:, self.find("mysa")] = paralength1
        dx[:, self.find("flut")] = paralength2
        dx[:, self.find("stin")] = interlength
        self.dx[:] = dx

        scale = torch.full((n_ax, n_c), 0.0001)
        scale[:, self.find("mysa")] = 0.001

        secd = fd.expand(n_ax, n_c).clone()
        secd[:, self.find("flut")] = axonD
        secd[:, self.find("mysa")] = nodeD
        secd[:, self.find("stin")] = axonD

        self.register_parametrization("cm", mrg_cm(secd, fd))
        self.register_parametrization("rhoa", mrg_rhoa(secd, fd))

        xc = (0.1 / (nl * 2)).expand(n_ax, n_c).clone()
        xc[:, self.find("node")] = 0.0

        xg = (0.001 / (nl * 2)).expand(n_ax, n_c).clone()
        xg[:, self.find("node")] = 1e10

        xraxial = torch.empty(n_ax, n_c).to(dtype=self.dtype(), device=self.device())
        xraxial[:, self.find("node")] = rpn0
        xraxial[:, self.find("flut")] = rpn2
        xraxial[:, self.find("stin")] = rpx
        xraxial[:, self.find("mysa")] = rpn1

        self.xc[..., 0] = xc.to(dtype=self.dtype(), device=self.device())
        self.xg[..., 0] = xg.to(dtype=self.dtype(), device=self.device())
        self.xraxial[..., 0] = xraxial

    def c(self, *args):
        locs = self.find("node", as_list=True)
        n = len(locs)
        return [locs[round((n - 1) * arg)] for arg in args]

    def steady_state(self, dt=1.0, tstop=200):
        return super().steady_state(dt, tstop)


class bigMRG(MRG):
    nl = lambda fd: torch.clamp(-0.4749 * fd**2 + 16.85 * fd - 0.7648, min=1)
    axonD = lambda fd: 0.02361 * fd**2 + 0.3673 * fd + 0.7122
    nodeD = lambda fd: 0.01093 * fd**2 + 0.1008 * fd + 1.099
    deltax = lambda fd: -8.215284e00 * fd**2 + 2.724201e02 * fd + -7.802411e02

    nodelength0 = lambda fd: 1.0
    paralength1 = lambda fd: 3.0
    paralength2 = lambda fd: -0.1652 * fd**2 + 6.354 * fd - 0.2862

    def __init__(
        self,
        diameters=[8.0],
        n_node=101,
        celsius=37.0,
        v_init=-80.0,
        integrator=None,
    ):
        if torch.any(torch.as_tensor(diameters) < 5.7):
            warnings.warn(
                "Fiber diameter should not be less than 5.7 um for bigMRG. Use smolMRG instead."
            )

        super().__init__(diameters, n_node, celsius, v_init, integrator)

        n_ax = self.n_ax

        fd = self.diameters.unsqueeze(1)
        nodeD = self.__class__.nodeD(fd)
        axonD = self.__class__.axonD(fd)

        n_mysa = self.mysa.numel() // n_ax
        node_scale = (nodeD / fd).expand(n_ax, n_mysa).flatten()

        n_stin = self.stin.numel() // n_ax
        stin_scale = (axonD / fd).expand(n_ax, n_stin).flatten()

        n_flut = self.flut.numel() // n_ax
        flut_scale = (axonD / fd).expand(n_ax, n_flut).flatten()

        self.flut.insert(pas, g=0.0001 * flut_scale, e=self.v_init)
        self.stin.insert(pas, g=0.0001 * stin_scale, e=self.v_init)
        self.mysa.insert(pas, g=0.001 * node_scale, e=self.v_init)

        self.node.insert(axnode_myel)

        self.x[:] = self._x()


class smolMRG(MRG):
    nl = lambda fd: torch.clamp(torch.floor(17.4 * (0.553 * fd - 0.024) - 1.74), min=1)
    axonD = lambda fd: 0.553 * fd - 0.024
    nodeD = lambda fd: 0.321 * smolMRG.axonD(fd) + 0.37
    deltax = lambda fd: -3.22 * fd**2 + 148 * fd - 128

    nodelength0 = lambda fd: 1.0
    paralength1 = lambda fd: 3.0
    paralength2 = lambda fd: -0.171 * fd**2 + 6.48 * fd - 0.935

    def __init__(
        self,
        diameters=[2.0],
        n_node=101,
        celsius=37.0,
        v_init=-80.0,
        integrator=None,
    ):
        if torch.any(
            (torch.as_tensor(diameters) > 5.7) | (torch.as_tensor(diameters) < 1.011)
        ):
            warnings.warn(
                "Fiber diameter should not be <1.011um or >5.7 um for smolMRG. Use bigMRG for larger fibers."
            )

        super().__init__(diameters, n_node, celsius, v_init, integrator)

        n_ax = self.n_ax

        fd = self.diameters.unsqueeze(1)
        nodeD = self.__class__.nodeD(fd)
        axonD = self.__class__.axonD(fd)

        n_mysa = self.mysa.numel() // n_ax
        node_scale = (nodeD / fd).expand(n_ax, n_mysa).flatten()

        n_stin = self.stin.numel() // n_ax
        stin_scale = (axonD / fd).expand(n_ax, n_stin).flatten()

        n_flut = self.flut.numel() // n_ax
        flut_scale = (axonD / fd).expand(n_ax, n_flut).flatten()

        self.flut.insert(pas, g=0.0001 * flut_scale, e=self.v_init)
        self.stin.insert(pas, g=0.0001 * stin_scale, e=self.v_init)
        self.mysa.insert(pas, g=0.001 * node_scale, e=self.v_init)

        self.node.insert(axnode_myel, gnabar=2.333333, gkbar=0.115556)

        self.x[:] = self._x()


Number = Union[int, float]


def make_substituter(
    keys: Sequence[Number],
    values: Sequence[Number],
    *,
    default: Optional[Number | str] = "raise",  # "identity" | "raise" | numeric fill
    tol: Optional[float] = None,  # for float keys: match nearest within atol
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Return a function that maps each occurrence of `keys[i]` to `values[i]`
    in a tensor `x` of ANY shape (ndim). Output has the same shape and device.

    - `default="identity"` leaves unmatched elements unchanged.
    - `default="raise"` raises KeyError if any element is unmatched.
    - `default=<number>` fills unmatched elements with that number.
    - If `tol` is set, float keys are matched to the *nearest* key within `atol=tol`.
    """
    k = torch.as_tensor(keys)
    v = torch.as_tensor(values)
    if k.ndim != 1 or v.ndim != 1 or k.numel() != v.numel():
        raise ValueError("`keys` and `values` must be 1D and the same length.")
    if torch.unique(k).numel() != k.numel():
        raise ValueError("`keys` must be unique.")

    # sort once so we can use searchsorted
    perm = torch.argsort(k)
    k_sorted = k[perm]
    v_sorted = v[perm]

    def substitute(x: torch.Tensor) -> torch.Tensor:
        # work on a flat view, then restore shape
        orig_shape = x.shape
        xx = x.reshape(-1)

        # compute in x's dtype/device
        kx = k_sorted.to(device=xx.device, dtype=xx.dtype)
        vx = v_sorted.to(device=xx.device, dtype=xx.dtype)

        # left insertion positions
        idx = torch.searchsorted(kx, xx)
        n = kx.numel()
        in_range_r = idx < n

        if tol is None:
            # exact match (no need to check left neighbor)
            matched = torch.zeros(xx.numel(), dtype=torch.bool, device=xx.device)
            probe_mask = in_range_r
            matched[probe_mask] = kx[idx[probe_mask]] == xx[probe_mask]
            chosen = idx  # valid only where matched==True
        else:
            # nearest-neighbor within atol=tol (check right and left candidates)
            x64 = xx.to(torch.float64)
            k64 = kx.to(torch.float64)

            # right candidate
            diff_r = torch.full_like(x64, float("inf"))
            mR = in_range_r
            diff_r[mR] = (k64[idx[mR]] - x64[mR]).abs()

            # left candidate
            diff_l = torch.full_like(x64, float("inf"))
            mL = idx > 0
            idxL = idx[mL] - 1
            diff_l[mL] = (k64[idxL] - x64[mL]).abs()

            # pick closer (ties → left)
            choose_left = diff_l <= diff_r
            chosen = idx.clone()
            chosen[choose_left] -= 1

            # matched if nearest is within tol
            nearest_diff = torch.minimum(diff_l, diff_r)
            matched = nearest_diff <= float(tol)

        out = xx.clone()
        if matched.any():
            out[matched] = vx[chosen[matched]]

        if not matched.all():
            if default == "raise":
                bad = xx[~matched][:8].tolist()
                raise KeyError(f"Unmapped values encountered (up to 8 shown): {bad}")
            elif default == "identity" or default is None:
                pass
            else:
                fill = torch.as_tensor(default, dtype=out.dtype, device=out.device)
                out[~matched] = fill

        return out.reshape(orig_shape)

    return substitute


class exactMRG(MRG):
    valid_diams = [1.0, 2.0, 5.7, 7.3, 8.7, 10.0, 11.5, 12.8, 14.0, 15.0, 16.0]
    nl_vals = [15, 30, 80, 100, 110, 120, 130, 135, 140, 145, 150]

    nl = make_substituter(valid_diams, nl_vals)

    axonD_vals = [0.8, 1.6, 3.4, 4.6, 5.8, 6.9, 8.1, 9.2, 10.4, 11.5, 12.7]
    nodeD_vals = [0.7, 1.4, 1.9, 2.4, 2.8, 3.3, 3.7, 4.2, 4.7, 5.0, 5.5]
    deltax_vals = [100, 200, 500, 750, 1000, 1150, 1250, 1350, 1400, 1450, 1500]

    axonD = make_substituter(valid_diams, axonD_vals)
    nodeD = make_substituter(valid_diams, nodeD_vals)
    deltax = make_substituter(valid_diams, deltax_vals)

    nodelength0 = lambda fd: 1.0
    paralength1 = lambda fd: 3.0
    paralength2_vals = [5, 10, 35, 38, 40, 46, 50, 54, 56, 58, 60]
    paralength2 = make_substituter(valid_diams, paralength2_vals)

    def __init__(
        self,
        diameters=[5.7],
        n_node=101,
        celsius=37.0,
        v_init=-80.0,
        integrator=None,
    ):
        valid_diams = torch.as_tensor(self.valid_diams)
        # Ensure that the diameters are valid
        if not torch.all(torch.isin(torch.as_tensor(diameters), valid_diams)):
            raise ValueError(
                f"Invalid diameters. Valid diameters are: {self.valid_diams}"
            )

        super().__init__(diameters, n_node, celsius, v_init, integrator)

        n_ax = self.n_ax

        fd = self.diameters.unsqueeze(1)
        nodeD = self.__class__.nodeD(fd)
        axonD = self.__class__.axonD(fd)

        n_mysa = self.mysa.numel() // n_ax
        node_scale = (nodeD / fd).expand(n_ax, n_mysa).flatten()

        n_stin = self.stin.numel() // n_ax
        stin_scale = (axonD / fd).expand(n_ax, n_stin).flatten()

        n_flut = self.flut.numel() // n_ax
        flut_scale = (axonD / fd).expand(n_ax, n_flut).flatten()

        self.flut.insert(pas, g=0.0001 * flut_scale, e=self.v_init)
        self.stin.insert(pas, g=0.0001 * stin_scale, e=self.v_init)
        self.mysa.insert(pas, g=0.001 * node_scale, e=self.v_init)

        self.node.insert(axnode_myel)

        self.x[:] = self._x()
