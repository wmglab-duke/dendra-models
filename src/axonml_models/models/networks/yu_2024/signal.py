from typing import Optional

import torch
import torch.nn.functional as F


def _sosfilt_torch(sos: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Direct Form II (transposed) SOS IIR filter in pure PyTorch.
    Differentiable w.r.t. x. Assumes a0 == 1 for each section.

    Parameters
    ----------
    sos : (S, 6) tensor
        Each row is [b0, b1, b2, a0(=1), a1, a2].
    x : (..., T) tensor
        Signal along the last dimension.

    Returns
    -------
    y : (..., T) tensor
    """
    if sos.ndim != 2 or sos.shape[-1] != 6:
        raise ValueError("sos must have shape (n_sections, 6)")
    if x.ndim < 1:
        raise ValueError("x must have at least 1 dimension")

    y = x
    # Iterate over sections; autograd flows through the recurrent ops
    for s in range(sos.shape[0]):
        b0, b1, b2, a0, a1, a2 = sos[s]
        if not torch.allclose(a0, torch.ones_like(a0)):
            # Normalize if needed (rare when using scipy.signal.butter(..., output='sos'))
            b0 = b0 / a0; b1 = b1 / a0; b2 = b2 / a0
            a1 = a1 / a0; a2 = a2 / a0

        # allocate state (z1, z2) per batch/sample prefix
        *batch, T = y.shape
        z1 = torch.zeros(*batch, dtype=y.dtype, device=y.device)
        z2 = torch.zeros_like(z1)

        out = []
        # Manual recurrent loop over time is fine; it preserves gradient.
        for t in range(T):
            xt = y[..., t]
            yt = b0 * xt + z1
            z1 = b1 * xt + z2 - a1 * yt
            z2 = b2 * xt - a2 * yt
            out.append(yt)
        y = torch.stack(out, dim=-1)
    return y


def _reflect_pad(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Reflect-pad along the last dimension (like SciPy filtfilt)."""
    if pad <= 0:
        return x
    if x.shape[-1] <= 1:
        raise ValueError("Input length must be > 1 to apply reflect padding.")
    # Reflect without repeating the edge value
    pre = x[..., 1:pad+1].flip(-1)
    post = x[..., -pad-1:-1].flip(-1)
    return torch.cat([pre, x, post], dim=-1)


def _sosfiltfilt_torch(sos: torch.Tensor, x: torch.Tensor, padlen: Optional[int] = None) -> torch.Tensor:
    """
    Zero-phase filtering: forward sosfilt, reverse, forward sosfilt, reverse.
    Uses reflection padding to mitigate edge transients (like SciPy).
    """
    n_sections = sos.shape[0]
    # A conservative pad length: similar scale to SciPy defaults
    if padlen is None:
        padlen = max(1, 3 * 2 * n_sections)  # 3*(filter order); 2 taps per biquad stage

    if x.shape[-1] <= padlen:
        raise ValueError(f"Input length {x.shape[-1]} must be > padlen {padlen}.")

    xp = _reflect_pad(x, padlen)
    y = _sosfilt_torch(sos, xp)
    y = torch.flip(y, dims=[-1])
    y = _sosfilt_torch(sos, y)
    y = torch.flip(y, dims=[-1])

    # remove padding
    return y[..., padlen:-padlen]


def butter_bandpass_filter_torch(
    data: torch.Tensor,
    lowcut: float,
    highcut: float,
    fs: float,
    order: int = 2,
    *,
    sos: Optional[torch.Tensor] = None,
    padlen: Optional[int] = None,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Butterworth bandpass + zero-phase filtering (sosfiltfilt) in PyTorch.
    Differentiable w.r.t. `data`.

    This computes the *filtering* in PyTorch. By default it uses SciPy once to
    design the SOS coefficients (not part of autograd). If you already have
    coefficients, pass them via `sos` to avoid SciPy entirely.

    Parameters
    ----------
    data : (..., T) torch.Tensor
        Input signal along the last dimension.
    lowcut : float
        Low cutoff frequency in Hz.
    highcut : float
        High cutoff frequency in Hz.
    fs : float
        Sampling rate in Hz.
    order : int, default 2
        Butterworth order (per SciPy, overall order is `order`).
    sos : (S, 6) torch.Tensor, optional
        Precomputed SOS coefficients [b0, b1, b2, a0, a1, a2] per row.
        If provided, `lowcut`, `highcut`, `fs`, `order` are used only for doc/logging.
    padlen : int, optional
        Reflection pad length used in filtfilt. Defaults to a value based on sections.
    dtype, device : optional
        Overrides for internal dtype/device. Defaults to those of `data`.

    Returns
    -------
    y : (..., T) torch.Tensor
        Zero-phase filtered signal; gradients propagate w.r.t. `data`.
    """
    if dtype is None:
        dtype = data.dtype
    if device is None:
        device = data.device

    if sos is None:
        # Design once via SciPy (not part of the autograd graph).
        try:
            from scipy import signal as _sp_signal
        except Exception as e:
            raise RuntimeError(
                "SciPy is required to DESIGN the Butterworth SOS (or pass `sos=`). "
                "Install scipy or precompute the SOS and pass it in."
            ) from e

        nyq = 0.5 * fs
        wn = [lowcut / nyq, highcut / nyq]
        sos_np = _sp_signal.butter(order, wn, btype="bandpass", output="sos")
        sos = torch.tensor(sos_np, dtype=dtype, device=device)
    else:
        sos = sos.to(device=device, dtype=dtype)

    data = data.to(device=device, dtype=dtype)
    return _sosfiltfilt_torch(sos, data, padlen=padlen)


def _as_bt1(x: torch.Tensor) -> torch.Tensor:
    """Ensure shape [B,1,T] without copying."""
    if x.dim() == 1:      # [T]
        return x.unsqueeze(0).unsqueeze(0)
    if x.dim() == 2:      # [B,T]
        return x.unsqueeze(1)
    if x.dim() == 3 and x.size(1) == 1:  # [B,1,T]
        return x
    raise ValueError("Vm must be [T], [B,T], or [B,1,T].")

def _conv1d_same(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Reflect-pad 'same' 1D conv; k is [K] (symmetric not required)."""
    k = k.to(device=x.device, dtype=x.dtype).view(1, 1, -1)
    pad = k.shape[-1] // 2
    x = F.pad(x, (pad, pad), mode="reflect")
    return F.conv1d(x, k)

def point_gain(a_m: float, R_m: float, sigma_Spm: float) -> float:
    """
    Geometric/conductive scale for a small sphere/point source:
        gain = a^2 / (sigma * R)
    Returns a dimensionless factor that maps membrane current density (A/m^2)
    times area into extracellular potential (V). For shape-only use, set gain=1.0.
    """
    return (a_m * a_m) / (sigma_Spm * R_m)

def lfp_raw_from_vm(
    Vm: torch.Tensor,
    dt: float,
    *,
    Cm: float = 1.0,               # keep =1.0 if you only care about shape
    gL: Optional[float] = None,    # optional leak (S/m^2) if you want it
    EL: Optional[float] = None,    # leak reversal (V)
    gain: float = 1.0,             # = a^2/(sigma*R), or 1.0 for shape-only
    smooth_sigma: float = 0.0      # optional pre-derivative Gaussian (s); 0 = off
) -> torch.Tensor:
    """
    Vm-only, differentiable 'raw LFP' proxy (no band-limiting).
    i_m(t) ≈ Cm * dV/dt [+ gL*(V-EL) if provided], then phi(t) = gain * i_m(t).

    Args
    ----
    Vm : [T], [B,T], or [B,1,T] tensor
    dt : sample interval (s)
    Cm : specific capacitance scale (use 1.0 for shape)
    gL, EL : optional leak term; ignored if gL or EL is None
    gain : point/sphere forward-model scale; use 1.0 for shape-only
    smooth_sigma : Gaussian pre-smoothing std (s) before d/dt; helps with hard resets
    """
    x = _as_bt1(Vm)
    dtype, device = x.dtype, x.device

    # Optional very gentle pre-smoothing to tame spike resets (still differentiable).
    if smooth_sigma and smooth_sigma > 0.0:
        radius = max(1, int(4.0 * smooth_sigma / dt))
        n = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        g = torch.exp(-0.5 * (n * dt / smooth_sigma) ** 2)
        g = g / g.sum()
        x = _conv1d_same(x, g)

    # Centered finite difference derivative (linear, differentiable op).
    dker = torch.tensor([-0.5, 0.0, 0.5], dtype=dtype, device=device) / dt
    dV = _conv1d_same(x, dker)

    i_m = Cm * dV
    if gL is not None and EL is not None:
        i_m = i_m + float(gL) * (x - float(EL))

    phi = float(gain) * i_m

    # restore original shape
    if Vm.dim() == 1:
        return phi[0, 0]
    if Vm.dim() == 2:
        return phi[:, 0]
    return phi  # [B,1,T]


def render_butterworth_loss(net, rec, sos, reference, tstop=13.0, tstart=0.0, dt=0.025):
    x = rec.stack('v')[:, *net.cells.L5.index].sum(axis=1)
    # x = x - x[:, 0]
    x = lfp_raw_from_vm(x, dt, smooth_sigma=0.01).unsqueeze(0)
    y = butter_bandpass_filter_torch(x, sos=sos)

    t = torch.arange(0, tstop+dt, dt)
    idx = torch.where(t >= tstart)[0][0]
    y = y[0, idx:idx+len(reference)]
    
    y = y / torch.max(torch.abs(y))

    return y.detach().numpy(), torch.mean(torch.square(reference - y))
