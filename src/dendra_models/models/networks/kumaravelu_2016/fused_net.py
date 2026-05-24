"""Fused Dendra builder for the Kumaravelu et al. CTX-BG-TH network."""

from __future__ import annotations

from copy import deepcopy
from typing import Iterable, Mapping

import numpy as np

import dendra as dn

from .params import params as default_parameter_dict
from .mechanisms.fused import make_kumaravelu_2016_fused


def _copy_params(params=None):
    return deepcopy(default_parameter_dict if params is None else params)


def _as_rng(params, rng=None):
    if rng is not None:
        return rng
    return np.random.default_rng(params.get("seed", None))


def _validate_equal_n(params: Mapping) -> int:
    n_by_type = params["cells"]["n_by_type"]
    expected = int(params.get("n", next(iter(n_by_type.values()))))
    required = params["cells"]["hh_celltypes"] + params["cells"]["ctx_celltypes"]
    bad = {name: n_by_type[name] for name in required if int(n_by_type[name]) != expected}
    if bad:
        raise ValueError(
            "The fused Kumaravelu backend assumes the same n in every population. "
            f"Expected n={expected}, got {bad}."
        )
    return expected


def _as_1d(values, n: int, name: str, *, dtype=float):
    arr = np.asarray(values, dtype=dtype).reshape(-1)
    if arr.size != n:
        raise ValueError(f"{name} must have length n={n}; got length {arr.size}.")
    return arr


def _as_2d(values, N: int, n: int, name: str, *, dtype=float):
    """Return an array shaped (N, n), accepting shared length-n inputs."""
    arr = np.asarray(values, dtype=dtype)
    if arr.ndim == 0:
        return np.full((N, n), arr.item(), dtype=dtype)

    flat = arr.reshape(-1)
    if flat.size == n:
        return np.broadcast_to(flat.reshape(1, n), (N, n)).copy()
    if flat.size == N * n:
        return flat.reshape(N, n).copy()

    # Common MATLAB/scipy shapes such as (n, 1), (1, n), or already (N, n)
    # are handled by the flatten branches above.  This final branch is mostly
    # for explicit batch-like arrays with singleton leading axes.
    if arr.shape[-1] == n:
        arr2 = arr.reshape(-1, n)
        if arr2.shape[0] == 1:
            return np.broadcast_to(arr2, (N, n)).copy()
        if arr2.shape[0] == N:
            return arr2.copy()

    raise ValueError(
        f"{name} must be length n={n} or shape (N, n)=({N}, {n}); "
        f"got shape {tuple(arr.shape)}."
    )


def _normalize_perm_rows(arr: np.ndarray, n: int, name: str):
    out = np.asarray(arr, dtype=int).copy()
    if out.ndim == 1:
        out = out.reshape(1, n)
    if out.shape[-1] != n:
        raise ValueError(f"{name} must have last dimension n={n}; got shape {tuple(out.shape)}.")

    out = out.reshape(-1, n)
    for row_idx in range(out.shape[0]):
        row = out[row_idx]
        # MATLAB exports 1..n. Python/Dendra uses 0..n-1.
        if row.size and row.min() >= 1 and row.max() == n:
            row = row - 1
            out[row_idx] = row
        if sorted(row.tolist()) != list(range(n)):
            raise ValueError(
                f"{name}[{row_idx}] is not a valid permutation of 0..{n - 1} or 1..{n}."
            )
    return out


def _as_perm(values, n: int, name: str, *, N: int = 1):
    """Return zero-based permutations, accepting shared or per-network values."""
    arr = np.asarray(values)
    flat = arr.reshape(-1)
    if flat.size == n:
        return _normalize_perm_rows(flat, n, name).reshape(n) if N == 1 else np.broadcast_to(_normalize_perm_rows(flat, n, name), (N, n)).copy()
    if flat.size == N * n:
        return _normalize_perm_rows(flat.reshape(N, n), n, name).reshape(N, n)
    if arr.shape[-1] == n:
        arr2 = _normalize_perm_rows(arr.reshape(-1, n), n, name)
        if arr2.shape[0] == 1:
            return arr2.reshape(n) if N == 1 else np.broadcast_to(arr2, (N, n)).copy()
        if arr2.shape[0] == N:
            return arr2.reshape(N, n)
    raise ValueError(
        f"{name} must be length n={n} or shape (N, n)=({N}, {n}); "
        f"got shape {tuple(arr.shape)}."
    )


def _lookup(mapping: Mapping, *names):
    for name in names:
        if name in mapping:
            return mapping[name]
    raise KeyError(f"None of the fields {names!r} were found. Available fields: {tuple(mapping.keys())!r}")


def _lookup_optional(mapping: Mapping, *names):
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _normalize_external_initial(params: Mapping, names: Iterable[str], n_by_type: Mapping[str, int], N: int):
    """Return a v_init matrix when params supplies exact MATLAB/Dendra initials."""
    initial = params.get("initial", None)
    if initial is None:
        initial = params.get("v_init_override", None)
    if initial is None:
        return None

    # Current MATLAB validation schema stores voltages in validation.initial.v.*.
    # Older helpers used flat validation.initial.* fields.  Accept both.
    if isinstance(initial, Mapping) and "v" in initial and isinstance(initial["v"], Mapping):
        voltage_initial = initial["v"]
    else:
        voltage_initial = initial

    aliases = {
        "TH": ("TH", "v1", "vth"),
        "STN": ("STN", "v2", "vsn"),
        "GPe": ("GPe", "v3", "vge"),
        "GPi": ("GPi", "v4", "vgi"),
        "StrD2": ("StrD2", "Striat_indr", "Striatum_indirect", "v5", "vstr_indr"),
        "StrD1": ("StrD1", "Striat_dr", "Striatum_direct", "v6", "vstr_dr"),
        "CTX_RS": ("CTX_RS", "Cortex_RS", "Cor_RS", "ve"),
        "CTX_FS": ("CTX_FS", "Cortex_FS", "Cor_FS", "vi"),
    }

    groups = []
    for name in names:
        n = int(n_by_type[name])
        vals = _lookup(voltage_initial, *aliases.get(name, (name,)))
        groups.append(_as_2d(vals, N, n, f"initial[{name}]"))
    return np.concatenate(groups, axis=1)


def _perm_group(mapping: Mapping, n: int, keys: tuple[str, ...], label: str, *, N: int = 1):
    return [_as_perm(_lookup(mapping, key), n, f"{label}.{key}", N=N) for key in keys]


def _normalize_external_realization(realization: Mapping, n: int, N: int = 1):
    """Normalize MATLAB- or Dendra-named realization structs into Dendra fields.

    Accepted schemas:
      1. Dendra canonical:
           str_d2_perms, str_d1_perms, fs_to_rs_perms, rs_to_fs_perms, gains...
      2. Flat MATLAB:
           all,bll,cll,dll, ell..hll, ill..lll, mll,nll,oll, gains...
      3. Current MATLAB validation export:
           str_d2_gaba_perms.{all,bll,cll,dll}
           ctx_fs_to_rs_perms.{ell,fll,gll,hll}
           ctx_rs_to_fs_perms.{ill,jll,kll,lll}
           str_d1_gaba_perms.{mll,nll,oll}
           random_gains.{gcorsna,...,gsngi}
    """
    if realization is None:
        return None

    # Current nested MATLAB validation schema.
    if any(k in realization for k in (
        "str_d2_gaba_perms", "ctx_fs_to_rs_perms", "ctx_rs_to_fs_perms", "str_d1_gaba_perms"
    )):
        d2 = _lookup(realization, "str_d2_gaba_perms")
        fsrs = _lookup(realization, "ctx_fs_to_rs_perms")
        rsfs = _lookup(realization, "ctx_rs_to_fs_perms")
        d1 = _lookup(realization, "str_d1_gaba_perms")
        out = {
            "str_d2_perms": _perm_group(d2, n, ("all", "bll", "cll", "dll"), "realization.str_d2_gaba_perms", N=N),
            "fs_to_rs_perms": _perm_group(fsrs, n, ("ell", "fll", "gll", "hll"), "realization.ctx_fs_to_rs_perms", N=N),
            "rs_to_fs_perms": _perm_group(rsfs, n, ("ill", "jll", "kll", "lll"), "realization.ctx_rs_to_fs_perms", N=N),
            "str_d1_perms": _perm_group(d1, n, ("mll", "nll", "oll"), "realization.str_d1_gaba_perms", N=N),
        }
        gains_src = _lookup_optional(realization, "random_gains") or realization

    # Flat MATLAB names.
    elif any(k in realization for k in ("all", "bll", "cll", "dll")):
        out = {
            "str_d2_perms": _perm_group(realization, n, ("all", "bll", "cll", "dll"), "realization", N=N),
            "fs_to_rs_perms": _perm_group(realization, n, ("ell", "fll", "gll", "hll"), "realization", N=N),
            "rs_to_fs_perms": _perm_group(realization, n, ("ill", "jll", "kll", "lll"), "realization", N=N),
            "str_d1_perms": _perm_group(realization, n, ("mll", "nll", "oll"), "realization", N=N),
        }
        gains_src = realization

    # Canonical Dendra names.
    else:
        out = {
            "str_d2_perms": [
                _as_perm(p, n, f"realization.str_d2_perms[{i}]", N=N)
                for i, p in enumerate(_lookup(realization, "str_d2_perms"))
            ],
            "str_d1_perms": [
                _as_perm(p, n, f"realization.str_d1_perms[{i}]", N=N)
                for i, p in enumerate(_lookup(realization, "str_d1_perms"))
            ],
            "fs_to_rs_perms": [
                _as_perm(p, n, f"realization.fs_to_rs_perms[{i}]", N=N)
                for i, p in enumerate(_lookup(realization, "fs_to_rs_perms"))
            ],
            "rs_to_fs_perms": [
                _as_perm(p, n, f"realization.rs_to_fs_perms[{i}]", N=N)
                for i, p in enumerate(_lookup(realization, "rs_to_fs_perms"))
            ],
        }
        gains_src = _lookup_optional(realization, "random_gains") or realization

    for key in ("gcorsna", "gcorsnn", "gcordrstr", "ggege", "gsngen", "gsngea", "gsngi"):
        out[key] = _as_2d(_lookup(gains_src, key), N, n, f"realization.{key}")

    return out


def _sample_initials(params: Mapping, names: Iterable[str], rng: np.random.Generator, N: int = 1):
    n_by_type = params["cells"]["n_by_type"]

    external = _normalize_external_initial(params, names, n_by_type, N)
    if external is not None:
        return external

    groups = []
    for name in names:
        n = int(n_by_type[name])
        mean, sd = params["cells"]["v_init"][name]
        if sd == 0:
            vals = np.full((N, n), mean, dtype=float)
        else:
            vals = mean + sd * rng.standard_normal((N, n))
        groups.append(vals)
    return np.concatenate(groups, axis=1)


def _label_population(pop, names: Iterable[str], n_by_type: Mapping[str, int]):
    idx = 0
    for name in names:
        n = int(n_by_type[name])
        pop[:, idx : idx + n].label(name)
        idx += n


def _realization(n: int, params: Mapping, rng: np.random.Generator, N: int = 1):
    external = params.get("realization", None)
    if external is not None:
        return _normalize_external_realization(external, n, N=N)

    pd = float(params.get("pd", 0.0))

    def per_network_perms(k: int):
        return [np.stack([rng.permutation(n) for _ in range(N)], axis=0) for _ in range(k)]

    realization = {
        "str_d2_perms": per_network_perms(4),
        "str_d1_perms": per_network_perms(3),
        "fs_to_rs_perms": per_network_perms(4),
        "rs_to_fs_perms": per_network_perms(4),
        "gcorsna": 0.3 * rng.random((N, n)),
        "gcorsnn": 0.003 * rng.random((N, n)),
        "gcordrstr": (0.07 - 0.044 * pd) + 0.001 * rng.random((N, n)),
        "ggege": rng.random((N, n)),
        "gsngen": np.zeros((N, n)),
        "gsngea": np.zeros((N, n)),
        "gsngi": np.zeros((N, n)),
    }
    n_ampa = min(2, n)
    n_gpi = min(5, n)
    for row in range(N):
        realization["gsngen"][row, rng.choice(n, size=n_ampa, replace=False)] = 0.002 * rng.random(n_ampa)
        realization["gsngea"][row, rng.choice(n, size=n_ampa, replace=False)] = 0.3 * rng.random(n_ampa)
        realization["gsngi"][row, rng.choice(n, size=n_gpi, replace=False)] = 0.15
    return realization


def _delay_steps(params: Mapping, dt_ref: float):
    sp = params["syn"]["pathways"]

    def steps(pathway: str):
        return int(round(float(sp[pathway]["delay"]) / float(dt_ref)))

    return {
        "th_ctx": steps("TH_CTX"),
        "stn_gpe": steps("STN_GPe_AMPA"),
        "stn_gpi": steps("STN_GPi"),
        "gpe_stn": steps("GPe_STN"),
        "gpe_gpi": steps("GPe_GPi"),
        "gpe_gpe": steps("GPe_GPe"),
        "gpi_th": steps("GPi_TH"),
        "d2_gpe": steps("D2_GPe"),
        "d1_gpi": steps("D1_GPi"),
        "ctx_d2": steps("CTX_D2"),
        "ctx_d1": steps("CTX_D1"),
        "ctx_stn": steps("CTX_STN_AMPA"),
    }


def _dbs_frequency(params: Mapping, pick_dbs_freq=None):
    if "dbs" in params and "freq_hz" in params["dbs"]:
        return float(params["dbs"]["freq_hz"])
    if pick_dbs_freq is None:
        pick_dbs_freq = params.get("pick_dbs_freq", 1)
    freqs = np.arange(0, 205, 5, dtype=float)
    idx = int(pick_dbs_freq) - 1  # MATLAB uses 1-based indexing.
    idx = max(0, min(idx, len(freqs) - 1))
    return float(freqs[idx])


def _fused_config(
    params: Mapping,
    realization,
    dt_ref: float,
    pick_dbs_freq=None,
    *,
    N: int = 1,
    synapse_discretization=None,
    synapse_update_mode=None,
    spike_update_mode=None,
    delay_mode=None,
):
    pd = float(params.get("pd", 0.0))
    corstim = float(params.get("corstim", 0.0))

    coupling = deepcopy(params["coupling"])
    coupling.setdefault("gm", 1.0)

    constants = dict(
        gl=[0.05, 0.35, 0.1, 0.1],
        El=[-70.0, -60.0, -65.0, -67.0],
        gna=[3.0, 49.0, 120.0, 100.0],
        Ena=[50.0, 60.0, 55.0, 50.0],
        gk=[5.0, 57.0, 30.0, 80.0],
        Ek=[-75.0, -90.0, -80.0, -100.0],
        gt=[5.0, 5.0, 0.5],
        Et=0.0,
        gca=[0.0, 2.0, 0.15],
        Eca=[0.0, 140.0, 120.0],
        Em=-100.0,
        gahp=[0.0, 20.0, 10.0],
        k1=[0.0, 15.0, 10.0],
        kca=[0.0, 22.5, 15.0],
        ga=5.0,
        gL=15.0,
        gcak=1.0,
        Kca=2.0e-3,
        alp=1.0 / (2.0 * 96485.0),
        con=(8314.0 * 298.0) / (2.0 * 96485.0),
        Cao=2000.0,
        Esyn=[-85.0, 0.0, -85.0, 0.0, -85.0, -85.0, -80.0],
    )

    ctx_rs = deepcopy(params["intrinsic"]["CTX_RS"])
    ctx_fs = deepcopy(params["intrinsic"]["CTX_FS"])
    ctx_rs.setdefault("v_peak", 30.0)
    ctx_fs.setdefault("v_peak", 30.0)

    dbs = dict(params.get("dbs", {}))
    dbs.setdefault("freq_hz", _dbs_frequency(params, pick_dbs_freq=pick_dbs_freq))
    dbs.setdefault("pulse_width_ms", 0.3)
    dbs.setdefault("amplitude", 300.0)

    ctx_stim = dict(params.get("ctx_stim", {}))
    ctx_stim.setdefault("enabled", bool(corstim))
    ctx_stim.setdefault("start_ms", 1000.0)
    ctx_stim.setdefault("duration_ms", 0.3)
    ctx_stim.setdefault("amplitude", 350.0)

    if synapse_discretization is None:
        synapse_discretization = params.get("synapse_discretization", "euler")
    if synapse_update_mode is None:
        synapse_update_mode = params.get("synapse_update_mode", "uncoalesced")
    if spike_update_mode is None:
        spike_update_mode = params.get("spike_update_mode", "uncoalesced")
    if delay_mode is None:
        delay_mode = params.get("delay_mode", "auto")

    return {
        "n": int(params["n"]),
        "N": int(N),
        "synapse_discretization": str(synapse_discretization).lower(),
        "synapse_update_mode": str(synapse_update_mode).lower(),
        "spike_update_mode": str(spike_update_mode).lower(),
        "delay_mode": str(delay_mode).lower(),
        "pd": pd,
        "corstim": corstim,
        "dt_ref": float(dt_ref),
        "realization": realization,
        "delay_steps": _delay_steps(params, dt_ref),
        "constants": constants,
        "coupling": coupling,
        "syn": {
            "gpeak": float(params["syn"]["gpeak"]),
            "gpeak1": float(params["syn"]["gpeak1"]),
            "tau_alpha": float(params["syn"].get("tau_alpha", 5.0)),
            "tau_i_striatum": float(params["syn"].get("tau_i_striatum", 13.0)),
        },
        "ctx_rs": ctx_rs,
        "ctx_fs": ctx_fs,
        "dbs": dbs,
        "ctx_stim": ctx_stim,
    }


def kumaravelu_2016_fused(
    params=None,
    rng=None,
    *,
    N: int = 1,
    dt: float = 0.01,
    pick_dbs_freq=None,
    differentiable_spikes: bool = True,
    synapse_discretization=None,
    synapse_update_mode=None,
    spike_update_mode=None,
    delay_mode=None,
):
    """Build the fused Kumaravelu CTX-BG-TH model as a single Dendra Population.

    Parameters
    ----------
    params : dict, optional
        Parameter dictionary. Defaults to ``kumaravelu_2016.params.default_params()``.
    rng : numpy.random.Generator, optional
        Random generator for initial conditions and realization arrays.
    N : int, optional
        Number of independent network realizations to simulate in parallel.  The
        returned population has shape ``(N, 8*n)``.
    dt : float, optional
        Reference timestep in ms used to discretize axonal delays. Run with the
        same ``dt`` for faithful delay timing.
    pick_dbs_freq : int, optional
        MATLAB-style 1-based index into ``0:5:200`` Hz. Ignored if
        ``params['dbs']['freq_hz']`` is present.
    differentiable_spikes : bool, optional
        If true, spike event tensors use Dendra's straight-through surrogate
        gradients. If false, the generated fused mechanism uses a separate hard
        threshold-event class and skips surrogate sigmoid/ReLU work entirely.
    synapse_discretization : {"euler", "backward_euler", "exact"}, optional
        Recursive filter update used for alpha and double-exponential synapses.
        Defaults to ``params.get('synapse_discretization', 'euler')`` to preserve
        the original fused backend behavior.
    synapse_update_mode : {"uncoalesced", "coalesced"}, optional
        Tensor layout used for synaptic filter updates. ``"coalesced"`` stacks
        same-family alpha and double-exponential filters into stream axes.
    spike_update_mode : {"uncoalesced", "coalesced"}, optional
        Tensor layout used for spike-event detection. ``"coalesced"`` stacks
        populations and thresholds before evaluating reset and crossing events.
    delay_mode : {"auto", "shift", "circular", "circular_eager"}, optional
        Fixed-delay queue backend. ``"auto"`` uses fast circular buffers in
        eval/no-grad mode and graph-safe shifted queues in training mode.
        ``"shift"`` forces the original functional shifted queue.
        ``"circular"`` forces the fast in-place circular buffer.
        ``"circular_eager"`` updates all fused pathway delays inside one
        ``torch._dynamo`` eager island while keeping the rest of the fused step
        compiled; this is intended for forward/eval simulations when JIT is
        otherwise beneficial but compiled circular-buffer mutation is costly.

    Returns
    -------
    dendra.models.Population
        A single ``scnv`` population with ``8*n`` exposed voltage entries. Labels
        are ``TH``, ``STN``, ``GPe``, ``GPi``, ``StrD2``, ``StrD1``, ``CTX_RS``,
        and ``CTX_FS``.
    """

    params = _copy_params(params)
    rng = _as_rng(params, rng)
    N = int(N if N is not None else params.get("N", 1))
    if N < 1:
        raise ValueError(f"N must be >= 1; got {N}.")
    params["N"] = N
    if "n" in params:
        n = int(params["n"])
        params["cells"]["n_by_type"] = {k: n for k in params["cells"]["n_by_type"]}
    n = _validate_equal_n(params)
    params["n"] = n

    names = params["cells"]["hh_celltypes"] + params["cells"]["ctx_celltypes"]
    v_init = _sample_initials(params, names, rng, N=N)
    realization = _realization(n, params, rng, N=N)
    config = _fused_config(
        params,
        realization,
        dt_ref=dt,
        pick_dbs_freq=pick_dbs_freq,
        N=N,
        synapse_discretization=synapse_discretization,
        synapse_update_mode=synapse_update_mode,
        spike_update_mode=spike_update_mode,
        delay_mode=delay_mode,
    )
    config["differentiable_spikes"] = bool(differentiable_spikes)

    mech_cls = make_kumaravelu_2016_fused(config)
    pop = dn.Population(N=N, C=8 * n, v_init=v_init, integrator=dn.scnv())
    _label_population(pop, names, params["cells"]["n_by_type"])

    spike_params = dict(params.get("spikedetect_hh", {}))
    tau_gate = float(spike_params.get("tau_gate", 0.5))
    ste_scale = float(spike_params.get("ste_scale", 1.0)) if differentiable_spikes else 0.0

    # Do not call ``mech_cls.rename("kumaravelu_fused")`` here.  Dendra's
    # Mechanism.rename cache is keyed only by alias name, and the generated fused
    # class carries the network realization/n in its class-level CONFIG.  Reusing
    # a cached renamed class from an earlier n=10 build would make the mechanism
    # expect 8*10 voltages even when the Population was correctly created with
    # 8*n entries.  ``make_kumaravelu_2016_fused`` already returns a fresh class
    # whose __name__ is the desired Dendra alias.
    pop.insert(mech_cls, tau_gate=tau_gate, ste_scale=ste_scale)

    # Attach useful metadata for downstream validation/recording. These are plain
    # Python attributes and not part of the Dendra parameter system.
    pop.kumaravelu_params = params
    pop.kumaravelu_realization = realization
    pop.kumaravelu_fused_config = config
    pop.kumaravelu_dt_ref = float(dt)
    pop.kumaravelu_N = int(N)
    pop.kumaravelu_spike_mode = "surrogate" if differentiable_spikes else "hard"
    pop.kumaravelu_synapse_discretization = config["synapse_discretization"]
    pop.kumaravelu_synapse_update_mode = config.get("synapse_update_mode", "uncoalesced")
    pop.kumaravelu_spike_update_mode = config.get("spike_update_mode", "uncoalesced")
    pop.kumaravelu_delay_mode = config["delay_mode"]
    return pop.double() if not params.get("fp32", False) else pop.float()


# Convenience aliases matching the existing builder naming style.
Kumaravelu2016Fused = kumaravelu_2016_fused
