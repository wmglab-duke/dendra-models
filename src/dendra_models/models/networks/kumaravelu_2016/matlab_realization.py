"""Load MATLAB Kumaravelu validation exports into Dendra params.

This helper consumes the validation struct written by ``simulate_network_model.m``.
It intentionally extracts the realized random arrays rather than trying to match
MATLAB's RNG stream.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.io import loadmat

try:
    from .params import params as default_parameter_dict
except Exception:  # allows running as a standalone helper
    default_parameter_dict = None


POP_TO_INITIAL_ALIASES = {
    "TH": ("TH", "v1", "vth"),
    "STN": ("STN", "v2", "vsn"),
    "GPe": ("GPe", "v3", "vge"),
    "GPi": ("GPi", "v4", "vgi"),
    "StrD2": ("StrD2", "Striat_indr", "Striatum_indirect", "v5", "vstr_indr"),
    "StrD1": ("StrD1", "Striat_dr", "Striatum_direct", "v6", "vstr_dr"),
    "CTX_RS": ("CTX_RS", "Cortex_RS", "Cor_RS", "ve"),
    "CTX_FS": ("CTX_FS", "Cortex_FS", "Cor_FS", "vi"),
}


def _as_dict(obj: Any) -> Any:
    """Recursively convert scipy/MATLAB structs to plain dictionaries."""
    if isinstance(obj, dict):
        return {k: _as_dict(v) for k, v in obj.items() if not str(k).startswith("__")}
    if hasattr(obj, "_fieldnames"):
        return {name: _as_dict(getattr(obj, name)) for name in obj._fieldnames}
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        if obj.shape == ():
            return {name: _as_dict(obj[name].item()) for name in obj.dtype.names}
        if obj.size == 1:
            item = obj.reshape(-1)[0]
            return {name: _as_dict(item[name]) for name in obj.dtype.names}
    return obj


def _field(mapping: Mapping[str, Any], *names: str):
    for name in names:
        if name in mapping:
            return mapping[name]
    raise KeyError(f"None of {names!r} found in fields {tuple(mapping.keys())!r}")


def _optional_field(mapping: Mapping[str, Any], *names: str):
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _scalar(x, default=None):
    if x is None:
        return default
    arr = np.asarray(x)
    if arr.size == 0:
        return default
    return arr.reshape(-1)[0].item() if hasattr(arr.reshape(-1)[0], "item") else arr.reshape(-1)[0]


def _vec(x, *, dtype=float):
    return np.asarray(x, dtype=dtype).reshape(-1)


def _one_based_or_zero_based_perm(x):
    arr = _vec(x, dtype=int)
    if arr.size and arr.min() >= 1 and arr.max() == arr.size:
        arr = arr - 1
    return arr


def _extract_initial(validation: Mapping[str, Any]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}

    if "initial" in validation:
        initial = validation["initial"]
        voltage_initial = initial.get("v", initial) if isinstance(initial, Mapping) else initial
        for pop, aliases in POP_TO_INITIAL_ALIASES.items():
            try:
                out[pop] = _vec(_field(voltage_initial, *aliases), dtype=float)
            except KeyError:
                pass

    # Robust fallback: use first time point from exported voltage traces.
    if len(out) < len(POP_TO_INITIAL_ALIASES):
        voltage = validation.get("voltage", {})
        for pop in POP_TO_INITIAL_ALIASES:
            if pop not in out and pop in voltage:
                v = np.asarray(voltage[pop], dtype=float)
                if v.ndim == 1:
                    out[pop] = v.reshape(-1)
                else:
                    out[pop] = v[0, :].reshape(-1)

    missing = [p for p in POP_TO_INITIAL_ALIASES if p not in out]
    if missing:
        raise KeyError(f"Could not recover initial voltages for {missing!r}")
    return out


def _perm_group(mapping: Mapping[str, Any], keys: tuple[str, ...]) -> list[np.ndarray]:
    return [_one_based_or_zero_based_perm(_field(mapping, key)) for key in keys]


def _extract_realization(validation: Mapping[str, Any]) -> dict[str, Any]:
    if "realization" not in validation:
        raise KeyError(
            "MATLAB validation file does not contain validation.realization. "
            "Re-run MATLAB with realization export enabled."
        )
    r = validation["realization"]

    out: dict[str, Any] = {}

    # Current MATLAB validation schema: nested permutation groups plus random_gains.
    if any(k in r for k in ("str_d2_gaba_perms", "ctx_fs_to_rs_perms", "ctx_rs_to_fs_perms", "str_d1_gaba_perms")):
        out["str_d2_perms"] = _perm_group(_field(r, "str_d2_gaba_perms"), ("all", "bll", "cll", "dll"))
        out["fs_to_rs_perms"] = _perm_group(_field(r, "ctx_fs_to_rs_perms"), ("ell", "fll", "gll", "hll"))
        out["rs_to_fs_perms"] = _perm_group(_field(r, "ctx_rs_to_fs_perms"), ("ill", "jll", "kll", "lll"))
        out["str_d1_perms"] = _perm_group(_field(r, "str_d1_gaba_perms"), ("mll", "nll", "oll"))
        gains = _optional_field(r, "random_gains") or r
    else:
        # Older flat MATLAB schema.
        flat_keys = [
            "all", "bll", "cll", "dll",
            "ell", "fll", "gll", "hll",
            "ill", "jll", "kll", "lll",
            "mll", "nll", "oll",
        ]
        if any(k in r for k in flat_keys):
            out["str_d2_perms"] = _perm_group(r, ("all", "bll", "cll", "dll"))
            out["fs_to_rs_perms"] = _perm_group(r, ("ell", "fll", "gll", "hll"))
            out["rs_to_fs_perms"] = _perm_group(r, ("ill", "jll", "kll", "lll"))
            out["str_d1_perms"] = _perm_group(r, ("mll", "nll", "oll"))
        else:
            # Already-Dendra-named realization structs.
            for key in ("str_d2_perms", "str_d1_perms", "fs_to_rs_perms", "rs_to_fs_perms"):
                vals = _field(r, key)
                if isinstance(vals, np.ndarray) and vals.dtype == object:
                    vals = vals.reshape(-1).tolist()
                out[key] = [_one_based_or_zero_based_perm(v) for v in vals]
        gains = _optional_field(r, "random_gains") or r

    for key in ("gcorsna", "gcorsnn", "gcordrstr", "ggege", "gsngen", "gsngea", "gsngi"):
        out[key] = _vec(_field(gains, key), dtype=float)

    return out


def load_matlab_validation(path: str | Path) -> dict[str, Any]:
    raw = loadmat(path, simplify_cells=True)
    raw = _as_dict(raw)
    if "validation" in raw:
        return raw["validation"]
    if "matlab_validation" in raw:
        return raw["matlab_validation"]
    raise KeyError("Could not find top-level 'validation' struct in MATLAB file.")


def params_from_matlab_validation(path: str | Path, base_params: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Return a Dendra params dict using MATLAB's exact initial/realization arrays."""
    validation = load_matlab_validation(path)

    if base_params is None:
        if default_parameter_dict is None:
            raise ValueError("base_params is required when this helper is used standalone.")
        params = deepcopy(default_parameter_dict)
    else:
        params = deepcopy(base_params)

    meta = validation.get("metadata", {}) if isinstance(validation.get("metadata", {}), Mapping) else {}

    n_val = _optional_field(validation, "n")
    if n_val is None:
        n_val = _optional_field(meta, "n")
    if n_val is not None:
        params["n"] = int(_scalar(n_val))

    params["initial"] = _extract_initial(validation)
    params["realization"] = _extract_realization(validation)

    # Preserve run settings so Dendra's fused config matches MATLAB.
    for key in ("pd", "corstim", "pick_dbs_freq"):
        val = _optional_field(validation, key)
        if val is None:
            val = _optional_field(meta, key)
        if val is not None:
            params[key] = int(_scalar(val)) if key in {"pick_dbs_freq", "corstim"} else float(_scalar(val))

    pattern_hz = _optional_field(meta, "pattern_hz")
    if pattern_hz is not None:
        params.setdefault("dbs", {})["freq_hz"] = float(_scalar(pattern_hz))

    return params
