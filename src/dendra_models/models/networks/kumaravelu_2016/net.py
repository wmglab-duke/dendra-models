"""
Dendra network builder for the Kumaravelu et al. CTX-BG-TH model.

The builder follows the module pattern in the supplied yu_2024 implementation:
subranges are labeled, mechanisms are inserted on labeled groups, the Network is
then constructed, and cross-mechanism references are bound with setreference.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import torch

import dendra as dn
from dendra.models.mod import spikedetect

from .params import params as default_parameter_dict
from .mechanisms.hh import thalamic, stn, gpe, gpi, striatal_d2, striatal_d1
from .mechanisms.izhikevich import regular_spiking_cortex, fast_spiking_interneuron, cortical_spike_router
from .mechanisms.synapses import (
    alpha_syn_current,
    alpha_syn_event,
    exp2syn_current,
    striatal_gaba_gate,
    striatal_recurrent_gaba,
)


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
            "This faithful MATLAB translation currently assumes the same n in "
            f"every population. Expected n={expected}, got {bad}."
        )
    return expected


def _sample_initials(params: Mapping, names: Iterable[str], rng: np.random.Generator) -> List[float]:
    out: List[float] = []
    n_by_type = params["cells"]["n_by_type"]
    for name in names:
        n = int(n_by_type[name])
        mean, sd = params["cells"]["v_init"][name]
        if sd == 0:
            vals = np.full(n, mean, dtype=float)
        else:
            vals = mean + sd * rng.standard_normal(n)
        out.extend(vals.tolist())
    return out


def _label_population(pop, names: Iterable[str], n_by_type: Mapping[str, int]):
    idx = 0
    for name in names:
        n = int(n_by_type[name])
        pop[0, idx:idx + n].label(name)
        idx += n


def _realization(n: int, params: Mapping, rng: np.random.Generator) -> Dict[str, object]:
    pd = float(params.get("pd", 0.0))
    realization = {
        "str_d2_perms": [rng.permutation(n) for _ in range(4)],
        "str_d1_perms": [rng.permutation(n) for _ in range(3)],
        "fs_to_rs_perms": [rng.permutation(n) for _ in range(4)],
        "rs_to_fs_perms": [rng.permutation(n) for _ in range(4)],
        "gcorsna": 0.3 * rng.random(n),
        "gcorsnn": 0.003 * rng.random(n),
        "gcordrstr": (0.07 - 0.044 * pd) + 0.001 * rng.random(n),
        "ggege": rng.random(n),
        "gsngen": np.zeros(n),
        "gsngea": np.zeros(n),
        "gsngi": np.zeros(n),
    }
    n_ampa = min(2, n)
    n_gpi = min(5, n)
    realization["gsngen"][rng.choice(n, size=n_ampa, replace=False)] = 0.002 * rng.random(n_ampa)
    realization["gsngea"][rng.choice(n, size=n_ampa, replace=False)] = 0.3 * rng.random(n_ampa)
    realization["gsngi"][rng.choice(n, size=n_gpi, replace=False)] = 0.15
    return realization


def _pathway_peak(params: Mapping, pathway: str) -> float:
    key = params["syn"]["pathways"][pathway]["peak"]
    return float(params["syn"][key])


def _insert_synapse(
    target_group,
    name: str,
    spec: Mapping,
    *,
    current_scale: float = 1.0,
    event_only: bool = False,
):
    if spec["kind"] == "alpha":
        cls = alpha_syn_event if event_only else alpha_syn_current
        target_group.insert(
            cls.rename(name),
            tau=spec["tau"],
            e=spec["e"],
            current_scale=current_scale,
        )
    elif spec["kind"] == "exp2":
        target_group.insert(
            exp2syn_current.rename(name),
            tau1=spec["tau1"],
            tau2=spec["tau2"],
            e=spec["e"],
            current_scale=current_scale,
        )
    else:
        raise ValueError(f"Unknown synapse kind for {name!r}: {spec['kind']!r}")


def _edge_arrays_from_shifts(n: int, shifts: Iterable[int]) -> Tuple[np.ndarray, np.ndarray]:
    post = np.arange(n, dtype=int)
    pres = []
    posts = []
    for shift in shifts:
        pres.append((post + int(shift)) % n)
        posts.append(post.copy())
    return np.concatenate(pres), np.concatenate(posts)


def _edge_arrays_from_perms(perms: Iterable[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    pres = []
    posts = []
    for perm in perms:
        perm = np.asarray(perm, dtype=int)
        pres.append(perm)
        posts.append(np.arange(len(perm), dtype=int))
    return np.concatenate(pres), np.concatenate(posts)


def _edge_values(values, post_idx: np.ndarray, n_edges: int, *, device=None, dtype=None):
    """Normalize connection weights without prematurely expanding scalars.

    Dendra's Network.connect expands scalar weights *after* applying its
    autapse/multapse policy. Returning a Python list for scalar weights can
    therefore become stale if the connection set is filtered. For heterogeneous
    weights, return a tensor whose length matches the explicit edge list.
    """
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 0:
        return float(arr)
    if len(arr) == n_edges:
        vals = arr.astype(float)
    else:
        vals = arr[post_idx].astype(float)
    return torch.as_tensor(vals, device=device, dtype=dtype)


def _delays(delay: float, n_edges: int):
    # Keep scalar delays scalar for the same reason as scalar weights above.
    return float(delay)


def _connect_one_to_one(
    net,
    pre_pop: str,
    pre_label: str,
    post_pop: str,
    post_label: str,
    syn_name: str,
    pre_idx: np.ndarray,
    post_idx: np.ndarray,
    weight,
    delay: float,
    *,
    pre_var: str | None = None,
    threshold: float | None = -10.0,
    allow_autapses: bool = False,
    allow_multapses: bool = False,
):
    n_edges = int(len(pre_idx))
    if n_edges == 0:
        return

    pre_model = getattr(net, pre_pop)
    post_model = getattr(net, post_pop)
    pre = getattr(pre_model, pre_label)[pre_idx]
    post = getattr(post_model, post_label)[post_idx]
    syn = getattr(post_model.mech, syn_name)
    kwargs = dict(
        weight=_edge_values(
            weight, post_idx, n_edges, device=post_model.device(), dtype=post_model.dtype()
        ),
        delay=_delays(delay, n_edges),
    )
    if pre_var is not None:
        kwargs.update(threshold=None, pre_var=pre_var)
    else:
        kwargs.update(threshold=threshold)
    net.connect_one_to_one(
        pre,
        post,
        syn,
        allow_autapses=allow_autapses,
        allow_multapses=allow_multapses,
        **kwargs,
    )


def _connect_shifted(
    net,
    n: int,
    pre_pop: str,
    pre_label: str,
    post_pop: str,
    post_label: str,
    syn_name: str,
    shifts: Iterable[int],
    weight,
    delay: float,
    *,
    pre_var: str | None = None,
    threshold: float | None = -10.0,
    allow_autapses: bool = False,
    allow_multapses: bool = False,
):
    pre_idx, post_idx = _edge_arrays_from_shifts(n, shifts)
    _connect_one_to_one(
        net, pre_pop, pre_label, post_pop, post_label, syn_name,
        pre_idx, post_idx, weight, delay, pre_var=pre_var, threshold=threshold,
        allow_autapses=allow_autapses, allow_multapses=allow_multapses,
    )


def _index_reference(getter: Callable[[], torch.Tensor], idx: np.ndarray):
    idx_np = np.asarray(idx, dtype=np.int64)

    def ref():
        x = getter()
        tidx = torch.as_tensor(idx_np, dtype=torch.long, device=x.device)
        return x[:, tidx]

    return ref


def _ctx_local_view(getter: Callable[[], torch.Tensor], start: int, stop: int):
    """Return a local RS/FS view from either local or full CTX tensors."""

    def ref():
        x = getter()
        width = int(stop - start)
        if x.shape[-1] == width:
            return x
        return x[..., start:stop]

    return ref


def _ctx_syn_current_reference(
    get_s: Callable[[], torch.Tensor],
    get_v: Callable[[], torch.Tensor],
    e: float,
    start: int,
    stop: int,
    current_scale: float = 1.0,
):
    """Compute CTX synaptic current directly from conductance and local voltage.

    For the scnv cortical population, ``mech.advance`` evaluates the
    Izhikevich state equations before the bookkeeping ``mech.i`` pass. Reading
    saved ``i_`` from a separate synapse mechanism can therefore lag by one
    step and is sensitive to full-vs-local current frames.  This helper computes
    the current from the synapse conductance state in the target-local RS/FS
    voltage frame instead.
    """

    def ref():
        v = get_v()
        s = get_s()
        width = v.shape[-1]
        if s.shape[-1] != width:
            s = s[..., start:stop]
        if tuple(s.shape) != tuple(v.shape) and s.numel() == v.numel():
            s = s.reshape_as(v)
        return current_scale * s * (v - e)

    return ref


def _bind_references(net, realization: Mapping, params: Mapping):
    """Bind state/current references after dn.Network construction."""
    mh = net.hh.mech
    mc = net.ctx.mech

    # Izhikevich scnv population: model.v is not the integrated voltage, so the
    # VoltageProcess state equation must explicitly read synaptic currents.
    # The cortical synapse mechanisms are full-population mechanisms; slice their
    # saved currents back to the mechanism-local RS/FS state-vector shapes.
    n_ctx = int(len(realization["gcorsna"]))
    mc.ctx_rs.DE["regular_spiking_cortex_states"].setreference(
        "i_ie",
        _ctx_syn_current_reference(
            lambda: mc.FS_RS.s, lambda: mc.ctx_rs.v_izh,
            e=float(params["syn"]["pathways"]["FS_RS"]["e"]),
            start=0, stop=n_ctx,
        ),
    )
    mc.ctx_rs.DE["regular_spiking_cortex_states"].setreference(
        "i_thcor",
        _ctx_syn_current_reference(
            lambda: mc.TH_CTX.s, lambda: mc.ctx_rs.v_izh,
            e=float(params["syn"]["pathways"]["TH_CTX"]["e"]),
            start=0, stop=n_ctx,
        ),
    )
    mc.ctx_fs.DE["fast_spiking_interneuron_states"].setreference(
        "i_ei",
        _ctx_syn_current_reference(
            lambda: mc.RS_FS.s, lambda: mc.ctx_fs.v_izh,
            e=float(params["syn"]["pathways"]["RS_FS"]["e"]),
            start=n_ctx, stop=2 * n_ctx,
        ),
    )
    mc.ctx_spikes.setreference("rs_spikes_local", lambda: mc.ctx_rs.spikes)
    mc.ctx_spikes.setreference("rs_syn_spikes_local", lambda: mc.ctx_rs.syn_spikes)
    mc.ctx_spikes.setreference("fs_spikes_local", lambda: mc.ctx_fs.spikes)
    mc.ctx_spikes.setreference("fs_syn_spikes_local", lambda: mc.ctx_fs.syn_spikes)

    # Recurrent striatal GABA: S1c/S8 are voltage-gated states owned by the
    # presynaptic MSN populations, not spike-triggered point-process synapses.
    d2_refs = mh.StrD2_recurrent_gaba.DE["striatal_recurrent_gaba_refs"]
    for k, perm in enumerate(realization["str_d2_perms"]):
        d2_refs.setreference(f"s{k}", _index_reference(lambda: mh.StrD2_gate.s, perm))

    d1_refs = mh.StrD1_recurrent_gaba.DE["striatal_recurrent_gaba_refs"]
    for k, perm in enumerate(realization["str_d1_perms"]):
        d1_refs.setreference(f"s{k}", _index_reference(lambda: mh.StrD1_gate.s, perm))


def kumaravelu_2016(params=None, rng=None):
    """Build the Kumaravelu CTX-BG-TH network as a Dendra Network."""
    params = _copy_params(params)
    rng = _as_rng(params, rng)
    # Treat params["n"] as the canonical MATLAB n and expand it into every
    # named population unless the caller has deliberately removed that key.
    if "n" in params:
        n = int(params["n"])
        params["cells"]["n_by_type"] = {k: n for k in params["cells"]["n_by_type"]}
    n = _validate_equal_n(params)

    # Disease/stimulation parameters that alter intrinsic currents.
    pd = float(params.get("pd", 0.0))
    corstim = float(params.get("corstim", 0.0))
    params["intrinsic"]["StrD2"]["pd"] = pd
    params["intrinsic"]["StrD1"]["pd"] = pd
    params["intrinsic"]["GPe"]["i_stim"] = 3.0 - 2.0 * corstim * (1.0 - pd)

    # MATLAB HH currents are uA/cm^2. Dendra's single-compartment voltage
    # solver uses mA/cm^2 current densities, so all HH-intrinsic and HH-target
    # synaptic currents get this multiplicative scale. The scnv/Izhikevich
    # cortical population keeps the MATLAB numeric current convention.
    hh_current_scale = float(params.get("units", {}).get("hh_current_scale", 1.0e-3))
    for _name in params["cells"]["hh_celltypes"]:
        params["intrinsic"][_name].setdefault("current_scale", hh_current_scale)

    hh_names = params["cells"]["hh_celltypes"]
    ctx_names = params["cells"]["ctx_celltypes"]
    n_by_type = params["cells"]["n_by_type"]
    realization = _realization(n, params, rng)
    params["realization"] = realization

    # HH population uses Dendra's standard v solver.
    hh_v_init = _sample_initials(params, hh_names, rng)
    hh_pop = dn.Population(C=len(hh_v_init), v_init=hh_v_init)
    _label_population(hh_pop, hh_names, n_by_type)

    # Cortical Izhikevich population uses scnv because v is a mechanism state.
    ctx_pop = dn.Population(C=2 * n, v_init=-65.0, integrator=dn.scnv())
    _label_population(ctx_pop, ctx_names, n_by_type)

    # Intrinsic mechanisms.
    hh_pop.TH.insert(thalamic.rename("thalamic"), **params["intrinsic"]["TH"])
    hh_pop.STN.insert(stn.rename("stn"), **params["intrinsic"]["STN"])
    hh_pop.GPe.insert(gpe.rename("gpe"), **params["intrinsic"]["GPe"])
    hh_pop.GPi.insert(gpi.rename("gpi"), **params["intrinsic"]["GPi"])
    hh_pop.StrD2.insert(striatal_d2.rename("str_d2"), **params["intrinsic"]["StrD2"])
    hh_pop.StrD1.insert(striatal_d1.rename("str_d1"), **params["intrinsic"]["StrD1"])

    # HH-origin synapses use a single presynaptic threshold-crossing detector
    # per HH compartment. All HH NetCons below read mech.spikedetect.spikes,
    # avoiding duplicated voltage threshold tests on every outgoing synapse.
    threshold = float(params.get("spike_threshold_hh", -10.0))
    detector_params = dict(params.get("spikedetect_hh", {}))
    detector_params.setdefault("threshold", threshold)
    hh_pop.insert(spikedetect, **detector_params)

    tau_i = params["syn"]["tau_i_striatum"]
    ggaba = params["coupling"]["ggaba"]
    hh_pop.StrD2.insert(striatal_gaba_gate.rename("StrD2_gate"), tau_i=tau_i)
    hh_pop.StrD1.insert(striatal_gaba_gate.rename("StrD1_gate"), tau_i=tau_i)
    hh_pop.StrD2.insert(striatal_recurrent_gaba.rename("StrD2_recurrent_gaba"), g=ggaba / 4.0, e=-80.0, current_scale=hh_current_scale)
    hh_pop.StrD1.insert(striatal_recurrent_gaba.rename("StrD1_recurrent_gaba"), g=ggaba / 3.0, e=-80.0, current_scale=hh_current_scale)

    ctx_pop.CTX_RS.insert(regular_spiking_cortex.rename("ctx_rs"), **params["intrinsic"]["CTX_RS"])
    ctx_pop.CTX_FS.insert(fast_spiking_interneuron.rename("ctx_fs"), **params["intrinsic"]["CTX_FS"])
    ctx_pop.insert(cortical_spike_router.rename("ctx_spikes"))

    # Synapses inserted on postsynaptic targets.
    sp = params["syn"]["pathways"]

    # CTX-target synapses are event-only conductance states.  The cortical
    # Izhikevich mechanisms own the voltage state under scnv and read these
    # conductances through setreference, so the synapses should not contribute
    # to the standard membrane-current bookkeeping pass.  Insert them on their
    # actual target slices so NetCon post indices, synapse state, and the
    # VoltageProcess reference frame are all length n.
    _insert_synapse(ctx_pop.CTX_RS, "TH_CTX", sp["TH_CTX"], event_only=True)
    _insert_synapse(ctx_pop.CTX_RS, "FS_RS", sp["FS_RS"], event_only=True)
    _insert_synapse(ctx_pop.CTX_FS, "RS_FS", sp["RS_FS"], event_only=True)

    for name in ["GPe_STN", "CTX_STN_AMPA", "CTX_STN_NMDA"]:
        _insert_synapse(hh_pop.STN, name, sp[name], current_scale=hh_current_scale)
    for name in ["STN_GPe_AMPA", "STN_GPe_NMDA", "GPe_GPe", "D2_GPe"]:
        _insert_synapse(hh_pop.GPe, name, sp[name], current_scale=hh_current_scale)
    for name in ["STN_GPi", "GPe_GPi", "D1_GPi"]:
        _insert_synapse(hh_pop.GPi, name, sp[name], current_scale=hh_current_scale)
    _insert_synapse(hh_pop.TH, "GPi_TH", sp["GPi_TH"], current_scale=hh_current_scale)
    _insert_synapse(hh_pop.StrD2, "CTX_D2", sp["CTX_D2"], current_scale=hh_current_scale)
    _insert_synapse(hh_pop.StrD1, "CTX_D1", sp["CTX_D1"], current_scale=hh_current_scale)

    if params.get("fp32", False):
        net = dn.Network({"hh": hh_pop, "ctx": ctx_pop}).float()
    else:
        net = dn.Network({"hh": hh_pop, "ctx": ctx_pop}).double()

    _bind_references(net, realization, params)

    c = params["coupling"]
    hh_spikes = "mech.spikedetect.spikes"

    # HH-to-HH and HH-to-cortex pathways. Thresholding has already been done by
    # the HH-population spikedetect mechanism, so each NetCon reads the cached
    # presynaptic spike flag instead of performing its own voltage comparison.
    _connect_shifted(net, n, "hh", "GPi", "hh", "TH", "GPi_TH", [0], _pathway_peak(params, "GPi_TH") * c["ggith"], sp["GPi_TH"]["delay"], pre_var=hh_spikes)
    _connect_shifted(net, n, "hh", "TH", "ctx", "CTX_RS", "TH_CTX", [0], _pathway_peak(params, "TH_CTX") * c["gthcor"], sp["TH_CTX"]["delay"], pre_var=hh_spikes)

    _connect_shifted(net, n, "hh", "GPe", "hh", "STN", "GPe_STN", [0, 1], _pathway_peak(params, "GPe_STN") * c["ggesn"], sp["GPe_STN"]["delay"], pre_var=hh_spikes)

    _connect_shifted(net, n, "hh", "STN", "hh", "GPe", "STN_GPe_AMPA", [0, -1], _pathway_peak(params, "STN_GPe_AMPA") * realization["gsngea"], sp["STN_GPe_AMPA"]["delay"], pre_var=hh_spikes)
    _connect_shifted(net, n, "hh", "STN", "hh", "GPe", "STN_GPe_NMDA", [0, -1], _pathway_peak(params, "STN_GPe_NMDA") * realization["gsngen"], sp["STN_GPe_NMDA"]["delay"], pre_var=hh_spikes)
    _connect_shifted(net, n, "hh", "STN", "hh", "GPi", "STN_GPi", [0, -1], _pathway_peak(params, "STN_GPi") * realization["gsngi"], sp["STN_GPi"]["delay"], pre_var=hh_spikes)

    gpe_gpe_weight = _pathway_peak(params, "GPe_GPe") * 0.25 * (pd * 3.0 + 1.0) * realization["ggege"]
    _connect_shifted(net, n, "hh", "GPe", "hh", "GPe", "GPe_GPe", [1, -2], gpe_gpe_weight, sp["GPe_GPe"]["delay"], pre_var=hh_spikes)
    _connect_shifted(net, n, "hh", "GPe", "hh", "GPi", "GPe_GPi", [1, -2], _pathway_peak(params, "GPe_GPi") * c["ggigi"], sp["GPe_GPi"]["delay"], pre_var=hh_spikes)

    # Striatal output uses every circular shift, reproducing S5 + S51..S59 and S9 + S91..S99.
    all_shifts = list(range(n))
    _connect_shifted(net, n, "hh", "StrD2", "hh", "GPe", "D2_GPe", all_shifts, _pathway_peak(params, "D2_GPe") * c["gstrgpe"], sp["D2_GPe"]["delay"], pre_var=hh_spikes)
    _connect_shifted(net, n, "hh", "StrD1", "hh", "GPi", "D1_GPi", all_shifts, _pathway_peak(params, "D1_GPi") * c["gstrgpi"], sp["D1_GPi"]["delay"], pre_var=hh_spikes)

    # Corticofugal pathways use the Izh reset spike flag, as in the MATLAB t_list_cor update.
    ctx_spikes = "mech.ctx_spikes.rs_spikes"
    _connect_shifted(net, n, "ctx", "CTX_RS", "hh", "StrD2", "CTX_D2", [0], _pathway_peak(params, "CTX_D2") * c["gcorindrstr"], sp["CTX_D2"]["delay"], pre_var=ctx_spikes)
    _connect_shifted(net, n, "ctx", "CTX_RS", "hh", "StrD1", "CTX_D1", [0], _pathway_peak(params, "CTX_D1") * realization["gcordrstr"], sp["CTX_D1"]["delay"], pre_var=ctx_spikes)
    _connect_shifted(net, n, "ctx", "CTX_RS", "hh", "STN", "CTX_STN_AMPA", [0, 1], _pathway_peak(params, "CTX_STN_AMPA") * realization["gcorsna"], sp["CTX_STN_AMPA"]["delay"], pre_var=ctx_spikes)
    _connect_shifted(net, n, "ctx", "CTX_RS", "hh", "STN", "CTX_STN_NMDA", [0, 1], _pathway_peak(params, "CTX_STN_NMDA") * realization["gcorsnn"], sp["CTX_STN_NMDA"]["delay"], pre_var=ctx_spikes)

    # Local cortical E/I alpha synapses use -10 mV crossing flags.  The four
    # independent MATLAB randsample/permutation arrays can choose the same
    # pre/post pair more than once, which is a real repeated contribution in the
    # original equations. Preserve those multapses rather than letting Network
    # filter them out and desynchronize the explicit per-edge weight vectors.
    pre_idx, post_idx = _edge_arrays_from_perms(realization["rs_to_fs_perms"])
    _connect_one_to_one(
        net, "ctx", "CTX_RS", "ctx", "CTX_FS", "RS_FS", pre_idx, post_idx,
        _pathway_peak(params, "RS_FS") * c["gei"], sp["RS_FS"]["delay"],
        pre_var="mech.ctx_spikes.rs_syn_spikes", allow_multapses=True,
    )
    pre_idx, post_idx = _edge_arrays_from_perms(realization["fs_to_rs_perms"])
    _connect_one_to_one(
        net, "ctx", "CTX_FS", "ctx", "CTX_RS", "FS_RS", pre_idx, post_idx,
        _pathway_peak(params, "FS_RS") * c["gie"], sp["FS_RS"]["delay"],
        pre_var="mech.ctx_spikes.fs_syn_spikes", allow_multapses=True,
    )

    # Stash realization/params for downstream inspection. This is not required by Dendra.
    net.kumaravelu_params = params
    return net


# Convenience alias matching the package name.
build = kumaravelu_2016
