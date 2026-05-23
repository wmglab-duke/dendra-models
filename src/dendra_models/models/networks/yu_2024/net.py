import numpy as np

import dendra as dn
from dendra.models.mod import exp2syn

from .mechanisms.esser import esser_mech_h
from .mechanisms.synapses import exp2NMDA
from .params import params as parameters
from .utils import PairwiseDistanceStore, Delay

# alias synapses
class exp2syn_r(exp2syn): 
    exp2syn.SAVE('i')
    exp2syn.EXPLICIT('i')

AMPA = exp2syn_r.rename('AMPA')
GABAA = exp2syn_r.rename('GABAA')
GABAB = exp2syn_r.rename('GABAB')
NMDA = exp2NMDA.rename('NMDA')

# helpers

def gaussian(distance, sigma):
    return np.exp(-0.5 * (distance / sigma)**2)

def gen_pos_circle(params):
    lattice = gen_triangular_lattice(params)
    return check_circle(lattice, params)

def gen_triangular_lattice(params):
    x_dist = params["spacing_column"]
    offset = params["diameter_column"] / 2
    y_dist = np.sqrt(x_dist**2 - (x_dist / 2)**2)
    y = np.arange(0, params["diameter_column"], y_dist) - offset

    x_even = np.arange(0, params["diameter_column"], x_dist) - offset
    x_odd = np.arange(x_dist / 2, params["diameter_column"], x_dist) - offset
    
    X, Y = np.meshgrid(x_even, x_odd)

    X[1::2] = x_odd
    
    return np.c_[X.ravel(), Y.ravel()]

def check_circle(points, params):
    r = params["diameter_column"] / 2
    circle = points[:, 0]**2 + points[:, 1]**2
    idx = circle <= r**2
    return points[idx, :]

def sample_between_uniform(low_high, n, rng):
    low, high = low_high
    r = rng.random(n)
    return low + r * (high - low)

def eliminate_self_connections(pre_idx: np.ndarray, post_idx: np.ndarray, *, return_mask=False):
    """
    Remove all positions i where pre_idx[i] == post_idx[i].

    Returns filtered (pre_idx, post_idx); optionally also the boolean mask of kept rows.
    """
    pre_idx  = np.asarray(pre_idx)
    post_idx = np.asarray(post_idx)
    if pre_idx.shape != post_idx.shape:
        raise ValueError(f"Shape mismatch: {pre_idx.shape=} vs {post_idx.shape=}")
    if pre_idx.ndim != 1:
        raise ValueError("Expected 1D arrays")

    keep = pre_idx != post_idx
    if return_mask:
        return pre_idx[keep], post_idx[keep], keep
    return pre_idx[keep], post_idx[keep]

# network builder

def yu_2024(params=None, rng=None):
    if params is None:
        params = parameters
    if rng is None:
        rng = np.random.default_rng(params["seed"])

    locs = gen_pos_circle(params)
    n_columns = len(locs)

    for name, npc in params['cells']['n_per_column'].items():
        n_total = n_columns * npc
        params['cells'].setdefault('n_total', {}).update({name: n_total})

    for name, npc in params['cells']['n_per_column'].items():
        cell_locs = np.repeat(locs, npc, axis=0)
        z = sample_between_uniform(params['depth'][name], len(cell_locs), rng)
        cell_locs = np.hstack([cell_locs, z[:,None]])
        params['cells'].setdefault('locs', {}).update({name: cell_locs})
    
    distances_2d = PairwiseDistanceStore()
    distances_3d = PairwiseDistanceStore()
    for name, arr in params['cells']['locs'].items():
        distances_2d.add(name, arr[:, :2])
        distances_3d.add(name, arr)

    n_cells = [n for n in params['cells']['n_total'].values()]
    n_cells_total = sum(n_cells)

    # netstims for intrinsic activity
    n_netstim_per_group = [n for n in params['netstim']['n_per_group'].values()]
    n_netstim_total = sum(n_netstim_per_group) + sum(n_cells)

    FR = params['FR']

    intervals = []
    for n in n_netstim_per_group:
        intervals.extend([1000.0/params['netstim']['rate']] * n)
    for name, n in params['cells']['n_total'].items():
        intervals.extend([1000.0 / FR[name]] * n)
        
    ns = dn.NetStim(n_netstim_total, interval=intervals, noise=1.0)

    # make subpops
    idx = 0
    for name, n in params['netstim']['n_per_group'].items():
        ns[idx:idx+n].label(f"{name}_input")
        idx += n
    for name, n in params['cells']['n_total'].items():
        ns[idx:idx+n].label(f"{name}_noise")
        idx += n

    # instantiate pop & biophysics

    idx = 0
    p = dn.Population(C=n_cells_total, v_init=-77.5, integrator=dn.scnv())

    for name, n in params['cells']['n_total'].items():
        end = idx + n
        p[0, idx:end].label(name)
        idx = end

    esser_params = {}
    esser_params["L5"] = dict(
        v_iaf0 = -78.33,
        theta_eq = -53.0,
        tau_theta = 0.5,
        tau_spike = 0.6,
        tau_m = 13.0,
        gNa_leak = 0.14,
        gK_leak = 1.3,
        tspike = 0.75
    )
    esser_params["L"] = dict(
        v_iaf0 = -75.263,
        theta_eq = -53,
        tau_theta = 2.0,
        tau_spike = 1.75,
        tau_m = 15.0,
        gNa_leak = 0.14,
        gK_leak = 1.0,
        tspike = 2.0
    )
    esser_params["I"] = dict(
        v_iaf0 = -70.0,
        theta_eq = -54.0,
        tau_theta = 1.0,
        tau_spike = 0.48,
        tau_m = 7.0,
        gNa_leak = 0.2,
        gK_leak = 1.0,
        tspike = 0.75
    )


    for celltype in params['cells']['n_total']:
        if celltype.startswith("L"):
            if celltype == "L5":
                esser_p = esser_params["L5"]
            else:
                esser_p = esser_params["L"]
        else:
            esser_p = esser_params["I"]
        getattr(p, celltype).insert(esser_mech_h.rename('esser_mech'), **esser_p)

    p.insert(AMPA, **params['syn']['AMPA'])
    p.insert(NMDA, **params['syn']['NMDA'])
    p.insert(GABAA, **params['syn']['GABAA'])
    p.insert(GABAB, **params['syn']['GABAB'])

    fp32 = params.get("fp32", False)
    if fp32:
        net = dn.Network({'cells':p}, netstim=ns).float()
    else:
        net = dn.Network({'cells':p}, netstim=ns).double()

    # set references
    m = net.cells.mech
    m.esser_mech.DE['esser_states'].setreference('i_ampa', lambda: m.AMPA.i_)
    m.esser_mech.DE['esser_states'].setreference('i_nmda', lambda: m.NMDA.i_)
    m.esser_mech.DE['esser_states'].setreference('i_gaba_a', lambda: m.GABAA.i_)
    m.esser_mech.DE['esser_states'].setreference('i_gaba_b', lambda: m.GABAB.i_)

    # connect noise netstims

    for name in params['cells']['n_total']:
        pre = f"{name}_noise"
        post = name
        net.connect_one_to_one(
            getattr(net.netstim, pre),
            getattr(net.cells, post),
            net.cells.mech.esser_mech,
            weight=10.0
        )
    
    # synapses
    pscale = params["pscale"]

    for pre_celltype in params['cells']['celltypes']:
        p_dict = params["con"]["p"][pre_celltype]
        s_dict = params["con"]["sigma"][pre_celltype]
        for post_celltype in p_dict:
            pmax = p_dict[post_celltype]
            sigma = s_dict[post_celltype]
            p = pscale * pmax * gaussian(distances_2d.get(pre_celltype, post_celltype), sigma)
            pre_idx, post_idx = np.where(rng.random(p.shape) < p)
            # avoid self-connections
            if params["forbid_autapses"]:
                if pre_celltype == post_celltype:
                    pre_idx, post_idx = eliminate_self_connections(pre_idx, post_idx)
            dist = distances_3d.get(pre_celltype, post_celltype)[pre_idx, post_idx]
            delay_ = Delay(dist, params['conduction_velocity'], params['syn_delay'])
            for syntype in params["con"]["syn_types"][pre_celltype][post_celltype]:
                weight = (1 / pscale) * params["syn"]["gpeak"][syntype] * params["con"]["strength"][pre_celltype][post_celltype][syntype]
                weight = [weight] * len(dist)
                net.connect_one_to_one(
                    getattr(net.cells, pre_celltype)[pre_idx],
                    getattr(net.cells, post_celltype)[post_idx],
                    getattr(net.cells.mech, syntype),
                    threshold=None,
                    delay=delay_,
                    weight=weight,
                    pre_var='mech.esser_mech.spikes'
                )

    return net