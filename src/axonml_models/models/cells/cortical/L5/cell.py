import re
from importlib.resources import files, as_file

import torch
import networkx as nx

import axonml as ax
from axonml.models.utils import distance
from axonml.models.mod import pas

from ..mech import *


_MORPH = files(__package__) / "morphologies"


def valid_ids():
    """Return sorted integer IDs from files like L5_<ID>.gml packaged in morphologies/."""
    ids = []
    for p in _MORPH.iterdir():
        m = re.fullmatch(r"L5_(\d+)\.gml", p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def L5_TTPC_cADpyr(ID, N, integrator=None):
    target = _MORPH / f"L5_{ID}.gml"
    if not target.is_file():
        raise FileNotFoundError(f"Missing morphology: {target}")

    # Give NetworkX a real filesystem path (extracted if needed)
    with as_file(target) as p:
        g = nx.read_gml(p, destringizer=int)

    cell = ax.Tree.from_graph(g, integrator=integrator, N=N, v_init=-70.0)
    for group in ["soma", "apic", "dend", "axon", "myelin", "unmyelin", "node"]:
        cell.slice(group).label(group)

    # insert mechanisms

    # pas
    g = 3e-5 * torch.ones(cell.nc)
    g[cell.find("myelin")] = 1 / 1.125e6

    cell.insert(pas, e=-75.0, g=g[None, :])

    # ih
    d = distance(cell, cell.find("soma"), cell.find("apic"))
    gbar_ih = (-0.869600 + 2.087000 * torch.exp((d) * 0.003100)) * 0.000080

    cell.apic.insert(ih, alias="apical", gbar=gbar_ih[None, :])
    cell.dend.insert(ih, alias="basal", gbar=0.00008)
    cell.soma.insert(ih, alias="soma", gbar=0.00008)

    # im
    cell.apic.insert(im, alias="apical", gbar=0.000143)

    # nats2_t
    cell.apic.insert(nats2_t, alias="apical", gbar=0.026145)
    cell.soma.insert(nats2_t, alias="soma", gbar=0.983955)

    # skv31
    cell.apic.insert(skv3_1, alias="apical", gbar=0.004226)
    cell.soma.insert(skv3_1, alias="soma", gbar=0.303472)
    cell.axon.insert(skv3_1, alias="axon", gbar=1.021945)
    cell.node.insert(skv3_1, alias="node", gbar=1.021945)
    cell.unmyelin.insert(skv3_1, alias="unmyelin", gbar=1.021945)

    # the following channels are inserted in the axon initial segment and soma
    # ca_hva
    cell.soma.insert(ca_hva, alias="soma", gbar=0.000994)
    cell.axon.insert(ca_hva, alias="axon", gbar=0.000990)

    # sk
    cell.soma.insert(sk_e2, alias="soma", gbar=0.008407)
    cell.axon.insert(sk_e2, alias="axon", gbar=0.007104)

    # ca_lva
    cell.soma.insert(ca_lva, alias="soma", gbar=0.000333)
    cell.axon.insert(ca_lva, alias="axon", gbar=0.008752)

    # cadynamics
    cell.soma.insert(cadynamics, alias="soma", gamma=0.000609, decay=210.485284)
    cell.axon.insert(cadynamics, alias="axon", gamma=0.002910, decay=287.198731)

    # these are inserted in the axon initial segment, nodes, and unmyelinated segments
    cell.axon.insert(nap_et2, alias="axon", gbar=0.006827)
    cell.node.insert(nap_et2, alias="node", gbar=0.006827)
    cell.unmyelin.insert(nap_et2, alias="unmyelin", gbar=0.006827)

    cell.axon.insert(k_t, alias="axon", gbar=0.089259)
    cell.node.insert(k_t, alias="node", gbar=0.089259)
    cell.unmyelin.insert(k_t, alias="unmyelin", gbar=0.089259)

    cell.axon.insert(k_p, alias="axon", gbar=0.973538)
    cell.node.insert(k_p, alias="node", gbar=0.973538)
    cell.unmyelin.insert(k_p, alias="unmyelin", gbar=0.973538)

    cell.axon.insert(nata_t, alias="axon", gbar=3.137968)
    cell.node.insert(nata_t, alias="node", gbar=3.137968 * 2.0)
    cell.unmyelin.insert(nata_t, alias="unmyelin", gbar=3.137968)

    cell.equilibria(ek=-85.0, ena=50.0)

    return cell


L5_TTPC_cADpyr.valid_ids = valid_ids
