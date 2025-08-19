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
    """Return sorted integer IDs from files like L23_<ID>.gml packaged in morphologies/."""
    ids = []
    for p in _MORPH.iterdir():
        m = re.fullmatch(r"L23_(\d+)\.gml", p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def L23_PC_cADpyr(ID, N, integrator=None):
    target = _MORPH / f"L23_{ID}.gml"
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
    cell.apic.insert(im, alias="apical", gbar=0.00074)

    # nats2_t
    cell.apic.insert(nats2_t, alias="apical", gbar=0.012009)
    cell.soma.insert(nats2_t, alias="soma", gbar=0.926705)

    # skv31
    cell.apic.insert(skv3_1, alias="apical", gbar=0.000513)
    cell.soma.insert(skv3_1, alias="soma", gbar=0.102517)
    cell.axon.insert(skv3_1, alias="axon", gbar=0.094971)
    cell.node.insert(skv3_1, alias="node", gbar=0.094971)
    cell.unmyelin.insert(skv3_1, alias="unmyelin", gbar=0.094971)

    # the following channels are inserted in the axon initial segment and soma
    # ca_hva
    cell.soma.insert(ca_hva, alias="soma", gbar=0.000374)
    cell.axon.insert(ca_hva, alias="axon", gbar=0.000306)

    # sk
    cell.soma.insert(sk_e2, alias="soma", gbar=0.099433)
    cell.axon.insert(sk_e2, alias="axon", gbar=0.008085)

    # ca_lva
    cell.soma.insert(ca_lva, alias="soma", gbar=0.000778)
    cell.axon.insert(ca_lva, alias="axon", gbar=0.000050)

    # cadynamics
    cell.soma.insert(cadynamics, alias="soma", gamma=0.000533, decay=342.544232)
    cell.axon.insert(cadynamics, alias="axon", gamma=0.016713, decay=384.114655)

    # these are inserted in the axon initial segment, nodes, and unmyelinated segments
    cell.axon.insert(nap_et2, alias="axon", gbar=0.009803)
    cell.node.insert(nap_et2, alias="node", gbar=0.009803)
    cell.unmyelin.insert(nap_et2, alias="unmyelin", gbar=0.009803)

    cell.axon.insert(k_t, alias="axon", gbar=0.001035)
    cell.node.insert(k_t, alias="node", gbar=0.001035)
    cell.unmyelin.insert(k_t, alias="unmyelin", gbar=0.001035)

    cell.axon.insert(k_p, alias="axon", gbar=0.959296)
    cell.node.insert(k_p, alias="node", gbar=0.959296)
    cell.unmyelin.insert(k_p, alias="unmyelin", gbar=0.959296)

    cell.axon.insert(nata_t, alias="axon", gbar=3.429725)
    cell.node.insert(nata_t, alias="node", gbar=3.429725 * 2.0)
    cell.unmyelin.insert(nata_t, alias="unmyelin", gbar=3.429725)

    cell.equilibria(ek=-85.0, ena=50.0)

    return cell


L23_PC_cADpyr.valid_ids = valid_ids
