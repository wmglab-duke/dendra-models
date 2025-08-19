import re
from importlib.resources import files, as_file

import torch
import networkx as nx

import axonml as ax
from axonml.models.utils import distance
from axonml.models.mod import pas

from ..mech import *


_MORPH = files(__package__) / "L4_SBC_bNAC"


def valid_ids():
    ids = []
    for p in _MORPH.iterdir():
        m = re.fullmatch(r"L4_SBC_bNAC_(\d+)\.gml", p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def L4_SBC_bNAC(ID, N, integrator=None):
    target = _MORPH / f"L4_SBC_bNAC_{ID}.gml"
    if not target.is_file():
        raise FileNotFoundError(f"Missing morphology: {target}. Valid IDs: {valid_ids()}")

    # Give NetworkX a real filesystem path (extracted if needed)
    with as_file(target) as p:
        g = nx.read_gml(p, destringizer=int)

    cell = ax.Tree.from_graph(g, integrator=integrator, N=N, v_init=-70.0)
    for group in ["soma", "apic", "dend", "axon", "myelin", "unmyelin", "node"]:
        cell.slice(group).label(group)

    # insert mechanisms

    # soma
    cell.soma.insert(pas, e=-67.128897, g=0.000100)
    cell.soma.insert(nats2_t, alias="soma", gbar=0.150747)
    cell.soma.insert(skv3_1, alias="soma", gbar=0.503893)
    cell.soma.insert(ca_hva, alias="soma", gbar=0.000174)
    cell.soma.insert(sk_e2, alias="soma", gbar=0.000523)
    cell.soma.insert(ca_lva, alias="soma", gbar=0.003242)
    cell.soma.insert(nap_et2, alias="soma", gbar=0.000001)
    cell.soma.insert(im, alias="soma", gbar=0.000478)
    cell.soma.insert(k_p, alias="soma", gbar=0.005446)
    cell.soma.insert(k_t, alias="soma", gbar=0.039863)
    cell.soma.insert(cadynamics, alias="soma", gamma=0.000500, decay=645.079741)

    # apic
    cell.apic.insert(pas, e=-60.295916, g=1e-6)
    cell.apic.insert(nats2_t, alias="apical", gbar=0.000229)
    cell.apic.insert(skv3_1, alias="apical", gbar=0.000083)
    d_apic = distance(cell, cell.find("soma"), cell.find("apic"))
    gbar_ih_apic = (-0.869600 + 2.087000 * torch.exp((d_apic) * 0.003)) * 0.000049
    cell.apic.insert(ih, alias="apical", gbar=gbar_ih_apic[None, :])
    cell.apic.insert(im, alias="apical", gbar=0.000022)
    cell.apic.insert(k_p, alias="apical", gbar=0.00001)
    cell.apic.insert(k_t, alias="apical", gbar=0.001511)

    # dend
    cell.dend.insert(pas, e=-60.295916, g=1e-6)
    cell.dend.insert(nats2_t, alias="basal", gbar=0.000229)
    cell.dend.insert(skv3_1, alias="basal", gbar=0.000083)
    d_dend = distance(cell, cell.find("soma"), cell.find("dend"))
    gbar_ih_dend = (-0.869600 + 2.087000 * torch.exp((d_dend) * 0.003)) * 0.000049
    cell.dend.insert(ih, alias="basal", gbar=gbar_ih_dend[None, :])
    cell.dend.insert(im, alias="basal", gbar=0.000022)
    cell.dend.insert(k_p, alias="basal", gbar=0.00001)
    cell.dend.insert(k_t, alias="basal", gbar=0.001511)

    # axon
    cell.axon.insert(pas, e=-63.854018, g=0.000008)
    cell.axon.insert(skv3_1, alias="axon", gbar=0.386953)
    cell.axon.insert(ca_hva, alias="axon", gbar=0.000400)
    cell.axon.insert(sk_e2, alias="axon", gbar=0.001224)
    cell.axon.insert(cadynamics, alias="axon", gamma=0.001739, decay=468.069681)
    cell.axon.insert(nap_et2, alias="axon", gbar=0.000001)
    cell.axon.insert(im, alias="axon", gbar=0.000554)
    cell.axon.insert(k_p, alias="axon", gbar=0.001693)
    cell.axon.insert(k_t, alias="axon", gbar=0.042115)
    cell.axon.insert(ca_lva, alias="axon", gbar=0.009017)
    cell.axon.insert(nata_t, alias="axon", gbar=3.999855)

    # myelin
    cell.myelin.insert(pas, e=-63.854018, g=1 / 1.125e6)

    # unmyelin
    cell.unmyelin.insert(pas, e=-63.854018, g=0.000008)
    cell.unmyelin.insert(skv3_1, alias="unmyelin", gbar=0.386953)
    cell.unmyelin.insert(k_p, alias="unmyelin", gbar=0.001693)
    cell.unmyelin.insert(k_t, alias="unmyelin", gbar=0.042115)
    cell.unmyelin.insert(nata_t, alias="unmyelin", gbar=3.999855)
    cell.unmyelin.insert(nap_et2, alias="unmyelin", gbar=0.000001)

    # node
    cell.node.insert(pas, e=-63.854018, g=0.000008)
    cell.node.insert(skv3_1, alias="node", gbar=0.386953)
    cell.node.insert(k_p, alias="node", gbar=0.001693)
    cell.node.insert(k_t, alias="node", gbar=0.042115)
    cell.node.insert(nata_t, alias="node", gbar=3.999855 * 2)
    cell.node.insert(nap_et2, alias="node", gbar=0.000001)

    cell.equilibria(ek=-85.0, ena=50.0)

    return cell


L4_SBC_bNAC.valid_ids = valid_ids
