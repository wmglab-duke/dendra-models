import re
from importlib.resources import files, as_file

import torch
import networkx as nx

import dendra as dn
from .._utils import distance_from_soma_0, myelin_g
from dendra.models.mod import pas

from ..mech import *


_MORPH = files(__package__) / "L4_NBC_dNAC"


def valid_ids():
    ids = []
    for p in _MORPH.iterdir():
        m = re.fullmatch(r"L4_NBC_dNAC_(\d+)\.gml", p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def L4_NBC_dNAC(ID, N=1, integrator=None):
    target = _MORPH / f"L4_NBC_dNAC_{ID}.gml"
    if not target.is_file():
        raise FileNotFoundError(f"Missing morphology: {target}. Valid IDs: {valid_ids()}")

    # Give NetworkX a real filesystem path (extracted if needed)
    with as_file(target) as p:
        g = nx.read_gml(p, destringizer=int)

    cell = dn.Tree.from_graph(g, integrator=integrator, N=N, v_init=-70.0)
    exclude = ["branchpoint"]
    for group, exc, name in [
        ("soma", [], "soma"), 
        ("apic", [], "apic"), 
        ("dend", [], "dend"), 
        ("axon_initial_segment", [], "ais"), 
        ("axon", ["axon_initial_segment"], "axon"), 
        ("myelin", [], "myelin"),
        ("node", [], "node")
    ]:
        cell.slice(group, exclude=exclude+exc).label(name, replace=True)

    # insert mechanisms

    # soma
    cell.soma.insert(pas, e=-62.442793, g=0.000091)
    cell.soma.insert(nats2_t, alias="soma", gbar=0.197999)
    cell.soma.insert(skv3_1, alias="soma", gbar=0.297559)
    cell.soma.insert(ca_hva, alias="soma", gbar=0.000032)
    cell.soma.insert(sk_e2, alias="soma", gbar=0.019726)
    cell.soma.insert(ca_lva, alias="soma", gbar=0.001067)
    cell.soma.insert(nap_et2, alias="soma", gbar=0.000001)
    cell.soma.insert(im, alias="soma", gbar=0.000008)
    cell.soma.insert(kdshu2007, alias="soma", gbar=0.000425)
    cell.soma.insert(k_p, alias="soma", gbar=0.156376)
    cell.soma.insert(k_t, alias="soma", gbar=0.092965)
    cell.soma.insert(cadynamics, alias="soma", gamma=0.000511, decay=731.707637)

    # axon initial segment
    cell.ais.insert(pas, e=-60.216510, g=0.000094)
    cell.ais.insert(skv3_1, alias="ais", gbar=0.317363)
    cell.ais.insert(ca_hva, alias="ais", gbar=0.000003)
    cell.ais.insert(sk_e2, alias="ais", gbar=0.003442)
    cell.ais.insert(cadynamics, alias="ais", gamma=0.010353, decay=64.277990)
    cell.ais.insert(im, alias="ais", gbar=0.000999)
    cell.ais.insert(k_p, alias="ais", gbar=0.004729)
    cell.ais.insert(k_t, alias="ais", gbar=0.098908)
    cell.ais.insert(ca_lva, alias="ais", gbar=0.000015)
    cell.ais.insert(nata_t, alias="ais", gbar=3.959764)

    # apic
    cell.apic.insert(pas, e=-79.315740, g=1e-6)
    cell.apic.insert(nats2_t, alias="apical", gbar=0.000010)
    cell.apic.insert(nap_et2, alias="apical", gbar=0.000001)
    cell.apic.insert(skv3_1, alias="apical", gbar=0.004399)
    d_apic = distance_from_soma_0(cell, cell.find("apic"))
    gbar_ih_apic = (-0.869600 + 2.087000 * torch.exp((d_apic) * 0.003)) * 0.000052
    cell.apic.insert(ih, alias="apical", gbar=gbar_ih_apic[None, :])
    cell.apic.insert(im, alias="apical", gbar=0.000008)
    cell.apic.insert(kdshu2007, alias="apical", gbar=0.000483)
    cell.apic.insert(k_p, alias="apical", gbar=0.00001)
    cell.apic.insert(k_t, alias="apical", gbar=0.009500)

    # dend
    cell.dend.insert(pas, e=-79.315740, g=1e-6)
    cell.dend.insert(nats2_t, alias="basal", gbar=0.000010)
    cell.dend.insert(nap_et2, alias="basal", gbar=0.000001)
    cell.dend.insert(skv3_1, alias="basal", gbar=0.004399)
    d_dend = distance_from_soma_0(cell, cell.find("dend"))
    gbar_ih_dend = (-0.869600 + 2.087000 * torch.exp((d_dend) * 0.003)) * 0.000023
    cell.dend.insert(ih, alias="basal", gbar=gbar_ih_dend[None, :])
    cell.dend.insert(im, alias="basal", gbar=0.000008)
    cell.dend.insert(kdshu2007, alias="basal", gbar=0.000483)
    cell.dend.insert(k_p, alias="basal", gbar=0.00001)
    cell.dend.insert(k_t, alias="basal", gbar=0.009500)

    # axon
    cell.axon.insert(pas, e=-60.216510, g=0.000094)
    cell.axon.insert(skv3_1, alias="axon", gbar=0.317363)
    cell.axon.insert(ca_hva, alias="axon", gbar=0.000003)
    cell.axon.insert(sk_e2, alias="axon", gbar=0.003442)
    cell.axon.insert(cadynamics, alias="axon", gamma=0.010353, decay=64.277990)
    cell.axon.insert(im, alias="axon", gbar=0.000999)
    cell.axon.insert(k_p, alias="axon", gbar=0.004729)
    cell.axon.insert(k_t, alias="axon", gbar=0.098908)
    cell.axon.insert(ca_lva, alias="axon", gbar=0.000015)
    cell.axon.insert(nata_t, alias="axon", gbar=3.959764)

    # myelin
    g = myelin_g(cell, cell.find("myelin")).squeeze()
    g_myelin = g[cell.find("myelin")]
    cell.myelin.insert(pas, e=-60.216510, g=g_myelin[None, :])

    # node
    cell.node.insert(pas, e=-60.216510, g=0.000094)
    cell.node.insert(skv3_1, alias="node", gbar=0.317363)
    cell.node.insert(k_p, alias="node", gbar=0.004729)
    cell.node.insert(k_t, alias="node", gbar=0.098908)
    cell.node.insert(nata_t, alias="node", gbar=3.959764 * 2)

    # axon
    cell.axon.insert(pas, e=-60.216510, g=0.000094)
    cell.axon.insert(skv3_1, alias="axon", gbar=0.317363)
    cell.axon.insert(k_p, alias="axon", gbar=0.004729)
    cell.axon.insert(k_t, alias="axon", gbar=0.098908)
    cell.axon.insert(nata_t, alias="axon", gbar=3.959764)

    cell.equilibria(ek=-85.0, ena=50.0)

    return cell


L4_NBC_dNAC.valid_ids = valid_ids
