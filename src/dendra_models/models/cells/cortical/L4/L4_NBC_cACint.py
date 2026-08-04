import re
from importlib.resources import files, as_file

import torch
import networkx as nx

import dendra as dn
from .._utils import distance_from_soma_0, myelin_g
from dendra.models.mod import pas

from ..mech import *


_MORPH = files(__package__) / "L4_NBC_cACint"


def valid_ids():
    ids = []
    for p in _MORPH.iterdir():
        m = re.fullmatch(r"L4_NBC_cACint_(\d+)\.gml", p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def L4_NBC_cACint(ID, N=1, integrator=None):
    target = _MORPH / f"L4_NBC_cACint_{ID}.gml"
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
    cell.soma.insert(pas, e=-69.781406, g=0.00002)
    cell.soma.insert(nats2_t, alias="soma", gbar=0.395881)
    cell.soma.insert(skv3_1, alias="soma", gbar=0.260872)
    cell.soma.insert(ca_hva, alias="soma", gbar=0.000028)
    cell.soma.insert(sk_e2, alias="soma", gbar=0.002099)
    cell.soma.insert(ca_lva, alias="soma", gbar=0.009728)
    cell.soma.insert(nap_et2, alias="soma", gbar=0.000001)
    cell.soma.insert(im, alias="soma", gbar=0.000032)
    cell.soma.insert(k_p, alias="soma", gbar=0.000114)
    cell.soma.insert(k_t, alias="soma", gbar=0.077616)
    cell.soma.insert(cadynamics, alias="soma", gamma=0.000814, decay=967.678789)

    # ais
    cell.ais.insert(pas, e=-64.601696, g=0.000063)
    cell.ais.insert(skv3_1, alias="ais", gbar=0.517764)
    cell.ais.insert(ca_hva, alias="ais", gbar=0.000501)
    cell.ais.insert(sk_e2, alias="ais", gbar=0.005611)
    cell.ais.insert(cadynamics, alias="ais", gamma=0.000503, decay=573.007045)
    cell.ais.insert(im, alias="ais", gbar=0.000345)
    cell.ais.insert(k_p, alias="ais", gbar=0.068460)
    cell.ais.insert(ca_lva, alias="ais", gbar=0.009986)
    cell.ais.insert(nata_t, alias="ais", gbar=3.993125)

    # apic
    cell.apic.insert(pas, e=-63.118492, g=1e-6)
    cell.apic.insert(nats2_t, alias="apical", gbar=0.001373)
    cell.apic.insert(skv3_1, alias="apical", gbar=0.000041)
    d_apic = distance_from_soma_0(cell, cell.find("apic"))
    gbar_ih_apic = (-0.869600 + 2.087000 * torch.exp((d_apic) * 0.003)) * 0.000023
    cell.apic.insert(ih, alias="apical", gbar=gbar_ih_apic[None, :])
    cell.apic.insert(im, alias="apical", gbar=0.000014)
    cell.apic.insert(k_p, alias="apical", gbar=0.00001)
    cell.apic.insert(k_t, alias="apical", gbar=0.007375)

    # dend
    cell.dend.insert(pas, e=-63.118492, g=1e-6)
    cell.dend.insert(nats2_t, alias="basal", gbar=0.001373)
    cell.dend.insert(skv3_1, alias="basal", gbar=0.000041)
    d_dend = distance_from_soma_0(cell, cell.find("dend"))
    gbar_ih_dend = (-0.869600 + 2.087000 * torch.exp((d_dend) * 0.003)) * 0.000023
    cell.dend.insert(ih, alias="basal", gbar=gbar_ih_dend[None, :])
    cell.dend.insert(im, alias="basal", gbar=0.000014)
    cell.dend.insert(k_p, alias="basal", gbar=0.00001)
    cell.dend.insert(k_t, alias="basal", gbar=0.007375)

    # axon
    cell.axon.insert(pas, e=-64.601696, g=0.000063)
    cell.axon.insert(skv3_1, alias="axon", gbar=0.517764)
    cell.axon.insert(k_p, alias="axon", gbar=0.068460)
    cell.axon.insert(nata_t, alias="axon", gbar=3.993125)

    # myelin
    g = myelin_g(cell).squeeze()
    g_myelin = g[cell.find("myelin")]
    cell.myelin.insert(pas, e=-64.601696, g=g_myelin[None, :])

    # node
    cell.node.insert(pas, e=-64.601696, g=0.000063)
    cell.node.insert(skv3_1, alias="node", gbar=0.517764)
    cell.node.insert(k_p, alias="node", gbar=0.068460)
    cell.node.insert(nata_t, alias="node", gbar=3.993125 * 2)

    cell.equilibria(ek=-85.0, ena=50.0)

    return cell


L4_NBC_cACint.valid_ids = valid_ids
