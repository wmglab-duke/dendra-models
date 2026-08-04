import re
from importlib.resources import files, as_file

import torch
import networkx as nx

import dendra as dn
from .._utils import distance_from_soma_0, myelin_g
from dendra.models.mod import pas

from ..mech import *


_MORPH = files(__package__) / "L23_PC_cADpyr"


def valid_ids():
    """Return sorted integer IDs from files like L23_PC_cADpyr<ID>.gml packaged in L23_PC_cADpyr/."""
    ids = []
    for p in _MORPH.iterdir():
        m = re.fullmatch(r"L23_PC_cADpyr(\d+)\.gml", p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def L23_PC_cADpyr(ID, N=1, integrator=None):
    target = _MORPH / f"L23_PC_cADpyr_{ID}.gml"
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

    # pas
    g = myelin_g(cell)
    cell.insert(pas, e=-75.0, g=g)

    # basal dendrites
    cell.dend.insert(ih, alias="basal", gbar=0.00008)

    # apical dendrites
    d = distance_from_soma_0(cell, cell.find("apic"))
    gbar_ih = (-0.869600 + 2.087000 * torch.exp((d) * 0.003100)) * 0.000080
    cell.apic.insert(ih, alias="apical", gbar=gbar_ih[None, :])
    cell.apic.insert(im, alias="apical", gbar=0.00074)
    cell.apic.insert(nats2_t, alias="apical", gbar=0.012009)
    cell.apic.insert(skv3_1, alias="apical", gbar=0.000513)

    # axon initial segment
    cell.ais.insert(ca_hva, alias="ais", gbar=0.000306)
    cell.ais.insert(skv3_1, alias="ais", gbar=0.094971)
    cell.ais.insert(sk_e2, alias="ais", gbar=0.008085)
    cell.ais.insert(cadynamics, alias="ais", gamma=0.016713, decay=384.114655)
    cell.ais.insert(nap_et2, alias="ais", gbar=0.009803)
    cell.ais.insert(k_p, alias="ais", gbar=0.959296)
    cell.ais.insert(k_t, alias="ais", gbar=0.001035)
    cell.ais.insert(ca_lva, alias="ais", gbar=0.000050)
    cell.ais.insert(nata_t, alias="ais", gbar=3.429725)

    # soma
    cell.soma.insert(ca_hva, alias="soma", gbar=0.000374)
    cell.soma.insert(skv3_1, alias="soma", gbar=0.102517)
    cell.soma.insert(sk_e2, alias="soma", gbar=0.099433)
    cell.soma.insert(ca_lva, alias="soma", gbar=0.000778)
    cell.soma.insert(ih, alias="soma", gbar=0.00008)
    cell.soma.insert(nats2_t, alias="soma", gbar=0.926705)
    cell.soma.insert(cadynamics, alias="soma", gamma=0.000533, decay=342.544232)

    # nodes of ranvier
    cell.node.insert(skv3_1, alias="node", gbar=0.094971)
    cell.node.insert(nap_et2, alias="node", gbar=0.009803)
    cell.node.insert(k_p, alias="node", gbar=0.959296)
    cell.node.insert(k_t, alias="node", gbar=0.001035)
    cell.node.insert(nata_t, alias="node", gbar=3.429725 * 2.0)

    # axon
    cell.axon.insert(skv3_1, alias="axon", gbar=0.094971)
    cell.axon.insert(nap_et2, alias="axon", gbar=0.009803)
    cell.axon.insert(k_p, alias="axon", gbar=0.959296)
    cell.axon.insert(k_t, alias="axon", gbar=0.001035)
    cell.axon.insert(nata_t, alias="axon", gbar=3.429725)

    cell.equilibria(ek=-85.0, ena=50.0)

    return cell


L23_PC_cADpyr.valid_ids = valid_ids
