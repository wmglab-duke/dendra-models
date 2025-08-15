from pathlib import Path

import torch
import networkx as nx

import axonml as ax
from axonml.models.utils import distance
from axonml.models.mod import pas

from ..mech import *

_PACKAGE_DIR = Path(__file__).resolve().parent
_MORPH_DIR = _PACKAGE_DIR / "L4_NBC_dNAC"


def valid_ids():
    all_morphs = _MORPH_DIR.glob("L4_NBC_dNAC_*.gml")
    ids = [int(morph.stem.split("_")[1]) for morph in all_morphs]
    return ids


def L4_NBC_dNAC(ID, N, integrator=None):
    gml_path = _MORPH_DIR / f"L4_NBC_dNAC_{ID}.gml"
    g = nx.read_gml(gml_path, destringizer=int)

    cell = ax.Tree.from_graph(g, integrator=integrator, N=N, v_init=-70.0)
    for group in ["soma", "apic", "dend", "axon", "myelin", "unmyelin", "node"]:
        cell.slice(group).label(group)

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

    # apic
    cell.apic.insert(pas, e=-79.315740, g=1e-6)
    cell.apic.insert(nats2_t, alias="apical", gbar=0.000010)
    cell.apic.insert(nap_et2, alias="apical", gbar=0.000001)
    cell.apic.insert(skv3_1, alias="apical", gbar=0.004399)
    d_apic = distance(cell, cell.find("soma"), cell.find("apic"))
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
    d_dend = distance(cell, cell.find("soma"), cell.find("dend"))
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
    cell.myelin.insert(pas, e=-60.216510, g=1 / 1.125e6)

    # unmyelin
    cell.unmyelin.insert(pas, e=-60.216510, g=0.000094)
    cell.unmyelin.insert(skv3_1, alias="unmyelin", gbar=0.317363)
    cell.unmyelin.insert(k_p, alias="unmyelin", gbar=0.004729)
    cell.unmyelin.insert(k_t, alias="unmyelin", gbar=0.098908)
    cell.unmyelin.insert(nata_t, alias="unmyelin", gbar=3.959764)

    # node
    cell.node.insert(pas, e=-60.216510, g=0.000094)
    cell.node.insert(skv3_1, alias="node", gbar=0.317363)
    cell.node.insert(k_p, alias="node", gbar=0.004729)
    cell.node.insert(k_t, alias="node", gbar=0.098908)
    cell.node.insert(nata_t, alias="node", gbar=3.959764*2)

    cell.equilibria(ek=-85.0, ena=50.0)

    return cell


L4_NBC_dNAC.valid_ids = valid_ids
