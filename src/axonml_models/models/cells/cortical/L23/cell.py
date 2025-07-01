from pathlib import Path

import torch
import networkx as nx

import axonml as ax
from axonml.models.utils import distance
from axonml.models.mod import pas
from axonml.models.mechanisms import equilibria

from ..mech import *

_PACKAGE_DIR = Path(__file__).resolve().parent
_MORPH_DIR   = _PACKAGE_DIR / "morphologies"


def L23(ID, N, integrator=None):
    gml_path = _MORPH_DIR / f"L23_{ID}.gml"
    g = nx.read_gml(gml_path, destringizer=int)

    cell = ax.Tree.from_graph(g, integrator=integrator, N=N, v_init=-70.0)
    for group in ['soma', 'apic', 'dend', 'axon', 'myelin', 'unmyelin', 'node']:
        cell[:, cell.find(group)].label(group)    

    # pas
    g = 3e-5 * torch.ones(cell.nc)
    g[cell.find('myelin')] = 1 / 1.125e6
    cell.insert(pas, e=-75.0, g=g[None, :])

    # ih
    d = distance(cell, cell.find('soma'), cell.find('apic'))
    gbar_ih = (-0.869600 + 2.087000*torch.exp((d)*0.003100))*0.000080
    cell.apic.insert(ih, alias='apical', gbar=gbar_ih[None, :])
    cell.dend.insert(ih, alias='basal',  gbar=0.00008)
    cell.soma.insert(ih, alias='soma',   gbar=0.00008)

    # im
    cell.apic.insert(im, alias='apical', gbar=0.00074)

    # nats2_t
    cell.apic.insert(nats2_t, alias='apical', gbar=0.012009)
    cell.soma.insert(nats2_t, alias='soma',   gbar=0.926705)

    # skv31
    cell.apic.insert(skv3_1,        alias='apical',     gbar=0.000513)
    cell.soma.insert(skv3_1,        alias='soma',       gbar=0.102517)
    cell.axon.insert(skv3_1,        alias='axon',       gbar=0.094971)
    cell.node.insert(skv3_1,        alias='node',       gbar=0.094971)
    cell.unmyelin.insert(skv3_1,    alias='unmyelin',   gbar=0.094971)

    # the following channels are inserted in the axon initial segment and soma
    # ca_hva
    cell.soma.insert(ca_hva,        alias='soma', gbar=0.000374)
    cell.axon.insert(ca_hva,        alias='axon', gbar=0.000306)

    # sk
    cell.soma.insert(sk_e2,         alias='soma', gbar=0.099433)
    cell.axon.insert(sk_e2,         alias='axon', gbar=0.008085)

    # ca_lva
    cell.soma.insert(ca_lva,        alias='soma', gbar=0.000778)
    cell.axon.insert(ca_lva,        alias='axon', gbar=0.000050)

    # cadynamics
    cell.soma.insert(cadynamics, alias='soma', gamma=0.000533, decay=342.544232)
    cell.axon.insert(cadynamics, alias='axon', gamma=0.016713, decay=384.114655)

    # these are inserted in the axon initial segment, nodes, and unmyelinated segments
    cell.axon.insert(nap_et2,       alias='axon',       gbar=0.009803)
    cell.node.insert(nap_et2,       alias='node',       gbar=0.009803)
    cell.unmyelin.insert(nap_et2,   alias='unmyelin',   gbar=0.009803)

    cell.axon.insert(k_t,           alias='axon',       gbar=0.001035)
    cell.node.insert(k_t,           alias='node',       gbar=0.001035)
    cell.unmyelin.insert(k_t,       alias='unmyelin',   gbar=0.001035)

    cell.axon.insert(k_p,           alias='axon',       gbar=0.959296)
    cell.node.insert(k_p,           alias='node',       gbar=0.959296)
    cell.unmyelin.insert(k_p,       alias='unmyelin',   gbar=0.959296)

    cell.axon.insert(nata_t,        alias='axon',       gbar=3.429725)
    cell.node.insert(nata_t,        alias='node',       gbar=3.429725*2.0)
    cell.unmyelin.insert(nata_t,    alias='unmyelin',   gbar=3.429725)

    with equilibria(ek=-85.0, ena=50.0):
        cell.build()

    return cell
