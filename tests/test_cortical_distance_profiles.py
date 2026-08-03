from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import as_file, files

import dendra as dn  # configure Dendra before importing torch
import networkx as nx
import pytest
import torch

from dendra_models.models.cells.cortical import (
    L23_PC_cADpyr,
    L4_LBC_cACint,
    L4_LBC_dNAC,
    L4_NBC_cACint,
    L4_NBC_dNAC,
    L4_SBC_bNAC,
    L4_SBC_cACint,
    L5_TTPC_cADpyr,
)


DTYPE = torch.float64


@dataclass(frozen=True)
class Profile:
    region: str
    alias: str
    scale: float
    exponent: float | None


@dataclass(frozen=True)
class CorticalCase:
    name: str
    constructor: object
    model_id: int
    package: str
    resource_parts: tuple[str, ...]
    profiles: tuple[Profile, ...]


L23_L5_PROFILES = (
    Profile("apic", "apical", 0.000080, 0.003100),
    Profile("dend", "basal", 0.000080, None),
)
CAC_PROFILES = (
    Profile("apic", "apical", 0.000023, 0.003000),
    Profile("dend", "basal", 0.000023, 0.003000),
)
DNAC_PROFILES = (
    Profile("apic", "apical", 0.000052, 0.003000),
    Profile("dend", "basal", 0.000023, 0.003000),
)
SBC_BNAC_PROFILES = (
    Profile("apic", "apical", 0.000049, 0.003000),
    Profile("dend", "basal", 0.000049, 0.003000),
)


CASES = (
    CorticalCase(
        "L23_PC_cADpyr",
        L23_PC_cADpyr,
        1,
        "dendra_models.models.cells.cortical.L23",
        ("L23_PC_cADpyr", "L23_PC_cADpyr_1.gml"),
        L23_L5_PROFILES,
    ),
    CorticalCase(
        "L5_TTPC_cADpyr",
        L5_TTPC_cADpyr,
        1,
        "dendra_models.models.cells.cortical.L5",
        ("morphologies", "L5_1.gml"),
        L23_L5_PROFILES,
    ),
    CorticalCase(
        "L4_LBC_cACint",
        L4_LBC_cACint,
        9477,
        "dendra_models.models.cells.cortical.L4",
        ("L4_LBC_cACint", "L4_LBC_cACint_9477.gml"),
        CAC_PROFILES,
    ),
    CorticalCase(
        "L4_LBC_dNAC",
        L4_LBC_dNAC,
        7899,
        "dendra_models.models.cells.cortical.L4",
        ("L4_LBC_dNAC", "L4_LBC_dNAC_7899.gml"),
        DNAC_PROFILES,
    ),
    CorticalCase(
        "L4_NBC_cACint",
        L4_NBC_cACint,
        103472,
        "dendra_models.models.cells.cortical.L4",
        ("L4_NBC_cACint", "L4_NBC_cACint_103472.gml"),
        CAC_PROFILES,
    ),
    CorticalCase(
        "L4_NBC_dNAC",
        L4_NBC_dNAC,
        8989,
        "dendra_models.models.cells.cortical.L4",
        ("L4_NBC_dNAC", "L4_NBC_dNAC_8989.gml"),
        DNAC_PROFILES,
    ),
    CorticalCase(
        "L4_SBC_bNAC",
        L4_SBC_bNAC,
        2,
        "dendra_models.models.cells.cortical.L4",
        ("L4_SBC_bNAC", "L4_SBC_bNAC_2.gml"),
        SBC_BNAC_PROFILES,
    ),
    CorticalCase(
        "L4_SBC_cACint",
        L4_SBC_cACint,
        11826,
        "dendra_models.models.cells.cortical.L4",
        ("L4_SBC_cACint", "L4_SBC_cACint_11826.gml"),
        CAC_PROFILES,
    ),
)


def _load_graph(case):
    resource = files(case.package).joinpath(*case.resource_parts)
    with as_file(resource) as path:
        return nx.read_gml(path, destringizer=int)


def _material_nodes(graph, region):
    token = f".{region}["
    return sorted(
        node
        for node, attributes in graph.nodes(data=True)
        if token in attributes["name"] and float(attributes["L"]) > 0.0
    )


def _distances_from_soma_0(graph, targets):
    soma_nodes = _material_nodes(graph, "soma")
    assert len(soma_nodes) == 1
    soma = soma_nodes[0]
    centre_distances = nx.single_source_dijkstra_path_length(
        graph.to_undirected(as_view=True),
        soma,
        weight="L",
    )
    soma_0_offset = 0.5 * float(graph.nodes[soma]["L"])
    return torch.tensor(
        [centre_distances[node] + soma_0_offset for node in targets],
        dtype=DTYPE,
    )


def _expected_conductance(profile, distances):
    if profile.exponent is None:
        return torch.full_like(distances, profile.scale)
    return (
        -0.869600
        + 2.087000 * torch.exp(profile.exponent * distances)
    ) * profile.scale


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
def test_cortical_ih_profiles_match_neuron_soma_0_distance(case):
    graph = _load_graph(case)

    with dn.ctx(DTYPE=DTYPE):
        cell = case.constructor(case.model_id, 1)
        cell.initialize()

    for profile in case.profiles:
        nodes = _material_nodes(graph, profile.region)
        assert nodes, f"{case.name} ID {case.model_id} has no {profile.region}"
        assert cell.find(profile.region, as_list=True) == nodes

        distances = _distances_from_soma_0(graph, nodes)
        expected = _expected_conductance(profile, distances)
        parameter = getattr(cell.mech.ih, f"gbar_{profile.alias}")()
        actual = parameter.detach().cpu().reshape(-1)
        if actual.numel() == 1:
            actual = actual.expand_as(expected)

        assert actual.shape == expected.shape
        torch.testing.assert_close(actual, expected, rtol=2.0e-6, atol=2.0e-10)


def test_cortical_distance_profile_handles_an_empty_apical_region():
    with dn.ctx(DTYPE=DTYPE):
        cell = L4_LBC_cACint(1, 1)
        cell.initialize()

    assert cell.find("apic", as_list=True) == []
    assert cell.find("dend", as_list=True)
