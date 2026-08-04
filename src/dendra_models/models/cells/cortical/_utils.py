"""Shared helpers for translated cortical-cell biophysics."""

from dendra.models.utils import distance
import torch


def distance_from_soma_0(cell, targets):
    """Return path distances from the source model's ``soma(0)`` origin.

    The imported cortical resistor graphs represent each one-segment soma at
    its computational centre, ``soma(0.5)``, and all dendrites attach there.
    SimNIBS/NEURON instead anchors distance-dependent biophysics at
    ``soma(0)``. Include the missing half-soma path explicitly.
    """
    if cell.soma.shape[-1] != 1:
        raise ValueError(
            "distance_from_soma_0 requires a single-compartment soma."
        )
    soma_0_offset = 0.5 * cell.soma.dx[0, 0].item()
    return distance(
        cell, cell.find("soma"), targets, origin_offset=soma_0_offset
    )


def myelin_g(cell):
    myelin_idx = cell.find("myelin")

    g = torch.full_like(cell.diam[0], 3e-5)
    diam = cell.myelin.diam[0]
    g_ratio = 0.58112771 + 0.08340535 * torch.log(diam[0])

    myelin_membrane_g_pas = 1.659e-3

    segment_radius = diam / 2
    segment_myelin_thickness = (diam * (1 / g_ratio) - diam) / 2

    layer_thickness = diam.new_tensor(0.005)

    n_layers = torch.floor(
        segment_myelin_thickness / layer_thickness
    ).long()

    if (n_layers < 1).any().item():
        raise ValueError("Every myelinated segment must contain at least one layer")

    # Construct enough columns for the segment with the most layers.
    layer_idx = torch.arange(
        int(n_layers.max().item()),
        device=diam.device,
    )

    layer_radii = (
        segment_radius[:, None]
        + (layer_idx.to(diam.dtype)[None, :] + 0.5) * layer_thickness
    )

    # Ignore padded layer positions.
    valid = layer_idx[None, :] < n_layers[:, None]

    reciprocal_sum = (
        layer_radii.reciprocal()
        .masked_fill(~valid, 0)
        .sum(dim=1)
    )

    hm = n_layers.to(diam.dtype) / reciprocal_sum

    factor = (
        g_ratio
        / (1 - g_ratio)
        * (layer_thickness / segment_radius.square())
        * hm
    )

    myelin_g_pas = myelin_membrane_g_pas * factor
    axon_g_pas = diam.new_tensor(3e-5)
    g[myelin_idx] = (myelin_g_pas * axon_g_pas) / (myelin_g_pas + axon_g_pas)

    return g[None, :]