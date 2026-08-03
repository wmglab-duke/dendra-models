"""Shared helpers for translated cortical-cell biophysics."""

from dendra.models.utils import distance


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
