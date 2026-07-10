"""Regression tests for the Zhang port's use of Dendra's core find API."""

from __future__ import annotations

import pytest

pytest.importorskip("neuron")

from .. import (  # noqa: E402
    ZHANG_SECTION_GROUPS,
    aguiar_in,
    aguiar_wdr,
    insert_aguiar_wdr_mechanisms,
    make_aguiar_wdr_neuron,
    melnick_sg,
)


@pytest.mark.parametrize(
    ("builder", "kwargs", "shape", "counts"),
    [
        (
            melnick_sg,
            {"sg_ampa_counts": (0, 0)},
            (1, 140),
            {"soma": 10, "dend": 50, "hillock": 30, "axon": 50},
        ),
        (
            aguiar_in,
            {},
            (1, 16),
            {"soma": 3, "dend": 5, "hillock": 3, "axon": 5},
        ),
        (
            aguiar_wdr,
            {},
            (1, 16),
            {"soma": 3, "dend": 5, "hillock": 3, "axon": 5},
        ),
    ],
)
def test_builders_expose_core_find_groups(builder, kwargs, shape, counts):
    cell = builder(
        N=1,
        insert_mechanisms=False,
        insert_synapses=False,
        template_name="Network_Zhang_Test_Cell",
        **kwargs,
    )

    assert tuple(cell.shape) == shape
    assert tuple(ZHANG_SECTION_GROUPS) == ("soma", "dend", "hillock", "axon")

    for group, count in counts.items():
        assert len(cell.find(group, as_list=True)) == count
        assert tuple(cell.slice(group).shape) == (1, count)
        assert tuple(getattr(cell, group).shape) == (1, count)

    # The corrected importer should not require the legacy Zhang repair.
    assert cell.zhang2014_branchpoint_repairs == []


def test_public_mechanism_inserter_does_not_require_preexisting_labels():
    """Mechanism insertion should resolve sections through ``tree.slice``."""
    import dendra as dn

    hoc_cell = make_aguiar_wdr_neuron("Unlabelled_Custom_WDR")
    tree = dn.Tree.from_NEURON(hoc_cell.soma, N=1, celsius=36.0)

    # Remove any convenience labels supplied by the importer. The public
    # mechanism inserter must still work solely through core name lookup.
    tree.clear_labels()
    assert not hasattr(tree, "soma")
    insert_aguiar_wdr_mechanisms(tree, insert_synapses=False)

    assert len(tree.find("soma", as_list=True)) == 3
    assert len(tree.find("dend", as_list=True)) == 5
