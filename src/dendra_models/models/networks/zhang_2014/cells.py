"""Dendra morphology/population builders for Zhang et al. 2014 / ModelDB 168414.

This module constructs the three morphology-bearing cell classes in the Zhang
spinal dorsal horn model as Dendra ``Tree`` populations by first creating the
same stylized NEURON sections and then calling ``dn.Tree.from_NEURON``.

The complete vector-defined network is assembled by ``network.py``; this
module remains focused on reusable morphology and mechanism construction.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Optional, Tuple, Literal


ZHANG_CELSIUS = 36.0
"""Simulation temperature used by the original Zhang Wind-Up model (degC)."""

ZHANG_SECTION_GROUPS = ("soma", "dend", "hillock", "axon")
"""Canonical morphology tokens exposed as labels on every Zhang cell."""

ZHANG_SYNAPSE_LAYOUTS = ("banked", "independent")
"""Supported Zhang synapse insertion layouts."""



@dataclass(frozen=True)
class SectionSpec:
    """Geometry/passive-section metadata used to instantiate a NEURON section."""

    name: str
    L: float
    diam: float
    nseg: int
    Ra: float
    cm: float = 1.0
    diam_distal: Optional[float] = None


def _require_neuron():
    os.environ.setdefault("NEURON_MODULE_OPTIONS", "-nogui")
    try:
        from neuron import h  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on optional NEURON install
        raise ImportError(
            "The Zhang 2014 Dendra morphology builders require NEURON because "
            "they intentionally route morphologies through dn.Tree.from_NEURON."
        ) from exc
    return h


def _set_tapered_diam(sec, proximal: float, distal: float) -> None:
    """Approximate HOC's ``diam(0:1)=proximal:distal`` over NEURON segments."""

    for seg in sec:
        seg.diam = proximal + (distal - proximal) * float(seg.x)


class StylizedHocCell:
    """Small holder for the NEURON sections used before Dendra conversion."""

    def __init__(self, template_name: str, sections: Iterable[SectionSpec]):
        h = _require_neuron()
        self.template_name = template_name
        self.sections: Dict[str, object] = {}

        for spec in sections:
            sec = h.Section(name=f"{template_name}_{spec.name}")
            sec.L = float(spec.L)
            sec.nseg = int(spec.nseg)
            sec.Ra = float(spec.Ra)
            sec.cm = float(spec.cm)
            sec.diam = float(spec.diam)
            if spec.diam_distal is not None:
                _set_tapered_diam(sec, float(spec.diam), float(spec.diam_distal))
            self.sections[spec.name] = sec
            setattr(self, spec.name, sec)

        # The biological direction is soma -> hillock -> axon, plus soma -> dend.
        # NEURON's root choice is then simply ``self.soma`` for dn.Tree.from_NEURON.
        self.hillock.connect(self.soma(1.0), 0.0)
        self.axon.connect(self.hillock(1.0), 0.0)
        self.dend.connect(self.soma(0.0), 0.0)


def make_melnick_sg_neuron(template_name: str = "MelnickSG") -> StylizedHocCell:
    """Create the NEURON stylized sections for one Melnick SG interneuron."""

    return StylizedHocCell(
        template_name,
        [
            SectionSpec("soma", L=10.0, diam=10.0, nseg=10, Ra=80.0),
            SectionSpec(
                "hillock", L=30.0, diam=1.0, diam_distal=0.5, nseg=30, Ra=80.0
            ),
            SectionSpec("axon", L=0.001, diam=0.001, nseg=50, Ra=80.0),
            SectionSpec("dend", L=1371.0, diam=1.4, nseg=50, Ra=80.0),
        ],
    )


def make_aguiar_in_neuron(template_name: str = "AguiarIN") -> StylizedHocCell:
    """Create the NEURON stylized sections for one Aguiar excitatory interneuron."""

    return StylizedHocCell(
        template_name,
        [
            SectionSpec("soma", L=20.0, diam=20.0, nseg=3, Ra=150.0),
            SectionSpec("dend", L=400.0, diam=3.0, nseg=5, Ra=150.0),
            SectionSpec(
                "hillock", L=9.0, diam=2.0, diam_distal=1.0, nseg=3, Ra=150.0
            ),
            SectionSpec("axon", L=1000.0, diam=1.0, nseg=5, Ra=150.0),
        ],
    )


def make_aguiar_wdr_neuron(template_name: str = "AguiarWDR") -> StylizedHocCell:
    """Create the NEURON stylized sections for one Aguiar WDR projection neuron."""

    return StylizedHocCell(
        template_name,
        [
            SectionSpec("soma", L=20.0, diam=20.0, nseg=3, Ra=150.0),
            SectionSpec("dend", L=350.0, diam=2.5, nseg=5, Ra=150.0),
            SectionSpec(
                "hillock", L=9.0, diam=2.0, diam_distal=1.0, nseg=3, Ra=150.0
            ),
            SectionSpec("axon", L=1000.0, diam=1.0, nseg=5, Ra=150.0),
        ],
    )


def repair_endpoint_branchpoint_resistances(
    tree,
    *,
    invalid_resistance_ohm: float = 1.0e30,
):
    """Repair infinite root-endpoint axial edges produced by ``from_NEURON``.

    The uploaded Dendra importer represents a child section attached at a parent
    section endpoint by inserting a zero-length branchpoint.  At a root endpoint
    such as ``soma(0)``, NEURON's endpoint ``ri()`` is effectively infinite, so
    the importer can assign the parent-compartment-to-branchpoint edge an axial
    resistance near ``1e36`` ohm.  That electrically disconnects the child
    section even though the original NEURON morphology is connected.

    For the stylized Zhang cells, the parent segment adjacent to this endpoint is
    cylindrical.  Recompute its center-to-end resistance from its axial
    resistivity, diameter, and the edge's center-to-end length.  The repair must
    run before ``tree.build()`` so the tree integrator sees the corrected graph.

    Returns
    -------
    list[dict]
        Metadata for every repaired edge.  It is also stored on
        ``tree.zhang2014_branchpoint_repairs`` for diagnostics.
    """

    graphs = tree.graph if isinstance(tree.graph, (list, tuple)) else [tree.graph]
    repairs = []

    for graph_index, graph in enumerate(graphs):
        for pre, post, edge_data in list(graph.edges(data=True)):
            resistance = float(edge_data.get("R_ohm", math.inf))
            if math.isfinite(resistance) and resistance < float(invalid_resistance_ohm):
                continue

            post_name = str(graph.nodes[post].get("name", "")).lower()
            if "branchpoint" not in post_name:
                continue

            parent = graph.nodes[pre]
            length_um = float(edge_data.get("L", 0.0))
            diameter_um = float(parent.get("diam", 0.0))
            ra_ohm_cm = float(parent.get("Ra", 0.0))
            if length_um <= 0.0 or diameter_um <= 0.0 or ra_ohm_cm <= 0.0:
                raise ValueError(
                    "Cannot repair endpoint branchpoint resistance: missing or "
                    f"non-positive geometry on edge {pre}->{post}."
                )

            length_cm = length_um * 1.0e-4
            radius_cm = 0.5 * diameter_um * 1.0e-4
            corrected = ra_ohm_cm * length_cm / (math.pi * radius_cm**2)
            edge_data["R_ohm"] = corrected

            repairs.append(
                {
                    "graph_index": graph_index,
                    "pre": pre,
                    "post": post,
                    "pre_name": str(parent.get("name", pre)),
                    "post_name": str(graph.nodes[post].get("name", post)),
                    "old_R_ohm": resistance,
                    "new_R_ohm": corrected,
                    "length_um": length_um,
                    "diameter_um": diameter_um,
                    "Ra_ohm_cm": ra_ohm_cm,
                }
            )

    tree.zhang2014_branchpoint_repairs = repairs
    return repairs


def _tree_from_hoc_cell(
    hoc_cell: StylizedHocCell,
    *,
    N: int,
    repair_endpoint_branchpoints: bool = True,
    **tree_kwargs,
):
    import dendra as dn

    # Dendra's generic default is 37 degC, whereas the Zhang Wind-Up scripts
    # explicitly run at 36 degC.  Set the model-specific default at construction
    # time so Q10 caches are populated with the correct temperature.
    tree_kwargs.setdefault("celsius", ZHANG_CELSIUS)

    tree = dn.Tree.from_NEURON(hoc_cell.soma, N=N, **tree_kwargs)
    if repair_endpoint_branchpoints:
        repair_endpoint_branchpoint_resistances(tree)
    else:
        tree.zhang2014_branchpoint_repairs = []

    # Use Dendra's public token-aware name API directly.  This intentionally
    # mirrors user code: ``tree.find("soma")`` and ``tree.slice("soma")`` work
    # regardless of the NEURON/template prefix in names such as
    # ``MelnickSG_soma(0.45)``.
    for group in ZHANG_SECTION_GROUPS:
        section = tree.slice(group)
        if section.is_empty:
            raise LookupError(
                f"Could not find Zhang morphology group {group!r}. "
                "This package requires Dendra's token-aware Population.find API."
            )
        section.label(group)
    return tree


def _pas_class():
    from dendra.models.mod import pas

    return pas


def _synapse_mechanism_name(syn_cls) -> str:
    return str(getattr(syn_cls, "_name", None) or syn_cls.__name__)


def _synapse_target_size(target) -> int:
    shape = tuple(getattr(target, "shape", ()))
    if not shape:
        return 1
    total = 1
    for dim in shape:
        total *= int(dim)
    return int(total)


def _ensure_synapse_metadata(tree):
    if not hasattr(tree, "zhang2014_synapses"):
        tree.zhang2014_synapses = {}
    if not hasattr(tree, "zhang2014_synapse_banks"):
        tree.zhang2014_synapse_banks = {}
    if not hasattr(tree, "_zhang2014_synapse_cursors"):
        tree._zhang2014_synapse_cursors = {}
    return tree.zhang2014_synapses, tree.zhang2014_synapse_banks, tree._zhang2014_synapse_cursors


def _insert_synapse_bank(
    tree,
    syn_cls,
    count: int,
    *,
    prefix: str,
    section: str = "dend",
    loc: float = 0.5,
    aliases: Optional[MutableMapping[str, List[str]]] = None,
    layout: Literal["banked", "independent"] = "banked",
    **params,
) -> List[str]:
    """Insert a Zhang HOC synlist bank.

    ``layout="banked"`` creates one vectorized Dendra mechanism per receptor
    class, with ``count`` independent colocated state slots at the selected
    morphology location.  This preserves independent A/B/P/Use state while
    allowing all contacts of the same receptor class to share one mechanism
    object and therefore one grouped NetCon per source/target/receptor pathway.

    ``layout="independent"`` retains the earlier literal representation: every
    HOC point process is a separately renamed Dendra mechanism object.
    """

    if layout not in ZHANG_SYNAPSE_LAYOUTS:
        raise ValueError(
            f"synapse_layout must be one of {ZHANG_SYNAPSE_LAYOUTS}; got {layout!r}."
        )

    synapses, bank_info, cursors = _ensure_synapse_metadata(tree)

    if count <= 0:
        if aliases is not None:
            aliases[prefix] = []
        synapses[prefix] = []
        bank_info[prefix] = {
            "layout": layout,
            "mechanism": _synapse_mechanism_name(syn_cls),
            "aliases": [],
            "local_indices": [],
            "section": section,
            "loc": float(loc),
            "count": 0,
        }
        return []

    target = tree.slice(section, loc=loc)
    inserted = [f"{prefix}_{idx:03d}" for idx in range(int(count))]

    if layout == "banked":
        mechanism_name = _synapse_mechanism_name(syn_cls)
        target_size = _synapse_target_size(target)
        if target_size != 1:
            raise ValueError(
                "The exact Zhang Wind-Up realization expects one target point per "
                f"population. Bank {prefix!r} selected {target_size} targets; "
                "multi-cell scaling should allocate per-cell banks explicitly."
            )
        start = int(cursors.get(mechanism_name, 0))
        local_indices = list(range(start, start + int(count)))
        target.insert(
            syn_cls,
            alias=prefix,
            preserve_duplicate_indices=True,
            copies=int(count),
            **params,
        )
        cursors[mechanism_name] = start + int(count)
        bank_info[prefix] = {
            "layout": "banked",
            "mechanism": mechanism_name,
            "aliases": inserted,
            "local_indices": local_indices,
            "section": section,
            "loc": float(loc),
            "count": int(count),
            "params": dict(params),
        }
    else:
        local_indices = []
        for idx, alias in enumerate(inserted):
            target.insert(syn_cls.rename(alias), **params)
            local_indices.append(0)
        bank_info[prefix] = {
            "layout": "independent",
            "mechanism": None,
            "aliases": inserted,
            "local_indices": local_indices,
            "section": section,
            "loc": float(loc),
            "count": int(count),
            "params": dict(params),
        }

    synapses[prefix] = inserted
    if aliases is not None:
        aliases[prefix] = inserted
    return inserted


def insert_melnick_sg_mechanisms(tree, *, insert_synapses: bool = True, sg_ampa_counts=(0, 0), synapse_layout: Literal["banked", "independent"] = "banked"):
    """Insert Melnick SG intrinsic channels and optional SG AMPA synapse banks."""

    from . import mechanisms as mech

    pas = _pas_class()

    soma = tree.slice("soma")
    hillock = tree.slice("hillock")
    axon = tree.slice("axon")
    dend = tree.slice("dend")

    # Intrinsic currents from MelnickSG in CellTemplates.hoc.  Potassium-current
    # reversal values are carried per alias because Dendra alias overrides require
    # RANGE parameters.
    for target, suffix, gnabar, kdrigbar, g_pas in [
        (soma, "soma", 0.008, 0.0043, 1.1e-5),
        (hillock, "hillock", 3.45, 0.076, 1.1e-5),
        (axon, "axon", 0.0, 0.0, 0.0),
    ]:
        target.insert(mech.B_Na, alias=f"B_Na_{suffix}", gnabar=gnabar, ena=60.0)
        target.insert(mech.B_A, alias=f"B_A_{suffix}", gkbar=0.0, ek=-84.0)
        target.insert(mech.B_DR, alias=f"B_DR_{suffix}", gkbar=0.0, ek=-84.0)
        target.insert(mech.KDR, alias=f"KDR_{suffix}", gkbar=0.0, ek=-84.0)
        target.insert(mech.KDRI, alias=f"KDRI_{suffix}", gkbar=kdrigbar, ek=-84.0)
        target.insert(pas, alias=f"pas_{suffix}", g=g_pas, e=-70.0)

    dend.insert(mech.SS, alias="SS_dend", gnabar=0.0, ena=60.0)
    dend.insert(mech.B_DR, alias="B_DR_dend", gkbar=0.0, ek=-84.0)
    dend.insert(mech.KDR, alias="KDR_dend", gkbar=0.0, ek=-84.0)
    dend.insert(mech.KDRI, alias="KDRI_dend", gkbar=0.034, ek=-84.0)
    dend.insert(pas, alias="pas_dend", g=1.1e-5, e=-70.0)

    syn_aliases: Dict[str, List[str]] = {}
    tree.zhang2014_synapses = syn_aliases
    tree.zhang2014_synapse_banks = {}
    tree._zhang2014_synapse_cursors = {}
    if insert_synapses:
        # The SG template reads two AMPA counts from SG_SynapseNumber.dat.
        # Keep this explicit instead of silently hard-coding a model-instance file.
        n_ampa_1, n_ampa_2 = sg_ampa_counts
        _insert_synapse_bank(
            tree,
            mech.AMPA_DynSyn,
            int(n_ampa_1),
            prefix="sg_ampa_primary",
            aliases=syn_aliases, layout=synapse_layout,
            tau_rise=0.1,
            tau_decay=5.0,
            e=0.0,
        )
        _insert_synapse_bank(
            tree,
            mech.AMPA_DynSyn,
            int(n_ampa_2),
            prefix="sg_ampa_secondary",
            aliases=syn_aliases, layout=synapse_layout,
            tau_rise=0.1,
            tau_decay=5.0,
            e=0.0,
        )
    tree.zhang2014_synapses = syn_aliases
    return tree


def insert_aguiar_in_mechanisms(tree, *, insert_synapses: bool = True, synapse_layout: Literal["banked", "independent"] = "banked"):
    """Insert AguiarIN intrinsic currents and optional synapse aliases."""

    from . import mechanisms as mech

    pas = _pas_class()

    soma = tree.slice("soma")
    dend = tree.slice("dend")
    hillock = tree.slice("hillock")
    axon = tree.slice("axon")

    soma.insert(mech.HH2, alias="HH2_soma", gnabar=0.0, gkbar=0.0043 / 4.0, vtraub=-55.0, ek=-70.0)
    soma.insert(mech.CaIntraCellDyn, alias="CaIntraCellDyn_soma", depth=0.1, cai_tau=1.0, cai_inf=50e-6)
    soma.insert(mech.iKCa, alias="iKCa_soma", gbar=0.002, ek=-70.0)
    soma.insert(pas, alias="pas_soma", g=4.2e-5, e=-65.0)

    dend.insert(mech.HH2, alias="HH2_dend", gnabar=0.0, gkbar=0.036, vtraub=-55.0, ek=-70.0)
    dend.insert(mech.CaIntraCellDyn, alias="CaIntraCellDyn_dend", depth=0.1, cai_tau=2.0, cai_inf=50e-6)
    dend.insert(mech.iKCa, alias="iKCa_dend", gbar=0.002, ek=-70.0)
    dend.insert(pas, alias="pas_dend", g=4.2e-5, e=-65.0)

    hillock.insert(mech.HH2, alias="HH2_hillock", gnabar=3.45, gkbar=0.076, vtraub=-55.0)
    hillock.insert(pas, alias="pas_hillock", g=4.2e-5, e=-65.0)

    axon.insert(mech.HH2, alias="HH2_axon", gnabar=0.0, gkbar=0.0, vtraub=-55.0)
    axon.insert(pas, alias="pas_axon", g=4.2e-5, e=-65.0)

    syn_aliases: Dict[str, List[str]] = {}
    tree.zhang2014_synapses = syn_aliases
    tree.zhang2014_synapse_banks = {}
    tree._zhang2014_synapse_cursors = {}
    if insert_synapses:
        _insert_synapse_bank(tree, mech.AMPA_DynSyn, 30, prefix="ex_ampa", aliases=syn_aliases, layout=synapse_layout, tau_rise=0.1, tau_decay=5.0, e=0.0)
        _insert_synapse_bank(tree, mech.NMDA_DynSyn, 30, prefix="ex_nmda", aliases=syn_aliases, layout=synapse_layout, tau_rise=2.0, tau_decay=100.0, e=0.0)
        _insert_synapse_bank(tree, mech.NK1_DynSyn, 30, prefix="ex_nk1", aliases=syn_aliases, layout=synapse_layout, tau_rise=100.0, tau_decay=3000.0, e=0.0)
        _insert_synapse_bank(tree, mech.GABAa_DynSyn, 2, prefix="ex_gaba_local", aliases=syn_aliases, layout=synapse_layout, tau_rise=0.1, tau_decay=20.0, e=-70.0)
        _insert_synapse_bank(tree, mech.GABAa_DynSyn, 2, prefix="ex_gaba_scs", aliases=syn_aliases, layout=synapse_layout, tau_rise=0.1, tau_decay=20.0, e=-70.0)
    tree.zhang2014_synapses = syn_aliases
    return tree


def insert_aguiar_wdr_mechanisms(tree, *, insert_synapses: bool = True, cascale: float = 1.0, synapse_layout: Literal["banked", "independent"] = "banked"):
    """Insert AguiarWDR intrinsic currents and optional synapse aliases."""

    from . import mechanisms as mech

    pas = _pas_class()
    cascale = float(cascale)

    soma = tree.slice("soma")
    dend = tree.slice("dend")
    hillock = tree.slice("hillock")
    axon = tree.slice("axon")

    soma.insert(mech.HH2, alias="HH2_soma", gnabar=0.0, gkbar=0.0043 * 6.0 / 24.0, vtraub=-55.0, ek=-70.0)
    soma.insert(mech.CaIntraCellDyn, alias="CaIntraCellDyn_soma", depth=0.1, cai_tau=1.0, cai_inf=50e-6)
    soma.insert(mech.iCaL, alias="iCaL_soma", pcabar=0.0001 * cascale)
    soma.insert(mech.iCaAN, alias="iCaAN_soma", gbar=0.0)
    soma.insert(mech.iKCa, alias="iKCa_soma", gbar=0.0001 * cascale, ek=-70.0)
    soma.insert(mech.iNaP, alias="iNaP_soma", gnabar=0.0001 * cascale)
    soma.insert(pas, alias="pas_soma", g=4.2e-5, e=-65.0)

    dend.insert(mech.CaIntraCellDyn, alias="CaIntraCellDyn_dend", depth=0.1, cai_tau=2.0, cai_inf=50e-6)
    dend.insert(mech.iCaL, alias="iCaL_dend", pcabar=0.00003 * cascale)
    dend.insert(mech.iCaAN, alias="iCaAN_dend", gbar=0.00007 * cascale * 1.3)
    dend.insert(mech.iKCa, alias="iKCa_dend", gbar=0.001 * cascale, ek=-70.0)
    dend.insert(mech.HH2, alias="HH2_dend", gnabar=0.0, gkbar=0.036, vtraub=-55.0, ek=-70.0)
    dend.insert(pas, alias="pas_dend", g=4.2e-5, e=-65.0)

    hillock.insert(mech.HH2, alias="HH2_hillock", gnabar=3.45, gkbar=0.076, vtraub=-55.0)
    hillock.insert(mech.B_A, alias="B_A_hillock", gkbar=0.0)
    hillock.insert(pas, alias="pas_hillock", g=4.2e-5, e=-65.0)

    axon.insert(mech.HH2, alias="HH2_axon", gnabar=0.0, gkbar=0.0, vtraub=-55.0)
    axon.insert(pas, alias="pas_axon", g=0.0, e=-65.0)

    syn_aliases: Dict[str, List[str]] = {}
    tree.zhang2014_synapses = syn_aliases
    tree.zhang2014_synapse_banks = {}
    tree._zhang2014_synapse_cursors = {}
    if insert_synapses:
        _insert_synapse_bank(tree, mech.AMPA_DynSyn, 15, prefix="wdr_abeta_ampa", aliases=syn_aliases, layout=synapse_layout, tau_rise=0.1, tau_decay=5.0, e=0.0)
        _insert_synapse_bank(tree, mech.NMDA_DynSyn, 15, prefix="wdr_abeta_nmda", aliases=syn_aliases, layout=synapse_layout, tau_rise=2.0, tau_decay=100.0, e=0.0)
        _insert_synapse_bank(tree, mech.AMPA_DynSyn, 15, prefix="wdr_c_ampa", aliases=syn_aliases, layout=synapse_layout, tau_rise=0.1, tau_decay=5.0, e=0.0)
        _insert_synapse_bank(tree, mech.NMDA_DynSyn, 15, prefix="wdr_c_nmda", aliases=syn_aliases, layout=synapse_layout, tau_rise=2.0, tau_decay=100.0, e=0.0)
        _insert_synapse_bank(tree, mech.NK1_DynSyn, 30, prefix="wdr_nk1", aliases=syn_aliases, layout=synapse_layout, tau_rise=200.0, tau_decay=3000.0, e=0.0)
        _insert_synapse_bank(tree, mech.AMPA_DynSyn, 30, prefix="wdr_ex_ampa", aliases=syn_aliases, layout=synapse_layout, tau_rise=0.1, tau_decay=5.0, e=0.0)
        _insert_synapse_bank(tree, mech.NMDA_DynSyn, 30, prefix="wdr_ex_nmda", aliases=syn_aliases, layout=synapse_layout, tau_rise=2.0, tau_decay=100.0, e=0.0)
        _insert_synapse_bank(tree, mech.Glycine_DynSyn, 1, prefix="wdr_glycine", aliases=syn_aliases, layout=synapse_layout, tau_rise=0.1, tau_decay=10.0, e=-70.0)
        _insert_synapse_bank(tree, mech.GABAa_DynSyn, 1, prefix="wdr_gaba_local", aliases=syn_aliases, layout=synapse_layout, tau_rise=0.1, tau_decay=20.0, e=-70.0)
        _insert_synapse_bank(tree, mech.GABAa_DynSyn, 1, prefix="wdr_gaba_surround", aliases=syn_aliases, layout=synapse_layout, tau_rise=0.1, tau_decay=20.0, e=-70.0)
        _insert_synapse_bank(tree, mech.GABAa_DynSyn, 1, prefix="wdr_gaba_scs", aliases=syn_aliases, layout=synapse_layout, tau_rise=0.1, tau_decay=20.0, e=-70.0)
    tree.zhang2014_synapses = syn_aliases
    return tree


def melnick_sg(
    *,
    N: int = 1,
    insert_mechanisms: bool = True,
    insert_synapses: bool = True,
    synapse_layout: Literal["banked", "independent"] = "banked",
    sg_ampa_counts: Tuple[int, int] = (0, 0),
    template_name: str = "MelnickSG",
    **tree_kwargs,
):
    """Return a Dendra ``Tree`` population for Melnick SG inhibitory cells."""

    tree = _tree_from_hoc_cell(make_melnick_sg_neuron(template_name), N=N, **tree_kwargs)
    if insert_mechanisms:
        insert_melnick_sg_mechanisms(
            tree, insert_synapses=insert_synapses, sg_ampa_counts=sg_ampa_counts, synapse_layout=synapse_layout
        )
    return tree


def aguiar_in(
    *,
    N: int = 1,
    insert_mechanisms: bool = True,
    insert_synapses: bool = True,
    synapse_layout: Literal["banked", "independent"] = "banked",
    template_name: str = "AguiarIN",
    **tree_kwargs,
):
    """Return a Dendra ``Tree`` population for Aguiar excitatory interneurons."""

    tree = _tree_from_hoc_cell(make_aguiar_in_neuron(template_name), N=N, **tree_kwargs)
    if insert_mechanisms:
        insert_aguiar_in_mechanisms(tree, insert_synapses=insert_synapses, synapse_layout=synapse_layout)
    return tree


def aguiar_wdr(
    *,
    N: int = 1,
    cascale: float = 1.0,
    insert_mechanisms: bool = True,
    insert_synapses: bool = True,
    synapse_layout: Literal["banked", "independent"] = "banked",
    template_name: str = "AguiarWDR",
    **tree_kwargs,
):
    """Return a Dendra ``Tree`` population for Aguiar WDR projection neurons."""

    tree = _tree_from_hoc_cell(make_aguiar_wdr_neuron(template_name), N=N, **tree_kwargs)
    if insert_mechanisms:
        insert_aguiar_wdr_mechanisms(
            tree, insert_synapses=insert_synapses, cascale=cascale, synapse_layout=synapse_layout
        )
    return tree


def build_cell_populations(
    *,
    n_sg: int = 1,
    n_sg_scs: int = 1,
    n_ex: int = 1,
    n_wdr: int = 1,
    sg_ampa_counts: Tuple[int, int] = (0, 0),
    sg_scs_ampa_counts: Tuple[int, int] = (0, 0),
    cascale: float = 1.0,
    insert_synapses: bool = True,
    synapse_layout: Literal["banked", "independent"] = "banked",
    **tree_kwargs,
):
    """Build the morphology-bearing Dendra populations used in the network.

    Returns a dictionary with keys matching the original model's population roles:
    ``SG`` and ``SGSCS`` are Melnick inhibitory interneuron populations, ``EX`` is
    the Aguiar excitatory interneuron population, and ``T_Cell`` is the Aguiar
    WDR projection-cell population.
    """

    pops = {
        "SG": melnick_sg(
            N=n_sg,
            insert_synapses=insert_synapses,
            synapse_layout=synapse_layout,
            sg_ampa_counts=sg_ampa_counts,
            template_name="MelnickSG",
            **tree_kwargs,
        ),
        "SGSCS": melnick_sg(
            N=n_sg_scs,
            insert_synapses=insert_synapses,
            synapse_layout=synapse_layout,
            sg_ampa_counts=sg_scs_ampa_counts,
            template_name="MelnickSGSCS",
            **tree_kwargs,
        ),
        "EX": aguiar_in(
            N=n_ex,
            insert_synapses=insert_synapses,
            synapse_layout=synapse_layout,
            template_name="AguiarIN",
            **tree_kwargs,
        ),
        "T_Cell": aguiar_wdr(
            N=n_wdr,
            cascale=cascale,
            insert_synapses=insert_synapses,
            synapse_layout=synapse_layout,
            template_name="AguiarWDR",
            **tree_kwargs,
        ),
    }
    return pops


__all__ = [
    "ZHANG_CELSIUS",
    "ZHANG_SECTION_GROUPS",
    "ZHANG_SYNAPSE_LAYOUTS",
    "SectionSpec",
    "repair_endpoint_branchpoint_resistances",
    "StylizedHocCell",
    "make_melnick_sg_neuron",
    "make_aguiar_in_neuron",
    "make_aguiar_wdr_neuron",
    "insert_melnick_sg_mechanisms",
    "insert_aguiar_in_mechanisms",
    "insert_aguiar_wdr_mechanisms",
    "melnick_sg",
    "aguiar_in",
    "aguiar_wdr",
    "build_cell_populations",
]
