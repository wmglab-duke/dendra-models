from __future__ import annotations

import gc
import importlib.util
from dataclasses import replace

import numpy as np
import pytest
import torch

from .. import (
    WINDUP_DT_MS,
    WINDUP_SPIKE_DETECTOR_CONSUMPTION_LAG_STEPS,
    WINDUP_SPIKE_PRE_VAR,
    WINDUP_SPIKE_THRESHOLD_MV,
    WindUpDataError,
    assemble_windup_network,
    attach_windup_afferents,
    build_windup_network,
    load_windup_data,
    schedule_windup_afferents,
    synapse_aliases,
)


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("neuron") is None,
    reason="The Zhang morphology builders intentionally require NEURON.",
)


@pytest.fixture(scope="module")
def windup_data():
    return load_windup_data(require_shared_afferent_schedules=True)


@pytest.fixture(scope="module")
def cellular_network_uninitialized():
    return build_windup_network(dtype=torch.float64)


@pytest.fixture(scope="module")
def full_network_uninitialized(cellular_network_uninitialized):
    # Reuse the same biological cells to keep the regression suite's memory
    # footprint bounded. This also exercises the primary user workflow:
    # construct/build the cellular circuit first, then attach afferents later.
    net = cellular_network_uninitialized
    if not net.built:
        net.build(WINDUP_DT_MS)
    attach_windup_afferents(net, schedule=True)
    return net


def test_packaged_vectors_and_spike_schedules(windup_data):
    data = windup_data
    assert data.n_connections == 291
    assert data.n_scheduled_connections == 225
    assert data.n_afferent_sources == 75
    assert data.tstop_ms == pytest.approx(21_000.0)
    assert data.can_share_afferent_sources

    unique_counts, multiplicities = np.unique(
        data.connection_spike_counts, return_counts=True
    )
    assert dict(zip(unique_counts.tolist(), multiplicities.tolist())) == {1: 15, 20: 210}
    assert sum(map(len, data.connection_spike_times)) == 4215
    assert sum(map(len, data.afferent_spike_times)) == 1215

    target_ids, target_counts = np.unique(data.to_cell, return_counts=True)
    assert dict(zip(target_ids.tolist(), target_counts.tolist())) == {
        75: 30,
        76: 15,
        77: 92,
        78: 154,
    }
    assert np.all(data.delay_ms == 1.0)
    assert np.all(data.threshold_mV == -30.0)


def test_default_builder_returns_cellular_network_only(cellular_network_uninitialized):
    net = cellular_network_uninitialized
    assert net.netstim is None
    assert not net.zhang2014.afferents_attached
    assert net.zhang2014.afferent_mode is None
    assert len(net.zhang2014.cellular_connections) == 66
    assert len(net.zhang2014.afferent_connections) == 0
    assert len(net.zhang2014.connections) == 66
    assert [c.index for c in net.zhang2014.connections] == list(range(225, 291))
    assert all(not c.is_external for c in net.zhang2014.connections)
    assert all(c.pre_var == WINDUP_SPIKE_PRE_VAR for c in net.zhang2014.connections)
    assert all(c.runtime_threshold_mV is None for c in net.zhang2014.connections)
    assert len(net.synapse_spec) == 5
    assert sum(len(specs) for specs in net.synapse_spec.values()) == 63
    assert sum(c.runtime_connected for c in net.zhang2014.connections) == 63

    # SG, SGSCS, and EX are the only biological sources in this realization.
    # Each computes its soma(1), -30 mV crossing once and shares that event
    # variable across all outgoing receptor-specific NetCons.
    assert net.zhang2014.spike_source_populations == ("SG", "SGSCS", "EX")
    assert net.zhang2014.spike_detector_alias == "spikedetect"
    assert net.zhang2014.spike_pre_var == WINDUP_SPIKE_PRE_VAR
    assert net.zhang2014.spike_threshold_mV == WINDUP_SPIKE_THRESHOLD_MV
    assert (
        net.zhang2014.spike_detector_consumption_lag_steps
        == WINDUP_SPIKE_DETECTOR_CONSUMPTION_LAG_STEPS
        == 1
    )
    assert net.zhang2014.spike_detector_consumption_lag_ms == pytest.approx(
        WINDUP_DT_MS
    )
    for name in net.zhang2014.spike_source_populations:
        detector = getattr(getattr(net, name).mech, "spikedetect")
        assert tuple(detector.shape_f) == (1, 1)
        assert detector.threshold.item() == WINDUP_SPIKE_THRESHOLD_MV
    assert "spikedetect" not in net.T_Cell.mech.mechanisms

    # Exact biological contact-level topology from rows 225:291.
    pair_counts = {}
    for connection in net.zhang2014.cellular_connections:
        pair = (connection.source_population, connection.target_population)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    assert pair_counts == {
        ("SG", "EX"): 1,
        ("SG", "T_Cell"): 2,
        ("SGSCS", "EX"): 1,
        ("SGSCS", "T_Cell"): 2,
        ("EX", "T_Cell"): 60,
    }


def test_explicit_schedule_without_afferent_layer_is_rejected():
    with pytest.raises(ValueError, match="requires include_afferents=True"):
        build_windup_network(
            dtype=torch.float64,
            include_afferents=False,
            schedule_afferents=True,
        )


def test_shared_detector_contract_rejects_nonuniform_biological_thresholds(
    cellular_network_uninitialized,
    windup_data,
):
    thresholds = windup_data.threshold_mV.copy()
    thresholds[windup_data.n_scheduled_connections] = -29.0
    incompatible = replace(windup_data, threshold_mV=thresholds)

    with pytest.raises(WindUpDataError, match="one biological threshold"):
        assemble_windup_network(
            cellular_network_uninitialized.populations,
            data=incompatible,
        )


def test_synapse_alias_order_matches_hoc_synlist(cellular_network_uninitialized):
    net = cellular_network_uninitialized

    sg = synapse_aliases("SG", net.SG)
    assert len(sg) == 30
    assert sg[0] == "sg_ampa_primary_000"
    assert sg[14] == "sg_ampa_primary_014"
    assert sg[15] == "sg_ampa_secondary_000"
    assert sg[29] == "sg_ampa_secondary_014"

    sgscs = synapse_aliases("SGSCS", net.SGSCS)
    assert len(sgscs) == 15
    assert sgscs[-1] == "sg_ampa_primary_014"

    ex = synapse_aliases("EX", net.EX)
    assert len(ex) == 94
    assert ex[0] == "ex_ampa_000"
    assert ex[30] == "ex_nmda_000"
    assert ex[60] == "ex_nk1_000"
    assert ex[90] == "ex_gaba_local_000"
    assert ex[91] == "ex_gaba_local_001"
    assert ex[92] == "ex_gaba_scs_000"

    wdr = synapse_aliases("T_Cell", net.T_Cell)
    assert len(wdr) == 154
    expected_boundaries = {
        0: "wdr_abeta_ampa_000",
        15: "wdr_abeta_nmda_000",
        30: "wdr_c_ampa_000",
        45: "wdr_c_nmda_000",
        60: "wdr_nk1_000",
        90: "wdr_ex_ampa_000",
        120: "wdr_ex_nmda_000",
        150: "wdr_glycine_000",
        151: "wdr_gaba_local_000",
        152: "wdr_gaba_surround_000",
        153: "wdr_gaba_scs_000",
    }
    for index, alias in expected_boundaries.items():
        assert wdr[index] == alias


def test_each_original_point_process_has_independent_state(cellular_network_uninitialized):
    net = cellular_network_uninitialized
    ampa = net.SG.mech.AMPA_DynSyn
    primary = net.SG.zhang2014_synapse_banks["sg_ampa_primary"]
    assert tuple(ampa.shape_f) == (30,)
    assert primary["local_indices"][:2] == [0, 1]
    ampa.A.zero_()
    ampa.A[0] = 1.0
    assert float(ampa.A[0]) == 1.0
    assert float(ampa.A[1]) == 0.0


def test_public_synapse_slots_address_banked_zhang_targets(cellular_network_uninitialized):
    net = cellular_network_uninitialized
    synapse = net.T_Cell.mech.AMPA_DynSyn
    bank = net.T_Cell.zhang2014_synapse_banks["wdr_ex_ampa"]
    first_slot = bank["local_indices"][0]

    slots = net.synapse_slots(net.T_Cell, synapse, slots=[first_slot])
    assert len(slots) == 1
    assert slots.local_index.tolist() == [first_slot]
    target = net.T_Cell.slice(bank["section"], loc=bank["loc"])
    flat_grid = torch.arange(
        net.T_Cell.v.numel(), device=net.T_Cell.device(), dtype=torch.long
    ).reshape(net.T_Cell.shape)
    expected_flat = flat_grid[target.index].reshape(-1)
    assert expected_flat.numel() == 1
    assert slots.flat_index.tolist() == expected_flat.tolist()

    # Physical compartment targeting is ambiguous for a banked point process;
    # the public slot object is the intended endpoint for NetCon construction.
    with pytest.raises(ValueError, match="multiple local slots"):
        net.connect_one_to_one(
            net.EX.slice("soma", loc=1.0),
            net.T_Cell.slice("dend", loc=0.5),
            synapse,
            threshold=None,
        )


@torch.no_grad()
def test_cellular_netcon_build_omits_static_zero_weights_by_default(
    cellular_network_uninitialized,
):
    net = cellular_network_uninitialized
    net.build(WINDUP_DT_MS)
    net.init_synapses()

    weights = torch.cat(
        [netcon.weight().detach().reshape(-1) for netcon in net.synapses.values()]
    )
    delays = torch.cat(
        [netcon.delay_ms().detach().reshape(-1) for netcon in net.synapses.values()]
    )
    assert weights.numel() == 63
    # Three biological SGSCS projections are explicitly disabled in the
    # packaged Wind-Up realization. Static exact-zero rows remain in metadata
    # but are omitted from runtime NetCons unless weights are trainable.
    assert int((weights == 0.0).sum()) == 0
    assert sum(not c.runtime_connected for c in net.zhang2014.cellular_connections) == 3
    expected_runtime_delay = (
        1.0 - WINDUP_SPIKE_DETECTOR_CONSUMPTION_LAG_STEPS * WINDUP_DT_MS
    )
    assert torch.allclose(
        delays, torch.full_like(delays, expected_runtime_delay)
    )
    for netcon in net.synapses.values():
        assert netcon.pre_var == WINDUP_SPIKE_PRE_VAR
        assert netcon.skip_thresholding
        assert bool(netcon.thresh_is_nan.all())
        assert netcon.has_spiked.numel() == 0


def test_afferents_attach_after_cellular_build_and_invalidate_netcons(
    full_network_uninitialized,
):
    net = full_network_uninitialized
    assert net.netstim.N == 75
    assert not net.built
    assert len(net.zhang2014.cellular_connections) == 66
    assert len(net.zhang2014.afferent_connections) == 225
    assert len(net.zhang2014.connections) == 291
    assert len(net.synapse_spec) == 13
    assert sum(len(specs) for specs in net.synapse_spec.values()) == 273
    assert sum(c.runtime_connected for c in net.zhang2014.connections) == 273
    with pytest.raises(RuntimeError, match="wiring has changed"):
        net.run(WINDUP_DT_MS)


def test_full_network_structure_and_schedule(full_network_uninitialized):
    net = full_network_uninitialized
    assert net.netstim.N == 75
    assert net.zhang2014.afferents_attached
    assert net.zhang2014.afferent_mode == "shared_source"
    assert len(net.zhang2014.cellular_connections) == 66
    assert len(net.zhang2014.afferent_connections) == 225
    assert len(net.zhang2014.connections) == 291
    assert [c.index for c in net.zhang2014.connections] == list(range(291))
    assert len(net.synapse_spec) == 13
    assert sum(len(specs) for specs in net.synapse_spec.values()) == 273
    assert sum(c.runtime_connected for c in net.zhang2014.connections) == 273
    assert sum(len(heap) for heap in net.netstim._sched_heaps) == 1215
    assert net.zhang2014.external_event_semantics == "netcon_event_delivery_time"
    assert net.zhang2014.external_schedule_lead_ms == pytest.approx(WINDUP_DT_MS)
    assert net.netstim._sched_heaps[0][0] == pytest.approx(
        net.zhang2014.data.afferent_spike_times[0][0] - WINDUP_DT_MS
    )

    first = net.zhang2014.connections[0]
    assert first.source_id == 0
    assert first.source_channel == 0
    assert first.target_population == "SG"
    assert first.target_synapse_alias == "sg_ampa_primary_000"
    assert first.vector_delay_ms == pytest.approx(1.0)
    assert first.runtime_delay_ms == pytest.approx(WINDUP_DT_MS)
    assert first.pre_var == "spike"
    assert first.runtime_threshold_mV is None

    last = net.zhang2014.connections[-1]
    assert last.source_population == "SGSCS"
    assert last.target_population == "T_Cell"
    assert last.target_synapse_alias == "wdr_gaba_scs_000"
    assert last.weight == 0.0
    assert last.vector_delay_ms == pytest.approx(1.0)
    assert last.runtime_delay_ms == pytest.approx(1.0 - WINDUP_DT_MS)
    assert last.pre_var == WINDUP_SPIKE_PRE_VAR
    assert last.runtime_threshold_mV is None


@torch.no_grad()
def test_full_netcon_build_preserves_event_delay_semantics(full_network_uninitialized):
    net = full_network_uninitialized
    net.build(WINDUP_DT_MS)
    net.init_synapses()

    weights = torch.cat(
        [netcon.weight().detach().reshape(-1) for netcon in net.synapses.values()]
    )
    delays = torch.cat(
        [netcon.delay_ms().detach().reshape(-1) for netcon in net.synapses.values()]
    )
    assert weights.numel() == 273
    assert int((weights == 0.0).sum()) == 0
    assert sum(not c.runtime_connected for c in net.zhang2014.connections) == 18
    assert int(
        torch.isclose(delays, torch.full_like(delays, WINDUP_DT_MS)).sum()
    ) == 210
    internal_delay = 1.0 - WINDUP_DT_MS
    assert int(
        torch.isclose(delays, torch.full_like(delays, internal_delay)).sum()
    ) == 63
    for name, netcon in net.synapses.items():
        assert netcon.skip_thresholding
        if name.startswith("netstim:"):
            assert netcon.pre_var == "spike"
        else:
            assert netcon.pre_var == WINDUP_SPIKE_PRE_VAR



def test_per_connection_source_mode(cellular_network_uninitialized, windup_data):
    # Reuse the already imported biological cells; this tests the wiring-only
    # API without duplicating hundreds of independent point-process mechanisms.
    net = assemble_windup_network(
        cellular_network_uninitialized.populations,
        data=windup_data,
        include_afferents=True,
        afferent_mode="per_connection",
    )
    assert net.netstim.N == 225
    assert sum(len(heap) for heap in net.netstim._sched_heaps) == 4215
    assert net.zhang2014.connections[0].source_channel == 0
    assert net.zhang2014.connections[224].source_channel == 224
    assert net.zhang2014.connections[225].source_channel is None
    # This network shares the large biological Population modules with the
    # cellular fixture. Release the temporary wrapper before the event tests.
    del net
    gc.collect()

EVENT_TEST_DT_MS = WINDUP_DT_MS


@torch.no_grad()
def test_external_and_biological_event_delivery_paths(full_network_uninitialized):
    """Exercise the three event paths on one repeatedly reset runtime network."""

    net = full_network_uninitialized

    def reset_runtime():
        net.t.zero_()
        for population in net.populations.values():
            population.initialize()
            population.detach()
        net.build(EVENT_TEST_DT_MS)
        net.init_synapses()
        net.netstim.set_dt(EVENT_TEST_DT_MS)
        net.netstim.clear_schedule()
        net.netstim.initialize()
        net.netstim.detach()

    # One shared-source event must fan out to the intended colocated receptor
    # contacts, but not to the adjacent independent AMPA point process.
    reset_runtime()
    net.netstim.schedule(0, [2 * EVENT_TEST_DT_MS])
    net.netstim.initialize()

    sg_ampa = net.SG.mech.AMPA_DynSyn
    wdr_ampa = net.T_Cell.mech.AMPA_DynSyn
    wdr_nmda = net.T_Cell.mech.NMDA_DynSyn
    sg0_slot = net.SG.zhang2014_synapse_banks["sg_ampa_primary"]["local_indices"][0]
    sg1_slot = net.SG.zhang2014_synapse_banks["sg_ampa_primary"]["local_indices"][1]
    wdr_ampa_slot = net.T_Cell.zhang2014_synapse_banks["wdr_abeta_ampa"]["local_indices"][0]
    wdr_nmda_slot = net.T_Cell.zhang2014_synapse_banks["wdr_abeta_nmda"]["local_indices"][0]
    for step in range(6):
        net.t.fill_(step * EVENT_TEST_DT_MS)
        net.netstim(net.t, bptt=False, dt=EVENT_TEST_DT_MS)
        for netcon in net.synapses.values():
            netcon.advance()

    assert int(net.netstim.spike_counts[0]) == 1
    assert float(sg_ampa.A[sg0_slot]) > 0.0
    assert float(wdr_ampa.A[wdr_ampa_slot]) > 0.0
    assert float(wdr_nmda.A[wdr_nmda_slot]) > 0.0
    assert float(sg_ampa.A[sg1_slot]) == 0.0

    # Biological EX->WDR transmission consumes the shared soma detector.
    # NetCons advance before mechanisms: the detector emits on step 1, the
    # NetCon consumes that pulse on step 2, and its 79-step runtime delay lands
    # on the same effective step as the previous direct 80-step threshold path.
    reset_runtime()
    netcon = net.synapses[
        "EX:mech_spikedetect_spikes->T_Cell:AMPA_DynSyn"
    ]
    assert netcon.pre_var == WINDUP_SPIKE_PRE_VAR
    assert netcon.skip_thresholding
    assert netcon.delay_steps.tolist() == [79] * 30

    source_slice = net.EX.slice("soma", loc=1.0)
    grid = torch.arange(
        net.EX.v.numel(), device=net.EX.device(), dtype=torch.long
    ).reshape(net.EX.shape)
    source_index = int(grid[source_slice.index].reshape(-1).item())
    assert "soma" in str(net.EX.names[source_index])
    detector = net.EX.mech.spikedetect

    wdr_ampa = net.T_Cell.mech.AMPA_DynSyn
    target_slot = net.T_Cell.zhang2014_synapse_banks["wdr_ex_ampa"]["local_indices"][0]
    unrelated_slot = net.T_Cell.zhang2014_synapse_banks["wdr_abeta_ampa"]["local_indices"][0]
    net.EX.v.view(-1)[source_index] = -65.0
    detector.initial(detector.get(net.EX.v))

    first_nonzero_step = None
    for step in range(90):
        net.t.fill_(step * EVENT_TEST_DT_MS)
        if step == 1:
            net.EX.v.view(-1)[source_index] = 20.0
        elif step == 3:
            net.EX.v.view(-1)[source_index] = -65.0

        for connection in net.synapses.values():
            connection.advance()
        detector.breakpoint(detector.get(net.EX.v))

        if first_nonzero_step is None and float(wdr_ampa.A[target_slot]) > 0.0:
            first_nonzero_step = step

    assert first_nonzero_step == 81
    assert float(wdr_ampa.A[unrelated_slot]) == 0.0

    # The one-step-early NetStim schedule plus one-step bridge delay must land
    # on the original NEURON NetCon.event(tdeliver) delivery step.
    reset_runtime()
    source_step = 2
    desired_delivery_ms = (source_step + 1) * EVENT_TEST_DT_MS
    net.netstim.schedule(0, [source_step * EVENT_TEST_DT_MS])
    net.netstim.initialize()

    sg_ampa = net.SG.mech.AMPA_DynSyn
    target_slot = net.SG.zhang2014_synapse_banks["sg_ampa_primary"]["local_indices"][0]
    first_nonzero_time = None
    for step in range(6):
        t_ms = step * EVENT_TEST_DT_MS
        net.t.fill_(t_ms)
        net.netstim(net.t, bptt=False, dt=EVENT_TEST_DT_MS)
        for netcon in net.synapses.values():
            netcon.advance()
        if first_nonzero_time is None and float(sg_ampa.A[target_slot]) > 0.0:
            first_nonzero_time = t_ms

    assert first_nonzero_time == pytest.approx(desired_delivery_ms)


def test_rescheduling_is_idempotent(full_network_uninitialized, windup_data):
    net = full_network_uninitialized
    schedule_windup_afferents(net, windup_data, clear=True)
    first = sum(len(heap) for heap in net.netstim._sched_heaps)
    schedule_windup_afferents(net, windup_data, clear=True)
    second = sum(len(heap) for heap in net.netstim._sched_heaps)
    assert first == second == 1215
