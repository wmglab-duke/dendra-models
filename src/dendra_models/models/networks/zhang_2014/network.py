"""Dendra network assembly for the Zhang et al. 2014 Wind-Up model.

The original NEURON realization is defined by parallel vectors describing the
source cell, target cell, target synapse index, weight, delay, and threshold for
every connection.  The default Dendra builder now creates only the four
biological cells and their interconnections.  Because every biological row uses
the same -30 mV source threshold, each projecting cell computes its crossing
once with ``dendra.models.mod.spikedetect`` and all outgoing receptor-specific
NetCons consume that shared event variable.  Artificial afferents are an optional
second layer that can be attached before or after the ``Network`` has been built.

External event times are supplied per ``NetCon`` with
``NetCon.event(tdeliver)``.  They are therefore *postsynaptic delivery times*,
not presynaptic spike times, and NEURON does not apply ``NetCon.delay`` to those
manually scheduled events.  The optional afferent layer preserves that data
model exactly.

Dendra routes a source event through a ``NetCon`` no earlier than the next
integration step.  To reproduce NEURON's delivery-time API, externally driven
connections use a one-step bridge delay and their source events are scheduled
one step early.  Biological connections retain the effective delays in ``DelayVector.txt``.
Dendra advances NetCons before population mechanisms, so the internal NetCon
delay is shortened by one fixed step to compensate for the one-step visibility
lag of a mechanism-produced spike pulse.

Two external-source representations are supported:

``shared_source`` (default)
    Use the 75 artificial source-cell IDs from the original realization as 75
    Dendra ``NetStim`` channels.  This is exact for the packaged Wind-Up data,
    because every externally driven connection sharing a source ID also shares
    the same spike train.

``per_connection``
    Use one ``NetStim`` channel for each externally scheduled connection (225
    channels in the packaged realization).  This mirrors ``NetCon.event``
    scheduling directly and remains exact for modified vector sets where two
    connections from the same artificial source have different schedules.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib.resources import files
from pathlib import Path
from types import MappingProxyType
from typing import Literal, Mapping, Sequence

import numpy as np

import dendra as dn
import torch
from dendra.models.mod import spikedetect
from dendra.models.parametric import PositiveParam

from .cells import ZHANG_CELSIUS, ZHANG_SYNAPSE_LAYOUTS, build_cell_populations


WINDUP_DT_MS = 0.0125
"""Fixed step used by ``RunSim_WindUp.hoc`` (ms)."""

WINDUP_SPIKE_THRESHOLD_MV = -30.0
"""Uniform biological source threshold in the published connection vectors."""

WINDUP_SPIKE_DETECTOR_ALIAS = "spikedetect"
"""Mechanism alias used for one shared detector per projecting cell."""

WINDUP_SPIKE_PRE_VAR = f"mech.{WINDUP_SPIKE_DETECTOR_ALIAS}.spikes"
"""Presynaptic event variable consumed by biological NetCons."""

WINDUP_SPIKE_SOURCE_SECTION = "soma"
WINDUP_SPIKE_SOURCE_LOC = 1.0
"""Published biological source location, equivalent to ``soma(1)``."""

WINDUP_SPIKE_DETECTOR_CONSUMPTION_LAG_STEPS = 1
"""Steps between detector evaluation and consumption by outgoing NetCons."""

WINDUP_TSTOP_MS = 21_000.0
"""Nominal stop time of the packaged Wind-Up protocol (ms)."""

WINDUP_N_AFFERENT_SOURCES = 75
"""Number of artificial ``S_NetStim`` source cells in the realization."""

WINDUP_EXTERNAL_EVENT_SEMANTICS = "netcon_event_delivery_time"
"""Meaning of times stored in ``SpikeTimesVector.txt``."""

WINDUP_GLOBAL_CELL_MAP: Mapping[int, str] = MappingProxyType(
    {
        75: "SG",
        76: "SGSCS",
        77: "EX",
        78: "T_Cell",
    }
)
"""Map original global biological-cell IDs to Dendra population names."""

WINDUP_SYNAPSE_BANK_ORDER: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "SG": (
            "sg_ampa_primary",
            "sg_ampa_secondary",
        ),
        "SGSCS": (
            "sg_ampa_primary",
            "sg_ampa_secondary",
        ),
        "EX": (
            "ex_ampa",
            "ex_nmda",
            "ex_nk1",
            "ex_gaba_local",
            "ex_gaba_scs",
        ),
        "T_Cell": (
            "wdr_abeta_ampa",
            "wdr_abeta_nmda",
            "wdr_c_ampa",
            "wdr_c_nmda",
            "wdr_nk1",
            "wdr_ex_ampa",
            "wdr_ex_nmda",
            "wdr_glycine",
            "wdr_gaba_local",
            "wdr_gaba_surround",
            "wdr_gaba_scs",
        ),
    }
)
"""HOC ``synlist`` order expressed in terms of Dendra mechanism banks."""

_AFFERENT_MODES = ("shared_source", "per_connection")
_CONNECTION_SUBSETS = ("all", "cellular", "afferent")


@dataclass(frozen=True, slots=True)
class WindUpData:
    """Parsed vector realization and external event schedules.

    Array entries are ordered exactly as in the original text files.  The first
    ``n_scheduled_connections`` rows of the connection vectors are the NetCons
    that receive explicitly scheduled events.  ``connection_spike_times`` and
    ``afferent_spike_times`` contain the original *delivery times* passed to
    NEURON's ``NetCon.event(tdeliver)``.
    """

    from_cell: np.ndarray
    to_cell: np.ndarray
    synapse_index: np.ndarray
    weight: np.ndarray
    delay_ms: np.ndarray
    threshold_mV: np.ndarray
    connection_spike_counts: np.ndarray
    connection_spike_times: tuple[np.ndarray, ...]
    tstop_ms: float
    n_afferent_sources: int
    afferent_spike_times: tuple[np.ndarray, ...] | None

    @property
    def n_connections(self) -> int:
        return int(self.from_cell.size)

    @property
    def n_scheduled_connections(self) -> int:
        return int(self.connection_spike_counts.size)

    @property
    def can_share_afferent_sources(self) -> bool:
        return self.afferent_spike_times is not None


@dataclass(frozen=True, slots=True)
class WindUpSynapseTarget:
    """One HOC ``synlist`` entry resolved to a Dendra mechanism slot.

    ``alias`` is the stable Zhang/HOC-style name exposed to users.
    ``mechanism`` is the compiled Dendra mechanism object name.
    ``local_index`` is the slot inside that mechanism that receives this
    connection.  In the optimized banked layout many aliases share one
    mechanism and differ only by local index.
    """

    alias: str
    bank: str
    mechanism: str
    local_index: int
    layout: str


@dataclass(frozen=True, slots=True)
class WindUpConnection:
    """Resolved description of one original connection-vector row."""

    index: int
    source_id: int
    source_population: str
    source_channel: int | None
    target_id: int
    target_population: str
    target_synapse_index: int
    target_synapse_alias: str
    target_synapse_mechanism: str
    target_synapse_local_index: int
    weight: float
    delay_ms: float
    runtime_delay_ms: float
    threshold_mV: float
    scheduled_connection_index: int | None
    runtime_connected: bool = True

    @property
    def is_external(self) -> bool:
        return self.scheduled_connection_index is not None

    @property
    def pre_var(self) -> str:
        """Runtime source variable used by the Dendra NetCon."""

        return "spike" if self.is_external else WINDUP_SPIKE_PRE_VAR

    @property
    def runtime_threshold_mV(self) -> None:
        """No NetCon-local threshold: sources already emit event variables."""

        return None

    @property
    def vector_delay_ms(self) -> float:
        """Delay declared in ``DelayVector.txt``.

        For externally scheduled rows this value is retained for provenance,
        although NEURON's ``NetCon.event`` does not use it.  The actual Dendra
        bridge delay is available as :attr:`runtime_delay_ms`.
        """

        return self.delay_ms


@dataclass(frozen=True, slots=True)
class WindUpNetworkMetadata:
    """Metadata attached to a Zhang cellular or afferent-driven network.

    ``cellular_connections`` always contains the biological wiring driven by
    shared source detectors. ``afferent_connections`` is empty until
    :func:`attach_windup_afferents` is called.  The :attr:`connections`
    property preserves the original connection-vector row order when the two
    subsets are combined.
    """

    data: WindUpData
    cellular_connections: tuple[WindUpConnection, ...]
    afferent_connections: tuple[WindUpConnection, ...]
    afferent_mode: str | None
    dt_ms: float
    tstop_ms: float
    external_event_semantics: str
    external_schedule_lead_ms: float
    source_cell_map: Mapping[int, str]
    synapse_bank_order: Mapping[str, tuple[str, ...]]
    spike_detector_alias: str
    spike_pre_var: str
    spike_threshold_mV: float
    spike_source_populations: tuple[str, ...]
    spike_detector_consumption_lag_steps: int

    @property
    def spike_detector_consumption_lag_ms(self) -> float:
        """Fixed mechanism-to-NetCon visibility lag in milliseconds."""

        return self.spike_detector_consumption_lag_steps * self.dt_ms

    @property
    def connections(self) -> tuple[WindUpConnection, ...]:
        """All currently attached connections in original vector-row order."""

        return tuple(
            sorted(
                self.afferent_connections + self.cellular_connections,
                key=lambda connection: connection.index,
            )
        )

    @property
    def afferents_attached(self) -> bool:
        """Whether artificial afferent sources and their NetCons are present."""

        return bool(self.afferent_connections)


class WindUpDataError(ValueError):
    """Raised when a vector realization is internally inconsistent."""


def _resource_root():
    return files(__package__) / "data" / "windup"


def _read_text(root, name: str) -> str:
    resource = root / name
    try:
        return resource.read_text(encoding="utf-8")
    except TypeError:  # pathlib.Path.read_text has no encoding issue, defensive.
        return resource.read_text()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing Zhang Wind-Up data file: {resource}") from exc


def _read_numeric_vector(root, name: str, *, dtype) -> np.ndarray:
    text = _read_text(root, name)
    values = np.fromstring(text, sep=" ", dtype=np.float64)
    if values.size == 0:
        raise WindUpDataError(f"{name} is empty or could not be parsed.")

    if np.issubdtype(np.dtype(dtype), np.integer):
        rounded = np.rint(values)
        if not np.allclose(values, rounded, rtol=0.0, atol=1.0e-9):
            bad = values[np.abs(values - rounded) > 1.0e-9][:5]
            raise WindUpDataError(
                f"{name} must contain integer-valued entries; examples: {bad.tolist()}"
            )
        values = rounded
    return np.asarray(values, dtype=dtype)


def _parse_spike_stats(root) -> tuple[np.ndarray, float]:
    stats = _read_numeric_vector(root, "SpikeStatsVector.txt", dtype=np.float64)
    n_schedules_f = float(stats[0])
    n_schedules = int(round(n_schedules_f))
    if abs(n_schedules_f - n_schedules) > 1.0e-9 or n_schedules < 0:
        raise WindUpDataError(
            "The first SpikeStatsVector entry must be a non-negative integer."
        )
    expected = n_schedules + 2
    if stats.size != expected:
        raise WindUpDataError(
            "SpikeStatsVector.txt must contain: schedule count, one spike count "
            f"per schedule, and tstop. Expected {expected} entries, got {stats.size}."
        )

    counts_f = stats[1 : 1 + n_schedules]
    counts = np.rint(counts_f).astype(np.int64)
    if np.any(counts < 0) or not np.allclose(
        counts_f, counts, rtol=0.0, atol=1.0e-9
    ):
        raise WindUpDataError("Spike counts must be non-negative integers.")

    tstop_ms = float(stats[-1])
    if not np.isfinite(tstop_ms) or tstop_ms <= 0.0:
        raise WindUpDataError(f"Invalid Wind-Up tstop: {tstop_ms!r}")
    return counts, tstop_ms


def _parse_spike_times(
    root,
    counts: np.ndarray,
    *,
    sentinel: float = -1.0e15,
) -> tuple[np.ndarray, ...]:
    values = _read_numeric_vector(root, "SpikeTimesVector.txt", dtype=np.float64)
    schedules: list[np.ndarray] = []
    start = 0
    sentinel_idx = np.flatnonzero(values <= sentinel)

    if sentinel_idx.size != counts.size:
        raise WindUpDataError(
            "SpikeTimesVector sentinel count does not match SpikeStatsVector: "
            f"{sentinel_idx.size} versus {counts.size}."
        )

    for schedule_idx, stop in enumerate(sentinel_idx):
        times = np.asarray(values[start:stop], dtype=np.float64).copy()
        expected = int(counts[schedule_idx])
        if times.size != expected:
            raise WindUpDataError(
                f"Spike schedule {schedule_idx} contains {times.size} times, "
                f"but SpikeStatsVector declares {expected}."
            )
        if times.size:
            if not np.all(np.isfinite(times)):
                raise WindUpDataError(
                    f"Spike schedule {schedule_idx} contains non-finite times."
                )
            if np.any(np.diff(times) < 0.0):
                raise WindUpDataError(
                    f"Spike schedule {schedule_idx} is not monotonically ordered."
                )
        schedules.append(times)
        start = int(stop) + 1

    if start != values.size:
        trailing = values[start:]
        if trailing.size and np.any(np.abs(trailing) > 0.0):
            raise WindUpDataError(
                "SpikeTimesVector contains values after its final sentinel."
            )
    return tuple(schedules)


def _collapse_afferent_schedules(
    from_cell: np.ndarray,
    schedules: Sequence[np.ndarray],
    *,
    n_afferent_sources: int,
    strict: bool,
) -> tuple[np.ndarray, ...] | None:
    """Collapse per-connection schedules onto artificial source IDs.

    Returns ``None`` when schedules disagree and ``strict`` is False.  In strict
    mode, disagreement raises a diagnostic error because the compact source
    representation would alter the model.
    """

    by_source: list[np.ndarray | None] = [None] * int(n_afferent_sources)
    rows_by_source: list[list[int]] = [[] for _ in range(int(n_afferent_sources))]

    for row, (source_id, schedule) in enumerate(zip(from_cell, schedules)):
        source = int(source_id)
        if source < 0 or source >= n_afferent_sources:
            raise WindUpDataError(
                f"Scheduled connection {row} has non-afferent source ID {source}."
            )
        rows_by_source[source].append(row)
        canonical = by_source[source]
        if canonical is None:
            by_source[source] = np.asarray(schedule, dtype=np.float64).copy()
            continue
        if not np.array_equal(canonical, schedule):
            if strict:
                raise WindUpDataError(
                    "Cannot use afferent_mode='shared_source': scheduled "
                    f"connections from source {source} have different spike trains "
                    f"(including row {row}). Use afferent_mode='per_connection'."
                )
            return None

    missing = [source for source, rows in enumerate(rows_by_source) if not rows]
    if missing:
        if strict:
            raise WindUpDataError(
                "The compact shared-source representation requires every artificial "
                f"source ID to be represented; missing IDs: {missing[:10]}"
            )
        return None

    return tuple(schedule for schedule in by_source if schedule is not None)


def load_windup_data(
    data_dir: str | Path | None = None,
    *,
    n_afferent_sources: int = WINDUP_N_AFFERENT_SOURCES,
    require_shared_afferent_schedules: bool = False,
) -> WindUpData:
    """Load and validate one Zhang Wind-Up vector realization.

    Parameters
    ----------
    data_dir:
        Directory containing the eight vector files.  By default the packaged
        ModelDB realization is used.
    n_afferent_sources:
        IDs ``0 .. n_afferent_sources-1`` are treated as artificial sources.
    require_shared_afferent_schedules:
        If True, fail when two scheduled connections from the same source ID
        have different event times.  This is required by the compact
        ``shared_source`` network representation.
    """

    if n_afferent_sources <= 0:
        raise ValueError("n_afferent_sources must be positive.")
    root = _resource_root() if data_dir is None else Path(data_dir)

    vectors = {
        "from_cell": _read_numeric_vector(root, "FromVector.txt", dtype=np.int64),
        "to_cell": _read_numeric_vector(root, "ToVector.txt", dtype=np.int64),
        "synapse_index": _read_numeric_vector(
            root, "SynapseVector.txt", dtype=np.int64
        ),
        "weight": _read_numeric_vector(root, "WeightVector.txt", dtype=np.float64),
        "delay_ms": _read_numeric_vector(root, "DelayVector.txt", dtype=np.float64),
        "threshold_mV": _read_numeric_vector(
            root, "ThresholdVector.txt", dtype=np.float64
        ),
    }

    lengths = {name: int(value.size) for name, value in vectors.items()}
    if len(set(lengths.values())) != 1:
        raise WindUpDataError(f"Connection vector lengths differ: {lengths}")
    n_connections = next(iter(lengths.values()))
    if n_connections == 0:
        raise WindUpDataError("The connection vectors contain no connections.")

    if np.any(vectors["from_cell"] < 0):
        raise WindUpDataError("FromVector contains a negative source ID.")
    if np.any(vectors["synapse_index"] < 0):
        raise WindUpDataError("SynapseVector contains a negative target index.")
    if np.any(~np.isfinite(vectors["weight"])) or np.any(vectors["weight"] < 0.0):
        raise WindUpDataError("WeightVector must contain finite non-negative values.")
    if np.any(~np.isfinite(vectors["delay_ms"])) or np.any(
        vectors["delay_ms"] < 0.0
    ):
        raise WindUpDataError("DelayVector must contain finite non-negative values.")
    if np.any(~np.isfinite(vectors["threshold_mV"])):
        raise WindUpDataError("ThresholdVector contains non-finite values.")

    counts, tstop_ms = _parse_spike_stats(root)
    if counts.size > n_connections:
        raise WindUpDataError(
            "There are more explicitly scheduled connections than connection rows: "
            f"{counts.size} > {n_connections}."
        )
    schedules = _parse_spike_times(root, counts)

    scheduled_sources = vectors["from_cell"][: counts.size]
    if np.any(scheduled_sources >= n_afferent_sources):
        bad_row = int(np.flatnonzero(scheduled_sources >= n_afferent_sources)[0])
        raise WindUpDataError(
            "Explicit schedules are attached to the leading NetCon rows in the "
            "original model, but one of those rows is biological: "
            f"row={bad_row}, source={int(scheduled_sources[bad_row])}."
        )
    if np.any(vectors["from_cell"][counts.size :] < n_afferent_sources):
        bad = int(
            counts.size
            + np.flatnonzero(
                vectors["from_cell"][counts.size :] < n_afferent_sources
            )[0]
        )
        raise WindUpDataError(
            "An artificial-source connection appears after the explicitly "
            f"scheduled prefix (row {bad})."
        )

    afferent_spike_times = _collapse_afferent_schedules(
        scheduled_sources,
        schedules,
        n_afferent_sources=n_afferent_sources,
        strict=require_shared_afferent_schedules,
    )

    return WindUpData(
        from_cell=vectors["from_cell"],
        to_cell=vectors["to_cell"],
        synapse_index=vectors["synapse_index"],
        weight=vectors["weight"],
        delay_ms=vectors["delay_ms"],
        threshold_mV=vectors["threshold_mV"],
        connection_spike_counts=counts,
        connection_spike_times=schedules,
        tstop_ms=tstop_ms,
        n_afferent_sources=int(n_afferent_sources),
        afferent_spike_times=afferent_spike_times,
    )


def _synapse_targets_from_bank_metadata(
    population_name: str,
    population,
) -> tuple[WindUpSynapseTarget, ...] | None:
    """Return optimized banked synapse targets when available."""

    banks = getattr(population, "zhang2014_synapse_banks", None)
    if not banks:
        return None

    targets: list[WindUpSynapseTarget] = []
    missing: list[str] = []
    for bank in WINDUP_SYNAPSE_BANK_ORDER[population_name]:
        info = banks.get(bank)
        if info is None:
            missing.append(bank)
            continue
        aliases = tuple(str(alias) for alias in info.get("aliases", ()))
        layout = str(info.get("layout", "banked"))
        if layout == "banked":
            mechanism = str(info["mechanism"])
            local_indices = tuple(int(i) for i in info.get("local_indices", ()))
        else:
            mechanism = None
            local_indices = tuple(int(i) for i in info.get("local_indices", ()))
        if len(aliases) != len(local_indices):
            raise WindUpDataError(
                f"Population {population_name!r} bank {bank!r} has "
                f"{len(aliases)} aliases but {len(local_indices)} local indices."
            )
        for alias, local_index in zip(aliases, local_indices):
            targets.append(
                WindUpSynapseTarget(
                    alias=alias,
                    bank=bank,
                    mechanism=(mechanism if mechanism is not None else alias),
                    local_index=int(local_index),
                    layout=layout,
                )
            )
    if missing:
        raise WindUpDataError(
            f"Population {population_name!r} is missing synapse banks: {missing}"
        )
    return tuple(targets)


def synapse_targets(population_name: str, population) -> tuple[WindUpSynapseTarget, ...]:
    """Return one population's HOC synlist entries as Dendra mechanism slots."""

    try:
        WINDUP_SYNAPSE_BANK_ORDER[population_name]
    except KeyError as exc:
        raise KeyError(f"Unknown Zhang population {population_name!r}.") from exc

    banked = _synapse_targets_from_bank_metadata(population_name, population)
    if banked is not None:
        return banked

    # Backward-compatible independent-layout fallback: every HOC point process
    # is its own mechanism alias and its only local slot is 0.
    banks = getattr(population, "zhang2014_synapses", None)
    if banks is None:
        raise AttributeError(
            f"Population {population_name!r} has no zhang2014_synapses metadata; "
            "build it with insert_synapses=True."
        )

    targets: list[WindUpSynapseTarget] = []
    missing: list[str] = []
    for bank in WINDUP_SYNAPSE_BANK_ORDER[population_name]:
        if bank not in banks:
            missing.append(bank)
            continue
        for alias in banks[bank]:
            alias = str(alias)
            targets.append(
                WindUpSynapseTarget(
                    alias=alias,
                    bank=bank,
                    mechanism=alias,
                    local_index=0,
                    layout="independent",
                )
            )
    if missing:
        raise WindUpDataError(
            f"Population {population_name!r} is missing synapse banks: {missing}"
        )
    return tuple(targets)


def synapse_aliases(population_name: str, population) -> tuple[str, ...]:
    """Return one population's target synapse names in original HOC order."""

    return tuple(target.alias for target in synapse_targets(population_name, population))


def _validate_target_synapses(data: WindUpData, populations) -> dict[str, tuple[WindUpSynapseTarget, ...]]:
    targets_by_population = {
        name: synapse_targets(name, population)
        for name, population in populations.items()
    }

    unknown_targets = sorted(set(map(int, data.to_cell)) - set(WINDUP_GLOBAL_CELL_MAP))
    if unknown_targets:
        raise WindUpDataError(
            f"ToVector contains unknown biological target IDs: {unknown_targets}"
        )

    for population_name, targets in targets_by_population.items():
        pop = populations[population_name]
        handler = getattr(pop, "mech", None)
        if handler is None:
            raise WindUpDataError(
                f"Population {population_name!r} has not been built with mechanisms."
            )
        for target in targets:
            try:
                syn = getattr(handler, target.mechanism)
            except AttributeError as exc:
                raise WindUpDataError(
                    f"Population {population_name!r} target {target.alias!r} "
                    f"requires compiled mechanism {target.mechanism!r}."
                ) from exc
            syn_numel = int(np.prod(tuple(syn.shape_f)))
            if int(target.local_index) < 0 or int(target.local_index) >= syn_numel:
                raise WindUpDataError(
                    f"Population {population_name!r} target {target.alias!r} "
                    f"uses local slot {target.local_index}, but mechanism "
                    f"{target.mechanism!r} has only {syn_numel} slots."
                )

    for row, (target_id, synapse_index) in enumerate(
        zip(data.to_cell, data.synapse_index)
    ):
        target_name = WINDUP_GLOBAL_CELL_MAP[int(target_id)]
        targets = targets_by_population[target_name]
        syn_idx = int(synapse_index)
        if syn_idx >= len(targets):
            raise WindUpDataError(
                f"Connection row {row} targets {target_name} synapse {syn_idx}, "
                f"but only {len(targets)} synapses were inserted."
            )
    return targets_by_population


def _positive_parameter(
    value: float,
    *,
    device,
    dtype,
    trainable: bool,
) -> PositiveParam:
    """Create a non-negative parameter that preserves exact zero values."""

    tensor = torch.as_tensor(value, device=device, dtype=dtype)
    return PositiveParam(
        tensor,
        include_zero=True,
        requires_grad=bool(trainable),
    )


def _make_netstim(
    N: int,
    *,
    tstop_ms: float,
    seed: int | None,
    device,
    dtype,
):
    # Intrinsic NetStim renewal events are disabled over the modeled horizon by
    # placing ``start`` far beyond tstop. Explicit schedules remain active.
    quiet_start = max(float(tstop_ms) + 1.0e6, 1.0e6)
    return dn.NetStim(
        N=N,
        interval=quiet_start,
        start=quiet_start,
        noise=0.0,
        max_spikes=1_000_000,
        seed=seed,
        device=device,
        dtype=dtype,
    )


def _label_netstim_ranges(netstim, *, afferent_mode: str, data: WindUpData) -> None:
    if afferent_mode == "shared_source":
        # Neutral labels avoid assigning uncertain sensory semantics while still
        # making the source blocks easy to inspect.
        for start, stop in ((0, 15), (15, 30), (30, 45), (45, 60), (60, 75)):
            if start < netstim.N:
                netstim[start : min(stop, netstim.N)].label(
                    f"source_{start:02d}_{min(stop, netstim.N) - 1:02d}"
                )
    else:
        netstim[: data.n_scheduled_connections].label("scheduled_connection")


def _pending_mechanism_for_alias(population, alias: str):
    """Return one pending ``Population.insert`` record for ``alias``."""

    matches = []
    for mechanism, (name, _ic, kwargs) in getattr(
        population, "_mech_everywhere", {}
    ).items():
        if name == alias:
            matches.append((mechanism, kwargs, None))
    for mechanism, records in getattr(population, "_mech_data", {}).items():
        for record in records:
            record_alias, kwargs, key = record[:3]
            effective_alias = record_alias or mechanism.__name__
            if effective_alias == alias:
                matches.append((mechanism, kwargs, key))
    if len(matches) > 1:
        raise WindUpDataError(
            f"Population {getattr(population, 'name', '<unnamed>')!r} has "
            f"multiple pending mechanisms using alias {alias!r}."
        )
    return matches[0] if matches else None


def _flat_indices_for_key(population, key) -> torch.Tensor:
    """Return population-flat indices selected by a mechanism insertion key."""

    grid = torch.arange(
        population.v.numel(),
        device=population.device(),
        dtype=torch.long,
    ).reshape(population.shape)
    if key is None:
        return grid.reshape(-1)
    return grid[key].reshape(-1)


def _spike_source_slice(population):
    source = population.slice(
        WINDUP_SPIKE_SOURCE_SECTION,
        loc=WINDUP_SPIKE_SOURCE_LOC,
    )
    if source.is_empty:
        raise WindUpDataError(
            f"Could not select {WINDUP_SPIKE_SOURCE_SECTION}"
            f"({WINDUP_SPIKE_SOURCE_LOC:g}) source location for "
            f"{getattr(population, 'name', '<unnamed>')}."
        )
    return source


def _spike_source_flat_indices(population) -> torch.Tensor:
    source = _spike_source_slice(population)
    return _flat_indices_for_key(population, source.index)


def _validate_detector_threshold(value, *, population_name: str) -> None:
    """Require an existing detector to use the published Zhang threshold."""

    value_t = torch.as_tensor(value).detach().to(dtype=torch.float64)
    expected = torch.full_like(value_t, WINDUP_SPIKE_THRESHOLD_MV)
    if value_t.numel() == 0 or not torch.allclose(
        value_t, expected, atol=0.0, rtol=0.0
    ):
        observed = value_t.reshape(-1)[:8].cpu().tolist()
        raise WindUpDataError(
            f"Population {population_name!r} already has a "
            f"{WINDUP_SPIKE_DETECTOR_ALIAS!r} mechanism, but its threshold is "
            f"{observed}; Zhang wiring requires "
            f"{WINDUP_SPIKE_THRESHOLD_MV} mV."
        )


def _validate_compiled_detector(population, detector, *, population_name: str) -> None:
    """Check that a compiled detector covers every Zhang source compartment."""

    grid = torch.arange(
        population.v.numel(),
        device=population.device(),
        dtype=torch.long,
    ).reshape(population.shape)
    detector_indices = detector.get(grid).reshape(-1)
    source_indices = _spike_source_flat_indices(population)

    positions = []
    for source_index in source_indices:
        found = torch.nonzero(
            detector_indices == source_index, as_tuple=False
        ).reshape(-1)
        if found.numel() != 1:
            raise WindUpDataError(
                f"Population {population_name!r} already has a "
                f"{WINDUP_SPIKE_DETECTOR_ALIAS!r} mechanism, but it does not "
                "cover every Zhang soma(1) source compartment exactly once."
            )
        positions.append(found[0])

    positions_t = torch.stack(positions).to(detector.threshold.device)
    thresholds = detector.threshold.reshape(-1).index_select(0, positions_t)
    _validate_detector_threshold(thresholds, population_name=population_name)


def _validate_pending_detector(
    population,
    kwargs,
    key,
    *,
    population_name: str,
) -> None:
    """Check source coverage for a detector not yet compiled by ``build``."""

    detector_indices = _flat_indices_for_key(population, key)
    source_indices = _spike_source_flat_indices(population)
    if not torch.isin(source_indices, detector_indices).all():
        raise WindUpDataError(
            f"Population {population_name!r} already reserves alias "
            f"{WINDUP_SPIKE_DETECTOR_ALIAS!r}, but that insertion does not "
            "cover every Zhang soma(1) source compartment."
        )
    _validate_detector_threshold(
        kwargs.get("threshold", 0.0),
        population_name=population_name,
    )


def _biological_source_populations(data: WindUpData) -> tuple[str, ...]:
    """Return projecting populations after validating one common threshold."""

    rows = range(data.n_scheduled_connections, data.n_connections)
    thresholds = {float(data.threshold_mV[row]) for row in rows}
    if thresholds != {WINDUP_SPIKE_THRESHOLD_MV}:
        raise WindUpDataError(
            "Shared Zhang spike detection requires one biological threshold. "
            f"Observed {sorted(thresholds)!r}; expected "
            f"[{WINDUP_SPIKE_THRESHOLD_MV!r}]."
        )

    source_ids = {int(data.from_cell[row]) for row in rows}
    return tuple(
        name
        for global_id, name in WINDUP_GLOBAL_CELL_MAP.items()
        if global_id in source_ids
    )


def _install_windup_spike_detectors(
    populations: Mapping[str, object],
    data: WindUpData,
) -> tuple[str, ...]:
    """Insert one shared soma detector per biologically projecting cell.

    Compatible pre-existing detectors are reused. Alias collisions, incomplete
    source coverage, and non-Zhang thresholds fail explicitly rather than
    silently changing event semantics.
    """

    source_names = _biological_source_populations(data)
    for name in source_names:
        population = populations[name]
        source_indices = _spike_source_flat_indices(population)
        expected_sources = int(getattr(population, "np", population.shape[0]))
        if source_indices.numel() != expected_sources:
            raise WindUpDataError(
                f"Population {name!r} must expose exactly one "
                f"{WINDUP_SPIKE_SOURCE_SECTION}({WINDUP_SPIKE_SOURCE_LOC:g}) "
                f"spike source per cell; found {source_indices.numel()} for "
                f"N={expected_sources}."
            )

        mechanisms = getattr(
            getattr(population, "mech", None), "mechanisms", {}
        )
        compiled = (
            mechanisms[WINDUP_SPIKE_DETECTOR_ALIAS]
            if WINDUP_SPIKE_DETECTOR_ALIAS in mechanisms
            else None
        )
        pending = _pending_mechanism_for_alias(
            population, WINDUP_SPIKE_DETECTOR_ALIAS
        )

        if compiled is not None:
            if not isinstance(compiled, spikedetect):
                raise WindUpDataError(
                    f"Population {name!r} already uses mechanism alias "
                    f"{WINDUP_SPIKE_DETECTOR_ALIAS!r} for "
                    f"{compiled.__class__.__name__!r}."
                )
            _validate_compiled_detector(
                population, compiled, population_name=name
            )
            continue

        if pending is not None:
            mechanism, kwargs, key = pending
            if mechanism is not spikedetect:
                raise WindUpDataError(
                    f"Population {name!r} already reserves alias "
                    f"{WINDUP_SPIKE_DETECTOR_ALIAS!r} for "
                    f"{mechanism.__name__!r}."
                )
            _validate_pending_detector(
                population,
                kwargs,
                key,
                population_name=name,
            )
            continue

        _spike_source_slice(population).insert(
            spikedetect,
            alias=WINDUP_SPIKE_DETECTOR_ALIAS,
            threshold=WINDUP_SPIKE_THRESHOLD_MV,
        )

    return source_names


def _require_compiled_windup_spike_detectors(
    populations: Mapping[str, object],
    data: WindUpData,
) -> tuple[str, ...]:
    """Validate detector availability before queuing biological NetCons."""

    source_names = _biological_source_populations(data)
    for name in source_names:
        population = populations[name]
        mechanisms = getattr(
            getattr(population, "mech", None), "mechanisms", {}
        )
        detector = (
            mechanisms[WINDUP_SPIKE_DETECTOR_ALIAS]
            if WINDUP_SPIKE_DETECTOR_ALIAS in mechanisms
            else None
        )
        if detector is None:
            raise WindUpDataError(
                f"Population {name!r} has no compiled shared spike detector. "
                "Use assemble_windup_network(), which installs detectors before "
                "constructing the Dendra Network."
            )
        if not isinstance(detector, spikedetect):
            raise WindUpDataError(
                f"Population {name!r} uses alias "
                f"{WINDUP_SPIKE_DETECTOR_ALIAS!r} for an incompatible mechanism."
            )
        _validate_compiled_detector(
            population, detector, population_name=name
        )
    return source_names


def _runtime_biological_delay_ms(
    published_delay_ms: float,
    dt_ms: float,
    *,
    row: int,
) -> float:
    """Compensate the source-mechanism visibility lag in integer-step space."""

    published_steps = int(round(float(published_delay_ms) / float(dt_ms)))
    minimum_steps = WINDUP_SPIKE_DETECTOR_CONSUMPTION_LAG_STEPS + 1
    if published_steps < minimum_steps:
        raise WindUpDataError(
            "A shared spikedetect source requires a biological delay of at "
            f"least {minimum_steps} timesteps; row {row} has "
            f"delay={published_delay_ms} ms at dt={dt_ms} ms "
            f"({published_steps} timestep(s))."
        )
    runtime_steps = (
        published_steps - WINDUP_SPIKE_DETECTOR_CONSUMPTION_LAG_STEPS
    )
    return float(runtime_steps * dt_ms)


def schedule_windup_afferents(
    network,
    data: WindUpData | None = None,
    *,
    dt_ms: float | None = None,
    clear: bool = True,
    allow_past: bool = False,
):
    """Install the realization's explicit afferent delivery times on ``network``.

    Schedules are retained by ``Network.initialize``.  Therefore the normal
    order is to build the network (which schedules by default), then call
    ``network.initialize(network.zhang2014.dt_ms)``.

    The source events are placed one integration step before each original
    ``NetCon.event(tdeliver)`` time.  Externally driven Dendra NetCons use a
    matching one-step bridge delay, so their target mechanisms receive the
    event on the original delivery step.  ``dt_ms`` must therefore match the
    timestep later supplied to ``Network.initialize``.
    """

    metadata = getattr(network, "zhang2014", None)
    if data is None:
        if metadata is None:
            raise AttributeError(
                "No Wind-Up data supplied and network has no zhang2014 metadata."
            )
        data = metadata.data
    if network.netstim is None:
        raise AttributeError(
            "The network has no NetStim source. Call "
            "attach_windup_afferents(network, schedule=False) first."
        )

    afferent_mode = (
        metadata.afferent_mode
        if metadata is not None
        else getattr(network, "zhang2014_afferent_mode", None)
    )
    if afferent_mode not in _AFFERENT_MODES:
        raise ValueError(
            "The network does not have Zhang afferents attached, or its "
            f"afferent mode is invalid: {afferent_mode!r}."
        )

    if dt_ms is None:
        dt_ms = metadata.dt_ms if metadata is not None else WINDUP_DT_MS
    dt_ms = float(dt_ms)
    if not np.isfinite(dt_ms) or dt_ms <= 0.0:
        raise ValueError(f"dt_ms must be finite and positive; got {dt_ms!r}.")

    def source_times(delivery_times: np.ndarray) -> list[float]:
        delivery_times = np.asarray(delivery_times, dtype=np.float64)
        shifted = delivery_times - dt_ms
        if np.any(shifted < 0.0):
            first = int(np.flatnonzero(shifted < 0.0)[0])
            raise WindUpDataError(
                "An external delivery occurs before one Dendra integration step "
                "and cannot be represented by the source-to-NetCon bridge: "
                f"delivery={float(delivery_times[first])} ms, dt={dt_ms} ms."
            )
        return shifted.tolist()

    if clear:
        network.netstim.clear_schedule()

    if afferent_mode == "shared_source":
        schedules = data.afferent_spike_times
        if schedules is None:
            raise WindUpDataError(
                "This data set cannot be represented by shared source schedules; "
                "rebuild with afferent_mode='per_connection'."
            )
        for source_id, times in enumerate(schedules):
            network.netstim.schedule(
                int(source_id),
                source_times(times),
                allow_past=allow_past,
            )
    else:
        for connection_index, times in enumerate(data.connection_spike_times):
            network.netstim.schedule(
                int(connection_index),
                source_times(times),
                allow_past=allow_past,
            )
    return network


def _source_endpoint(
    network,
    data: WindUpData,
    *,
    row: int,
    source_id: int,
    afferent_mode: str,
):
    if row < data.n_scheduled_connections:
        if network.netstim is None:
            raise WindUpDataError(
                "Afferent connection rows were requested, but the Network has "
                "no NetStim. Call network.attach_netstim(...) or use "
                "attach_windup_afferents(...)."
            )
        source_channel = source_id if afferent_mode == "shared_source" else row
        return network.netstim[int(source_channel)], "netstim", int(source_channel)

    try:
        source_population = WINDUP_GLOBAL_CELL_MAP[int(source_id)]
    except KeyError as exc:
        raise WindUpDataError(
            f"Connection row {row} has unknown biological source ID {source_id}."
        ) from exc
    source = network.populations[source_population].slice("soma", loc=1.0)
    if source.is_empty:
        raise WindUpDataError(
            f"Could not select soma(1) source location for {source_population}."
        )
    return source, source_population, None


def connect_windup_network(
    network,
    data: WindUpData | None = None,
    *,
    subset: Literal["all", "cellular", "afferent"] = "all",
    afferent_mode: Literal["shared_source", "per_connection"] = "shared_source",
    dt_ms: float = WINDUP_DT_MS,
    trainable_weights: bool = False,
    trainable_delays: bool = False,
    omit_static_zero_weight_connections: bool = True,
) -> tuple[WindUpConnection, ...]:
    """Translate selected connection-vector rows into Dendra NetCons.

    Parameters
    ----------
    subset : {"all", "cellular", "afferent"}, default "all"
        ``"cellular"`` creates only shared-detector connections among SG,
        SGSCS, EX, and T_Cell. ``"afferent"`` creates only the explicitly
        scheduled source rows and therefore requires an attached NetStim.
        ``"all"`` retains the complete original-vector behavior.

    Notes
    -----
    The four biological populations must already contain the independent
    synapse mechanism instances inserted by :func:`build_cell_populations`.
    Biological sources must also contain the shared ``spikedetect`` mechanism;
    :func:`assemble_windup_network` installs and validates it automatically.
    """

    if subset not in _CONNECTION_SUBSETS:
        raise ValueError(
            f"subset must be one of {_CONNECTION_SUBSETS}; got {subset!r}."
        )
    if afferent_mode not in _AFFERENT_MODES:
        raise ValueError(
            f"afferent_mode must be one of {_AFFERENT_MODES}; got {afferent_mode!r}."
        )
    dt_ms = float(dt_ms)
    if not np.isfinite(dt_ms) or dt_ms <= 0.0:
        raise ValueError(f"dt_ms must be finite and positive; got {dt_ms!r}.")
    if data is None:
        data = load_windup_data(
            require_shared_afferent_schedules=(
                subset in ("all", "afferent")
                and afferent_mode == "shared_source"
            )
        )
    if (
        subset in ("all", "afferent")
        and afferent_mode == "shared_source"
        and not data.can_share_afferent_sources
    ):
        raise WindUpDataError(
            "The supplied connection schedules differ within at least one "
            "artificial source ID; use afferent_mode='per_connection'."
        )

    if set(network.populations) != set(WINDUP_GLOBAL_CELL_MAP.values()):
        missing = set(WINDUP_GLOBAL_CELL_MAP.values()) - set(network.populations)
        extra = set(network.populations) - set(WINDUP_GLOBAL_CELL_MAP.values())
        raise WindUpDataError(
            f"Unexpected biological populations. Missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )

    if subset in ("all", "cellular"):
        _require_compiled_windup_spike_detectors(network.populations, data)

    targets_by_population = _validate_target_synapses(data, network.populations)
    records: list[WindUpConnection] = []

    if subset == "cellular":
        rows = range(data.n_scheduled_connections, data.n_connections)
    elif subset == "afferent":
        rows = range(0, data.n_scheduled_connections)
    else:
        rows = range(data.n_connections)

    for row in rows:
        source_id = int(data.from_cell[row])
        target_id = int(data.to_cell[row])
        target_population = WINDUP_GLOBAL_CELL_MAP[target_id]
        target_synapse_index = int(data.synapse_index[row])
        target_synapse = targets_by_population[target_population][target_synapse_index]
        target_synapse_alias = target_synapse.alias

        source, source_population, source_channel = _source_endpoint(
            network,
            data,
            row=row,
            source_id=source_id,
            afferent_mode=afferent_mode,
        )
        target_model = network.populations[target_population]
        try:
            synapse = getattr(target_model.mech, target_synapse.mechanism)
        except AttributeError as exc:
            raise WindUpDataError(
                f"Target synapse {target_population}.{target_synapse_alias} "
                f"requires mechanism {target_synapse.mechanism!r}."
            ) from exc

        # PositiveParam(include_zero=True) is supplied explicitly so original
        # zero-weight connections remain mathematically zero rather than being
        # reconstructed as a tiny positive softplus value.
        weight = _positive_parameter(
            float(data.weight[row]),
            device=target_model.device(),
            dtype=target_model.dtype(),
            trainable=trainable_weights,
        )
        is_external = row < data.n_scheduled_connections
        # The original scheduled rows are activated with NetCon.event(tdeliver),
        # which bypasses their declared NetCon.delay. Dendra's source-event path
        # has a minimum one-step transit, so those rows use one dt and their
        # source events are scheduled one dt early. Biological rows consume a
        # mechanism-produced event one step after that detector is evaluated;
        # shorten the internal NetCon delay by one step to preserve the original
        # effective voltage-crossing-to-delivery latency.
        runtime_delay_ms = (
            dt_ms
            if is_external
            else _runtime_biological_delay_ms(
                float(data.delay_ms[row]), dt_ms, row=row
            )
        )
        delay = _positive_parameter(
            runtime_delay_ms,
            device=target_model.device(),
            dtype=target_model.dtype(),
            trainable=trainable_delays,
        )

        source_count = int(np.prod(tuple(getattr(source, "shape", ()))) or 0)
        if source_count != 1:
            raise WindUpDataError(
                f"Connection row {row} resolved to {source_count} presynaptic "
                "source indices; the exact Wind-Up realization expects one "
                "source slot per row."
            )

        target_slots = network.synapse_slots(
            target_model,
            synapse,
            slots=[int(target_synapse.local_index)],
        )
        if len(target_slots) != 1:
            raise WindUpDataError(
                f"Connection row {row} resolved to {len(target_slots)} "
                "postsynaptic synapse slots; the exact Wind-Up realization "
                "expects one target slot per row."
            )

        runtime_connected = not (
            bool(omit_static_zero_weight_connections)
            and not bool(trainable_weights)
            and float(data.weight[row]) == 0.0
        )
        if runtime_connected:
            network.connect_one_to_one_slots(
                source,
                target_slots,
                threshold=None,
                weight=weight,
                delay=delay,
                pre_var=("spike" if is_external else WINDUP_SPIKE_PRE_VAR),
            )

        records.append(
            WindUpConnection(
                index=row,
                source_id=source_id,
                source_population=source_population,
                source_channel=source_channel,
                target_id=target_id,
                target_population=target_population,
                target_synapse_index=target_synapse_index,
                target_synapse_alias=target_synapse_alias,
                target_synapse_mechanism=target_synapse.mechanism,
                target_synapse_local_index=int(target_synapse.local_index),
                weight=float(data.weight[row]),
                delay_ms=float(data.delay_ms[row]),
                runtime_delay_ms=runtime_delay_ms,
                threshold_mV=float(data.threshold_mV[row]),
                scheduled_connection_index=(row if is_external else None),
                runtime_connected=runtime_connected,
            )
        )

    return tuple(records)


def _validate_biological_populations(populations: Mapping[str, object]) -> dict[str, object]:
    populations = dict(populations)
    required = set(WINDUP_GLOBAL_CELL_MAP.values())
    if set(populations) != required:
        missing = required - set(populations)
        extra = set(populations) - required
        raise WindUpDataError(
            f"Unexpected biological populations. Missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    for name, population in populations.items():
        shape = tuple(getattr(population, "shape", ()))
        if not shape or int(shape[0]) != 1:
            raise WindUpDataError(
                f"The packaged realization requires exactly one {name} cell; "
                f"got population shape {shape!r}."
            )
    return populations


def _set_metadata_aliases(network) -> None:
    metadata = network.zhang2014
    network.zhang2014_connections = metadata.connections
    network.zhang2014_cellular_connections = metadata.cellular_connections
    network.zhang2014_afferent_connections = metadata.afferent_connections
    network.zhang2014_afferent_mode = metadata.afferent_mode
    network.zhang2014_afferents_attached = metadata.afferents_attached
    network.zhang2014_dt_ms = metadata.dt_ms
    network.zhang2014_tstop_ms = metadata.tstop_ms
    network.zhang2014_spike_detector_alias = metadata.spike_detector_alias
    network.zhang2014_spike_pre_var = metadata.spike_pre_var
    network.zhang2014_spike_threshold_mV = metadata.spike_threshold_mV
    network.zhang2014_spike_source_populations = metadata.spike_source_populations
    network.zhang2014_spike_detector_consumption_lag_steps = (
        metadata.spike_detector_consumption_lag_steps
    )


def _connection_vectors_match(a: WindUpData, b: WindUpData) -> bool:
    """Return True when two data sets define the same biological/vector wiring."""

    integer_fields = ("from_cell", "to_cell", "synapse_index")
    float_fields = ("weight", "delay_ms", "threshold_mV")
    integer_match = all(
        np.array_equal(getattr(a, name), getattr(b, name))
        for name in integer_fields
    )
    float_match = all(
        np.array_equal(getattr(a, name), getattr(b, name))
        for name in float_fields
    )
    return integer_match and float_match


def attach_windup_afferents(
    network,
    *,
    data: WindUpData | None = None,
    data_dir: str | Path | None = None,
    netstim=None,
    afferent_mode: Literal["shared_source", "per_connection"] = "shared_source",
    schedule: bool = True,
    dt_ms: float | None = None,
    seed: int | None = None,
    trainable_weights: bool = False,
    trainable_delays: bool = False,
    omit_static_zero_weight_connections: bool = True,
):
    """Attach the original artificial afferents to a cellular Zhang network.

    This function is intentionally separate from cellular network assembly.
    It may be called before or after :meth:`dn.Network.build`.  Attachment and
    the new connection specifications invalidate materialized NetCons, so call
    ``network.initialize(dt)`` before simulation (or ``network.build(dt)`` when
    only inspecting NetCons).  Reinitialization is required if the network had
    already been initialized, because all NetCon runtime queues are rebuilt.

    A caller-supplied ``netstim`` may be used; otherwise a quiet NetStim with
    the canonical number of channels is created on the biological cells'
    device/dtype.  Explicit schedules are loaded by default but can be omitted
    for programmatic stimulation via ``schedule=False``.
    """

    metadata = getattr(network, "zhang2014", None)
    if metadata is None:
        raise AttributeError(
            "attach_windup_afferents expects a network created by "
            "assemble_windup_network or build_windup_network."
        )
    if metadata.afferents_attached:
        raise RuntimeError("Zhang afferents are already attached to this network.")
    if afferent_mode not in _AFFERENT_MODES:
        raise ValueError(
            f"afferent_mode must be one of {_AFFERENT_MODES}; got {afferent_mode!r}."
        )
    if data is not None and data_dir is not None:
        raise ValueError("Provide either data or data_dir, not both.")
    if data is None:
        data = (
            metadata.data
            if data_dir is None
            else load_windup_data(
                data_dir,
                require_shared_afferent_schedules=(afferent_mode == "shared_source"),
            )
        )
    if not _connection_vectors_match(metadata.data, data):
        raise WindUpDataError(
            "The afferent data's connection vectors do not match the vectors "
            "used to assemble the cellular network."
        )
    if afferent_mode == "shared_source" and not data.can_share_afferent_sources:
        raise WindUpDataError(
            "The supplied schedules cannot share artificial source channels; "
            "use afferent_mode='per_connection'."
        )

    dt_ms = metadata.dt_ms if dt_ms is None else float(dt_ms)
    if not np.isfinite(dt_ms) or dt_ms <= 0.0:
        raise ValueError(f"dt_ms must be finite and positive; got {dt_ms!r}.")
    if not np.isclose(dt_ms, metadata.dt_ms, rtol=0.0, atol=1.0e-12):
        raise WindUpDataError(
            "Afferent bridge delays must use the same dt as cellular network "
            f"metadata ({metadata.dt_ms} ms); got {dt_ms} ms."
        )

    expected_n = (
        data.n_afferent_sources
        if afferent_mode == "shared_source"
        else data.n_scheduled_connections
    )
    if netstim is None and network.netstim is None:
        first_population = next(iter(network.populations.values()))
        netstim = _make_netstim(
            expected_n,
            tstop_ms=data.tstop_ms,
            seed=seed,
            device=first_population.device(),
            dtype=first_population.dtype(),
        )
    elif netstim is None:
        netstim = network.netstim

    if network.netstim is None:
        if not hasattr(network, "attach_netstim"):
            raise RuntimeError(
                "This Dendra Network does not support post-construction NetStim "
                "attachment. Apply the bundled Network.attach_netstim patch."
            )
        network.attach_netstim(netstim)
    elif network.netstim is not netstim:
        raise RuntimeError(
            "The Network already has a different NetStim attached. Pass that "
            "same object or construct a fresh cellular network."
        )

    if int(network.netstim.N) != int(expected_n):
        raise WindUpDataError(
            f"{afferent_mode!r} mode requires NetStim.N == {expected_n}; "
            f"got {network.netstim.N}."
        )
    _label_netstim_ranges(network.netstim, afferent_mode=afferent_mode, data=data)

    afferent_connections = connect_windup_network(
        network,
        data,
        subset="afferent",
        afferent_mode=afferent_mode,
        dt_ms=dt_ms,
        trainable_weights=trainable_weights,
        trainable_delays=trainable_delays,
        omit_static_zero_weight_connections=omit_static_zero_weight_connections,
    )
    network.zhang2014 = replace(
        metadata,
        afferent_connections=afferent_connections,
        afferent_mode=afferent_mode,
        external_schedule_lead_ms=dt_ms,
    )
    _set_metadata_aliases(network)

    if schedule:
        schedule_windup_afferents(network, data, dt_ms=dt_ms, clear=True)
    return network


def assemble_windup_network(
    populations: Mapping[str, object],
    *,
    data: WindUpData | None = None,
    data_dir: str | Path | None = None,
    include_afferents: bool = False,
    afferent_mode: Literal["shared_source", "per_connection"] = "shared_source",
    schedule_afferents: bool | None = None,
    dt_ms: float = WINDUP_DT_MS,
    seed: int | None = None,
    trainable_weights: bool = False,
    trainable_delays: bool = False,
    omit_static_zero_weight_connections: bool = True,
    track_netcon_events: bool = False,
    netcon_delay_backend: Literal[
        "dense", "sparse_calendar", "bitpacked_history"
    ] = "dense",
    netcon_train_backend: Literal["dense", "source_history", "auto"] = "auto",
):
    """Assemble the four-cell Zhang network around prebuilt populations.

    By default this returns a *cellular-only* network: SG, SGSCS, EX, and
    T_Cell plus the 66 biological connection-vector rows among them. Source
    threshold crossings are shared per projecting cell.
    No NetStim is constructed and no external events are scheduled.  This is
    the appropriate form for intracellular current injection and other direct
    perturbations.

    Set ``include_afferents=True`` to reproduce the original external drive at
    construction time, or call :func:`attach_windup_afferents` later—even after
    ``Network.build`` has already been called.
    """

    if afferent_mode not in _AFFERENT_MODES:
        raise ValueError(
            f"afferent_mode must be one of {_AFFERENT_MODES}; got {afferent_mode!r}."
        )
    dt_ms = float(dt_ms)
    if not np.isfinite(dt_ms) or dt_ms <= 0.0:
        raise ValueError(f"dt_ms must be finite and positive; got {dt_ms!r}.")
    if data is not None and data_dir is not None:
        raise ValueError("Provide either data or data_dir, not both.")
    if schedule_afferents is None:
        schedule_afferents = bool(include_afferents)
    elif schedule_afferents and not include_afferents:
        raise ValueError(
            "schedule_afferents=True requires include_afferents=True. For a "
            "cellular-only network, attach afferents later with "
            "attach_windup_afferents(...)."
        )
    if data is None:
        data = load_windup_data(
            data_dir,
            require_shared_afferent_schedules=(
                bool(include_afferents) and afferent_mode == "shared_source"
            ),
        )

    populations = _validate_biological_populations(populations)
    spike_source_populations = _install_windup_spike_detectors(populations, data)
    network = dn.Network(
        populations,
        seed=seed,
        track_netcon_events=track_netcon_events,
        netcon_delay_backend=netcon_delay_backend,
        netcon_train_backend=netcon_train_backend,
    )
    cellular_connections = connect_windup_network(
        network,
        data,
        subset="cellular",
        afferent_mode=afferent_mode,
        dt_ms=dt_ms,
        trainable_weights=trainable_weights,
        trainable_delays=trainable_delays,
        omit_static_zero_weight_connections=omit_static_zero_weight_connections,
    )

    network.zhang2014 = WindUpNetworkMetadata(
        data=data,
        cellular_connections=cellular_connections,
        afferent_connections=(),
        afferent_mode=None,
        dt_ms=dt_ms,
        tstop_ms=float(data.tstop_ms),
        external_event_semantics=WINDUP_EXTERNAL_EVENT_SEMANTICS,
        external_schedule_lead_ms=dt_ms,
        source_cell_map=WINDUP_GLOBAL_CELL_MAP,
        synapse_bank_order=WINDUP_SYNAPSE_BANK_ORDER,
        spike_detector_alias=WINDUP_SPIKE_DETECTOR_ALIAS,
        spike_pre_var=WINDUP_SPIKE_PRE_VAR,
        spike_threshold_mV=WINDUP_SPIKE_THRESHOLD_MV,
        spike_source_populations=spike_source_populations,
        spike_detector_consumption_lag_steps=(
            WINDUP_SPIKE_DETECTOR_CONSUMPTION_LAG_STEPS
        ),
    )
    _set_metadata_aliases(network)

    if include_afferents:
        attach_windup_afferents(
            network,
            data=data,
            afferent_mode=afferent_mode,
            schedule=bool(schedule_afferents),
            dt_ms=dt_ms,
            seed=seed,
            trainable_weights=trainable_weights,
            trainable_delays=trainable_delays,
            omit_static_zero_weight_connections=omit_static_zero_weight_connections,
        )
    return network


def build_windup_network(
    *,
    data: WindUpData | None = None,
    data_dir: str | Path | None = None,
    include_afferents: bool = False,
    afferent_mode: Literal["shared_source", "per_connection"] = "shared_source",
    schedule_afferents: bool | None = None,
    dt_ms: float = WINDUP_DT_MS,
    celsius: float = ZHANG_CELSIUS,
    v_init: float = -65.0,
    cascale: float = 1.0,
    synapse_layout: Literal["banked", "independent"] = "banked",
    integrator=None,
    device=None,
    dtype=None,
    seed: int | None = None,
    trainable_weights: bool = False,
    trainable_delays: bool = False,
    omit_static_zero_weight_connections: bool = True,
    track_netcon_events: bool = False,
    netcon_delay_backend: Literal[
        "dense", "sparse_calendar", "bitpacked_history"
    ] = "dense",
    netcon_train_backend: Literal["dense", "source_history", "auto"] = "auto",
):
    """Build the canonical four-cell Zhang network.

    The default result contains only the biological cells and their recurrent
    connectivity; ``network.netstim is None``.  Synapse banks default to
    ``synapse_layout="banked"``, which preserves one independent state slot per
    original HOC synlist entry while grouping contacts by receptor class. This
    makes direct somatic or
    dendritic current injection the uncomplicated default::

        from dendra import units as U

        net = build_windup_network(dtype=torch.float64)
        assert net.netstim is None
        net.T_Cell.slice("soma", loc=0.5).inject(
            dn.mono_rect(amp=1.0 * U.nA, delay=10.0, pw=1.0)
        )
        net.initialize(net.zhang2014.dt_ms)

    To include the original vector-driven afferents immediately, pass
    ``include_afferents=True``.  Alternatively, attach them later with
    :func:`attach_windup_afferents`.

    ``synapse_layout="banked"`` is the default optimized representation: one
    vectorized Dendra mechanism per receptor class, with independent local state
    slots for every original HOC point process. Use ``"independent"`` to keep
    the earlier one-renamed-mechanism-per-contact representation.
    """

    if afferent_mode not in _AFFERENT_MODES:
        raise ValueError(
            f"afferent_mode must be one of {_AFFERENT_MODES}; got {afferent_mode!r}."
        )
    if synapse_layout not in ZHANG_SYNAPSE_LAYOUTS:
        raise ValueError(
            f"synapse_layout must be one of {ZHANG_SYNAPSE_LAYOUTS}; "
            f"got {synapse_layout!r}."
        )
    dt_ms = float(dt_ms)
    if not np.isfinite(dt_ms) or dt_ms <= 0.0:
        raise ValueError(f"dt_ms must be finite and positive; got {dt_ms!r}.")
    if data is not None and data_dir is not None:
        raise ValueError("Provide either data or data_dir, not both.")
    if schedule_afferents is not None and schedule_afferents and not include_afferents:
        raise ValueError(
            "schedule_afferents=True requires include_afferents=True. For a "
            "cellular-only network, attach afferents later with "
            "attach_windup_afferents(...)."
        )
    if data is None:
        data = load_windup_data(
            data_dir,
            require_shared_afferent_schedules=(
                bool(include_afferents) and afferent_mode == "shared_source"
            ),
        )

    tree_kwargs = {
        "v_init": float(v_init),
        "celsius": float(celsius),
    }
    if integrator is not None:
        tree_kwargs["integrator"] = integrator
    if device is not None:
        tree_kwargs["device"] = device
    if dtype is not None:
        tree_kwargs["dtype"] = dtype

    populations = build_cell_populations(
        n_sg=1,
        n_sg_scs=1,
        n_ex=1,
        n_wdr=1,
        sg_ampa_counts=(15, 15),
        sg_scs_ampa_counts=(15, 0),
        cascale=float(cascale),
        insert_synapses=True,
        synapse_layout=synapse_layout,
        **tree_kwargs,
    )

    return assemble_windup_network(
        populations,
        data=data,
        include_afferents=include_afferents,
        afferent_mode=afferent_mode,
        schedule_afferents=schedule_afferents,
        dt_ms=dt_ms,
        seed=seed,
        trainable_weights=trainable_weights,
        trainable_delays=trainable_delays,
        omit_static_zero_weight_connections=omit_static_zero_weight_connections,
        track_netcon_events=track_netcon_events,
        netcon_delay_backend=netcon_delay_backend,
        netcon_train_backend=netcon_train_backend,
    )


# A concise alias matching the style of other Dendra model repositories.
zhang2014_windup = build_windup_network


__all__ = [
    "WINDUP_DT_MS",
    "WINDUP_SPIKE_THRESHOLD_MV",
    "WINDUP_SPIKE_DETECTOR_ALIAS",
    "WINDUP_SPIKE_PRE_VAR",
    "WINDUP_SPIKE_SOURCE_SECTION",
    "WINDUP_SPIKE_SOURCE_LOC",
    "WINDUP_SPIKE_DETECTOR_CONSUMPTION_LAG_STEPS",
    "WINDUP_TSTOP_MS",
    "WINDUP_N_AFFERENT_SOURCES",
    "WINDUP_EXTERNAL_EVENT_SEMANTICS",
    "WINDUP_GLOBAL_CELL_MAP",
    "WINDUP_SYNAPSE_BANK_ORDER",
    "WindUpData",
    "WindUpConnection",
    "WindUpSynapseTarget",
    "WindUpNetworkMetadata",
    "WindUpDataError",
    "load_windup_data",
    "synapse_aliases",
    "synapse_targets",
    "schedule_windup_afferents",
    "attach_windup_afferents",
    "connect_windup_network",
    "assemble_windup_network",
    "build_windup_network",
    "zhang2014_windup",
]
