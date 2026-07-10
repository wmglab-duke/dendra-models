"""Build and inspect the cellular-first Zhang Wind-Up network."""

from __future__ import annotations

import dendra as dn
from dendra import units as U
import torch

from zhang2014_dendra import build_windup_network


def main() -> None:
    network = build_windup_network(
        dtype=torch.float64,
        celsius=36.0,
        dt_ms=0.0125,
    )

    print("Populations:")
    for name, population in network.populations.items():
        print(f"  {name:7s} shape={tuple(population.shape)}")

    metadata = network.zhang2014
    print(f"Cellular connections: {len(metadata.cellular_connections)}")
    print(f"Afferent connections: {len(metadata.afferent_connections)}")
    print(f"NetStim attached: {network.netstim is not None}")
    print(f"dt: {metadata.dt_ms} ms")
    print("Shared spike sources:", metadata.spike_source_populations)
    print("Spike pre_var:", metadata.spike_pre_var)
    print("Threshold:", metadata.spike_threshold_mV, "mV")
    print(
        "First biological delay (published/runtime):",
        metadata.cellular_connections[0].delay_ms,
        metadata.cellular_connections[0].runtime_delay_ms,
        "ms",
    )

    # A single NEURON-like point current injection at WDR soma(0.5).
    network.T_Cell.slice("soma", loc=0.5).inject(
        dn.mono_rect(amp=1.0 * U.nA, delay=10.0, pw=1.0)
    )

    # Full integration requires a compatible Dendra tree solver.
    # network.initialize(metadata.dt_ms)
    # network.run(100.0, progressbar=True)


if __name__ == "__main__":
    main()
