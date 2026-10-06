"""Attach the original Zhang afferents after cellular NetCons were built."""

from __future__ import annotations

import torch

from dendra_models.models.networks.zhang_2014 import (
    attach_windup_afferents,
    build_windup_network,
)


def main() -> None:
    network = build_windup_network(dtype=torch.float64)
    dt = network.zhang2014.dt_ms

    network.build(dt)
    print("Before attachment:")
    print("  built:", network.built)
    print("  NetStim:", network.netstim)
    print("  connections:", len(network.zhang2014.connections))

    attach_windup_afferents(
        network,
        afferent_mode="shared_source",
        schedule=True,
    )

    print("After attachment:")
    print("  built:", network.built)
    print("  NetStim channels:", network.netstim.N)
    print("  connections:", len(network.zhang2014.connections))
    print(
        "  queued events:",
        sum(len(heap) for heap in network.netstim._sched_heaps),
    )

    # Required before run(): attachment and the new connections invalidate the
    # old materialized NetCons and their runtime queues.
    # network.initialize(dt)


if __name__ == "__main__":
    main()
