"""Modify Zhang cells before assembling the cellular network."""

from __future__ import annotations

from zhang2014_dendra import assemble_windup_network, build_cell_populations
import torch


def main() -> None:
    populations = build_cell_populations(
        n_sg=1,
        n_sg_scs=1,
        n_ex=1,
        n_wdr=1,
        sg_ampa_counts=(15, 15),
        sg_scs_ampa_counts=(15, 0),
        insert_synapses=True,
        celsius=36.0,
        v_init=-65.0,
        dtype=torch.float64,
    )

    populations["T_Cell"].slice("soma").label("recording_site")

    network = assemble_windup_network(populations, dt_ms=0.0125)
    print("Cellular connections:", len(network.zhang2014.connections))
    print("NetStim attached:", network.netstim is not None)
    print(
        "WDR persistent sodium gbar:",
        network.T_Cell.mech.iNaP.gnabar_iNaP_soma,
    )


if __name__ == "__main__":
    main()
