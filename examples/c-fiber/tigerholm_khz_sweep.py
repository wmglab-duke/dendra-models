#!/usr/bin/env python3
"""Run a kHz-frequency/amplitude sweep with the Tigerholm 2014 C-fiber model."""

from __future__ import annotations

import itertools
from pathlib import Path

import dendra as dn
from dendra.units import Hz, kHz, mA, mm, ms, nA, um
from dendra_models.models import Tigerholm2014

import numpy as np
import pandas as pd
import torch


# Set this to True to record selected membrane voltages to HDF5. Voltage
# recording adds runtime and can produce a large file.
RECORD_VOLTAGE = False

FREQUENCIES_KHZ = (1, 2, 5, 10, 20, 30, 40, 50, 60, 80, 100)
AMPLITUDES_MA = tuple(np.arange(0.0, 26.1, 1.0))

MODEL_NAME = "tigerholm2014"
DIAMETER = 1.0 * um
LENGTH = 40.0 * mm
DX = 10.0 * um

TSTOP = 2000 * ms
DT = 0.001 * ms
STIM_START = 200 * ms
STIM_STOP = 610 * ms
CHUNK_LENGTH = 1000

TEST_PULSE_FREQUENCY = 10 * Hz
TEST_PULSE_START = 15 * ms


def make_parameter_grid() -> list[tuple[float, float]]:
    """Return ``(frequency_kHz, amplitude_mA)`` pairs for the sweep."""
    return list(itertools.product(FREQUENCIES_KHZ, AMPLITUDES_MA))


def make_model(n_simulations: int) -> Tigerholm2014:
    """Create one identical fiber per sweep point."""
    return Tigerholm2014(
        [DIAMETER] * n_simulations,
        L=LENGTH,
        dx=DX,
    )


def run_sweep(
    model: Tigerholm2014,
    parameters: list[tuple[float, float]],
    *,
    record_voltage: bool = RECORD_VOLTAGE,
) -> np.ndarray:
    """Run the sweep and return a ``[time, simulation, site]`` spike raster."""
    if len(parameters) != model.n_ax:
        raise ValueError(
            f"Expected one parameter pair per fiber ({model.n_ax}); "
            f"received {len(parameters)}."
        )

    # This is an inference workload; keep autograd disabled during the long run.
    model.eval()

    sweep = torch.as_tensor(
        parameters,
        device=model.device(),
        dtype=model.dtype(),
    )

    # A final singleton component axis is important here. Modern dn.sin treats
    # a 1-D tensor as oscillator components to sum; [simulation, 1] creates one
    # independent sinusoid per fiber instead.
    extracellular_stimulus = dn.sin(
        freq=sweep[:, 0:1] * kHz,
        amp=sweep[:, 1:2] * mA,
        delay=STIM_START,
        off=STIM_STOP,
    )

    test_pulse = dn.mono_rect(amp=1 * nA, pw=1 * ms).repeat(
        TEST_PULSE_FREQUENCY,
        delay=TEST_PULSE_START,
    )
    model.csl(0.1).inject(test_pulse)
    model.initialize()

    # The point-source field is in mV/mA and the temporal stimulus is in mA.
    extracellular_field = dn.isotropic_point(
        z=200.0 * um,
        rhoe=100 / 1.79,
    )(model)

    raster = dn.callbacks.Raster(node_check=model.c(0.9), dt=DT)
    callbacks = [raster]

    voltage_recorder = None
    if record_voltage:
        voltage_nodes = model.c(
            0.1,
            0.4,
            0.5,
            0.501,
            0.502,
            0.505,
            0.51,
            0.55,
            0.6,
            0.65,
            0.7,
            0.9,
        )
        voltage_recorder = dn.callbacks.Recorder(
            ["v"],
            node_indices=voltage_nodes,
        ).set_hdf5(
            f"{MODEL_NAME}_voltage.h5",
            cache_every=10_000,
        )
        callbacks.append(voltage_recorder)

    try:
        model.longrun(
            tstop=TSTOP,
            dt=DT,
            extra=(extracellular_field, extracellular_stimulus),
            chunklength=CHUNK_LENGTH,
            callbacks=callbacks,
            progressbar=True,
        )
    finally:
        if voltage_recorder is not None:
            voltage_recorder.close()

    return raster.numpy()


def make_results(
    parameters: list[tuple[float, float]],
    raster: np.ndarray,
) -> pd.DataFrame:
    """Convert a spike raster into one summary row per sweep point."""
    rows = []

    for simulation, (frequency_khz, amplitude_ma) in enumerate(parameters):
        # Raster callbacks run after each completed solver step, so sample zero
        # corresponds to DT rather than t=0.
        spike_steps = np.flatnonzero(raster[:, simulation, 0])
        spike_times = (spike_steps + 1) * DT
        during_stimulus = (spike_times > STIM_START) & (spike_times < STIM_STOP)

        rows.append(
            {
                "all spike times (ms)": spike_times.tolist(),
                "# APs during kHz": int(np.count_nonzero(during_stimulus)),
                "freq (kHz)": float(frequency_khz),
                "amplitude (mA)": float(amplitude_ma),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parameters = make_parameter_grid()
    model = make_model(len(parameters))
    raster = run_sweep(model, parameters)
    results = make_results(parameters, raster)

    output_path = Path(f"{MODEL_NAME}_khz_df.pkl")
    results.to_pickle(output_path)
    print(f"Saved {len(results)} sweep results to {output_path}")


if __name__ == "__main__":
    main()
