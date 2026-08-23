"""Compare batched S-MF kHz simulations against precomputed NEURON results."""

from __future__ import annotations

import argparse
from pathlib import Path

import dendra as dn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from dendra.units import Hz, ms, nA
from dendra_models.models.cells.peripheral import SMF
from matplotlib.lines import Line2D


FIELD_DIRECTORY = Path("fields")
GROUND_TRUTH_DIRECTORY = Path("ground_truth")
OUTPUT_PATH = Path("khz_stim_example.png")

STIMULATING_CONTACT = 0
NODES = 101
FIELD_OFFSET_UM = 37_500

FREQUENCIES_KHZ = (1, 2, 5, 10)
AXON_INDICES = (0, 10, 15, 25, 32, 33, 45)
AMPLITUDES_MA_BY_DIAMETER = {
    5.7: np.arange(0, 10, 0.01),
    8.7: np.arange(0, 5, 0.01),
    14.0: np.arange(0, 2, 0.01),
}

TSTOP = 100 * ms
DT = 0.001 * ms

AP_THRESHOLD_MV = -20.0
AP_COUNT_START = 50 * ms

INTRA_AMPLITUDE = 2.0 * nA
INTRA_PULSE_WIDTH = 0.1 * ms
INTRA_FREQUENCY = 100.0 * Hz
INTRA_DELAY = 50.0 * ms

DIAMETER_TITLES = (
    r"5.7 $\mu m$",
    r"8.7 $\mu m$",
    r"14.0 $\mu m$",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chunks",
        type=int,
        default=1000,
        help="Number of chunks used to cover the simulation.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="PyTorch device for the simulation (default: cuda).",
    )
    return parser.parse_args()


def deltax(diameter: float) -> float:
    """Return the S-MF internodal spacing in micrometres."""
    return -8.215284e00 * diameter**2 + 2.724201e02 * diameter - 7.802411e02


def make_interpolator(field_directory: Path = FIELD_DIRECTORY):
    """Load the selected contact field and construct its 1-D interpolator."""
    field_data = np.load(field_directory / f"{STIMULATING_CONTACT}.npy")
    field_x = np.load(field_directory / "fiber_zs.npy")
    return dn.precomputed_interpolate_1d(field_data * 1000, field_x).float()


def field_at_nodes(
    interpolator,
    diameter: float,
    axon_index: int,
    *,
    nodes: int = NODES,
    offset: float = FIELD_OFFSET_UM,
) -> np.ndarray:
    """Interpolate one axon's extracellular basis onto its model nodes."""
    half_length = (deltax(diameter) * (nodes - 1)) / 2
    positions = torch.linspace(-half_length, half_length, nodes) + offset
    field = interpolator.interp(positions, indices=[axon_index])
    return field.detach().cpu().numpy()


def build_sweep_cases(
    field_directory: Path = FIELD_DIRECTORY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build amplitude-scaled fields, diameters, and diameter split indices."""
    interpolator = make_interpolator(field_directory)
    field_blocks = []
    diameter_blocks = []
    split_lengths = []

    # Ordering is significant downstream:
    # diameter -> axon index -> amplitude.
    for diameter, amplitudes in AMPLITUDES_MA_BY_DIAMETER.items():
        split_length = 0
        for axon_index in AXON_INDICES:
            basis = field_at_nodes(interpolator, diameter, axon_index)
            block = np.einsum("i,j->ij", amplitudes, basis)
            field_blocks.append(block)
            diameter_blocks.append(np.full(len(block), diameter))
            split_length += len(block)
        split_lengths.append(split_length)

    fields = np.vstack(field_blocks)
    diameters = np.concatenate(diameter_blocks)
    diameter_splits = np.cumsum(split_lengths)
    return fields, diameters, diameter_splits


def run_simulation(
    base_fields: np.ndarray,
    base_diameters: np.ndarray,
    *,
    chunks: int,
    device: str | torch.device = "cuda",
) -> np.ndarray:
    """Run all frequency blocks and return AP counts in sweep order."""
    if chunks <= 0:
        raise ValueError("chunks must be a positive integer.")

    simulation_device = torch.device(device)
    simulation_dtype = torch.float32

    # Construct on CPU and move the complete model once. The old sequence sent
    # diameters GPU -> CPU during SMF construction, then moved the model to GPU.
    repeated_diameters = np.tile(base_diameters, len(FREQUENCIES_KHZ))
    diameter_tensor = torch.as_tensor(repeated_diameters, dtype=simulation_dtype)
    model = SMF(diameters=diameter_tensor, n_node=NODES).to(
        device=simulation_device,
        dtype=simulation_dtype,
    )

    model_device = model.device()
    model_dtype = model.dtype()

    # Frequency-major layout matches the repeated field and diameter blocks.
    frequencies = np.repeat(FREQUENCIES_KHZ, len(base_fields))[:, None]
    frequency_tensor = torch.as_tensor(
        frequencies,
        device=model_device,
        dtype=model_dtype,
    )
    extracellular_stimulus = dn.sin(
        amp=1.0,
        freq=frequency_tensor,
        delay=0.5 * ms,
    ).to(device=model_device, dtype=model_dtype)

    extracellular_field = torch.as_tensor(
        base_fields,
        device=model_device,
        dtype=model_dtype,
    ).repeat(len(FREQUENCIES_KHZ), 1)

    intracellular_stimulus = dn.mono_rect(
        amp=INTRA_AMPLITUDE,
        pw=INTRA_PULSE_WIDTH,
    ).repeat(
        INTRA_FREQUENCY,
        delay=INTRA_DELAY,
    )
    intracellular_stimulus = intracellular_stimulus.to(
        device=model_device,
        dtype=model_dtype,
    )
    model[:, 5].inject(intracellular_stimulus)

    count = dn.callbacks.APCount(
        node_check=[-5],
        threshold=AP_THRESHOLD_MV,
        t_start_check=AP_COUNT_START,
    )
    count.reset()

    chunklength = int(TSTOP / DT / chunks)
    if chunklength <= 0:
        raise ValueError(
            f"chunks={chunks} produces a zero-length chunk for "
            f"{int(TSTOP / DT)} simulation steps."
        )

    model.initialize()
    model.longrun(
        tstop=TSTOP,
        dt=DT,
        extra=(extracellular_field, extracellular_stimulus),
        chunklength=chunklength,
        callbacks=[count],
        progressbar=True,
    )
    return count.numpy()


def build_comparison_frame(
    all_counts: np.ndarray,
    diameter_splits: np.ndarray,
    ground_truth_directory: Path = GROUND_TRUTH_DIRECTORY,
) -> pd.DataFrame:
    """Combine S-MF predictions and NEURON ground truth in long form."""
    rows = []
    diameter_values = tuple(AMPLITUDES_MA_BY_DIAMETER)
    counts_by_frequency = np.split(all_counts, len(FREQUENCIES_KHZ))

    for frequency, frequency_counts in zip(FREQUENCIES_KHZ, counts_by_frequency):
        counts_by_diameter = np.split(frequency_counts, diameter_splits[:-1])

        for diameter, diameter_counts in zip(diameter_values, counts_by_diameter):
            counts_by_axon = np.split(diameter_counts, len(AXON_INDICES))

            for axon_index, surrogate_counts in zip(AXON_INDICES, counts_by_axon):
                amplitudes = AMPLITUDES_MA_BY_DIAMETER[diameter]
                neuron_counts = np.load(
                    ground_truth_directory
                    / f"{frequency}khz_{diameter}um_2_3_0_{axon_index}.npy"
                ).flatten()

                if len(surrogate_counts) != len(amplitudes):
                    raise ValueError(
                        "Surrogate count length does not match the amplitude sweep: "
                        f"{len(surrogate_counts)} != {len(amplitudes)}."
                    )
                for amplitude, count in zip(amplitudes, surrogate_counts):
                    rows.append(
                        {
                            "frequency": frequency,
                            "a_idx": axon_index,
                            "amplitude": amplitude,
                            "diameter": diameter,
                            "# APs": count[0],
                            "method": "Surrogate",
                        }
                    )

                # Some ground-truth files cover a longer amplitude sweep. As in
                # the original script, use the prefix matching this diameter's
                # configured amplitudes.
                for amplitude, count in zip(amplitudes, neuron_counts):
                    rows.append(
                        {
                            "frequency": frequency,
                            "a_idx": axon_index,
                            "amplitude": amplitude,
                            "diameter": diameter,
                            "# APs": count,
                            "method": "NEURON",
                        }
                    )

    return pd.DataFrame(rows)


def plot_comparison(
    data: pd.DataFrame,
    output_path: Path = OUTPUT_PATH,
) -> None:
    """Recreate the original four-by-three comparison figure."""
    diameter_values = tuple(AMPLITUDES_MA_BY_DIAMETER)
    row_titles = tuple(f"{frequency} kHz" for frequency in FREQUENCIES_KHZ)

    figure, axes = plt.subplots(
        len(FREQUENCIES_KHZ),
        len(diameter_values),
        dpi=200,
        figsize=(5, 5),
        sharex="col",
        sharey=True,
    )

    for column, diameter in enumerate(diameter_values):
        for row, frequency in enumerate(FREQUENCIES_KHZ):
            selection = data[
                (data["diameter"] == diameter) & (data["frequency"] == frequency)
            ]
            sns.lineplot(
                data=selection,
                x="amplitude",
                y="# APs",
                hue="method",
                hue_order=["NEURON", "Surrogate"],
                legend=False,
                palette="colorblind",
                ax=axes[row, column],
                errorbar="sd",
            )
            axes[row, column].set_ylabel("")
            axes[row, column].set_xlabel("")

    label_axis = figure.add_subplot(111, frameon=False)
    label_axis.tick_params(
        labelcolor="none",
        which="both",
        top=False,
        bottom=False,
        left=False,
        right=False,
    )
    label_axis.set_xlabel("Amplitude $(mA)$")
    label_axis.set_ylabel("# APs", labelpad=5)

    for axis, title in zip(axes[0], DIAMETER_TITLES):
        axis.set_title(title, size="medium")

    for axis, title in zip(axes[:, 0], row_titles):
        axis.set_ylabel(title, rotation=45, size="medium", labelpad=35)

    colors = sns.color_palette("colorblind")
    legend_lines = [
        Line2D([0], [0], color=colors[0], lw=2),
        Line2D([0], [0], color=colors[1], lw=2),
    ]
    label_axis.legend(
        legend_lines,
        ["NEURON", "S-MF"],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncols=2,
    )

    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    fields, diameters, diameter_splits = build_sweep_cases()
    all_counts = run_simulation(
        fields,
        diameters,
        chunks=args.chunks,
        device=args.device,
    )
    comparison = build_comparison_frame(all_counts, diameter_splits)
    plot_comparison(comparison)


if __name__ == "__main__":
    main()
