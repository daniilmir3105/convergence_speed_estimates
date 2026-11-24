# -*- coding: utf-8 -*-
"""Visualize a two-dimensional slice of a maximum of convex quadratics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.colors import ListedColormap

from .core import validate_matrices


# Saving must work even when the local Python installation has no Tcl/Tk GUI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "data" / "instances.npz"
DEFAULT_RESULTS_DIR = BASE_DIR / "results_visualization"


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Plot f(x)=max_j 0.5*x^T*A_j*x as a two-dimensional slice"
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Instance index in an NPZ dataset; by default the first 2D instance",
    )
    parser.add_argument(
        "--axes",
        type=int,
        nargs=2,
        default=(0, 1),
        metavar=("FIRST", "SECOND"),
        help="Coordinates used for a 2D slice",
    )
    parser.add_argument("--limit", type=float, default=2.0)
    parser.add_argument("--grid-size", type=int, default=180)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--raw-scale",
        action="store_true",
        help="Plot original values instead of the experiment's A_j/L scale",
    )
    return parser


def validate_args(args):
    if args.limit <= 0.0:
        raise ValueError("limit must be positive")
    if args.grid_size < 40:
        raise ValueError("grid-size must be at least 40")
    if args.axes[0] == args.axes[1] or min(args.axes) < 0:
        raise ValueError("axes must contain two distinct non-negative indices")


def load_matrices(path, requested_index):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file does not exist: {path}")

    if path.suffix.lower() == ".npy":
        matrices = validate_matrices(np.load(path))
        return matrices, None
    if path.suffix.lower() != ".npz":
        raise ValueError("Expected a .npy or .npz file")

    with np.load(path) as data:
        if "matrices" not in data:
            raise ValueError("NPZ file must contain a 'matrices' array")
        values = data["matrices"]
        if values.ndim == 3:
            return validate_matrices(values), None
        if values.ndim != 4:
            raise ValueError("matrices must be a 3D instance or a 4D dataset")
        if "dimensions" not in data or "n_pieces" not in data:
            raise ValueError("Padded dataset requires dimensions and n_pieces")

        dimensions = data["dimensions"].astype(int)
        n_pieces = data["n_pieces"].astype(int)
        if requested_index is None:
            two_dimensional = np.flatnonzero(dimensions == 2)
            index = int(two_dimensional[0]) if len(two_dimensional) else 0
        else:
            index = int(requested_index)
        if not 0 <= index < len(values):
            raise IndexError(f"index must belong to [0, {len(values) - 1}]")

        dimension = int(dimensions[index])
        pieces = int(n_pieces[index])
        matrices = values[index, :pieces, :dimension, :dimension]
        return validate_matrices(matrices), index


def evaluate_slice(matrices, first_axis, second_axis, limit, grid_size):
    dimension = matrices.shape[1]
    if first_axis >= dimension or second_axis >= dimension:
        raise ValueError(
            f"axes must be smaller than the instance dimension {dimension}"
        )

    coordinates = np.linspace(-limit, limit, grid_size)
    horizontal, vertical = np.meshgrid(coordinates, coordinates)
    points = np.zeros((grid_size * grid_size, dimension), dtype=np.float64)
    points[:, first_axis] = horizontal.ravel()
    points[:, second_axis] = vertical.ravel()
    component_values = 0.5 * np.einsum(
        "pi,mij,pj->pm", points, matrices, points, optimize=True
    )
    function_values = np.max(component_values, axis=1).reshape(horizontal.shape)
    active_components = np.argmax(component_values, axis=1).reshape(horizontal.shape)
    return horizontal, vertical, function_values, active_components


def plot_function(
    matrices,
    first_axis,
    second_axis,
    limit,
    grid_size,
    instance_index,
    output,
    normalized_scale,
):
    horizontal, vertical, values, active = evaluate_slice(
        matrices, first_axis, second_axis, limit, grid_size
    )
    figure = plt.figure(figsize=(15, 6))

    surface_axis = figure.add_subplot(1, 2, 1, projection="3d")
    surface = surface_axis.plot_surface(
        horizontal,
        vertical,
        values,
        cmap="viridis",
        linewidth=0,
        antialiased=True,
        alpha=0.95,
    )
    surface_axis.scatter([0.0], [0.0], [0.0], color="red", s=55, label="minimum")
    surface_axis.set_xlabel(f"x[{first_axis}]")
    surface_axis.set_ylabel(f"x[{second_axis}]")
    surface_axis.set_zlabel("f(x)")
    surface_axis.set_title("Surface of the maximum of quadratics")
    surface_axis.legend()
    figure.colorbar(surface, ax=surface_axis, shrink=0.65, pad=0.10)

    contour_axis = figure.add_subplot(1, 2, 2)
    filled = contour_axis.contourf(
        horizontal, vertical, values, levels=35, cmap="viridis"
    )
    colors = plt.get_cmap("tab10").colors
    region_cmap = ListedColormap(colors[: min(len(colors), matrices.shape[0])])
    contour_axis.imshow(
        active,
        extent=(-limit, limit, -limit, limit),
        origin="lower",
        cmap=region_cmap,
        alpha=0.18,
        interpolation="nearest",
        aspect="auto",
    )
    if matrices.shape[0] > 1:
        contour_axis.contour(
            horizontal,
            vertical,
            active,
            levels=np.arange(matrices.shape[0] - 1) + 0.5,
            colors="white",
            linewidths=0.8,
            alpha=0.85,
        )
    contour_axis.scatter([0.0], [0.0], color="red", s=55, label="minimum")
    contour_axis.set_xlabel(f"x[{first_axis}]")
    contour_axis.set_ylabel(f"x[{second_axis}]")
    contour_axis.set_title("Level sets and active-component regions")
    contour_axis.set_aspect("equal")
    contour_axis.legend()
    figure.colorbar(filled, ax=contour_axis, label="f(x)")

    slice_note = ""
    if matrices.shape[1] > 2:
        slice_note = "; all other coordinates are fixed at zero"
    scale_note = "; matrices normalized by global L" if normalized_scale else ""
    index_note = "external instance" if instance_index is None else f"instance {instance_index}"
    figure.suptitle(
        f"f(x) = max_j 0.5 x^T A_j x, {index_note}, "
        f"{matrices.shape[0]} components{slice_note}{scale_note}"
    )
    figure.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    matrices, instance_index = load_matrices(args.input, args.index)
    normalized_scale = not args.raw_scale
    if normalized_scale:
        global_L = float(np.max(np.linalg.eigvalsh(matrices)))
        matrices = matrices / global_L
    if args.output is None:
        suffix = "external" if instance_index is None else str(instance_index)
        output = DEFAULT_RESULTS_DIR / f"function_instance_{suffix}.png"
    else:
        output = args.output

    plot_function(
        matrices=matrices,
        first_axis=args.axes[0],
        second_axis=args.axes[1],
        limit=args.limit,
        grid_size=args.grid_size,
        instance_index=instance_index,
        output=output,
        normalized_scale=normalized_scale,
    )
    print(f"Saved visualization: {output.resolve()}")
    print("The algorithm minimizes the plotted maximum; its minimum is x=0.")


if __name__ == "__main__":
    main()
