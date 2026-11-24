# -*- coding: utf-8 -*-
"""Core of the component-wise Li/li hypodifferential-rate experiment."""

from __future__ import annotations

import time

import numpy as np

from .algorithm import (
    generate_max_quadratic_instance,
    make_start_points,
    run_hypodifferential_descent,
    validate_matrices,
)


FEATURE_SCHEMA_VERSION = "component_bounds_v1"


def build_feature_names(max_pieces):
    """Return the fixed-width feature schema for up to max_pieces components."""
    if max_pieces < 2:
        raise ValueError("max_pieces must be at least two")
    names = ["dimension", "n_pieces", "log_global_kappa"]
    for slot in range(1, max_pieces + 1):
        suffix = f"{slot:02d}"
        names.extend(
            [
                f"piece_present_{suffix}",
                f"piece_log_li_over_global_l_{suffix}",
                f"piece_log_Li_over_global_l_{suffix}",
                f"piece_log_kappa_{suffix}",
            ]
        )
    return names


def compute_component_features(matrices, max_pieces):
    """Encode every component's li and Li in a permutation-stable vector.

    Components are sorted by decreasing Li and then decreasing li. Logarithms
    are measured relative to the global l, removing a common scale that cannot
    affect exact-line-search iterates. Missing slots are zero-padded and have
    an explicit presence mask.
    """
    matrices = validate_matrices(matrices)
    n_pieces, dimension, _ = matrices.shape
    if n_pieces > max_pieces:
        raise ValueError(
            f"instance has {n_pieces} pieces but max_pieces={max_pieces}"
        )

    eigenvalues = np.linalg.eigvalsh(matrices)
    li_values = eigenvalues[:, 0]
    Li_values = eigenvalues[:, -1]
    global_l = float(np.min(li_values))
    global_L = float(np.max(Li_values))
    order = np.lexsort((-li_values, -Li_values))

    features = {
        "dimension": float(dimension),
        "n_pieces": float(n_pieces),
        "log_global_kappa": float(np.log(global_L / global_l)),
    }
    for slot in range(1, max_pieces + 1):
        suffix = f"{slot:02d}"
        if slot <= n_pieces:
            component = int(order[slot - 1])
            li_value = float(li_values[component])
            Li_value = float(Li_values[component])
            features[f"piece_present_{suffix}"] = 1.0
            features[f"piece_log_li_over_global_l_{suffix}"] = float(
                np.log(li_value / global_l)
            )
            features[f"piece_log_Li_over_global_l_{suffix}"] = float(
                np.log(Li_value / global_l)
            )
            features[f"piece_log_kappa_{suffix}"] = float(
                np.log(Li_value / li_value)
            )
        else:
            features[f"piece_present_{suffix}"] = 0.0
            features[f"piece_log_li_over_global_l_{suffix}"] = 0.0
            features[f"piece_log_Li_over_global_l_{suffix}"] = 0.0
            features[f"piece_log_kappa_{suffix}"] = 0.0

    feature_names = build_feature_names(max_pieces)
    values = np.asarray([features[name] for name in feature_names], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("all component-wise features must be finite")
    return features


def evaluate_instance(
    instance_id,
    matrices,
    max_pieces,
    n_starts=16,
    steps=300,
    stopping_tolerance=1e-10,
    projection_tolerance=1e-11,
    projection_max_iterations=300,
    seed=43,
    keep_trajectories=False,
):
    """Calculate all Li/li features and the unchanged J_300 target."""
    started = time.perf_counter()
    matrices = validate_matrices(matrices)
    features = compute_component_features(matrices, max_pieces=max_pieces)
    global_L = float(np.max(np.linalg.eigvalsh(matrices)))
    normalized = matrices / global_L
    starts = make_start_points(matrices.shape[1], n_starts=n_starts, seed=seed)

    per_start_rates = []
    effective_iterations = []
    projection_iterations = []
    statuses = []
    trajectories = []

    for start_id, x0 in enumerate(starts):
        result = run_hypodifferential_descent(
            normalized,
            x0,
            steps=steps,
            stopping_tolerance=stopping_tolerance,
            projection_tolerance=projection_tolerance,
            projection_max_iterations=projection_max_iterations,
        )
        initial = max(float(result.history[0]), 1e-300)
        ratio = max(float(result.history[-1]) / initial, 1e-300)
        per_start_rates.append(np.log10(ratio) / steps)
        effective_iterations.append(result.effective_iterations)
        projection_iterations.append(result.projection_iterations)
        statuses.append(result.status)

        if keep_trajectories:
            for iteration, value in enumerate(result.history):
                trajectories.append(
                    {
                        "instance_id": int(instance_id),
                        "start_id": int(start_id),
                        "iteration": int(iteration),
                        "relative_gap": max(float(value) / initial, 1e-300),
                    }
                )

    # Exactly the established finite-horizon target:
    # J_300 = sum_i log10(max(f(x_300^i)/f(x_0^i), 1e-300)) / (16*300).
    J300 = float(np.mean(per_start_rates))
    row = {
        "instance_id": int(instance_id),
        **features,
        "J300": J300,
        "rho": float(10.0**J300),
        "effective_iterations_mean": float(np.mean(effective_iterations)),
        "projection_iterations_mean": float(np.mean(projection_iterations)),
        "stationary_fraction": float(np.mean(np.asarray(statuses) == "stationary")),
        "evaluation_seconds": float(time.perf_counter() - started),
    }
    return row, trajectories


__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "build_feature_names",
    "compute_component_features",
    "evaluate_instance",
    "generate_max_quadratic_instance",
    "validate_matrices",
]
