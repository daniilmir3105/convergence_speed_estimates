# -*- coding: utf-8 -*-
"""Enhanced, leakage-free features for the hypodifferential-rate task.

The numerical descent implementation is deliberately reused from the original
experiment. This module changes only the information available to the ML model;
the definition and calculation of J_300 remain unchanged.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from hypodifferential_rate.core import (
    exact_line_search_max_quadratics,
    generate_max_quadratic_instance,
    hypodifferential_descent_pair,
    make_start_points,
    max_quadratic_value,
    quadratic_values_and_gradients,
    run_hypodifferential_descent,
    validate_matrices,
)


# log_l and log_L are intentionally excluded: a common positive scaling of all
# matrices changes neither exact-line-search iterates nor J_300. sqrt_kappa is
# stored separately so validation can test whether it adds anything beyond
# log_kappa, but it is not part of the default feature set.
FEATURE_NAMES = [
    "dimension",
    "n_pieces",
    "log_kappa",
    "eigenvalue_q10",
    "eigenvalue_q25",
    "eigenvalue_q50",
    "eigenvalue_q75",
    "eigenvalue_q90",
    "piece_log_condition_mean",
    "piece_log_condition_std",
    "piece_log_condition_max",
    "trace_over_L_mean",
    "trace_over_L_std",
    "frobenius_over_L_mean",
    "frobenius_over_L_std",
    "pairwise_distance_mean",
    "pairwise_distance_max",
    "commutator_mean",
    "commutator_max",
    "pairwise_relative_log_condition_mean",
    "pairwise_relative_log_condition_std",
    "pairwise_relative_log_condition_max",
    "principal_direction_difference_mean",
    "principal_direction_difference_std",
    "principal_direction_difference_max",
    "start_active_piece_fraction",
    "start_active_entropy",
    "start_dominant_active_share",
    "start_active_margin_mean",
    "start_active_margin_std",
    "start_active_margin_min",
    "initial_pair_norm_mean",
    "initial_pair_norm_std",
    "initial_pair_norm_min",
    "initial_pair_norm_max",
    "initial_direction_norm_mean",
    "initial_direction_norm_std",
    "initial_direction_norm_min",
    "initial_direction_norm_max",
    "initial_a_abs_mean",
    "initial_a_abs_std",
    "first_alpha_log1p_mean",
    "first_alpha_log1p_std",
    "first_alpha_log1p_min",
    "first_alpha_log1p_max",
    "first_relative_value_mean",
    "first_relative_value_std",
    "first_relative_value_min",
    "first_relative_value_max",
    "first_log10_decrease_mean",
    "first_log10_decrease_std",
    "first_log10_decrease_min",
    "first_log10_decrease_max",
    "first_step_norm_mean",
    "first_step_norm_std",
    "first_step_norm_max",
    "first_active_switch_fraction",
    "first_zero_direction_fraction",
]

OPTIONAL_FEATURE_NAMES = ["sqrt_kappa"]
ALL_FEATURE_NAMES = FEATURE_NAMES + OPTIONAL_FEATURE_NAMES


def _summary(values, prefix, include_min=True, include_max=True):
    values = np.asarray(values, dtype=np.float64)
    result = {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
    }
    if include_min:
        result[f"{prefix}_min"] = float(np.min(values))
    if include_max:
        result[f"{prefix}_max"] = float(np.max(values))
    return result


def _start_activity(values):
    n_starts, n_pieces = values.shape
    active = np.argmax(values, axis=1)
    counts = np.bincount(active, minlength=n_pieces).astype(np.float64)
    probabilities = counts / max(float(n_starts), 1.0)
    nonzero = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(nonzero * np.log(nonzero))) / max(
        np.log(n_pieces), 1e-12
    )
    sorted_values = np.sort(values, axis=1)
    scale = np.maximum(np.abs(sorted_values[:, -1]), 1e-300)
    margins = (sorted_values[:, -1] - sorted_values[:, -2]) / scale
    return active, margins, {
        "start_active_piece_fraction": float(np.count_nonzero(counts) / n_pieces),
        "start_active_entropy": entropy,
        "start_dominant_active_share": float(np.max(probabilities)),
        "start_active_margin_mean": float(np.mean(margins)),
        "start_active_margin_std": float(np.std(margins)),
        "start_active_margin_min": float(np.min(margins)),
    }


def count_active_pieces_on_starts(matrices, n_starts=16, seed=43):
    """Cheap generation-time activity check using the exact target starts."""
    matrices = validate_matrices(matrices)
    starts = make_start_points(matrices.shape[1], n_starts=n_starts, seed=seed)
    values = 0.5 * np.einsum(
        "si,mij,sj->sm", starts, matrices, starts, optimize=True
    )
    return int(np.unique(np.argmax(values, axis=1)).size)


def _pairwise_geometry(normalized, eigvectors):
    distances = []
    commutators = []
    relative_log_conditions = []
    principal_differences = []

    for first, second in combinations(range(len(normalized)), 2):
        first_matrix = normalized[first]
        second_matrix = normalized[second]
        distances.append(float(np.linalg.norm(first_matrix - second_matrix)))
        commutator = first_matrix @ second_matrix - second_matrix @ first_matrix
        commutators.append(float(np.linalg.norm(commutator)))

        values, vectors = np.linalg.eigh(first_matrix)
        inverse_root = (vectors * (1.0 / np.sqrt(values))) @ vectors.T
        relative = inverse_root @ second_matrix @ inverse_root
        relative = 0.5 * (relative + relative.T)
        relative_values = np.linalg.eigvalsh(relative)
        relative_condition = float(
            np.max(relative_values) / max(np.min(relative_values), 1e-300)
        )
        relative_log_conditions.append(float(np.log(relative_condition)))

        first_principal = eigvectors[first, :, -1]
        second_principal = eigvectors[second, :, -1]
        cosine = float(np.clip(abs(first_principal @ second_principal), 0.0, 1.0))
        principal_differences.append(float(np.sqrt(max(0.0, 1.0 - cosine**2))))

    return {
        "pairwise_distance_mean": float(np.mean(distances)),
        "pairwise_distance_max": float(np.max(distances)),
        "commutator_mean": float(np.mean(commutators)),
        "commutator_max": float(np.max(commutators)),
        "pairwise_relative_log_condition_mean": float(
            np.mean(relative_log_conditions)
        ),
        "pairwise_relative_log_condition_std": float(
            np.std(relative_log_conditions)
        ),
        "pairwise_relative_log_condition_max": float(
            np.max(relative_log_conditions)
        ),
        "principal_direction_difference_mean": float(
            np.mean(principal_differences)
        ),
        "principal_direction_difference_std": float(
            np.std(principal_differences)
        ),
        "principal_direction_difference_max": float(
            np.max(principal_differences)
        ),
    }


def _first_step_features(
    normalized,
    starts,
    active_before,
    projection_tolerance,
    projection_max_iterations,
):
    pair_norms = []
    direction_norms = []
    a_absolute = []
    alpha_log1p = []
    relative_values = []
    log10_decreases = []
    step_norms = []
    active_switches = []
    zero_directions = []

    for start_id, x0 in enumerate(starts):
        pair, _, _, _, initial_value = hypodifferential_descent_pair(
            normalized,
            x0,
            initial_weights=None,
            projection_tolerance=projection_tolerance,
            projection_max_iterations=projection_max_iterations,
        )
        direction = pair[1:]
        direction_norm = float(np.linalg.norm(direction))
        pair_norms.append(float(np.linalg.norm(pair)))
        direction_norms.append(direction_norm)
        a_absolute.append(float(abs(pair[0])))

        if direction_norm <= np.finfo(np.float64).eps:
            alpha = 0.0
            candidate = x0
            candidate_value = initial_value
            zero_directions.append(1.0)
        else:
            alpha = exact_line_search_max_quadratics(normalized, x0, direction)
            candidate = x0 + alpha * direction
            candidate_value = max_quadratic_value(normalized, candidate)
            zero_directions.append(0.0)

        ratio = float(candidate_value / max(initial_value, 1e-300))
        ratio = max(ratio, 1e-300)
        candidate_values, _ = quadratic_values_and_gradients(normalized, candidate)
        active_after = int(np.argmax(candidate_values))

        alpha_log1p.append(float(np.log1p(max(alpha, 0.0))))
        relative_values.append(ratio)
        log10_decreases.append(float(np.log10(ratio)))
        step_norms.append(float(alpha * direction_norm))
        active_switches.append(float(active_after != int(active_before[start_id])))

    return {
        **_summary(pair_norms, "initial_pair_norm"),
        **_summary(direction_norms, "initial_direction_norm"),
        **_summary(
            a_absolute, "initial_a_abs", include_min=False, include_max=False
        ),
        **_summary(alpha_log1p, "first_alpha_log1p"),
        **_summary(relative_values, "first_relative_value"),
        **_summary(log10_decreases, "first_log10_decrease"),
        **_summary(step_norms, "first_step_norm", include_min=False),
        "first_active_switch_fraction": float(np.mean(active_switches)),
        "first_zero_direction_fraction": float(np.mean(zero_directions)),
    }


def compute_instance_features(
    matrices,
    n_starts=16,
    seed=43,
    projection_tolerance=1e-11,
    projection_max_iterations=300,
):
    """Calculate only matrix, initial-point and first-step information."""
    matrices = validate_matrices(matrices)
    n_pieces, dimension, _ = matrices.shape
    eigenvalues, eigvectors = np.linalg.eigh(matrices)
    l_value = float(np.min(eigenvalues))
    L_value = float(np.max(eigenvalues))
    kappa = L_value / l_value
    normalized = matrices / L_value
    normalized_eigenvalues = eigenvalues / L_value

    flattened_eigenvalues = normalized_eigenvalues.ravel()
    quantiles = np.quantile(flattened_eigenvalues, [0.10, 0.25, 0.50, 0.75, 0.90])
    piece_log_conditions = np.log(
        np.max(eigenvalues, axis=1) / np.min(eigenvalues, axis=1)
    )
    traces = np.trace(normalized, axis1=1, axis2=2)
    frobenius = np.linalg.norm(normalized, axis=(1, 2))

    starts = make_start_points(dimension, n_starts=n_starts, seed=seed)
    start_values = 0.5 * np.einsum(
        "si,mij,sj->sm", starts, normalized, starts, optimize=True
    )
    active_before, _, activity = _start_activity(start_values)

    features = {
        "dimension": float(dimension),
        "n_pieces": float(n_pieces),
        "log_kappa": float(np.log(kappa)),
        "sqrt_kappa": float(np.sqrt(kappa)),
        "eigenvalue_q10": float(quantiles[0]),
        "eigenvalue_q25": float(quantiles[1]),
        "eigenvalue_q50": float(quantiles[2]),
        "eigenvalue_q75": float(quantiles[3]),
        "eigenvalue_q90": float(quantiles[4]),
        "piece_log_condition_mean": float(np.mean(piece_log_conditions)),
        "piece_log_condition_std": float(np.std(piece_log_conditions)),
        "piece_log_condition_max": float(np.max(piece_log_conditions)),
        "trace_over_L_mean": float(np.mean(traces)),
        "trace_over_L_std": float(np.std(traces)),
        "frobenius_over_L_mean": float(np.mean(frobenius)),
        "frobenius_over_L_std": float(np.std(frobenius)),
        **_pairwise_geometry(normalized, eigvectors),
        **activity,
        **_first_step_features(
            normalized,
            starts,
            active_before,
            projection_tolerance,
            projection_max_iterations,
        ),
    }
    missing = [name for name in ALL_FEATURE_NAMES if name not in features]
    if missing:
        raise RuntimeError(f"Internal error: missing features {missing}")
    if not np.all(np.isfinite([features[name] for name in ALL_FEATURE_NAMES])):
        raise FloatingPointError("Calculated features must all be finite")
    return features


def _tail_slope_and_r2(history, tail_length=100, floor=1e-300):
    if len(history) < 3 or history[0] <= 0.0:
        return np.nan, np.nan
    begin = max(0, len(history) - 1 - tail_length)
    indices = np.arange(begin, len(history), dtype=np.float64)
    values = np.log10(np.maximum(history[begin:] / history[0], floor))
    if len(values) < 3:
        return np.nan, np.nan
    slope, intercept = np.polyfit(indices, values, 1)
    prediction = slope * indices + intercept
    residual = float(np.sum((values - prediction) ** 2))
    total = float(np.sum((values - np.mean(values)) ** 2))
    r2 = 1.0 - residual / total if total > 1e-30 else np.nan
    return float(slope), float(r2)


def evaluate_instance(
    instance_id,
    matrices,
    n_starts=16,
    steps=300,
    stopping_tolerance=1e-10,
    projection_tolerance=1e-11,
    projection_max_iterations=300,
    seed=43,
    keep_trajectories=False,
):
    """Calculate the unchanged J_300 target and enhanced prior features."""
    matrices = validate_matrices(matrices)
    features = compute_instance_features(
        matrices,
        n_starts=n_starts,
        seed=seed,
        projection_tolerance=projection_tolerance,
        projection_max_iterations=projection_max_iterations,
    )
    eigenvalues = np.linalg.eigvalsh(matrices)
    normalized = matrices / float(np.max(eigenvalues))
    starts = make_start_points(matrices.shape[1], n_starts=n_starts, seed=seed)

    per_start_rates = []
    tail_slopes = []
    tail_r2 = []
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
        slope, linearity = _tail_slope_and_r2(result.history)
        tail_slopes.append(slope)
        tail_r2.append(linearity)
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

    # This is exactly the original definition:
    # mean_i(log10(max(f(x_300^i)/f(x_0^i), 1e-300))) / 300.
    J300 = float(np.mean(per_start_rates))
    slopes = np.asarray(tail_slopes, dtype=np.float64)
    linearities = np.asarray(tail_r2, dtype=np.float64)
    row = {
        "instance_id": int(instance_id),
        **features,
        "J300": J300,
        "rho": float(10.0**J300),
        "tail_log10_slope_mean": float(np.nanmean(slopes))
        if np.any(np.isfinite(slopes))
        else np.nan,
        "tail_linearity_r2_mean": float(np.nanmean(linearities))
        if np.any(np.isfinite(linearities))
        else np.nan,
        "effective_iterations_mean": float(np.mean(effective_iterations)),
        "projection_iterations_mean": float(np.mean(projection_iterations)),
        "stationary_fraction": float(np.mean(np.asarray(statuses) == "stationary")),
    }
    return row, trajectories


__all__ = [
    "ALL_FEATURE_NAMES",
    "FEATURE_NAMES",
    "OPTIONAL_FEATURE_NAMES",
    "compute_instance_features",
    "count_active_pieces_on_starts",
    "evaluate_instance",
    "generate_max_quadratic_instance",
    "validate_matrices",
]
