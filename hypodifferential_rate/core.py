# -*- coding: utf-8 -*-
"""Numerical core for maxima of convex quadratic functions.

Each problem has the form

    f(x) = max_j 0.5 * x.T @ A_j @ x,

where every A_j is symmetric positive definite. Hence x*=0 and f*=0 are
known exactly. The normalized hypodifferential is

    C(x) = conv{(q_j(x)-f(x), grad q_j(x))},

and the descent pair [a, g] is minus the projection of zero onto C(x).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


FEATURE_NAMES = [
    "dimension",
    "n_pieces",
    "log_l",
    "log_L",
    "log_kappa",
    "sqrt_kappa",
    "trace_over_L_mean",
    "trace_over_L_std",
    "frobenius_over_L_mean",
    "frobenius_over_L_std",
    "pairwise_distance_mean",
    "pairwise_distance_max",
    "commutator_mean",
    "commutator_max",
    "active_piece_fraction",
    "active_entropy",
    "dominant_active_share",
    "active_margin_mean",
]


@dataclass
class RunResult:
    history: np.ndarray
    effective_iterations: int
    status: str
    projection_iterations: int
    final_point: np.ndarray


def validate_matrices(matrices):
    values = np.asarray(matrices, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("matrices must have shape (n_pieces, dimension, dimension)")
    if values.shape[0] < 2 or values.shape[1] < 2:
        raise ValueError("at least two matrices of dimension >= 2 are required")
    if not np.all(np.isfinite(values)):
        raise ValueError("matrices must contain only finite values")
    symmetric = 0.5 * (values + np.swapaxes(values, 1, 2))
    if not np.allclose(values, symmetric, rtol=1e-10, atol=1e-12):
        raise ValueError("all quadratic matrices must be symmetric")
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if np.min(eigenvalues) <= 0.0:
        raise ValueError("all quadratic matrices must be positive definite")
    return symmetric


def quadratic_values_and_gradients(matrices, x):
    gradients = np.einsum("mij,j->mi", matrices, x, optimize=True)
    values = 0.5 * np.einsum("mi,i->m", gradients, x, optimize=True)
    return values, gradients


def max_quadratic_value(matrices, x):
    values, _ = quadratic_values_and_gradients(matrices, x)
    return float(np.max(values))


def project_origin_to_convex_hull(
    points,
    initial_weights=None,
    tolerance=1e-11,
    max_iterations=300,
):
    """Approximate the minimum-norm point in a convex hull by Gilbert/FW.

    The returned Frank-Wolfe gap certifies the projection accuracy. Reusing
    weights from the previous optimization step considerably reduces work.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or len(points) < 1:
        raise ValueError("points must be a non-empty two-dimensional array")

    n_points = len(points)
    if initial_weights is None:
        index = int(np.argmin(np.einsum("ij,ij->i", points, points)))
        weights = np.zeros(n_points, dtype=np.float64)
        weights[index] = 1.0
    else:
        weights = np.asarray(initial_weights, dtype=np.float64).copy()
        if weights.shape != (n_points,) or np.any(weights < 0.0):
            raise ValueError("initial_weights must belong to the simplex")
        total = float(np.sum(weights))
        if not np.isfinite(total) or total <= 0.0:
            raise ValueError("initial_weights must have a positive finite sum")
        weights /= total

    projection = weights @ points
    # The hull contracts as x approaches the minimizer.  An absolute scale
    # here would make the projection artificially inaccurate near zero and
    # could stop the outer method too early.  The squared diameter scale
    # keeps the certificate invariant under a common scaling of all points.
    point_scale = max(
        float(np.max(np.einsum("ij,ij->i", points, points))),
        np.finfo(np.float64).tiny,
    )
    gap = np.inf
    for iteration in range(1, max_iterations + 1):
        dot_products = points @ projection
        vertex = int(np.argmin(dot_products))
        norm_squared = float(projection @ projection)
        gap = norm_squared - float(dot_products[vertex])
        if gap <= tolerance * point_scale:
            return projection, weights, iteration, max(gap, 0.0)

        direction = points[vertex] - projection
        denominator = float(direction @ direction)
        if denominator <= np.finfo(np.float64).tiny:
            return projection, weights, iteration, max(gap, 0.0)
        step = float(np.clip(-(projection @ direction) / denominator, 0.0, 1.0))
        weights *= 1.0 - step
        weights[vertex] += step
        projection += step * direction

    return projection, weights, max_iterations, max(float(gap), 0.0)


def hypodifferential_descent_pair(
    matrices,
    x,
    initial_weights=None,
    projection_tolerance=1e-11,
    projection_max_iterations=300,
):
    values, gradients = quadratic_values_and_gradients(matrices, x)
    function_value = float(np.max(values))
    points = np.column_stack([values - function_value, gradients])
    projection, weights, iterations, gap = project_origin_to_convex_hull(
        points,
        initial_weights=initial_weights,
        tolerance=projection_tolerance,
        max_iterations=projection_max_iterations,
    )
    descent_pair = -projection
    return descent_pair, weights, iterations, gap, function_value


def _positive_quadratic_roots(a, b, c, tolerance=1e-13):
    scale = max(abs(a), abs(b), abs(c), 1.0)
    if abs(a) <= tolerance * scale:
        if abs(b) <= tolerance * scale:
            return []
        root = -c / b
        return [root] if root > 0.0 and np.isfinite(root) else []

    discriminant = b * b - 4.0 * a * c
    if discriminant < -tolerance * scale * scale:
        return []
    discriminant = max(discriminant, 0.0)
    root_disc = np.sqrt(discriminant)
    roots = [(-b - root_disc) / (2.0 * a), (-b + root_disc) / (2.0 * a)]
    return [root for root in roots if root > 0.0 and np.isfinite(root)]


def exact_line_search_max_quadratics(matrices, x, direction):
    """Exactly minimize the upper envelope of univariate quadratics.

    Along a line every component is a convex quadratic. The minimum of their
    upper envelope is attained at zero, at a component vertex, or at an
    intersection of two components. Evaluating all such points is therefore
    sufficient and avoids iterative scalar optimization.
    """
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= np.finfo(np.float64).eps:
        return 0.0

    Ax = np.einsum("mij,j->mi", matrices, x, optimize=True)
    Ad = np.einsum("mij,j->mi", matrices, direction, optimize=True)
    coefficient_2 = 0.5 * np.einsum("mi,i->m", Ad, direction, optimize=True)
    coefficient_1 = np.einsum("mi,i->m", Ax, direction, optimize=True)
    coefficient_0 = 0.5 * np.einsum("mi,i->m", Ax, x, optimize=True)

    candidates = [0.0]
    vertices = -coefficient_1 / np.maximum(2.0 * coefficient_2, 1e-300)
    candidates.extend(vertices[(vertices > 0.0) & np.isfinite(vertices)].tolist())

    for first, second in combinations(range(len(matrices)), 2):
        candidates.extend(
            _positive_quadratic_roots(
                coefficient_2[first] - coefficient_2[second],
                coefficient_1[first] - coefficient_1[second],
                coefficient_0[first] - coefficient_0[second],
            )
        )

    candidates = np.asarray(candidates, dtype=np.float64)
    candidates = candidates[
        np.isfinite(candidates) & (candidates >= 0.0) & (candidates <= 1e12)
    ]
    candidates = np.unique(candidates)
    envelope = np.max(
        coefficient_2[None, :] * candidates[:, None] ** 2
        + coefficient_1[None, :] * candidates[:, None]
        + coefficient_0[None, :],
        axis=1,
    )
    return float(candidates[int(np.argmin(envelope))])


def run_hypodifferential_descent(
    matrices,
    x0,
    steps=300,
    stopping_tolerance=1e-10,
    projection_tolerance=1e-11,
    projection_max_iterations=300,
):
    matrices = validate_matrices(matrices)
    x = np.asarray(x0, dtype=np.float64).copy()
    if x.shape != (matrices.shape[1],) or not np.all(np.isfinite(x)):
        raise ValueError("x0 must be a finite vector matching matrix dimension")
    if steps < 1:
        raise ValueError("steps must be positive")

    history = [max_quadratic_value(matrices, x)]
    weights = None
    projection_iterations = 0
    status = "max_steps"

    for _ in range(steps):
        descent_pair, weights, iterations, _, current_value = (
            hypodifferential_descent_pair(
                matrices,
                x,
                initial_weights=weights,
                projection_tolerance=projection_tolerance,
                projection_max_iterations=projection_max_iterations,
            )
        )
        projection_iterations += iterations
        if float(np.linalg.norm(descent_pair)) <= stopping_tolerance:
            status = "stationary"
            break

        direction = descent_pair[1:]
        if float(np.linalg.norm(direction)) <= stopping_tolerance:
            status = "zero_direction"
            break
        alpha = exact_line_search_max_quadratics(matrices, x, direction)
        if alpha <= np.finfo(np.float64).eps:
            status = "line_search_stalled"
            break

        candidate = x + alpha * direction
        candidate_value = max_quadratic_value(matrices, candidate)
        if candidate_value > current_value + 1e-10 * max(1.0, current_value):
            raise FloatingPointError("exact line search produced an increasing step")
        x = candidate
        history.append(candidate_value)

    return RunResult(
        history=np.asarray(history, dtype=np.float64),
        effective_iterations=len(history) - 1,
        status=status,
        projection_iterations=projection_iterations,
        final_point=x,
    )


def _orthogonal_matrix(dimension, rng):
    matrix = rng.normal(size=(dimension, dimension))
    q, r = np.linalg.qr(matrix)
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def generate_max_quadratic_instance(
    dimension,
    n_pieces,
    l_value,
    kappa,
    seed,
):
    if dimension < 2 or n_pieces < 2:
        raise ValueError("dimension and n_pieces must be at least two")
    if not 0.0 < l_value or not 1.0 < kappa:
        raise ValueError("expected l_value > 0 and kappa > 1")

    rng = np.random.default_rng(seed)
    L_value = l_value * kappa
    matrices = []
    for piece in range(n_pieces):
        eigenvalues = np.exp(
            rng.uniform(np.log(l_value), np.log(L_value), size=dimension)
        )
        low_axis = piece % dimension
        high_axis = (piece + 1) % dimension
        eigenvalues[low_axis] = l_value * rng.uniform(1.0, 1.35)
        eigenvalues[high_axis] = L_value * rng.uniform(0.72, 1.0)
        eigenvalues = np.clip(eigenvalues, l_value, L_value)
        if piece == 0:
            eigenvalues[low_axis] = l_value
            eigenvalues[high_axis] = L_value
        rotation = _orthogonal_matrix(dimension, rng)
        matrices.append((rotation * eigenvalues) @ rotation.T)
    return np.asarray(matrices, dtype=np.float64)


def make_start_points(dimension, n_starts=16, seed=43):
    rng = np.random.default_rng(seed + 1009 * dimension)
    starts = rng.normal(size=(n_starts, dimension))
    norms = np.linalg.norm(starts, axis=1, keepdims=True)
    starts /= np.maximum(norms, 1e-300)
    return starts.astype(np.float64)


def _active_geometry(matrices, n_directions=256):
    n_pieces, dimension, _ = matrices.shape
    rng = np.random.default_rng(17_729 + 101 * dimension + n_pieces)
    directions = rng.normal(size=(n_directions, dimension))
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-300)
    values = 0.5 * np.einsum(
        "di,mij,dj->dm", directions, matrices, directions, optimize=True
    )
    active = np.argmax(values, axis=1)
    counts = np.bincount(active, minlength=n_pieces).astype(np.float64)
    probabilities = counts / np.sum(counts)
    nonzero = probabilities[probabilities > 0.0]
    entropy = -float(np.sum(nonzero * np.log(nonzero))) / max(np.log(n_pieces), 1e-12)
    sorted_values = np.sort(values, axis=1)
    scale = np.maximum(np.abs(sorted_values[:, -1]), 1e-300)
    margins = (sorted_values[:, -1] - sorted_values[:, -2]) / scale
    return {
        "active_piece_fraction": float(np.count_nonzero(counts) / n_pieces),
        "active_entropy": entropy,
        "dominant_active_share": float(np.max(probabilities)),
        "active_margin_mean": float(np.mean(margins)),
    }


def compute_instance_features(matrices):
    matrices = validate_matrices(matrices)
    n_pieces, dimension, _ = matrices.shape
    eigenvalues = np.linalg.eigvalsh(matrices)
    l_value = float(np.min(eigenvalues))
    L_value = float(np.max(eigenvalues))
    kappa = L_value / l_value
    normalized = matrices / L_value
    traces = np.trace(normalized, axis1=1, axis2=2)
    frobenius = np.linalg.norm(normalized, axis=(1, 2))

    distances = []
    commutators = []
    for first, second in combinations(range(n_pieces), 2):
        distances.append(float(np.linalg.norm(normalized[first] - normalized[second])))
        commutator = normalized[first] @ normalized[second] - normalized[second] @ normalized[first]
        commutators.append(float(np.linalg.norm(commutator)))
    geometry = _active_geometry(matrices)

    features = {
        "dimension": float(dimension),
        "n_pieces": float(n_pieces),
        "log_l": float(np.log(l_value)),
        "log_L": float(np.log(L_value)),
        "log_kappa": float(np.log(kappa)),
        "sqrt_kappa": float(np.sqrt(kappa)),
        "trace_over_L_mean": float(np.mean(traces)),
        "trace_over_L_std": float(np.std(traces)),
        "frobenius_over_L_mean": float(np.mean(frobenius)),
        "frobenius_over_L_std": float(np.std(frobenius)),
        "pairwise_distance_mean": float(np.mean(distances)),
        "pairwise_distance_max": float(np.max(distances)),
        "commutator_mean": float(np.mean(commutators)),
        "commutator_max": float(np.max(commutators)),
        **geometry,
    }
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
    seed=42,
    keep_trajectories=False,
):
    matrices = validate_matrices(matrices)
    features = compute_instance_features(matrices)
    # Common positive scaling does not change exact-line-search iterates. This
    # normalization improves numerical conditioning and makes stopping rules
    # comparable across generated objective scales.
    normalized_matrices = matrices / np.exp(features["log_L"])
    starts = make_start_points(matrices.shape[1], n_starts, seed=seed)
    per_start_rates = []
    tail_slopes = []
    tail_r2 = []
    effective_iterations = []
    projection_iterations = []
    statuses = []
    trajectories = []

    for start_id, x0 in enumerate(starts):
        result = run_hypodifferential_descent(
            normalized_matrices,
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
    J300 = float(np.mean(per_start_rates))
    valid_slopes = np.asarray(tail_slopes, dtype=np.float64)
    valid_r2 = np.asarray(tail_r2, dtype=np.float64)
    row = {
        "instance_id": int(instance_id),
        **features,
        "J300": J300,
        "rho": float(10.0**J300),
        "tail_log10_slope_mean": float(np.nanmean(valid_slopes))
        if np.any(np.isfinite(valid_slopes))
        else np.nan,
        "tail_linearity_r2_mean": float(np.nanmean(valid_r2))
        if np.any(np.isfinite(valid_r2))
        else np.nan,
        "effective_iterations_mean": float(np.mean(effective_iterations)),
        "projection_iterations_mean": float(np.mean(projection_iterations)),
        "stationary_fraction": float(np.mean(np.asarray(statuses) == "stationary")),
    }
    return row, trajectories
