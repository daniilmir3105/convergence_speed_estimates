# -*- coding: utf-8 -*-
"""Self-contained numerical implementation of hypodifferential descent."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np


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
    if float(np.min(np.linalg.eigvalsh(symmetric))) <= 0.0:
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
    """Find the minimum-norm point in a convex hull by Gilbert/Frank-Wolfe."""
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
    return -projection, weights, iterations, gap, function_value


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
    root_disc = np.sqrt(max(discriminant, 0.0))
    roots = [(-b - root_disc) / (2.0 * a), (-b + root_disc) / (2.0 * a)]
    return [root for root in roots if root > 0.0 and np.isfinite(root)]


def exact_line_search_max_quadratics(matrices, x, direction):
    """Exactly minimize the upper envelope along a positive ray."""
    if float(np.linalg.norm(direction)) <= np.finfo(np.float64).eps:
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


def generate_max_quadratic_instance(dimension, n_pieces, l_value, kappa, seed):
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
    starts /= np.maximum(np.linalg.norm(starts, axis=1, keepdims=True), 1e-300)
    return starts.astype(np.float64)
