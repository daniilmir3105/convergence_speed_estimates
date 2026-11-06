# -*- coding: utf-8 -*-
"""Optimized finite-horizon Heavy Ball inverse-learning experiment.

The experiment deliberately keeps the historical K-step trajectory target.
It adds five safeguards around it:

1. an exact oracle diagnostic, independent of the ML surrogate;
2. sqrt(kappa) target scaling for the forward regressor;
3. exact re-scoring and refinement after surrogate candidate ranking;
4. stable dimensionless direct targets q and beta;
5. evaluation on actual K-step trajectories, not parameter R2 alone.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

KAPPA_MIN = 1.1
KAPPA_MAX = 20_000.0
STABILITY_SAFETY = 0.99999
BETA_MAX = 0.99
STEPS = 300
N_STARTS = 16

FORWARD_FEATURES = [
    "kappa",
    "log_kappa",
    "sqrt_kappa",
    "inv_sqrt_kappa",
    "alpha_L",
    "beta",
    "q",
]
META_FEATURES = ["log_L", "log_l", "log_kappa", "sqrt_kappa"]


def stratified_log_uniform(size, low, high, rng):
    """Draw one value from every equal-width logarithmic stratum."""
    if not 0 < low < high:
        raise ValueError("Expected 0 < low < high")
    edges = np.linspace(np.log(low), np.log(high), size + 1)
    values = np.exp(rng.uniform(edges[:-1], edges[1:]))
    rng.shuffle(values)
    return values


def theory_from_kappa(kappa):
    """Return Polyak's dimensionless alpha*L and beta."""
    kappa = np.asarray(kappa, dtype=np.float64)
    root = np.sqrt(kappa)
    alpha_L = 4.0 * kappa / (root + 1.0) ** 2
    beta = ((root - 1.0) / (root + 1.0)) ** 2
    return alpha_L, beta


def theory_from_L_l(L, l):
    L = np.asarray(L, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    alpha_L, beta = theory_from_kappa(L / l)
    return alpha_L / L, beta


def make_starts(n_starts=N_STARTS, seed=43):
    rng = np.random.default_rng(seed)
    starts = rng.normal(size=(n_starts, 2)).astype(np.float64)
    tiny = np.linalg.norm(starts, axis=1) < 1e-8
    starts[tiny, 0] = 1.0
    return starts


def terminal_rate(
    kappa,
    alpha_L,
    beta,
    starts,
    steps=STEPS,
    chunk_size=32_768,
):
    """Compute the historical mean log10(f_K/f_0)/K target exactly.

    For zero initial momentum every eigendirection is a scalar multiple of
    its initial coordinate.  We therefore propagate two scalar multipliers
    per candidate instead of all start points.  This is algebraically equal
    to the full trajectory and is substantially faster.
    """
    kappa, alpha_L, beta = np.broadcast_arrays(
        np.asarray(kappa, dtype=np.float64),
        np.asarray(alpha_L, dtype=np.float64),
        np.asarray(beta, dtype=np.float64),
    )
    original_shape = kappa.shape
    kappa = kappa.reshape(-1)
    alpha_L = alpha_L.reshape(-1)
    beta = beta.reshape(-1)
    result = np.empty(len(kappa), dtype=np.float64)

    x1_sq = starts[:, 0] ** 2
    x2_sq = starts[:, 1] ** 2

    for begin in range(0, len(kappa), chunk_size):
        end = min(begin + chunk_size, len(kappa))
        current_kappa = kappa[begin:end]
        current_alpha = alpha_L[begin:end]
        current_beta = beta[begin:end]

        coefficient_L = 1.0 - current_alpha + current_beta
        coefficient_l = 1.0 - current_alpha / current_kappa + current_beta

        # x_{-1}=x_0: both scalar multipliers start at one.
        previous_L = np.ones_like(current_kappa)
        current_L = np.ones_like(current_kappa)
        previous_l = np.ones_like(current_kappa)
        current_l = np.ones_like(current_kappa)

        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            for _ in range(steps):
                next_L = coefficient_L * current_L - current_beta * previous_L
                next_l = coefficient_l * current_l - current_beta * previous_l
                previous_L, current_L = current_L, next_L
                previous_l, current_l = current_l, next_l

            f0 = 0.5 * (
                current_kappa[:, None] * x1_sq[None, :] + x2_sq[None, :]
            )
            fK = 0.5 * (
                current_kappa[:, None]
                * (current_L[:, None] ** 2)
                * x1_sq[None, :]
                + (current_l[:, None] ** 2) * x2_sq[None, :]
            )
            ratio = fK / np.maximum(f0, 1e-300)
            log_ratio = np.log10(np.maximum(ratio, 1e-300))
            score = np.mean(log_ratio, axis=1) / steps

        result[begin:end] = np.nan_to_num(
            score,
            nan=1.0,
            posinf=1.0,
            neginf=-1.0,
        )

    return result.reshape(original_shape)


def forward_feature_matrix(kappa, alpha_L, beta):
    kappa, alpha_L, beta = np.broadcast_arrays(
        np.asarray(kappa, dtype=np.float64),
        np.asarray(alpha_L, dtype=np.float64),
        np.asarray(beta, dtype=np.float64),
    )
    q = alpha_L / np.maximum(2.0 * (1.0 + beta), 1e-12)
    root = np.sqrt(kappa)
    return np.column_stack(
        [
            kappa.reshape(-1),
            np.log(kappa).reshape(-1),
            root.reshape(-1),
            (1.0 / root).reshape(-1),
            alpha_L.reshape(-1),
            beta.reshape(-1),
            q.reshape(-1),
        ]
    ).astype(np.float32)


def meta_feature_matrix(L, l):
    L = np.asarray(L, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    kappa = L / l
    return np.column_stack(
        [np.log(L), np.log(l), np.log(kappa), np.sqrt(kappa)]
    ).astype(np.float32)


def generate_forward_dataset(
    starts,
    n_groups,
    candidates_per_group,
    kappa_min,
    kappa_max,
    beta_max,
    safety,
    steps,
    seed,
):
    if candidates_per_group < 24:
        raise ValueError("forward-candidates must be at least 24")

    rng = np.random.default_rng(seed)
    group_kappa = stratified_log_uniform(n_groups, kappa_min, kappa_max, rng)
    kappa = np.repeat(group_kappa, candidates_per_group)
    group_id = np.repeat(np.arange(n_groups), candidates_per_group)
    theory_alpha_group, theory_beta_group = theory_from_kappa(group_kappa)
    theory_alpha = np.repeat(theory_alpha_group, candidates_per_group)
    theory_beta = np.repeat(theory_beta_group, candidates_per_group)

    n = len(kappa)
    beta = rng.uniform(0.0, beta_max, size=n)
    q = rng.uniform(0.002, safety, size=n)
    alpha_L = 2.0 * (1.0 + beta) * q
    weight = np.ones(n, dtype=np.float64)
    kind = np.full(n, "global", dtype=object)

    # Half of every group is concentrated near the theoretical reference.
    local_start = candidates_per_group // 2
    local = (np.arange(n) % candidates_per_group) >= local_start
    beta[local] = np.clip(
        theory_beta[local] + rng.normal(0.0, 0.10, local.sum()),
        0.0,
        beta_max,
    )
    alpha_L[local] = theory_alpha[local] * np.exp(
        rng.normal(0.0, 0.18, local.sum())
    )
    alpha_L[local] = np.clip(
        alpha_L[local],
        1e-8,
        safety * 2.0 * (1.0 + beta[local]),
    )
    weight[local] = 4.0
    kind[local] = "local"

    anchors = [
        (1.00, 0.00),
        (0.97, 0.00),
        (1.03, 0.00),
        (1.00, -0.02),
        (1.00, 0.02),
        (0.95, -0.04),
        (1.05, 0.04),
        (0.90, 0.00),
    ]
    for group in range(n_groups):
        base = group * candidates_per_group
        for index, (alpha_scale, beta_shift) in enumerate(anchors):
            row = base + index
            beta[row] = np.clip(
                theory_beta_group[group] + beta_shift, 0.0, beta_max
            )
            alpha_L[row] = min(
                theory_alpha_group[group] * alpha_scale,
                safety * 2.0 * (1.0 + beta[row]),
            )
            weight[row] = 8.0
            kind[row] = "anchor"

    started = time.perf_counter()
    target = terminal_rate(kappa, alpha_L, beta, starts, steps=steps)
    scaled_target = np.sqrt(kappa) * target
    elapsed = time.perf_counter() - started
    print(
        f"Generated {n:,} exact finite-horizon rows in {elapsed:.1f}s "
        f"({n / max(elapsed, 1e-12):,.0f} rows/s)."
    )

    return pd.DataFrame(
        {
            "group_id": group_id,
            "kappa": kappa,
            "alpha_L": alpha_L,
            "beta": beta,
            "q": alpha_L / (2.0 * (1.0 + beta)),
            "target_rate": target,
            "scaled_target_rate": scaled_target,
            "sample_weight": weight,
            "sample_kind": kind,
        }
    )


def metric_dict(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "corr": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def print_metrics(name, y_true, y_pred):
    metrics = metric_dict(y_true, y_pred)
    print(
        f"{name}: R2={metrics['r2']:.4f} | corr={metrics['corr']:.4f} | "
        f"MAE={metrics['mae']:.8f} | RMSE={metrics['rmse']:.8f}"
    )
    return metrics


def train_forward_model(frame, seed, max_iter):
    groups = frame["group_id"].unique()
    train_groups, validation_groups = train_test_split(
        groups, test_size=0.2, random_state=seed, shuffle=True
    )
    train_mask = frame["group_id"].isin(train_groups).to_numpy()
    validation_mask = frame["group_id"].isin(validation_groups).to_numpy()

    X = forward_feature_matrix(frame["kappa"], frame["alpha_L"], frame["beta"])
    y = frame["scaled_target_rate"].to_numpy()
    weights = frame["sample_weight"].to_numpy()

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.055,
        max_iter=max_iter,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=0.10,
        random_state=seed,
    )
    model.fit(X[train_mask], y[train_mask], sample_weight=weights[train_mask])

    prediction = model.predict(X[validation_mask])
    raw_prediction = prediction / np.sqrt(frame.loc[validation_mask, "kappa"])
    local_validation = validation_mask & (frame["sample_kind"] != "global").to_numpy()
    local_prediction = model.predict(X[local_validation])

    return model, {
        "scaled_validation": print_metrics(
            "Forward validation (scaled)", y[validation_mask], prediction
        ),
        "raw_validation": print_metrics(
            "Forward validation (raw J_K)",
            frame.loc[validation_mask, "target_rate"],
            raw_prediction,
        ),
        "scaled_near_optimum_validation": print_metrics(
            "Forward validation near optimum",
            frame.loc[local_validation, "scaled_target_rate"],
            local_prediction,
        ),
    }


def global_grid(batch_size, n_q, n_beta, safety, beta_max):
    q_axis = np.linspace(1e-8, safety, n_q)
    beta_axis = np.linspace(0.0, beta_max, n_beta)
    q = np.broadcast_to(q_axis[None, None, :], (batch_size, n_beta, n_q))
    beta = np.broadcast_to(
        beta_axis[None, :, None], (batch_size, n_beta, n_q)
    )
    alpha_L = 2.0 * q * (1.0 + beta)
    return (
        alpha_L.reshape(batch_size, -1),
        beta.reshape(batch_size, -1),
        q.reshape(batch_size, -1),
    )


def local_grid(
    best_q,
    best_beta,
    n_q,
    n_beta,
    q_radius,
    beta_radius,
    safety,
    beta_max,
):
    q_offsets = np.linspace(-1.0, 1.0, n_q)
    beta_offsets = np.linspace(-1.0, 1.0, n_beta)
    q = np.clip(
        best_q[:, None, None] + q_radius * q_offsets[None, None, :],
        1e-8,
        safety,
    )
    beta = np.clip(
        best_beta[:, None, None] + beta_radius * beta_offsets[None, :, None],
        0.0,
        beta_max,
    )
    shape = (len(best_q), n_beta, n_q)
    q = np.broadcast_to(q, shape)
    beta = np.broadcast_to(beta, shape)
    alpha_L = 2.0 * q * (1.0 + beta)
    return (
        alpha_L.reshape(len(best_q), -1),
        beta.reshape(len(best_q), -1),
        q.reshape(len(best_q), -1),
    )


def predicted_scores(model, kappa, alpha_L, beta):
    kappa_rows = np.broadcast_to(kappa[:, None], alpha_L.shape)
    X = forward_feature_matrix(kappa_rows, alpha_L, beta)
    return model.predict(X).reshape(alpha_L.shape)


def exact_scores(kappa, alpha_L, beta, starts, steps):
    kappa_rows = np.broadcast_to(kappa[:, None], alpha_L.shape)
    return terminal_rate(
        kappa_rows,
        alpha_L,
        beta,
        starts,
        steps=steps,
    )


def take_top_candidates(alpha_L, beta, q, score, top_k):
    top_k = min(top_k, score.shape[1])
    indices = np.argpartition(score, top_k - 1, axis=1)[:, :top_k]
    return (
        np.take_along_axis(alpha_L, indices, axis=1),
        np.take_along_axis(beta, indices, axis=1),
        np.take_along_axis(q, indices, axis=1),
    )


def select_best(alpha_L, beta, q, score):
    indices = np.argmin(score, axis=1)
    rows = np.arange(len(indices))
    return (
        alpha_L[rows, indices],
        beta[rows, indices],
        q[rows, indices],
        score[rows, indices],
    )


def oracle_search(
    kappa,
    starts,
    steps,
    n_grid,
    refine_size,
    refine_stages,
    batch_size,
    safety,
    beta_max,
):
    """Minimize the true finite-horizon score without an ML model."""
    kappa = np.asarray(kappa, dtype=np.float64)
    outputs = [np.empty(len(kappa), dtype=np.float64) for _ in range(4)]
    started = time.perf_counter()

    for begin in range(0, len(kappa), batch_size):
        end = min(begin + batch_size, len(kappa))
        current_kappa = kappa[begin:end]
        alpha_L, beta, q = global_grid(
            len(current_kappa), n_grid, n_grid, safety, beta_max
        )

        # The theoretical point is evaluated, but it wins only if its actual
        # J_K is no worse than the finite-horizon alternatives.
        theory_alpha, theory_beta = theory_from_kappa(current_kappa)
        theory_q = theory_alpha / (2.0 * (1.0 + theory_beta))
        alpha_L = np.column_stack([alpha_L, theory_alpha])
        beta = np.column_stack([beta, theory_beta])
        q = np.column_stack([q, theory_q])

        score = exact_scores(current_kappa, alpha_L, beta, starts, steps)
        best_alpha, best_beta, best_q, best_score = select_best(
            alpha_L, beta, q, score
        )

        q_radius = 2.0 * safety / max(n_grid - 1, 1)
        beta_radius = 2.0 * beta_max / max(n_grid - 1, 1)
        for _ in range(refine_stages):
            alpha_ref, beta_ref, q_ref = local_grid(
                best_q,
                best_beta,
                refine_size,
                refine_size,
                q_radius,
                beta_radius,
                safety,
                beta_max,
            )
            score_ref = exact_scores(
                current_kappa, alpha_ref, beta_ref, starts, steps
            )
            candidate = select_best(alpha_ref, beta_ref, q_ref, score_ref)
            improved = candidate[3] < best_score
            best_alpha[improved] = candidate[0][improved]
            best_beta[improved] = candidate[1][improved]
            best_q[improved] = candidate[2][improved]
            best_score[improved] = candidate[3][improved]
            q_radius *= 0.35
            beta_radius *= 0.35

        for target, values in zip(outputs, [best_alpha, best_beta, best_q, best_score]):
            target[begin:end] = values

        elapsed = time.perf_counter() - started
        rate = end / max(elapsed, 1e-12)
        eta = (len(kappa) - end) / max(rate, 1e-12)
        print(f"  oracle {end:5d}/{len(kappa)} | ETA {eta:6.1f}s")

    return tuple(outputs)


def hybrid_inverse_search(
    model,
    kappa,
    starts,
    steps,
    n_grid,
    predicted_refine,
    exact_top_k,
    exact_refine,
    exact_stages,
    batch_size,
    safety,
    beta_max,
):
    """Use ML for ranking, then determine labels with the exact J_K."""
    kappa = np.asarray(kappa, dtype=np.float64)
    outputs = [np.empty(len(kappa), dtype=np.float64) for _ in range(4)]
    started = time.perf_counter()

    for begin in range(0, len(kappa), batch_size):
        end = min(begin + batch_size, len(kappa))
        current_kappa = kappa[begin:end]
        alpha_L, beta, q = global_grid(
            len(current_kappa), n_grid, n_grid, safety, beta_max
        )
        score = predicted_scores(model, current_kappa, alpha_L, beta)
        predicted_best = select_best(alpha_L, beta, q, score)

        half_top = max(1, exact_top_k // 2)
        coarse_top = take_top_candidates(
            alpha_L, beta, q, score, half_top
        )

        q_radius = 2.0 * safety / max(n_grid - 1, 1)
        beta_radius = 2.0 * beta_max / max(n_grid - 1, 1)
        alpha_ref, beta_ref, q_ref = local_grid(
            predicted_best[2],
            predicted_best[1],
            predicted_refine,
            predicted_refine,
            q_radius,
            beta_radius,
            safety,
            beta_max,
        )
        score_ref = predicted_scores(model, current_kappa, alpha_ref, beta_ref)
        refined_top = take_top_candidates(
            alpha_ref,
            beta_ref,
            q_ref,
            score_ref,
            exact_top_k - half_top,
        )

        theory_alpha, theory_beta = theory_from_kappa(current_kappa)
        theory_q = theory_alpha / (2.0 * (1.0 + theory_beta))
        candidates_alpha = np.column_stack(
            [coarse_top[0], refined_top[0], theory_alpha]
        )
        candidates_beta = np.column_stack(
            [coarse_top[1], refined_top[1], theory_beta]
        )
        candidates_q = np.column_stack(
            [coarse_top[2], refined_top[2], theory_q]
        )
        exact = exact_scores(
            current_kappa, candidates_alpha, candidates_beta, starts, steps
        )
        best_alpha, best_beta, best_q, best_score = select_best(
            candidates_alpha, candidates_beta, candidates_q, exact
        )

        for _ in range(exact_stages):
            alpha_exact, beta_exact, q_exact = local_grid(
                best_q,
                best_beta,
                exact_refine,
                exact_refine,
                q_radius,
                beta_radius,
                safety,
                beta_max,
            )
            exact_score = exact_scores(
                current_kappa, alpha_exact, beta_exact, starts, steps
            )
            candidate = select_best(
                alpha_exact, beta_exact, q_exact, exact_score
            )
            improved = candidate[3] < best_score
            best_alpha[improved] = candidate[0][improved]
            best_beta[improved] = candidate[1][improved]
            best_q[improved] = candidate[2][improved]
            best_score[improved] = candidate[3][improved]
            q_radius *= 0.35
            beta_radius *= 0.35

        for target, values in zip(outputs, [best_alpha, best_beta, best_q, best_score]):
            target[begin:end] = values

        elapsed = time.perf_counter() - started
        rate = end / max(elapsed, 1e-12)
        eta = (len(kappa) - end) / max(rate, 1e-12)
        print(f"  inverse {end:5d}/{len(kappa)} | ETA {eta:6.1f}s")

    return tuple(outputs)


def sample_meta_pairs(n_pairs, kappa_min, kappa_max, seed):
    rng = np.random.default_rng(seed)
    l = rng.uniform(1.0, 50.0, size=n_pairs)
    kappa = stratified_log_uniform(n_pairs, kappa_min, kappa_max, rng)
    L = l * kappa
    alpha_theory, beta_theory = theory_from_L_l(L, l)
    alpha_L_theory = alpha_theory * L
    return pd.DataFrame(
        {
            "L": L,
            "l": l,
            "kappa": kappa,
            "alpha_theory": alpha_theory,
            "beta_theory": beta_theory,
            "q_theory": alpha_L_theory / (2.0 * (1.0 + beta_theory)),
        }
    )


def train_direct_model(X, y, seed, n_estimators):
    model = GradientBoostingRegressor(
        loss="huber",
        n_estimators=n_estimators,
        learning_rate=0.035,
        max_depth=3,
        min_samples_leaf=5,
        subsample=0.90,
        random_state=seed,
    )
    model.fit(X, y)
    return model


def metrics_by_kappa(frame, truth, prediction, kappa_max):
    edges = [KAPPA_MIN, 10.0, 100.0, 1000.0, 10_000.0, kappa_max]
    result = {}
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        final = index == len(edges) - 2
        mask = (frame["kappa"] >= lower) & (
            frame["kappa"] <= upper if final else frame["kappa"] < upper
        )
        current = frame.loc[mask]
        if len(current) < 2:
            continue
        result[f"[{lower:g}, {upper:g}{']' if final else ')'}"] = {
            "count": int(len(current)),
            **metric_dict(current[truth], current[prediction]),
        }
    return result


def performance_summary(model_score, reference_score):
    model_score = np.asarray(model_score)
    reference_score = np.asarray(reference_score)
    return {
        "model_mean_JK": float(np.mean(model_score)),
        "reference_mean_JK": float(np.mean(reference_score)),
        "model_better_fraction": float(np.mean(model_score < reference_score)),
        "mean_regret": float(np.mean(model_score - reference_score)),
    }


def save_plots(oracle_frame, test_frame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(
        oracle_frame["alpha_theory"], oracle_frame["alpha_found"], s=12, alpha=0.5
    )
    axes[1].scatter(
        oracle_frame["beta_theory"], oracle_frame["beta_found"], s=12, alpha=0.5
    )
    for axis, name in zip(axes, ["alpha", "beta"]):
        low = min(oracle_frame[f"{name}_theory"].min(), oracle_frame[f"{name}_found"].min())
        high = max(oracle_frame[f"{name}_theory"].max(), oracle_frame[f"{name}_found"].max())
        axis.plot([low, high], [low, high], "r--")
        axis.set_xlabel(f"{name} Polyak")
        axis.set_ylabel(f"{name} exact finite-horizon oracle")
        axis.grid(True)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "oracle_vs_polyak.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, name in zip(axes, ["alpha", "beta"]):
        axis.scatter(
            test_frame[f"{name}_found"], test_frame[f"{name}_pred"], s=12, alpha=0.45
        )
        low = min(test_frame[f"{name}_found"].min(), test_frame[f"{name}_pred"].min())
        high = max(test_frame[f"{name}_found"].max(), test_frame[f"{name}_pred"].max())
        axis.plot([low, high], [low, high], "r--")
        axis.set_xlabel(f"{name} finite-horizon label")
        axis.set_ylabel(f"{name} direct model")
        axis.grid(True)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "direct_models_vs_finite_labels.png", dpi=160)
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Optimized exact finite-horizon Heavy Ball ML pipeline"
    )
    parser.add_argument("--kappa-min", type=float, default=KAPPA_MIN)
    parser.add_argument("--kappa-max", type=float, default=KAPPA_MAX)
    parser.add_argument("--stability-safety", type=float, default=STABILITY_SAFETY)
    parser.add_argument("--beta-max", type=float, default=BETA_MAX)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--starts", type=int, default=N_STARTS)
    parser.add_argument("--forward-groups", type=int, default=2400)
    parser.add_argument("--forward-candidates", type=int, default=96)
    parser.add_argument("--forward-iterations", type=int, default=350)
    parser.add_argument("--oracle-points", type=int, default=256)
    parser.add_argument("--oracle-grid", type=int, default=64)
    parser.add_argument("--oracle-refine", type=int, default=25)
    parser.add_argument("--oracle-stages", type=int, default=2)
    parser.add_argument("--meta-pairs", type=int, default=10000)
    parser.add_argument("--meta-estimators", type=int, default=600)
    parser.add_argument("--search-grid", type=int, default=64)
    parser.add_argument("--predicted-refine", type=int, default=32)
    parser.add_argument("--exact-top-k", type=int, default=24)
    parser.add_argument("--exact-refine", type=int, default=17)
    parser.add_argument("--exact-stages", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(args):
    if not 0 < args.kappa_min < args.kappa_max:
        raise ValueError("Expected 0 < kappa-min < kappa-max")
    if not 0 < args.stability_safety < 1:
        raise ValueError("stability-safety must be in (0, 1)")
    if not 0 < args.beta_max < 1:
        raise ValueError("beta-max must be in (0, 1)")
    required_q = args.kappa_max / (args.kappa_max + 1.0)
    required_beta = float(theory_from_kappa(args.kappa_max)[1])
    if args.stability_safety <= required_q:
        raise ValueError("stability-safety excludes the Polyak point")
    if args.beta_max <= required_beta:
        raise ValueError("beta-max excludes the Polyak point")
    if args.steps < 1 or args.starts < 1:
        raise ValueError("steps and starts must be positive")
    if args.forward_groups < 10 or args.forward_candidates < 24:
        raise ValueError("forward dataset is too small")
    if args.oracle_points < 2 or args.meta_pairs < 10:
        raise ValueError("oracle-points and meta-pairs are too small")
    if min(args.oracle_grid, args.oracle_refine, args.search_grid) < 3:
        raise ValueError("search grids must contain at least three points")
    if min(args.predicted_refine, args.exact_refine) < 3:
        raise ValueError("refinement grids must contain at least three points")
    if args.exact_top_k < 2:
        raise ValueError("exact-top-k must be at least two")
    if min(args.oracle_stages, args.exact_stages, args.batch_size) < 1:
        raise ValueError("stage counts and batch-size must be positive")


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    starts = make_starts(args.starts, seed=args.seed + 1)
    np.save(DATA_DIR / "initial_points.npy", starts)

    print("\n1/6 Exact oracle diagnostic")
    rng = np.random.default_rng(args.seed + 100)
    oracle_kappa = stratified_log_uniform(
        args.oracle_points, args.kappa_min, args.kappa_max, rng
    )
    oracle_alpha_L, oracle_beta, oracle_q, oracle_score = oracle_search(
        oracle_kappa,
        starts,
        args.steps,
        args.oracle_grid,
        args.oracle_refine,
        args.oracle_stages,
        args.batch_size,
        args.stability_safety,
        args.beta_max,
    )
    oracle_l = np.ones_like(oracle_kappa)
    oracle_L = oracle_kappa.copy()
    oracle_alpha = oracle_alpha_L / oracle_L
    theory_alpha, theory_beta = theory_from_L_l(oracle_L, oracle_l)
    theory_alpha_L = theory_alpha * oracle_L
    theory_score = terminal_rate(
        oracle_kappa, theory_alpha_L, theory_beta, starts, steps=args.steps
    )
    oracle_frame = pd.DataFrame(
        {
            "L": oracle_L,
            "l": oracle_l,
            "kappa": oracle_kappa,
            "alpha_found": oracle_alpha,
            "beta_found": oracle_beta,
            "q_found": oracle_q,
            "J_found": oracle_score,
            "alpha_theory": theory_alpha,
            "beta_theory": theory_beta,
            "J_theory": theory_score,
        }
    )
    oracle_frame.to_csv(DATA_DIR / "oracle_diagnostic.csv", index=False)
    oracle_metrics = {
        "alpha_vs_polyak": print_metrics(
            "Oracle alpha vs Polyak", theory_alpha, oracle_alpha
        ),
        "beta_vs_polyak": print_metrics(
            "Oracle beta vs Polyak", theory_beta, oracle_beta
        ),
        "performance_vs_polyak": performance_summary(oracle_score, theory_score),
    }

    print("\n2/6 Generating exact finite-horizon forward dataset")
    forward_frame = generate_forward_dataset(
        starts,
        args.forward_groups,
        args.forward_candidates,
        args.kappa_min,
        args.kappa_max,
        args.beta_max,
        args.stability_safety,
        args.steps,
        args.seed,
    )
    forward_frame.to_pickle(DATA_DIR / "forward_dataset.pkl")

    print("\n3/6 Training sqrt(kappa)-scaled forward model")
    forward_model, forward_metrics = train_forward_model(
        forward_frame, args.seed, args.forward_iterations
    )
    joblib.dump(
        forward_model, MODELS_DIR / "hist_gradient_boosting_finite_rate.joblib"
    )

    print("\n4/6 Building exact-refined inverse labels")
    all_pairs = sample_meta_pairs(
        args.meta_pairs, args.kappa_min, args.kappa_max, args.seed + 10
    )
    found_alpha_L, found_beta, found_q, found_score = hybrid_inverse_search(
        forward_model,
        all_pairs["kappa"].to_numpy(),
        starts,
        args.steps,
        args.search_grid,
        args.predicted_refine,
        args.exact_top_k,
        args.exact_refine,
        args.exact_stages,
        args.batch_size,
        args.stability_safety,
        args.beta_max,
    )
    all_pairs["alpha_found"] = found_alpha_L / all_pairs["L"].to_numpy()
    all_pairs["beta_found"] = found_beta
    all_pairs["q_found"] = found_q
    all_pairs["J_found"] = found_score
    theory_alpha_L = all_pairs["alpha_theory"].to_numpy() * all_pairs["L"].to_numpy()
    all_pairs["J_theory"] = terminal_rate(
        all_pairs["kappa"].to_numpy(),
        theory_alpha_L,
        all_pairs["beta_theory"].to_numpy(),
        starts,
        steps=args.steps,
    )
    all_pairs.to_csv(DATA_DIR / "inverse_labels_all.csv", index=False)

    train_index, test_index = train_test_split(
        np.arange(len(all_pairs)),
        test_size=0.2,
        random_state=args.seed,
        shuffle=True,
    )
    train_frame = all_pairs.iloc[train_index].copy().reset_index(drop=True)
    test_frame = all_pairs.iloc[test_index].copy().reset_index(drop=True)

    print("\n5/6 Training stable q/beta direct models")
    X_train = meta_feature_matrix(train_frame["L"], train_frame["l"])
    q_model = train_direct_model(
        X_train, train_frame["q_found"], args.seed, args.meta_estimators
    )
    beta_model = train_direct_model(
        X_train, train_frame["beta_found"], args.seed, args.meta_estimators
    )
    joblib.dump(q_model, MODELS_DIR / "gradient_boosting_q.joblib")
    joblib.dump(beta_model, MODELS_DIR / "gradient_boosting_beta.joblib")

    print("\n6/6 Evaluating parameters and actual 300-step performance")
    X_test = meta_feature_matrix(test_frame["L"], test_frame["l"])
    test_frame["q_pred"] = np.clip(
        q_model.predict(X_test), 1e-8, args.stability_safety
    )
    test_frame["beta_pred"] = np.clip(
        beta_model.predict(X_test), 0.0, args.beta_max
    )
    test_frame["alpha_L_pred"] = (
        2.0 * test_frame["q_pred"] * (1.0 + test_frame["beta_pred"])
    )
    test_frame["alpha_pred"] = (
        test_frame["alpha_L_pred"] / test_frame["L"]
    )
    test_frame["J_pred"] = terminal_rate(
        test_frame["kappa"].to_numpy(),
        test_frame["alpha_L_pred"].to_numpy(),
        test_frame["beta_pred"].to_numpy(),
        starts,
        steps=args.steps,
    )
    test_frame.to_csv(DATA_DIR / "direct_model_test.csv", index=False)

    final_metrics = {
        "alpha_vs_finite_label": print_metrics(
            "Direct alpha vs finite label",
            test_frame["alpha_found"],
            test_frame["alpha_pred"],
        ),
        "beta_vs_finite_label": print_metrics(
            "Direct beta vs finite label",
            test_frame["beta_found"],
            test_frame["beta_pred"],
        ),
        "q_vs_finite_label": print_metrics(
            "Direct q vs finite label", test_frame["q_found"], test_frame["q_pred"]
        ),
        "alpha_vs_polyak": print_metrics(
            "Direct alpha vs Polyak",
            test_frame["alpha_theory"],
            test_frame["alpha_pred"],
        ),
        "beta_vs_polyak": print_metrics(
            "Direct beta vs Polyak",
            test_frame["beta_theory"],
            test_frame["beta_pred"],
        ),
        "performance_vs_finite_label": performance_summary(
            test_frame["J_pred"], test_frame["J_found"]
        ),
        "performance_vs_polyak": performance_summary(
            test_frame["J_pred"], test_frame["J_theory"]
        ),
        "unstable_fraction": float(
            np.mean(test_frame["q_pred"].to_numpy() >= 1.0)
        ),
    }

    save_plots(oracle_frame, test_frame)
    manifest = {
        "description": "Exact-refined finite-horizon Heavy Ball experiment",
        "external_inputs": ["L", "l"],
        "internal_direct_features": META_FEATURES,
        "saved_models": {
            "q": str(MODELS_DIR / "gradient_boosting_q.joblib"),
            "beta": str(MODELS_DIR / "gradient_boosting_beta.joblib"),
        },
        "target": {
            "definition": "mean_i(log10(max(f_K/f_0, 1e-300))) / K",
            "steps": args.steps,
            "starts": args.starts,
            "forward_training_target": "sqrt(kappa) * finite_horizon_target",
        },
        "config": vars(args),
        "metrics": {
            "oracle": oracle_metrics,
            "forward": forward_metrics,
            "inverse_labels_vs_polyak": {
                "alpha": metric_dict(
                    all_pairs["alpha_theory"], all_pairs["alpha_found"]
                ),
                "beta": metric_dict(
                    all_pairs["beta_theory"], all_pairs["beta_found"]
                ),
                "performance": performance_summary(
                    all_pairs["J_found"], all_pairs["J_theory"]
                ),
            },
            "final": final_metrics,
        },
        "metrics_by_kappa": {
            "alpha_vs_finite_label": metrics_by_kappa(
                test_frame, "alpha_found", "alpha_pred", args.kappa_max
            ),
            "beta_vs_finite_label": metrics_by_kappa(
                test_frame, "beta_found", "beta_pred", args.kappa_max
            ),
            "alpha_vs_polyak": metrics_by_kappa(
                test_frame, "alpha_theory", "alpha_pred", args.kappa_max
            ),
            "beta_vs_polyak": metrics_by_kappa(
                test_frame, "beta_theory", "beta_pred", args.kappa_max
            ),
        },
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nSaved:")
    print(" ", RESULTS_DIR / "metrics.json")
    print(" ", RESULTS_DIR / "oracle_vs_polyak.png")
    print(" ", RESULTS_DIR / "direct_models_vs_finite_labels.png")
    print(" ", MODELS_DIR / "gradient_boosting_q.joblib")
    print(" ", MODELS_DIR / "gradient_boosting_beta.joblib")


if __name__ == "__main__":
    main()
