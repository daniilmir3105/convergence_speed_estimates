# -*- coding: utf-8 -*-
"""Optimized finite-horizon Gradient Descent inverse-learning experiment.

The first model learns sqrt(kappa) * J_K from a candidate normalized step.
It is then used to rank candidates.  Exact J_K re-scoring/refinement produces
labels for the final public model (L, l) -> alpha.  Classical GD theory is
kept as an independent evaluation baseline and is not used as a label.
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
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from .alpha_model import GDAlphaModel, META_FEATURES, gd_theory, meta_feature_matrix


matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

KAPPA_MIN = 1.1
KAPPA_MAX = 20_000.0
STABILITY_SAFETY = 0.99999
STEPS = 300
N_STARTS = 16

FORWARD_FEATURES = [
    "kappa",
    "log_kappa",
    "sqrt_kappa",
    "inv_sqrt_kappa",
    "alpha_L",
    "q",
]


def stratified_log_uniform(size, low, high, rng):
    """Draw one value from every equal-width logarithmic stratum."""
    if not 0 < low < high:
        raise ValueError("Expected 0 < low < high")
    edges = np.linspace(np.log(low), np.log(high), size + 1)
    values = np.exp(rng.uniform(edges[:-1], edges[1:]))
    rng.shuffle(values)
    return values


def theory_from_kappa(kappa):
    """Return q*=alpha*L/2 and alpha*L for classical constant-step GD."""
    kappa = np.asarray(kappa, dtype=np.float64)
    q = kappa / (kappa + 1.0)
    return q, 2.0 * q


def make_starts(n_starts=N_STARTS, seed=43):
    rng = np.random.default_rng(seed)
    starts = rng.normal(size=(n_starts, 2)).astype(np.float64)
    tiny = np.linalg.norm(starts, axis=1) < 1e-8
    starts[tiny, 0] = 1.0
    return starts


def terminal_rate(
    kappa,
    alpha_L,
    starts,
    steps=STEPS,
    chunk_size=65_536,
):
    """Compute the requested finite-horizon target J_K exactly.

    The normalized quadratic is
        f(x) = 0.5 * (kappa*x_1^2 + x_2^2).
    For GD, the two coordinates are multiplied at every step by
    (1-alpha*L) and (1-alpha*l).  Their K-step values therefore have a
    closed form, so no Python loop over 300 iterations is needed.
    """
    kappa, alpha_L = np.broadcast_arrays(
        np.asarray(kappa, dtype=np.float64),
        np.asarray(alpha_L, dtype=np.float64),
    )
    original_shape = kappa.shape
    kappa = kappa.reshape(-1)
    alpha_L = alpha_L.reshape(-1)
    result = np.empty(len(kappa), dtype=np.float64)

    x1_sq = starts[:, 0] ** 2
    x2_sq = starts[:, 1] ** 2

    for begin in range(0, len(kappa), chunk_size):
        end = min(begin + chunk_size, len(kappa))
        current_kappa = kappa[begin:end]
        current_alpha = alpha_L[begin:end]

        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            decay_L = np.power(np.abs(1.0 - current_alpha), 2 * steps)
            decay_l = np.power(
                np.abs(1.0 - current_alpha / current_kappa), 2 * steps
            )
            f0 = 0.5 * (
                current_kappa[:, None] * x1_sq[None, :] + x2_sq[None, :]
            )
            fK = 0.5 * (
                current_kappa[:, None] * decay_L[:, None] * x1_sq[None, :]
                + decay_l[:, None] * x2_sq[None, :]
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


def forward_feature_matrix(kappa, alpha_L):
    kappa, alpha_L = np.broadcast_arrays(
        np.asarray(kappa, dtype=np.float64),
        np.asarray(alpha_L, dtype=np.float64),
    )
    root = np.sqrt(kappa)
    q = alpha_L / 2.0
    return np.column_stack(
        [
            kappa.reshape(-1),
            np.log(kappa).reshape(-1),
            root.reshape(-1),
            (1.0 / root).reshape(-1),
            alpha_L.reshape(-1),
            q.reshape(-1),
        ]
    ).astype(np.float32)


def generate_forward_dataset(
    starts,
    n_groups,
    candidates_per_group,
    kappa_min,
    kappa_max,
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
    n = len(kappa)
    row_in_group = np.arange(n) % candidates_per_group
    q = rng.uniform(1e-8, safety, size=n)
    weight = np.ones(n, dtype=np.float64)
    kind = np.full(n, "global", dtype=object)

    # Half of every group is sampled logarithmically by its gap 1-q.  This
    # resolves the narrow stable region near q=1 without using the theoretical
    # optimum as either a target or a sampling centre.
    focused = row_in_group >= candidates_per_group // 2
    minimum_gap = max(1.0 - safety, 1e-12)
    gaps = np.exp(
        rng.uniform(np.log(minimum_gap), np.log(0.5), focused.sum())
    )
    q[focused] = 1.0 - gaps
    weight[focused] = 4.0
    kind[focused] = "focused"

    # Deterministic scale anchors are also independent of the GD formula.
    anchors = [
        0.25,
        0.50,
        0.65,
        0.80,
        0.90,
        0.97,
        0.99,
        0.999,
        0.9999,
        safety,
    ]
    for group in range(n_groups):
        base = group * candidates_per_group
        for index, value in enumerate(anchors):
            row = base + index
            q[row] = np.clip(value, 1e-8, safety)
            weight[row] = 8.0
            kind[row] = "anchor"

    alpha_L = 2.0 * q
    started = time.perf_counter()
    target = terminal_rate(kappa, alpha_L, starts, steps=steps)
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
            "q": q,
            "target_rate": target,
            "scaled_target_rate": scaled_target,
            "sample_weight": weight,
            "sample_kind": kind,
        }
    )


def print_r2(name, y_true, y_pred):
    score = float(r2_score(y_true, y_pred))
    print(f"{name}: R2={score:.6f}")
    return score


def train_forward_model(frame, seed, max_iter):
    groups = frame["group_id"].unique()
    train_groups, validation_groups = train_test_split(
        groups, test_size=0.2, random_state=seed, shuffle=True
    )
    train_mask = frame["group_id"].isin(train_groups).to_numpy()
    validation_mask = frame["group_id"].isin(validation_groups).to_numpy()
    # "Near optimum" is defined by the true target, not by a theoretical
    # formula: retain the best 20% candidates inside every validation kappa.
    target_rank = frame.groupby("group_id")["target_rate"].rank(
        method="first", pct=True
    )
    near_optimum_validation = validation_mask & (target_rank.to_numpy() <= 0.20)

    X = forward_feature_matrix(frame["kappa"], frame["alpha_L"])
    y = frame["scaled_target_rate"].to_numpy()
    weights = frame["sample_weight"].to_numpy()

    model = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=max_iter,
        max_leaf_nodes=63,
        min_samples_leaf=20,
        l2_regularization=0.10,
        random_state=seed,
    )
    model.fit(X[train_mask], y[train_mask], sample_weight=weights[train_mask])

    prediction = model.predict(X[validation_mask])
    raw_prediction = prediction / np.sqrt(frame.loc[validation_mask, "kappa"])
    near_optimum_prediction = model.predict(X[near_optimum_validation])
    return model, {
        "validation_scaled_r2": print_r2(
            "Forward validation (scaled)", y[validation_mask], prediction
        ),
        "near_optimum_scaled_r2": print_r2(
            "Forward validation near optimum (scaled)",
            frame.loc[near_optimum_validation, "scaled_target_rate"],
            near_optimum_prediction,
        ),
        "validation_raw_J300_r2": print_r2(
            "Forward validation (raw J_K)",
            frame.loc[validation_mask, "target_rate"],
            raw_prediction,
        ),
    }


def global_q_axis(size, safety):
    """Mixed grid: broad linear coverage plus dense coverage near q=1."""
    linear_size = max(3, size // 2)
    boundary_size = max(3, size - linear_size)
    linear_high = min(0.98, safety)
    linear = np.linspace(1e-8, linear_high, linear_size)
    min_gap = max(1.0 - safety, 1e-12)
    boundary_gap = np.geomspace(min_gap, 0.5, boundary_size)
    boundary = 1.0 - boundary_gap
    return np.unique(np.clip(np.concatenate([linear, boundary]), 1e-8, safety))


def global_grid(batch_size, size, safety):
    axis = global_q_axis(size, safety)
    q = np.broadcast_to(axis[None, :], (batch_size, len(axis)))
    return 2.0 * q, q


def initial_radius(best_q, safety):
    best_q = np.asarray(best_q, dtype=np.float64)
    return np.minimum(
        0.25,
        np.maximum(4.0 * (1.0 - best_q), 4.0 * (1.0 - safety)),
    )


def local_grid(best_q, size, radius, safety):
    offsets = np.linspace(-1.0, 1.0, size)
    radius = np.asarray(radius, dtype=np.float64)
    if radius.ndim == 0:
        radius = np.full(len(best_q), float(radius))
    q = np.clip(
        np.asarray(best_q)[:, None] + radius[:, None] * offsets[None, :],
        1e-8,
        safety,
    )
    return 2.0 * q, q


def predicted_scores(model, kappa, alpha_L):
    kappa_rows = np.broadcast_to(kappa[:, None], alpha_L.shape)
    X = forward_feature_matrix(kappa_rows, alpha_L)
    return model.predict(X).reshape(alpha_L.shape)


def exact_scores(kappa, alpha_L, starts, steps):
    kappa_rows = np.broadcast_to(kappa[:, None], alpha_L.shape)
    return terminal_rate(kappa_rows, alpha_L, starts, steps=steps)


def take_top_candidates(alpha_L, q, score, top_k):
    top_k = min(top_k, score.shape[1])
    indices = np.argpartition(score, top_k - 1, axis=1)[:, :top_k]
    return (
        np.take_along_axis(alpha_L, indices, axis=1),
        np.take_along_axis(q, indices, axis=1),
    )


def select_best(alpha_L, q, score):
    indices = np.argmin(score, axis=1)
    rows = np.arange(len(indices))
    return alpha_L[rows, indices], q[rows, indices], score[rows, indices]


def oracle_search(
    kappa,
    starts,
    steps,
    n_grid,
    refine_size,
    refine_stages,
    batch_size,
    safety,
):
    """Minimize true J_K without either ML model or theoretical labels."""
    kappa = np.asarray(kappa, dtype=np.float64)
    outputs = [np.empty(len(kappa), dtype=np.float64) for _ in range(3)]
    started = time.perf_counter()

    for begin in range(0, len(kappa), batch_size):
        end = min(begin + batch_size, len(kappa))
        current_kappa = kappa[begin:end]
        alpha_L, q = global_grid(len(current_kappa), n_grid, safety)
        score = exact_scores(current_kappa, alpha_L, starts, steps)
        best_alpha, best_q, best_score = select_best(alpha_L, q, score)

        radius = initial_radius(best_q, safety)
        for _ in range(refine_stages):
            alpha_ref, q_ref = local_grid(
                best_q, refine_size, radius, safety
            )
            score_ref = exact_scores(current_kappa, alpha_ref, starts, steps)
            candidate = select_best(alpha_ref, q_ref, score_ref)
            improved = candidate[2] < best_score
            best_alpha[improved] = candidate[0][improved]
            best_q[improved] = candidate[1][improved]
            best_score[improved] = candidate[2][improved]
            radius *= 0.25

        for target, values in zip(outputs, [best_alpha, best_q, best_score]):
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
):
    """Rank candidates with ML, then create labels using exact J_K only."""
    kappa = np.asarray(kappa, dtype=np.float64)
    outputs = [np.empty(len(kappa), dtype=np.float64) for _ in range(3)]
    started = time.perf_counter()

    for begin in range(0, len(kappa), batch_size):
        end = min(begin + batch_size, len(kappa))
        current_kappa = kappa[begin:end]
        alpha_L, q = global_grid(len(current_kappa), n_grid, safety)
        score = predicted_scores(model, current_kappa, alpha_L)
        predicted_best = select_best(alpha_L, q, score)

        half_top = max(1, exact_top_k // 2)
        coarse_top = take_top_candidates(alpha_L, q, score, half_top)
        predicted_radius = initial_radius(predicted_best[1], safety)
        alpha_ref, q_ref = local_grid(
            predicted_best[1], predicted_refine, predicted_radius, safety
        )
        score_ref = predicted_scores(model, current_kappa, alpha_ref)
        refined_top = take_top_candidates(
            alpha_ref, q_ref, score_ref, exact_top_k - half_top
        )

        candidates_alpha = np.column_stack([coarse_top[0], refined_top[0]])
        candidates_q = np.column_stack([coarse_top[1], refined_top[1]])
        exact = exact_scores(current_kappa, candidates_alpha, starts, steps)
        best_alpha, best_q, best_score = select_best(
            candidates_alpha, candidates_q, exact
        )

        radius = initial_radius(best_q, safety)
        for _ in range(exact_stages):
            alpha_exact, q_exact = local_grid(
                best_q, exact_refine, radius, safety
            )
            exact_score = exact_scores(
                current_kappa, alpha_exact, starts, steps
            )
            candidate = select_best(alpha_exact, q_exact, exact_score)
            improved = candidate[2] < best_score
            best_alpha[improved] = candidate[0][improved]
            best_q[improved] = candidate[1][improved]
            best_score[improved] = candidate[2][improved]
            radius *= 0.25

        for target, values in zip(outputs, [best_alpha, best_q, best_score]):
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
    alpha_theory = gd_theory(L, l)
    q_theory, _ = theory_from_kappa(kappa)
    return pd.DataFrame(
        {
            "L": L,
            "l": l,
            "kappa": kappa,
            "alpha_theory": alpha_theory,
            "q_theory": q_theory,
        }
    )


def train_q_model(X, y, seed, n_estimators):
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


def save_plots(oracle_frame, test_frame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(
        oracle_frame["alpha_theory"], oracle_frame["alpha_found"],
        s=12, alpha=0.5,
    )
    axes[1].scatter(
        oracle_frame["q_theory"], oracle_frame["q_found"],
        s=12, alpha=0.5,
    )
    for axis, name in zip(axes, ["alpha", "q"]):
        low = min(
            oracle_frame[f"{name}_theory"].min(),
            oracle_frame[f"{name}_found"].min(),
        )
        high = max(
            oracle_frame[f"{name}_theory"].max(),
            oracle_frame[f"{name}_found"].max(),
        )
        axis.plot([low, high], [low, high], "r--")
        axis.set_xlabel(f"{name} classical GD theory")
        axis.set_ylabel(f"{name} exact finite-horizon oracle")
        axis.grid(True)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "oracle_vs_theory.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].scatter(
        test_frame["alpha_found"], test_frame["alpha_pred"],
        s=12, alpha=0.45,
    )
    axes[1].scatter(
        test_frame["q_found"], test_frame["q_pred"],
        s=12, alpha=0.45,
    )
    for axis, name in zip(axes, ["alpha", "q"]):
        low = min(
            test_frame[f"{name}_found"].min(),
            test_frame[f"{name}_pred"].min(),
        )
        high = max(
            test_frame[f"{name}_found"].max(),
            test_frame[f"{name}_pred"].max(),
        )
        axis.plot([low, high], [low, high], "r--")
        axis.set_xlabel(f"{name} finite-horizon label")
        axis.set_ylabel(f"{name} final model")
        axis.grid(True)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "alpha_model_vs_finite_labels.png", dpi=160)
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Optimized exact finite-horizon Gradient Descent ML pipeline"
    )
    parser.add_argument("--kappa-min", type=float, default=KAPPA_MIN)
    parser.add_argument("--kappa-max", type=float, default=KAPPA_MAX)
    parser.add_argument("--stability-safety", type=float, default=STABILITY_SAFETY)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--starts", type=int, default=N_STARTS)
    parser.add_argument("--forward-groups", type=int, default=2400)
    parser.add_argument("--forward-candidates", type=int, default=96)
    parser.add_argument("--forward-iterations", type=int, default=350)
    parser.add_argument("--oracle-points", type=int, default=512)
    parser.add_argument("--oracle-grid", type=int, default=256)
    parser.add_argument("--oracle-refine", type=int, default=65)
    parser.add_argument("--oracle-stages", type=int, default=3)
    parser.add_argument("--meta-pairs", type=int, default=10000)
    parser.add_argument("--meta-estimators", type=int, default=600)
    parser.add_argument("--search-grid", type=int, default=192)
    parser.add_argument("--predicted-refine", type=int, default=64)
    parser.add_argument("--exact-top-k", type=int, default=24)
    parser.add_argument("--exact-refine", type=int, default=41)
    parser.add_argument("--exact-stages", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(args):
    if not 1.0 < args.kappa_min < args.kappa_max:
        raise ValueError("Expected 1 < kappa-min < kappa-max")
    if not 0.0 < args.stability_safety < 1.0:
        raise ValueError("stability-safety must be in (0, 1)")
    required_q = args.kappa_max / (args.kappa_max + 1.0)
    if args.stability_safety <= required_q:
        raise ValueError("stability-safety excludes the classical GD optimum")
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
    oracle_alpha_L, oracle_q, oracle_score = oracle_search(
        oracle_kappa,
        starts,
        args.steps,
        args.oracle_grid,
        args.oracle_refine,
        args.oracle_stages,
        args.batch_size,
        args.stability_safety,
    )
    oracle_l = np.ones_like(oracle_kappa)
    oracle_L = oracle_kappa.copy()
    oracle_alpha = oracle_alpha_L / oracle_L
    oracle_alpha_theory = gd_theory(oracle_L, oracle_l)
    oracle_q_theory, oracle_alpha_L_theory = theory_from_kappa(oracle_kappa)
    oracle_theory_score = terminal_rate(
        oracle_kappa,
        oracle_alpha_L_theory,
        starts,
        steps=args.steps,
    )
    oracle_frame = pd.DataFrame(
        {
            "L": oracle_L,
            "l": oracle_l,
            "kappa": oracle_kappa,
            "alpha_found": oracle_alpha,
            "q_found": oracle_q,
            "J_found": oracle_score,
            "alpha_theory": oracle_alpha_theory,
            "q_theory": oracle_q_theory,
            "J_theory": oracle_theory_score,
        }
    )
    oracle_frame.to_csv(DATA_DIR / "oracle_diagnostic.csv", index=False)
    print("\n2/6 Generating exact finite-horizon forward dataset")
    forward_frame = generate_forward_dataset(
        starts,
        args.forward_groups,
        args.forward_candidates,
        args.kappa_min,
        args.kappa_max,
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
        forward_model,
        MODELS_DIR / "hist_gradient_boosting_finite_rate.joblib",
    )

    print("\n4/6 Building exact-refined inverse labels")
    all_pairs = sample_meta_pairs(
        args.meta_pairs, args.kappa_min, args.kappa_max, args.seed + 10
    )
    found_alpha_L, found_q, found_score = hybrid_inverse_search(
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
    )
    all_pairs["alpha_found"] = found_alpha_L / all_pairs["L"].to_numpy()
    all_pairs["q_found"] = found_q
    all_pairs["J_found"] = found_score
    all_pairs["J_theory"] = terminal_rate(
        all_pairs["kappa"].to_numpy(),
        2.0 * all_pairs["q_theory"].to_numpy(),
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

    print("\n5/6 Training final (L, l) -> alpha model")
    X_train = meta_feature_matrix(train_frame["L"], train_frame["l"])
    q_regressor = train_q_model(
        X_train, train_frame["q_found"], args.seed, args.meta_estimators
    )
    alpha_model = GDAlphaModel(q_regressor, args.stability_safety)
    joblib.dump(alpha_model, MODELS_DIR / "gradient_boosting_alpha.joblib")

    print("\n6/6 Evaluating parameters and actual 300-step performance")
    raw_test = test_frame[["L", "l"]].to_numpy()
    test_frame["q_pred"] = alpha_model.predict_q(raw_test)
    test_frame["alpha_pred"] = alpha_model.predict(raw_test)
    test_frame["alpha_L_pred"] = (
        test_frame["alpha_pred"] * test_frame["L"]
    )
    test_frame["J_pred"] = terminal_rate(
        test_frame["kappa"].to_numpy(),
        test_frame["alpha_L_pred"].to_numpy(),
        starts,
        steps=args.steps,
    )
    test_frame.to_csv(DATA_DIR / "direct_model_test.csv", index=False)

    final_alpha_vs_theory_r2 = print_r2(
        "Final alpha vs classical GD theory",
        test_frame["alpha_theory"],
        test_frame["alpha_pred"],
    )

    save_plots(oracle_frame, test_frame)
    manifest = {
        "description": "Exact-refined finite-horizon Gradient Descent experiment",
        "external_inputs": ["L", "l"],
        "internal_direct_features": META_FEATURES,
        "forward_features": FORWARD_FEATURES,
        "saved_models": {
            "forward": str(
                MODELS_DIR / "hist_gradient_boosting_finite_rate.joblib"
            ),
            "alpha": str(MODELS_DIR / "gradient_boosting_alpha.joblib"),
        },
        "target": {
            "definition": "mean_i(log10(max(f_K/f_0, 1e-300))) / K",
            "steps": args.steps,
            "starts": args.starts,
            "forward_training_target": "sqrt(kappa) * finite_horizon_target",
        },
        "theory_baseline": "alpha = 2 / (L + l)",
        "theory_is_used_as_training_label": False,
        "config": vars(args),
        "metrics": {
            "forward_model": forward_metrics,
            "final_model_vs_classical_gd_theory": {
                "alpha_r2": final_alpha_vs_theory_r2,
            },
        },
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nSaved:")
    print(" ", RESULTS_DIR / "metrics.json")
    print(" ", RESULTS_DIR / "oracle_vs_theory.png")
    print(" ", RESULTS_DIR / "alpha_model_vs_finite_labels.png")
    print(" ", MODELS_DIR / "hist_gradient_boosting_finite_rate.joblib")
    print(" ", MODELS_DIR / "gradient_boosting_alpha.joblib")


if __name__ == "__main__":
    main()
