# -*- coding: utf-8 -*-
"""Alternative Heavy Ball pipeline tailored to the inverse problem.

Pipeline:
1. Generate exact terminal convergence-rate data without early stopping.
2. Train a dedicated sklearn gradient-boosting surrogate for candidate ranking.
3. Jointly minimize that surrogate over (alpha, beta).
4. Train two final gradient-boosting models: (L, l) -> alpha and beta.
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
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

FORWARD_FEATURES = [
    "kappa",
    "log_kappa",
    "alpha_L",
    "beta",
    "stability_ratio",
]
META_FEATURES = ["L", "l"]


def theory_from_kappa(kappa):
    """Return dimensionless alpha*L and beta* for Polyak Heavy Ball."""
    kappa = np.asarray(kappa, dtype=np.float64)
    sqrt_kappa = np.sqrt(kappa)
    alpha_L = 4.0 * kappa / (sqrt_kappa + 1.0) ** 2
    beta = ((sqrt_kappa - 1.0) / (sqrt_kappa + 1.0)) ** 2
    return alpha_L, beta


def theory_from_L_l(L, l):
    L = np.asarray(L, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    alpha_L, beta = theory_from_kappa(L / l)
    return alpha_L / L, beta


def make_start_points(n_starts, seed):
    """Fixed starts make every candidate score deterministic and comparable."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(n_starts, 2))
    tiny = np.linalg.norm(x0, axis=1) < 1e-8
    x0[tiny, 0] = 1.0
    return x0.astype(np.float64)


def terminal_rate(
    kappa,
    alpha_L,
    beta,
    x0,
    steps=300,
    chunk_size=4096,
):
    """Exact mean log10(f_K/f_0)/K for normalized spectrum (kappa, 1).

    No convergence-based early stopping is used. The 1e-300 floor preserves
    the dynamic range needed to distinguish fast candidates at iteration 300.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    alpha_L = np.asarray(alpha_L, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    result = np.empty(len(kappa), dtype=np.float64)

    for start in range(0, len(kappa), chunk_size):
        end = min(start + chunk_size, len(kappa))
        k = kappa[start:end]
        u = alpha_L[start:end]
        b = beta[start:end]
        n = len(k)

        x = np.broadcast_to(x0[None, :, :], (n, len(x0), 2)).copy()
        x_prev = x.copy()
        coefficient = np.stack(
            [
                1.0 - u + b,
                1.0 - u / k + b,
            ],
            axis=1,
        )[:, None, :]

        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            for _ in range(steps):
                x_next = coefficient * x - b[:, None, None] * x_prev
                x_prev, x = x, x_next

            f0 = 0.5 * (
                k[:, None] * x0[None, :, 0] ** 2
                + x0[None, :, 1] ** 2
            )
            fK = 0.5 * (
                k[:, None] * x[..., 0] ** 2
                + x[..., 1] ** 2
            )
            ratio = fK / np.maximum(f0, 1e-300)
            log_ratio = np.log10(np.maximum(ratio, 1e-300))
            score = np.mean(log_ratio, axis=1) / steps

        result[start:end] = np.nan_to_num(
            score,
            nan=1.0,
            posinf=1.0,
            neginf=-1.0,
        )

    return result


def forward_feature_matrix(kappa, alpha_L, beta):
    kappa = np.asarray(kappa, dtype=np.float64)
    alpha_L = np.asarray(alpha_L, dtype=np.float64)
    beta = np.asarray(beta, dtype=np.float64)
    stability_ratio = alpha_L / np.maximum(2.0 * (1.0 + beta), 1e-12)
    return np.column_stack(
        [
            kappa,
            np.log(kappa),
            alpha_L,
            beta,
            stability_ratio,
        ]
    ).astype(np.float32)


def generate_forward_dataset(
    n_groups=1200,
    candidates_per_group=64,
    n_starts=16,
    steps=300,
    safety=0.95,
    seed=42,
):
    """Generate broad and optimum-focused candidate data for the surrogate."""
    if candidates_per_group < 16:
        raise ValueError("candidates_per_group must be at least 16")

    rng = np.random.default_rng(seed)
    group_kappa = rng.uniform(1.1, 20.0, size=n_groups)
    kappa = np.repeat(group_kappa, candidates_per_group)
    group_id = np.repeat(np.arange(n_groups), candidates_per_group)

    alpha_theory_group, beta_theory_group = theory_from_kappa(group_kappa)
    alpha_theory = np.repeat(alpha_theory_group, candidates_per_group)
    beta_theory = np.repeat(beta_theory_group, candidates_per_group)

    n = len(kappa)
    beta = rng.uniform(0.0, 0.99, size=n)
    stability_ratio = rng.uniform(0.005, safety, size=n)
    alpha_L = 2.0 * (1.0 + beta) * stability_ratio
    sample_weight = np.ones(n, dtype=np.float64)
    sample_kind = np.full(n, "global", dtype=object)

    # Half of every group densely samples the neighborhood of the optimum.
    local_start = candidates_per_group // 2
    local_mask = (np.arange(n) % candidates_per_group) >= local_start
    beta[local_mask] = np.clip(
        beta_theory[local_mask] + rng.normal(0.0, 0.10, local_mask.sum()),
        0.0,
        0.99,
    )
    alpha_L[local_mask] = alpha_theory[local_mask] * np.exp(
        rng.normal(0.0, 0.16, local_mask.sum())
    )
    stable_max = safety * 2.0 * (1.0 + beta[local_mask])
    alpha_L[local_mask] = np.clip(alpha_L[local_mask], 1e-8, stable_max)
    sample_weight[local_mask] = 4.0
    sample_kind[local_mask] = "local"

    # Include the exact theoretical point and small deterministic offsets in
    # every group so the model sees the bottom of the objective basin.
    offsets = [
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
        for offset_index, (alpha_scale, beta_shift) in enumerate(offsets):
            index = base + offset_index
            beta[index] = np.clip(beta_theory_group[group] + beta_shift, 0.0, 0.99)
            alpha_L[index] = alpha_theory_group[group] * alpha_scale
            alpha_L[index] = min(
                alpha_L[index],
                safety * 2.0 * (1.0 + beta[index]),
            )
            sample_weight[index] = 8.0
            sample_kind[index] = "anchor"

    x0 = make_start_points(n_starts, seed + 1)
    started = time.perf_counter()
    target = terminal_rate(
        kappa,
        alpha_L,
        beta,
        x0,
        steps=steps,
    )
    elapsed = time.perf_counter() - started
    print(
        f"Generated {n:,} forward rows in {elapsed:.1f}s "
        f"({n / max(elapsed, 1e-12):,.0f} rows/s)."
    )

    return pd.DataFrame(
        {
            "group_id": group_id,
            "kappa": kappa,
            "alpha_L": alpha_L,
            "beta": beta,
            "stability_ratio": alpha_L / (2.0 * (1.0 + beta)),
            "target_rate": target,
            "sample_weight": sample_weight,
            "sample_kind": sample_kind,
        }
    )


def train_forward_model(frame, seed=42, n_estimators=400):
    """Split by kappa groups so validation measures real generalization."""
    groups = frame["group_id"].unique()
    train_groups, validation_groups = train_test_split(
        groups,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )
    train_mask = frame["group_id"].isin(train_groups)
    validation_mask = frame["group_id"].isin(validation_groups)

    X = forward_feature_matrix(frame["kappa"], frame["alpha_L"], frame["beta"])
    y = frame["target_rate"].to_numpy()
    weights = frame["sample_weight"].to_numpy()

    model = GradientBoostingRegressor(
        loss="squared_error",
        n_estimators=n_estimators,
        learning_rate=0.04,
        max_depth=5,
        min_samples_leaf=5,
        subsample=0.85,
        random_state=seed,
    )
    model.fit(X[train_mask], y[train_mask], sample_weight=weights[train_mask])

    prediction = model.predict(X[validation_mask])
    print_metrics("Forward validation", y[validation_mask], prediction)

    local_validation = validation_mask & (frame["sample_kind"] != "global").to_numpy()
    local_prediction = model.predict(X[local_validation])
    print_metrics("Forward validation near optimum", y[local_validation], local_prediction)
    return model


def predict_forward_scores(model, kappa, alpha_L, beta):
    shape = alpha_L.shape
    kappa_rows = np.broadcast_to(kappa[:, None], shape)
    X = forward_feature_matrix(
        kappa_rows.reshape(-1),
        alpha_L.reshape(-1),
        beta.reshape(-1),
    )
    return model.predict(X).reshape(shape)


def coarse_grid(kappa, n_alpha, n_beta, beta_max=0.99, safety=0.95):
    beta_axis = np.linspace(0.0, beta_max, n_beta)
    fractions = np.linspace(0.0, safety, n_alpha)
    beta = np.broadcast_to(
        beta_axis[None, :, None],
        (len(kappa), n_beta, n_alpha),
    )
    alpha_L = 1e-8 + (
        2.0 * (1.0 + beta) - 1e-8
    ) * fractions[None, None, :]
    return alpha_L.reshape(len(kappa), -1), beta.reshape(len(kappa), -1)


def refined_grid(
    best_alpha_L,
    best_beta,
    coarse_n_alpha,
    coarse_n_beta,
    refine_n_alpha,
    refine_n_beta,
    beta_max=0.99,
    safety=0.95,
):
    beta_step = beta_max / max(coarse_n_beta - 1, 1)
    beta_offsets = np.linspace(-2.0, 2.0, refine_n_beta)
    beta = np.clip(
        best_beta[:, None] + beta_step * beta_offsets[None, :],
        0.0,
        beta_max,
    )

    alpha_step = (
        safety * 2.0 * (1.0 + best_beta) / max(coarse_n_alpha - 1, 1)
    )
    alpha_offsets = np.linspace(-2.0, 2.0, refine_n_alpha)
    alpha_L = best_alpha_L[:, None, None] + (
        alpha_step[:, None, None] * alpha_offsets[None, None, :]
    )
    alpha_L = np.maximum(alpha_L, 1e-8)
    stable_max = safety * 2.0 * (1.0 + beta)
    alpha_L = np.minimum(alpha_L, stable_max[:, :, None])

    beta = np.broadcast_to(
        beta[:, :, None],
        (len(best_beta), refine_n_beta, refine_n_alpha),
    )
    return alpha_L.reshape(len(best_beta), -1), beta.reshape(len(best_beta), -1)


def select_best(alpha_L, beta, score):
    index = np.argmin(score, axis=1)
    rows = np.arange(len(index))
    return alpha_L[rows, index], beta[rows, index], score[rows, index]


def inverse_search(
    model,
    kappa,
    n_alpha=32,
    n_beta=32,
    refine_n_alpha=16,
    refine_n_beta=16,
    batch_size=256,
    beta_max=0.99,
    safety=0.95,
):
    kappa = np.asarray(kappa, dtype=np.float64)
    alpha_result = np.empty(len(kappa), dtype=np.float64)
    beta_result = np.empty(len(kappa), dtype=np.float64)
    score_result = np.empty(len(kappa), dtype=np.float64)
    started = time.perf_counter()

    for start in range(0, len(kappa), batch_size):
        end = min(start + batch_size, len(kappa))
        current_kappa = kappa[start:end]

        alpha_L, beta = coarse_grid(
            current_kappa,
            n_alpha,
            n_beta,
            beta_max=beta_max,
            safety=safety,
        )
        score = predict_forward_scores(model, current_kappa, alpha_L, beta)
        best_alpha, best_beta, best_score = select_best(alpha_L, beta, score)

        alpha_refined, beta_refined = refined_grid(
            best_alpha,
            best_beta,
            n_alpha,
            n_beta,
            refine_n_alpha,
            refine_n_beta,
            beta_max=beta_max,
            safety=safety,
        )
        refined_score = predict_forward_scores(
            model,
            current_kappa,
            alpha_refined,
            beta_refined,
        )
        refined_alpha, refined_beta, refined_best_score = select_best(
            alpha_refined,
            beta_refined,
            refined_score,
        )
        improved = refined_best_score < best_score
        best_alpha[improved] = refined_alpha[improved]
        best_beta[improved] = refined_beta[improved]
        best_score[improved] = refined_best_score[improved]

        alpha_result[start:end] = best_alpha
        beta_result[start:end] = best_beta
        score_result[start:end] = best_score

        elapsed = time.perf_counter() - started
        rate = end / max(elapsed, 1e-12)
        eta = (len(kappa) - end) / max(rate, 1e-12)
        print(
            f"  inverse {end:5d}/{len(kappa)} | {rate:7.1f} pairs/s | "
            f"ETA {eta:5.1f}s"
        )

    return alpha_result, beta_result, score_result


def sample_meta_pairs(n_pairs, seed):
    rng = np.random.default_rng(seed)
    l = rng.uniform(1.0, 50.0, size=n_pairs)
    kappa = rng.uniform(1.1, 20.0, size=n_pairs)
    L = l * kappa
    alpha_theory, beta_theory = theory_from_L_l(L, l)
    return pd.DataFrame(
        {
            "L": L,
            "l": l,
            "kappa": kappa,
            "alpha_theory": alpha_theory,
            "beta_theory": beta_theory,
        }
    )


def train_meta_model(X, y, seed, n_estimators):
    model = GradientBoostingRegressor(
        loss="squared_error",
        n_estimators=n_estimators,
        learning_rate=0.035,
        max_depth=4,
        min_samples_leaf=4,
        subsample=0.9,
        random_state=seed,
    )
    model.fit(X, y)
    return model


def print_metrics(name, y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    print(
        f"{name}: R2={r2:.4f} | corr={corr:.4f} | "
        f"MAE={mae:.8f} | RMSE={rmse:.8f}"
    )
    return {"r2": float(r2), "corr": float(corr), "mae": float(mae), "rmse": float(rmse)}


def save_result_plot(test_frame, path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, name in zip(axes, ["alpha", "beta"]):
        theory = test_frame[f"{name}_theory"].to_numpy()
        prediction = test_frame[f"{name}_pred"].to_numpy()
        lower = min(theory.min(), prediction.min())
        upper = max(theory.max(), prediction.max())
        score = r2_score(theory, prediction)
        axis.scatter(theory, prediction, alpha=0.45, s=12)
        axis.plot([lower, upper], [lower, upper], "r--")
        axis.set_xlabel(f"{name} theory")
        axis.set_ylabel(f"{name} model")
        axis.set_title(f"{name}: R2={score:.4f}")
        axis.grid(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description="Alternative Heavy Ball ML pipeline")
    parser.add_argument("--forward-groups", type=int, default=1200)
    parser.add_argument("--forward-candidates", type=int, default=64)
    parser.add_argument("--forward-starts", type=int, default=16)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--forward-estimators", type=int, default=400)
    parser.add_argument("--meta-pairs", type=int, default=5000)
    parser.add_argument("--meta-estimators", type=int, default=600)
    parser.add_argument("--n-alpha", type=int, default=32)
    parser.add_argument("--n-beta", type=int, default=32)
    parser.add_argument("--refine-alpha", type=int, default=16)
    parser.add_argument("--refine-beta", type=int, default=16)
    parser.add_argument("--inverse-batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(args=None):
    args = build_parser().parse_args(args)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n1/4 Generating inverse-oriented forward dataset")
    forward_frame = generate_forward_dataset(
        n_groups=args.forward_groups,
        candidates_per_group=args.forward_candidates,
        n_starts=args.forward_starts,
        steps=args.steps,
        seed=args.seed,
    )
    forward_frame.to_pickle(DATA_DIR / "forward_dataset.pkl")

    print("\n2/4 Training terminal-rate sklearn GradientBoosting surrogate")
    forward_model = train_forward_model(
        forward_frame,
        seed=args.seed,
        n_estimators=args.forward_estimators,
    )
    joblib.dump(forward_model, MODELS_DIR / "gradient_boosting_terminal_rate.joblib")

    print("\n3/4 Solving the joint inverse problem through the surrogate")
    all_pairs = sample_meta_pairs(args.meta_pairs, args.seed + 10)
    train_index, test_index = train_test_split(
        np.arange(len(all_pairs)),
        test_size=0.2,
        random_state=args.seed,
        shuffle=True,
    )
    train_frame = all_pairs.iloc[train_index].reset_index(drop=True)
    test_frame = all_pairs.iloc[test_index].reset_index(drop=True)

    alpha_L_found, beta_found, inverse_score = inverse_search(
        forward_model,
        train_frame["kappa"].to_numpy(),
        n_alpha=args.n_alpha,
        n_beta=args.n_beta,
        refine_n_alpha=args.refine_alpha,
        refine_n_beta=args.refine_beta,
        batch_size=args.inverse_batch_size,
    )
    train_frame["alpha_found"] = alpha_L_found / train_frame["L"].to_numpy()
    train_frame["beta_found"] = beta_found
    train_frame["inverse_score"] = inverse_score

    inverse_alpha_metrics = print_metrics(
        "Inverse alpha vs theory",
        train_frame["alpha_theory"],
        train_frame["alpha_found"],
    )
    inverse_beta_metrics = print_metrics(
        "Inverse beta  vs theory",
        train_frame["beta_theory"],
        train_frame["beta_found"],
    )

    print("\n4/4 Training final (L, l) -> alpha/beta models")
    X_train = train_frame[META_FEATURES].to_numpy()
    alpha_model = train_meta_model(
        X_train,
        train_frame["alpha_found"],
        args.seed,
        args.meta_estimators,
    )
    beta_model = train_meta_model(
        X_train,
        train_frame["beta_found"],
        args.seed,
        args.meta_estimators,
    )

    X_test = test_frame[META_FEATURES].to_numpy()
    test_frame["alpha_pred"] = np.maximum(alpha_model.predict(X_test), 1e-12)
    test_frame["beta_pred"] = np.clip(beta_model.predict(X_test), 0.0, 0.99)

    alpha_metrics = print_metrics(
        "Final alpha model vs theory",
        test_frame["alpha_theory"],
        test_frame["alpha_pred"],
    )
    beta_metrics = print_metrics(
        "Final beta  model vs theory",
        test_frame["beta_theory"],
        test_frame["beta_pred"],
    )

    alpha_model_path = MODELS_DIR / "gradient_boosting_alpha.joblib"
    beta_model_path = MODELS_DIR / "gradient_boosting_beta.joblib"
    joblib.dump(alpha_model, alpha_model_path)
    joblib.dump(beta_model, beta_model_path)
    train_frame.to_csv(DATA_DIR / "inverse_train.csv", index=False)
    test_frame.to_csv(DATA_DIR / "meta_test.csv", index=False)
    save_result_plot(test_frame, RESULTS_DIR / "meta_vs_theory.png")

    manifest = {
        "inputs": META_FEATURES,
        "alpha_model": str(alpha_model_path),
        "beta_model": str(beta_model_path),
        "config": vars(args),
        "metrics": {
            "inverse_alpha": inverse_alpha_metrics,
            "inverse_beta": inverse_beta_metrics,
            "final_alpha": alpha_metrics,
            "final_beta": beta_metrics,
        },
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSaved final models:")
    print(" ", alpha_model_path)
    print(" ", beta_model_path)
    print("Results:")
    print(" ", RESULTS_DIR / "metrics.json")
    print(" ", RESULTS_DIR / "meta_vs_theory.png")


if __name__ == "__main__":
    main()

