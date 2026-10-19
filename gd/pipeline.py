# -*- coding: utf-8 -*-
"""Inverse-oriented Gradient Descent pipeline.

The internal sklearn gradient-boosting model predicts the terminal convergence
rate for (kappa, alpha*L). Its predictions are minimized over alpha, and the
resulting labels train the final two-input model (L, l) -> alpha.
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
from sklearn.compose import TransformedTargetRegressor


matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

FORWARD_FEATURES = ["kappa", "log_kappa", "alpha_L", "stability_ratio"]
META_FEATURES = ["L", "l"]
KAPPA_MIN = 1.1
KAPPA_MAX = 20_000.0
STABILITY_SAFETY = 0.99999


def stratified_log_uniform(size, low, high, rng):
    """Cover every logarithmic part of a wide positive interval."""
    if not 0 < low < high:
        raise ValueError("Expected 0 < low < high")
    edges = np.linspace(np.log(low), np.log(high), size + 1)
    values = np.exp(rng.uniform(edges[:-1], edges[1:]))
    rng.shuffle(values)
    return values


def theory_from_kappa(kappa):
    """Dimensionless optimal step alpha*L for gradient descent."""
    kappa = np.asarray(kappa, dtype=np.float64)
    return 2.0 * kappa / (kappa + 1.0)


def theory_from_L_l(L, l):
    L = np.asarray(L, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    return 2.0 / (L + l)


def asymptotic_rate(kappa, alpha_L):
    """Return 2*log10(rho), the asymptotic per-step function-value rate."""
    kappa = np.asarray(kappa, dtype=np.float64)
    alpha_L = np.asarray(alpha_L, dtype=np.float64)
    rho = np.maximum(
        np.abs(1.0 - alpha_L),
        np.abs(1.0 - alpha_L / kappa),
    )
    return 2.0 * np.log10(np.maximum(rho, 1e-15))


def forward_feature_matrix(kappa, alpha_L):
    kappa = np.asarray(kappa, dtype=np.float64)
    alpha_L = np.asarray(alpha_L, dtype=np.float64)
    return np.column_stack(
        [
            kappa,
            np.log(kappa),
            alpha_L,
            alpha_L / 2.0,
        ]
    ).astype(np.float32)


def generate_forward_dataset(
    n_groups=2000,
    candidates_per_group=64,
    kappa_min=KAPPA_MIN,
    kappa_max=KAPPA_MAX,
    safety=STABILITY_SAFETY,
    seed=42,
):
    if candidates_per_group < 12:
        raise ValueError("candidates_per_group must be at least 12")

    rng = np.random.default_rng(seed)
    group_kappa = stratified_log_uniform(
        n_groups, kappa_min, kappa_max, rng
    )
    kappa = np.repeat(group_kappa, candidates_per_group)
    group_id = np.repeat(np.arange(n_groups), candidates_per_group)
    alpha_theory_group = theory_from_kappa(group_kappa)
    alpha_theory = np.repeat(alpha_theory_group, candidates_per_group)

    n = len(kappa)
    alpha_L = rng.uniform(1e-8, 2.0 * safety, size=n)
    sample_weight = np.ones(n, dtype=np.float64)
    sample_kind = np.full(n, "global", dtype=object)

    local_start = candidates_per_group // 2
    local_mask = (np.arange(n) % candidates_per_group) >= local_start
    alpha_L[local_mask] = alpha_theory[local_mask] * np.exp(
        rng.normal(0.0, 0.13, local_mask.sum())
    )
    alpha_L[local_mask] = np.clip(alpha_L[local_mask], 1e-8, 2.0 * safety)
    sample_weight[local_mask] = 4.0
    sample_kind[local_mask] = "local"

    anchor_scales = [1.00, 0.99, 1.01, 0.97, 1.03, 0.95, 1.05, 0.90]
    for group in range(n_groups):
        base = group * candidates_per_group
        for offset, scale in enumerate(anchor_scales):
            index = base + offset
            alpha_L[index] = np.clip(
                alpha_theory_group[group] * scale,
                1e-8,
                2.0 * safety,
            )
            sample_weight[index] = 8.0
            sample_kind[index] = "anchor"

    started = time.perf_counter()
    target = asymptotic_rate(kappa, alpha_L)
    scaled_target = kappa * target
    elapsed = time.perf_counter() - started
    print(
        f"Generated {n:,} forward rows in {elapsed:.2f}s "
        f"({n / max(elapsed, 1e-12):,.0f} rows/s)."
    )

    return pd.DataFrame(
        {
            "group_id": group_id,
            "kappa": kappa,
            "alpha_L": alpha_L,
            "stability_ratio": alpha_L / 2.0,
            "target_rate": target,
            "scaled_target_rate": scaled_target,
            "sample_weight": sample_weight,
            "sample_kind": sample_kind,
        }
    )


def train_forward_model(frame, seed=42, n_estimators=400):
    groups = frame["group_id"].unique()
    train_groups, validation_groups = train_test_split(
        groups,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )
    train_mask = frame["group_id"].isin(train_groups).to_numpy()
    validation_mask = frame["group_id"].isin(validation_groups).to_numpy()

    X = forward_feature_matrix(frame["kappa"], frame["alpha_L"])
    y = frame["scaled_target_rate"].to_numpy()
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
    model.fit(
        X[train_mask],
        y[train_mask],
        sample_weight=weights[train_mask],
    )

    prediction = model.predict(X[validation_mask])
    scaled_metrics = print_metrics(
        "Forward validation (scaled)", y[validation_mask], prediction
    )
    raw_prediction = prediction / frame.loc[validation_mask, "kappa"].to_numpy()
    raw_metrics = print_metrics(
        "Forward validation (raw rate)",
        frame.loc[validation_mask, "target_rate"],
        raw_prediction,
    )
    local_mask = validation_mask & (frame["sample_kind"] != "global").to_numpy()
    local_metrics = print_metrics(
        "Forward validation near optimum (scaled)",
        y[local_mask],
        model.predict(X[local_mask]),
    )
    return model, {
        "scaled_validation": scaled_metrics,
        "raw_rate_validation": raw_metrics,
        "scaled_near_optimum_validation": local_metrics,
    }


def predict_forward_scores(model, kappa, alpha_L):
    shape = alpha_L.shape
    kappa_rows = np.broadcast_to(kappa[:, None], shape)
    X = forward_feature_matrix(kappa_rows.reshape(-1), alpha_L.reshape(-1))
    return model.predict(X).reshape(shape)


def inverse_search(
    model,
    kappa,
    n_alpha=192,
    refine_n_alpha=96,
    batch_size=512,
    safety=STABILITY_SAFETY,
):
    kappa = np.asarray(kappa, dtype=np.float64)
    alpha_result = np.empty(len(kappa), dtype=np.float64)
    score_result = np.empty(len(kappa), dtype=np.float64)
    coarse_axis = np.linspace(1e-8, 2.0 * safety, n_alpha)
    coarse_step = coarse_axis[1] - coarse_axis[0]
    started = time.perf_counter()

    for start in range(0, len(kappa), batch_size):
        end = min(start + batch_size, len(kappa))
        current_kappa = kappa[start:end]
        coarse = np.broadcast_to(coarse_axis[None, :], (len(current_kappa), n_alpha))
        coarse_score = predict_forward_scores(model, current_kappa, coarse)
        coarse_index = np.argmin(coarse_score, axis=1)
        rows = np.arange(len(current_kappa))
        best_alpha = coarse[rows, coarse_index]
        best_score = coarse_score[rows, coarse_index]

        offsets = np.linspace(-2.0, 2.0, refine_n_alpha)
        refined = np.clip(
            best_alpha[:, None] + coarse_step * offsets[None, :],
            1e-8,
            2.0 * safety,
        )
        refined_score = predict_forward_scores(model, current_kappa, refined)
        refined_index = np.argmin(refined_score, axis=1)
        refined_alpha = refined[rows, refined_index]
        refined_best_score = refined_score[rows, refined_index]
        improved = refined_best_score < best_score
        best_alpha[improved] = refined_alpha[improved]
        best_score[improved] = refined_best_score[improved]

        alpha_result[start:end] = best_alpha
        score_result[start:end] = best_score
        elapsed = time.perf_counter() - started
        rate = end / max(elapsed, 1e-12)
        eta = (len(kappa) - end) / max(rate, 1e-12)
        print(
            f"  inverse {end:5d}/{len(kappa)} | {rate:8.1f} pairs/s | "
            f"ETA {eta:5.1f}s"
        )

    return alpha_result, score_result


def sample_meta_pairs(
    n_pairs,
    seed,
    kappa_min=KAPPA_MIN,
    kappa_max=KAPPA_MAX,
):
    rng = np.random.default_rng(seed)
    l = rng.uniform(1.0, 50.0, size=n_pairs)
    kappa = stratified_log_uniform(n_pairs, kappa_min, kappa_max, rng)
    L = l * kappa
    return pd.DataFrame(
        {
            "L": L,
            "l": l,
            "kappa": kappa,
            "alpha_theory": theory_from_L_l(L, l),
        }
    )


def train_meta_model(X, y, seed=42, n_estimators=600):
    base_model = GradientBoostingRegressor(
        loss="squared_error",
        n_estimators=n_estimators,
        learning_rate=0.035,
        max_depth=4,
        min_samples_leaf=4,
        subsample=0.9,
        random_state=seed,
    )
    model = TransformedTargetRegressor(
        regressor=base_model,
        func=np.log,
        inverse_func=np.exp,
        check_inverse=False,
    )
    model.fit(X, np.maximum(np.asarray(y), 1e-15))
    return model


def print_metrics(name, y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    metrics = {
        "r2": float(r2_score(y_true, y_pred)),
        "corr": float(np.corrcoef(y_true, y_pred)[0, 1]),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }
    print(
        f"{name}: R2={metrics['r2']:.4f} | corr={metrics['corr']:.4f} | "
        f"MAE={metrics['mae']:.8f} | RMSE={metrics['rmse']:.8f}"
    )
    return metrics


def metrics_by_kappa(frame, truth_column, prediction_column):
    """Report quality separately on each order of magnitude of kappa."""
    edges = [KAPPA_MIN, 10.0, 100.0, 1000.0, 10_000.0, KAPPA_MAX]
    result = {}
    for index, (lower, upper) in enumerate(zip(edges[:-1], edges[1:])):
        inclusive = index == len(edges) - 2
        mask = (frame["kappa"] >= lower) & (
            frame["kappa"] <= upper if inclusive else frame["kappa"] < upper
        )
        current = frame.loc[mask]
        if len(current) < 2:
            continue
        y_true = current[truth_column].to_numpy()
        y_pred = current[prediction_column].to_numpy()
        key = f"[{lower:g}, {upper:g}{']' if inclusive else ')'}"
        result[key] = {
            "count": int(len(current)),
            "r2": float(r2_score(y_true, y_pred)),
            "mae": float(mean_absolute_error(y_true, y_pred)),
            "mean_relative_error": float(
                np.mean(np.abs(y_pred - y_true) / np.maximum(np.abs(y_true), 1e-15))
            ),
        }
    return result


def save_plot(frame, path):
    theory = frame["alpha_theory"].to_numpy()
    prediction = frame["alpha_pred"].to_numpy()
    lower = min(theory.min(), prediction.min())
    upper = max(theory.max(), prediction.max())

    plt.figure(figsize=(6, 6))
    plt.scatter(theory, prediction, alpha=0.45, s=12)
    plt.plot([lower, upper], [lower, upper], "r--")
    plt.xlabel("alpha theory")
    plt.ylabel("alpha model")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(f"Gradient Descent alpha: R2={r2_score(theory, prediction):.4f}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def build_parser():
    parser = argparse.ArgumentParser(description="Alternative GD inverse pipeline")
    parser.add_argument("--kappa-min", type=float, default=KAPPA_MIN)
    parser.add_argument("--kappa-max", type=float, default=KAPPA_MAX)
    parser.add_argument("--stability-safety", type=float, default=STABILITY_SAFETY)
    parser.add_argument("--forward-groups", type=int, default=2000)
    parser.add_argument("--forward-candidates", type=int, default=64)
    parser.add_argument("--forward-estimators", type=int, default=400)
    parser.add_argument("--meta-pairs", type=int, default=20000)
    parser.add_argument("--meta-estimators", type=int, default=600)
    parser.add_argument("--n-alpha", type=int, default=192)
    parser.add_argument("--refine-alpha", type=int, default=96)
    parser.add_argument("--inverse-batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(args=None):
    args = build_parser().parse_args(args)
    if not 0 < args.kappa_min < args.kappa_max:
        raise ValueError("Expected 0 < kappa-min < kappa-max")
    if not 0 < args.stability_safety < 1:
        raise ValueError("stability-safety must be in (0, 1)")
    required_safety = args.kappa_max / (args.kappa_max + 1.0)
    if args.stability_safety <= required_safety:
        raise ValueError(
            "stability-safety must exceed kappa-max/(kappa-max+1) "
            "so the theoretical optimum remains in the search domain"
        )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n1/4 Generating inverse-oriented GD forward dataset")
    forward_frame = generate_forward_dataset(
        n_groups=args.forward_groups,
        candidates_per_group=args.forward_candidates,
        kappa_min=args.kappa_min,
        kappa_max=args.kappa_max,
        safety=args.stability_safety,
        seed=args.seed,
    )
    forward_frame.to_pickle(DATA_DIR / "forward_dataset.pkl")

    print("\n2/4 Training GD terminal-rate surrogate")
    forward_model, forward_metrics = train_forward_model(
        forward_frame,
        seed=args.seed,
        n_estimators=args.forward_estimators,
    )
    joblib.dump(forward_model, MODELS_DIR / "gradient_boosting_terminal_rate.joblib")

    print("\n3/4 Solving the inverse problem through the surrogate")
    all_pairs = sample_meta_pairs(
        args.meta_pairs,
        args.seed + 10,
        kappa_min=args.kappa_min,
        kappa_max=args.kappa_max,
    )
    train_index, test_index = train_test_split(
        np.arange(len(all_pairs)),
        test_size=0.2,
        random_state=args.seed,
        shuffle=True,
    )
    train_frame = all_pairs.iloc[train_index].reset_index(drop=True)
    test_frame = all_pairs.iloc[test_index].reset_index(drop=True)

    alpha_L_found, inverse_score = inverse_search(
        forward_model,
        train_frame["kappa"].to_numpy(),
        n_alpha=args.n_alpha,
        refine_n_alpha=args.refine_alpha,
        batch_size=args.inverse_batch_size,
        safety=args.stability_safety,
    )
    train_frame["alpha_found"] = alpha_L_found / train_frame["L"].to_numpy()
    train_frame["inverse_score"] = inverse_score
    inverse_metrics = print_metrics(
        "Inverse alpha vs theory",
        train_frame["alpha_theory"],
        train_frame["alpha_found"],
    )

    print("\n4/4 Training final (L, l) -> alpha model")
    X_train = train_frame[META_FEATURES].to_numpy()
    alpha_model = train_meta_model(
        X_train,
        train_frame["alpha_found"],
        seed=args.seed,
        n_estimators=args.meta_estimators,
    )
    X_test = test_frame[META_FEATURES].to_numpy()
    test_frame["alpha_pred"] = np.maximum(alpha_model.predict(X_test), 1e-12)
    final_metrics = print_metrics(
        "Final alpha model vs theory",
        test_frame["alpha_theory"],
        test_frame["alpha_pred"],
    )

    model_path = MODELS_DIR / "gradient_boosting_alpha.joblib"
    joblib.dump(alpha_model, model_path)
    train_frame.to_csv(DATA_DIR / "inverse_train.csv", index=False)
    test_frame.to_csv(DATA_DIR / "meta_test.csv", index=False)
    save_plot(test_frame, RESULTS_DIR / "meta_vs_theory.png")

    manifest = {
        "inputs": META_FEATURES,
        "alpha_model": str(model_path),
        "config": vars(args),
        "metrics": {
            "forward": forward_metrics,
            "inverse_alpha": inverse_metrics,
            "final_alpha": final_metrics,
        },
        "metrics_by_kappa": {
            "inverse_alpha": metrics_by_kappa(
                train_frame, "alpha_theory", "alpha_found"
            ),
            "final_alpha": metrics_by_kappa(
                test_frame, "alpha_theory", "alpha_pred"
            ),
        },
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nSaved final model:")
    print(" ", model_path)
    print("Results:")
    print(" ", RESULTS_DIR / "metrics.json")
    print(" ", RESULTS_DIR / "meta_vs_theory.png")


if __name__ == "__main__":
    main()
