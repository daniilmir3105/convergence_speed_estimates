# -*- coding: utf-8 -*-
"""Empirical learning-rate selection for a fixed sklearn MLP on Boston Housing."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
OPENML_CACHE_DIR = BASE_DIR / "openml_cache"
RAW_DATA_PATH = DATA_DIR / "boston_openml_531.csv"
SURROGATE_MODEL_PATH = MODELS_DIR / "gradient_boosting_lr_surrogate.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessing.joblib"
TARGET_NAME = "MEDV"


def load_boston_data():
    """Load the 506-row OpenML Boston dataset and keep a local CSV copy."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if RAW_DATA_PATH.exists():
        frame = pd.read_csv(RAW_DATA_PATH)
        source = "local CSV cache created from OpenML data_id=531"
    else:
        dataset = fetch_openml(
            data_id=531,
            as_frame=True,
            data_home=OPENML_CACHE_DIR,
        )
        features = dataset.data.copy()
        features.columns = [str(column) for column in features.columns]
        frame = features
        frame[TARGET_NAME] = dataset.target
        frame.to_csv(RAW_DATA_PATH, index=False)
        source = "OpenML data_id=531"

    if len(frame) != 506:
        raise RuntimeError(f"Expected 506 Boston rows, got {len(frame)}")
    if TARGET_NAME not in frame:
        raise RuntimeError(f"Target column {TARGET_NAME!r} is missing")

    feature_names = [column for column in frame.columns if column != TARGET_NAME]
    numeric = frame.copy()
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric[feature_names], numeric[TARGET_NAME], feature_names, source


def clip_to_train_quantiles(values, lower_bounds, upper_bounds):
    clipped = np.clip(values, lower_bounds, upper_bounds)
    changed = int(np.count_nonzero(~np.isclose(values, clipped, equal_nan=True)))
    return clipped, changed


def prepare_data(seed, lower_quantile=0.01, upper_quantile=0.99):
    """Split first; then fit imputation, clipping and scaling on train only."""
    X, y, feature_names, source = load_boston_data()
    indices = np.arange(len(X))
    train_indices, temporary_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        shuffle=True,
    )
    validation_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=0.50,
        random_state=seed,
        shuffle=True,
    )
    raw_splits = {
        "train": (X.iloc[train_indices], y.iloc[train_indices]),
        "validation": (X.iloc[validation_indices], y.iloc[validation_indices]),
        "test": (X.iloc[test_indices], y.iloc[test_indices]),
    }

    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(raw_splits["train"][0])
    lower_bounds = np.quantile(train_imputed, lower_quantile, axis=0)
    upper_bounds = np.quantile(train_imputed, upper_quantile, axis=0)
    train_clipped, _ = clip_to_train_quantiles(
        train_imputed, lower_bounds, upper_bounds
    )

    x_scaler = StandardScaler().fit(train_clipped)
    y_train = raw_splits["train"][1].to_numpy(dtype=np.float64).reshape(-1, 1)
    y_scaler = StandardScaler().fit(y_train)

    splits = {}
    preprocessing_report = {}
    for name, (raw_features, raw_targets) in raw_splits.items():
        missing_before = int(raw_features.isna().sum().sum())
        imputed = imputer.transform(raw_features)
        clipped, clipped_values = clip_to_train_quantiles(
            imputed, lower_bounds, upper_bounds
        )
        scaled_features = x_scaler.transform(clipped).astype(np.float64)
        scaled_targets = y_scaler.transform(
            raw_targets.to_numpy(dtype=np.float64).reshape(-1, 1)
        ).reshape(-1)
        splits[name] = (scaled_features, scaled_targets)
        preprocessing_report[name] = {
            "rows": int(len(raw_features)),
            "missing_feature_values_imputed": missing_before,
            "feature_values_clipped_as_outliers": clipped_values,
        }

    preprocessing = {
        "imputer": imputer,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds,
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "feature_names": feature_names,
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
    }
    metadata = {
        "source": source,
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "feature_names": feature_names,
        "raw_missing_feature_values": int(X.isna().sum().sum()),
        "preprocessing_report": preprocessing_report,
    }
    return splits, preprocessing, metadata


def build_mlp(learning_rate, seed, train_size):
    """Fixed 13-32-32-1 MLP; sklearn creates the scalar output layer."""
    return MLPRegressor(
        hidden_layer_sizes=(32, 32),
        activation="relu",
        solver="adam",
        learning_rate="constant",
        learning_rate_init=float(learning_rate),
        batch_size=train_size,
        max_iter=1,
        shuffle=False,
        random_state=seed,
        early_stopping=False,
        tol=0.0,
        n_iter_no_change=1_000_000,
    )


def mse(model, split):
    features, targets = split
    prediction = model.predict(features)
    return float(np.mean((prediction - targets) ** 2))


def train_with_lr(
    splits,
    learning_rate,
    epochs,
    seed,
    record_curve=False,
    evaluate_test=True,
):
    """Train a fresh MLP with one persistent Adam optimizer via partial_fit."""
    model = build_mlp(learning_rate, seed, len(splits["train"][0]))
    history = {"train": [], "validation": []}
    diverged = False

    try:
        for _ in range(epochs):
            model.partial_fit(*splits["train"])
            if not np.isfinite(model.loss_) or model.loss_ > 1e12:
                diverged = True
                break
            if record_curve:
                train_loss = mse(model, splits["train"])
                validation_loss = mse(model, splits["validation"])
                history["train"].append(train_loss)
                history["validation"].append(validation_loss)
                if not np.isfinite(train_loss + validation_loss):
                    diverged = True
                    break
    except (FloatingPointError, ValueError, OverflowError):
        diverged = True

    evaluated_names = (
        ("train", "validation", "test")
        if evaluate_test
        else ("train", "validation")
    )
    if diverged:
        losses = {name: float("inf") for name in evaluated_names}
    else:
        losses = {name: mse(model, splits[name]) for name in evaluated_names}
        if not all(np.isfinite(value) for value in losses.values()):
            diverged = True
            losses = {name: float("inf") for name in evaluated_names}
    return model, history, losses, diverged


def sample_surrogate_learning_rates(lr_min, lr_max, count, seed):
    rng = np.random.default_rng(seed)
    edges = np.linspace(np.log10(lr_min), np.log10(lr_max), count + 1)
    log_rates = rng.uniform(edges[:-1], edges[1:])
    rng.shuffle(log_rates)
    return np.power(10.0, log_rates)


def evaluate_candidates(label, rates, splits, epochs, seed):
    rows = []
    started = time.perf_counter()
    for index, learning_rate in enumerate(rates, start=1):
        _, _, losses, diverged = train_with_lr(
            splits,
            learning_rate,
            epochs,
            seed,
            record_curve=False,
            evaluate_test=False,
        )
        rows.append(
            {
                "learning_rate": float(learning_rate),
                "train_mse": losses["train"],
                "validation_mse": losses["validation"],
                "diverged": diverged,
            }
        )
        print(
            f"  {label} {index:2d}/{len(rates)} | lr={learning_rate:.8g} | "
            f"val={losses['validation']:.6g}"
        )
    print(f"{label} completed in {time.perf_counter() - started:.1f}s")
    return pd.DataFrame(rows).sort_values("learning_rate").reset_index(drop=True)


def surrogate_features(learning_rates):
    rates = np.asarray(learning_rates, dtype=np.float64)
    return pd.DataFrame({"log10_learning_rate": np.log10(rates)})


def new_surrogate(seed):
    return GradientBoostingRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        loss="squared_error",
        min_samples_leaf=2,
        random_state=seed,
    )


def fit_surrogate(training_data, seed):
    finite = training_data[
        np.isfinite(training_data["validation_mse"])
    ].reset_index(drop=True)
    if len(finite) < 10:
        raise RuntimeError("Too few finite runs to train the surrogate")
    X = surrogate_features(finite["learning_rate"])
    y = finite["validation_mse"].to_numpy()

    oof = np.empty(len(finite), dtype=np.float64)
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    for train_index, validation_index in splitter.split(X):
        fold_model = new_surrogate(seed)
        fold_model.fit(X.iloc[train_index], y[train_index])
        oof[validation_index] = fold_model.predict(X.iloc[validation_index])

    model = new_surrogate(seed)
    model.fit(X, y)
    fitted = model.predict(X)
    quality = {
        "train_r2": float(r2_score(y, fitted)),
        "train_mae": float(mean_absolute_error(y, fitted)),
        "oof_r2": float(r2_score(y, oof)),
        "oof_mae": float(mean_absolute_error(y, oof)),
        "finite_training_runs": int(len(finite)),
    }
    return model, quality


def minimize_surrogate(model, lr_min, lr_max, points=20_000):
    rates = np.geomspace(lr_min, lr_max, points)
    predictions = model.predict(surrogate_features(rates))
    minimum = float(np.min(predictions))
    mask = np.isclose(predictions, minimum, rtol=1e-12, atol=1e-14)
    indices = np.flatnonzero(mask)
    learning_rate = float(np.sqrt(rates[indices[0]] * rates[indices[-1]]))
    curve = pd.DataFrame(
        {
            "learning_rate": rates,
            "predicted_validation_mse": predictions,
        }
    )
    return learning_rate, minimum, curve


def grid_search(splits, lr_min, lr_max, epochs, count, seed):
    rates = np.geomspace(lr_min, lr_max, count)
    frame = evaluate_candidates("grid", rates, splits, epochs, seed)
    finite = frame[np.isfinite(frame["validation_mse"])]
    if finite.empty:
        raise RuntimeError("Every Grid Search candidate diverged")
    best = finite.loc[finite["validation_mse"].idxmin()]
    return frame, float(best["learning_rate"])


def set_adam_learning_rate(model, learning_rate):
    """Change the base step during an LR range test without resetting Adam."""
    model.learning_rate_init = float(learning_rate)
    if hasattr(model, "_optimizer"):
        model._optimizer.learning_rate_init = float(learning_rate)
        model._optimizer.learning_rate = float(learning_rate)


def learning_rate_finder(
    splits,
    lr_min,
    lr_max,
    steps,
    suggestion_factor,
    seed,
):
    model = build_mlp(lr_min, seed, len(splits["train"][0]))
    rates = np.geomspace(lr_min, lr_max, steps)
    beta = 0.95
    average_loss = 0.0
    best_loss = float("inf")
    rows = []

    for index, learning_rate in enumerate(rates, start=1):
        set_adam_learning_rate(model, learning_rate)
        try:
            model.partial_fit(*splits["train"])
            train_loss = mse(model, splits["train"])
            validation_loss = mse(model, splits["validation"])
        except (FloatingPointError, ValueError, OverflowError):
            break
        if not np.isfinite(train_loss + validation_loss):
            break
        average_loss = beta * average_loss + (1.0 - beta) * train_loss
        smooth_loss = average_loss / (1.0 - beta**index)
        best_loss = min(best_loss, smooth_loss)
        rows.append(
            {
                "step": index,
                "learning_rate": float(learning_rate),
                "train_mse": train_loss,
                "smoothed_train_mse": smooth_loss,
                "validation_mse": validation_loss,
            }
        )
        if index > 20 and smooth_loss > 4.0 * best_loss:
            break

    frame = pd.DataFrame(rows)
    if len(frame) < 20:
        raise RuntimeError("LR Finder stopped before collecting enough points")
    log_rates = np.log10(frame["learning_rate"].to_numpy())
    smooth_losses = frame["smoothed_train_mse"].to_numpy()
    gradients = np.gradient(smooth_losses, log_rates)
    margin = max(5, len(frame) // 10)
    eligible = np.arange(margin, len(frame) - margin)
    steepest_index = int(eligible[np.argmin(gradients[eligible])])
    steepest_lr = float(frame.iloc[steepest_index]["learning_rate"])
    suggested_lr = float(np.clip(steepest_lr * suggestion_factor, lr_min, lr_max))
    frame["loss_gradient"] = gradients
    frame["is_steepest_point"] = False
    frame.loc[steepest_index, "is_steepest_point"] = True
    return frame, suggested_lr, steepest_lr


def evaluate_r2(model, splits):
    scores = {}
    for name, (features, targets) in splits.items():
        scores[name] = float(r2_score(targets, model.predict(features)))
    return scores


def train_final_run(name, learning_rate, splits, epochs, seed, y_scale):
    model, history, losses, diverged = train_with_lr(
        splits,
        learning_rate,
        epochs,
        seed,
        record_curve=True,
        evaluate_test=True,
    )
    r2 = (
        evaluate_r2(model, splits)
        if not diverged
        else {split: float("nan") for split in splits}
    )
    joblib.dump(model, MODELS_DIR / f"{name}_mlp.joblib")
    metrics = {
        "learning_rate": float(learning_rate),
        "diverged": diverged,
        "losses": losses,
        "r2": r2,
        "test_rmse_original_target": float(np.sqrt(losses["test"]) * y_scale),
    }
    return history, metrics


def combine_histories(histories, epochs):
    frame = pd.DataFrame({"epoch": np.arange(1, epochs + 1)})
    for method, history in histories.items():
        for split in ("train", "validation"):
            values = np.full(epochs, np.nan)
            recorded = history[split]
            values[: len(recorded)] = recorded
            frame[f"{method}_{split}"] = values
    return frame


def save_plot(
    training_data,
    prediction_curve,
    grid_data,
    finder_data,
    histories,
    selected_rates,
    path,
):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].scatter(
        training_data["learning_rate"],
        training_data["validation_mse"],
        s=28,
        label="surrogate training runs",
    )
    axes[0, 0].plot(
        prediction_curve["learning_rate"],
        prediction_curve["predicted_validation_mse"],
        color="black",
        linewidth=1.5,
        label="GradientBoosting prediction",
    )
    axes[0, 0].scatter(
        grid_data["learning_rate"],
        grid_data["validation_mse"],
        s=14,
        alpha=0.6,
        label="independent grid",
    )
    for method, learning_rate in selected_rates.items():
        axes[0, 0].axvline(learning_rate, linestyle="--", label=method)
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("learning rate")
    axes[0, 0].set_ylabel("validation MSE after selection epochs")
    axes[0, 0].set_title("Learned objective landscape")
    axes[0, 0].grid(True)
    axes[0, 0].legend(fontsize=8)

    axes[0, 1].plot(
        finder_data["learning_rate"],
        finder_data["smoothed_train_mse"],
        label="smoothed train MSE",
    )
    axes[0, 1].plot(
        finder_data["learning_rate"],
        finder_data["validation_mse"],
        alpha=0.65,
        label="validation MSE",
    )
    axes[0, 1].axvline(
        selected_rates["LR Finder"],
        color="tab:red",
        linestyle="--",
        label="suggestion",
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("learning rate")
    axes[0, 1].set_ylabel("MSE")
    axes[0, 1].set_title("Learning Rate Finder")
    axes[0, 1].grid(True)
    axes[0, 1].legend(fontsize=8)

    for method in selected_rates:
        key = method.lower().replace(" ", "_")
        axes[1, 0].plot(histories["epoch"], histories[f"{key}_train"], label=method)
        axes[1, 1].plot(
            histories["epoch"], histories[f"{key}_validation"], label=method
        )
    for axis, title, ylabel in (
        (axes[1, 0], "Final training curves", "train MSE"),
        (axes[1, 1], "Final validation curves", "validation MSE"),
    ):
        axis.set_xlabel("epoch")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(True)
        axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Boston Housing sklearn MLP learning-rate experiment"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-max", type=float, default=0.1)
    parser.add_argument("--surrogate-runs", type=int, default=36)
    parser.add_argument("--grid-candidates", type=int, default=41)
    parser.add_argument("--selection-epochs", type=int, default=300)
    parser.add_argument("--lr-finder-steps", type=int, default=140)
    parser.add_argument("--lr-finder-factor", type=float, default=1.0)
    parser.add_argument("--compare-epochs", type=int, default=500)
    parser.add_argument("--outlier-lower-quantile", type=float, default=0.01)
    parser.add_argument("--outlier-upper-quantile", type=float, default=0.99)
    return parser


def main(args=None):
    args = build_parser().parse_args(args)
    if not 0 < args.lr_min < args.lr_max:
        raise ValueError("Expected 0 < lr-min < lr-max")
    if not 0 <= args.outlier_lower_quantile < args.outlier_upper_quantile <= 1:
        raise ValueError("Invalid outlier quantiles")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("1/6 Loading and preprocessing Boston Housing")
    splits, preprocessing, dataset_metadata = prepare_data(
        args.seed,
        args.outlier_lower_quantile,
        args.outlier_upper_quantile,
    )
    joblib.dump(preprocessing, PREPROCESSOR_PATH)
    print(
        f"train={len(splits['train'][0])}, validation={len(splits['validation'][0])}, "
        f"test={len(splits['test'][0])}"
    )
    print(
        "raw missing values=",
        dataset_metadata["raw_missing_feature_values"],
        "| clipped train values=",
        dataset_metadata["preprocessing_report"]["train"][
            "feature_values_clipped_as_outliers"
        ],
    )

    print("\n2/6 Generating supervised data for the surrogate")
    surrogate_rates = sample_surrogate_learning_rates(
        args.lr_min, args.lr_max, args.surrogate_runs, args.seed + 1
    )
    surrogate_data = evaluate_candidates(
        "surrogate",
        surrogate_rates,
        splits,
        args.selection_epochs,
        args.seed,
    )
    surrogate_data.to_csv(RESULTS_DIR / "surrogate_training_data.csv", index=False)

    print("\n3/6 Training and minimizing GradientBoosting(validation MSE | LR)")
    surrogate_model, surrogate_quality = fit_surrogate(surrogate_data, args.seed)
    joblib.dump(surrogate_model, SURROGATE_MODEL_PATH)
    surrogate_lr, predicted_minimum, prediction_curve = minimize_surrogate(
        surrogate_model, args.lr_min, args.lr_max
    )
    prediction_curve.to_csv(
        RESULTS_DIR / "surrogate_prediction_curve.csv", index=False
    )
    print(f"Surrogate LR = {surrogate_lr:.10f}")
    print(
        f"Surrogate fit: train R2={surrogate_quality['train_r2']:.4f}, "
        f"OOF R2={surrogate_quality['oof_r2']:.4f}"
    )

    print("\n4/6 Independent Grid Search")
    grid_data, grid_lr = grid_search(
        splits,
        args.lr_min,
        args.lr_max,
        args.selection_epochs,
        args.grid_candidates,
        args.seed,
    )
    grid_data.to_csv(RESULTS_DIR / "grid_search.csv", index=False)
    print(f"Grid Search LR = {grid_lr:.10f}")

    print("\n5/6 Independent Learning Rate Finder")
    finder_data, finder_lr, finder_steepest_lr = learning_rate_finder(
        splits,
        args.lr_min,
        args.lr_max,
        args.lr_finder_steps,
        args.lr_finder_factor,
        args.seed,
    )
    finder_data.to_csv(RESULTS_DIR / "lr_finder.csv", index=False)
    print(f"LR Finder steepest point = {finder_steepest_lr:.10f}")
    print(f"LR Finder suggestion     = {finder_lr:.10f}")

    print("\n6/6 Final comparison from identical sklearn initializations")
    selected_rates = {
        "Surrogate": surrogate_lr,
        "Grid Search": grid_lr,
        "LR Finder": finder_lr,
    }
    histories = {}
    final_runs = {}
    y_scale = float(preprocessing["y_scaler"].scale_[0])
    for display_name, learning_rate in selected_rates.items():
        key = display_name.lower().replace(" ", "_")
        history, metrics = train_final_run(
            key,
            learning_rate,
            splits,
            args.compare_epochs,
            args.seed,
            y_scale,
        )
        histories[key] = history
        final_runs[key] = metrics
        print(
            f"{display_name:11s} | lr={learning_rate:.8g} | "
            f"val MSE={metrics['losses']['validation']:.6f} | "
            f"test R2={metrics['r2']['test']:.4f} | "
            f"test RMSE={metrics['test_rmse_original_target']:.3f}"
        )

    history_data = combine_histories(histories, args.compare_epochs)
    history_data.to_csv(RESULTS_DIR / "learning_curves.csv", index=False)
    save_plot(
        surrogate_data,
        prediction_curve,
        grid_data,
        finder_data,
        history_data,
        selected_rates,
        RESULTS_DIR / "comparison.png",
    )

    result = {
        "config": vars(args),
        "dataset": dataset_metadata,
        "architecture": "13 -> 32 -> 32 -> 1, ReLU",
        "estimator": "sklearn.neural_network.MLPRegressor",
        "optimizer": "Adam, constant base learning rate, full-batch, no shuffling",
        "loss": "MSE on standardized target",
        "selection_metric": "validation MSE after fixed selection_epochs",
        "split_sizes": {name: int(len(data[0])) for name, data in splits.items()},
        "preprocessing": {
            "missing_values": "median imputation fitted on train",
            "outliers": (
                "feature-wise winsorization fitted on train at "
                f"[{args.outlier_lower_quantile}, {args.outlier_upper_quantile}]"
            ),
            "scaling": "StandardScaler fitted on train for X and y",
            "artifact": str(PREPROCESSOR_PATH),
        },
        "surrogate": {
            "feature": "log10(learning_rate); learning rate is the only input",
            "target": "validation_mse",
            "model_path": str(SURROGATE_MODEL_PATH),
            "quality": surrogate_quality,
            "learning_rate": surrogate_lr,
            "predicted_validation_mse": predicted_minimum,
        },
        "grid_search": {"learning_rate": grid_lr},
        "lr_finder": {
            "learning_rate": finder_lr,
            "steepest_descent_learning_rate": finder_steepest_lr,
            "suggestion_factor": args.lr_finder_factor,
        },
        "comparisons": {
            "absolute_difference_surrogate_vs_grid": abs(surrogate_lr - grid_lr),
            "ratio_surrogate_to_grid": surrogate_lr / grid_lr,
            "absolute_difference_surrogate_vs_lr_finder": abs(
                surrogate_lr - finder_lr
            ),
            "ratio_surrogate_to_lr_finder": surrogate_lr / finder_lr,
        },
        "final_runs": final_runs,
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Saved results to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
