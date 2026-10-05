# -*- coding: utf-8 -*-
"""Learn a surrogate validation-loss landscape for the SGD learning rate."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.datasets import load_diabetes
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn


matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"
SURROGATE_MODEL_PATH = MODELS_DIR / "gradient_boosting_lr_surrogate.joblib"


class DiabetesMLP(nn.Module):
    """Fixed architecture: 10 -> 16 -> 16 -> 1 with ReLU."""

    def __init__(self, n_features: int = 10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def set_deterministic(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)


def prepare_data(seed: int):
    dataset = load_diabetes()
    X = dataset.data.astype(np.float64)
    y = dataset.target.astype(np.float64).reshape(-1, 1)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=seed
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=seed
    )

    x_scaler = StandardScaler().fit(X_train)
    y_scaler = StandardScaler().fit(y_train)
    splits = {}
    for name, features, targets in (
        ("train", X_train, y_train),
        ("validation", X_val, y_val),
        ("test", X_test, y_test),
    ):
        splits[name] = (
            torch.tensor(x_scaler.transform(features), dtype=torch.float64),
            torch.tensor(y_scaler.transform(targets), dtype=torch.float64),
        )
    return splits, x_scaler, y_scaler


def make_initial_state(n_features: int, seed: int):
    torch.manual_seed(seed)
    model = DiabetesMLP(n_features).to(dtype=torch.float64)
    return copy.deepcopy(model.state_dict())


def build_model(initial_state, n_features: int) -> DiabetesMLP:
    model = DiabetesMLP(n_features).to(dtype=torch.float64)
    model.load_state_dict(copy.deepcopy(initial_state))
    return model


def train_with_lr(
    initial_state,
    splits,
    learning_rate: float,
    epochs: int,
    record_curve: bool = False,
    evaluate_test: bool = True,
):
    """Train from the shared initialization with constant full-batch SGD."""
    model = build_model(initial_state, splits["train"][0].shape[1])
    optimizer = torch.optim.SGD(model.parameters(), lr=float(learning_rate))
    criterion = nn.MSELoss()
    history = {"train": [], "validation": []}
    diverged = False

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(splits["train"][0]), splits["train"][1])
        if not torch.isfinite(loss) or loss.item() > 1e8:
            diverged = True
            break
        loss.backward()
        optimizer.step()

        if record_curve:
            model.eval()
            with torch.no_grad():
                history["train"].append(
                    criterion(model(splits["train"][0]), splits["train"][1]).item()
                )
                history["validation"].append(
                    criterion(
                        model(splits["validation"][0]), splits["validation"][1]
                    ).item()
                )

    model.eval()
    evaluated_splits = splits if evaluate_test else {
        name: splits[name] for name in ("train", "validation")
    }
    with torch.no_grad():
        losses = {
            name: criterion(model(features), targets).item()
            for name, (features, targets) in evaluated_splits.items()
        }
    if diverged or not all(np.isfinite(value) for value in losses.values()):
        diverged = True
        losses = {name: float("inf") for name in evaluated_splits}
    return model, history, losses, diverged


def sample_surrogate_learning_rates(
    lr_min: float, lr_max: float, count: int, seed: int
) -> np.ndarray:
    """Stratified random log-uniform design, distinct from Grid Search."""
    rng = np.random.default_rng(seed)
    edges = np.linspace(np.log10(lr_min), np.log10(lr_max), count + 1)
    log_rates = rng.uniform(edges[:-1], edges[1:])
    rng.shuffle(log_rates)
    return np.power(10.0, log_rates)


def evaluate_candidates(
    label: str,
    learning_rates: np.ndarray,
    initial_state,
    splits,
    epochs: int,
) -> pd.DataFrame:
    rows = []
    started = time.perf_counter()
    for index, learning_rate in enumerate(learning_rates, start=1):
        _, _, losses, diverged = train_with_lr(
            initial_state,
            splits,
            learning_rate,
            epochs,
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
            f"  {label} {index:2d}/{len(learning_rates)} | "
            f"lr={learning_rate:.8g} | val={losses['validation']:.6g}"
        )
    print(f"{label} completed in {time.perf_counter() - started:.1f}s")
    return pd.DataFrame(rows).sort_values("learning_rate").reset_index(drop=True)


def surrogate_features(learning_rates) -> pd.DataFrame:
    """One underlying feature: learning rate, represented on a log scale."""
    rates = np.asarray(learning_rates, dtype=float)
    return pd.DataFrame({"log10_learning_rate": np.log10(rates)})


def new_surrogate(seed: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        loss="squared_error",
        min_samples_leaf=2,
        random_state=seed,
    )


def fit_surrogate(training_data: pd.DataFrame, seed: int):
    finite = training_data[
        np.isfinite(training_data["validation_mse"])
    ].reset_index(drop=True)
    if len(finite) < 10:
        raise RuntimeError("Too few finite runs to train the sklearn surrogate")

    X = surrogate_features(finite["learning_rate"])
    y = finite["validation_mse"].to_numpy()

    # OOF scores describe interpolation quality without evaluating on test data.
    oof = np.empty(len(finite), dtype=float)
    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    for train_index, validation_index in splitter.split(X):
        fold_model = new_surrogate(seed)
        fold_model.fit(X.iloc[train_index], y[train_index])
        oof[validation_index] = fold_model.predict(X.iloc[validation_index])

    model = new_surrogate(seed)
    model.fit(X, y)
    fitted = model.predict(X)
    metrics = {
        "train_r2": float(r2_score(y, fitted)),
        "train_mae": float(mean_absolute_error(y, fitted)),
        "oof_r2": float(r2_score(y, oof)),
        "oof_mae": float(mean_absolute_error(y, oof)),
        "finite_training_runs": int(len(finite)),
    }
    return model, metrics


def minimize_surrogate(
    model: GradientBoostingRegressor,
    lr_min: float,
    lr_max: float,
    points: int = 20000,
):
    learning_rates = np.geomspace(lr_min, lr_max, points)
    predictions = model.predict(surrogate_features(learning_rates))
    minimum = float(np.min(predictions))

    # A one-dimensional tree model is constant on intervals. Use the geometric
    # midpoint of its minimum plateau instead of an arbitrary interval edge.
    minimum_mask = np.isclose(predictions, minimum, rtol=1e-12, atol=1e-14)
    minimum_indices = np.flatnonzero(minimum_mask)
    learning_rate = float(
        np.sqrt(
            learning_rates[minimum_indices[0]]
            * learning_rates[minimum_indices[-1]]
        )
    )
    curve = pd.DataFrame(
        {
            "learning_rate": learning_rates,
            "predicted_validation_mse": predictions,
        }
    )
    return learning_rate, minimum, curve


def grid_search(
    initial_state,
    splits,
    lr_min: float,
    lr_max: float,
    epochs: int,
    count: int,
):
    rates = np.geomspace(lr_min, lr_max, count)
    frame = evaluate_candidates("grid", rates, initial_state, splits, epochs)
    finite = frame[np.isfinite(frame["validation_mse"])]
    if finite.empty:
        raise RuntimeError("Every Grid Search candidate diverged")
    best = finite.loc[finite["validation_mse"].idxmin()]
    return frame, float(best["learning_rate"])


def learning_rate_finder(
    initial_state,
    splits,
    lr_min: float,
    lr_max: float,
    steps: int,
    suggestion_factor: float,
):
    """Exponential LR range test with an EMA-smoothed full-batch loss."""
    model = build_model(initial_state, splits["train"][0].shape[1])
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_min)
    criterion = nn.MSELoss()
    rates = np.geomspace(lr_min, lr_max, steps)
    beta = 0.95
    average_loss = 0.0
    best_loss = float("inf")
    rows = []

    for index, learning_rate in enumerate(rates, start=1):
        optimizer.param_groups[0]["lr"] = float(learning_rate)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(splits["train"][0]), splits["train"][1])
        if not torch.isfinite(loss):
            break
        loss.backward()
        optimizer.step()

        raw_loss = float(loss.item())
        average_loss = beta * average_loss + (1.0 - beta) * raw_loss
        smooth_loss = average_loss / (1.0 - beta**index)
        best_loss = min(best_loss, smooth_loss)
        with torch.no_grad():
            validation_loss = criterion(
                model(splits["validation"][0]), splits["validation"][1]
            ).item()
        rows.append(
            {
                "step": index,
                "learning_rate": float(learning_rate),
                "train_mse": raw_loss,
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


def evaluate_r2(model: DiabetesMLP, splits) -> dict[str, float]:
    model.eval()
    scores = {}
    with torch.no_grad():
        for name, (features, targets) in splits.items():
            actual = targets.cpu().numpy().reshape(-1)
            predicted = model(features).cpu().numpy().reshape(-1)
            scores[name] = float(r2_score(actual, predicted))
    return scores


def train_final_run(name, learning_rate, initial_state, splits, epochs, y_scale):
    model, history, losses, diverged = train_with_lr(
        initial_state, splits, learning_rate, epochs, record_curve=True
    )
    r2 = evaluate_r2(model, splits) if not diverged else {
        split: float("nan") for split in splits
    }
    torch.save(model.state_dict(), MODELS_DIR / f"{name}_model.pt")
    metrics = {
        "learning_rate": float(learning_rate),
        "diverged": diverged,
        "losses": losses,
        "r2": r2,
        "test_rmse_original_target": float(np.sqrt(losses["test"]) * y_scale),
    }
    return history, metrics


def combine_histories(histories: dict[str, dict], epochs: int) -> pd.DataFrame:
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
        selected_rates["LR Finder"], color="tab:red", linestyle="--", label="suggestion"
    )
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("learning rate")
    axes[0, 1].set_ylabel("MSE")
    axes[0, 1].set_title("Learning Rate Finder")
    axes[0, 1].grid(True)
    axes[0, 1].legend(fontsize=8)

    for method in selected_rates:
        key = method.lower().replace(" ", "_")
        axes[1, 0].plot(
            histories["epoch"], histories[f"{key}_train"], label=method
        )
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
        description="Learn validation MSE as a function of SGD learning rate"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr-min", type=float, default=1e-4)
    parser.add_argument("--lr-max", type=float, default=0.8)
    parser.add_argument("--surrogate-runs", type=int, default=36)
    parser.add_argument("--grid-candidates", type=int, default=41)
    parser.add_argument("--selection-epochs", type=int, default=300)
    parser.add_argument("--lr-finder-steps", type=int, default=140)
    parser.add_argument("--lr-finder-factor", type=float, default=0.1)
    parser.add_argument("--compare-epochs", type=int, default=500)
    return parser


def main(args=None):
    args = build_parser().parse_args(args)
    if not 0 < args.lr_min < args.lr_max:
        raise ValueError("Expected 0 < lr-min < lr-max")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    set_deterministic(args.seed)

    print("1/6 Loading Diabetes and creating the shared initialization")
    splits, x_scaler, y_scaler = prepare_data(args.seed)
    initial_state = make_initial_state(splits["train"][0].shape[1], args.seed)
    print(
        f"train={len(splits['train'][0])}, validation={len(splits['validation'][0])}, "
        f"test={len(splits['test'][0])}"
    )

    print("\n2/6 Generating supervised data for the sklearn surrogate")
    surrogate_rates = sample_surrogate_learning_rates(
        args.lr_min, args.lr_max, args.surrogate_runs, args.seed + 1
    )
    surrogate_data = evaluate_candidates(
        "surrogate",
        surrogate_rates,
        initial_state,
        splits,
        args.selection_epochs,
    )
    surrogate_data.to_csv(RESULTS_DIR / "surrogate_training_data.csv", index=False)

    print("\n3/6 Training and minimizing GradientBoosting(validation MSE | learning rate)")
    surrogate_model, surrogate_metrics = fit_surrogate(surrogate_data, args.seed)
    joblib.dump(surrogate_model, SURROGATE_MODEL_PATH)
    surrogate_lr, predicted_minimum, prediction_curve = minimize_surrogate(
        surrogate_model, args.lr_min, args.lr_max
    )
    prediction_curve.to_csv(
        RESULTS_DIR / "surrogate_prediction_curve.csv", index=False
    )
    print(f"Surrogate LR = {surrogate_lr:.10f}")
    print(
        f"Surrogate fit: train R2={surrogate_metrics['train_r2']:.4f}, "
        f"OOF R2={surrogate_metrics['oof_r2']:.4f}"
    )

    print("\n4/6 Independent Grid Search")
    grid_data, grid_lr = grid_search(
        initial_state,
        splits,
        args.lr_min,
        args.lr_max,
        args.selection_epochs,
        args.grid_candidates,
    )
    grid_data.to_csv(RESULTS_DIR / "grid_search.csv", index=False)
    print(f"Grid Search LR = {grid_lr:.10f}")

    print("\n5/6 Independent Learning Rate Finder")
    finder_data, finder_lr, finder_steepest_lr = learning_rate_finder(
        initial_state,
        splits,
        args.lr_min,
        args.lr_max,
        args.lr_finder_steps,
        args.lr_finder_factor,
    )
    finder_data.to_csv(RESULTS_DIR / "lr_finder.csv", index=False)
    print(f"LR Finder steepest point = {finder_steepest_lr:.10f}")
    print(f"LR Finder suggestion     = {finder_lr:.10f}")

    print("\n6/6 Final comparison from the same initialization")
    selected_rates = {
        "Surrogate": surrogate_lr,
        "Grid Search": grid_lr,
        "LR Finder": finder_lr,
    }
    histories = {}
    final_runs = {}
    y_scale = float(y_scaler.scale_[0])
    for display_name, learning_rate in selected_rates.items():
        key = display_name.lower().replace(" ", "_")
        history, metrics = train_final_run(
            key,
            learning_rate,
            initial_state,
            splits,
            args.compare_epochs,
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
        "architecture": "10 -> 16 -> 16 -> 1, ReLU",
        "optimizer": "torch.optim.SGD, full batch, no momentum, constant LR",
        "loss": "MSE on standardized target",
        "selection_metric": "validation MSE after fixed selection_epochs",
        "split_sizes": {name: len(data[0]) for name, data in splits.items()},
        "surrogate": {
            "feature": "log10(learning_rate); learning rate is the only input",
            "target": "validation_mse",
            "model_path": str(SURROGATE_MODEL_PATH),
            "quality": surrogate_metrics,
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
        "preprocessing": {
            "x_mean": x_scaler.mean_.tolist(),
            "x_scale": x_scaler.scale_.tolist(),
            "y_mean": y_scaler.mean_.tolist(),
            "y_scale": y_scaler.scale_.tolist(),
        },
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Saved results to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
