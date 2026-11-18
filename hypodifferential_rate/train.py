# -*- coding: utf-8 -*-
"""Train a model that predicts J_300 for hypodifferential descent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from .core import FEATURE_NAMES


matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


CANDIDATE_CONFIGS = [
    {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 2},
    {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 2},
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 2},
    {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 3},
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3},
    {"n_estimators": 400, "learning_rate": 0.03, "max_depth": 4},
]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train J_300 predictor for hypodifferential descent"
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser


def make_model(config, seed):
    return GradientBoostingRegressor(
        loss="huber",
        n_estimators=config["n_estimators"],
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        min_samples_leaf=5,
        subsample=0.90,
        random_state=seed,
    )


def three_way_split(frame, seed):
    indices = np.arange(len(frame))
    n_bins = min(10, max(2, len(frame) // 20))
    bins = pd.qcut(
        frame["log_kappa"], q=n_bins, labels=False, duplicates="drop"
    ).to_numpy()
    try:
        train_validation, test = train_test_split(
            indices,
            test_size=0.20,
            random_state=seed,
            shuffle=True,
            stratify=bins,
        )
        train, validation = train_test_split(
            train_validation,
            test_size=0.25,
            random_state=seed + 1,
            shuffle=True,
            stratify=bins[train_validation],
        )
    except ValueError:
        train_validation, test = train_test_split(
            indices, test_size=0.20, random_state=seed, shuffle=True
        )
        train, validation = train_test_split(
            train_validation,
            test_size=0.25,
            random_state=seed + 1,
            shuffle=True,
        )
    return np.asarray(train), np.asarray(validation), np.asarray(test)


def select_model(X_train, y_train, X_validation, y_validation, seed):
    records = []
    best_model = None
    best_mse = np.inf
    best_config = None
    for number, config in enumerate(CANDIDATE_CONFIGS, start=1):
        model = make_model(config, seed + number)
        model.fit(X_train, y_train)
        prediction = model.predict(X_validation)
        mse = float(mean_squared_error(y_validation, prediction))
        record = {"candidate": number, **config, "validation_mse": mse}
        records.append(record)
        print(
            f"candidate={number} depth={config['max_depth']} "
            f"estimators={config['n_estimators']} "
            f"lr={config['learning_rate']:.3f} validation MSE={mse:.10g}"
        )
        if mse < best_mse:
            best_mse = mse
            best_model = model
            best_config = config.copy()
    return best_model, best_config, best_mse, pd.DataFrame(records)


def metric_block(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }


def save_plots(test_frame, feature_importance):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, true_name, pred_name, title in [
        (axes[0], "J300", "J300_pred", "Finite-horizon rate J_300"),
        (axes[1], "rho", "rho_pred", "Per-iteration contraction rho"),
    ]:
        axis.scatter(test_frame[true_name], test_frame[pred_name], s=18, alpha=0.55)
        low = min(test_frame[true_name].min(), test_frame[pred_name].min())
        high = max(test_frame[true_name].max(), test_frame[pred_name].max())
        axis.plot([low, high], [low, high], "r--")
        axis.set_xlabel(f"true {true_name}")
        axis.set_ylabel(f"predicted {true_name}")
        axis.set_title(title)
        axis.grid(True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "test_predictions.png", dpi=170)
    plt.close(fig)

    ordered = feature_importance.sort_values("importance", ascending=True).tail(15)
    fig, axis = plt.subplots(figsize=(9, 7))
    axis.barh(ordered["feature"], ordered["importance"])
    axis.set_xlabel("Gradient Boosting feature importance")
    axis.grid(True, axis="x")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=170)
    plt.close(fig)

    trajectory_path = DATA_DIR / "sample_trajectories.csv"
    if trajectory_path.exists():
        trajectories = pd.read_csv(trajectory_path)
        if not trajectories.empty:
            if "relative_gap" in trajectories.columns:
                relative_column = "relative_gap"
            elif "relative_error" in trajectories.columns:
                # Compatibility with datasets produced by the briefly used
                # full-trajectory format. Both columns contain the same
                # normalized quantity needed only for this diagnostic plot.
                relative_column = "relative_error"
            else:
                relative_column = None
        if not trajectories.empty and relative_column is not None:
            fig, axis = plt.subplots(figsize=(10, 6))
            for (instance_id, start_id), group in trajectories.groupby(
                ["instance_id", "start_id"]
            ):
                if start_id >= 3:
                    continue
                axis.plot(
                    group["iteration"],
                    np.log10(np.maximum(group[relative_column], 1e-300)),
                    alpha=0.55,
                    linewidth=1.0,
                    label=f"instance {instance_id}" if start_id == 0 else None,
                )
            axis.set_xlabel("iteration")
            axis.set_ylabel("log10 relative objective gap")
            axis.set_title("Sample convergence trajectories")
            axis.grid(True)
            handles, labels = axis.get_legend_handles_labels()
            if handles:
                axis.legend(handles, labels, fontsize=8, ncol=2)
            fig.tight_layout()
            fig.savefig(RESULTS_DIR / "sample_convergence_curves.png", dpi=170)
            plt.close(fig)


def main(argv=None):
    args = build_parser().parse_args(argv)
    dataset_path = DATA_DIR / "dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Dataset is missing. Run: python -m hypodifferential_rate.generate_dataset"
        )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(dataset_path)
    missing = [name for name in FEATURE_NAMES + ["J300", "rho"] if name not in frame]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    if frame[FEATURE_NAMES + ["J300"]].isna().any().any():
        raise ValueError("Features and J300 must not contain missing values")

    train_index, validation_index, test_index = three_way_split(frame, args.seed)
    frame["split"] = ""
    frame.loc[train_index, "split"] = "train"
    frame.loc[validation_index, "split"] = "validation"
    frame.loc[test_index, "split"] = "test"
    frame.to_csv(DATA_DIR / "dataset_with_splits.csv", index=False)

    X = frame[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = frame["J300"].to_numpy(dtype=np.float64)
    _, best_config, best_validation_mse, selection = select_model(
        X[train_index],
        y[train_index],
        X[validation_index],
        y[validation_index],
        args.seed,
    )
    selection.to_csv(RESULTS_DIR / "model_selection.csv", index=False)

    fit_index = np.concatenate([train_index, validation_index])
    final_model = make_model(best_config, args.seed + 100)
    final_model.fit(X[fit_index], y[fit_index])

    test_frame = frame.iloc[test_index].copy().reset_index(drop=True)
    test_frame["J300_pred"] = final_model.predict(
        test_frame[FEATURE_NAMES].to_numpy(dtype=np.float64)
    )
    test_frame["rho_pred"] = 10.0 ** test_frame["J300_pred"]
    test_frame.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    J_metrics = metric_block(test_frame["J300"], test_frame["J300_pred"])
    rho_metrics = metric_block(test_frame["rho"], test_frame["rho_pred"])
    print(f"TEST J_300 R2={J_metrics['r2']:.6f} MAE={J_metrics['mae']:.8g}")
    print(f"TEST rho   R2={rho_metrics['r2']:.6f} MAE={rho_metrics['mae']:.8g}")

    model_path = MODELS_DIR / "gradient_boosting_J300.joblib"
    bundle = {
        "model": final_model,
        "feature_names": FEATURE_NAMES,
        "target": "J300",
        "rho_transform": "10 ** J300",
        "best_config": best_config,
    }
    joblib.dump(bundle, model_path)

    feature_importance = pd.DataFrame(
        {"feature": FEATURE_NAMES, "importance": final_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    save_plots(test_frame, feature_importance)

    tail_values = test_frame["tail_linearity_r2_mean"].dropna()
    diagnostics = {
        "tail_linearity_r2_mean": float(tail_values.mean())
        if len(tail_values)
        else None,
        "tail_linearity_r2_median": float(tail_values.median())
        if len(tail_values)
        else None,
        "fraction_tail_r2_at_least_0_95": float(np.mean(tail_values >= 0.95))
        if len(tail_values)
        else None,
    }
    manifest = {
        "description": "Learned finite-horizon rate of hypodifferential descent",
        "external_input": "matrices A_j of f(x)=max_j 0.5*x^T*A_j*x",
        "output": ["J300", "rho=10**J300"],
        "data_split": {
            "train_fraction": 0.60,
            "validation_fraction": 0.20,
            "test_fraction": 0.20,
            "model_selection_metric": "validation MSE on J300",
            "reported_metrics_partition": "test only",
        },
        "model_selection": {
            "best_config": best_config,
            "best_validation_mse": best_validation_mse,
        },
        "test_metrics": {"J300": J_metrics, "rho": rho_metrics},
        "linearity_diagnostics_on_test": diagnostics,
        "feature_names": FEATURE_NAMES,
        "saved_model": str(model_path),
        "seed": args.seed,
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Saved:")
    print(" ", model_path)
    print(" ", RESULTS_DIR / "metrics.json")
    print(" ", RESULTS_DIR / "test_predictions.csv")
    print(" ", RESULTS_DIR / "test_predictions.png")
    print(" ", RESULTS_DIR / "sample_convergence_curves.png")


if __name__ == "__main__":
    main()
