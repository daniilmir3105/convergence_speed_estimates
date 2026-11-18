# -*- coding: utf-8 -*-
"""Train an Extra Trees model that predicts hypodifferential J_300."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .core import FEATURE_NAMES
from .train import three_way_split


matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models_extratrees"
RESULTS_DIR = BASE_DIR / "results_extratrees"


# All candidates use the same features, split, loss and random seed. Therefore
# validation MSE compares only Extra Trees capacity and regularization.
CANDIDATE_CONFIGS = [
    {
        "n_estimators": 300,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "max_depth": None,
    },
    {
        "n_estimators": 500,
        "min_samples_leaf": 2,
        "max_features": 0.9,
        "max_depth": None,
    },
    {
        "n_estimators": 800,
        "min_samples_leaf": 2,
        "max_features": 1.0,
        "max_depth": None,
    },
    {
        "n_estimators": 500,
        "min_samples_leaf": 4,
        "max_features": 0.9,
        "max_depth": None,
    },
    {
        "n_estimators": 800,
        "min_samples_leaf": 2,
        "max_features": 0.7,
        "max_depth": None,
    },
    {
        "n_estimators": 500,
        "min_samples_leaf": 2,
        "max_features": 0.9,
        "max_depth": 12,
    },
    {
        "n_estimators": 800,
        "min_samples_leaf": 3,
        "max_features": 1.0,
        "max_depth": None,
    },
    {
        "n_estimators": 800,
        "min_samples_leaf": 1,
        "max_features": 0.7,
        "max_depth": None,
    },
]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train an ExtraTreesRegressor J_300 predictor"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser


def validate_args(args):
    if args.n_jobs == 0:
        raise ValueError("n-jobs cannot be zero")


def make_model(config, seed, n_jobs):
    return ExtraTreesRegressor(
        criterion="squared_error",
        n_estimators=config["n_estimators"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
        max_depth=config["max_depth"],
        bootstrap=False,
        random_state=seed,
        n_jobs=n_jobs,
    )


def metric_block(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }


def select_model(
    X_train,
    y_train,
    X_validation,
    y_validation,
    seed,
    n_jobs,
):
    records = []
    best_config = None
    best_mse = np.inf

    for number, config in enumerate(CANDIDATE_CONFIGS, start=1):
        started = time.perf_counter()
        model = make_model(config, seed, n_jobs)
        model.fit(X_train, y_train)
        prediction = model.predict(X_validation)
        mse = float(mean_squared_error(y_validation, prediction))
        elapsed = time.perf_counter() - started
        if not np.isfinite(mse):
            raise FloatingPointError(
                f"Non-finite validation MSE for candidate={number}"
            )
        records.append(
            {
                "candidate": number,
                **config,
                "validation_mse": mse,
                "fit_seconds": float(elapsed),
            }
        )
        depth = "None" if config["max_depth"] is None else config["max_depth"]
        print(
            f"candidate={number} trees={config['n_estimators']} "
            f"leaf={config['min_samples_leaf']} "
            f"features={config['max_features']:.2f} depth={depth} "
            f"validation MSE={mse:.10g} fit={elapsed:.2f}s"
        )
        if mse < best_mse:
            best_mse = mse
            best_config = config.copy()

    selection = pd.DataFrame(records).sort_values(
        ["validation_mse", "candidate"], kind="stable"
    )
    selected_number = int(selection.iloc[0]["candidate"])
    print(
        f"Selected candidate={selected_number} "
        f"validation MSE={best_mse:.10g}"
    )
    return best_config, best_mse, selection


def save_plots(test_frame, selection, feature_importance):
    ordered = selection.sort_values("candidate")
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.plot(ordered["candidate"], ordered["validation_mse"], marker="o")
    best = selection.iloc[0]
    axis.scatter(
        [best["candidate"]], [best["validation_mse"]], color="red", zorder=3
    )
    axis.set_xlabel("Extra Trees candidate")
    axis.set_ylabel("Validation MSE")
    axis.set_title("Extra Trees model selection")
    axis.grid(True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "model_selection.png", dpi=170)
    plt.close(fig)

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

    ordered_importance = feature_importance.sort_values(
        "importance", ascending=True
    )
    fig, axis = plt.subplots(figsize=(9, 7))
    axis.barh(ordered_importance["feature"], ordered_importance["importance"])
    axis.set_xlabel("Extra Trees feature importance")
    axis.grid(True, axis="x")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=170)
    plt.close(fig)


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_args(args)

    dataset_path = DATA_DIR / "dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(
            "Dataset is missing. Run: python -m hypodifferential_rate.generate_dataset"
        )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(dataset_path)
    required = FEATURE_NAMES + ["J300", "rho"]
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    if frame[FEATURE_NAMES + ["J300"]].isna().any().any():
        raise ValueError("Features and J300 must not contain missing values")

    train_index, validation_index, test_index = three_way_split(frame, args.seed)
    split_frame = pd.DataFrame(
        {
            "instance_id": frame["instance_id"],
            "split": "",
        }
    )
    split_frame.loc[train_index, "split"] = "train"
    split_frame.loc[validation_index, "split"] = "validation"
    split_frame.loc[test_index, "split"] = "test"
    split_frame.to_csv(RESULTS_DIR / "split_assignments.csv", index=False)

    X = frame[FEATURE_NAMES].to_numpy(dtype=np.float64)
    y = frame["J300"].to_numpy(dtype=np.float64)
    best_config, best_validation_mse, selection = select_model(
        X[train_index],
        y[train_index],
        X[validation_index],
        y[validation_index],
        args.seed,
        args.n_jobs,
    )
    selection.to_csv(RESULTS_DIR / "model_selection.csv", index=False)

    fit_index = np.concatenate([train_index, validation_index])
    final_model = make_model(best_config, args.seed, args.n_jobs)
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

    model_path = MODELS_DIR / "extra_trees_J300.joblib"
    joblib.dump(
        {
            "model": final_model,
            "feature_names": FEATURE_NAMES,
            "target": "J300",
            "rho_transform": "10 ** J300",
            "best_config": best_config,
        },
        model_path,
    )

    feature_importance = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "importance": final_model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(
        RESULTS_DIR / "feature_importance.csv", index=False
    )

    metrics = {
        "description": (
            "Extra Trees finite-horizon rate model for "
            "hypodifferential descent"
        ),
        "estimator": "ExtraTreesRegressor",
        "target": "J300",
        "data_split": {
            "train_fraction": 0.60,
            "validation_fraction": 0.20,
            "test_fraction": 0.20,
            "selection_metric": "validation MSE on J300",
            "reported_metrics_partition": "test only",
        },
        "model_selection": {
            "candidate_count": len(CANDIDATE_CONFIGS),
            "best_config": best_config,
            "best_validation_mse": best_validation_mse,
        },
        "test_metrics": {"J300": J_metrics, "rho": rho_metrics},
        "feature_names": FEATURE_NAMES,
        "saved_model": str(model_path),
        "seed": args.seed,
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_plots(test_frame, selection, feature_importance)

    print("Saved:")
    print(" ", model_path)
    print(" ", RESULTS_DIR / "metrics.json")
    print(" ", RESULTS_DIR / "model_selection.csv")
    print(" ", RESULTS_DIR / "split_assignments.csv")
    print(" ", RESULTS_DIR / "test_predictions.csv")
    print(" ", RESULTS_DIR / "model_selection.png")
    print(" ", RESULTS_DIR / "test_predictions.png")
    print(" ", RESULTS_DIR / "feature_importance.csv")
    print(" ", RESULTS_DIR / "feature_importance.png")


if __name__ == "__main__":
    main()
