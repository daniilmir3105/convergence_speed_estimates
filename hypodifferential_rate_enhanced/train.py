# -*- coding: utf-8 -*-
"""Train Extra Trees on the enhanced balanced J_300 dataset."""

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
from sklearn.model_selection import train_test_split

from .core import FEATURE_NAMES, OPTIONAL_FEATURE_NAMES


matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


FEATURE_SETS = {
    "log_kappa_only": FEATURE_NAMES,
    "log_and_sqrt_kappa": FEATURE_NAMES + OPTIONAL_FEATURE_NAMES,
}

CANDIDATE_CONFIGS = [
    {
        "n_estimators": 400,
        "min_samples_leaf": 1,
        "max_features": 1.0,
        "max_depth": None,
    },
    {
        "n_estimators": 800,
        "min_samples_leaf": 2,
        "max_features": 1.0,
        "max_depth": None,
    },
    {
        "n_estimators": 800,
        "min_samples_leaf": 2,
        "max_features": 0.75,
        "max_depth": None,
    },
    {
        "n_estimators": 1000,
        "min_samples_leaf": 3,
        "max_features": 1.0,
        "max_depth": None,
    },
    {
        "n_estimators": 1000,
        "min_samples_leaf": 1,
        "max_features": 0.70,
        "max_depth": None,
    },
    {
        "n_estimators": 800,
        "min_samples_leaf": 4,
        "max_features": 0.90,
        "max_depth": None,
    },
]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train enhanced ExtraTreesRegressor J_300 predictor"
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


def _combined_strata(frame):
    if "design_kappa_bin" in frame:
        kappa_bin = frame["design_kappa_bin"].astype(int).astype(str)
    else:
        kappa_bin = pd.qcut(
            frame["log_kappa"], q=10, labels=False, duplicates="drop"
        ).astype(str)
    return (
        frame["dimension"].astype(int).astype(str)
        + "_"
        + frame["n_pieces"].astype(int).astype(str)
        + "_"
        + kappa_bin
    ).to_numpy()


def three_way_split(frame, seed):
    indices = np.arange(len(frame))
    strata = _combined_strata(frame)
    try:
        train_validation, test = train_test_split(
            indices,
            test_size=0.20,
            random_state=seed,
            shuffle=True,
            stratify=strata,
        )
        train, validation = train_test_split(
            train_validation,
            test_size=0.25,
            random_state=seed + 1,
            shuffle=True,
            stratify=strata[train_validation],
        )
    except ValueError:
        # This fallback is meant only for deliberately tiny technical datasets.
        bins = pd.qcut(
            frame["log_kappa"], q=min(10, max(2, len(frame) // 20)),
            labels=False,
            duplicates="drop",
        ).to_numpy()
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
    return np.asarray(train), np.asarray(validation), np.asarray(test)


def metric_block(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }


def select_model(frame, train_index, validation_index, seed, n_jobs):
    y = frame["J300"].to_numpy(dtype=np.float64)
    records = []
    best = None
    best_mse = np.inf
    candidate_number = 0

    for feature_set_name, feature_names in FEATURE_SETS.items():
        X = frame[feature_names].to_numpy(dtype=np.float64)
        for config in CANDIDATE_CONFIGS:
            candidate_number += 1
            started = time.perf_counter()
            model = make_model(config, seed, n_jobs)
            model.fit(X[train_index], y[train_index])
            prediction = model.predict(X[validation_index])
            mse = float(mean_squared_error(y[validation_index], prediction))
            elapsed = time.perf_counter() - started
            record = {
                "candidate": candidate_number,
                "feature_set": feature_set_name,
                "n_features": len(feature_names),
                **config,
                "validation_mse": mse,
                "fit_seconds": float(elapsed),
            }
            records.append(record)
            print(
                f"candidate={candidate_number:2d} features={feature_set_name} "
                f"trees={config['n_estimators']} leaf={config['min_samples_leaf']} "
                f"max_features={config['max_features']:.2f} "
                f"validation MSE={mse:.10g} fit={elapsed:.2f}s"
            )
            if mse < best_mse:
                best_mse = mse
                best = {
                    "candidate": candidate_number,
                    "feature_set": feature_set_name,
                    "feature_names": feature_names.copy(),
                    "config": config.copy(),
                }

    selection = pd.DataFrame(records).sort_values(
        ["validation_mse", "candidate"], kind="stable"
    )
    print(
        f"Selected candidate={best['candidate']} "
        f"features={best['feature_set']} validation MSE={best_mse:.10g}"
    )
    return best, best_mse, selection


def save_plots(test_frame, selection, feature_importance):
    ordered = selection.sort_values("candidate")
    fig, axis = plt.subplots(figsize=(10, 5))
    for feature_set, group in ordered.groupby("feature_set", sort=False):
        axis.plot(
            group["candidate"],
            group["validation_mse"],
            marker="o",
            label=feature_set,
        )
    best = selection.iloc[0]
    axis.scatter(
        [best["candidate"]], [best["validation_mse"]], color="red", zorder=3
    )
    axis.set_xlabel("candidate")
    axis.set_ylabel("validation MSE")
    axis.set_title("Extra Trees and kappa-feature ablation")
    axis.grid(True)
    axis.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "model_selection.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, true_name, pred_name, title in [
        (axes[0], "J300", "J300_pred", "Finite-horizon rate J_300"),
        (axes[1], "rho", "rho_pred", "Per-iteration contraction rho"),
    ]:
        axis.scatter(test_frame[true_name], test_frame[pred_name], s=16, alpha=0.5)
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
    ).tail(25)
    fig, axis = plt.subplots(figsize=(10, 9))
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
            "Dataset is missing. Run: "
            "python -m hypodifferential_rate_enhanced.generate_dataset"
        )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(dataset_path)
    required = sorted(
        set(FEATURE_NAMES + OPTIONAL_FEATURE_NAMES + ["J300", "rho"])
    )
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    if frame[required].isna().any().any():
        raise ValueError("Features and targets must not contain missing values")

    train_index, validation_index, test_index = three_way_split(frame, args.seed)
    split_frame = pd.DataFrame(
        {"instance_id": frame["instance_id"].astype(int), "split": ""}
    )
    split_frame.loc[train_index, "split"] = "train"
    split_frame.loc[validation_index, "split"] = "validation"
    split_frame.loc[test_index, "split"] = "test"
    split_frame.to_csv(RESULTS_DIR / "split_assignments.csv", index=False)

    best, best_validation_mse, selection = select_model(
        frame,
        train_index,
        validation_index,
        args.seed,
        args.n_jobs,
    )
    selection.to_csv(RESULTS_DIR / "model_selection.csv", index=False)

    feature_names = best["feature_names"]
    X = frame[feature_names].to_numpy(dtype=np.float64)
    y = frame["J300"].to_numpy(dtype=np.float64)
    fit_index = np.concatenate([train_index, validation_index])
    final_model = make_model(best["config"], args.seed, args.n_jobs)
    final_model.fit(X[fit_index], y[fit_index])

    test_frame = frame.iloc[test_index].copy().reset_index(drop=True)
    test_frame["J300_pred"] = final_model.predict(X[test_index])
    test_frame["rho_pred"] = 10.0 ** test_frame["J300_pred"]
    test_frame.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    J_metrics = metric_block(test_frame["J300"], test_frame["J300_pred"])
    rho_metrics = metric_block(test_frame["rho"], test_frame["rho_pred"])
    print(f"TEST J_300 R2={J_metrics['r2']:.6f} MAE={J_metrics['mae']:.8g}")
    print(f"TEST rho   R2={rho_metrics['r2']:.6f} MAE={rho_metrics['mae']:.8g}")

    model_path = MODELS_DIR / "extra_trees_enhanced_J300.joblib"
    manifest_path = DATA_DIR / "dataset_manifest.json"
    if manifest_path.exists():
        dataset_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        generation_config = dataset_manifest.get("config", {})
        feature_config = {
            "n_starts": int(generation_config.get("starts", 16)),
            "seed": int(dataset_manifest.get("feature_seed", args.seed + 1)),
            "projection_tolerance": float(
                generation_config.get("projection_tolerance", 1e-11)
            ),
            "projection_max_iterations": int(
                generation_config.get("projection_max_iterations", 300)
            ),
        }
    else:
        feature_config = {
            "n_starts": 16,
            "seed": args.seed + 1,
            "projection_tolerance": 1e-11,
            "projection_max_iterations": 300,
        }
    joblib.dump(
        {
            "model": final_model,
            "feature_names": feature_names,
            "feature_set": best["feature_set"],
            "feature_config": feature_config,
            "target": "J300",
            "rho_transform": "10 ** J300",
            "best_config": best["config"],
        },
        model_path,
    )

    feature_importance = pd.DataFrame(
        {"feature": feature_names, "importance": final_model.feature_importances_}
    ).sort_values("importance", ascending=False)
    feature_importance.to_csv(
        RESULTS_DIR / "feature_importance.csv", index=False
    )

    metrics = {
        "description": "Enhanced balanced Extra Trees J_300 experiment",
        "estimator": "ExtraTreesRegressor",
        "target": (
            "sum_i log10(max(f(x_300^i)/f(x_0^i), 1e-300)) / (16*300)"
        ),
        "target_changed_from_original": False,
        "data_split": {
            "train_fraction": 0.60,
            "validation_fraction": 0.20,
            "test_fraction": 0.20,
            "stratification": "dimension x n_pieces x log-kappa bin",
            "selection_metric": "validation MSE on J300",
            "reported_metrics_partition": "test only",
        },
        "model_selection": {
            "candidate_count": len(selection),
            "selected_candidate": best["candidate"],
            "selected_feature_set": best["feature_set"],
            "sqrt_kappa_selected": "sqrt_kappa" in feature_names,
            "best_config": best["config"],
            "best_validation_mse": best_validation_mse,
        },
        "test_metrics": {"J300": J_metrics, "rho": rho_metrics},
        "feature_names": feature_names,
        "feature_config": feature_config,
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
