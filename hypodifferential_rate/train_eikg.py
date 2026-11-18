# -*- coding: utf-8 -*-
"""Train an EIKG polynomial model that predicts hypodifferential J_300."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from eikg import EIKGPolynomialRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .core import FEATURE_NAMES
from .train import three_way_split


matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models_eikg"
RESULTS_DIR = BASE_DIR / "results_eikg"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train an EIKGPolynomialRegressor J_300 predictor"
    )
    parser.add_argument(
        "--degrees", type=int, nargs="+", default=list(range(1, 11))
    )
    parser.add_argument("--alpha-ridge", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(args):
    if not args.degrees or any(degree < 1 for degree in args.degrees):
        raise ValueError("degrees must contain positive integers")
    args.degrees = sorted(set(args.degrees))
    if not np.isfinite(args.alpha_ridge) or args.alpha_ridge < 0.0:
        raise ValueError("alpha-ridge must be finite and non-negative")


def make_model(degree, alpha_ridge):
    return EIKGPolynomialRegressor(
        degree=int(degree),
        regularization="ridge",
        alpha_ridge=float(alpha_ridge),
        fit_intercept=True,
        scale=True,
        scale_y=True,
        normalize_latent=True,
        dtype=np.float64,
        copy=True,
        check_input=True,
    )


def metric_block(y_true, y_pred):
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
    }


def select_degree(X_train, y_train, X_validation, y_validation, degrees, alpha):
    records = []
    for degree in degrees:
        started = time.perf_counter()
        model = make_model(degree, alpha)
        model.fit(X_train, y_train)
        prediction = np.asarray(model.predict(X_validation), dtype=np.float64)
        mse = float(mean_squared_error(y_validation, prediction))
        if not np.isfinite(mse):
            raise FloatingPointError(
                f"Non-finite validation MSE for EIKG degree={degree}"
            )
        elapsed = time.perf_counter() - started
        records.append(
            {
                "degree": int(degree),
                "validation_mse": mse,
                "fit_seconds": float(elapsed),
            }
        )
        print(
            f"degree={degree:2d} validation MSE={mse:.10g} "
            f"fit={elapsed:.3f}s"
        )

    selection = pd.DataFrame(records).sort_values(
        ["validation_mse", "degree"], kind="stable"
    )
    best_degree = int(selection.iloc[0]["degree"])
    best_mse = float(selection.iloc[0]["validation_mse"])
    print(f"Selected degree={best_degree} validation MSE={best_mse:.10g}")
    return best_degree, best_mse, selection


def save_plots(test_frame, selection):
    ordered = selection.sort_values("degree")
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.plot(ordered["degree"], ordered["validation_mse"], marker="o")
    best = selection.iloc[0]
    axis.scatter(
        [best["degree"]], [best["validation_mse"]], color="red", zorder=3
    )
    axis.set_xlabel("EIKG polynomial degree")
    axis.set_ylabel("Validation MSE")
    axis.set_title("EIKG degree selection")
    axis.grid(True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "degree_selection.png", dpi=170)
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
    best_degree, best_validation_mse, selection = select_degree(
        X[train_index],
        y[train_index],
        X[validation_index],
        y[validation_index],
        args.degrees,
        args.alpha_ridge,
    )
    selection.to_csv(RESULTS_DIR / "degree_selection.csv", index=False)

    fit_index = np.concatenate([train_index, validation_index])
    final_model = make_model(best_degree, args.alpha_ridge)
    final_model.fit(X[fit_index], y[fit_index])

    test_frame = frame.iloc[test_index].copy().reset_index(drop=True)
    test_frame["J300_pred"] = np.asarray(
        final_model.predict(test_frame[FEATURE_NAMES].to_numpy(dtype=np.float64)),
        dtype=np.float64,
    )
    test_frame["rho_pred"] = 10.0 ** test_frame["J300_pred"]
    test_frame.to_csv(RESULTS_DIR / "test_predictions.csv", index=False)

    J_metrics = metric_block(test_frame["J300"], test_frame["J300_pred"])
    rho_metrics = metric_block(test_frame["rho"], test_frame["rho_pred"])
    print(f"TEST J_300 R2={J_metrics['r2']:.6f} MAE={J_metrics['mae']:.8g}")
    print(f"TEST rho   R2={rho_metrics['r2']:.6f} MAE={rho_metrics['mae']:.8g}")

    model_path = MODELS_DIR / "eikg_J300.joblib"
    joblib.dump(
        {
            "model": final_model,
            "feature_names": FEATURE_NAMES,
            "target": "J300",
            "rho_transform": "10 ** J300",
            "selected_degree": best_degree,
            "alpha_ridge": args.alpha_ridge,
        },
        model_path,
    )

    metrics = {
        "description": "EIKG finite-horizon rate model for hypodifferential descent",
        "estimator": "EIKGPolynomialRegressor",
        "package": "eikgpolynomial",
        "target": "J300",
        "data_split": {
            "train_fraction": 0.60,
            "validation_fraction": 0.20,
            "test_fraction": 0.20,
            "selection_metric": "validation MSE on J300",
            "reported_metrics_partition": "test only",
        },
        "degree_selection": {
            "candidate_degrees": args.degrees,
            "selected_degree": best_degree,
            "best_validation_mse": best_validation_mse,
        },
        "model_parameters": {
            "regularization": "ridge",
            "alpha_ridge": args.alpha_ridge,
            "scale": True,
            "scale_y": True,
            "normalize_latent": True,
        },
        "test_metrics": {"J300": J_metrics, "rho": rho_metrics},
        "feature_names": FEATURE_NAMES,
        "saved_model": str(model_path),
        "seed": args.seed,
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_plots(test_frame, selection)

    print("Saved:")
    print(" ", model_path)
    print(" ", RESULTS_DIR / "metrics.json")
    print(" ", RESULTS_DIR / "degree_selection.csv")
    print(" ", RESULTS_DIR / "test_predictions.csv")
    print(" ", RESULTS_DIR / "degree_selection.png")
    print(" ", RESULTS_DIR / "test_predictions.png")


if __name__ == "__main__":
    main()
