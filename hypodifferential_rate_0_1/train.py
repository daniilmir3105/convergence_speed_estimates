# -*- coding: utf-8 -*-
"""Train Gradient Boosting, Extra Trees and EIKG on all component Li/li."""

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
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

GLOBAL_FEATURES = ["dimension", "n_pieces", "log_global_kappa"]

GRADIENT_BOOSTING_CONFIGS = [
    {"n_estimators": 300, "learning_rate": 0.03, "max_depth": 2},
    {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 2},
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 2},
    {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 3},
    {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3},
    {"n_estimators": 500, "learning_rate": 0.03, "max_depth": 4},
]

EXTRA_TREES_CONFIGS = [
    {"n_estimators": 500, "min_samples_leaf": 1, "max_features": 1.0},
    {"n_estimators": 800, "min_samples_leaf": 2, "max_features": 1.0},
    {"n_estimators": 800, "min_samples_leaf": 2, "max_features": 0.75},
    {"n_estimators": 1000, "min_samples_leaf": 3, "max_features": 1.0},
    {"n_estimators": 1000, "min_samples_leaf": 1, "max_features": 0.70},
    {"n_estimators": 800, "min_samples_leaf": 4, "max_features": 0.90},
]


def build_parser():
    parser = argparse.ArgumentParser(description="Train all Li/li rate models")
    parser.add_argument(
        "--eikg-degrees", type=int, nargs="+", default=list(range(1, 11))
    )
    parser.add_argument("--eikg-alpha-ridge", type=float, default=1e-5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(args):
    if not args.eikg_degrees or any(value < 1 for value in args.eikg_degrees):
        raise ValueError("eikg-degrees must contain positive integers")
    args.eikg_degrees = sorted(set(args.eikg_degrees))
    if args.eikg_alpha_ridge < 0.0 or not np.isfinite(args.eikg_alpha_ridge):
        raise ValueError("eikg-alpha-ridge must be finite and non-negative")
    if args.n_jobs == 0:
        raise ValueError("n-jobs cannot be zero")


def load_dataset_and_manifest():
    dataset_path = DATA_DIR / "dataset.csv"
    manifest_path = DATA_DIR / "dataset_manifest.json"
    if not dataset_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "Dataset is missing. Run: "
            "python -m hypodifferential_rate_0_1.generate_dataset"
        )
    frame = pd.read_csv(dataset_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    feature_names = list(manifest["feature_names"])
    required = feature_names + ["J300", "rho"]
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    if frame[required].isna().any().any():
        raise ValueError("Features and targets must not contain missing values")
    return frame, manifest, feature_names


def _strata(frame):
    coarse_kappa = (frame["design_kappa_bin"].astype(int) // 2).astype(str)
    return (
        frame["dimension"].astype(int).astype(str)
        + "_"
        + frame["n_pieces"].astype(int).astype(str)
        + "_"
        + coarse_kappa
    ).to_numpy()


def three_way_split(frame, seed):
    indices = np.arange(len(frame))
    strata = _strata(frame)
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
        fallback = pd.qcut(
            frame["log_global_kappa"], q=10, labels=False, duplicates="drop"
        ).to_numpy()
        train_validation, test = train_test_split(
            indices,
            test_size=0.20,
            random_state=seed,
            shuffle=True,
            stratify=fallback,
        )
        train, validation = train_test_split(
            train_validation,
            test_size=0.25,
            random_state=seed + 1,
            shuffle=True,
            stratify=fallback[train_validation],
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


def make_gradient_boosting(config, seed):
    return GradientBoostingRegressor(
        loss="squared_error",
        n_estimators=config["n_estimators"],
        learning_rate=config["learning_rate"],
        max_depth=config["max_depth"],
        min_samples_leaf=5,
        subsample=1.0,
        random_state=seed,
    )


def make_extra_trees(config, seed, n_jobs):
    return ExtraTreesRegressor(
        criterion="squared_error",
        n_estimators=config["n_estimators"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
        max_depth=None,
        bootstrap=False,
        random_state=seed,
        n_jobs=n_jobs,
    )


def make_eikg(degree, alpha_ridge):
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


def select_tree_model(
    family,
    configs,
    factory,
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    records = []
    best_config = None
    best_mse = np.inf
    for candidate, config in enumerate(configs, start=1):
        started = time.perf_counter()
        model = factory(config)
        model.fit(X_train, y_train)
        prediction = model.predict(X_validation)
        mse = float(mean_squared_error(y_validation, prediction))
        elapsed = time.perf_counter() - started
        records.append(
            {
                "family": family,
                "candidate": candidate,
                **config,
                "validation_mse": mse,
                "fit_seconds": elapsed,
            }
        )
        print(
            f"{family} candidate={candidate} validation MSE={mse:.10g} "
            f"fit={elapsed:.2f}s"
        )
        if mse < best_mse:
            best_mse = mse
            best_config = config.copy()
    return best_config, best_mse, records


def select_eikg(
    degrees,
    alpha_ridge,
    X_train,
    y_train,
    X_validation,
    y_validation,
):
    records = []
    best_degree = None
    best_mse = np.inf
    for candidate, degree in enumerate(degrees, start=1):
        started = time.perf_counter()
        model = make_eikg(degree, alpha_ridge)
        model.fit(X_train, y_train)
        prediction = np.asarray(model.predict(X_validation), dtype=np.float64)
        mse = float(mean_squared_error(y_validation, prediction))
        elapsed = time.perf_counter() - started
        if not np.isfinite(mse):
            raise FloatingPointError(f"non-finite EIKG MSE for degree={degree}")
        records.append(
            {
                "family": "eikg",
                "candidate": candidate,
                "degree": int(degree),
                "validation_mse": mse,
                "fit_seconds": elapsed,
            }
        )
        print(
            f"eikg candidate={candidate} degree={degree} "
            f"validation MSE={mse:.10g} fit={elapsed:.2f}s"
        )
        if mse < best_mse:
            best_mse = mse
            best_degree = int(degree)
    return best_degree, best_mse, records


def evaluate_family(
    family,
    full_model,
    baseline_model,
    X_full_fit,
    X_baseline_fit,
    y_fit,
    X_full_test,
    X_baseline_test,
    test_frame,
    feature_names,
    selection,
    feature_metadata,
):
    full_model.fit(X_full_fit, y_fit)
    baseline_model.fit(X_baseline_fit, y_fit)
    full_prediction = np.asarray(full_model.predict(X_full_test), dtype=np.float64)
    baseline_prediction = np.asarray(
        baseline_model.predict(X_baseline_test), dtype=np.float64
    )

    output = test_frame.copy()
    output["J300_pred_all_Li_li"] = full_prediction
    output["rho_pred_all_Li_li"] = 10.0**full_prediction
    output["J300_pred_global_only"] = baseline_prediction
    output["rho_pred_global_only"] = 10.0**baseline_prediction

    family_dir = RESULTS_DIR / family
    family_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(family_dir / "test_predictions.csv", index=False)
    pd.DataFrame(selection).to_csv(family_dir / "model_selection.csv", index=False)

    metrics = {
        "all_component_Li_li": {
            "J300": metric_block(output["J300"], output["J300_pred_all_Li_li"]),
            "rho": metric_block(output["rho"], output["rho_pred_all_Li_li"]),
        },
        "global_L_l_baseline": {
            "J300": metric_block(output["J300"], output["J300_pred_global_only"]),
            "rho": metric_block(output["rho"], output["rho_pred_global_only"]),
        },
    }

    model_path = MODELS_DIR / f"{family}_J300.joblib"
    joblib.dump(
        {
            "model": full_model,
            "family": family,
            "feature_names": feature_names,
            "feature_metadata": feature_metadata,
            "target": "J300",
            "rho_transform": "10 ** J300",
        },
        model_path,
    )
    return metrics, model_path


def save_comparison_plot(summary):
    families = list(summary["test_metrics"])
    full_values = [
        summary["test_metrics"][name]["all_component_Li_li"]["J300"]["r2"]
        for name in families
    ]
    baseline_values = [
        summary["test_metrics"][name]["global_L_l_baseline"]["J300"]["r2"]
        for name in families
    ]
    positions = np.arange(len(families))
    width = 0.36
    fig, axis = plt.subplots(figsize=(9, 5))
    axis.bar(positions - width / 2, full_values, width, label="all Li/li")
    axis.bar(positions + width / 2, baseline_values, width, label="global L/l")
    axis.set_xticks(positions, families)
    axis.set_ylabel("Test R2 for J_300")
    axis.set_title("Information gain from component-wise spectral bounds")
    axis.grid(True, axis="y")
    axis.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "test_r2_feature_comparison.png", dpi=170)
    plt.close(fig)


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    frame, manifest, feature_names = load_dataset_and_manifest()
    train_index, validation_index, test_index = three_way_split(frame, args.seed)
    split_frame = pd.DataFrame(
        {"instance_id": frame["instance_id"].astype(int), "split": ""}
    )
    split_frame.loc[train_index, "split"] = "train"
    split_frame.loc[validation_index, "split"] = "validation"
    split_frame.loc[test_index, "split"] = "test"
    split_frame.to_csv(RESULTS_DIR / "split_assignments.csv", index=False)

    X_full = frame[feature_names].to_numpy(dtype=np.float64)
    X_baseline = frame[GLOBAL_FEATURES].to_numpy(dtype=np.float64)
    y = frame["J300"].to_numpy(dtype=np.float64)
    fit_index = np.concatenate([train_index, validation_index])
    test_frame = frame.iloc[test_index].copy().reset_index(drop=True)

    gb_config, gb_validation_mse, gb_records = select_tree_model(
        "gradient_boosting",
        GRADIENT_BOOSTING_CONFIGS,
        lambda config: make_gradient_boosting(config, args.seed),
        X_full[train_index],
        y[train_index],
        X_full[validation_index],
        y[validation_index],
    )
    for record in gb_records:
        record["feature_set"] = "all_component_Li_li"
    gb_baseline_config, gb_baseline_validation_mse, gb_baseline_records = (
        select_tree_model(
            "gradient_boosting_global_only",
            GRADIENT_BOOSTING_CONFIGS,
            lambda config: make_gradient_boosting(config, args.seed),
            X_baseline[train_index],
            y[train_index],
            X_baseline[validation_index],
            y[validation_index],
        )
    )
    for record in gb_baseline_records:
        record["feature_set"] = "global_L_l_baseline"
    et_config, et_validation_mse, et_records = select_tree_model(
        "extra_trees",
        EXTRA_TREES_CONFIGS,
        lambda config: make_extra_trees(config, args.seed, args.n_jobs),
        X_full[train_index],
        y[train_index],
        X_full[validation_index],
        y[validation_index],
    )
    for record in et_records:
        record["feature_set"] = "all_component_Li_li"
    et_baseline_config, et_baseline_validation_mse, et_baseline_records = (
        select_tree_model(
            "extra_trees_global_only",
            EXTRA_TREES_CONFIGS,
            lambda config: make_extra_trees(config, args.seed, args.n_jobs),
            X_baseline[train_index],
            y[train_index],
            X_baseline[validation_index],
            y[validation_index],
        )
    )
    for record in et_baseline_records:
        record["feature_set"] = "global_L_l_baseline"
    eikg_degree, eikg_validation_mse, eikg_records = select_eikg(
        args.eikg_degrees,
        args.eikg_alpha_ridge,
        X_full[train_index],
        y[train_index],
        X_full[validation_index],
        y[validation_index],
    )
    for record in eikg_records:
        record["feature_set"] = "all_component_Li_li"
    (
        eikg_baseline_degree,
        eikg_baseline_validation_mse,
        eikg_baseline_records,
    ) = select_eikg(
        args.eikg_degrees,
        args.eikg_alpha_ridge,
        X_baseline[train_index],
        y[train_index],
        X_baseline[validation_index],
        y[validation_index],
    )
    for record in eikg_baseline_records:
        record["family"] = "eikg_global_only"
        record["feature_set"] = "global_L_l_baseline"

    feature_metadata = {
        "schema": manifest["feature_schema_version"],
        "max_pieces": manifest["config"]["pieces_max"],
        "component_order": "decreasing Li, then decreasing li",
        "padding": "zero with explicit presence mask",
    }
    metrics = {}
    model_paths = {}
    metrics["gradient_boosting"], model_paths["gradient_boosting"] = evaluate_family(
        "gradient_boosting",
        make_gradient_boosting(gb_config, args.seed),
        make_gradient_boosting(gb_baseline_config, args.seed),
        X_full[fit_index],
        X_baseline[fit_index],
        y[fit_index],
        X_full[test_index],
        X_baseline[test_index],
        test_frame,
        feature_names,
        gb_records + gb_baseline_records,
        feature_metadata,
    )
    metrics["extra_trees"], model_paths["extra_trees"] = evaluate_family(
        "extra_trees",
        make_extra_trees(et_config, args.seed, args.n_jobs),
        make_extra_trees(et_baseline_config, args.seed, args.n_jobs),
        X_full[fit_index],
        X_baseline[fit_index],
        y[fit_index],
        X_full[test_index],
        X_baseline[test_index],
        test_frame,
        feature_names,
        et_records + et_baseline_records,
        feature_metadata,
    )
    metrics["eikg"], model_paths["eikg"] = evaluate_family(
        "eikg",
        make_eikg(eikg_degree, args.eikg_alpha_ridge),
        make_eikg(eikg_baseline_degree, args.eikg_alpha_ridge),
        X_full[fit_index],
        X_baseline[fit_index],
        y[fit_index],
        X_full[test_index],
        X_baseline[test_index],
        test_frame,
        feature_names,
        eikg_records + eikg_baseline_records,
        feature_metadata,
    )

    selection_summary = {
        "gradient_boosting": {
            "all_component_Li_li": {
                "best_config": gb_config,
                "best_validation_mse": gb_validation_mse,
            },
            "global_L_l_baseline": {
                "best_config": gb_baseline_config,
                "best_validation_mse": gb_baseline_validation_mse,
            },
        },
        "extra_trees": {
            "all_component_Li_li": {
                "best_config": et_config,
                "best_validation_mse": et_validation_mse,
            },
            "global_L_l_baseline": {
                "best_config": et_baseline_config,
                "best_validation_mse": et_baseline_validation_mse,
            },
        },
        "eikg": {
            "alpha_ridge": args.eikg_alpha_ridge,
            "all_component_Li_li": {
                "selected_degree": eikg_degree,
                "best_validation_mse": eikg_validation_mse,
            },
            "global_L_l_baseline": {
                "selected_degree": eikg_baseline_degree,
                "best_validation_mse": eikg_baseline_validation_mse,
            },
        },
    }
    summary = {
        "description": "All component Li/li versus global L/l experiment",
        "target_changed_from_original": False,
        "data_split": {
            "train_fraction": 0.60,
            "validation_fraction": 0.20,
            "test_fraction": 0.20,
            "reported_metrics_partition": "test only",
        },
        "feature_comparison": {
            "main": "all sorted component Li/li pairs with masks",
            "baseline": "dimension, n_pieces, global L/l only",
        },
        "model_selection": selection_summary,
        "test_metrics": metrics,
        "saved_models": {name: str(path) for name, path in model_paths.items()},
        "seed": args.seed,
    }
    (RESULTS_DIR / "summary_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_comparison_plot(summary)

    print("Test R2 for J_300 (all Li/li versus global-only baseline):")
    for family, values in metrics.items():
        full_r2 = values["all_component_Li_li"]["J300"]["r2"]
        baseline_r2 = values["global_L_l_baseline"]["J300"]["r2"]
        print(f"  {family}: {full_r2:.6f} versus {baseline_r2:.6f}")
    print("Saved:")
    print(" ", RESULTS_DIR / "summary_metrics.json")
    print(" ", RESULTS_DIR / "test_r2_feature_comparison.png")
    for path in model_paths.values():
        print(" ", path)


if __name__ == "__main__":
    main()
