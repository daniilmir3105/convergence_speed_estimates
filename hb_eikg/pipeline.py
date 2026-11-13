# -*- coding: utf-8 -*-
"""Finite-horizon Heavy Ball pipeline with EIKG polynomial models.

The numerical experiment is shared with ``hb.pipeline`` so the substantive
model change is Gradient Boosting -> EIKG. Polynomial degrees are selected by
minimum validation MSE. Reported R2 metrics use untouched test partitions.
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
from eikg import EIKGPolynomialRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from hb.pipeline import (
    BETA_MAX,
    FORWARD_FEATURES,
    KAPPA_MAX,
    KAPPA_MIN,
    N_STARTS,
    STABILITY_SAFETY,
    STEPS,
    exact_scores,
    forward_feature_matrix,
    generate_forward_dataset,
    global_grid,
    local_grid,
    make_starts,
    oracle_search,
    predicted_scores,
    sample_meta_pairs,
    select_best,
    stratified_log_uniform,
    take_top_candidates,
    terminal_rate,
    theory_from_L_l,
)

from .model import EIKGHeavyBallModel, META_FEATURES, meta_feature_matrix


matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


def three_way_split(items, seed):
    """Return deterministic 60%/20%/20% train/validation/test partitions."""
    items = np.asarray(items)
    train_validation, test = train_test_split(
        items, test_size=0.20, random_state=seed, shuffle=True
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        random_state=seed + 1,
        shuffle=True,
    )
    return train, validation, test


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


def expand_integer_weights(X, y, weights):
    """Emulate unsupported sample_weight by exact integer row repetition."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if weights is None:
        return X, y
    weights = np.asarray(weights, dtype=np.float64)
    repeats = np.rint(weights).astype(np.int64)
    if not np.allclose(weights, repeats) or np.any(repeats < 1):
        raise ValueError("EIKG weight expansion requires positive integer weights")
    return np.repeat(X, repeats, axis=0), np.repeat(y, repeats)


def select_degree_and_refit(
    name,
    X_train,
    y_train,
    X_validation,
    y_validation,
    degrees,
    alpha_ridge,
    train_weights=None,
    validation_weights=None,
):
    """Select degree by plain validation MSE and refit on train+validation."""
    X_fit, y_fit = expand_integer_weights(X_train, y_train, train_weights)
    records = []
    for degree in degrees:
        started = time.perf_counter()
        model = make_eikg(degree, alpha_ridge)
        model.fit(X_fit, y_fit)
        prediction = model.predict(np.asarray(X_validation, dtype=np.float64))
        mse = float(mean_squared_error(y_validation, prediction))
        if not np.isfinite(mse):
            raise FloatingPointError(
                f"Non-finite validation MSE for {name}, degree={degree}"
            )
        elapsed = time.perf_counter() - started
        records.append(
            {
                "degree": int(degree),
                "validation_mse": mse,
                "fit_seconds": float(elapsed),
                "weighted_train_rows": int(len(X_fit)),
            }
        )
        print(
            f"  {name}: degree={degree:2d} | validation MSE={mse:.10g} "
            f"| {elapsed:.2f}s"
        )

    selection = pd.DataFrame(records).sort_values(
        ["validation_mse", "degree"], kind="stable"
    )
    best_degree = int(selection.iloc[0]["degree"])
    best_validation_mse = float(selection.iloc[0]["validation_mse"])
    print(
        f"  {name}: selected degree={best_degree} "
        f"(validation MSE={best_validation_mse:.10g})"
    )

    X_refit = np.vstack(
        [np.asarray(X_train, dtype=np.float64), np.asarray(X_validation, dtype=np.float64)]
    )
    y_refit = np.concatenate(
        [np.asarray(y_train, dtype=np.float64), np.asarray(y_validation, dtype=np.float64)]
    )
    if train_weights is None and validation_weights is None:
        refit_weights = None
    else:
        refit_weights = np.concatenate(
            [
                np.ones(len(X_train))
                if train_weights is None
                else np.asarray(train_weights),
                np.ones(len(X_validation))
                if validation_weights is None
                else np.asarray(validation_weights),
            ]
        )
    X_refit, y_refit = expand_integer_weights(X_refit, y_refit, refit_weights)
    final_model = make_eikg(best_degree, alpha_ridge)
    final_model.fit(X_refit, y_refit)
    return final_model, best_degree, best_validation_mse, selection


def print_r2(name, y_true, y_pred):
    score = float(r2_score(y_true, y_pred))
    print(f"{name}: TEST R2={score:.6f}")
    return score


def theory_blind_inverse_search(
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
    beta_max,
):
    """Create inverse labels without inserting the Polyak point.

    The forward model ranks a global grid and a local refinement. Only those
    candidates are evaluated with the exact finite-horizon criterion. Further
    local refinement also minimizes exact J_K and uses no analytical optimum.
    """
    kappa = np.asarray(kappa, dtype=np.float64)
    outputs = [np.empty(len(kappa), dtype=np.float64) for _ in range(4)]
    started = time.perf_counter()

    for begin in range(0, len(kappa), batch_size):
        end = min(begin + batch_size, len(kappa))
        current_kappa = kappa[begin:end]
        alpha_L, beta, q = global_grid(
            len(current_kappa), n_grid, n_grid, safety, beta_max
        )
        score = predicted_scores(model, current_kappa, alpha_L, beta)
        predicted_best = select_best(alpha_L, beta, q, score)

        half_top = max(1, exact_top_k // 2)
        coarse_top = take_top_candidates(
            alpha_L, beta, q, score, half_top
        )

        q_radius = 2.0 * safety / max(n_grid - 1, 1)
        beta_radius = 2.0 * beta_max / max(n_grid - 1, 1)
        alpha_ref, beta_ref, q_ref = local_grid(
            predicted_best[2],
            predicted_best[1],
            predicted_refine,
            predicted_refine,
            q_radius,
            beta_radius,
            safety,
            beta_max,
        )
        score_ref = predicted_scores(model, current_kappa, alpha_ref, beta_ref)
        refined_top = take_top_candidates(
            alpha_ref,
            beta_ref,
            q_ref,
            score_ref,
            exact_top_k - half_top,
        )

        # Deliberately no Polyak/theoretical candidate in this matrix.
        candidates_alpha = np.column_stack([coarse_top[0], refined_top[0]])
        candidates_beta = np.column_stack([coarse_top[1], refined_top[1]])
        candidates_q = np.column_stack([coarse_top[2], refined_top[2]])
        exact = exact_scores(
            current_kappa,
            candidates_alpha,
            candidates_beta,
            starts,
            steps,
        )
        best_alpha, best_beta, best_q, best_score = select_best(
            candidates_alpha,
            candidates_beta,
            candidates_q,
            exact,
        )

        for _ in range(exact_stages):
            alpha_exact, beta_exact, q_exact = local_grid(
                best_q,
                best_beta,
                exact_refine,
                exact_refine,
                q_radius,
                beta_radius,
                safety,
                beta_max,
            )
            exact_score = exact_scores(
                current_kappa,
                alpha_exact,
                beta_exact,
                starts,
                steps,
            )
            candidate = select_best(
                alpha_exact,
                beta_exact,
                q_exact,
                exact_score,
            )
            improved = candidate[3] < best_score
            best_alpha[improved] = candidate[0][improved]
            best_beta[improved] = candidate[1][improved]
            best_q[improved] = candidate[2][improved]
            best_score[improved] = candidate[3][improved]
            q_radius *= 0.35
            beta_radius *= 0.35

        for target, values in zip(
            outputs,
            [best_alpha, best_beta, best_q, best_score],
        ):
            target[begin:end] = values

        elapsed = time.perf_counter() - started
        rate = end / max(elapsed, 1e-12)
        eta = (len(kappa) - end) / max(rate, 1e-12)
        print(f"  theory-blind inverse {end:5d}/{len(kappa)} | ETA {eta:6.1f}s")

    return tuple(outputs)


def save_plots(
    oracle_frame,
    test_frame,
    forward_selection,
    q_selection,
    beta_selection,
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, parameter in zip(axes, ["alpha", "beta"]):
        axis.scatter(
            oracle_frame[f"{parameter}_theory"],
            oracle_frame[f"{parameter}_found"],
            s=12,
            alpha=0.5,
        )
        low = min(
            oracle_frame[f"{parameter}_theory"].min(),
            oracle_frame[f"{parameter}_found"].min(),
        )
        high = max(
            oracle_frame[f"{parameter}_theory"].max(),
            oracle_frame[f"{parameter}_found"].max(),
        )
        axis.plot([low, high], [low, high], "r--")
        axis.set_xlabel(f"{parameter} Polyak theory")
        axis.set_ylabel(f"{parameter} exact finite-horizon oracle")
        axis.grid(True)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "oracle_vs_polyak.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, parameter in zip(axes, ["alpha", "beta"]):
        axis.scatter(
            test_frame[f"{parameter}_theory"],
            test_frame[f"{parameter}_pred"],
            s=12,
            alpha=0.45,
        )
        low = min(
            test_frame[f"{parameter}_theory"].min(),
            test_frame[f"{parameter}_pred"].min(),
        )
        high = max(
            test_frame[f"{parameter}_theory"].max(),
            test_frame[f"{parameter}_pred"].max(),
        )
        axis.plot([low, high], [low, high], "r--")
        axis.set_xlabel(f"{parameter} Polyak theory")
        axis.set_ylabel(f"{parameter} EIKG prediction")
        axis.grid(True)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "eikg_parameters_test.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for axis, frame, title in [
        (axes[0], forward_selection, "Forward EIKG degree"),
        (axes[1], q_selection, "Final q EIKG degree"),
        (axes[2], beta_selection, "Final beta EIKG degree"),
    ]:
        ordered = frame.sort_values("degree")
        axis.plot(ordered["degree"], ordered["validation_mse"], marker="o")
        axis.set_xlabel("Polynomial degree")
        axis.set_ylabel("Validation MSE")
        axis.set_title(title)
        axis.grid(True)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "degree_selection.png", dpi=160)
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Finite-horizon Heavy Ball pipeline with EIKG models"
    )
    parser.add_argument("--kappa-min", type=float, default=KAPPA_MIN)
    parser.add_argument("--kappa-max", type=float, default=KAPPA_MAX)
    parser.add_argument("--stability-safety", type=float, default=STABILITY_SAFETY)
    parser.add_argument("--beta-max", type=float, default=BETA_MAX)
    parser.add_argument("--steps", type=int, default=STEPS)
    parser.add_argument("--starts", type=int, default=N_STARTS)
    parser.add_argument("--forward-groups", type=int, default=2400)
    parser.add_argument("--forward-candidates", type=int, default=96)
    parser.add_argument(
        "--forward-degrees", type=int, nargs="+", default=list(range(1, 11))
    )
    parser.add_argument(
        "--q-degrees", type=int, nargs="+", default=list(range(1, 11))
    )
    parser.add_argument(
        "--beta-degrees", type=int, nargs="+", default=list(range(1, 11))
    )
    parser.add_argument("--alpha-ridge", type=float, default=1e-5)
    parser.add_argument(
        "--disable-forward-weight-repetition",
        action="store_true",
        help="Do not emulate original integer sample weights by row repetition",
    )
    parser.add_argument("--oracle-points", type=int, default=256)
    parser.add_argument("--oracle-grid", type=int, default=64)
    parser.add_argument("--oracle-refine", type=int, default=25)
    parser.add_argument("--oracle-stages", type=int, default=2)
    parser.add_argument("--meta-pairs", type=int, default=10000)
    parser.add_argument("--search-grid", type=int, default=64)
    parser.add_argument("--predicted-refine", type=int, default=32)
    parser.add_argument("--exact-top-k", type=int, default=24)
    parser.add_argument("--exact-refine", type=int, default=17)
    parser.add_argument("--exact-stages", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(args):
    if not 1.0 < args.kappa_min < args.kappa_max:
        raise ValueError("Expected 1 < kappa-min < kappa-max")
    if not 0.0 < args.stability_safety < 1.0:
        raise ValueError("stability-safety must be in (0, 1)")
    if not 0.0 < args.beta_max < 1.0:
        raise ValueError("beta-max must be in (0, 1)")
    required_q = args.kappa_max / (args.kappa_max + 1.0)
    _, required_beta = theory_from_L_l(args.kappa_max, 1.0)
    if args.stability_safety <= required_q:
        raise ValueError("stability-safety excludes the Polyak point")
    if args.beta_max <= float(required_beta):
        raise ValueError("beta-max excludes the Polyak point")
    if args.steps < 1 or args.starts < 1:
        raise ValueError("steps and starts must be positive")
    if args.forward_groups < 20 or args.forward_candidates < 24:
        raise ValueError("forward dataset is too small for a three-way split")
    if args.oracle_points < 2 or args.meta_pairs < 30:
        raise ValueError("oracle-points or meta-pairs are too small")
    if min(args.oracle_grid, args.oracle_refine, args.search_grid) < 3:
        raise ValueError("search grids must contain at least three points")
    if min(args.predicted_refine, args.exact_refine) < 3:
        raise ValueError("refinement grids must contain at least three points")
    if args.exact_top_k < 2:
        raise ValueError("exact-top-k must be at least two")
    if min(args.oracle_stages, args.exact_stages, args.batch_size) < 1:
        raise ValueError("stage counts and batch-size must be positive")
    if not np.isfinite(args.alpha_ridge) or args.alpha_ridge < 0.0:
        raise ValueError("alpha-ridge must be finite and non-negative")
    for name in ["forward_degrees", "q_degrees", "beta_degrees"]:
        degrees = getattr(args, name)
        if not degrees or any(value < 1 for value in degrees):
            raise ValueError(f"{name} must contain positive integers")
        setattr(args, name, sorted(set(degrees)))


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    starts = make_starts(args.starts, seed=args.seed + 1)
    np.save(DATA_DIR / "initial_points.npy", starts)

    print("\n1/6 Exact finite-horizon oracle")
    rng = np.random.default_rng(args.seed + 100)
    oracle_kappa = stratified_log_uniform(
        args.oracle_points, args.kappa_min, args.kappa_max, rng
    )
    oracle_alpha_L, oracle_beta, oracle_q, oracle_score = oracle_search(
        oracle_kappa,
        starts,
        args.steps,
        args.oracle_grid,
        args.oracle_refine,
        args.oracle_stages,
        args.batch_size,
        args.stability_safety,
        args.beta_max,
    )
    oracle_l = np.ones_like(oracle_kappa)
    oracle_L = oracle_kappa.copy()
    oracle_alpha_theory, oracle_beta_theory = theory_from_L_l(
        oracle_L, oracle_l
    )
    oracle_frame = pd.DataFrame(
        {
            "L": oracle_L,
            "l": oracle_l,
            "kappa": oracle_kappa,
            "alpha_found": oracle_alpha_L / oracle_L,
            "beta_found": oracle_beta,
            "q_found": oracle_q,
            "J_found": oracle_score,
            "alpha_theory": oracle_alpha_theory,
            "beta_theory": oracle_beta_theory,
            "J_theory": terminal_rate(
                oracle_kappa,
                oracle_alpha_theory * oracle_L,
                oracle_beta_theory,
                starts,
                steps=args.steps,
            ),
        }
    )
    oracle_frame.to_csv(DATA_DIR / "oracle_diagnostic.csv", index=False)

    print("\n2/6 Building forward dataset and selecting EIKG degree")
    forward_frame = generate_forward_dataset(
        starts,
        args.forward_groups,
        args.forward_candidates,
        args.kappa_min,
        args.kappa_max,
        args.beta_max,
        args.stability_safety,
        args.steps,
        args.seed,
    )
    groups = forward_frame["group_id"].unique()
    train_groups, validation_groups, test_groups = three_way_split(
        groups, args.seed
    )
    train_mask = forward_frame["group_id"].isin(train_groups).to_numpy()
    validation_mask = forward_frame["group_id"].isin(validation_groups).to_numpy()
    test_mask = forward_frame["group_id"].isin(test_groups).to_numpy()
    forward_frame["split"] = ""
    forward_frame.loc[train_mask, "split"] = "train"
    forward_frame.loc[validation_mask, "split"] = "validation"
    forward_frame.loc[test_mask, "split"] = "test"

    X_forward = forward_feature_matrix(
        forward_frame["kappa"],
        forward_frame["alpha_L"],
        forward_frame["beta"],
    ).astype(np.float64)
    y_forward = forward_frame["scaled_target_rate"].to_numpy(dtype=np.float64)
    forward_weights = None
    if not args.disable_forward_weight_repetition:
        forward_weights = forward_frame["sample_weight"].to_numpy()

    forward_model, forward_degree, forward_validation_mse, forward_selection = (
        select_degree_and_refit(
            "forward",
            X_forward[train_mask],
            y_forward[train_mask],
            X_forward[validation_mask],
            y_forward[validation_mask],
            args.forward_degrees,
            args.alpha_ridge,
            None if forward_weights is None else forward_weights[train_mask],
            None if forward_weights is None else forward_weights[validation_mask],
        )
    )
    forward_frame.to_pickle(DATA_DIR / "forward_dataset.pkl")
    forward_selection.to_csv(
        DATA_DIR / "forward_degree_selection.csv", index=False
    )
    joblib.dump(forward_model, MODELS_DIR / "eikg_forward.joblib")

    print("\n3/6 Evaluating forward EIKG on untouched test groups")
    forward_test_prediction = forward_model.predict(X_forward[test_mask])
    raw_test_prediction = forward_test_prediction / np.sqrt(
        forward_frame.loc[test_mask, "kappa"].to_numpy()
    )
    target_rank = forward_frame.groupby("group_id")["target_rate"].rank(
        method="first", pct=True
    )
    near_optimum_test = test_mask & (target_rank.to_numpy() <= 0.20)
    near_optimum_prediction = forward_model.predict(
        X_forward[near_optimum_test]
    )
    forward_test_metrics = {
        "test_scaled_r2": print_r2(
            "Forward scaled target",
            y_forward[test_mask],
            forward_test_prediction,
        ),
        "test_near_optimum_scaled_r2": print_r2(
            "Forward scaled target near optimum",
            y_forward[near_optimum_test],
            near_optimum_prediction,
        ),
        "test_raw_J300_r2": print_r2(
            "Forward raw J_300",
            forward_frame.loc[test_mask, "target_rate"],
            raw_test_prediction,
        ),
    }

    print("\n4/6 Building exact-refined inverse labels")
    all_pairs = sample_meta_pairs(
        args.meta_pairs, args.kappa_min, args.kappa_max, args.seed + 10
    )
    found_alpha_L, found_beta, found_q, found_score = theory_blind_inverse_search(
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
        args.beta_max,
    )
    all_pairs["alpha_found"] = found_alpha_L / all_pairs["L"].to_numpy()
    all_pairs["beta_found"] = found_beta
    all_pairs["q_found"] = found_q
    all_pairs["J_found"] = found_score
    theory_alpha_L = all_pairs["alpha_theory"].to_numpy() * all_pairs["L"].to_numpy()
    all_pairs["J_theory"] = terminal_rate(
        all_pairs["kappa"].to_numpy(),
        theory_alpha_L,
        all_pairs["beta_theory"].to_numpy(),
        starts,
        steps=args.steps,
    )

    row_indices = np.arange(len(all_pairs))
    train_index, validation_index, test_index = three_way_split(
        row_indices, args.seed + 20
    )
    all_pairs["split"] = ""
    all_pairs.loc[train_index, "split"] = "train"
    all_pairs.loc[validation_index, "split"] = "validation"
    all_pairs.loc[test_index, "split"] = "test"
    all_pairs.to_csv(DATA_DIR / "inverse_labels_all.csv", index=False)

    print("\n5/6 Selecting final q/beta EIKG degrees and refitting")
    X_meta = meta_feature_matrix(all_pairs["L"], all_pairs["l"])
    q_target = all_pairs["q_found"].to_numpy(dtype=np.float64)
    beta_target = all_pairs["beta_found"].to_numpy(dtype=np.float64)
    q_model, q_degree, q_validation_mse, q_selection = select_degree_and_refit(
        "final q",
        X_meta[train_index],
        q_target[train_index],
        X_meta[validation_index],
        q_target[validation_index],
        args.q_degrees,
        args.alpha_ridge,
    )
    beta_model, beta_degree, beta_validation_mse, beta_selection = (
        select_degree_and_refit(
            "final beta",
            X_meta[train_index],
            beta_target[train_index],
            X_meta[validation_index],
            beta_target[validation_index],
            args.beta_degrees,
            args.alpha_ridge,
        )
    )
    q_selection.to_csv(DATA_DIR / "q_degree_selection.csv", index=False)
    beta_selection.to_csv(DATA_DIR / "beta_degree_selection.csv", index=False)
    joblib.dump(q_model, MODELS_DIR / "eikg_q.joblib")
    joblib.dump(beta_model, MODELS_DIR / "eikg_beta.joblib")
    parameter_model = EIKGHeavyBallModel(
        q_model,
        beta_model,
        args.stability_safety,
        args.beta_max,
    )
    joblib.dump(parameter_model, MODELS_DIR / "eikg_heavy_ball.joblib")

    print("\n6/6 Evaluating final EIKG models on untouched test rows")
    test_frame = all_pairs.iloc[test_index].copy().reset_index(drop=True)
    raw_test_inputs = test_frame[["L", "l"]].to_numpy(dtype=np.float64)
    test_frame["q_pred"], test_frame["beta_pred"] = (
        parameter_model.predict_coordinates(raw_test_inputs)
    )
    predictions = parameter_model.predict(raw_test_inputs)
    test_frame["alpha_pred"] = predictions[:, 0]
    test_frame["alpha_L_pred"] = test_frame["alpha_pred"] * test_frame["L"]
    test_frame["J_pred"] = terminal_rate(
        test_frame["kappa"].to_numpy(),
        test_frame["alpha_L_pred"].to_numpy(),
        test_frame["beta_pred"].to_numpy(),
        starts,
        steps=args.steps,
    )
    test_frame.to_csv(DATA_DIR / "final_models_test.csv", index=False)
    final_test_metrics = {
        "test_alpha_vs_polyak_r2": print_r2(
            "Final alpha vs Polyak theory",
            test_frame["alpha_theory"],
            test_frame["alpha_pred"],
        ),
        "test_beta_vs_polyak_r2": print_r2(
            "Final beta vs Polyak theory",
            test_frame["beta_theory"],
            test_frame["beta_pred"],
        ),
    }

    save_plots(
        oracle_frame,
        test_frame,
        forward_selection,
        q_selection,
        beta_selection,
    )
    manifest = {
        "description": "Finite-horizon Heavy Ball experiment with EIKG models",
        "external_inputs": ["L", "l"],
        "forward_features": FORWARD_FEATURES,
        "internal_final_features": META_FEATURES,
        "target": {
            "definition": "mean_i(log10(max(f_300/f_0, 1e-300))) / 300",
            "forward_training_target": "sqrt(kappa) * J_300",
            "steps": args.steps,
            "starts": args.starts,
        },
        "inverse_label_search": {
            "polyak_candidate_included": False,
            "selection_criterion": "exact finite-horizon J_300",
            "polyak_used_in_inverse_label_candidates": False,
            "polyak_reference_computed_separately": True,
        },
        "data_split": {
            "train_fraction": 0.60,
            "validation_fraction": 0.20,
            "test_fraction": 0.20,
            "degree_selection_criterion": "minimum validation MSE",
            "reported_metrics_partition": "test only",
        },
        "model_selection": {
            "forward": {
                "candidate_degrees": args.forward_degrees,
                "selected_degree": forward_degree,
                "best_validation_mse": forward_validation_mse,
            },
            "final_q": {
                "candidate_degrees": args.q_degrees,
                "selected_degree": q_degree,
                "best_validation_mse": q_validation_mse,
            },
            "final_beta": {
                "candidate_degrees": args.beta_degrees,
                "selected_degree": beta_degree,
                "best_validation_mse": beta_validation_mse,
            },
        },
        "eikg_parameters": {
            "regularization": "ridge",
            "alpha_ridge": args.alpha_ridge,
            "scale": True,
            "scale_y": True,
            "normalize_latent": True,
            "forward_integer_weights_repeated": (
                not args.disable_forward_weight_repetition
            ),
        },
        "saved_models": {
            "forward": str(MODELS_DIR / "eikg_forward.joblib"),
            "q": str(MODELS_DIR / "eikg_q.joblib"),
            "beta": str(MODELS_DIR / "eikg_beta.joblib"),
            "public_parameters": str(MODELS_DIR / "eikg_heavy_ball.joblib"),
        },
        "config": vars(args),
        "metrics": {
            "forward_model_test": forward_test_metrics,
            "final_models_test": final_test_metrics,
        },
    }
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("\nSaved:")
    print(" ", MODELS_DIR / "eikg_forward.joblib")
    print(" ", MODELS_DIR / "eikg_q.joblib")
    print(" ", MODELS_DIR / "eikg_beta.joblib")
    print(" ", MODELS_DIR / "eikg_heavy_ball.joblib")
    print(" ", RESULTS_DIR / "metrics.json")
    print(" ", RESULTS_DIR / "degree_selection.png")
    print(" ", RESULTS_DIR / "oracle_vs_polyak.png")
    print(" ", RESULTS_DIR / "eikg_parameters_test.png")


if __name__ == "__main__":
    main()
