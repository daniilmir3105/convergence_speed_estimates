# -*- coding: utf-8 -*-
"""Generate a wide component-count dataset with all component Li/li values."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend

from .core import (
    FEATURE_SCHEMA_VERSION,
    build_feature_names,
    evaluate_instance,
    generate_max_quadratic_instance,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate the component-wise Li/li J_300 dataset"
    )
    parser.add_argument("--instances", type=int, default=5000)
    parser.add_argument("--starts", type=int, default=16)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--dimension-min", type=int, default=2)
    parser.add_argument("--dimension-max", type=int, default=6)
    parser.add_argument("--pieces-min", type=int, default=2)
    parser.add_argument("--pieces-max", type=int, default=20)
    parser.add_argument("--kappa-min", type=float, default=1.1)
    parser.add_argument("--kappa-max", type=float, default=20_000.0)
    parser.add_argument("--kappa-bins", type=int, default=10)
    parser.add_argument("--generation-attempts", type=int, default=4)
    parser.add_argument("--activity-directions", type=int, default=128)
    parser.add_argument("--stopping-tolerance", type=float, default=1e-10)
    parser.add_argument("--projection-tolerance", type=float, default=1e-11)
    parser.add_argument("--projection-max-iterations", type=int, default=300)
    parser.add_argument("--trajectory-instances", type=int, default=8)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(args):
    if args.instances < 100:
        raise ValueError("instances must be at least 100")
    if args.starts != 16 or args.steps != 300:
        raise ValueError("the unchanged target requires starts=16 and steps=300")
    if not 2 <= args.dimension_min <= args.dimension_max:
        raise ValueError("expected 2 <= dimension-min <= dimension-max")
    if not 2 <= args.pieces_min <= args.pieces_max:
        raise ValueError("expected 2 <= pieces-min <= pieces-max")
    if not 1.0 < args.kappa_min < args.kappa_max:
        raise ValueError("expected 1 < kappa-min < kappa-max")
    if args.kappa_bins < 2:
        raise ValueError("kappa-bins must be at least two")
    if args.generation_attempts < 1 or args.activity_directions < 16:
        raise ValueError("generation attempts and activity directions are too small")
    if args.stopping_tolerance <= 0.0 or args.projection_tolerance <= 0.0:
        raise ValueError("numeric tolerances must be positive")
    if args.projection_max_iterations < 1:
        raise ValueError("projection-max-iterations must be positive")
    if args.trajectory_instances < 0:
        raise ValueError("trajectory-instances cannot be negative")
    if args.n_jobs == 0:
        raise ValueError("n-jobs cannot be zero")


def stratified_log_uniform(size, low, high, rng):
    edges = np.linspace(np.log(low), np.log(high), size + 1)
    values = np.exp(rng.uniform(edges[:-1], edges[1:]))
    rng.shuffle(values)
    return values


def balanced_discrete_pairs(size, first_values, second_values, rng):
    cells = np.asarray(
        [(first, second) for first in first_values for second in second_values],
        dtype=np.int64,
    )
    repeats = int(np.ceil(size / len(cells)))
    schedule = np.tile(cells, (repeats, 1))[:size]
    rng.shuffle(schedule)
    return schedule


def active_piece_count(matrices, n_directions, seed):
    rng = np.random.default_rng(seed)
    dimension = matrices.shape[1]
    directions = rng.normal(size=(n_directions, dimension))
    directions /= np.maximum(
        np.linalg.norm(directions, axis=1, keepdims=True), 1e-300
    )
    values = 0.5 * np.einsum(
        "di,mij,dj->dm", directions, matrices, directions, optimize=True
    )
    return int(np.unique(np.argmax(values, axis=1)).size)


def generate_instances(args):
    rng = np.random.default_rng(args.seed)
    kappas = stratified_log_uniform(
        args.instances, args.kappa_min, args.kappa_max, rng
    )
    pairs = balanced_discrete_pairs(
        args.instances,
        range(args.dimension_min, args.dimension_max + 1),
        range(args.pieces_min, args.pieces_max + 1),
        rng,
    )
    log_edges = np.linspace(
        np.log(args.kappa_min), np.log(args.kappa_max), args.kappa_bins + 1
    )

    instances = []
    design_rows = []
    for index, ((dimension, n_pieces), kappa) in enumerate(zip(pairs, kappas)):
        best = None
        best_active = -1
        for attempt in range(args.generation_attempts):
            candidate = generate_max_quadratic_instance(
                int(dimension),
                int(n_pieces),
                l_value=1.0,
                kappa=float(kappa),
                seed=args.seed + 10_007 * index + 97 * attempt,
            )
            active_count = active_piece_count(
                candidate,
                n_directions=args.activity_directions,
                seed=args.seed + 31_337 * index + attempt,
            )
            if active_count > best_active:
                best = candidate
                best_active = active_count
        kappa_bin = int(
            np.clip(np.searchsorted(log_edges, np.log(kappa), side="right") - 1, 0, args.kappa_bins - 1)
        )
        instances.append(best)
        design_rows.append(
            {
                "instance_id": index,
                "design_dimension": int(dimension),
                "design_n_pieces": int(n_pieces),
                "design_kappa": float(kappa),
                "design_kappa_bin": kappa_bin,
                "active_piece_count_probe": int(best_active),
            }
        )
    return instances, pd.DataFrame(design_rows), log_edges


def save_instances(instances):
    dimensions = np.asarray([item.shape[1] for item in instances], dtype=np.int64)
    n_pieces = np.asarray([item.shape[0] for item in instances], dtype=np.int64)
    padded = np.zeros(
        (
            len(instances),
            int(n_pieces.max()),
            int(dimensions.max()),
            int(dimensions.max()),
        ),
        dtype=np.float64,
    )
    for index, matrices in enumerate(instances):
        padded[
            index,
            : matrices.shape[0],
            : matrices.shape[1],
            : matrices.shape[2],
        ] = matrices
    np.savez_compressed(
        DATA_DIR / "instances.npz",
        matrices=padded,
        dimensions=dimensions,
        n_pieces=n_pieces,
    )


def _evaluate(index, matrices, args):
    return evaluate_instance(
        instance_id=index,
        matrices=matrices,
        max_pieces=args.pieces_max,
        n_starts=args.starts,
        steps=args.steps,
        stopping_tolerance=args.stopping_tolerance,
        projection_tolerance=args.projection_tolerance,
        projection_max_iterations=args.projection_max_iterations,
        seed=args.seed + 1,
        keep_trajectories=index < args.trajectory_instances,
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    print("Generating balanced dimension and component-count design...")
    instances, design, log_edges = generate_instances(args)
    save_instances(instances)

    print("Calculating the unchanged J_300 target...")
    tasks = (
        delayed(_evaluate)(index, matrices, args)
        for index, matrices in enumerate(instances)
    )
    with parallel_backend("loky", inner_max_num_threads=1):
        outputs = Parallel(n_jobs=args.n_jobs, verbose=10)(tasks)

    rows = [item[0] for item in outputs]
    trajectories = [row for item in outputs for row in item[1]]
    frame = pd.DataFrame(rows).merge(design, on="instance_id", validate="one_to_one")
    frame = frame.sort_values("instance_id").reset_index(drop=True)
    frame.to_csv(DATA_DIR / "dataset.csv", index=False)
    pd.DataFrame(trajectories).to_csv(
        DATA_DIR / "sample_trajectories.csv", index=False
    )

    runtime = (
        frame.groupby("n_pieces", observed=True)["evaluation_seconds"]
        .agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    runtime.to_csv(DATA_DIR / "runtime_by_n_pieces.csv", index=False)

    elapsed = time.perf_counter() - started
    feature_names = build_feature_names(args.pieces_max)
    manifest = {
        "description": "Component-wise Li/li finite-horizon dataset",
        "objective": "max_j 0.5*x^T*A_j*x",
        "known_minimizer": "x*=0, f*=0",
        "target": (
            "J_300 = sum_i log10(max(f(x_300^i)/f(x_0^i), 1e-300)) "
            "/ (16*300)"
        ),
        "target_changed_from_original": False,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_representation": (
            "all component li/Li pairs sorted by decreasing Li, represented "
            "relative to global l, zero-padded with presence masks"
        ),
        "feature_names": feature_names,
        "config": vars(args),
        "log_kappa_bin_edges": log_edges.tolist(),
        "elapsed_seconds": elapsed,
        "saved_rows": len(frame),
    }
    (DATA_DIR / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(frame):,} rows in {elapsed:.1f}s")
    print(" ", DATA_DIR / "dataset.csv")
    print(" ", DATA_DIR / "runtime_by_n_pieces.csv")
    print(" ", DATA_DIR / "dataset_manifest.json")


if __name__ == "__main__":
    main()
