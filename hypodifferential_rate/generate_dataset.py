# -*- coding: utf-8 -*-
"""Generate max-quadratic problems and exact finite-horizon rate labels."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend

from .core import (
    FEATURE_NAMES,
    compute_instance_features,
    evaluate_instance,
    generate_max_quadratic_instance,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


def stratified_log_uniform(size, low, high, rng):
    edges = np.linspace(np.log(low), np.log(high), size + 1)
    values = np.exp(rng.uniform(edges[:-1], edges[1:]))
    rng.shuffle(values)
    return values


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate hypodifferential-descent rate dataset"
    )
    parser.add_argument("--instances", type=int, default=1000)
    parser.add_argument("--starts", type=int, default=16)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--dimension-min", type=int, default=2)
    parser.add_argument("--dimension-max", type=int, default=6)
    parser.add_argument("--pieces-min", type=int, default=2)
    parser.add_argument("--pieces-max", type=int, default=6)
    parser.add_argument("--kappa-min", type=float, default=1.1)
    parser.add_argument("--kappa-max", type=float, default=20_000.0)
    parser.add_argument("--l-min", type=float, default=1.0)
    parser.add_argument("--l-max", type=float, default=50.0)
    parser.add_argument("--stopping-tolerance", type=float, default=1e-10)
    parser.add_argument("--projection-tolerance", type=float, default=1e-11)
    parser.add_argument("--projection-max-iterations", type=int, default=300)
    parser.add_argument("--trajectory-instances", type=int, default=8)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(args):
    if args.instances < 30:
        raise ValueError("instances must be at least 30 for train/validation/test")
    if args.starts < 1:
        raise ValueError("starts must be positive")
    if args.steps != 300:
        raise ValueError("this experiment defines J_300, so steps must equal 300")
    if not 2 <= args.dimension_min <= args.dimension_max:
        raise ValueError("expected 2 <= dimension-min <= dimension-max")
    if not 2 <= args.pieces_min <= args.pieces_max:
        raise ValueError("expected 2 <= pieces-min <= pieces-max")
    if not 1.0 < args.kappa_min < args.kappa_max:
        raise ValueError("expected 1 < kappa-min < kappa-max")
    if not 0.0 < args.l_min <= args.l_max:
        raise ValueError("expected 0 < l-min <= l-max")
    if args.stopping_tolerance <= 0.0 or args.projection_tolerance <= 0.0:
        raise ValueError("numeric tolerances must be positive")
    if args.projection_max_iterations < 1:
        raise ValueError("projection-max-iterations must be positive")
    if args.trajectory_instances < 0:
        raise ValueError("trajectory-instances cannot be negative")
    if args.n_jobs == 0:
        raise ValueError("n-jobs cannot be zero")


def generate_instances(args):
    rng = np.random.default_rng(args.seed)
    kappas = stratified_log_uniform(
        args.instances, args.kappa_min, args.kappa_max, rng
    )
    l_values = np.exp(
        rng.uniform(np.log(args.l_min), np.log(args.l_max), size=args.instances)
    )
    dimensions = rng.integers(
        args.dimension_min, args.dimension_max + 1, size=args.instances
    )
    n_pieces = rng.integers(
        args.pieces_min, args.pieces_max + 1, size=args.instances
    )

    instances = []
    for index in range(args.instances):
        best = None
        best_active = -1.0
        required_active = min(int(n_pieces[index]), 3)
        for attempt in range(12):
            candidate = generate_max_quadratic_instance(
                int(dimensions[index]),
                int(n_pieces[index]),
                float(l_values[index]),
                float(kappas[index]),
                seed=args.seed + 10_007 * index + 97 * attempt,
            )
            active_fraction = compute_instance_features(candidate)[
                "active_piece_fraction"
            ]
            active_count = int(round(active_fraction * n_pieces[index]))
            if active_fraction > best_active:
                best = candidate
                best_active = active_fraction
            if active_count >= required_active:
                break
        instances.append(best)
    return instances


def save_instances(instances):
    dimensions = np.asarray([item.shape[1] for item in instances], dtype=np.int64)
    n_pieces = np.asarray([item.shape[0] for item in instances], dtype=np.int64)
    padded = np.zeros(
        (len(instances), int(n_pieces.max()), int(dimensions.max()), int(dimensions.max())),
        dtype=np.float64,
    )
    for index, matrices in enumerate(instances):
        padded[index, : matrices.shape[0], : matrices.shape[1], : matrices.shape[2]] = matrices
    np.savez_compressed(
        DATA_DIR / "instances.npz",
        matrices=padded,
        dimensions=dimensions,
        n_pieces=n_pieces,
    )


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    print("Generating convex max-quadratic instances...")
    instances = generate_instances(args)
    save_instances(instances)

    print("Running hypodifferential descent and calculating J_300...")
    tasks = (
        delayed(evaluate_instance)(
            index,
            matrices,
            n_starts=args.starts,
            steps=args.steps,
            stopping_tolerance=args.stopping_tolerance,
            projection_tolerance=args.projection_tolerance,
            projection_max_iterations=args.projection_max_iterations,
            seed=args.seed + 1,
            keep_trajectories=index < args.trajectory_instances,
        )
        for index, matrices in enumerate(instances)
    )
    with parallel_backend("loky", inner_max_num_threads=1):
        outputs = Parallel(n_jobs=args.n_jobs, verbose=10)(tasks)

    rows = [item[0] for item in outputs]
    trajectories = [row for item in outputs for row in item[1]]
    frame = pd.DataFrame(rows).sort_values("instance_id").reset_index(drop=True)
    frame.to_csv(DATA_DIR / "dataset.csv", index=False)
    pd.DataFrame(trajectories).to_csv(
        DATA_DIR / "sample_trajectories.csv", index=False
    )

    elapsed = time.perf_counter() - started
    manifest = {
        "description": "Finite-horizon hypodifferential-descent dataset",
        "objective": "max_j 0.5*x^T*A_j*x",
        "known_minimizer": "x*=0, f*=0",
        "target": "J_K = mean_i(log10(max(f(x_K)/f(x_0), 1e-300))) / K",
        "features": FEATURE_NAMES,
        "config": vars(args),
        "elapsed_seconds": elapsed,
        "saved_rows": len(frame),
    }
    (DATA_DIR / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(frame):,} rows in {elapsed:.1f}s")
    print(" ", DATA_DIR / "dataset.csv")
    print(" ", DATA_DIR / "instances.npz")
    print(" ", DATA_DIR / "sample_trajectories.csv")
    print(" ", DATA_DIR / "dataset_manifest.json")


if __name__ == "__main__":
    main()
