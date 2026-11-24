# -*- coding: utf-8 -*-
"""Generate a balanced 5,000-instance enhanced hypodifferential dataset."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend

from .core import (
    ALL_FEATURE_NAMES,
    count_active_pieces_on_starts,
    evaluate_instance,
    generate_max_quadratic_instance,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"


@dataclass(frozen=True)
class InstanceSpec:
    instance_id: int
    dimension: int
    n_pieces: int
    kappa_bin: int
    kappa: float
    l_value: float
    hard_oversampled: bool


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate the enhanced balanced J_300 dataset"
    )
    parser.add_argument("--instances", type=int, default=5000)
    parser.add_argument("--starts", type=int, default=16)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--dimension-min", type=int, default=2)
    parser.add_argument("--dimension-max", type=int, default=6)
    parser.add_argument("--pieces-min", type=int, default=2)
    parser.add_argument("--pieces-max", type=int, default=6)
    parser.add_argument("--kappa-min", type=float, default=1.1)
    parser.add_argument("--kappa-max", type=float, default=20_000.0)
    parser.add_argument("--kappa-bins", type=int, default=10)
    parser.add_argument("--hard-bin-fraction", type=float, default=0.30)
    parser.add_argument("--hard-weight", type=float, default=2.0)
    parser.add_argument("--l-min", type=float, default=1.0)
    parser.add_argument("--l-max", type=float, default=50.0)
    parser.add_argument("--stopping-tolerance", type=float, default=1e-10)
    parser.add_argument("--projection-tolerance", type=float, default=1e-11)
    parser.add_argument("--projection-max-iterations", type=int, default=300)
    parser.add_argument("--generation-attempts", type=int, default=8)
    parser.add_argument("--trajectory-instances", type=int, default=8)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def validate_args(args):
    if args.instances < 30:
        raise ValueError("instances must be at least 30")
    if args.starts != 16:
        raise ValueError("the unchanged target definition requires starts=16")
    if args.steps != 300:
        raise ValueError("the unchanged target definition requires steps=300")
    if not 2 <= args.dimension_min <= args.dimension_max:
        raise ValueError("expected 2 <= dimension-min <= dimension-max")
    if not 2 <= args.pieces_min <= args.pieces_max:
        raise ValueError("expected 2 <= pieces-min <= pieces-max")
    if not 1.0 < args.kappa_min < args.kappa_max:
        raise ValueError("expected 1 < kappa-min < kappa-max")
    if args.kappa_bins < 2:
        raise ValueError("kappa-bins must be at least two")
    if not 0.0 < args.hard_bin_fraction <= 1.0:
        raise ValueError("hard-bin-fraction must belong to (0, 1]")
    if args.hard_weight < 1.0:
        raise ValueError("hard-weight must be at least one")
    if not 0.0 < args.l_min <= args.l_max:
        raise ValueError("expected 0 < l-min <= l-max")
    if args.stopping_tolerance <= 0.0 or args.projection_tolerance <= 0.0:
        raise ValueError("numeric tolerances must be positive")
    if args.projection_max_iterations < 1 or args.generation_attempts < 1:
        raise ValueError("iteration and attempt counts must be positive")
    if args.trajectory_instances < 0:
        raise ValueError("trajectory-instances cannot be negative")
    if args.n_jobs == 0:
        raise ValueError("n-jobs cannot be zero")


def _allocate_counts(total, weights):
    weights = np.asarray(weights, dtype=np.float64)
    raw = total * weights / np.sum(weights)
    counts = np.floor(raw).astype(np.int64)
    remaining = int(total - np.sum(counts))
    if remaining:
        order = np.argsort(-(raw - counts), kind="stable")
        counts[order[:remaining]] += 1
    if int(np.sum(counts)) != total:
        raise RuntimeError("internal allocation error")
    return counts


def build_balanced_specs(args):
    rng = np.random.default_rng(args.seed)
    log_edges = np.linspace(
        np.log(args.kappa_min), np.log(args.kappa_max), args.kappa_bins + 1
    )
    hard_bin_start = max(
        0, int(np.floor(args.kappa_bins * (1.0 - args.hard_bin_fraction)))
    )

    cells = []
    weights = []
    for dimension in range(args.dimension_min, args.dimension_max + 1):
        for n_pieces in range(args.pieces_min, args.pieces_max + 1):
            for kappa_bin in range(args.kappa_bins):
                is_hard = n_pieces in (2, 3) and kappa_bin >= hard_bin_start
                cells.append((dimension, n_pieces, kappa_bin, is_hard))
                weights.append(args.hard_weight if is_hard else 1.0)

    counts = _allocate_counts(args.instances, weights)
    raw_specs = []
    for cell, count in zip(cells, counts):
        dimension, n_pieces, kappa_bin, is_hard = cell
        for _ in range(int(count)):
            kappa = float(
                np.exp(rng.uniform(log_edges[kappa_bin], log_edges[kappa_bin + 1]))
            )
            l_value = float(
                np.exp(rng.uniform(np.log(args.l_min), np.log(args.l_max)))
            )
            raw_specs.append(
                (dimension, n_pieces, kappa_bin, kappa, l_value, is_hard)
            )

    rng.shuffle(raw_specs)
    specs = [
        InstanceSpec(index, *values) for index, values in enumerate(raw_specs)
    ]
    return specs, log_edges


def generate_instances(specs, args):
    instances = []
    feature_seed = args.seed + 1
    for spec in specs:
        best = None
        best_active_count = -1
        required_active = min(spec.n_pieces, 3)
        for attempt in range(args.generation_attempts):
            candidate = generate_max_quadratic_instance(
                spec.dimension,
                spec.n_pieces,
                spec.l_value,
                spec.kappa,
                seed=args.seed + 10_007 * spec.instance_id + 97 * attempt,
            )
            active_count = count_active_pieces_on_starts(
                candidate, n_starts=args.starts, seed=feature_seed
            )
            if active_count > best_active_count:
                best = candidate
                best_active_count = active_count
            if active_count >= required_active:
                break
        instances.append(best)
    return instances


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


def _evaluate(spec, matrices, args):
    row, trajectories = evaluate_instance(
        spec.instance_id,
        matrices,
        n_starts=args.starts,
        steps=args.steps,
        stopping_tolerance=args.stopping_tolerance,
        projection_tolerance=args.projection_tolerance,
        projection_max_iterations=args.projection_max_iterations,
        seed=args.seed + 1,
        keep_trajectories=spec.instance_id < args.trajectory_instances,
    )
    row["design_kappa_bin"] = int(spec.kappa_bin)
    row["design_hard_oversampled"] = int(spec.hard_oversampled)
    return row, trajectories


def main(argv=None):
    args = build_parser().parse_args(argv)
    validate_args(args)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    print("Building balanced dimension/pieces/log-kappa design...")
    specs, log_edges = build_balanced_specs(args)
    print("Generating convex max-quadratic instances...")
    instances = generate_instances(specs, args)
    save_instances(instances)

    print("Calculating prior features and the unchanged J_300 target...")
    tasks = (
        delayed(_evaluate)(spec, matrices, args)
        for spec, matrices in zip(specs, instances)
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

    design_counts = (
        frame.groupby(
            ["dimension", "n_pieces", "design_kappa_bin"], observed=True
        )
        .size()
        .rename("count")
        .reset_index()
    )
    design_counts.to_csv(DATA_DIR / "design_counts.csv", index=False)

    elapsed = time.perf_counter() - started
    manifest = {
        "description": "Enhanced balanced finite-horizon dataset",
        "objective": "max_j 0.5*x^T*A_j*x",
        "known_minimizer": "x*=0, f*=0",
        "target": (
            "J_300 = sum_i log10(max(f(x_300^i)/f(x_0^i), 1e-300)) "
            "/ (16*300)"
        ),
        "target_changed_from_original": False,
        "features_use_full_300_step_trajectory": False,
        "features": ALL_FEATURE_NAMES,
        "feature_seed": args.seed + 1,
        "design": {
            "strategy": "balanced dimension x pieces x logarithmic kappa bin",
            "log_kappa_bin_edges": log_edges.tolist(),
            "hard_regime": "n_pieces in {2,3} and top hard-bin-fraction",
            "hard_weight": args.hard_weight,
        },
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
    print(" ", DATA_DIR / "design_counts.csv")
    print(" ", DATA_DIR / "dataset_manifest.json")


if __name__ == "__main__":
    main()
