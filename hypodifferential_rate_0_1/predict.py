# -*- coding: utf-8 -*-
"""Predict J_300 and rho for one saved max-of-quadratics instance."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from .core import compute_component_features


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "models" / "extra_trees_J300.joblib"


def build_parser():
    parser = argparse.ArgumentParser(description="Predict the convergence rate")
    parser.add_argument(
        "--instance",
        type=Path,
        required=True,
        help="NPY file containing matrices with shape (n_pieces, n, n)",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    bundle = joblib.load(args.model)
    matrices = np.load(args.instance, allow_pickle=False)
    max_pieces = int(bundle["feature_metadata"]["max_pieces"])
    features = compute_component_features(matrices, max_pieces=max_pieces)
    X = np.asarray(
        [[features[name] for name in bundle["feature_names"]]],
        dtype=np.float64,
    )
    J300 = float(np.asarray(bundle["model"].predict(X)).reshape(-1)[0])
    print(f"model: {args.model}")
    print(f"n_pieces: {matrices.shape[0]}")
    print(f"J_300: {J300:.12g}")
    print(f"rho: {10.0**J300:.12g}")


if __name__ == "__main__":
    main()
