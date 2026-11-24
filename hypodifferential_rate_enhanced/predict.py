# -*- coding: utf-8 -*-
"""Predict J_300 and rho with the enhanced Extra Trees model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from .core import compute_instance_features, validate_matrices


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "models" / "extra_trees_enhanced_J300.joblib"


def build_parser():
    parser = argparse.ArgumentParser(description="Predict enhanced J_300 and rho")
    parser.add_argument("input", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    return parser


def load_instance(path, index):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        matrices = np.load(path)
        if matrices.ndim != 3:
            raise ValueError("A .npy file must contain one (pieces,n,n) array")
        return validate_matrices(matrices)
    if path.suffix.lower() != ".npz":
        raise ValueError("Expected a .npy or .npz file")

    with np.load(path) as data:
        if "matrices" not in data:
            raise ValueError("The .npz file must contain a 'matrices' array")
        values = data["matrices"]
        if values.ndim == 3:
            return validate_matrices(values)
        if values.ndim != 4:
            raise ValueError("matrices must be 3D (one instance) or 4D (dataset)")
        if index is None:
            raise ValueError("Use --index when the .npz file contains a dataset")
        if not 0 <= index < len(values):
            raise IndexError(f"index must be in [0, {len(values) - 1}]")
        if "dimensions" not in data or "n_pieces" not in data:
            raise ValueError("A padded dataset requires dimensions and n_pieces")
        dimension = int(data["dimensions"][index])
        n_pieces = int(data["n_pieces"][index])
        matrices = values[index, :n_pieces, :dimension, :dimension]
        return validate_matrices(matrices)


def predict(matrices, model_path=DEFAULT_MODEL):
    bundle = joblib.load(model_path)
    config = bundle["feature_config"]
    features = compute_instance_features(
        matrices,
        n_starts=config["n_starts"],
        seed=config["seed"],
        projection_tolerance=config["projection_tolerance"],
        projection_max_iterations=config["projection_max_iterations"],
    )
    feature_names = bundle["feature_names"]
    X = np.asarray([[features[name] for name in feature_names]], dtype=np.float64)
    J300 = float(bundle["model"].predict(X)[0])
    return J300, float(10.0**J300), features, bundle


def main(argv=None):
    args = build_parser().parse_args(argv)
    matrices = load_instance(args.input, args.index)
    J300, rho, features, bundle = predict(matrices, args.model)
    result = {
        "input": {
            "matrix_file": str(args.input.resolve()),
            "index": args.index,
            "dimension": matrices.shape[1],
            "n_pieces": matrices.shape[0],
        },
        "model": {
            "path": str(args.model.resolve()),
            "feature_set": bundle["feature_set"],
        },
        "prediction": {
            "J300": J300,
            "rho": rho,
            "interpretation": "smaller J300/rho means faster convergence",
        },
        "features": {name: features[name] for name in bundle["feature_names"]},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

