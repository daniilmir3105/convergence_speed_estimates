# -*- coding: utf-8 -*-
"""Predict J_300 and rho for a maximum of convex quadratics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from .core import compute_instance_features, validate_matrices


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "models" / "gradient_boosting_J300.joblib"


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
    features = compute_instance_features(matrices)
    feature_names = bundle["feature_names"]
    X = np.asarray([[features[name] for name in feature_names]], dtype=np.float64)
    J300 = float(bundle["model"].predict(X)[0])
    return J300, float(10.0**J300), features


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict a finite-horizon hypodifferential-descent rate"
    )
    parser.add_argument(
        "matrix_file",
        help=".npy with one instance or .npz with one/generated instances",
    )
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args(argv)

    matrices = load_instance(args.matrix_file, args.index)
    J300, rho, features = predict(matrices, args.model)
    result = {
        "input": {
            "matrix_file": str(Path(args.matrix_file).resolve()),
            "index": args.index,
            "dimension": matrices.shape[1],
            "n_pieces": matrices.shape[0],
        },
        "prediction": {
            "J300": J300,
            "rho": rho,
            "interpretation": "smaller J300/rho means faster convergence",
        },
        "features": features,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
