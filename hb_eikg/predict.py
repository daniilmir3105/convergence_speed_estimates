# -*- coding: utf-8 -*-
"""Use the saved EIKG Heavy Ball model and compare it with Polyak theory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from .model import polyak_theory


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "eikg_heavy_ball.joblib"


def predict(L, l):
    model = joblib.load(MODEL_PATH)
    X = np.array([[L, l]], dtype=np.float64)
    alpha, beta = model.predict(X)[0]
    return float(alpha), float(beta)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict Heavy Ball alpha and beta with EIKG"
    )
    parser.add_argument("L", type=float, help="largest curvature")
    parser.add_argument("l", type=float, help="smallest curvature")
    args = parser.parse_args(argv)

    L = max(args.L, args.l)
    l = min(args.L, args.l)
    if l <= 0.0:
        raise ValueError("L and l must be positive")

    alpha_model, beta_model = predict(L, l)
    alpha_theory, beta_theory = polyak_theory(L, l)
    result = {
        "inputs": {"L": L, "l": l, "kappa": L / l},
        "finite_horizon_eikg_model": {
            "alpha": alpha_model,
            "beta": beta_model,
        },
        "polyak_reference": {
            "alpha": float(alpha_theory),
            "beta": float(beta_theory),
        },
        "absolute_difference": {
            "alpha": abs(alpha_model - float(alpha_theory)),
            "beta": abs(beta_model - float(beta_theory)),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
