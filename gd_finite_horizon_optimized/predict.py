# -*- coding: utf-8 -*-
"""Predict finite-horizon GD alpha and compare it with classical theory."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from .alpha_model import gd_theory


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "gradient_boosting_alpha.joblib"


def predict(L, l):
    model = joblib.load(MODEL_PATH)
    return float(model.predict(np.array([[L, l]], dtype=np.float64))[0])


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Predict alpha with the finite-horizon GD model"
    )
    parser.add_argument("L", type=float, help="largest curvature")
    parser.add_argument("l", type=float, help="smallest curvature")
    args = parser.parse_args(argv)

    alpha_model = predict(args.L, args.l)
    alpha_theory = float(gd_theory(args.L, args.l))
    L = max(args.L, args.l)
    l = min(args.L, args.l)
    kappa = L / l

    print(f"L={L:.10g}, l={l:.10g}, kappa={kappa:.10g}")
    print("Finite-horizon ML model:")
    print(f"  alpha = {alpha_model:.12g}")
    print("Classical GD theory:")
    print(f"  alpha = {alpha_theory:.12g}")
    print("Difference:")
    print(f"  absolute = {abs(alpha_model - alpha_theory):.12g}")
    print(
        "  relative = "
        f"{abs(alpha_model - alpha_theory) / max(abs(alpha_theory), 1e-300):.6%}"
    )


if __name__ == "__main__":
    main()
