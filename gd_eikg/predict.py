# -*- coding: utf-8 -*-
"""Use the saved EIKG alpha model and compare it with GD theory."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np

from .model import gd_theory


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "eikg_alpha.joblib"


def predict(L, l):
    model = joblib.load(MODEL_PATH)
    X = np.array([[L, l]], dtype=np.float64)
    return float(model.predict(X)[0])


def main(argv=None):
    parser = argparse.ArgumentParser(description="Predict GD alpha with EIKG")
    parser.add_argument("L", type=float, help="largest curvature")
    parser.add_argument("l", type=float, help="smallest curvature")
    args = parser.parse_args(argv)

    L = max(args.L, args.l)
    l = min(args.L, args.l)
    if l <= 0.0:
        raise ValueError("L and l must be positive")

    alpha_model = predict(L, l)
    alpha_theory = float(gd_theory(L, l))
    print(f"L={L:.10g}, l={l:.10g}, kappa={L / l:.10g}")
    print("Finite-horizon EIKG model:")
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
