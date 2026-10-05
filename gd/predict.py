# -*- coding: utf-8 -*-
"""Predict the GD step and compare it with the theoretical optimum."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "gradient_boosting_alpha.joblib"


def normalize_L_l(L, l):
    if L <= 0 or l <= 0:
        raise ValueError("L and l must be positive")
    if L < l:
        L, l = l, L
    return float(L), float(l)


def gd_theory(L, l):
    L, l = normalize_L_l(L, l)
    return 2.0 / (L + l)


def predict(L, l):
    L, l = normalize_L_l(L, l)
    X = np.array([[L, l]], dtype=np.float64)
    model = joblib.load(MODEL_PATH)
    return max(float(model.predict(X)[0]), 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("L", type=float)
    parser.add_argument("l", type=float)
    args = parser.parse_args()

    alpha_model = predict(args.L, args.l)
    alpha_theory = gd_theory(args.L, args.l)
    print("ML model:")
    print(f"  alpha = {alpha_model:.10f}")
    print("GD theory:")
    print(f"  alpha = {alpha_theory:.10f}")
    print("Absolute error:")
    print(f"  alpha = {abs(alpha_model - alpha_theory):.10f}")


if __name__ == "__main__":
    main()
