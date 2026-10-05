# -*- coding: utf-8 -*-
"""Load the two final models and predict Heavy Ball alpha and beta."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


def normalize_L_l(L, l):
    if L <= 0 or l <= 0:
        raise ValueError("L and l must be positive")
    if L < l:
        L, l = l, L
    return float(L), float(l)


def polyak_theory(L, l):
    """Return the theoretical Heavy Ball parameters for a quadratic."""
    L, l = normalize_L_l(L, l)
    sqrt_L = np.sqrt(L)
    sqrt_l = np.sqrt(l)
    alpha = 4.0 / (sqrt_L + sqrt_l) ** 2
    beta = ((sqrt_L - sqrt_l) / (sqrt_L + sqrt_l)) ** 2
    return float(alpha), float(beta)


def predict(L, l):
    L, l = normalize_L_l(L, l)

    X = np.array([[L, l]], dtype=np.float64)
    alpha_model = joblib.load(MODELS_DIR / "gradient_boosting_alpha.joblib")
    beta_model = joblib.load(MODELS_DIR / "gradient_boosting_beta.joblib")
    alpha = max(float(alpha_model.predict(X)[0]), 1e-12)
    beta = min(max(float(beta_model.predict(X)[0]), 0.0), 0.99)
    return alpha, beta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("L", type=float)
    parser.add_argument("l", type=float)
    args = parser.parse_args()
    alpha_model, beta_model = predict(args.L, args.l)
    alpha_theory, beta_theory = polyak_theory(args.L, args.l)

    print("ML model:")
    print(f"  alpha = {alpha_model:.10f}")
    print(f"  beta  = {beta_model:.10f}")
    print("Polyak theory:")
    print(f"  alpha = {alpha_theory:.10f}")
    print(f"  beta  = {beta_theory:.10f}")
    print("Absolute error:")
    print(f"  alpha = {abs(alpha_model - alpha_theory):.10f}")
    print(f"  beta  = {abs(beta_model - beta_theory):.10f}")


if __name__ == "__main__":
    main()
