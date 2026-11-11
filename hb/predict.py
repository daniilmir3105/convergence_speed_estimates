# -*- coding: utf-8 -*-
"""Predict stable Heavy Ball alpha and beta from external inputs L and l."""

from __future__ import annotations

import argparse
import json

import joblib
import numpy as np

from pipeline import (
    BETA_MAX,
    MODELS_DIR,
    STABILITY_SAFETY,
    meta_feature_matrix,
    theory_from_L_l,
)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Predict finite-horizon Heavy Ball parameters"
    )
    parser.add_argument("L", type=float, help="largest Hessian eigenvalue")
    parser.add_argument("l", type=float, help="smallest Hessian eigenvalue")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 0 < args.l < args.L:
        raise ValueError("Expected 0 < l < L")

    q_model = joblib.load(MODELS_DIR / "gradient_boosting_q.joblib")
    beta_model = joblib.load(MODELS_DIR / "gradient_boosting_beta.joblib")
    X = meta_feature_matrix(np.array([args.L]), np.array([args.l]))

    q = float(np.clip(q_model.predict(X)[0], 1e-8, STABILITY_SAFETY))
    beta = float(np.clip(beta_model.predict(X)[0], 0.0, BETA_MAX))
    alpha_L = 2.0 * q * (1.0 + beta)
    alpha = alpha_L / args.L
    alpha_theory, beta_theory = theory_from_L_l(args.L, args.l)

    result = {
        "inputs": {"L": args.L, "l": args.l, "kappa": args.L / args.l},
        "finite_horizon_model": {
            "alpha": float(alpha),
            "beta": beta,
            "q": q,
            "stable": bool(q < 1.0),
        },
        "polyak_reference": {
            "alpha": float(alpha_theory),
            "beta": float(beta_theory),
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
