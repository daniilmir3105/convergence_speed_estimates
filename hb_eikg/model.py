# -*- coding: utf-8 -*-
"""Public ``(L, l) -> (alpha, beta)`` wrapper for the EIKG experiment."""

from __future__ import annotations

import numpy as np


META_FEATURES = ["log_L", "log_l", "log_kappa", "sqrt_kappa"]


def normalize_pairs(X):
    values = np.asarray(X, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("X must have shape (n_samples, 2) with columns [L, l]")
    if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("L and l must be finite positive numbers")
    L = np.maximum(values[:, 0], values[:, 1])
    l = np.minimum(values[:, 0], values[:, 1])
    return L, l


def meta_feature_matrix(L, l):
    L = np.asarray(L, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    kappa = L / l
    return np.column_stack(
        [np.log(L), np.log(l), np.log(kappa), np.sqrt(kappa)]
    ).astype(np.float64)


def polyak_theory(L, l):
    L = np.asarray(L, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    high = np.maximum(L, l)
    low = np.minimum(L, l)
    if np.any(~np.isfinite(high)) or np.any(low <= 0.0):
        raise ValueError("L and l must be finite positive numbers")
    root_L = np.sqrt(high)
    root_l = np.sqrt(low)
    alpha = 4.0 / (root_L + root_l) ** 2
    beta = ((root_L - root_l) / (root_L + root_l)) ** 2
    return alpha, beta


class EIKGHeavyBallModel:
    """Saved pair of EIKG models exposing raw external inputs ``[L, l]``.

    The inner regressors predict stable coordinates q and beta. The public
    ``predict`` method returns columns ``[alpha, beta]``.
    """

    def __init__(
        self,
        q_regressor,
        beta_regressor,
        stability_safety=0.99999,
        beta_max=0.99,
    ):
        self.q_regressor = q_regressor
        self.beta_regressor = beta_regressor
        self.stability_safety = float(stability_safety)
        self.beta_max = float(beta_max)

    def predict_coordinates(self, X):
        L, l = normalize_pairs(X)
        features = meta_feature_matrix(L, l)
        q = np.asarray(self.q_regressor.predict(features), dtype=np.float64)
        beta = np.asarray(
            self.beta_regressor.predict(features), dtype=np.float64
        )
        q = np.clip(q, 1e-8, self.stability_safety)
        beta = np.clip(beta, 0.0, self.beta_max)
        return q, beta

    def predict(self, X):
        L, _ = normalize_pairs(X)
        q, beta = self.predict_coordinates(X)
        alpha = 2.0 * q * (1.0 + beta) / L
        return np.column_stack([alpha, beta])
