# -*- coding: utf-8 -*-
"""Public ``(L, l) -> alpha`` wrapper for the EIKG experiment."""

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


def gd_theory(L, l):
    L = np.asarray(L, dtype=np.float64)
    l = np.asarray(l, dtype=np.float64)
    high = np.maximum(L, l)
    low = np.minimum(L, l)
    if np.any(~np.isfinite(high)) or np.any(low <= 0.0):
        raise ValueError("L and l must be finite positive numbers")
    return 2.0 / (high + low)


class EIKGAlphaModel:
    """Saved EIKG model exposing raw external inputs ``[L, l]``.

    The internal EIKG regressor predicts q = alpha*L/2.  The public ``predict``
    method clips q to the stable interval and reconstructs ordinary alpha.
    """

    def __init__(self, q_regressor, stability_safety=0.99999):
        self.q_regressor = q_regressor
        self.stability_safety = float(stability_safety)

    def predict_q(self, X):
        L, l = normalize_pairs(X)
        features = meta_feature_matrix(L, l)
        q = np.asarray(self.q_regressor.predict(features), dtype=np.float64)
        return np.clip(q, 1e-8, self.stability_safety)

    def predict(self, X):
        L, _ = normalize_pairs(X)
        return 2.0 * self.predict_q(X) / L
