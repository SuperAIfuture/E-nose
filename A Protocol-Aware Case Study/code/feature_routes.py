#!/usr/bin/env python3
"""Feature routes and model builders for Protocol-B baseline suite."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, QuantileTransformer, RobustScaler, StandardScaler
from sklearn.svm import SVC


MODEL_ORDER = [
    "power_yeojohnson_svm",
    "signed_log_abs_svm",
    "routear_combined_svm",
    "quantile_normal_svm",
    "ratio_median_svm",
    "ss_svm_rbf",
    "robust_svm_rbf",
]


def _safe_iqr(x: np.ndarray) -> np.ndarray:
    q1 = np.percentile(x, 25, axis=0)
    q3 = np.percentile(x, 75, axis=0)
    iqr = q3 - q1
    iqr[iqr == 0.0] = 1.0
    return iqr


class RatioMedianTransformer(BaseEstimator, TransformerMixin):
    """Per-feature ratio to train median."""

    def __init__(self, eps: float = 1e-9):
        self.eps = float(eps)

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "RatioMedianTransformer":
        med = np.median(x, axis=0)
        med[np.abs(med) < self.eps] = self.eps
        self.median_ = med
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return x / self.median_


class SignedLogAbsTransformer(BaseEstimator, TransformerMixin):
    """Robust-centered signed-log transform."""

    def __init__(self, eps: float = 1e-9):
        self.eps = float(eps)

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "SignedLogAbsTransformer":
        self.median_ = np.median(x, axis=0)
        self.scale_ = _safe_iqr(x)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        z = (x - self.median_) / self.scale_
        return np.sign(z) * np.log1p(np.abs(z))


class RouteARCombinedTransformer(BaseEstimator, TransformerMixin):
    """Route A-R combined view: robust signed-log + block composition."""

    def __init__(self, block_size: int = 8, eps: float = 1e-9):
        self.block_size = int(block_size)
        self.eps = float(eps)

    def fit(self, x: np.ndarray, y: np.ndarray | None = None) -> "RouteARCombinedTransformer":
        if x.shape[1] % self.block_size != 0:
            raise ValueError(f"Feature dimension {x.shape[1]} not divisible by block_size={self.block_size}.")
        self.median_ = np.median(x, axis=0)
        self.scale_ = _safe_iqr(x)
        self.n_features_in_ = x.shape[1]
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if x.shape[1] != self.n_features_in_:
            raise ValueError("Input feature dimension mismatch with fitted transformer.")

        signed_log = np.sign((x - self.median_) / self.scale_) * np.log1p(np.abs((x - self.median_) / self.scale_))

        n_samples, n_features = x.shape
        n_blocks = n_features // self.block_size
        reshaped = x.reshape(n_samples, n_blocks, self.block_size)
        denom = np.sum(np.abs(reshaped), axis=2, keepdims=True) + self.eps
        block_comp = (reshaped / denom).reshape(n_samples, n_features)

        return np.hstack([signed_log, block_comp])


def _svc() -> SVC:
    return SVC(kernel="rbf")


def build_model(model_name: str) -> Pipeline:
    if model_name == "ss_svm_rbf":
        steps = [("scaler", StandardScaler()), ("svc", _svc())]
    elif model_name == "robust_svm_rbf":
        steps = [("scaler", RobustScaler()), ("svc", _svc())]
    elif model_name == "quantile_normal_svm":
        steps = [
            ("quantile", QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=42)),
            ("svc", _svc()),
        ]
    elif model_name == "power_yeojohnson_svm":
        steps = [("power", PowerTransformer(method="yeo-johnson", standardize=True)), ("svc", _svc())]
    elif model_name == "ratio_median_svm":
        steps = [("ratio", RatioMedianTransformer()), ("scaler", StandardScaler()), ("svc", _svc())]
    elif model_name == "signed_log_abs_svm":
        steps = [("signedlog", SignedLogAbsTransformer()), ("scaler", StandardScaler()), ("svc", _svc())]
    elif model_name == "routear_combined_svm":
        steps = [("routear", RouteARCombinedTransformer(block_size=8)), ("scaler", StandardScaler()), ("svc", _svc())]
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return Pipeline(steps=steps)

