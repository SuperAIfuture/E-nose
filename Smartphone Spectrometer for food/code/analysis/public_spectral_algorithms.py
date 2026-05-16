"""Public spectral-analysis algorithms for the smartphone spectrometer study.

This script contains reusable preprocessing, validation, and evaluation routines
for the released article data tables. 
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR


FEATURE_PREFIX = "wl_"


def feature_columns(df: pd.DataFrame, prefix: str = FEATURE_PREFIX) -> list[str]:
    """Return spectral feature columns sorted by their numeric suffix."""
    cols = [col for col in df.columns if col.startswith(prefix)]
    return sorted(cols, key=lambda name: float(name.replace(prefix, "")))


def spectra_matrix(df: pd.DataFrame, prefix: str = FEATURE_PREFIX) -> tuple[np.ndarray, list[str]]:
    """Extract the spectral matrix and matching feature-column names."""
    cols = feature_columns(df, prefix=prefix)
    if not cols:
        raise ValueError(f"No spectral columns with prefix {prefix!r}.")
    return df[cols].to_numpy(dtype=float), cols


def sg_smooth(X: np.ndarray, window_length: int = 11, polyorder: int = 2) -> np.ndarray:
    """Apply Savitzky-Golay smoothing along the spectral axis."""
    return savgol_filter(X, window_length=window_length, polyorder=polyorder, axis=1, mode="interp")


def snv(X: np.ndarray) -> np.ndarray:
    """Apply standard normal variate normalization row by row."""
    mean = X.mean(axis=1, keepdims=True)
    std = X.std(axis=1, ddof=1, keepdims=True)
    std[std == 0] = 1.0
    return (X - mean) / std


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the regression metrics reported in the article tables."""
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the classification metrics reported in the article tables."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def leave_one_group_regression(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Run leave-one-group-out regression and return predictions and metrics."""
    pred = np.full_like(y, np.nan, dtype=float)
    for group in pd.unique(groups):
        test = groups == group
        train = ~test
        model = clone(estimator)
        model.fit(X[train], y[train])
        pred[test] = model.predict(X[test]).ravel()
    return pred, regression_metrics(y, pred)


def leave_one_group_classification(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Run leave-one-group-out classification and return predictions and metrics."""
    pred = np.empty_like(y, dtype=object)
    for group in pd.unique(groups):
        test = groups == group
        train = ~test
        model = clone(estimator)
        model.fit(X[train], y[train])
        pred[test] = model.predict(X[test])
    return pred, classification_metrics(y, pred)


def regression_candidates() -> dict[str, object]:
    """Return representative regression estimators used for model screening."""
    return {
        "linear": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
        "ridge": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))]),
        "lasso": Pipeline([("scale", StandardScaler()), ("model", Lasso(alpha=0.001, max_iter=50000))]),
        "svr_rbf": Pipeline([("scale", StandardScaler()), ("model", SVR(kernel="rbf", C=10.0, gamma="scale"))]),
        "pls": Pipeline([("scale", StandardScaler()), ("model", PLSRegression(n_components=5))]),
        "rf": RandomForestRegressor(n_estimators=500, random_state=42),
    }


def classification_candidates() -> dict[str, object]:
    """Return representative classifiers used for screening-band evaluation."""
    return {
        "logistic": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=1.0, max_iter=5000, multi_class="auto")),
        ]),
        "svc_rbf": Pipeline([
            ("scale", StandardScaler()),
            ("model", SVC(kernel="rbf", C=10.0, gamma="scale")),
        ]),
    }


def window_occlusion_delta_rmse(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list[str],
    window_size: int = 10,
) -> pd.DataFrame:
    """Estimate model sensitivity by median-replacing consecutive spectral windows."""
    base_pred, base_metrics = leave_one_group_regression(estimator, X, y, groups)
    base_rmse = base_metrics["rmse"]
    rows = []
    for start in range(0, X.shape[1], window_size):
        stop = min(start + window_size, X.shape[1])
        X_occ = X.copy()
        X_occ[:, start:stop] = np.median(X[:, start:stop], axis=0, keepdims=True)
        pred, metrics = leave_one_group_regression(estimator, X_occ, y, groups)
        rows.append(
            {
                "window_start": feature_names[start],
                "window_end": feature_names[stop - 1],
                "base_rmse": base_rmse,
                "occluded_rmse": metrics["rmse"],
                "delta_rmse": metrics["rmse"] - base_rmse,
            }
        )
    return pd.DataFrame(rows)


@dataclass
class TableLocation:
    module: str
    file_name: str


def load_public_table(package_root: Path, location: TableLocation) -> pd.DataFrame:
    """Load one released data table from the public package layout."""
    return pd.read_csv(package_root / "data" / location.module / location.file_name)
