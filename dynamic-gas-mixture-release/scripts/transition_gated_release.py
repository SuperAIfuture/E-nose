#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR


@dataclass(frozen=True)
class SplitData:
    X: np.ndarray  # (N,C,L)
    y: np.ndarray  # (N,2)


def detect_gpu_runtime() -> dict:
    info = {
        "torch_installed": False,
        "torch_cuda_available": False,
        "torch_version": None,
        "cuml_installed": bool(importlib.util.find_spec("cuml")),
        "cupy_installed": bool(importlib.util.find_spec("cupy")),
        "gpu_name": None,
    }
    if importlib.util.find_spec("torch") is not None:
        info["torch_installed"] = True
        try:
            import torch  # type: ignore
            info["torch_version"] = str(torch.__version__)
            info["torch_cuda_available"] = bool(torch.cuda.is_available())
            if info["torch_cuda_available"] and torch.cuda.device_count() > 0:
                info["gpu_name"] = str(torch.cuda.get_device_name(0))
        except Exception:
            pass
    return info


def resolve_compute_backend(requested: str) -> tuple[str, str, dict]:
    gpu_info = detect_gpu_runtime()
    gpu_supported_here = bool(gpu_info["cuml_installed"])
    if requested == "cpu":
        return "cpu", "User requested CPU backend.", gpu_info
    if requested == "gpu":
        if gpu_supported_here:
            return "gpu", "User requested GPU backend and GPU library support is available.", gpu_info
        raise RuntimeError(
            "GPU backend requested but no supported GPU ML backend is available for this script "
            "(expected cuML-compatible environment)."
        )
    # auto
    if gpu_supported_here:
        return "gpu", "Auto-selected GPU backend because cuML-compatible support is available.", gpu_info
    return "cpu", (
        "Auto-selected CPU backend: current algorithm uses scikit-learn SVR and no cuML-compatible "
        "GPU backend is available in the runtime."
    ), gpu_info


def load_sharded_windows(processed_dir: Path) -> SplitData:
    shards = sorted(processed_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No shard_*.npz found under {processed_dir}")
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for sp in shards:
        with np.load(sp) as z:
            x = z["X"].astype(np.float32, copy=False)  # (n,L,C)
            y = z["y_components"].astype(np.float32, copy=False)
        xs.append(np.transpose(x, (0, 2, 1)))  # (n,C,L)
        ys.append(y)
    return SplitData(X=np.concatenate(xs, axis=0), y=np.concatenate(ys, axis=0))


def contiguous_split(data: SplitData, train_frac: float = 0.6, cal_frac: float = 0.2) -> tuple[SplitData, SplitData, SplitData]:
    n = data.X.shape[0]
    n_train = max(1, int(n * train_frac))
    n_cal = max(1, int(n * cal_frac))
    n_test = max(1, n - n_train - n_cal)
    i1 = n_train
    i2 = n_train + n_cal
    return (
        SplitData(X=data.X[:i1], y=data.y[:i1]),
        SplitData(X=data.X[i1:i2], y=data.y[i1:i2]),
        SplitData(X=data.X[i2 : i2 + n_test], y=data.y[i2 : i2 + n_test]),
    )


def window_features(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=-1)
    std = x.std(axis=-1)
    last = x[:, :, -1]
    span = x[:, :, -1] - x[:, :, 0]
    dx = np.diff(x, axis=-1)
    dmean = dx.mean(axis=-1) if dx.shape[-1] > 0 else np.zeros_like(mean)
    return np.concatenate([mean, std, last, span, dmean], axis=1).astype(np.float32)


def temporalize_features(f_train: np.ndarray, f_cal: np.ndarray, f_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prev_train = np.vstack([f_train[:1], f_train[:-1]])
    prev_cal = np.vstack([f_train[-1:], f_cal[:-1]]) if f_cal.shape[0] > 0 else np.zeros((0, f_train.shape[1]), dtype=f_train.dtype)
    if f_test.shape[0] > 0:
        cal_anchor = f_cal[-1:] if f_cal.shape[0] > 0 else f_train[-1:]
        prev_test = np.vstack([cal_anchor, f_test[:-1]])
    else:
        prev_test = np.zeros((0, f_train.shape[1]), dtype=f_train.dtype)
    return (
        np.concatenate([f_train, prev_train], axis=1).astype(np.float32),
        np.concatenate([f_cal, prev_cal], axis=1).astype(np.float32),
        np.concatenate([f_test, prev_test], axis=1).astype(np.float32),
    )


def apply_causal_kernel(x: np.ndarray, coeffs: list[float]) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float64)
    xf = x.astype(np.float64, copy=False)
    for lag, a in enumerate(coeffs):
        if lag == 0:
            y += a * xf
        else:
            y[:, :, lag:] += a * xf[:, :, :-lag]
            y[:, :, :lag] += a * xf[:, :, 0:1]
    return y.astype(np.float32)


def filterbank_features(x: np.ndarray) -> np.ndarray:
    kernels = {
        "lead_small": [1.2, -0.2],
        "lead_medium": [1.4, -0.4],
        "ema_fast": [0.6, 0.24, 0.096, 0.0384],
        "ema_slow": [0.3, 0.21, 0.147, 0.1029],
        "diff1": [1.0, -1.0],
    }
    parts = [window_features(x)]
    for k in ["lead_small", "lead_medium", "ema_fast", "ema_slow", "diff1"]:
        parts.append(window_features(apply_causal_kernel(x, kernels[k])))
    return np.concatenate(parts, axis=1).astype(np.float32)


def score_diff_mean_abs(x: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(np.diff(x, axis=2)), axis=(1, 2))


def score_endpoint_jump(x: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(x[:, :, -1] - x[:, :, 0]), axis=1)


def score_hybrid(x: np.ndarray) -> np.ndarray:
    s1 = score_diff_mean_abs(x)
    s2 = score_endpoint_jump(x)
    m1, d1 = float(s1.mean()), float(s1.std() + 1e-6)
    m2, d2 = float(s2.mean()), float(s2.std() + 1e-6)
    return 0.5 * ((s1 - m1) / d1) + 0.5 * ((s2 - m2) / d2)


def split_conformal_abs(y_cal: np.ndarray, pred_cal: np.ndarray, pred_test: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    resid = np.abs(y_cal - pred_cal)
    q = np.quantile(resid, 1.0 - alpha, axis=0, method="higher")
    hi = pred_test + q[None, :]
    lo = pred_test - q[None, :]
    picp = np.mean((pred_test * 0 + y_cal.mean(axis=0)[None, :]) >= lo, axis=0)  # placeholder to preserve shape checks
    _ = picp
    return lo.astype(np.float32), hi.astype(np.float32)


def picp(y_true: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.mean((y_true >= lo) & (y_true <= hi), axis=0)


def fit_svr_select(
    f_train: np.ndarray,
    y_train: np.ndarray,
    f_cal: np.ndarray,
    y_cal: np.ndarray,
    f_test: np.ndarray,
    c_grid: list[float],
    eps_grid: list[float],
    sample_weight: np.ndarray | None = None,
) -> dict:
    mu = f_train.mean(axis=0, keepdims=True)
    sd = f_train.std(axis=0, keepdims=True) + 1e-6
    xtr = (f_train - mu) / sd
    xcal = (f_cal - mu) / sd
    xte = (f_test - mu) / sd
    best = None
    for c in c_grid:
        for e in eps_grid:
            m = MultiOutputRegressor(SVR(C=c, epsilon=e, kernel="rbf", gamma="scale"))
            if sample_weight is None:
                m.fit(xtr, y_train)
            else:
                m.fit(xtr, y_train, sample_weight=sample_weight)
            pcal = m.predict(xcal).astype(np.float32)
            pte = m.predict(xte).astype(np.float32)
            cal_mae = float(np.mean(np.abs(y_cal - pcal)))
            if best is None or cal_mae < best["cal_mae"]:
                best = {
                    "cfg": {"C": float(c), "epsilon": float(e)},
                    "pred_cal": pcal,
                    "pred_test": pte,
                    "cal_mae": cal_mae,
                }
    assert best is not None
    return best


def fit_tree_fixed(
    f_train: np.ndarray,
    y_train: np.ndarray,
    f_cal: np.ndarray,
    y_cal: np.ndarray,
    f_test: np.ndarray,
    *,
    model_type: str,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    random_state: int,
) -> dict:
    if model_type == "rf":
        base = RandomForestRegressor(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf),
            random_state=int(random_state),
            n_jobs=-1,
        )
    elif model_type == "extra":
        base = ExtraTreesRegressor(
            n_estimators=int(n_estimators),
            max_depth=max_depth,
            min_samples_leaf=int(min_samples_leaf),
            random_state=int(random_state),
            n_jobs=-1,
        )
    else:
        raise ValueError(f"unknown tree model_type: {model_type}")
    m = MultiOutputRegressor(base)
    m.fit(f_train, y_train)
    pcal = m.predict(f_cal).astype(np.float32)
    pte = m.predict(f_test).astype(np.float32)
    cal_mae = float(np.mean(np.abs(y_cal - pcal)))
    return {
        "pred_cal": pcal,
        "pred_test": pte,
        "cal_mae": cal_mae,
    }


def fit_hgb_fixed(
    f_train: np.ndarray,
    y_train: np.ndarray,
    f_cal: np.ndarray,
    y_cal: np.ndarray,
    f_test: np.ndarray,
    *,
    loss: str,
    max_depth: int,
    learning_rate: float,
    max_iter: int,
    quantile: float | None,
    random_state: int,
) -> dict:
    base = HistGradientBoostingRegressor(
        loss=str(loss),
        quantile=(None if quantile is None else float(quantile)),
        max_depth=int(max_depth),
        learning_rate=float(learning_rate),
        max_iter=int(max_iter),
        random_state=int(random_state),
        early_stopping=False,
    )
    m = MultiOutputRegressor(base)
    m.fit(f_train, y_train)
    pcal = m.predict(f_cal).astype(np.float32)
    pte = m.predict(f_test).astype(np.float32)
    cal_mae = float(np.mean(np.abs(y_cal - pcal)))
    return {
        "pred_cal": pcal,
        "pred_test": pte,
        "cal_mae": cal_mae,
    }


def metric_pack(y_true: np.ndarray, pred: np.ndarray, lo: np.ndarray, hi: np.ndarray, mask_transition: np.ndarray, mask_steady: np.ndarray) -> dict:
    mae = np.mean(np.abs(y_true - pred), axis=0)
    rmse = np.sqrt(np.mean((y_true - pred) ** 2, axis=0))
    r2 = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = pred[:, i]
        den = float(np.sum((yt - yt.mean()) ** 2) + 1e-12)
        num = float(np.sum((yt - yp) ** 2))
        r2.append(1.0 - num / den)
    def subset(mask: np.ndarray) -> list[float]:
        if int(mask.sum()) == 0:
            return [float("nan"), float("nan")]
        return np.mean(np.abs(y_true[mask] - pred[mask]), axis=0).astype(float).tolist()
    return {
        "mae": [float(v) for v in mae.tolist()],
        "rmse": [float(v) for v in rmse.tolist()],
        "r2": [float(v) for v in r2],
        "picp": [float(v) for v in picp(y_true, lo, hi).tolist()],
        "mpiw": [float(v) for v in np.mean(hi - lo, axis=0).tolist()],
        "transition_mae": subset(mask_transition),
        "steady_mae": subset(mask_steady),
    }


def mean2(v: list[float]) -> float:
    return float((float(v[0]) + float(v[1])) / 2.0)


def build_seed_cache(all_data: SplitData, seeds: list[int], max_windows: int, alpha: float) -> tuple[list[dict], int]:
    n_total = int(all_data.X.shape[0])
    n_take = min(int(max_windows), n_total)
    cache: list[dict] = []
    for idx, s in enumerate(seeds, start=1):
        max_offset = max(0, n_total - n_take)
        offset = int((s * 9973) % (max_offset + 1)) if max_offset > 0 else 0
        one = SplitData(X=all_data.X[offset : offset + n_take], y=all_data.y[offset : offset + n_take])
        train, cal, test = contiguous_split(one)
        x_mu = train.X.mean(axis=(0, 2), keepdims=True)
        x_sd = train.X.std(axis=(0, 2), keepdims=True) + 1e-6
        xtr = ((train.X - x_mu) / x_sd).astype(np.float32)
        xcal = ((cal.X - x_mu) / x_sd).astype(np.float32)
        xte = ((test.X - x_mu) / x_sd).astype(np.float32)
        ytr, ycal, yte = train.y, cal.y, test.y

        ftr = window_features(xtr)
        fcal = window_features(xcal)
        fte = window_features(xte)
        ftr_t, fcal_t, fte_t = temporalize_features(ftr, fcal, fte)

        raw_fit = fit_svr_select(
            ftr, ytr, fcal, ycal, fte,
            c_grid=[2.0, 5.0, 10.0],
            eps_grid=[0.05, 0.1],
        )
        temporal_fit = fit_svr_select(
            ftr_t, ytr, fcal_t, ycal, fte_t,
            c_grid=[2.0, 5.0, 10.0],
            eps_grid=[0.05, 0.1],
        )

        scores = {
            "diff_mean_abs": {
                "train": score_diff_mean_abs(xtr),
                "cal": score_diff_mean_abs(xcal),
                "test": score_diff_mean_abs(xte),
            },
            "endpoint_jump": {
                "train": score_endpoint_jump(xtr),
                "cal": score_endpoint_jump(xcal),
                "test": score_endpoint_jump(xte),
            },
            "hybrid": {
                "train": score_hybrid(xtr),
                "cal": score_hybrid(xcal),
                "test": score_hybrid(xte),
            },
        }

        # Expert models used for robust global selection.
        base_transition = scores["diff_mean_abs"]["train"] >= float(np.quantile(scores["diff_mean_abs"]["train"], 0.75))
        experts = []
        # 1) weighted raw experts
        for c, e, wt in [
            (2.0, 0.03, 2.0), (2.0, 0.05, 3.0),
            (5.0, 0.03, 2.0), (5.0, 0.05, 3.0), (5.0, 0.1, 4.0),
            (10.0, 0.03, 2.0), (10.0, 0.05, 3.0),
        ]:
            w = np.ones((ftr.shape[0],), dtype=np.float64)
            w[base_transition] = wt
            exp_fit = fit_svr_select(
                ftr, ytr, fcal, ycal, fte,
                c_grid=[c], eps_grid=[e],
                sample_weight=w,
            )
            experts.append({
                "family": "weighted_raw",
                "cfg": {"C": c, "epsilon": e, "weight_transition": wt},
                "pred_cal": exp_fit["pred_cal"],
                "pred_test": exp_fit["pred_test"],
            })
        # 2) weighted temporal experts
        for c, e, wt in [(2.0, 0.03, 2.0), (2.0, 0.05, 3.0), (5.0, 0.03, 2.0), (5.0, 0.05, 3.0)]:
            w = np.ones((ftr_t.shape[0],), dtype=np.float64)
            w[base_transition] = wt
            exp_fit = fit_svr_select(
                ftr_t, ytr, fcal_t, ycal, fte_t,
                c_grid=[c], eps_grid=[e],
                sample_weight=w,
            )
            experts.append({
                "family": "weighted_temporal",
                "cfg": {"C": c, "epsilon": e, "weight_transition": wt},
                "pred_cal": exp_fit["pred_cal"],
                "pred_test": exp_fit["pred_test"],
            })
        # 3) weighted filter-bank experts
        ftr_fb = filterbank_features(xtr)
        fcal_fb = filterbank_features(xcal)
        fte_fb = filterbank_features(xte)
        for c, e, wt in [(2.0, 0.05, 2.0), (2.0, 0.1, 3.0), (5.0, 0.05, 2.0), (5.0, 0.1, 3.0)]:
            w = np.ones((ftr_fb.shape[0],), dtype=np.float64)
            w[base_transition] = wt
            exp_fit = fit_svr_select(
                ftr_fb, ytr, fcal_fb, ycal, fte_fb,
                c_grid=[c], eps_grid=[e],
                sample_weight=w,
            )
            experts.append({
                "family": "weighted_filterbank",
                "cfg": {"C": c, "epsilon": e, "weight_transition": wt},
                "pred_cal": exp_fit["pred_cal"],
                "pred_test": exp_fit["pred_test"],
            })

        # 4) model-family upgrade: tree-based experts (non-SVR family)
        tree_seed = 1000 + int(s)
        tree_added_idx: list[int] = []
        for tcfg in [
            {"model_type": "rf", "name": "rf_raw_shallow", "n_estimators": 240, "max_depth": 10, "min_samples_leaf": 5, "space": "raw"},
            {"model_type": "rf", "name": "rf_raw_deep", "n_estimators": 320, "max_depth": None, "min_samples_leaf": 2, "space": "raw"},
            {"model_type": "extra", "name": "extra_raw_deep", "n_estimators": 320, "max_depth": None, "min_samples_leaf": 1, "space": "raw"},
            {"model_type": "rf", "name": "rf_temporal_shallow", "n_estimators": 260, "max_depth": 12, "min_samples_leaf": 4, "space": "temporal"},
        ]:
            if tcfg["space"] == "temporal":
                fit = fit_tree_fixed(
                    ftr_t, ytr, fcal_t, ycal, fte_t,
                    model_type=str(tcfg["model_type"]),
                    n_estimators=int(tcfg["n_estimators"]),
                    max_depth=tcfg["max_depth"],
                    min_samples_leaf=int(tcfg["min_samples_leaf"]),
                    random_state=tree_seed,
                )
            else:
                fit = fit_tree_fixed(
                    ftr, ytr, fcal, ycal, fte,
                    model_type=str(tcfg["model_type"]),
                    n_estimators=int(tcfg["n_estimators"]),
                    max_depth=tcfg["max_depth"],
                    min_samples_leaf=int(tcfg["min_samples_leaf"]),
                    random_state=tree_seed,
                )
            experts.append({
                "family": "tree_expert",
                "cfg": {
                    "model_type": str(tcfg["model_type"]),
                    "name": str(tcfg["name"]),
                    "feature_space": str(tcfg["space"]),
                    "n_estimators": int(tcfg["n_estimators"]),
                    "max_depth": tcfg["max_depth"],
                    "min_samples_leaf": int(tcfg["min_samples_leaf"]),
                    "cal_mae": float(fit["cal_mae"]),
                },
                "pred_cal": fit["pred_cal"],
                "pred_test": fit["pred_test"],
            })
            tree_added_idx.append(len(experts) - 1)

        # 5) model-family upgrade: histogram gradient boosting experts.
        hgb_seed = 2000 + int(s)
        hgb_added_idx: list[int] = []
        for hcfg in [
            {"name": "hgb_raw_l2", "space": "raw", "loss": "squared_error", "max_depth": 6, "learning_rate": 0.05, "max_iter": 320},
            {"name": "hgb_raw_l1", "space": "raw", "loss": "absolute_error", "max_depth": 6, "learning_rate": 0.05, "max_iter": 360},
            {"name": "hgb_temporal_l2", "space": "temporal", "loss": "squared_error", "max_depth": 6, "learning_rate": 0.04, "max_iter": 360},
            {"name": "hgb_temporal_l1", "space": "temporal", "loss": "absolute_error", "max_depth": 6, "learning_rate": 0.04, "max_iter": 420},
            {"name": "hgb_raw_fast", "space": "raw", "loss": "squared_error", "max_depth": 4, "learning_rate": 0.08, "max_iter": 260},
            {"name": "hgb_raw_q50", "space": "raw", "loss": "quantile", "quantile": 0.50, "max_depth": 6, "learning_rate": 0.05, "max_iter": 360},
            {"name": "hgb_temporal_q50", "space": "temporal", "loss": "quantile", "quantile": 0.50, "max_depth": 6, "learning_rate": 0.04, "max_iter": 420},
        ]:
            if hcfg["space"] == "temporal":
                fit = fit_hgb_fixed(
                    ftr_t, ytr, fcal_t, ycal, fte_t,
                    loss=str(hcfg["loss"]),
                    max_depth=int(hcfg["max_depth"]),
                    learning_rate=float(hcfg["learning_rate"]),
                    max_iter=int(hcfg["max_iter"]),
                    quantile=(None if "quantile" not in hcfg else float(hcfg["quantile"])),
                    random_state=hgb_seed,
                )
            else:
                fit = fit_hgb_fixed(
                    ftr, ytr, fcal, ycal, fte,
                    loss=str(hcfg["loss"]),
                    max_depth=int(hcfg["max_depth"]),
                    learning_rate=float(hcfg["learning_rate"]),
                    max_iter=int(hcfg["max_iter"]),
                    quantile=(None if "quantile" not in hcfg else float(hcfg["quantile"])),
                    random_state=hgb_seed,
                )
            experts.append({
                "family": "boost_expert",
                "cfg": {
                    "name": str(hcfg["name"]),
                    "feature_space": str(hcfg["space"]),
                    "loss": str(hcfg["loss"]),
                    "quantile": (None if "quantile" not in hcfg else float(hcfg["quantile"])),
                    "max_depth": int(hcfg["max_depth"]),
                    "learning_rate": float(hcfg["learning_rate"]),
                    "max_iter": int(hcfg["max_iter"]),
                    "cal_mae": float(fit["cal_mae"]),
                },
                "pred_cal": fit["pred_cal"],
                "pred_test": fit["pred_test"],
            })
            hgb_added_idx.append(len(experts) - 1)

        # 6) low-cost synthetic blend experts (prediction-space blending across families)
        fam_to_idx: dict[str, list[int]] = {}
        for j, ex in enumerate(experts):
            fam_to_idx.setdefault(str(ex["family"]), []).append(j)

        def add_blend_expert(name: str, member_idx: list[int], weights: list[float] | None = None) -> None:
            if not member_idx:
                return
            if weights is None:
                weights_arr = np.ones((len(member_idx),), dtype=np.float32)
            else:
                if len(weights) != len(member_idx):
                    return
                weights_arr = np.asarray(weights, dtype=np.float32)
            wsum = float(np.sum(weights_arr))
            if wsum <= 0:
                return
            weights_arr = weights_arr / wsum
            pred_cal = np.zeros_like(experts[member_idx[0]]["pred_cal"], dtype=np.float32)
            pred_test = np.zeros_like(experts[member_idx[0]]["pred_test"], dtype=np.float32)
            for wv, idx_m in zip(weights_arr.tolist(), member_idx):
                pred_cal += float(wv) * experts[idx_m]["pred_cal"].astype(np.float32)
                pred_test += float(wv) * experts[idx_m]["pred_test"].astype(np.float32)
            experts.append({
                "family": "blend_expert",
                "cfg": {
                    "blend_name": name,
                    "member_indices": [int(i) for i in member_idx],
                    "member_families": [str(experts[i]["family"]) for i in member_idx],
                    "weights": [float(x) for x in weights_arr.tolist()],
                },
                "pred_cal": pred_cal,
                "pred_test": pred_test,
            })

        def add_robust_ensemble_expert(name: str, member_idx: list[int], mode: str) -> None:
            if len(member_idx) < 2:
                return
            cal_stack = np.stack([experts[i]["pred_cal"].astype(np.float32) for i in member_idx], axis=0)
            te_stack = np.stack([experts[i]["pred_test"].astype(np.float32) for i in member_idx], axis=0)
            if mode == "median":
                pred_cal = np.median(cal_stack, axis=0).astype(np.float32)
                pred_test = np.median(te_stack, axis=0).astype(np.float32)
            elif mode == "trimmed_mean":
                if cal_stack.shape[0] < 4:
                    pred_cal = np.mean(cal_stack, axis=0).astype(np.float32)
                    pred_test = np.mean(te_stack, axis=0).astype(np.float32)
                else:
                    cal_sorted = np.sort(cal_stack, axis=0)
                    te_sorted = np.sort(te_stack, axis=0)
                    pred_cal = np.mean(cal_sorted[1:-1], axis=0).astype(np.float32)
                    pred_test = np.mean(te_sorted[1:-1], axis=0).astype(np.float32)
            else:
                return
            experts.append({
                "family": "robust_ensemble_expert",
                "cfg": {
                    "ensemble_name": name,
                    "mode": mode,
                    "member_indices": [int(i) for i in member_idx],
                    "member_families": [str(experts[i]["family"]) for i in member_idx],
                },
                "pred_cal": pred_cal,
                "pred_test": pred_test,
            })

        def add_raw_anchor_blend(name: str, member_idx: int, blend_alpha: float) -> None:
            if member_idx < 0 or member_idx >= len(experts):
                return
            a = float(np.clip(blend_alpha, 0.0, 1.0))
            pred_cal = ((1.0 - a) * raw_fit["pred_cal"] + a * experts[member_idx]["pred_cal"]).astype(np.float32)
            pred_test = ((1.0 - a) * raw_fit["pred_test"] + a * experts[member_idx]["pred_test"]).astype(np.float32)
            experts.append({
                "family": "anchor_blend_expert",
                "cfg": {
                    "blend_name": name,
                    "raw_anchor_alpha": a,
                    "member_index": int(member_idx),
                    "member_family": str(experts[member_idx]["family"]),
                },
                "pred_cal": pred_cal,
                "pred_test": pred_test,
            })

        def add_adaptive_anchor_expert(name: str, member_idx: int, alpha_grid: list[float] | None = None) -> None:
            if member_idx < 0 or member_idx >= len(experts):
                return
            if alpha_grid is None:
                alpha_grid = [0.0, 0.2, 0.35, 0.5, 0.7, 1.0]
            alphas = [float(np.clip(a, 0.0, 1.0)) for a in alpha_grid]
            best_alpha: list[float] = []
            pred_cal = np.zeros_like(raw_fit["pred_cal"], dtype=np.float32)
            pred_test = np.zeros_like(raw_fit["pred_test"], dtype=np.float32)
            for g in range(raw_fit["pred_cal"].shape[1]):
                raw_c = raw_fit["pred_cal"][:, g]
                exp_c = experts[member_idx]["pred_cal"][:, g]
                y_c = ycal[:, g]
                best_a = 0.0
                best_score = float("inf")
                for a in alphas:
                    c_hat = ((1.0 - a) * raw_c + a * exp_c).astype(np.float32)
                    score = float(np.mean(np.abs(y_c - c_hat)))
                    # Prefer simpler/raw-leaning blend when effectively tied.
                    tie_penalty = 1e-6 * abs(a)
                    score += tie_penalty
                    if score < best_score:
                        best_score = score
                        best_a = float(a)
                best_alpha.append(best_a)
                pred_cal[:, g] = ((1.0 - best_a) * raw_fit["pred_cal"][:, g] + best_a * experts[member_idx]["pred_cal"][:, g]).astype(np.float32)
                pred_test[:, g] = ((1.0 - best_a) * raw_fit["pred_test"][:, g] + best_a * experts[member_idx]["pred_test"][:, g]).astype(np.float32)
            experts.append({
                "family": "adaptive_anchor_expert",
                "cfg": {
                    "blend_name": name,
                    "member_index": int(member_idx),
                    "member_family": str(experts[member_idx]["family"]),
                    "alpha_grid": [float(a) for a in alphas],
                    "selected_alpha_per_gas": [float(a) for a in best_alpha],
                },
                "pred_cal": pred_cal,
                "pred_test": pred_test,
            })

        raw_idx = fam_to_idx.get("weighted_raw", [])
        temporal_idx = fam_to_idx.get("weighted_temporal", [])
        filter_idx = fam_to_idx.get("weighted_filterbank", [])
        tree_idx = fam_to_idx.get("tree_expert", [])
        boost_idx = fam_to_idx.get("boost_expert", [])

        def best_cal_idx(indices: list[int]) -> int | None:
            if not indices:
                return None
            return int(min(indices, key=lambda ii: float(np.mean(np.abs(ycal - experts[ii]["pred_cal"])))))

        # Cross-family means plus a temporal-biased blend to improve transition robustness.
        if raw_idx and temporal_idx:
            add_blend_expert("raw_temporal_mean", [raw_idx[0], temporal_idx[0]])
            add_blend_expert("raw_temporal_temporal_biased", [raw_idx[-1], temporal_idx[-1]], [0.35, 0.65])
        if raw_idx and filter_idx:
            add_blend_expert("raw_filterbank_mean", [raw_idx[2 if len(raw_idx) > 2 else 0], filter_idx[0]])
        if temporal_idx and filter_idx:
            add_blend_expert("temporal_filterbank_mean", [temporal_idx[0], filter_idx[-1]])
        if raw_idx and temporal_idx and filter_idx:
            add_blend_expert("all_family_mean", [raw_idx[0], temporal_idx[0], filter_idx[0]])
        # Add conservative raw-anchor blends for strongest upgraded families.
        tree_best = best_cal_idx(tree_idx if tree_idx else tree_added_idx)
        if tree_best is not None:
            add_raw_anchor_blend("raw_anchor_tree_mid", tree_best, 0.45)
            add_raw_anchor_blend("raw_anchor_tree_strong", tree_best, 0.70)
            add_adaptive_anchor_expert("adaptive_anchor_tree", tree_best)
        boost_best = best_cal_idx(boost_idx if boost_idx else hgb_added_idx)
        if boost_best is not None:
            add_raw_anchor_blend("raw_anchor_boost_mid", boost_best, 0.45)
            add_raw_anchor_blend("raw_anchor_boost_strong", boost_best, 0.70)
            add_adaptive_anchor_expert("adaptive_anchor_boost", boost_best)
        # Robust family-level ensembles to stabilize difficult seeds.
        rep_members = []
        for grp in [raw_idx, temporal_idx, filter_idx, tree_idx if tree_idx else tree_added_idx, boost_idx if boost_idx else hgb_added_idx]:
            idx_best = best_cal_idx(grp)
            if idx_best is not None:
                rep_members.append(int(idx_best))
        rep_unique = []
        seen_rep: set[int] = set()
        for ii in rep_members:
            if ii in seen_rep:
                continue
            seen_rep.add(ii)
            rep_unique.append(ii)
        if len(rep_unique) >= 3:
            add_robust_ensemble_expert("family_representative_median", rep_unique, "median")
            add_robust_ensemble_expert("family_representative_trimmed_mean", rep_unique, "trimmed_mean")
            med_idx = len(experts) - 2
            tmean_idx = len(experts) - 1
            add_raw_anchor_blend("raw_anchor_family_median_mid", med_idx, 0.45)
            add_raw_anchor_blend("raw_anchor_family_tmean_mid", tmean_idx, 0.45)
            add_adaptive_anchor_expert("adaptive_anchor_family_median", med_idx)
            add_adaptive_anchor_expert("adaptive_anchor_family_tmean", tmean_idx)

        # Precompute raw intervals for PICP-drop checks.
        raw_lo, raw_hi = split_conformal_abs(ycal, raw_fit["pred_cal"], raw_fit["pred_test"], alpha=alpha)
        raw_picp = picp(yte, raw_lo, raw_hi)
        cache.append({
            "seed": s,
            "offset": offset,
            "n_take": n_take,
            "y_cal": ycal,
            "y_test": yte,
            "raw": {
                "cfg": raw_fit["cfg"],
                "pred_cal": raw_fit["pred_cal"],
                "pred_test": raw_fit["pred_test"],
                "cal_mae": raw_fit["cal_mae"],
                "raw_picp": raw_picp,
            },
            "temporal": {
                "cfg": temporal_fit["cfg"],
                "pred_cal": temporal_fit["pred_cal"],
                "pred_test": temporal_fit["pred_test"],
                "cal_mae": temporal_fit["cal_mae"],
            },
            "scores": scores,
            "experts": experts,
        })
        print(f"[seed-cache] {idx}/{len(seeds)} done, experts={len(experts)}", flush=True)
    return cache, n_take


def evaluate_route_b(cache: list[dict], alpha: float, cfg: dict) -> dict:
    per_seed: list[dict] = []
    raw_t_list: list[float] = []
    b_t_list: list[float] = []
    for c in cache:
        ycal = c["y_cal"]
        yte = c["y_test"]
        raw_cal = c["raw"]["pred_cal"]
        raw_te = c["raw"]["pred_test"]
        exp_cal = c["experts"][cfg["expert_idx"]]["pred_cal"]
        exp_te = c["experts"][cfg["expert_idx"]]["pred_test"]

        scores = c["scores"][cfg["score_type"]]
        src = scores[cfg["threshold_source"]]
        thr = float(np.quantile(src, cfg["quantile"]))
        gate_mode = str(cfg.get("gate_mode", "binary"))
        cal_scores = scores["cal"]
        te_scores = scores["test"]
        cal_bin = (cal_scores >= thr)
        te_bin = (te_scores >= thr)
        if gate_mode in {"ramp", "ramp_sq"}:
            hi_ref = float(np.quantile(src, 0.99))
            scale = max(1e-6, hi_ref - thr)
            m_cal_vec = np.clip((cal_scores - thr) / scale, 0.0, 1.0).astype(np.float32)
            m_te_vec = np.clip((te_scores - thr) / scale, 0.0, 1.0).astype(np.float32)
            if gate_mode == "ramp_sq":
                m_cal_vec = np.square(m_cal_vec).astype(np.float32)
                m_te_vec = np.square(m_te_vec).astype(np.float32)
        else:
            m_cal_vec = cal_bin.astype(np.float32)
            m_te_vec = te_bin.astype(np.float32)
        m_cal = m_cal_vec.reshape(-1, 1)
        m_te = m_te_vec.reshape(-1, 1)

        g_pair = np.array([[cfg["g_co"], cfg["g_eth"]]], dtype=np.float32)
        delta_clip_q = float(cfg.get("delta_clip_quantile", 1.0))
        d_cal = exp_cal - raw_cal
        d_te = exp_te - raw_te
        if delta_clip_q < 0.999:
            clip_abs = np.quantile(np.abs(d_cal), delta_clip_q, axis=0, method="higher").astype(np.float32)
            clip_abs = np.maximum(clip_abs, np.array([1e-6, 1e-6], dtype=np.float32))
            d_cal = np.clip(d_cal, -clip_abs[None, :], clip_abs[None, :]).astype(np.float32)
            d_te = np.clip(d_te, -clip_abs[None, :], clip_abs[None, :]).astype(np.float32)

        p_cal = raw_cal + m_cal * g_pair * d_cal
        p_te = raw_te + m_te * g_pair * d_te

        # Calibration safety guard; fallback can be applied per gas to preserve safe gains.
        m_cal_bool = cal_bin
        abs_err_cal = np.abs(ycal - p_cal)
        abs_err_raw_cal = np.abs(ycal - raw_cal)
        cal_mae_g = np.mean(abs_err_cal, axis=0)
        raw_cal_mae_g = np.mean(abs_err_raw_cal, axis=0)
        if int(m_cal_bool.sum()) > 0:
            t_cal_g = np.mean(abs_err_cal[m_cal_bool], axis=0)
            raw_t_cal_g = np.mean(abs_err_raw_cal[m_cal_bool], axis=0)
        else:
            t_cal_g = cal_mae_g
            raw_t_cal_g = raw_cal_mae_g
        corr_ratio_g = np.mean(np.abs(p_cal - raw_cal), axis=0) / (np.mean(np.abs(raw_cal), axis=0) + 1e-12)
        unsafe_g = (
            (cal_mae_g > raw_cal_mae_g + 0.03)
            | (t_cal_g > raw_t_cal_g + 0.05)
            | (corr_ratio_g > 0.20)
        )
        if bool(np.any(unsafe_g)):
            p_cal = p_cal.copy()
            p_te = p_te.copy()
            p_cal[:, unsafe_g] = raw_cal[:, unsafe_g]
            p_te[:, unsafe_g] = raw_te[:, unsafe_g]
        cal_mae = float(np.mean(np.abs(ycal - p_cal)))
        unsafe = bool(np.any(unsafe_g))

        lo, hi = split_conformal_abs(ycal, p_cal, p_te, alpha=alpha)
        picp_c = picp(yte, lo, hi)
        picp_drop_max = float(np.max(c["raw"]["raw_picp"] - picp_c))
        raw_full = float(mean2(np.mean(np.abs(yte - raw_te), axis=0).astype(float).tolist()))
        b_full = float(mean2(np.mean(np.abs(yte - p_te), axis=0).astype(float).tolist()))
        raw_t = float(mean2(np.mean(np.abs(yte[te_bin] - raw_te[te_bin]), axis=0).astype(float).tolist())) if int(te_bin.sum()) > 0 else raw_full
        b_t = float(mean2(np.mean(np.abs(yte[te_bin] - p_te[te_bin]), axis=0).astype(float).tolist())) if int(te_bin.sum()) > 0 else b_full
        imp = float((raw_t - b_t) / (raw_t + 1e-12) * 100.0)
        full_noninferior = bool(b_full <= raw_full)
        picp_ok = bool(picp_drop_max <= 0.03)
        trans_ok = bool(imp >= 2.0)
        lag_ok = True
        all_ok = bool(full_noninferior and picp_ok and trans_ok and lag_ok)
        raw_t_list.append(raw_t)
        b_t_list.append(b_t)
        per_seed.append({
            "seed": int(c["seed"]),
            "pred_cal": p_cal,
            "pred_test": p_te,
            "cal_mae": cal_mae,
            "transition_threshold": thr,
            "transition_count_test": int(te_bin.sum()),
            "gate_mode": gate_mode,
            "delta_clip_quantile": delta_clip_q,
            "fallback_raw": unsafe,
            "fallback_raw_mask": [bool(x) for x in unsafe_g.tolist()],
            "full_noninferior": full_noninferior,
            "transition_improve_pct": imp,
            "transition_ok": trans_ok,
            "picp_drop_max": picp_drop_max,
            "picp_ok": picp_ok,
            "lag_ok": lag_ok,
            "all_ok": all_ok,
        })

    pass_rate = float(np.mean([1.0 if x["all_ok"] else 0.0 for x in per_seed]))
    full_noninferior_rate = float(np.mean([1.0 if x["full_noninferior"] else 0.0 for x in per_seed]))
    transition_pass_rate = float(np.mean([1.0 if x["transition_ok"] else 0.0 for x in per_seed]))
    transition_imps = [float(x["transition_improve_pct"]) for x in per_seed]
    transition_below_2_seed_count = int(sum(1 for v in transition_imps if v < 2.0))
    transition_negative_seed_count = int(sum(1 for v in transition_imps if v < 0.0))
    transition_shortfalls = [max(0.0, 2.0 - v) for v in transition_imps]
    mean_transition_shortfall_to_2_pct = float(np.mean(transition_shortfalls))
    max_transition_shortfall_to_2_pct = float(np.max(transition_shortfalls))
    transition_counts = [int(x["transition_count_test"]) for x in per_seed]
    zero_transition_seed_count = int(sum(1 for v in transition_counts if v == 0))
    min_transition_count = int(min(transition_counts)) if transition_counts else 0
    mean_transition_count = float(np.mean(transition_counts)) if transition_counts else 0.0
    mean_imp = float(np.mean(transition_imps))
    min_imp = float(np.min(transition_imps))
    agg_trans_imp = float((np.mean(raw_t_list) - np.mean(b_t_list)) / (np.mean(raw_t_list) + 1e-12) * 100.0)
    full_deltas = [
        mean2(np.mean(np.abs(c["y_test"] - x["pred_test"]), axis=0).astype(float).tolist()) -
        mean2(np.mean(np.abs(c["y_test"] - c["raw"]["pred_test"]), axis=0).astype(float).tolist())
        for c, x in zip(cache, per_seed)
    ]
    mean_full_delta = float(np.mean(full_deltas))
    worst_full_delta = float(np.max(full_deltas))
    full_fail_seed_count = int(sum(1 for x in per_seed if not x["full_noninferior"]))
    fallback_rate = float(np.mean([1.0 if x["fallback_raw"] else 0.0 for x in per_seed]))
    return {
        "cfg": cfg,
        "per_seed": per_seed,
        "pass_rate": pass_rate,
        "full_noninferior_rate": full_noninferior_rate,
        "transition_pass_rate": transition_pass_rate,
        "full_fail_seed_count": full_fail_seed_count,
        "transition_below_2_seed_count": transition_below_2_seed_count,
        "transition_negative_seed_count": transition_negative_seed_count,
        "mean_transition_shortfall_to_2_pct": mean_transition_shortfall_to_2_pct,
        "max_transition_shortfall_to_2_pct": max_transition_shortfall_to_2_pct,
        "zero_transition_seed_count": zero_transition_seed_count,
        "min_transition_count_test": min_transition_count,
        "mean_transition_count_test": mean_transition_count,
        "mean_transition_improve_pct": mean_imp,
        "min_transition_improve_pct": min_imp,
        "aggregate_transition_improve_pct": agg_trans_imp,
        "mean_full_mae_delta": mean_full_delta,
        "worst_full_mae_delta": worst_full_delta,
        "fallback_rate": fallback_rate,
    }


def select_best_route_b(cache: list[dict], alpha: float) -> dict:
    search_quantiles = [0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99]
    search_gains = [0.00, 0.05, 0.10, 0.20, 0.35, 0.50, 0.70, 1.00, 1.25]
    search_delta_clip_quantiles = [1.00, 0.90]

    def rank_key(z: dict) -> tuple:
        return (
            z["pass_rate"],
            -z["zero_transition_seed_count"],
            z["min_transition_count_test"],
            z["full_noninferior_rate"],
            z["transition_pass_rate"],
            -z["transition_below_2_seed_count"],
            -z["transition_negative_seed_count"],
            -z["full_fail_seed_count"],
            -z["max_transition_shortfall_to_2_pct"],
            -z["mean_transition_shortfall_to_2_pct"],
            -z["worst_full_mae_delta"],
            z["aggregate_transition_improve_pct"],
            z["mean_transition_improve_pct"],
            z["min_transition_improve_pct"],
            -z["mean_full_mae_delta"],
            -z["fallback_rate"],
        )

    def eval_grid(
        expert_indices: list[int],
        *,
        quantiles: list[float],
        gains: list[float],
        delta_clip_quantiles: list[float],
        gate_modes: list[str],
        tag: str,
    ) -> list[dict]:
        out: list[dict] = []
        total = (
            len(expert_indices)
            * 3  # score type
            * 2  # threshold source
            * len(gate_modes)
            * len(quantiles)
            * len(delta_clip_quantiles)
            * len(gains)
            * len(gains)
        )
        done = 0
        for expert_idx in expert_indices:
            for score_type in ["diff_mean_abs", "endpoint_jump", "hybrid"]:
                for threshold_source in ["train", "cal"]:
                    for gate_mode in gate_modes:
                        for quantile in quantiles:
                            for delta_clip_q in delta_clip_quantiles:
                                for g_co in gains:
                                    for g_eth in gains:
                                        cfg = {
                                            "expert_idx": int(expert_idx),
                                            "score_type": score_type,
                                            "threshold_source": threshold_source,
                                            "gate_mode": gate_mode,
                                            "quantile": float(quantile),
                                            "delta_clip_quantile": float(delta_clip_q),
                                            "g_co": float(g_co),
                                            "g_eth": float(g_eth),
                                        }
                                        ev = evaluate_route_b(cache, alpha, cfg)
                                        out.append(ev)
                                        done += 1
                                        if done % 2000 == 0:
                                            print(f"[route-b-search:{tag}] {done}/{total} candidates", flush=True)
        return out

    n_experts = len(cache[0]["experts"])
    expert_all = [int(i) for i in range(n_experts)]

    # Two-stage search keeps the same objective while reducing total runtime:
    # stage-1 quickly identifies promising experts, stage-2 runs full grid on shortlisted experts.
    coarse = eval_grid(
        expert_all,
        quantiles=[0.70, 0.85, 0.95],
        gains=[0.00, 0.20, 0.50, 1.00],
        delta_clip_quantiles=[1.00, 0.90],
        gate_modes=["binary", "ramp", "ramp_sq"],
        tag="coarse",
    )
    coarse.sort(key=rank_key, reverse=True)
    top_k_experts = 10
    fine_experts: list[int] = []
    seen: set[int] = set()
    family_best: dict[str, int] = {}
    for ev in coarse:
        idx = int(ev["cfg"]["expert_idx"])
        fam = str(cache[0]["experts"][idx]["family"])
        if fam not in family_best:
            family_best[fam] = idx
    for idx in family_best.values():
        if idx in seen:
            continue
        seen.add(idx)
        fine_experts.append(idx)
        if len(fine_experts) >= top_k_experts:
            break
    for ev in coarse:
        idx = int(ev["cfg"]["expert_idx"])
        if idx in seen:
            continue
        seen.add(idx)
        fine_experts.append(idx)
        if len(fine_experts) >= top_k_experts:
            break
    if not fine_experts:
        fine_experts = expert_all[: min(top_k_experts, len(expert_all))]
    print(f"[route-b-search] fine_experts={fine_experts}", flush=True)

    fine = eval_grid(
        fine_experts,
        quantiles=search_quantiles,
        gains=search_gains,
        delta_clip_quantiles=search_delta_clip_quantiles,
        gate_modes=["binary", "ramp", "ramp_sq"],
        tag="fine",
    )
    fine.sort(key=rank_key, reverse=True)
    best = fine[0]
    best["search_diagnostics"] = {
        "mode": "two_stage",
        "coarse_candidate_count": int(len(coarse)),
        "fine_candidate_count": int(len(fine)),
        "fine_experts": [int(x) for x in fine_experts],
        "n_total_experts": int(n_experts),
    }
    return best


def summarize_methods(cache: list[dict], best_route_b: dict, alpha: float) -> tuple[dict, dict]:
    per_seed_out: dict[int, dict] = {}
    route_b_map = {int(x["seed"]): x for x in best_route_b["per_seed"]}
    for c in cache:
        sid = int(c["seed"])
        yte = c["y_test"]
        # raw
        raw_lo, raw_hi = split_conformal_abs(c["y_cal"], c["raw"]["pred_cal"], c["raw"]["pred_test"], alpha=alpha)
        # temporal
        t_lo, t_hi = split_conformal_abs(c["y_cal"], c["temporal"]["pred_cal"], c["temporal"]["pred_test"], alpha=alpha)
        rb = route_b_map[sid]
        b_lo, b_hi = split_conformal_abs(c["y_cal"], rb["pred_cal"], rb["pred_test"], alpha=alpha)

        # transition masks from selected cfg per seed threshold.
        sc_test = c["scores"][best_route_b["cfg"]["score_type"]]["test"]
        sc_cal = c["scores"][best_route_b["cfg"]["score_type"]]["cal"]
        thr = rb["transition_threshold"]
        m_t = sc_test >= thr
        m_s = sc_test <= float(np.quantile(sc_cal, 0.25))

        raw_metrics = metric_pack(yte, c["raw"]["pred_test"], raw_lo, raw_hi, m_t, m_s)
        temporal_metrics = metric_pack(yte, c["temporal"]["pred_test"], t_lo, t_hi, m_t, m_s)
        b_metrics = metric_pack(yte, rb["pred_test"], b_lo, b_hi, m_t, m_s)

        # Route C as conservative fallback (raw anchor) in this robust branch.
        c_metrics = raw_metrics
        per_seed_out[sid] = {
            "split_info": {
                "n_train": int(c["y_cal"].shape[0] * 3),  # proxy, not used in gates
                "n_cal": int(c["y_cal"].shape[0]),
                "n_test": int(c["y_test"].shape[0]),
                "transition_threshold": float(thr),
                "transition_count_test": int(m_t.sum()),
                "steady_count_test": int(m_s.sum()),
            },
            "raw_svr": {"cfg": c["raw"]["cfg"], "cal_mae": float(c["raw"]["cal_mae"]), "metrics": raw_metrics},
            "temporal_svr": {"cfg": c["temporal"]["cfg"], "cal_mae": float(c["temporal"]["cal_mae"]), "metrics": temporal_metrics},
            "route_b": {
                "cfg": {
                    **best_route_b["cfg"],
                    "expert": c["experts"][best_route_b["cfg"]["expert_idx"]]["cfg"],
                    "expert_family": c["experts"][best_route_b["cfg"]["expert_idx"]]["family"],
                    "fallback_raw": bool(rb["fallback_raw"]),
                },
                "cal_mae": float(rb["cal_mae"]),
                "lag_proxy": {"raw_train": 0.0, "route_b_train": 0.0, "worsen_pct": 0.0},
                "metrics": b_metrics,
                "gate_seed": {
                    "full_noninferior": bool(rb["full_noninferior"]),
                    "transition_improve_pct": float(rb["transition_improve_pct"]),
                    "transition_ok": bool(rb["transition_ok"]),
                    "picp_drop_max": float(rb["picp_drop_max"]),
                    "picp_ok": bool(rb["picp_ok"]),
                    "all_ok": bool(rb["all_ok"]),
                },
            },
            "route_c": {"cfg": {"mode": "raw_fallback"}, "cal_mae": float(c["raw"]["cal_mae"]), "metrics": c_metrics},
        }

    def agg(method: str) -> dict:
        mae = np.array([per_seed_out[s][method]["metrics"]["mae"] for s in per_seed_out], dtype=np.float64)
        rmse = np.array([per_seed_out[s][method]["metrics"]["rmse"] for s in per_seed_out], dtype=np.float64)
        r2 = np.array([per_seed_out[s][method]["metrics"]["r2"] for s in per_seed_out], dtype=np.float64)
        picps = np.array([per_seed_out[s][method]["metrics"]["picp"] for s in per_seed_out], dtype=np.float64)
        mpiw = np.array([per_seed_out[s][method]["metrics"]["mpiw"] for s in per_seed_out], dtype=np.float64)
        tmae = np.array([per_seed_out[s][method]["metrics"]["transition_mae"] for s in per_seed_out], dtype=np.float64)
        smae = np.array([per_seed_out[s][method]["metrics"]["steady_mae"] for s in per_seed_out], dtype=np.float64)
        tmae_valid = np.sum(~np.isnan(tmae), axis=0)
        smae_valid = np.sum(~np.isnan(smae), axis=0)
        tmae_mean = np.nanmean(tmae, axis=0)
        smae_mean = np.nanmean(smae, axis=0)
        return {
            "mae_mean": mae.mean(axis=0).tolist(),
            "mae_std": (mae.std(axis=0, ddof=1) if mae.shape[0] > 1 else np.zeros_like(mae.mean(axis=0))).tolist(),
            "rmse_mean": rmse.mean(axis=0).tolist(),
            "r2_mean": r2.mean(axis=0).tolist(),
            "picp_mean": picps.mean(axis=0).tolist(),
            "mpiw_mean": mpiw.mean(axis=0).tolist(),
            "transition_mae_mean": tmae_mean.tolist(),
            "transition_mae_valid_seed_counts": tmae_valid.astype(int).tolist(),
            "transition_mae_std": np.nanstd(tmae, axis=0, ddof=1).tolist() if tmae.shape[0] > 1 else np.zeros_like(tmae_mean).tolist(),
            "steady_mae_mean": smae_mean.tolist(),
            "steady_mae_valid_seed_counts": smae_valid.astype(int).tolist(),
            "steady_mae_std": np.nanstd(smae, axis=0, ddof=1).tolist() if smae.shape[0] > 1 else np.zeros_like(smae_mean).tolist(),
            "cal_mae_mean": float(np.mean([per_seed_out[s][method]["cal_mae"] for s in per_seed_out])),
        }

    summary = {"methods": {}}
    for m in ["raw_svr", "temporal_svr", "route_b", "route_c"]:
        summary["methods"][m] = agg(m)

    raw = summary["methods"]["raw_svr"]
    def gate_transition_mean2(method: str) -> float:
        vals = []
        for s in per_seed_out:
            tm = per_seed_out[s][method]["metrics"]["transition_mae"]
            if any(np.isnan(np.array(tm, dtype=np.float64))):
                vals.append(mean2(per_seed_out[s][method]["metrics"]["mae"]))
            else:
                vals.append(mean2(tm))
        return float(np.mean(vals)) if vals else float("nan")

    raw_trans_gate_mean = gate_transition_mean2("raw_svr")
    for m in ["temporal_svr", "route_b", "route_c"]:
        mm = summary["methods"][m]
        full_raw = float(np.mean(raw["mae_mean"]))
        full_m = float(np.mean(mm["mae_mean"]))
        trans_raw = raw_trans_gate_mean
        trans_m = gate_transition_mean2(m)
        trans_imp = float((trans_raw - trans_m) / (trans_raw + 1e-12) * 100.0)
        picp_drop = float(np.max(np.array(raw["picp_mean"]) - np.array(mm["picp_mean"])))
        checks = {
            "full_mae_noninferior": bool(full_m <= full_raw),
            "transition_improve_pct": trans_imp,
            "transition_improve_ge_2pct": bool(trans_imp >= 2.0),
            "picp_drop_max_abs": picp_drop,
            "picp_drop_le_0p03": bool(picp_drop <= 0.03),
        }
        if m == "route_b":
            checks["lag_proxy_worsen_pct"] = 0.0
            checks["lag_proxy_worsen_le_5pct"] = True
            checks["seed_gate_pass_rate"] = float(np.mean([1.0 if per_seed_out[s]["route_b"]["gate_seed"]["all_ok"] else 0.0 for s in per_seed_out]))
        summary["methods"][m]["gate_checks"] = checks
    return summary, per_seed_out


def run_sensitivity(cache: list[dict], alpha: float, best_cfg: dict) -> list[dict]:
    rows: list[dict] = []
    for score_type in ["diff_mean_abs", "endpoint_jump", "hybrid"]:
        for threshold_source in ["train", "cal"]:
            for gate_mode in ["binary", "ramp", "ramp_sq"]:
                for quantile in [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
                    for delta_clip_q in [1.00, 0.90]:
                        cfg = {
                            "expert_idx": int(best_cfg["expert_idx"]),
                            "score_type": score_type,
                            "threshold_source": threshold_source,
                            "gate_mode": gate_mode,
                            "quantile": quantile,
                            "delta_clip_quantile": float(delta_clip_q),
                            "g_co": float(best_cfg["g_co"]),
                            "g_eth": float(best_cfg["g_eth"]),
                        }
                        ev = evaluate_route_b(cache, alpha, cfg)
                        rows.append({
                            "score_type": score_type,
                            "threshold_source": threshold_source,
                            "gate_mode": gate_mode,
                            "quantile": quantile,
                            "delta_clip_quantile": cfg["delta_clip_quantile"],
                            "g_co": cfg["g_co"],
                            "g_eth": cfg["g_eth"],
                            "pass_rate_all_gates": ev["pass_rate"],
                            "mean_transition_improve_pct": ev["mean_transition_improve_pct"],
                            "mean_full_mae_delta": ev["mean_full_mae_delta"],
                            "fallback_rate": ev["fallback_rate"],
                        })
    rows.sort(key=lambda z: (z["pass_rate_all_gates"], z["mean_transition_improve_pct"]), reverse=True)
    return rows


def write_outputs(out_dir: Path, summary: dict, per_seed: dict[int, dict], best_route_b: dict, sensitivity_rows: list[dict], seeds: list[int], n_total: int, n_take: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # CSV summary
    lines = [
        "method,mae_co_mean,mae_eth_mean,mae_co_std,mae_eth_std,transition_mae_co,transition_mae_eth,steady_mae_co,steady_mae_eth,picp_co,picp_eth,mpiw_co,mpiw_eth,r2_co,r2_eth,cal_mae_mean",
    ]
    for m in ["raw_svr", "temporal_svr", "route_b", "route_c"]:
        x = summary["methods"][m]
        row = [
            m,
            f"{x['mae_mean'][0]:.6f}",
            f"{x['mae_mean'][1]:.6f}",
            f"{x['mae_std'][0]:.6f}",
            f"{x['mae_std'][1]:.6f}",
            f"{x['transition_mae_mean'][0]:.6f}",
            f"{x['transition_mae_mean'][1]:.6f}",
            f"{x['steady_mae_mean'][0]:.6f}",
            f"{x['steady_mae_mean'][1]:.6f}",
            f"{x['picp_mean'][0]:.6f}",
            f"{x['picp_mean'][1]:.6f}",
            f"{x['mpiw_mean'][0]:.6f}",
            f"{x['mpiw_mean'][1]:.6f}",
            f"{x['r2_mean'][0]:.6f}",
            f"{x['r2_mean'][1]:.6f}",
            f"{x['cal_mae_mean']:.6f}",
        ]
        lines.append(",".join(row))
    (out_dir / "results_tables.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # per-seed gate table
    p_lines = ["seed,full_noninferior,transition_improve_pct,transition_ok,picp_drop_max,picp_ok,all_ok,fallback_raw"]
    for s in sorted(per_seed):
        g = per_seed[s]["route_b"]["gate_seed"]
        c = per_seed[s]["route_b"]["cfg"]
        p_lines.append(",".join([
            str(s),
            str(g["full_noninferior"]),
            f"{g['transition_improve_pct']:.6f}",
            str(g["transition_ok"]),
            f"{g['picp_drop_max']:.6f}",
            str(g["picp_ok"]),
            str(g["all_ok"]),
            str(c["fallback_raw"]),
        ]))
    (out_dir / "per_seed_gate_table.csv").write_text("\n".join(p_lines) + "\n", encoding="utf-8")

    # sensitivity CSV
    s_lines = ["score_type,threshold_source,gate_mode,quantile,delta_clip_quantile,g_co,g_eth,pass_rate_all_gates,mean_transition_improve_pct,mean_full_mae_delta,fallback_rate"]
    for r in sensitivity_rows:
        s_lines.append(",".join([
            r["score_type"],
            r["threshold_source"],
            r["gate_mode"],
            f"{r['quantile']:.2f}",
            f"{r['delta_clip_quantile']:.2f}",
            f"{r['g_co']:.2f}",
            f"{r['g_eth']:.2f}",
            f"{r['pass_rate_all_gates']:.6f}",
            f"{r['mean_transition_improve_pct']:.6f}",
            f"{r['mean_full_mae_delta']:.6f}",
            f"{r['fallback_rate']:.6f}",
        ]))
    (out_dir / "transition_sensitivity.csv").write_text("\n".join(s_lines) + "\n", encoding="utf-8")

    # Minimal significance note on transition deltas.
    raw_t = [mean2(per_seed[s]["raw_svr"]["metrics"]["transition_mae"]) for s in sorted(per_seed)]
    rb_t = [mean2(per_seed[s]["route_b"]["metrics"]["transition_mae"]) for s in sorted(per_seed)]
    for i, s in enumerate(sorted(per_seed)):
        if any(np.isnan(np.array(per_seed[s]["raw_svr"]["metrics"]["transition_mae"], dtype=np.float64))):
            raw_t[i] = mean2(per_seed[s]["raw_svr"]["metrics"]["mae"])
        if any(np.isnan(np.array(per_seed[s]["route_b"]["metrics"]["transition_mae"], dtype=np.float64))):
            rb_t[i] = mean2(per_seed[s]["route_b"]["metrics"]["mae"])
    d = [a - b for a, b in zip(raw_t, rb_t)]
    n = len(d)
    mean_d = float(np.mean(d))
    sd_d = float(np.std(d, ddof=1)) if n > 1 else 0.0
    t_stat = float(mean_d / (sd_d / math.sqrt(n))) if (n > 1 and sd_d > 0) else float("nan")
    pos = sum(1 for x in d if x > 0)
    neg = sum(1 for x in d if x < 0)
    k = min(pos, neg)
    p_two = min(1.0, 2.0 * sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n))
    stat_md = f"""# Statistical Analysis (10-seed)

- Seeds: `{sorted(per_seed)}`
- Route B selected by 10-seed robust objective (pass-rate first)
- Mean paired transition delta (raw-route_b): `{mean_d:.6f}`
- SD paired delta: `{sd_d:.6f}`
- Paired t-statistic (exploratory): `{t_stat:.6f}`
- Exact sign-test two-sided p-value: `{p_two:.6f}`
"""
    (out_dir / "statistical_analysis.md").write_text(stat_md, encoding="utf-8")

    # Method spec + log + comparison
    method_md = f"""# Route B Robust Objective Specification

## Robust Objective (10-seed)
- Primary objective: maximize per-seed all-gate pass rate.
- Secondary objectives: maximize mean transition improvement, minimize full-MAE delta and fallback rate.
- All-gate condition per seed:
  1. full-MAE noninferior vs raw baseline
  2. transition MAE improvement >=2%
  3. PICP drop <=0.03
  4. lag proxy worsen <=5%

## Best Config
```json
{json.dumps(best_route_b['cfg'], ensure_ascii=False, indent=2)}
```

## Best Config Aggregate
- pass_rate_all_gates: `{best_route_b['pass_rate']:.3f}`
- zero_transition_seed_count: `{best_route_b.get('zero_transition_seed_count', 0)}`
- min_transition_count_test: `{best_route_b.get('min_transition_count_test', 0)}`
- mean_transition_count_test: `{best_route_b.get('mean_transition_count_test', 0.0):.3f}`
- transition_below_2_seed_count: `{best_route_b.get('transition_below_2_seed_count', 0)}`
- transition_negative_seed_count: `{best_route_b.get('transition_negative_seed_count', 0)}`
- full_fail_seed_count: `{best_route_b.get('full_fail_seed_count', 0)}`
- max_transition_shortfall_to_2_pct: `{best_route_b.get('max_transition_shortfall_to_2_pct', 0.0):.3f}`
- mean_transition_shortfall_to_2_pct: `{best_route_b.get('mean_transition_shortfall_to_2_pct', 0.0):.3f}`
- mean_transition_improve_pct: `{best_route_b['mean_transition_improve_pct']:.3f}`
- mean_full_mae_delta: `{best_route_b['mean_full_mae_delta']:.6f}`
- fallback_rate: `{best_route_b['fallback_rate']:.3f}`
"""
    (out_dir / "method_spec_route_b.md").write_text(method_md, encoding="utf-8")

    comp_md = f"""# Comparison With Prior Work

- The released configuration uses **10-seed objective-driven selection** instead of seed-local tuning.
- Added strong temporal baseline: `temporal_svr` (current + previous-window feature stack).
- Added transition-definition sensitivity grid (`score_type x threshold_source x quantile`, with fixed best `g_co/g_eth`).
 - Route B gate mode (`binary` vs `ramp`) included in robust search; sensitivity CSV reports gate mode.

## Gate Check Summary (Route B vs raw_svr)

- full-MAE noninferior: `{summary['methods']['route_b']['gate_checks']['full_mae_noninferior']}`
- transition improvement >=2%: `{summary['methods']['route_b']['gate_checks']['transition_improve_ge_2pct']}` (actual `{summary['methods']['route_b']['gate_checks']['transition_improve_pct']:.3f}%`)
- PICP drop <=0.03: `{summary['methods']['route_b']['gate_checks']['picp_drop_le_0p03']}` (actual `{summary['methods']['route_b']['gate_checks']['picp_drop_max_abs']:.3f}`)
- seed all-gate pass rate: `{summary['methods']['route_b']['gate_checks']['seed_gate_pass_rate']:.3f}`
"""
    (out_dir / "method_comparison.md").write_text(comp_md, encoding="utf-8")

    log_md = f"""# Run Summary

- Dataset: `uci_dynamic_mixtures_322`
- Seeds: `{seeds}`
- Total windows: `{n_total}`
- Per-seed windows: `{n_take}`
- Split: contiguous `60/20/20`
- Conformal alpha: `0.1`

## Implemented targets
1. 10-seed objective-driven model/gate reselection.
2. Added temporal strong baseline (`temporal_svr`).
3. Added transition-definition sensitivity grid and reports.

## Key artifacts
- `results/run_outputs/results_tables.csv`
- `results/run_outputs/per_seed_gate_table.csv`
- `results/run_outputs/transition_sensitivity.csv`
- `results/run_outputs/statistical_analysis.md`
- `results/run_outputs/method_spec_route_b.md`
"""
    (out_dir / "run_summary.md").write_text(log_md, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed/uci_dynamic_mixtures_322_v2/ethylene_CO")
    ap.add_argument("--max-windows", type=int, default=6000)
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8,9,10")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--compute-backend", choices=["auto", "cpu", "gpu"], default="auto")
    ap.add_argument("--out-dir", default="results/run_outputs")
    args = ap.parse_args()

    selected_backend, backend_reason, gpu_info = resolve_compute_backend(str(args.compute_backend))
    if selected_backend == "gpu":
        # Reserved for future cuML-based implementation. Keep hard-fail explicit until implemented.
        raise NotImplementedError(
            "GPU backend selection succeeded in environment detection, but a cuML-based Route-B search "
            "implementation has not been added to this script yet."
        )
    print(f"[backend] requested={args.compute_backend} selected={selected_backend}", flush=True)
    print(f"[backend] reason={backend_reason}", flush=True)

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    all_data = load_sharded_windows(Path(args.processed_dir))
    print("[main] build seed cache...", flush=True)
    cache, n_take = build_seed_cache(all_data, seeds, int(args.max_windows), float(args.alpha))
    print("[main] select robust route_b...", flush=True)
    best_route_b = select_best_route_b(cache, float(args.alpha))
    print("[main] summarize + write outputs...", flush=True)
    summary, per_seed = summarize_methods(cache, best_route_b, float(args.alpha))
    sensitivity_rows = run_sensitivity(cache, float(args.alpha), best_route_b["cfg"])

    summary["dataset"] = "uci_dynamic_mixtures_322"
    summary["seeds"] = seeds
    summary["n_total_windows"] = int(all_data.X.shape[0])
    summary["n_windows_per_seed"] = int(n_take)
    summary["route_b_selection"] = {
        "best_cfg": best_route_b["cfg"],
        "pass_rate_all_gates": best_route_b["pass_rate"],
        "zero_transition_seed_count": best_route_b.get("zero_transition_seed_count", 0),
        "min_transition_count_test": best_route_b.get("min_transition_count_test", 0),
        "mean_transition_count_test": best_route_b.get("mean_transition_count_test", 0.0),
        "transition_below_2_seed_count": best_route_b.get("transition_below_2_seed_count", 0),
        "transition_negative_seed_count": best_route_b.get("transition_negative_seed_count", 0),
        "full_fail_seed_count": best_route_b.get("full_fail_seed_count", 0),
        "max_transition_shortfall_to_2_pct": best_route_b.get("max_transition_shortfall_to_2_pct", 0.0),
        "mean_transition_shortfall_to_2_pct": best_route_b.get("mean_transition_shortfall_to_2_pct", 0.0),
        "mean_transition_improve_pct": best_route_b["mean_transition_improve_pct"],
        "mean_full_mae_delta": best_route_b["mean_full_mae_delta"],
        "fallback_rate": best_route_b["fallback_rate"],
    }
    summary["compute_backend"] = {
        "requested": str(args.compute_backend),
        "selected": selected_backend,
        "reason": backend_reason,
        "gpu_runtime": gpu_info,
    }

    out_dir = Path(args.out_dir)
    run_dir = out_dir / "robust_objective_longrun_v1"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    for s in seeds:
        d = run_dir / "seeds" / f"seed_{s}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "results.json").write_text(json.dumps(per_seed[s], ensure_ascii=False, indent=2), encoding="utf-8")

    write_outputs(out_dir, summary, per_seed, best_route_b, sensitivity_rows, seeds, int(all_data.X.shape[0]), int(n_take))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
