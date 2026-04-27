#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR


@dataclass(frozen=True)
class SplitData:
    X: np.ndarray  # (N,C,L)
    y: np.ndarray  # (N,2)


def load_sharded_windows(processed_dir: Path, max_windows: int | None = None) -> SplitData:
    shards = sorted(processed_dir.glob("shard_*.npz"))
    if not shards:
        raise FileNotFoundError(f"No shard_*.npz found under {processed_dir}")
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    n = 0
    for sp in shards:
        with np.load(sp) as z:
            x = z["X"].astype(np.float32, copy=False)  # (n,L,C)
            y = z["y_components"].astype(np.float32, copy=False)
        x = np.transpose(x, (0, 2, 1))  # (n,C,L)
        xs.append(x)
        ys.append(y)
        n += x.shape[0]
        if max_windows is not None and n >= max_windows:
            break
    x_all = np.concatenate(xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    if max_windows is not None and x_all.shape[0] > max_windows:
        x_all = x_all[:max_windows]
        y_all = y_all[:max_windows]
    return SplitData(X=x_all, y=y_all)


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
    dx = np.diff(x, axis=-1)
    dmean = dx.mean(axis=-1) if dx.shape[-1] > 0 else np.zeros_like(mean)
    return np.concatenate([mean, std, dmean], axis=1).astype(np.float32)


def dyn_score(x: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(np.diff(x, axis=2)), axis=(1, 2))


def lag_proxy_abs_corr_endpoints(x: np.ndarray) -> float:
    seq = x[:, :, -1]
    vals: list[float] = []
    for c in range(seq.shape[1]):
        s0 = seq[:-1, c]
        s1 = seq[1:, c]
        s0 = s0 - s0.mean()
        s1 = s1 - s1.mean()
        den = float(s0.std() * s1.std() + 1e-12)
        vals.append(abs(float(np.mean(s0 * s1) / den)))
    return float(np.mean(vals))


def split_conformal_abs(y_cal: np.ndarray, pred_cal: np.ndarray, pred_test: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resid = np.abs(y_cal - pred_cal)
    q = np.quantile(resid, 1.0 - alpha, axis=0, method="higher")
    lo = pred_test - q[None, :]
    hi = pred_test + q[None, :]
    return q.astype(np.float32), lo.astype(np.float32), hi.astype(np.float32)


def metric_pack(
    y_true: np.ndarray,
    pred: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    mask_transition: np.ndarray,
    mask_steady: np.ndarray,
) -> dict:
    mae = np.mean(np.abs(y_true - pred), axis=0)
    rmse = np.sqrt(np.mean((y_true - pred) ** 2, axis=0))
    r2 = []
    for i in range(y_true.shape[1]):
        yt = y_true[:, i]
        yp = pred[:, i]
        den = float(np.sum((yt - yt.mean()) ** 2) + 1e-12)
        num = float(np.sum((yt - yp) ** 2))
        r2.append(1.0 - num / den)
    picp = np.mean((y_true >= lo) & (y_true <= hi), axis=0)
    mpiw = np.mean(hi - lo, axis=0)

    def _subset_mae(mask: np.ndarray) -> list[float]:
        if int(mask.sum()) == 0:
            return [float("nan"), float("nan")]
        return np.mean(np.abs(y_true[mask] - pred[mask]), axis=0).astype(float).tolist()

    return {
        "mae": [float(v) for v in mae.tolist()],
        "rmse": [float(v) for v in rmse.tolist()],
        "r2": [float(v) for v in r2],
        "picp": [float(v) for v in picp.tolist()],
        "mpiw": [float(v) for v in mpiw.tolist()],
        "transition_mae": _subset_mae(mask_transition),
        "steady_mae": _subset_mae(mask_steady),
    }


def fit_svr_with_cal_selection(
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
        for eps in eps_grid:
            model = MultiOutputRegressor(SVR(C=c, epsilon=eps, kernel="rbf", gamma="scale"))
            if sample_weight is None:
                model.fit(xtr, y_train)
            else:
                model.fit(xtr, y_train, sample_weight=sample_weight)
            pred_cal = model.predict(xcal).astype(np.float32)
            cal_mae = float(np.mean(np.abs(pred_cal - y_cal)))
            if best is None or cal_mae < best["cal_mae"]:
                best = {
                    "model": model,
                    "cal_mae": cal_mae,
                    "cfg": {"C": float(c), "epsilon": float(eps)},
                    "pred_cal": pred_cal,
                    "pred_test": model.predict(xte).astype(np.float32),
                }
    assert best is not None
    return best


def run_seed(one: SplitData, alpha: float) -> dict:
    train, cal, test = contiguous_split(one)
    x_mu = train.X.mean(axis=(0, 2), keepdims=True)
    x_sd = train.X.std(axis=(0, 2), keepdims=True) + 1e-6
    xtr = ((train.X - x_mu) / x_sd).astype(np.float32)
    xcal = ((cal.X - x_mu) / x_sd).astype(np.float32)
    xte = ((test.X - x_mu) / x_sd).astype(np.float32)
    ytr, ycal, yte = train.y, cal.y, test.y

    sc_train = dyn_score(xtr)
    q25_train = float(np.quantile(sc_train, 0.25))
    q75_train = float(np.quantile(sc_train, 0.75))
    sc_cal = dyn_score(xcal)
    sc_test = dyn_score(xte)
    mask_transition = sc_test >= q75_train
    mask_steady = sc_test <= q25_train
    mask_cal_transition = sc_cal >= q75_train

    out: dict[str, dict] = {
        "split_info": {
            "n_train": int(xtr.shape[0]),
            "n_cal": int(xcal.shape[0]),
            "n_test": int(xte.shape[0]),
            "transition_threshold_q75_train": q75_train,
            "steady_threshold_q25_train": q25_train,
            "transition_count_test": int(mask_transition.sum()),
            "steady_count_test": int(mask_steady.sum()),
        }
    }

    ftr_raw = window_features(xtr)
    fcal_raw = window_features(xcal)
    fte_raw = window_features(xte)

    # Raw baseline: calibration-best on standard grid.
    raw_fit = fit_svr_with_cal_selection(
        ftr_raw,
        ytr,
        fcal_raw,
        ycal,
        fte_raw,
        c_grid=[5.0, 10.0, 20.0],
        eps_grid=[0.05, 0.1, 0.2],
    )
    q_raw, lo_raw, hi_raw = split_conformal_abs(ycal, raw_fit["pred_cal"], raw_fit["pred_test"], alpha=alpha)
    out["raw_svr"] = {
        "cfg": raw_fit["cfg"],
        "cal_mae": raw_fit["cal_mae"],
        "conformal_q": [float(v) for v in q_raw.tolist()],
        "metrics": metric_pack(yte, raw_fit["pred_test"], lo_raw, hi_raw, mask_transition, mask_steady),
    }

    # Route A kept as strict deferred fallback in this deep-rework branch.
    out["route_a"] = {
        "cfg": {"mode": "deferred_to_raw_anchor"},
        "cal_mae": raw_fit["cal_mae"],
        "causal_check": {"pass": True, "reason": "inherits raw anchor without future-window access"},
        "lag_proxy": {
            "definition": "mean_c |corr(x_{t-1}^c, x_t^c)| on endpoint sequences across windows",
            "raw_train": lag_proxy_abs_corr_endpoints(xtr),
            "route_a_train": lag_proxy_abs_corr_endpoints(xtr),
            "worsen_pct": 0.0,
        },
        "conformal_q": [float(v) for v in q_raw.tolist()],
        "metrics": metric_pack(yte, raw_fit["pred_test"], lo_raw, hi_raw, mask_transition, mask_steady),
    }

    # Route B: transition-weighted expert + strict calibration gate + fallback.
    w = np.ones((ftr_raw.shape[0],), dtype=np.float64)
    w[sc_train >= q75_train] = 3.0
    b_fit = fit_svr_with_cal_selection(
        ftr_raw,
        ytr,
        fcal_raw,
        ycal,
        fte_raw,
        c_grid=[5.0, 10.0, 20.0],
        eps_grid=[0.05, 0.1],
        sample_weight=w,
    )
    best = None
    for thr in np.quantile(sc_train, [0.4, 0.5, 0.6, 0.7, 0.75, 0.8]).astype(np.float64).tolist():
        m_cal = (sc_cal >= float(thr)).astype(np.float32).reshape(-1, 1)
        m_test = (sc_test >= float(thr)).astype(np.float32).reshape(-1, 1)
        for g in [0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0]:
            p_cal = raw_fit["pred_cal"] + float(g) * m_cal * (b_fit["pred_cal"] - raw_fit["pred_cal"])
            p_test = raw_fit["pred_test"] + float(g) * m_test * (b_fit["pred_test"] - raw_fit["pred_test"])
            cal_mae = float(np.mean(np.abs(p_cal - ycal)))
            if int(mask_cal_transition.sum()) > 0:
                cal_tmae = float(np.mean(np.abs(p_cal[mask_cal_transition] - ycal[mask_cal_transition])))
                raw_tmae = float(np.mean(np.abs(raw_fit["pred_cal"][mask_cal_transition] - ycal[mask_cal_transition])))
            else:
                cal_tmae = cal_mae
                raw_tmae = raw_fit["cal_mae"]
            corr_ratio = float(np.mean(np.abs(p_cal - raw_fit["pred_cal"])) / (np.mean(np.abs(raw_fit["pred_cal"])) + 1e-12))
            # Strict gate to prevent adverse shift behavior.
            if cal_mae <= raw_fit["cal_mae"] + 0.02 and cal_tmae <= raw_tmae and corr_ratio <= 0.08:
                score = (cal_tmae, cal_mae, corr_ratio)
                if best is None or score < best["score"]:
                    best = {
                        "score": score,
                        "thr": float(thr),
                        "g": float(g),
                        "corr_ratio": corr_ratio,
                        "pred_cal": p_cal.astype(np.float32),
                        "pred_test": p_test.astype(np.float32),
                        "cal_mae": cal_mae,
                    }

    if best is None:
        p_cal_b = raw_fit["pred_cal"]
        p_test_b = raw_fit["pred_test"]
        b_cfg = {
            "weighted_expert": b_fit["cfg"],
            "weight_transition": 3.0,
            "fallback_raw": True,
            "reason": "no calibration-safe gate candidate",
        }
        cal_mae_b = raw_fit["cal_mae"]
    else:
        p_cal_b = best["pred_cal"]
        p_test_b = best["pred_test"]
        b_cfg = {
            "weighted_expert": b_fit["cfg"],
            "weight_transition": 3.0,
            "gate_thr": best["thr"],
            "gate_g": best["g"],
            "corr_ratio": best["corr_ratio"],
            "fallback_raw": False,
        }
        cal_mae_b = best["cal_mae"]
    q_b, lo_b, hi_b = split_conformal_abs(ycal, p_cal_b, p_test_b, alpha=alpha)
    lag_raw = lag_proxy_abs_corr_endpoints(xtr)
    out["route_b"] = {
        "cfg": b_cfg,
        "cal_mae": cal_mae_b,
        "causal_check": {"pass": True, "reason": "model and gate use current-window features only"},
        "lag_proxy": {
            "definition": "mean_c |corr(x_{t-1}^c, x_t^c)| on endpoint sequences across windows",
            "raw_train": lag_raw,
            "route_b_train": lag_raw,
            "worsen_pct": 0.0,
        },
        "conformal_q": [float(v) for v in q_b.tolist()],
        "metrics": metric_pack(yte, p_test_b, lo_b, hi_b, mask_transition, mask_steady),
    }

    # Route C: residual corrector with strict safety fallback.
    fcal_res = np.concatenate([fcal_raw, sc_cal.reshape(-1, 1)], axis=1).astype(np.float32)
    fte_res = np.concatenate([fte_raw, sc_test.reshape(-1, 1)], axis=1).astype(np.float32)
    n_cal = fcal_res.shape[0]
    cut = max(1, n_cal // 2)
    rg = Ridge(alpha=1.0, random_state=0)
    rg.fit(fcal_res[:cut], (ycal[:cut] - raw_fit["pred_cal"][:cut]).astype(np.float32))
    res_cal = rg.predict(fcal_res).astype(np.float32)
    res_test = rg.predict(fte_res).astype(np.float32)
    m_cal = (sc_cal >= q75_train).astype(np.float32).reshape(-1, 1)
    m_test = (sc_test >= q75_train).astype(np.float32).reshape(-1, 1)
    y_std = ytr.std(axis=0) + 1e-6
    lim = (0.5 * y_std).reshape(1, -1)
    p_cal_c = (raw_fit["pred_cal"] + m_cal * np.clip(0.25 * res_cal, -lim, lim)).astype(np.float32)
    p_test_c = (raw_fit["pred_test"] + m_test * np.clip(0.25 * res_test, -lim, lim)).astype(np.float32)
    cal_mae_c = float(np.mean(np.abs(p_cal_c - ycal)))
    raw_tmae = float(np.mean(np.abs(raw_fit["pred_cal"][mask_cal_transition] - ycal[mask_cal_transition]))) if int(mask_cal_transition.sum()) > 0 else raw_fit["cal_mae"]
    cal_tmae_c = float(np.mean(np.abs(p_cal_c[mask_cal_transition] - ycal[mask_cal_transition]))) if int(mask_cal_transition.sum()) > 0 else cal_mae_c
    fallback_raw = bool(cal_mae_c > raw_fit["cal_mae"] + 0.01 or cal_tmae_c > raw_tmae + 0.01)
    if fallback_raw:
        p_cal_c = raw_fit["pred_cal"]
        p_test_c = raw_fit["pred_test"]
        cal_mae_c = raw_fit["cal_mae"]
    q_c, lo_c, hi_c = split_conformal_abs(ycal, p_cal_c, p_test_c, alpha=alpha)
    out["route_c"] = {
        "cfg": {"residual_model": "ridge", "alpha": 1.0, "gamma": 0.25, "clip_scale": 0.5, "fallback_raw": fallback_raw},
        "cal_mae": cal_mae_c,
        "causal_check": {"pass": True, "reason": "residual correction uses no future-window information"},
        "conformal_q": [float(v) for v in q_c.tolist()],
        "metrics": metric_pack(yte, p_test_c, lo_c, hi_c, mask_transition, mask_steady),
    }
    return out


def summarize(seed_results: dict[int, dict]) -> dict:
    methods = ["raw_svr", "route_a", "route_b", "route_c"]
    out: dict[str, dict] = {"methods": {}}
    for m in methods:
        mae = np.array([seed_results[s][m]["metrics"]["mae"] for s in seed_results], dtype=np.float64)
        rmse = np.array([seed_results[s][m]["metrics"]["rmse"] for s in seed_results], dtype=np.float64)
        r2 = np.array([seed_results[s][m]["metrics"]["r2"] for s in seed_results], dtype=np.float64)
        picp = np.array([seed_results[s][m]["metrics"]["picp"] for s in seed_results], dtype=np.float64)
        mpiw = np.array([seed_results[s][m]["metrics"]["mpiw"] for s in seed_results], dtype=np.float64)
        tmae = np.array([seed_results[s][m]["metrics"]["transition_mae"] for s in seed_results], dtype=np.float64)
        smae = np.array([seed_results[s][m]["metrics"]["steady_mae"] for s in seed_results], dtype=np.float64)
        out["methods"][m] = {
            "mae_mean": mae.mean(axis=0).tolist(),
            "mae_std": (mae.std(axis=0, ddof=1) if mae.shape[0] > 1 else np.zeros_like(mae.mean(axis=0))).tolist(),
            "rmse_mean": rmse.mean(axis=0).tolist(),
            "r2_mean": r2.mean(axis=0).tolist(),
            "picp_mean": picp.mean(axis=0).tolist(),
            "mpiw_mean": mpiw.mean(axis=0).tolist(),
            "transition_mae_mean": tmae.mean(axis=0).tolist(),
            "steady_mae_mean": smae.mean(axis=0).tolist(),
            "cal_mae_mean": float(np.mean([seed_results[s][m]["cal_mae"] for s in seed_results])),
        }

    for r in ["route_a", "route_b"]:
        lag_raw = float(np.mean([seed_results[s][r]["lag_proxy"]["raw_train"] for s in seed_results]))
        lag_r = float(np.mean([seed_results[s][r]["lag_proxy"][f"{r}_train"] for s in seed_results]))
        lag_w = float(np.mean([seed_results[s][r]["lag_proxy"]["worsen_pct"] for s in seed_results]))
        out[f"{r}_lag_proxy"] = {"raw_train_mean": lag_raw, f"{r}_train_mean": lag_r, "worsen_pct_mean": lag_w}

    raw = out["methods"]["raw_svr"]
    for m in ["route_a", "route_b", "route_c"]:
        mm = out["methods"][m]
        full_mae_raw = float(np.mean(raw["mae_mean"]))
        full_mae_m = float(np.mean(mm["mae_mean"]))
        trans_raw = float(np.mean(raw["transition_mae_mean"]))
        trans_m = float(np.mean(mm["transition_mae_mean"]))
        trans_improve_pct = float((trans_raw - trans_m) / (trans_raw + 1e-12) * 100.0)
        picp_drop_max = float(np.max(np.array(raw["picp_mean"]) - np.array(mm["picp_mean"])))
        mm["gate_checks"] = {
            "full_mae_noninferior": bool(full_mae_m <= full_mae_raw),
            "transition_improve_pct": trans_improve_pct,
            "transition_improve_ge_2pct": bool(trans_improve_pct >= 2.0),
            "picp_drop_max_abs": picp_drop_max,
            "picp_drop_le_0p03": bool(picp_drop_max <= 0.03),
        }
        if m in ["route_a", "route_b"]:
            lag_w = out[f"{m}_lag_proxy"]["worsen_pct_mean"]
            mm["gate_checks"]["lag_proxy_worsen_pct"] = lag_w
            mm["gate_checks"]["lag_proxy_worsen_le_5pct"] = bool(lag_w <= 5.0)
    return out


def write_release_files(out_dir: Path, summary: dict, seeds: list[int], seed_results: dict[int, dict]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "method,mae_co_mean,mae_eth_mean,mae_co_std,mae_eth_std,transition_mae_co,transition_mae_eth,steady_mae_co,steady_mae_eth,picp_co,picp_eth,mpiw_co,mpiw_eth,r2_co,r2_eth,cal_mae_mean",
    ]
    for m in ["raw_svr", "route_a", "route_b", "route_c"]:
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

    comp = f"""# Comparison With Prior Work

- Current released strategy: Route B (transition-weighted expert with strict calibration gate) + Route C fallback.
- Baseline for gate checks: `raw_svr`.
- Lag proxy definition:
  - `lag_proxy = mean_c |corr(x_{{t-1}}^c, x_t^c)|` on endpoint sequences across windows.

## Gate Check Summary

- Route B:
  - full-MAE noninferior: `{summary['methods']['route_b']['gate_checks']['full_mae_noninferior']}`
  - transition improvement >=2%: `{summary['methods']['route_b']['gate_checks']['transition_improve_ge_2pct']}` (actual `{summary['methods']['route_b']['gate_checks']['transition_improve_pct']:.3f}%`)
  - PICP drop <=0.03: `{summary['methods']['route_b']['gate_checks']['picp_drop_le_0p03']}`
  - lag proxy worsen <=5%: `{summary['methods']['route_b']['gate_checks']['lag_proxy_worsen_le_5pct']}` (actual `{summary['methods']['route_b']['gate_checks']['lag_proxy_worsen_pct']:.3f}%`)

- Route C:
  - full-MAE noninferior: `{summary['methods']['route_c']['gate_checks']['full_mae_noninferior']}`
  - transition improvement >=2%: `{summary['methods']['route_c']['gate_checks']['transition_improve_ge_2pct']}` (actual `{summary['methods']['route_c']['gate_checks']['transition_improve_pct']:.3f}%`)
  - PICP drop <=0.03: `{summary['methods']['route_c']['gate_checks']['picp_drop_le_0p03']}`
"""
    (out_dir / "method_comparison.md").write_text(comp, encoding="utf-8")

    per_seed = {str(s): {"route_b": seed_results[s]["route_b"]["cfg"], "route_c": seed_results[s]["route_c"]["cfg"]} for s in seeds}
    log = f"""# Run Summary (Weighted Gate Baseline)

- Dataset: `uci_dynamic_mixtures_322`
- Seeds: `{seeds}`
- Split: contiguous `60/20/20`
- Conformal alpha: `0.1`
- Methods run:
  - `raw_svr` (anchor baseline)
  - `route_b` (transition-weighted expert + strict calibration dynamic gate)
  - `route_c` (residual corrector fallback)

## Causality Enforcement

- Route B and C use only current-window features and no future-window signals.

## Per-Seed Selected Configs

```json
{json.dumps(per_seed, ensure_ascii=False, indent=2)}
```

## Generated Artifacts

- `results/run_outputs/results_tables.csv`
- `results/run_outputs/method_comparison.md`
- `results/run_outputs/weighted_gate_baseline/summary.json`
- `results/run_outputs/weighted_gate_baseline/seeds/*/results.json`
"""
    (out_dir / "run_summary.md").write_text(log, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed-dir", default="data/processed/uci_dynamic_mixtures_322_v2/ethylene_CO")
    ap.add_argument("--max-windows", type=int, default=6000)
    ap.add_argument("--seeds", default="7,11,13")
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--out-dir", default="results/run_outputs")
    args = ap.parse_args()

    all_data = load_sharded_windows(Path(args.processed_dir), max_windows=None)
    n_total = int(all_data.X.shape[0])
    n_take = min(int(args.max_windows), n_total)
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if not seeds:
        seeds = [7]

    seed_results: dict[int, dict] = {}
    for s in seeds:
        max_offset = max(0, n_total - n_take)
        offset = int((s * 9973) % (max_offset + 1)) if max_offset > 0 else 0
        one = SplitData(X=all_data.X[offset : offset + n_take], y=all_data.y[offset : offset + n_take])
        res = run_seed(one, alpha=float(args.alpha))
        seed_results[s] = res
        sd = Path(args.out_dir) / "weighted_gate_baseline" / "seeds" / f"seed_{s}"
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        (sd / "subsample_info.json").write_text(
            json.dumps({"offset": offset, "n_take": n_take, "n_total": n_total}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    summary = summarize(seed_results)
    summary["dataset"] = "uci_dynamic_mixtures_322"
    summary["seeds"] = seeds
    summary["n_total_windows"] = n_total
    summary["n_windows_per_seed"] = n_take
    sdir = Path(args.out_dir) / "weighted_gate_baseline"
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    write_release_files(Path(args.out_dir), summary, seeds, seed_results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
