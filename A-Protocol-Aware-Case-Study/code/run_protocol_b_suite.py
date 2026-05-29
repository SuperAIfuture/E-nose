#!/usr/bin/env python3
"""Run Protocol-B consistent-policy baseline suite on UCI drift data.

This script reproduces submission-facing experiment tables from raw batch files.
It focuses on experiment execution and excludes any figure-generation workflow.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from feature_routes import MODEL_ORDER, build_model
from parse_uci_drift import batches_to_frames, discover_batch_files, feature_columns


CLASS_LABELS = [1, 2, 3, 4, 5, 6]
GRID_C = [1.0, 10.0, 50.0]
GRID_GAMMA = [0.001, 1.0 / 256.0, 1.0 / 128.0, 0.01]


def _fixed_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(f1_score(y_true, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0))


def _present_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    labels = sorted(np.unique(y_true).tolist())
    return float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))


def _best_hyperparams_key(score: float, c: float, gamma: float) -> tuple[float, float, float]:
    # maximize score, then pick smaller C/gamma for deterministic tie-break.
    return (score, -c, -gamma)


def _concat_batches(frames: dict[int, pd.DataFrame], batch_ids: Iterable[int]) -> pd.DataFrame:
    parts = [frames[b] for b in batch_ids]
    return pd.concat(parts, axis=0, ignore_index=True)


def run_protocol_b_suite(
    data_root: Path,
    output_dir: Path,
    dataset_name: str = "uci270",
    random_seed: int = 42,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_pairs = discover_batch_files(data_root)
    if not batch_pairs:
        raise FileNotFoundError(f"No batch*.dat files found in {data_root}")
    frames = batches_to_frames(batch_pairs=batch_pairs, dataset_name=dataset_name, n_features=128)
    all_batch_ids = sorted(frames.keys())
    if max(all_batch_ids) < 2:
        raise ValueError("Need at least two batches for Protocol-B evaluation.")

    feat_cols = feature_columns(128)

    per_target_rows: list[dict[str, object]] = []
    tuning_rows: list[dict[str, object]] = []

    for target_batch in [b for b in all_batch_ids if b >= 2]:
        train_batch_ids = [b for b in all_batch_ids if b < target_batch]
        full_train = _concat_batches(frames, train_batch_ids)
        test_df = frames[target_batch]

        if target_batch == 2:
            x_base = full_train[feat_cols].to_numpy(dtype=float)
            y_base = full_train["class_label"].to_numpy(dtype=int)
            x_inner_train, x_val, y_inner_train, y_val = train_test_split(
                x_base,
                y_base,
                test_size=0.2,
                random_state=random_seed,
                stratify=y_base,
            )
            tuning_policy = "row_level_stratified_fallback_single_train_batch"
            inner_train_batches = str([1])
            inner_val_batches = str([1])
            n_val = int(len(y_val))
        else:
            val_batch_id = target_batch - 1
            inner_train_ids = [b for b in all_batch_ids if b < val_batch_id]
            inner_train_df = _concat_batches(frames, inner_train_ids)
            val_df = frames[val_batch_id]

            x_inner_train = inner_train_df[feat_cols].to_numpy(dtype=float)
            y_inner_train = inner_train_df["class_label"].to_numpy(dtype=int)
            x_val = val_df[feat_cols].to_numpy(dtype=float)
            y_val = val_df["class_label"].to_numpy(dtype=int)

            tuning_policy = "holdout_latest_train_batch"
            inner_train_batches = str(inner_train_ids)
            inner_val_batches = str([val_batch_id])
            n_val = int(len(y_val))

        x_full_train = full_train[feat_cols].to_numpy(dtype=float)
        y_full_train = full_train["class_label"].to_numpy(dtype=int)
        x_test = test_df[feat_cols].to_numpy(dtype=float)
        y_test = test_df["class_label"].to_numpy(dtype=int)

        for model_name in MODEL_ORDER:
            best_key: tuple[float, float, float] | None = None
            best_params: tuple[float, float] | None = None

            for c in GRID_C:
                for gamma in GRID_GAMMA:
                    model = build_model(model_name)
                    model.set_params(svc__C=float(c), svc__gamma=float(gamma))
                    model.fit(x_inner_train, y_inner_train)
                    y_val_pred = model.predict(x_val)
                    val_fixed = _fixed_macro_f1(y_val, y_val_pred)

                    tuning_rows.append(
                        {
                            "protocol": "B_rolling_update",
                            "target_batch": int(target_batch),
                            "tuning_policy": tuning_policy,
                            "train_batches": str(train_batch_ids),
                            "inner_train_batches": inner_train_batches,
                            "inner_val_batches": inner_val_batches,
                            "model_name": model_name,
                            "C": float(c),
                            "gamma": float(gamma),
                            "val_macro_f1_fixed6": float(val_fixed),
                        }
                    )

                    key = _best_hyperparams_key(val_fixed, c, gamma)
                    if best_key is None or key > best_key:
                        best_key = key
                        best_params = (float(c), float(gamma))

            assert best_params is not None
            best_c, best_gamma = best_params

            model = build_model(model_name)
            model.set_params(svc__C=best_c, svc__gamma=best_gamma)
            model.fit(x_full_train, y_full_train)
            y_pred = model.predict(x_test)

            row = {
                "dataset": dataset_name,
                "protocol": "B_rolling_update",
                "target_batch": int(target_batch),
                "model_name": model_name,
                "n_train": int(len(y_full_train)),
                "n_val": int(n_val),
                "n_test": int(len(y_test)),
                "macro_f1_fixed6": _fixed_macro_f1(y_test, y_pred),
                "macro_f1_present": _present_macro_f1(y_test, y_pred),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
                "selected_C": best_c,
                "selected_gamma": best_gamma,
            }
            per_target_rows.append(row)

    per_target_df = pd.DataFrame(per_target_rows)
    tuning_df = pd.DataFrame(tuning_rows)

    # Summary table aligned with submission Table 1 fields.
    summary_rows: list[dict[str, object]] = []
    baseline = per_target_df[per_target_df["model_name"] == "ss_svm_rbf"].set_index("target_batch")

    for model_name in MODEL_ORDER:
        sub = per_target_df[per_target_df["model_name"] == model_name].sort_values("target_batch")
        if sub.empty:
            continue
        deltas = (
            sub.set_index("target_batch")["macro_f1_fixed6"] - baseline["macro_f1_fixed6"]
        ).dropna()
        wins = int((deltas > 0).sum())
        losses = int((deltas < 0).sum())
        ties = int((deltas == 0).sum())
        summary_rows.append(
            {
                "model_name": model_name,
                "n_targets": int(len(sub)),
                "mean_macro_f1_fixed": float(sub["macro_f1_fixed6"].mean()),
                "range_fixed": f"{sub['macro_f1_fixed6'].min():.3f}-{sub['macro_f1_fixed6'].max():.3f}",
                "mean_macro_f1_present": float(sub["macro_f1_present"].mean()),
                "mean_accuracy": float(sub["accuracy"].mean()),
                "mean_balanced_accuracy": float(sub["balanced_accuracy"].mean()),
                "delta_vs_ss_mean": float(deltas.mean()),
                "wins": wins,
                "losses": losses,
                "ties": ties,
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    out_per_target = output_dir / "protocol_b_baseline_per_target_verification.csv"
    out_tuning = output_dir / "protocol_b_baseline_tuning_log_verification.csv"
    out_summary = output_dir / "protocol_b_baseline_summary_verification.csv"
    out_table1 = output_dir / "table1_submission_ready_verification.csv"

    per_target_df.to_csv(out_per_target, index=False, encoding="utf-8")
    tuning_df.to_csv(out_tuning, index=False, encoding="utf-8")
    summary_df.to_csv(out_summary, index=False, encoding="utf-8")

    # Friendly projection for direct manuscript table updates.
    table1_df = summary_df.copy()
    table1_df["wins_losses_vs_ss"] = table1_df.apply(
        lambda r: "baseline"
        if r["model_name"] == "ss_svm_rbf"
        else ("0/0 (9 ties)" if r["wins"] == 0 and r["losses"] == 0 and r["ties"] == 9 else f"{int(r['wins'])}/{int(r['losses'])}"),
        axis=1,
    )
    table1_df.to_csv(out_table1, index=False, encoding="utf-8")

    return {
        "per_target": out_per_target,
        "tuning_log": out_tuning,
        "summary": out_summary,
        "table1": out_table1,
    }


def write_reference_comparison(
    verification_summary_path: Path,
    reference_summary_path: Path,
    output_path: Path,
) -> Path:
    verification = pd.read_csv(verification_summary_path)
    reference = pd.read_csv(reference_summary_path)
    keep_cols = [
        "model_name",
        "mean_macro_f1_fixed",
        "mean_macro_f1_present",
        "mean_accuracy",
        "mean_balanced_accuracy",
        "delta_vs_ss_mean",
    ]
    ref = reference.copy()
    rename_map = {
        "mean_macro_f1_fixed": "ref_mean_macro_f1_fixed",
        "mean_macro_f1_present": "ref_mean_macro_f1_present",
        "mean_accuracy": "ref_mean_accuracy",
        "mean_balanced_accuracy": "ref_mean_balanced_accuracy",
        "delta_vs_ss_mean": "ref_delta_vs_ss_mean",
    }
    for col in list(rename_map):
        if col not in ref.columns:
            if col == "mean_macro_f1_fixed":
                if "mean_macro_f1_fixed6" in ref.columns:
                    ref[col] = ref["mean_macro_f1_fixed6"]
                elif "mean_macro_f1" in ref.columns:
                    ref[col] = ref["mean_macro_f1"]
            elif col == "delta_vs_ss_mean":
                if "delta_vs_ss_mean" in ref.columns:
                    ref[col] = ref["delta_vs_ss_mean"]
                elif "delta_vs_ss" in ref.columns:
                    ref[col] = ref["delta_vs_ss"]
    ref = ref[["model_name"] + [c for c in keep_cols if c != "model_name"]].rename(columns=rename_map)
    merged = verification.merge(ref, on="model_name", how="left")

    for col in [
        "mean_macro_f1_fixed",
        "mean_macro_f1_present",
        "mean_accuracy",
        "mean_balanced_accuracy",
        "delta_vs_ss_mean",
    ]:
        merged[f"abs_diff_{col}"] = (merged[col] - merged[f"ref_{col}"]).abs()

    merged.to_csv(output_path, index=False, encoding="utf-8")
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Protocol-B baseline suite from raw UCI batches.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "raw" / "uci270",
        help="Directory containing batch*.dat files for UCI-270.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "verification",
        help="Output directory for verification CSV files.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="uci270",
        help="Dataset name tag written to output tables.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used in target-batch-2 fallback split.",
    )
    parser.add_argument(
        "--reference-summary",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results" / "reference" / "protocol_b_baseline_summary.csv",
        help="Reference summary CSV to compare against (optional).",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    outputs = run_protocol_b_suite(
        data_root=args.data_root,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        random_seed=args.random_seed,
    )

    if args.reference_summary.exists():
        cmp_out = args.output_dir / "summary_vs_reference.csv"
        write_reference_comparison(
            verification_summary_path=outputs["summary"],
            reference_summary_path=args.reference_summary,
            output_path=cmp_out,
        )
        print(f"[ok] Reference comparison written: {cmp_out}")
    else:
        print(f"[warn] Reference summary not found, skip comparison: {args.reference_summary}")

    for key, path in outputs.items():
        print(f"[ok] {key}: {path}")


if __name__ == "__main__":
    main()
