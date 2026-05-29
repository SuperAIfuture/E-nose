#!/usr/bin/env python3
"""Parse UCI gas-sensor drift batch files into tabular format.

Supported formats:
- UCI-270 lines: "<class>;<concentration> 1:<v1> ... 128:<v128>"
- UCI-224 lines: "<class> 1:<v1> ... 128:<v128>"
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ParsedRecord:
    class_label: int
    concentration: float | None
    features: np.ndarray


def _parse_feature_tokens(tokens: Sequence[str], n_features: int = 128) -> np.ndarray:
    vec = np.zeros(n_features, dtype=float)
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            continue
        idx_str, value_str = token.split(":", 1)
        idx = int(idx_str)
        if idx < 1 or idx > n_features:
            raise ValueError(f"Feature index {idx} outside 1..{n_features}.")
        vec[idx - 1] = float(value_str)
    return vec


def parse_record(line: str, n_features: int = 128) -> ParsedRecord:
    line = line.strip()
    if not line:
        raise ValueError("Empty line cannot be parsed.")

    tokens = line.split()
    head = tokens[0]
    feature_tokens = tokens[1:]

    if ";" in head:
        cls_str, conc_str = head.split(";", 1)
        class_label = int(cls_str)
        concentration = float(conc_str)
    else:
        class_label = int(head)
        concentration = None

    features = _parse_feature_tokens(feature_tokens, n_features=n_features)
    return ParsedRecord(class_label=class_label, concentration=concentration, features=features)


def load_batch(
    path: Path,
    dataset_name: str,
    batch_id: int,
    n_features: int = 128,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_idx, raw in enumerate(fh, start=1):
            parsed = parse_record(raw, n_features=n_features)
            row: dict[str, object] = {
                "dataset": dataset_name,
                "batch_id": int(batch_id),
                "row_index": int(line_idx),
                "measurement_id": f"{dataset_name}_B{batch_id:02d}_R{line_idx:06d}",
                "class_label": int(parsed.class_label),
                "concentration": parsed.concentration,
                "source_file": str(path.as_posix()),
                "source_line": int(line_idx),
            }
            for i, value in enumerate(parsed.features, start=1):
                row[f"f{i:03d}"] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def discover_batch_files(root: Path) -> list[tuple[int, Path]]:
    pairs: list[tuple[int, Path]] = []
    for path in root.glob("batch*.dat"):
        stem = path.stem.lower()
        suffix = stem.replace("batch", "")
        if suffix.isdigit():
            pairs.append((int(suffix), path))
    return sorted(pairs, key=lambda x: x[0])


def load_dataset(root: Path, dataset_name: str, n_features: int = 128) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for batch_id, path in discover_batch_files(root):
        parts.append(load_batch(path=path, dataset_name=dataset_name, batch_id=batch_id, n_features=n_features))
    if not parts:
        raise FileNotFoundError(f"No batch*.dat files found under {root}")
    return pd.concat(parts, axis=0, ignore_index=True)


def feature_columns(n_features: int = 128) -> list[str]:
    return [f"f{i:03d}" for i in range(1, n_features + 1)]


def batches_to_frames(
    batch_pairs: Iterable[tuple[int, Path]],
    dataset_name: str,
    n_features: int = 128,
) -> dict[int, pd.DataFrame]:
    out: dict[int, pd.DataFrame] = {}
    for batch_id, path in batch_pairs:
        out[batch_id] = load_batch(path=path, dataset_name=dataset_name, batch_id=batch_id, n_features=n_features)
    return out

