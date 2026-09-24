# Electronic-Nose Drift Benchmark Data and Analysis

This package contains gas-sensor-array batch data, analysis code, and tabular results for chronology-aware drift-compensation evaluation using the UCI-270 and UCI-224 records.

## Contents

- `data/raw/uci270/` and `data/raw/uci224/`: batch-level input records.
- `code/parse_uci_drift.py`: parser for the batch files.
- `code/feature_routes.py`: preprocessing routes and model builders.
- `code/run_protocol_b_suite.py`: Protocol-B baseline analysis across target batches 2-10.
- `code/run_route_family_ablation.py`: full route-family ablation and class-support sensitivity analysis across target batches 2-10.
- `results/reference/`: reference tables, including per-target metrics, tuning records, route-family ablation, and class-support sensitivity.
- `results/verification/`: outputs produced by the analysis runners.

The route-family reference tables are `route_family_ablation_b2_b10_per_target.csv` (45 target-by-model rows), `route_family_ablation_b2_b10_summary.csv` (five model summaries), and `metric_label_support_sensitivity.csv` (nine target-period class-support records).

## Environment

Use Python 3.10 or later and install the listed dependencies:

```bash
pip install -r requirements.txt
```

The route-family reference tables were generated with Python 3.12.3, NumPy 2.2.6, pandas 2.3.3, scikit-learn 1.7.2, and SciPy 1.15.3. The batch-2 stratified fallback uses random seed 42.

## Protocol-B baseline analysis

Run from this directory:

```bash
python code/run_protocol_b_suite.py \
  --data-root data/raw/uci270 \
  --output-dir results/verification \
  --dataset-name uci270 \
  --random-seed 42
```

The runner writes per-target metrics, tuning records, aggregate summaries, the Table 1 summary, and a comparison with the reference summary when one is supplied.

## Route-family ablation and class-support analysis

Run from this directory:

```bash
python code/run_route_family_ablation.py \
  --data-root data/raw/uci270 \
  --output-dir results/verification \
  --random-seed 42
```

The runner evaluates target batches 2-10. For each target batch, model selection uses the latest prior batch as validation; target batch 2 uses a stratified row-level split of batch 1. Preprocessing is fitted within the training data used for each fit. The outputs are per-target route-family metrics, aggregate summaries, and fixed-label versus present-label metrics with class support.
