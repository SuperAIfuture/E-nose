# Dynamic Gas Mixture Quantification Release

GitHub-ready repository for the released code, processed data, configuration files, deterministic seed logic, and result tables used in the public release.

## Repository Layout
- `scripts/`: released experiment scripts.
- `configs/`: operating-point configuration, seed logic, and repository manifest.
- `data/processed/`: processed shards required to rerun the released scripts.
- `data/raw/`: metadata for the original raw files and size-limited source notes.
- `results/main/`: main 10-seed outputs.
- `results/extensions/`: 30-seed fixed-configuration robustness results, ungated ablation results, and additional temporal baseline results.
- `docs/`: repository notes for data layout and results layout.

## Quick Start
1. Create a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Recompute the main 10-seed release:

```bash
python scripts/transition_gated_release.py --processed-dir data/processed/uci_dynamic_mixtures_322_v2/ethylene_CO --seeds 1,2,3,4,5,6,7,8,9,10 --max-windows 9000 --out-dir results/recomputed_main
```

## Data Notes
- The processed shards needed for the released runs are included in this repository.
- The original raw text files and original zip archive are not committed here because each file exceeds standard GitHub size limits.
- Raw-file names, sizes, and SHA-256 hashes are listed in `data/raw/raw_data_manifest.json`.

## Main Release Files
- `results/main/summary.json`
- `results/main/results_tables.csv`
- `results/main/per_seed_gate_table.csv`
- `results/main/per_seed_outputs/`
- `results/extensions/fixed_config_30seed/`
- `results/extensions/ungated_ablation_10seed/`
- `results/extensions/temporal_baselines_10seed/`
