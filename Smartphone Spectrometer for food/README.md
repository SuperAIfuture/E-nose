# Smartphone spectrometer article data and code

This package contains the released article data tables and public analysis algorithms used for the manuscript:

An Integrated Clamp-Type Smartphone Spectrometer for Traceable Field Screening of Water, Soil and Produce Safety

## Structure

- `data/stability/`: repeatability spectra and article-level stability result tables.
- `data/dimethoate/`: dimethoate standard and blind spectra, labels, model predictions, classification outputs, route-comparison outputs, and perturbation summaries.
- `data/soil/`: phosphorus and potassium spectra, labels, model comparisons, and locked-model predictions.
- `data/produce/`: produce effective spectra, red-line metrics, and summary table.
- `code/analysis/`: public preprocessing, validation, regression, classification, and sensitivity-analysis algorithms.


## Environment

The analyses were run with Python 3.12.3. Install the listed packages with:

```bash
python -m pip install -r requirements.txt
```

The public algorithms assume the package root as the working directory or equivalent path adjustments for the `data/` folders.
