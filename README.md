# SessionAdapt — Reproducible Session-Based Recommendation Experiments

This repository contains the code, experiment configurations, evaluation protocol, and reported results for SessionAdapt, a session-based next-item recommendation project.

## Setup

The instructions below work on Windows, Linux, macOS, and Google Colab.

```bash
python -m venv .venv
```

Activate the environment:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# Linux / macOS
source .venv/bin/activate
```

Install dependencies and the package:

```bash
pip install -r requirements.txt
pip install -e .
python -m pytest -q
```

In VS Code, select the `.venv` interpreter using **Python: Select Interpreter**.

## Evaluation protocol

All reported experiments follow these principles:

- Chronological splitting by session start-time quantiles; sessions crossing a split boundary are dropped.
- The item vocabulary is built from training data only. Out-of-vocabulary events in validation and test are removed and counted.
- Every position \(t \geq 1\) in each evaluation session is treated as a prediction instance.
- Ranking is performed over the full item catalogue.
- Ties are ranked pessimistically.
- Metrics: HR@20, MRR@20, and NDCG@20.
- 95% confidence intervals are computed using session-level bootstrap.
- Hyperparameters are selected using validation data only.
- Test data are used for final evaluation, not hyperparameter selection.

## Repository structure

```text
configs/       Dataset and experiment configurations
scripts/       Data preparation, training, evaluation, and analysis
src/sessrec/   Core Python package
tests/         Automated tests
data/          Raw and processed datasets (not included)
results/       Experiment outputs, tables, and figures
docs/          Project documentation
```

## Datasets

The experiments cover:

- Diginetica
- Yoochoose 1/64
- Yoochoose 1/8

Raw datasets are not included in this repository. Download or obtain the datasets according to their respective terms of use, and place them in the expected `data/raw/` locations before running the preparation scripts.

For example, the Yoochoose pipeline is:

```bash
python scripts/download_data.py yoochoose
python scripts/prepare_data.py configs/yoochoose_f64.yaml
```

For Diginetica, place `train-item-views.csv` and `product-categories.csv` in `data/raw/diginetica/`, then run:

```bash
python scripts/prepare_data.py configs/diginetica.yaml
```

## Experiments and reproduction

### Yoochoose 1/64

Run the baseline and analysis experiments:

```bash
python scripts/run_baselines.py configs/yoochoose_f64.yaml --final
python scripts/run_adaptive.py configs/yoochoose_f64.yaml --final
python scripts/run_slots.py configs/yoochoose_f64.yaml --final
python scripts/run_curve.py configs/yoochoose_f64.yaml --final
python scripts/run_backoff.py configs/yoochoose_f64.yaml --final
python scripts/run_time.py configs/yoochoose_f64.yaml --final
```

Train the neural models using the full experimental protocol (8 trials, 3 seeds, up to 30 epochs per trial):

```bash
python scripts/train_neural.py configs/yoochoose_f64.yaml --model gru --trials 8 --epochs 30 --seeds 3 --final
python scripts/train_neural.py configs/yoochoose_f64.yaml --model sasrec --trials 8 --epochs 30 --seeds 3 --final
python scripts/compare_models.py configs/yoochoose_f64.yaml
```

### Yoochoose 1/8

The released results include baseline and time-based experiments:

```bash
python scripts/prepare_data.py configs/yoochoose_f8.yaml
python scripts/run_baselines.py configs/yoochoose_f8.yaml --final
python scripts/run_time.py configs/yoochoose_f8.yaml --final
```

Neural-model results for Yoochoose 1/8 are not included in this release.

### Diginetica

Run the baseline, adaptive, curve, time, and backoff experiments:

```bash
python scripts/run_baselines.py configs/diginetica.yaml --models pop,spop,markov --final
python scripts/diagnose_shift.py configs/diginetica.yaml
python scripts/run_adaptive.py configs/diginetica.yaml --final
python scripts/run_curve.py configs/diginetica.yaml --final
python scripts/run_time.py configs/diginetica.yaml --scheme dow7 --final
python scripts/run_backoff.py configs/diginetica.yaml --final
```

Train the neural models using the full experimental protocol:

```bash
python scripts/train_neural.py configs/diginetica.yaml --model gru --trials 8 --epochs 30 --seeds 3 --final
python scripts/train_neural.py configs/diginetica.yaml --model sasrec --trials 8 --epochs 30 --seeds 3 --final
python scripts/compare_models.py configs/diginetica.yaml
```

**Experiment scope:** SKNN was not run on Diginetica. No SKNN result is reported for that dataset.

## Neural-model smoke test

A separate small CPU experiment was run on a fixed Diginetica sample to check the neural training pipeline:

- Training: 5,000 sessions (23,971 events)
- Validation: 1,000 sessions (4,377 events)
- Test: 1,000 sessions (4,692 events)
- Models: GRU and SASRec
- Training: one trial, one epoch, one seed
- Device: CPU

The sample experiment can be run with:

```bash
python scripts/train_neural.py configs/diginetica_sample.yaml --model gru --trials 1 --epochs 1 --patience 1 --seeds 1 --device cpu --final
python scripts/train_neural.py configs/diginetica_sample.yaml --model sasrec --trials 1 --epochs 1 --patience 1 --seeds 1 --device cpu --final
python scripts/compare_models.py configs/diginetica_sample.yaml
```

These are **smoke-test results only**, not the full-dataset results reported in the paper. The models were trained for only one epoch with one seed, and the results should not be used to compare final model performance.

The sample comparison is stored in:

```text
results/diginetica_sample_neural_vs_markov_test.csv
```

The full-dataset neural results are stored separately in the corresponding `diginetica_gru_*` and `diginetica_sasrec_*` files.

## Results, tables, and figures

The `results/` directory contains the released CSV and JSON experiment outputs, the consolidated `tables.md` file, and generated figures in `results/figures/`.

To regenerate the consolidated tables:

```bash
python scripts/make_tables.py yoochoose_f64 yoochoose_f8 diginetica
```

Prediction arrays and raw datasets are not included in the repository.

## Tests

Run the automated test suite from the project root:

```bash
python -m pytest -q
```
