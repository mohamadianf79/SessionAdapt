# sessrec — session-based recommendation: protocol + baselines

## Setup (VS Code, local or Colab)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest -q
```

VS Code: open the folder, press `Ctrl+Shift+P`, select "Python: Select Interpreter", then choose `.venv`.

## Pipeline

```bash
python scripts/download_data.py yoochoose
python scripts/prepare_data.py configs/yoochoose_f64.yaml
python scripts/run_baselines.py configs/yoochoose_f64.yaml
python scripts/run_baselines.py configs/yoochoose_f64.yaml --final
```

## Evaluation protocol

- Chronological split by session start-time quantiles; sessions crossing a boundary are dropped.
- Item vocabulary is built from TRAIN only; OOV events in validation/test are removed and counted.
- Every position t >= 1 in every evaluation session is an instance.
- Ranking is over the full catalogue.
- Ties are ranked pessimistically.
- Metrics: HR@20, MRR@20, NDCG@20.
- 95% confidence intervals use session-level bootstrap.
- Hyperparameters are selected on validation only.
- TEST is evaluated only for final runs.

## Project layout

```text
configs/       Dataset and experiment configurations
scripts/       Data preparation, training, evaluation and analysis
src/sessrec/   Core package
tests/         Automated tests
data/          Raw and processed datasets
results/       Experiment outputs and tables
docs/          Project documentation
```

## Yoochoose experiments

The project includes the following experiments:

- Popularity, session popularity and Markov baselines.
- Adaptive recommendation experiments.
- Cold-start and adaptation curves.
- Time-based controls.
- Category-prior backoff experiments.
- GRU and SASRec neural-model experiments.

Useful commands:

```bash
python scripts/run_adaptive.py configs/yoochoose_f64.yaml --final
python scripts/run_slots.py configs/yoochoose_f64.yaml --final
python scripts/run_curve.py configs/yoochoose_f64.yaml --final
python scripts/run_time.py configs/yoochoose_f8.yaml --final
python scripts/run_backoff.py configs/yoochoose_f64.yaml --final
python scripts/make_tables.py yoochoose_f64 yoochoose_f8 diginetica
```

## Diginetica replication

Place `train-item-views.csv` and `product-categories.csv` in `data/raw/diginetica/`.

```bash
python scripts/prepare_data.py configs/diginetica.yaml
python scripts/run_baselines.py configs/diginetica.yaml --models pop,spop,markov --final
python scripts/diagnose_shift.py configs/diginetica.yaml
python scripts/run_adaptive.py configs/diginetica.yaml --final
python scripts/run_curve.py configs/diginetica.yaml --final
python scripts/run_time.py configs/diginetica.yaml --scheme dow7 --final
python scripts/run_backoff.py configs/diginetica.yaml --final
```

## Neural models: Diginetica sample experiment

A small CPU proof-of-concept experiment was run on a fixed random sample:

- Train: 5,000 sessions (23,971 events)
- Validation: 1,000 sessions (4,377 events)
- Test: 1,000 sessions (4,692 events)
- Models: GRU and SASRec
- Training: one trial, one epoch, one seed
- Device: CPU

Run commands:

```bash
python scripts/train_neural.py configs/diginetica_sample.yaml --model gru --trials 1 --epochs 1 --patience 1 --seeds 1 --device cpu --final
python scripts/train_neural.py configs/diginetica_sample.yaml --model sasrec --trials 1 --epochs 1 --patience 1 --seeds 1 --device cpu --final
python scripts/compare_models.py configs/diginetica_sample.yaml
```

### Sample test results

| Metric | Markov | GRU | SASRec |
|---|---:|---:|---:|
| HR@20 | 0.0737 | 0.0003 | 0.0948 |
| MRR@20 | 0.0383 | 0.0000 | 0.0941 |
| NDCG@20 | 0.0466 | 0.0001 | 0.0943 |

Markov parameters were copied from full-dataset tuning and were not retuned on this sample.

These are preliminary sample results, not full-dataset or fully tuned neural results. Both neural models were trained for only one epoch with one seed; conclusions should not be generalized beyond this experiment.

Comparison output: `results/diginetica_sample_neural_vs_markov_test.csv`.

## Testing

Run the automated test suite from the project root:

```bash
python -m pytest -q
```
