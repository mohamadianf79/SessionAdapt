# sessrec — session-based recommendation: protocol + baselines (step 0)

## Setup (VS Code, local or Colab)
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .                   # makes `import sessrec` work everywhere
pytest -q                          # must be green before you trust any number
```
VS Code: open the folder, `Ctrl+Shift+P` -> "Python: Select Interpreter" -> `.venv`.

## Pipeline
```bash
python scripts/download_data.py yoochoose                # raw file -> data/raw/yoochoose/
python scripts/prepare_data.py configs/yoochoose_f64.yaml # -> data/processed/yoochoose_f64/
python scripts/run_baselines.py configs/yoochoose_f64.yaml          # tune on VAL only
python scripts/run_baselines.py configs/yoochoose_f64.yaml --final  # touch TEST once, at the end
```
Smoke test with synthetic data (no download): `pytest -q`.

## Protocol (one place, never change it silently)
* Chronological split by session start-time quantiles; sessions crossing a boundary are dropped.
* Item vocabulary from TRAIN only; OOV events removed from val/test and counted in `meta.json`.
* Every position t>=1 of every eval session is an instance; ranking over the full catalogue.
* Ties are ranked pessimistically. Metrics: HR@20 (= P@20), MRR@20, NDCG@20.
* 95% CIs: bootstrap over sessions. Hyper-parameters: VAL only. TEST: once, with `--final`.

## Layout
```
configs/   datasets.yaml (sources), <dataset>.yaml (split/eval settings)
scripts/   download_data, prepare_data, run_baselines, make_fake_yoochoose
src/sessrec/{data,eval,models}
tests/     leakage, tie-handling, metrics, end-to-end smoke
notebooks/ plots / exploration only (no logic lives here)
results/   CSVs written by scripts (commit these)
docs/      architecture_spec.md  <- fill this in before step 1
```

## Step 1c/1d/2/3 commands (after prepare_data + run_baselines)
```bash
# --- CPU, local ---
python scripts/run_adaptive.py configs/yoochoose_f64.yaml [--final]   # detector + score-mixing rules (null result)
python scripts/run_slots.py    configs/yoochoose_f64.yaml [--final]   # slot-reservation rule (last adaptive rule)
python scripts/run_curve.py    configs/yoochoose_f64.yaml [--final]   # cold-start / adaptation curve (C1), classical models

# --- GPU, Colab: upload this repo + data/processed/yoochoose_f64/ and results/yoochoose_f64_best_params.json ---
pip install -r requirements.txt            # torch is preinstalled on Colab
pytest -q                                  # neural test must pass
python scripts/train_neural.py configs/yoochoose_f64.yaml --model gru    --trials 8 --final
python scripts/train_neural.py configs/yoochoose_f64.yaml --model sasrec --trials 8 --final
python scripts/compare_models.py configs/yoochoose_f64.yaml            # paired neural-vs-Markov table
```
Rule: `--final` touches TEST. Run it once per model, after tuning; never tune on its output.
