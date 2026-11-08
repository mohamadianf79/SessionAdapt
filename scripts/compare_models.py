"""Paired comparison (session bootstrap) of neural models vs the tuned Markov baseline on the SAME test instances.
Run after train_neural.py --final for the models you want. Usage: python scripts/compare_models.py configs/yoochoose_f64.yaml"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sessrec.data.prepare import load_processed, to_sessions
from sessrec.eval.protocol import make_instances, ranks_of_targets
from sessrec.eval.metrics import per_instance, paired_diff_ci
from sessrec.models.markov import LagMarkov

cfg = yaml.safe_load(open(sys.argv[1])); K = cfg["eval"]["ks"][0]
data, meta = load_processed(f"data/processed/{cfg['name']}")
tr, _ = to_sessions(data["train"])
it, ts = to_sessions(data["test"])
inst = make_instances(it, ts, cfg["eval"]["max_sessions_test"])          # identical instance set as train_neural.py
best = json.load(open(f"results/{cfg['name']}_best_params.json"))["markov"]
ref = per_instance(ranks_of_targets(LagMarkov(**best).fit(tr, meta["n_items"]), inst), K)
rows = []
for model in ("gru", "sasrec"):
    files = sorted(Path("results/preds").glob(f"{cfg['name']}_{model}_test_seed*.npy"))
    if not files:
        continue
    ranks = [np.load(f) for f in files]
    assert len(ranks[0]) == len(inst), "instance set differs: same config/cap/seed required"
    avg = {k: np.mean([per_instance(r, K)[k] for r in ranks], 0) for k in ref}
    for k in ref:
        d, lo, hi = paired_diff_ci(avg[k], ref[k], inst.sess, n_boot=500)
        rows.append({"model": model, "metric": k, "neural": avg[k].mean(), "markov": ref[k].mean(),
                     "diff": d, "lo": lo, "hi": hi, "seeds": len(files)})
df = pd.DataFrame(rows); print(df.round(4).to_string(index=False))
df.to_csv(f"results/{cfg['name']}_neural_vs_markov_test.csv", index=False)
