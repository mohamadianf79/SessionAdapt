"""Tune baselines on VAL only, then (with --final) evaluate the chosen config once on TEST.
Usage: python scripts/run_baselines.py configs/yoochoose_f64.yaml [--final]"""
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sessrec.data.prepare import load_processed, to_sessions
from sessrec.eval.protocol import make_instances, ranks_of_targets
from sessrec.eval.metrics import per_instance, mean_ci
from sessrec.models.popularity import Pop, SPop
from sessrec.models.markov import LagMarkov
from sessrec.models.sknn import SKNN

ap = argparse.ArgumentParser()
ap.add_argument("config"); ap.add_argument("--final", action="store_true")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config))
data, meta = load_processed(f"data/processed/{cfg['name']}")
n_items = meta["n_items"]
tr_items, tr_ts = to_sessions(data["train"])
K = cfg["eval"]["ks"][0]

GRID = {
    "pop": (lambda: Pop(), [{}]),
    "spop": (lambda: SPop(), [{}]),
    "markov": (lambda **p: LagMarkov(**p), [dict(L=L, decay=d) for L in (1, 2, 3, 5) for d in (0.5, 0.7, 1.0)]),
    "sknn": (lambda **p: SKNN(**p), [dict(k=k) for k in (50, 100, 500)]),
}


def evaluate(model, split, cap, seed=0):
    it, ts = to_sessions(data[split])
    inst = make_instances(it, ts, cap, seed)
    r = ranks_of_targets(model, inst)
    m = per_instance(r, K)
    row = {"n_instances": len(inst), "n_sessions": int(inst.sess.max() + 1)}
    for k, v in m.items():
        mu, lo, hi = mean_ci(v, inst.sess, n_boot=500)
        row.update({k: mu, k + "_lo": lo, k + "_hi": hi})
    for lab, mask in {"n=1": inst.n == 1, "n=2": inst.n == 2, "n=3": inst.n == 3, "n>=5": inst.n >= 5}.items():
        if mask.any():
            row[f"HR@{K}[{lab}]"] = m[f"HR@{K}"][mask].mean()
            row[f"MRR@{K}[{lab}]"] = m[f"MRR@{K}"][mask].mean()
    return row


best = {}
rows = []
for name, (make, grid) in GRID.items():
    scored = []
    for p in grid:
        model = make(**p).fit(tr_items, n_items)
        r = evaluate(model, "val", cfg["eval"]["max_sessions_val"])
        scored.append((r[f"MRR@{K}"], p, r))
        print(f"[val] {name} {p} MRR@{K}={r[f'MRR@{K}']:.4f} HR@{K}={r[f'HR@{K}']:.4f}", flush=True)
    s, p, r = max(scored, key=lambda x: x[0])
    best[name] = p
    rows.append({"model": name, "params": json.dumps(p), "split": "val", **r})

Path("results").mkdir(exist_ok=True)
json.dump(best, open(f"results/{cfg['name']}_best_params.json", "w"), indent=2)
pd.DataFrame(rows).to_csv(f"results/{cfg['name']}_baselines_val.csv", index=False)

if a.final:
    rows = []
    for name, (make, _) in GRID.items():
        model = make(**best[name]).fit(tr_items, n_items)
        r = evaluate(model, "test", cfg["eval"]["max_sessions_test"])
        rows.append({"model": name, "params": json.dumps(best[name]), "split": "test", **r})
        print(f"[TEST] {name} MRR@{K}={r[f'MRR@{K}']:.4f} HR@{K}={r[f'HR@{K}']:.4f}")
    pd.DataFrame(rows).to_csv(f"results/{cfg['name']}_baselines_test.csv", index=False)
else:
    print("\n(test set untouched; add --final when tuning is finished)")
