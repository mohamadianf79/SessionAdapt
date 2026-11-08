"""Cold-start / adaptation curves for the classical models (neural models write the same CSVs from train_neural.py).
Usage: python scripts/run_curve.py configs/yoochoose_f64.yaml [--final] [--models pop,spop,markov] [--N 5]
Default split = VAL (exploratory); --final = TEST."""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sessrec.data.prepare import load_processed, to_sessions
from sessrec.eval.protocol import make_instances, ranks_of_targets
from sessrec.eval.curve import adaptation_curve, population_curve
from sessrec.models.popularity import Pop, SPop
from sessrec.models.markov import LagMarkov
from sessrec.models.sknn import SKNN

ap = argparse.ArgumentParser(); ap.add_argument("config")
ap.add_argument("--final", action="store_true"); ap.add_argument("--models", default="pop,spop,markov")
ap.add_argument("--N", type=int, default=5); ap.add_argument("--max_sessions", type=int, default=None)
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config)); K = cfg["eval"]["ks"][0]
split = "test" if a.final else "val"
data, meta = load_processed(f"data/processed/{cfg['name']}")
best = json.load(open(f"results/{cfg['name']}_best_params.json"))
tr, _ = to_sessions(data["train"])
makers = {"pop": lambda: Pop(), "spop": lambda: SPop(),
          "markov": lambda: LagMarkov(**best["markov"]), "sknn": lambda: SKNN(**best["sknn"])}
it, ts = to_sessions(data[split])
inst = make_instances(it, ts, a.max_sessions, start=0)
print(f"{split}: {inst.sess.max()+1} sessions, {len(inst)} instances (incl. n=0)")
curves, summs, pops = [], [], []
for name in a.models.split(","):
    m = makers[name]().fit(tr, meta["n_items"])
    r = ranks_of_targets(m, inst)
    c, s = adaptation_curve(r, inst, K=K, N=a.N)
    p = population_curve(r, inst, K=K, N=a.N)
    for df, lst in ((c, curves), (s, summs), (p, pops)):
        df.insert(0, "model", name); lst.append(df)
    print(f"\n== {name}")
    print(c[c.metric == f"HR@{K}"][["n", "value", "lo", "hi"]].round(4).to_string(index=False))
    print(s.round(4).to_string(index=False))
Path("results").mkdir(exist_ok=True)
for nm, lst in (("curve", curves), ("curve_summary", summs), ("curve_population", pops)):
    pd.concat(lst).to_csv(f"results/{cfg['name']}_{nm}_{split}.csv", index=False)
print(f"\nwritten results/{cfg['name']}_curve*_{split}.csv")
