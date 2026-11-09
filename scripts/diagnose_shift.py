"""How much does a strong base model lose when the user shifts category inside a session?
Usage: python scripts/diagnose_shift.py configs/yoochoose_f64.yaml
(needs data re-prepared with the category column: run prepare_data.py again once)"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sessrec.data.prepare import load_processed, to_sessions
from sessrec.eval.protocol import make_instances, ranks_of_targets
from sessrec.eval.metrics import per_instance, mean_ci
from sessrec.models.markov import LagMarkov
from sessrec.analysis.shift import item_category_map, shift_labels, cat_valid_from_cfg

cfg = yaml.safe_load(open(sys.argv[1]))
data, meta = load_processed(f"data/processed/{cfg['name']}")
assert "cat" in data["train"].columns, "re-run scripts/prepare_data.py (category column missing)"
K = cfg["eval"]["ks"][0]
best = json.load(open(f"results/{cfg['name']}_best_params.json"))["markov"]
tr_items, _ = to_sessions(data["train"])
model = LagMarkov(**best).fit(tr_items, meta["n_items"])
item_cat = item_category_map(data["train"], meta["n_items"], cat_valid_from_cfg(cfg))
print(f"items with a real category: {(item_cat >= 0).mean():.1%}")

it, ts = to_sessions(data["val"])          # VAL only (diagnosis, not a reported result)
inst = make_instances(it, ts, cfg["eval"]["max_sessions_val"])
m = per_instance(ranks_of_targets(model, inst), K)
lab = shift_labels(inst, item_cat)
print(f"instances={len(inst)}  unknown={np.mean(lab<0):.1%}  shift={np.mean(lab==1):.1%} of all, "
      f"{np.sum(lab==1)/max(1,np.sum(lab>=0)):.1%} of known")

rows = []
groups = {"all_known": lab >= 0, "no_shift": lab == 0, "shift": lab == 1}
for n_lab, nm in {"n=1": inst.n == 1, "n=2": inst.n == 2, "n=3": inst.n == 3, "n>=5": inst.n >= 5}.items():
    groups[f"no_shift {n_lab}"] = (lab == 0) & nm
    groups[f"shift {n_lab}"] = (lab == 1) & nm
for g, mask in groups.items():
    if mask.sum() < 30:
        continue
    row = {"group": g, "n_inst": int(mask.sum())}
    for k in (f"HR@{K}", f"MRR@{K}"):
        mu, lo, hi = mean_ci(m[k][mask], inst.sess[mask], n_boot=300)
        row[k] = round(mu, 4); row[k + "_CI"] = f"[{lo:.3f},{hi:.3f}]"
    rows.append(row)
df = pd.DataFrame(rows); print(df.to_string(index=False))
Path("results").mkdir(exist_ok=True)
df.to_csv(f"results/{cfg['name']}_shift_diagnosis_val.csv", index=False)
