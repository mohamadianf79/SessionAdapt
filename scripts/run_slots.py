"""Step C: slot-reservation rule. Tune (m, theta) on VAL with the PRE-SPECIFIED criterion HR@K[all];
--final evaluates the chosen setting once on TEST. STOP RULE (decide before looking): if there is no
significant gain on the shift subset, or a significant loss on no_shift, report a NEGATIVE result and stop.
Usage: python scripts/run_slots.py configs/yoochoose_f64.yaml [--final]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sessrec.data.prepare import load_processed, to_sessions
from sessrec.eval.protocol import make_instances, ranks_of_targets
from sessrec.eval.metrics import per_instance, paired_diff_ci
from sessrec.models.markov import LagMarkov
from sessrec.models.shift_aware import CategoryPrior
from sessrec.models.slots import SlotReserve
from sessrec.analysis.shift import item_category_map, shift_labels, cat_valid_from_cfg
from sessrec.analysis.detector import PrefixStats, prefix_features, ShiftDetector

ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--final", action="store_true")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config)); K = cfg["eval"]["ks"][0]
data, meta = load_processed(f"data/processed/{cfg['name']}")
n_items = meta["n_items"]
best = json.load(open(f"results/{cfg['name']}_best_params.json"))["markov"]
tr_items, _ = to_sessions(data["train"])
pop_counts = np.bincount(np.concatenate(tr_items), minlength=n_items).astype(float)
item_cat = item_category_map(data["train"], n_items, cat_valid_from_cfg(cfg))
base = LagMarkov(**best).fit(tr_items, n_items)
stats = PrefixStats(base.T[0], pop_counts, item_cat)
cp = CategoryPrior(tr_items, item_cat, pop_counts)


def build(split, cap):
    it, ts = to_sessions(data[split])
    inst = make_instances(it, ts, cap)
    return inst, shift_labels(inst, item_cat), prefix_features(inst.items, inst.ts, stats)


def summarize(m, inst, lab, ref=None):
    out = {}
    for grp, mask in {"all": np.ones(len(lab), bool), "shift": lab == 1, "no_shift": lab == 0}.items():
        for k in (f"HR@{K}", f"MRR@{K}"):
            out[f"{k}[{grp}]"] = m[k][mask].mean()
            if ref is not None:
                d, lo, hi = paired_diff_ci(m[k][mask], ref[k][mask], inst.sess[mask], n_boot=300)
                out[f"d{k}[{grp}]"] = d; out[f"d{k}[{grp}]_CI"] = f"[{lo:+.4f},{hi:+.4f}]"
    return out


vinst, vlab, vX = build("val", cfg["eval"]["max_sessions_val"])
known = vlab >= 0
det = ShiftDetector().fit(vX[known], vlab[known])
gate = lambda P, T: det.prob(prefix_features(P, T, stats))
vbase = per_instance(ranks_of_targets(base, vinst), K)
grid = [dict(m=0, theta=0.3)] + [dict(m=m, theta=t) for m in (2, 5, 10) for t in (0.2, 0.3, 0.4, 0.5)]
rows = []
for p in grid:
    mod = SlotReserve(base, gate, cp, K=K, **p)
    r = summarize(per_instance(ranks_of_targets(mod, vinst), K), vinst, vlab, vbase)
    rows.append({**p, **r})
    print(f"[val] {p} HR[all]={r[f'HR@{K}[all]']:.4f} HR[shift]={r[f'HR@{K}[shift]']:.4f} HR[no_shift]={r[f'HR@{K}[no_shift]']:.4f}", flush=True)
df = pd.DataFrame(rows).sort_values(f"HR@{K}[all]", ascending=False)
Path("results").mkdir(exist_ok=True)
df.to_csv(f"results/{cfg['name']}_slots_val.csv", index=False)
bp = df.iloc[0]; bp = dict(m=int(bp.m), theta=float(bp.theta))
json.dump(bp, open(f"results/{cfg['name']}_slots_best.json", "w"))
print("\nbest on VAL (criterion HR[all]):", bp)

if a.final:
    tinst, tlab, _ = build("test", cfg["eval"]["max_sessions_test"])
    tb = per_instance(ranks_of_targets(base, tinst), K)
    mod = SlotReserve(base, gate, cp, K=K, **bp)
    tm = per_instance(ranks_of_targets(mod, tinst), K)
    out = pd.DataFrame([{"model": "base_markov", **summarize(tb, tinst, tlab)},
                        {"model": mod.name, **summarize(tm, tinst, tlab, tb)}])
    out.to_csv(f"results/{cfg['name']}_slots_test.csv", index=False)
    print(out.T.to_string())
else:
    print("(test untouched; add --final)")
