"""Step 1b/1c: (1) can a PREFIX-ONLY detector predict the shift?  (2) does a shift-aware rule help?
Everything is fitted/tuned on VAL; with --final the chosen config is evaluated once on TEST
(the detector is then fitted on all of VAL and applied to TEST).
Usage: python scripts/run_adaptive.py configs/yoochoose_f64.yaml [--final]"""
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sessrec.data.prepare import load_processed, to_sessions
from sessrec.eval.protocol import make_instances, ranks_of_targets
from sessrec.eval.metrics import per_instance, mean_ci, paired_diff_ci
from sessrec.models.markov import LagMarkov
from sessrec.models.shift_aware import ShiftAware, CategoryPrior
from sessrec.analysis.shift import item_category_map, shift_labels
from sessrec.analysis.detector import FEATURES, PrefixStats, prefix_features, ShiftDetector

ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--final", action="store_true")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config)); K = cfg["eval"]["ks"][0]
data, meta = load_processed(f"data/processed/{cfg['name']}")
n_items = meta["n_items"]
best = json.load(open(f"results/{cfg['name']}_best_params.json"))["markov"]
tr_items, _ = to_sessions(data["train"])
pop_counts = np.bincount(np.concatenate(tr_items), minlength=n_items).astype(float)
item_cat = item_category_map(data["train"], n_items)
long_m = LagMarkov(**best).fit(tr_items, n_items)
short_m = LagMarkov(L=1, decay=1.0).fit(tr_items, n_items)
stats = PrefixStats(long_m.T[0] if long_m.L >= 1 else short_m.T[0], pop_counts, item_cat)
cp = CategoryPrior(tr_items, item_cat, pop_counts)
Path("results").mkdir(exist_ok=True)


def build(split, cap):
    it, ts = to_sessions(data[split])
    inst = make_instances(it, ts, cap)
    return inst, shift_labels(inst, item_cat), prefix_features(inst.items, inst.ts, stats)


vinst, vlab, vX = build("val", cfg["eval"]["max_sessions_val"])
known = vlab >= 0
print(f"VAL instances={len(vinst)} known={known.sum()} shift={np.sum(vlab==1)}")

# ---- (1) detector quality: AUC for predicting shift from the prefix only (grouped CV on VAL) ----
rows = []
for j, f in enumerate(FEATURES):
    rows.append({"detector": f"single:{f}", "AUC": roc_auc_score(vlab[known], vX[known, j])})
oof = np.zeros(len(vlab))
for tr, te in GroupKFold(5).split(vX[known], vlab[known], vinst.sess[known]):
    idx = np.flatnonzero(known)
    oof[idx[te]] = ShiftDetector().fit(vX[idx[tr]], vlab[idx[tr]]).prob(vX[idx[te]])
rows.append({"detector": "logistic(all features)", "AUC": roc_auc_score(vlab[known], oof[known])})
det_df = pd.DataFrame(rows); print(det_df.to_string(index=False))
det_df.to_csv(f"results/{cfg['name']}_detector_auc_val.csv", index=False)
print("(AUC<0.5 for a single feature just means the opposite direction; 0.5 = no signal)\n")

# ---- (2) tune the shift-aware rule on VAL ----
det = ShiftDetector().fit(vX[known], vlab[known])
base_rank = ranks_of_targets(long_m, vinst)
base_m = per_instance(base_rank, K)


def summarize(m, inst, lab, base=None):
    out = {}
    for grp, mask in {"all": np.ones(len(lab), bool), "shift": lab == 1, "no_shift": lab == 0}.items():
        for k in (f"HR@{K}", f"MRR@{K}"):
            out[f"{k}[{grp}]"] = m[k][mask].mean()
            if base is not None:
                d, lo, hi = paired_diff_ci(m[k][mask], base[k][mask], inst.sess[mask], n_boot=300)
                out[f"d{k}[{grp}]"] = d; out[f"d{k}[{grp}]_CI"] = f"[{lo:+.4f},{hi:+.4f}]"
    return out


grid = [dict(w=w, mu=mu, theta=th) for w in (0, 0.5, 1.0) for mu in (0, 0.1, 0.3, 1.0) for th in (None, 0.3)
        if not (w == 0 and mu == 0 and th is not None)]
res = []
for p in grid:
    m = ShiftAware(long_m, short_m, det, stats, cp, **p)
    pm = per_instance(ranks_of_targets(m, vinst), K)
    r = summarize(pm, vinst, vlab, base_m)
    res.append({**p, **r})
    print(f"[val] {p}  MRR={r[f'MRR@{K}[all]']:.4f} (shift {r[f'MRR@{K}[shift]']:.4f})", flush=True)
res_df = pd.DataFrame(res).sort_values(f"MRR@{K}[all]", ascending=False)
res_df.to_csv(f"results/{cfg['name']}_adaptive_val.csv", index=False)
bp = res_df.iloc[0]; bp = dict(w=float(bp.w), mu=float(bp.mu), theta=None if pd.isna(bp.theta) else float(bp.theta))
json.dump(bp, open(f"results/{cfg['name']}_adaptive_best.json", "w"))
print("\nbest on VAL:", bp)

# ---- final: TEST, once ----
if a.final:
    tinst, tlab, tX = build("test", cfg["eval"]["max_sessions_test"])
    tb = per_instance(ranks_of_targets(long_m, tinst), K)
    rows = [{"model": "base_markov", **summarize(tb, tinst, tlab)}]
    tm = ShiftAware(long_m, short_m, det, stats, cp, **bp)
    tp = per_instance(ranks_of_targets(tm, tinst), K)
    rows.append({"model": tm.name, **summarize(tp, tinst, tlab, tb)})
    kn = tlab >= 0
    pr = det.prob(tX)
    print(f"[TEST] detector AUC (fit on VAL) = {roc_auc_score(tlab[kn], pr[kn]):.4f}")
    out = pd.DataFrame(rows); out.to_csv(f"results/{cfg['name']}_adaptive_test.csv", index=False)
    print(out.T.to_string())
else:
    print("(test untouched; add --final)")
