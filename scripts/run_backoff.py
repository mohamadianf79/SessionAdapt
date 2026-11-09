"""Is the gain of the category prior (seen on Diginetica) a SHIFT-adaptation effect or just a generic back-off?
Variants added to the tuned Markov base (all tuned on VAL over the same mu grid that INCLUDES mu=0):
   cat_ungated : base + mu * S_cat                    (category prior on every instance, no detector)
   cat_gated   : base + mu * g * S_cat                (g = prefix-only shift detector, as in run_adaptive)
   cat_shuffled: like cat_ungated, categories randomly permuted among items   (control: do real categories matter?)
   pop_prior   : base + mu * popularity               (control: is any prior enough?)
Reported on TEST for all / shift / no_shift instances with paired session-bootstrap CIs.
Usage: python scripts/run_backoff.py configs/diginetica.yaml [--final] [--tune_sessions 2000]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sessrec.data.prepare import load_processed, to_sessions
from sessrec.eval.protocol import make_instances, ranks_of_targets
from sessrec.eval.metrics import per_instance, paired_diff_ci
from sessrec.models.base import Recommender
from sessrec.models.markov import LagMarkov
from sessrec.models.shift_aware import CategoryPrior
from sessrec.analysis.shift import item_category_map, shift_labels, cat_valid_from_cfg
from sessrec.analysis.detector import PrefixStats, prefix_features, ShiftDetector

ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--final", action="store_true")
ap.add_argument("--tune_sessions", type=int, default=2000)
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
rng = np.random.default_rng(0)
shuf = item_cat.copy(); kn = np.flatnonzero(item_cat >= 0); shuf[kn] = item_cat[rng.permutation(kn)]
cp_shuf = CategoryPrior(tr_items, shuf, pop_counts)
pop_norm = pop_counts / (pop_counts.max() + 1.0)


class AddPrior(Recommender):
    def __init__(self, base, prior_fn, mu, gate_fn=None):
        self.base, self.prior_fn, self.mu, self.gate_fn = base, prior_fn, mu, gate_fn

    def score(self, prefixes, ts=None):
        S = self.base.score(prefixes, ts)
        if self.mu == 0:
            return S
        P = self.prior_fn(prefixes)
        if self.gate_fn is not None:
            P = P * self.gate_fn(prefixes, ts)[:, None]
        return S + self.mu * P


last = lambda P: np.array([p[-1] for p in P])
priors = {"cat_ungated": lambda P: cp.score(last(P)), "cat_gated": lambda P: cp.score(last(P)),
          "cat_shuffled": lambda P: cp_shuf.score(last(P)), "pop_prior": lambda P: np.tile(pop_norm, (len(P), 1))}
MUS = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)


def build(split, cap):
    it, ts = to_sessions(data[split]); inst = make_instances(it, ts, cap)
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


vinst, vlab, vX = build("val", a.tune_sessions)
known = vlab >= 0
det = ShiftDetector().fit(vX[known], vlab[known])
gate = lambda P, T: det.prob(prefix_features(P, T, stats))
vbase = per_instance(ranks_of_targets(base, vinst), K)
chosen, rows = {}, []
for name, fn in priors.items():
    scored = []
    for mu in MUS:
        mod = AddPrior(base, fn, mu, gate if name == "cat_gated" else None)
        m = per_instance(ranks_of_targets(mod, vinst), K)
        scored.append((m[f"MRR@{K}"].mean(), mu))
        rows.append({"variant": name, "mu": mu, **summarize(m, vinst, vlab, vbase)})
        print(f"[val] {name} mu={mu}: MRR={scored[-1][0]:.4f}", flush=True)
    chosen[name] = max(scored)[1]
    if chosen[name] == MUS[-1]:
        print(f"  NOTE: best mu for {name} is at the edge of the grid; extend MUS before trusting it")
Path("results").mkdir(exist_ok=True)
pd.DataFrame(rows).to_csv(f"results/{cfg['name']}_backoff_val.csv", index=False)
json.dump(chosen, open(f"results/{cfg['name']}_backoff_best.json", "w"))
print("chosen on VAL:", chosen)
if not a.final:
    print("(test untouched; add --final)"); sys.exit()

tinst, tlab, _ = build("test", cfg["eval"]["max_sessions_test"])
tb = per_instance(ranks_of_targets(base, tinst), K)
out = [{"variant": "base_markov", "mu": 0.0, **summarize(tb, tinst, tlab)}]
for name, fn in priors.items():
    mod = AddPrior(base, fn, chosen[name], gate if name == "cat_gated" else None)
    out.append({"variant": name, "mu": chosen[name], **summarize(per_instance(ranks_of_targets(mod, tinst), K), tinst, tlab, tb)})
df = pd.DataFrame(out); df.to_csv(f"results/{cfg['name']}_backoff_test.csv", index=False)
pd.set_option("display.width", 250)
cols = ["variant", "mu"] + [f"d{k}[{g}]" for g in ("all", "shift", "no_shift") for k in (f"HR@{K}",)] + [f"dHR@{K}[{g}]_CI" for g in ("all", "shift", "no_shift")]
print(df[cols].round(4).to_string(index=False))
