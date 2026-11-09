"""Test of the hypothesis  H3: 'clock time has no direct effect on next-item recommendation'.
Primary comparisons (fixed in advance, 2 tests):
   P1  n=0  (first click, time is the ONLY context)  : HR@K  of TimePop  vs  global Pop
   P2  n>=1 (after >=1 click)                          : MRR@K of Markov+TimePop  vs  Markov
Controls: 'global' (adds a popularity prior without time) and 'shuffled' (time labels randomly permuted).
Rule: GAIN only if lower 95% CI > 0; 'no effect' (EQUIVALENT) only if the 90% CI lies inside +-margin.
Everything is tuned on VAL; --final evaluates TEST once.
Usage: python scripts/run_time.py configs/yoochoose_f64.yaml [--final] [--scheme daypart4_wkend] [--margin_hr 0.005 --margin_mrr 0.003]"""
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sessrec.data.prepare import load_processed, to_sessions
from sessrec.eval.protocol import make_instances, ranks_of_targets
from sessrec.eval.metrics import per_instance, paired_diff_ci, verdict
from sessrec.models.popularity import Pop
from sessrec.models.markov import LagMarkov
from sessrec.models.timepop import TimePop, TimeBlend

ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--final", action="store_true")
ap.add_argument("--scheme", default="daypart4_wkend"); ap.add_argument("--margin_hr", type=float, default=0.005)
ap.add_argument("--margin_mrr", type=float, default=0.003)
ap.add_argument("--tune_sessions", type=int, default=2000, help="val sessions used for tuning (speed)")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config)); K = cfg["eval"]["ks"][0]
data, meta = load_processed(f"data/processed/{cfg['name']}")
n_items = meta["n_items"]
best = json.load(open(f"results/{cfg['name']}_best_params.json"))["markov"]
tr_items, tr_ts = to_sessions(data["train"])
span_days = (max(t.max() for t in tr_ts) - min(t.min() for t in tr_ts)) / 86400
print(f"train span = {span_days:.1f} days  (scheme={a.scheme})")
if cfg.get("has_clock_time", True) is False and a.scheme not in ("dow7", "none"):
    sys.exit(f"{cfg['name']} has day-resolution timestamps only: use --scheme dow7 (hour/daypart bins would be meaningless)")
if span_days < 14 and "dow" in a.scheme + "wkend":
    print("WARNING: <14 days of training data: the weekday/weekend part of the bins is NOT reliably testable; "
          "use a longer slice (configs/yoochoose_f8.yaml) for the weekday claim.")

pop = Pop().fit(tr_items, n_items)
mk = LagMarkov(**best).fit(tr_items, n_items)


def insts(split, cap):
    it, ts = to_sessions(data[split]); return make_instances(it, ts, cap, start=0)


vi, ti = insts("val", min(a.tune_sessions, cfg["eval"]["max_sessions_val"])), insts("test", cfg["eval"]["max_sessions_test"])


def metrics(model, inst):
    return per_instance(ranks_of_targets(model, inst), K)


# w=0 (= the base model) is IN the grid, so the tuned variant can never be worse than the base on VAL by construction
ALPHAS, WS = (10.0, 1000.0), (0.0, 1e-4, 1e-3, 1e-2, 1e-1)
m0v, m1v = vi.n == 0, vi.n >= 1
base0v, base1v = metrics(pop, vi), metrics(mk, vi)

# ---- tuning on VAL ----
tuned = {}
for mode in ("time", "shuffled"):
    best0 = max(ALPHAS, key=lambda al: metrics(TimePop(a.scheme, al, mode).fit(tr_items, n_items, tr_ts), vi)[f"HR@{K}"][m0v].mean())
    print(f'tuning mode={mode} ...', flush=True)
    cands = [(al, w) for al in ALPHAS for w in WS]
    scored = []
    for al, w in cands:
        tp = TimePop(a.scheme, al, mode).fit(tr_items, n_items, tr_ts)
        scored.append((metrics(TimeBlend(mk, tp, w), vi)[f"MRR@{K}"][m1v].mean(), al, w))
        print(f'  alpha={al} w={w}: val MRR[n>=1]={scored[-1][0]:.4f}', flush=True)
    _, al1, w1 = max(scored)
    tuned[mode] = dict(alpha0=best0, alpha1=al1, w=w1)
scored = [(metrics(TimeBlend(mk, TimePop(a.scheme, 1.0, "global").fit(tr_items, n_items, tr_ts), w), vi)[f"MRR@{K}"][m1v].mean(), w) for w in WS]
tuned["global"] = dict(w=max(scored)[1])
print("tuned on VAL:", json.dumps(tuned))
Path("results").mkdir(exist_ok=True)
json.dump(tuned, open(f"results/{cfg['name']}_time_{a.scheme}_tuned.json", "w"))

if not a.final:
    print("(test untouched; add --final)"); sys.exit()

# ---- TEST, once ----
b0, b1 = metrics(pop, ti), metrics(mk, ti)
rows = []


def compare(label, var, base, mask, metric, margin):
    d, lo, hi = paired_diff_ci(var[metric][mask], base[metric][mask], ti.sess[mask], n_boot=1000)
    _, lo9, hi9 = paired_diff_ci(var[metric][mask], base[metric][mask], ti.sess[mask], n_boot=1000, alpha=0.10)
    rows.append({"comparison": label, "metric": metric, "base": base[metric][mask].mean(), "variant": var[metric][mask].mean(),
                 "diff": d, "rel_diff": d / max(base[metric][mask].mean(), 1e-12), "ci95": f"[{lo:+.4f},{hi:+.4f}]", "ci90": f"[{lo9:+.4f},{hi9:+.4f}]",
                 "n_inst": int(mask.sum()), "verdict": verdict(d, lo, hi, lo9, hi9, margin)})


for mode in ("time", "shuffled"):
    t = tuned[mode]
    tp0 = TimePop(a.scheme, t["alpha0"], mode).fit(tr_items, n_items, tr_ts)
    compare(f"P1 n=0: timepop[{mode}] vs pop", metrics(tp0, ti), b0, ti.n == 0, f"HR@{K}", a.margin_hr)
    tp1 = TimePop(a.scheme, t["alpha1"], mode).fit(tr_items, n_items, tr_ts)
    blend = metrics(TimeBlend(mk, tp1, t["w"]), ti)
    compare(f"P2 n>=1: markov+timepop[{mode}] vs markov", blend, b1, ti.n >= 1, f"MRR@{K}", a.margin_mrr)
    compare(f"P2 n>=1: markov+timepop[{mode}] vs markov", blend, b1, ti.n >= 1, f"HR@{K}", a.margin_hr)
    if mode == "time":
        for lab, mk_ in {"n=1": ti.n == 1, "n=2-3": (ti.n >= 2) & (ti.n <= 3), "n>=4": ti.n >= 4}.items():
            compare(f"  by n {lab}: markov+timepop[time] vs markov", blend, b1, mk_, f"MRR@{K}", a.margin_mrr)
gl = TimePop(a.scheme, 1.0, "global").fit(tr_items, n_items, tr_ts)
compare("P2 control n>=1: markov+GLOBALpop vs markov", metrics(TimeBlend(mk, gl, tuned["global"]["w"]), ti), b1, ti.n >= 1, f"MRR@{K}", a.margin_mrr)
df = pd.DataFrame(rows)
df.to_csv(f"results/{cfg['name']}_time_{a.scheme}_test.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
print(df[["comparison", "metric", "base", "variant", "diff", "rel_diff", "ci95", "verdict"]].round(4).to_string(index=False))
