"""Tune (random search, equal budget) and evaluate GRU4Rec-style / SASRec-style baselines.
Use a GPU (Colab). Everything is chosen on VAL; --final evaluates on TEST once per seed.
Usage:
  python scripts/train_neural.py configs/yoochoose_f64.yaml --model sasrec --trials 8            # tuning only
  python scripts/train_neural.py configs/yoochoose_f64.yaml --model sasrec --trials 8 --final    # + test, 3 seeds, curve
Outputs: results/<name>_<model>_trials.csv, _best.json, _test.csv, preds/<name>_<model>_test_seed<k>.npy (ranks),
         <name>_<model>_curve*_test.csv"""
import argparse, itertools, json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sessrec.data.prepare import load_processed, to_sessions
from sessrec.eval.protocol import make_instances, ranks_of_targets
from sessrec.eval.metrics import per_instance, mean_ci
from sessrec.eval.curve import adaptation_curve, population_curve
from sessrec.models.neural import NeuralRec

SPACES = {
    "gru": dict(emb=[64, 128], hidden=[64, 128, 256], dropout=[0.0, 0.2, 0.4], lr=[1e-3, 3e-3]),
    "sasrec": dict(d=[64, 128], layers=[1, 2], heads=[1, 2], dropout=[0.2, 0.4], lr=[1e-3, 3e-3]),
}
ap = argparse.ArgumentParser(); ap.add_argument("config")
ap.add_argument("--model", required=True, choices=list(SPACES))
ap.add_argument("--trials", type=int, default=8); ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--patience", type=int, default=3); ap.add_argument("--seeds", type=int, default=3)
ap.add_argument("--final", action="store_true"); ap.add_argument("--device", default=None)
ap.add_argument("--use_best", action="store_true", help="skip the random search; read hp from results/<name>_<model>_best.json (e.g. after a lost Colab session)")
a = ap.parse_args()
cfg = yaml.safe_load(open(a.config)); K = cfg["eval"]["ks"][0]
data, meta = load_processed(f"data/processed/{cfg['name']}")
n_items = meta["n_items"]
tr, _ = to_sessions(data["train"])
vit, vts = to_sessions(data["val"])
val_inst = make_instances(vit, vts, cfg["eval"]["max_sessions_val"])
Path("results/preds").mkdir(parents=True, exist_ok=True)
tag = f"results/{cfg['name']}_{a.model}"

keys = list(SPACES[a.model])
grid = [dict(zip(keys, v)) for v in itertools.product(*SPACES[a.model].values())]
grid = [g for g in grid if not (a.model == "sasrec" and g["d"] % g["heads"])]
order = [] if a.use_best else np.random.default_rng(0).permutation(len(grid))[: a.trials]
trials = []
for j, gi in enumerate(order):
    hp = grid[gi]
    print(f"\n=== trial {j+1}/{len(order)} {hp}", flush=True)
    m = NeuralRec(a.model, hp, seed=0, max_epochs=a.epochs, patience=a.patience, device=a.device)
    m.fit(tr, n_items, val_inst=val_inst)
    trials.append({**hp, "val_MRR": m.best_val, "epochs": m.epochs_run})
    pd.DataFrame(trials).to_csv(f"{tag}_trials.csv", index=False)
if a.use_best:
    best_hp = json.load(open(f"{tag}_best.json"))["hp"]
    print("using hp from", f"{tag}_best.json", best_hp)
else:
    best = max(trials, key=lambda t: t["val_MRR"])
    best_hp = {k: best[k] for k in keys}
    json.dump({"hp": best_hp, "val_MRR": best["val_MRR"], "epochs": best["epochs"]}, open(f"{tag}_best.json", "w"), indent=1)
    print("\nbest on VAL:", best_hp, f"MRR@{K}={best['val_MRR']:.4f}")

if a.final:
    tit, tts = to_sessions(data["test"])
    test_inst = make_instances(tit, tts, cfg["eval"]["max_sessions_test"])
    curve_inst = make_instances(tit, tts, None, start=0)
    per_seed, ranks_all = [], []
    for s in range(a.seeds):
        m = NeuralRec(a.model, best_hp, seed=s, max_epochs=a.epochs, patience=a.patience, device=a.device, verbose=False)
        m.fit(tr, n_items, val_inst=val_inst)
        r = ranks_of_targets(m, test_inst)
        np.save(f"results/preds/{cfg['name']}_{a.model}_test_seed{s}.npy", r)
        ranks_all.append(r)
        pm = per_instance(r, K)
        per_seed.append({"seed": s, **{k: v.mean() for k, v in pm.items()}})
        print(f"[TEST] seed {s}: " + " ".join(f"{k}={v.mean():.4f}" for k, v in pm.items()), flush=True)
        if s == 0:
            rc = ranks_of_targets(m, curve_inst)
            c, sm = adaptation_curve(rc, curve_inst, K=K)
            p = population_curve(rc, curve_inst, K=K)
            for df, nm in ((c, "curve"), (sm, "curve_summary"), (p, "curve_population")):
                df.insert(0, "model", a.model); df.to_csv(f"{tag}_{nm}_test.csv", index=False)
    ps = pd.DataFrame(per_seed)
    # seed-averaged per-instance metric -> session bootstrap CI
    avg = {k: np.mean([per_instance(r, K)[k] for r in ranks_all], 0) for k in per_instance(ranks_all[0], K)}
    row = {"model": a.model, "hp": json.dumps(best_hp), "seeds": a.seeds}
    for k, v in avg.items():
        mu, lo, hi = mean_ci(v, test_inst.sess, n_boot=500)
        row.update({k: mu, k + "_lo": lo, k + "_hi": hi, k + "_seed_std": ps[k].std()})
    pd.DataFrame([row]).to_csv(f"{tag}_test.csv", index=False)
    print(pd.DataFrame([row]).T.to_string())
else:
    print("(test untouched; add --final)")
