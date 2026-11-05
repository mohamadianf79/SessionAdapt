"""Writes a small synthetic file in the real Yoochoose clicks.dat format (for smoke tests only).
Sessions follow a noisy first-order Markov chain so that Markov > Pop must hold."""
import sys
from pathlib import Path
import numpy as np, pandas as pd

out = Path(sys.argv[1] if len(sys.argv) > 1 else "data/raw/fake/yoochoose-clicks.dat")
out.parent.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(0)
n_items, n_sess = 300, 6000
succ = {i: rng.choice(n_items, 3, replace=False) for i in range(n_items)}
rows, t0 = [], pd.Timestamp("2014-04-01")
for s in range(n_sess):
    start = t0 + pd.Timedelta(seconds=int(s * 40 + rng.integers(0, 30)))
    L = int(rng.integers(2, 12)); it = int(rng.integers(0, n_items))
    for k in range(L):
        ts = (start + pd.Timedelta(seconds=int(k * 20))).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        rows.append((s + 1, ts, 1000 + it, (it % 12) + 1))
        it = int(rng.choice(succ[it])) if rng.random() < 0.8 else int(rng.integers(0, n_items))
pd.DataFrame(rows).to_csv(out, header=False, index=False)
print("wrote", out, len(rows), "rows")
