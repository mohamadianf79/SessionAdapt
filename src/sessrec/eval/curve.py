"""Cold-start / adaptation curve (contribution C1).

Instances must be built with make_instances(..., start=0) so that n = 0,1,2,... are all present.
Main curve = COHORT curve: only sessions with >= N+1 instances, so exactly the SAME sessions contribute at
every n (otherwise the curve mixes different populations of sessions).
Summary metrics (all with session-bootstrap CIs), c = curve over n=0..N, max = max(c):
  gain_first_click : c[1] - c[0]            (what the first click buys)
  n_q              : first n with c[n] >= q*max(c)   (adaptation speed; q=0.9)
  adapt_auc        : mean(c)/max(c)         (1 = instantly at full performance, small = slow to adapt)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .metrics import mean_ci


def _matrices(ranks, inst, K, N):
    S = int(inst.sess.max()) + 1
    L = np.bincount(inst.sess, minlength=S)
    cohort = L >= N + 1
    cid = np.full(S, -1); cid[cohort] = np.arange(cohort.sum())
    sel = (inst.n <= N) & cohort[inst.sess]
    hit = (ranks <= K).astype(float)
    rr = np.where(ranks <= K, 1.0 / ranks, 0.0)
    Sc = int(cohort.sum())
    H = np.zeros((Sc, N + 1)); R = np.zeros((Sc, N + 1))
    H[cid[inst.sess[sel]], inst.n[sel]] = hit[sel]
    R[cid[inst.sess[sel]], inst.n[sel]] = rr[sel]
    return H, R


def _summ(c, q):
    mx = c.max()
    if mx <= 0:
        return {"gain_first_click": np.nan, "n_q": np.nan, "adapt_auc": np.nan}
    return {"gain_first_click": c[1] - c[0], "n_q": float(np.argmax(c >= q * mx)), "adapt_auc": c.mean() / mx}


def adaptation_curve(ranks, inst, K=20, N=5, q=0.9, n_boot=300, seed=0):
    H, R = _matrices(ranks, inst, K, N)
    Sc = H.shape[0]
    if Sc < 30:
        raise ValueError(f"only {Sc} cohort sessions with length >= {N+1}; use more sessions or a smaller N")
    rng = np.random.default_rng(seed)
    curve_rows, summ_rows = [], []
    for name, M in (("HR", H), ("MRR", R)):
        c = M.mean(0)
        B = np.array([M[rng.integers(0, Sc, Sc)].mean(0) for _ in range(n_boot)])
        lo, hi = np.quantile(B, [0.025, 0.975], axis=0)
        for n in range(N + 1):
            curve_rows.append({"metric": f"{name}@{K}", "n": n, "value": c[n], "lo": lo[n], "hi": hi[n], "cohort_sessions": Sc})
        s0 = _summ(c, q)
        sb = pd.DataFrame([_summ(b, q) for b in B])
        for key, v in s0.items():
            summ_rows.append({"metric": f"{name}@{K}", "summary": key, "value": v,
                              "lo": sb[key].quantile(0.025), "hi": sb[key].quantile(0.975)})
    return pd.DataFrame(curve_rows), pd.DataFrame(summ_rows)


def population_curve(ranks, inst, K=20, N=5):
    """Same metric on ALL instances, bucketed by n (0..N-1, then N+). Different sessions per bucket: descriptive only."""
    rows = []
    hit = (ranks <= K).astype(float); rr = np.where(ranks <= K, 1.0 / ranks, 0.0)
    for lab, mask in [(str(n), inst.n == n) for n in range(N)] + [(f"{N}+", inst.n >= N)]:
        if mask.sum() < 30:
            continue
        for name, v in (("HR", hit), ("MRR", rr)):
            mu, lo, hi = mean_ci(v[mask], inst.sess[mask], n_boot=200)
            rows.append({"metric": f"{name}@{K}", "n": lab, "value": mu, "lo": lo, "hi": hi, "n_instances": int(mask.sum())})
    return pd.DataFrame(rows)
