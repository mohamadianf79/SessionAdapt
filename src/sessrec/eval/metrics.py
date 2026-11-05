from __future__ import annotations
import numpy as np


def per_instance(ranks: np.ndarray, k: int) -> dict:
    hit = ranks <= k
    return {
        f"HR@{k}": hit.astype(float),
        f"MRR@{k}": np.where(hit, 1.0 / ranks, 0.0),
        f"NDCG@{k}": np.where(hit, 1.0 / np.log2(ranks + 1.0), 0.0),
    }


def _session_sums(vals, sess):
    S = sess.max() + 1
    return np.bincount(sess, weights=vals, minlength=S), np.bincount(sess, minlength=S).astype(float)


def mean_ci(vals, sess, n_boot=1000, seed=0, alpha=0.05):
    """Cluster bootstrap over SESSIONS (instances within a session are not independent)."""
    num, den = _session_sums(vals, sess)
    rng = np.random.default_rng(seed)
    S = len(num)
    est = []
    for _ in range(n_boot):
        r = rng.integers(0, S, S)
        est.append(num[r].sum() / den[r].sum())
    lo, hi = np.quantile(est, [alpha / 2, 1 - alpha / 2])
    return num.sum() / den.sum(), lo, hi


def paired_diff_ci(a, b, sess, n_boot=1000, seed=0, alpha=0.05):
    """mean(a-b) with a session-level paired bootstrap CI."""
    return mean_ci(np.asarray(a) - np.asarray(b), sess, n_boot, seed, alpha)
