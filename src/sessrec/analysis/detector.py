"""Prefix-only behaviour-shift detector (layer 5). Uses ONLY the current session's prefix
(item ids + inter-click gaps) and statistics fitted on TRAIN. Clock time (hour/weekday) is NOT used."""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

FEATURES = ["trans_entropy", "trans_maxp", "cat_changed_in_prefix", "log_n", "log_last_gap", "log_last_pop"]


class PrefixStats:
    """Per-item statistics from the TRAIN first-order transition matrix."""
    def __init__(self, T1, pop_counts, item_cat):
        M = T1.copy().tocsr()
        M.data = -M.data * np.log(M.data)
        self.entropy = np.asarray(M.sum(1)).ravel()
        self.maxp = T1.max(axis=1).toarray().ravel()
        self.log_pop = np.log1p(pop_counts)
        self.item_cat = item_cat


def prefix_features(prefixes, ts, st: PrefixStats) -> np.ndarray:
    last = np.array([p[-1] for p in prefixes])
    X = np.zeros((len(prefixes), len(FEATURES)))
    X[:, 0] = st.entropy[last]
    X[:, 1] = st.maxp[last]
    for i, (p, t) in enumerate(zip(prefixes, ts)):
        if len(p) >= 2:
            a, b = st.item_cat[p[-1]], st.item_cat[p[-2]]
            X[i, 2] = float(a >= 0 and b >= 0 and a != b)
            X[i, 4] = np.log1p(max(int(t[-1]) - int(t[-2]), 0))
    X[:, 3] = np.log([len(p) for p in prefixes])
    X[:, 5] = st.log_pop[last]
    return X


class ShiftDetector:
    def __init__(self, C=1.0):
        self.clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=1000))

    def fit(self, X, y):
        self.clf.fit(X, y)
        return self

    def prob(self, X):
        return self.clf.predict_proba(X)[:, 1]
