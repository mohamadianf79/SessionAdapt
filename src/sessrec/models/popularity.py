import numpy as np
from .base import Recommender


class Pop(Recommender):
    name = "pop"

    def fit(self, train_items, n_items, train_ts=None):
        self.n_items = n_items
        c = np.bincount(np.concatenate(train_items), minlength=n_items).astype(np.float64)
        self.pop = c / (c.max() + 1.0)   # strictly < 1
        return self

    def score(self, prefixes, ts=None):
        return np.tile(self.pop, (len(prefixes), 1))


class SPop(Pop):
    """Items already in the session first (by in-session count), then global popularity."""
    name = "spop"

    def score(self, prefixes, ts=None):
        S = super().score(prefixes)
        for i, p in enumerate(prefixes):
            u, c = np.unique(p, return_counts=True)
            S[i, u] += c
        return S
