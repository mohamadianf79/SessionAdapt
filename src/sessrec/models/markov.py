import numpy as np
import scipy.sparse as sp
from .base import Recommender


class LagMarkov(Recommender):
    """score(j) = sum_{l=1..L} decay^(l-1) * P(next=j | item at lag l). L=1 is a first-order Markov chain."""
    def __init__(self, L=3, decay=0.7, pop_eps=1e-6):
        self.L, self.decay, self.pop_eps = L, decay, pop_eps
        self.name = f"markov(L={L},decay={decay})"

    def fit(self, train_items, n_items, train_ts=None):
        self.n_items = n_items
        pop = np.bincount(np.concatenate(train_items), minlength=n_items).astype(np.float64)
        self.pop = pop / (pop.max() + 1.0)
        self.T = []
        for l in range(1, self.L + 1):
            src = np.concatenate([s[:-l] for s in train_items if len(s) > l] or [np.empty(0, int)])
            dst = np.concatenate([s[l:] for s in train_items if len(s) > l] or [np.empty(0, int)])
            M = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(n_items, n_items)).tocsr()
            d = np.asarray(M.sum(1)).ravel(); d[d == 0] = 1
            self.T.append(sp.diags(1.0 / d) @ M)
        return self

    def score(self, prefixes, ts=None):
        B = len(prefixes)
        S = np.zeros((B, self.n_items))
        for l in range(1, self.L + 1):
            rows = [i for i, p in enumerate(prefixes) if len(p) >= l]
            if not rows:
                break
            last = np.array([prefixes[i][-l] for i in rows])
            S[rows] += self.decay ** (l - 1) * self.T[l - 1][last].toarray()
        return S + self.pop_eps * self.pop
