import numpy as np
import scipy.sparse as sp
from .base import Recommender


class SKNN(Recommender):
    """Session-kNN: cosine similarity between the current prefix (as an item set) and train sessions;
    score(j) = sum of similarities of the k nearest sessions that contain j."""
    def __init__(self, k=100, pool=None, pop_eps=1e-6):
        self.k, self.pool, self.pop_eps = k, pool, pop_eps
        self.name = f"sknn(k={k})"

    def _binary(self, sessions, n_items):
        rows = np.concatenate([np.full(len(np.unique(s)), i) for i, s in enumerate(sessions)])
        cols = np.concatenate([np.unique(s) for s in sessions])
        return sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(sessions), n_items))

    def fit(self, train_items, n_items, train_ts=None):
        self.n_items = n_items
        sess = train_items[-self.pool:] if self.pool else train_items  # train sessions are time-ordered
        self.S = self._binary(sess, n_items)
        self.slen = np.asarray(self.S.sum(1)).ravel()
        self.ST = self.S.T.tocsr()
        pop = np.bincount(np.concatenate(train_items), minlength=n_items).astype(np.float64)
        self.pop = pop / (pop.max() + 1.0)
        return self

    def score(self, prefixes, ts=None):
        Q = self._binary(prefixes, self.n_items)
        qlen = np.asarray(Q.sum(1)).ravel()
        D = (Q @ self.ST).tocsr()          # B x n_train_sessions, co-occurrence counts
        r, c, v = [], [], []
        for i in range(D.shape[0]):
            a, b = D.indptr[i], D.indptr[i + 1]
            idx, val = D.indices[a:b], D.data[a:b]
            if len(idx) == 0:
                continue
            sim = val / np.sqrt(qlen[i] * self.slen[idx])
            if len(sim) > self.k:
                top = np.argpartition(-sim, self.k)[: self.k]
                idx, sim = idx[top], sim[top]
            r.append(np.full(len(idx), i)); c.append(idx); v.append(sim)
        if not r:
            return np.tile(self.pop_eps * self.pop, (len(prefixes), 1))
        W = sp.csr_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))),
                          shape=(D.shape[0], self.S.shape[0]))
        return (W @ self.S).toarray() + self.pop_eps * self.pop
