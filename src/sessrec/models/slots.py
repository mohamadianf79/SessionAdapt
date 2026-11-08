"""Slot reservation (C2c, last adaptive rule): when the gate says 'shift likely', keep the top K-q items of the
base model and fill q <= m reserved slots with the best items of the category-transition prior that are NOT
already in the base top-K.  Rows where the gate is closed are returned unchanged.  m=0 -> identical to base."""
from __future__ import annotations
import numpy as np
from .base import Recommender


class SlotReserve(Recommender):
    def __init__(self, base, gate_fn, cat_prior, m=2, theta=0.3, K=20):
        self.base, self.gate_fn, self.cp, self.m, self.theta, self.K = base, gate_fn, cat_prior, m, theta, K
        self.name = f"slots(m={m},theta={theta})"

    def score(self, prefixes, ts=None):
        S = self.base.score(prefixes, ts)
        if self.m == 0:
            return S
        g = self.gate_fn(prefixes, ts)
        rows = np.flatnonzero(g > self.theta)
        if len(rows) == 0:
            return S
        S = S.copy()
        C = self.cp.score(np.array([prefixes[i][-1] for i in rows]))
        K, m = self.K, self.m
        for k, i in enumerate(rows):
            top = np.argpartition(-S[i], K)[:K]
            top = top[np.argsort(-S[i, top])]            # base top-K, best first
            c = C[k].copy(); c[top] = 0.0                # reserved slots must be NEW items
            cand = np.argpartition(-c, m)[:m]
            cand = cand[c[cand] > 0]
            q = len(cand)
            keep = top[: K - q]
            S[i, keep] += 1000.0                         # keeps their relative order, puts them first
            S[i, cand] = 500.0 + c[cand]                 # then the reserved items, by prior score
        return S
