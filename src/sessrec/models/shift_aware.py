"""Shift-aware session model = lag-Markov base + (i) gated switch to a short-memory model
+ (ii) gated category-transition prior.
   g      = detector probability that the NEXT click changes category   (prefix only)
   score  = (1 - w*g) * S_long + w*g * S_short + mu * g * S_cat
   S_cat(j) = P(cat(j) | cat(last)) * P(j | cat(j))       (two-stage generative prior, TRAIN statistics)
 w=0, mu=0  -> exactly the base model (ablation).  theta (optional) turns g into a hard gate 1[g>theta]."""
from __future__ import annotations
import numpy as np
from .base import Recommender
from ..analysis.detector import prefix_features


class CategoryPrior:
    def __init__(self, train_items, item_cat, pop_counts, n_cat=13, alpha=1.0):
        self.item_cat = item_cat
        self.n_cat = n_cat
        C = np.zeros((n_cat, n_cat))
        for s in train_items:
            c = item_cat[s]
            a, b = c[:-1], c[1:]
            ok = (a >= 0) & (b >= 0)
            np.add.at(C, (a[ok], b[ok]), 1.0)
        C += alpha
        self.C = C / C.sum(1, keepdims=True)                 # P(next cat | last cat)
        self.marg = C.sum(0) / C.sum()                       # fallback when last cat unknown
        known = item_cat >= 0
        tot = np.bincount(item_cat[known], weights=pop_counts[known], minlength=n_cat) + 1e-9
        self.p_item_in_cat = np.where(known, pop_counts / tot[np.where(known, item_cat, 0)], 0.0)

    def score(self, last_items):
        cl = self.item_cat[last_items]
        rowp = np.where((cl >= 0)[:, None], self.C[np.where(cl >= 0, cl, 0)], self.marg[None, :])
        ic = np.where(self.item_cat >= 0, self.item_cat, 0)
        S = rowp[:, ic] * self.p_item_in_cat[None, :]
        S[:, self.item_cat < 0] = 0.0
        return S


class ShiftAware(Recommender):
    def __init__(self, long_model, short_model, detector, stats, cat_prior, w=0.5, mu=0.1, theta=None):
        self.long, self.short, self.det, self.st, self.cp = long_model, short_model, detector, stats, cat_prior
        self.w, self.mu, self.theta = w, mu, theta
        self.name = f"shift_aware(w={w},mu={mu},theta={theta})"

    def gate(self, prefixes, ts):
        g = self.det.prob(prefix_features(prefixes, ts, self.st))
        return (g > self.theta).astype(float) if self.theta is not None else g

    def score(self, prefixes, ts=None):
        S = self.long.score(prefixes, ts)
        if self.w == 0 and self.mu == 0:
            return S
        g = self.gate(prefixes, ts)[:, None]
        out = (1 - self.w * g) * S
        if self.w:
            out += self.w * g * self.short.score(prefixes, ts)
        if self.mu:
            out += self.mu * g * self.cp.score(np.array([p[-1] for p in prefixes]))
        return out
