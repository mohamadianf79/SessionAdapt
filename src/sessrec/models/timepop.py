"""Time-conditioned popularity (+ controls) used to test 'clock time has no direct effect'.

  TimePop   : p_bin(j) = (c_bin[j] + alpha * p_glob(j)) / (N_bin + alpha)   (Dirichlet smoothing toward global)
  mode='global'   : ignores time  -> control: any gain must come from time, not from 'adding a popularity prior'
  mode='shuffled' : time bins of TRAIN events are randomly permuted -> noise floor of the whole procedure
  TimeBlend : base_score + w * p_bin / max(p_glob)            (w=0 -> exactly the base model)
Both need the 'current time' of the request: instances carry t_next (= timestamp of the target click)."""
from __future__ import annotations
import numpy as np
from .base import Recommender
from ..data.timebins import time_bin, N_BINS


class TimePop(Recommender):
    uses_t_next = True

    def __init__(self, scheme="daypart4_wkend", alpha=100.0, mode="time", seed=0):
        assert mode in ("time", "global", "shuffled")
        self.scheme, self.alpha, self.mode, self.seed = scheme, alpha, mode, seed
        self.name = f"timepop({scheme},alpha={alpha},{mode})"

    def fit(self, train_items, n_items, train_ts=None):
        items = np.concatenate(train_items)
        bins = time_bin(np.concatenate(train_ts), self.scheme)
        if self.mode == "shuffled":
            bins = np.random.default_rng(self.seed).permutation(bins)
        B = N_BINS[self.scheme]
        C = np.zeros((B, n_items)); np.add.at(C, (bins, items), 1.0)
        glob = C.sum(0) / C.sum()
        Nb = C.sum(1, keepdims=True)
        self.p_bin = (C + self.alpha * glob[None]) / (Nb + self.alpha)
        self.p_glob = glob
        self.scale = glob.max()
        return self

    def score(self, prefixes, ts=None, t_next=None):
        if self.mode == "global":
            return np.tile(self.p_glob, (len(prefixes), 1)) / self.scale
        return self.p_bin[time_bin(t_next, self.scheme)] / self.scale


class TimeBlend(Recommender):
    uses_t_next = True

    def __init__(self, base, timepop, w):
        self.base, self.tp, self.w = base, timepop, w
        self.name = f"{base.name}+{w}*{timepop.name}"

    def score(self, prefixes, ts=None, t_next=None):
        S = self.base.score(prefixes, ts)
        if self.w == 0:
            return S
        return S + self.w * self.tp.score(prefixes, ts, t_next)
