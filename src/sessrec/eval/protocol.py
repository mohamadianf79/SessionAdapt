"""Evaluation protocol: every position t>=1 of every eval session is one instance
(prefix = items[:t], target = items[t]); ranking is over the FULL catalogue.
Ties are resolved pessimistically (target ranked below all items with an equal score), so a
model that outputs many zeros cannot gain from arbitrary tie-breaking."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Instances:
    items: list      # prefix item arrays (views)
    ts: list         # prefix timestamp arrays (views)
    target: np.ndarray
    n: np.ndarray    # prefix length
    sess: np.ndarray # session index (0..S-1) for cluster bootstrap

    def __len__(self):
        return len(self.target)


def make_instances(items, ts, max_sessions=None, seed=0) -> Instances:
    idx = np.arange(len(items))
    if max_sessions is not None and len(idx) > max_sessions:
        idx = np.sort(np.random.default_rng(seed).choice(idx, max_sessions, replace=False))
    P, T, tg, n, s = [], [], [], [], []
    for new_id, i in enumerate(idx):
        a, t = items[i], ts[i]
        for k in range(1, len(a)):
            P.append(a[:k]); T.append(t[:k]); tg.append(a[k]); n.append(k); s.append(new_id)
    return Instances(P, T, np.array(tg), np.array(n), np.array(s))


def ranks_of_targets(model, inst: Instances, batch_size=512) -> np.ndarray:
    out = np.empty(len(inst), dtype=np.int64)
    for a in range(0, len(inst), batch_size):
        b = min(a + batch_size, len(inst))
        S = model.score(inst.items[a:b], inst.ts[a:b])
        tgt = S[np.arange(b - a), inst.target[a:b]][:, None]
        greater = (S > tgt).sum(1)
        equal = (S == tgt).sum(1) - 1
        out[a:b] = 1 + greater + equal
    return out
