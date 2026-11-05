"""In-session behaviour-shift labels (used ONLY to evaluate/diagnose, never as model input).

Definition (document this in the paper):
  an instance is a SHIFT if the target item's category differs from the category of the last item
  of the prefix.  Item category = most frequent real category (1..12) among TRAIN events of that item;
  items without a real category are 'unknown' and their instances are excluded from shift/no-shift
  comparisons (coverage is reported).
Label: 1 = shift, 0 = same category, -1 = unknown.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def item_category_map(train_df: pd.DataFrame, n_items: int, valid=range(1, 13)) -> np.ndarray:
    d = train_df[train_df["cat"].isin(list(valid))]
    mode = d.groupby("item")["cat"].agg(lambda s: s.value_counts().idxmax())
    out = np.full(n_items, -1, dtype=np.int64)
    out[mode.index.to_numpy()] = mode.to_numpy()
    return out


def shift_labels(inst, item_cat: np.ndarray) -> np.ndarray:
    last = np.array([p[-1] for p in inst.items])
    a, b = item_cat[last], item_cat[inst.target]
    lab = np.where((a < 0) | (b < 0), -1, (a != b).astype(int))
    return lab
