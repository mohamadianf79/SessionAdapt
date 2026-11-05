import numpy as np


class Recommender:
    """score() must return a (B, n_items) float array: higher = better. No state from the
    evaluation set may be used; fit() sees TRAIN sessions only."""
    name = "base"

    def fit(self, train_items: list, n_items: int, train_ts: list | None = None):
        raise NotImplementedError

    def score(self, prefixes: list, ts: list | None = None) -> np.ndarray:
        raise NotImplementedError
