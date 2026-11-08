import numpy as np
from sessrec.models.base import Recommender
from sessrec.models.slots import SlotReserve


class FixedBase(Recommender):
    def score(self, p, ts=None):
        S = np.tile(np.linspace(1, 0, 50), (len(p), 1))   # item i has rank i+1
        return S


class FixedPrior:
    def score(self, last):
        C = np.zeros((len(last), 50)); C[:, 40] = 0.9; C[:, 41] = 0.5; C[:, 3] = 1.0  # item 3 is already in base top-K
        return C


def _rank_list(S, k):
    return list(np.argsort(-S[k], kind="stable")[:10])


def test_slots_replace_tail_with_new_prior_items_and_respect_gate():
    P = [np.array([1]), np.array([2])]
    g = lambda P_, T_: np.array([0.9, 0.0])           # row 0 gated open, row 1 closed
    m = SlotReserve(FixedBase(), g, FixedPrior(), m=2, theta=0.3, K=10)
    S = m.score(P, None)
    assert _rank_list(S, 0) == [0, 1, 2, 3, 4, 5, 6, 7, 40, 41]   # top-8 kept in order, then the 2 NEW items
    assert _rank_list(S, 1) == list(range(10))                    # closed gate -> unchanged
    assert 3 in _rank_list(S, 0)                                  # in-top item not duplicated


def test_m0_is_identity():
    P = [np.array([1])]
    S = SlotReserve(FixedBase(), lambda a, b: np.ones(1), FixedPrior(), m=0).score(P, None)
    assert np.allclose(S, FixedBase().score(P))
