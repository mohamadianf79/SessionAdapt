import numpy as np
from sessrec.data.timebins import time_bin
from sessrec.eval.protocol import make_instances, ranks_of_targets
from sessrec.eval.metrics import per_instance, verdict
from sessrec.models.popularity import Pop
from sessrec.models.timepop import TimePop


def test_time_bins_known_dates():
    # 1970-01-01 00:00 UTC was a Thursday (weekday) -> daypart 0, not weekend
    assert time_bin([0], "daypart4_wkend")[0] == 0
    assert time_bin([7 * 3600], "daypart4_wkend")[0] == 1
    assert time_bin([2 * 86400 + 13 * 3600], "daypart4_wkend")[0] == 2 + 4     # Saturday 13:00 -> afternoon, weekend
    assert time_bin([2 * 86400], "dow7")[0] == 5 and time_bin([4 * 86400], "dow7")[0] == 0   # Sat=5, Mon=0


def _world(time_matters, n_sess=6000, seed=0):
    """Single-click sessions over 10 days; if time_matters, items 0-9 are popular in the morning, 10-19 in the evening."""
    rng = np.random.default_rng(seed)
    items, ts = [], []
    for s in range(n_sess):
        t = int(rng.integers(0, 10 * 86400)); hour = (t // 3600) % 24
        if time_matters:
            pool = np.arange(0, 10) if hour < 12 else np.arange(10, 20)
            it = int(rng.choice(pool)) if rng.random() < 0.9 else int(rng.integers(0, 40))
        else:
            it = int(rng.integers(0, 40))
        items.append(np.array([it, int(rng.integers(0, 40))])); ts.append(np.array([t, t + 5]))
    return items, ts


def _hr(model, tr_i, tr_t, te_i, te_t):
    inst = make_instances(te_i, te_t, start=0)
    m = model.fit(tr_i, 40, tr_t) if isinstance(model, TimePop) else model.fit(tr_i, 40)
    return per_instance(ranks_of_targets(m, inst), 5)["HR@5"][inst.n == 0].mean()


def test_positive_control_detects_time_and_null_world_does_not():
    for matters in (True, False):
        items, ts = _world(matters)
        tr_i, tr_t, te_i, te_t = items[:4500], ts[:4500], items[4500:], ts[4500:]
        g = _hr(Pop(), tr_i, tr_t, te_i, te_t)
        t = _hr(TimePop("daypart4_wkend", 10.0), tr_i, tr_t, te_i, te_t)
        sh = _hr(TimePop("daypart4_wkend", 10.0, "shuffled"), tr_i, tr_t, te_i, te_t)
        if matters:
            assert t > g + 0.1 and t > sh + 0.1
        else:
            assert abs(t - g) < 0.05


def test_verdict_strings():
    assert verdict(0.01, 0.005, 0.015, 0.006, 0.014, 0.02).startswith("GAIN")
    assert "EQUIVALENT" in verdict(0.0, -0.002, 0.002, -0.0015, 0.0015, 0.005) and "NOT" not in verdict(0.0, -0.002, 0.002, -0.0015, 0.0015, 0.005)
    assert "NOT shown" in verdict(0.0, -0.02, 0.02, -0.015, 0.015, 0.005)
