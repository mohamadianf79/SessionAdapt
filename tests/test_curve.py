import numpy as np
from sessrec.eval.protocol import make_instances, ranks_of_targets
from sessrec.eval.curve import adaptation_curve, population_curve
from sessrec.models.popularity import Pop
from sessrec.models.markov import LagMarkov


def _data(seed=0, n_sess=3000):
    rng = np.random.default_rng(seed)
    succ = {i: rng.choice(30, 2, replace=False) for i in range(30)}
    out = []
    for _ in range(n_sess):
        it = int(rng.integers(30)); s = [it]
        for _ in range(8):
            it = int(rng.choice(succ[it])); s.append(it)
        out.append(np.array(s))
    return out


def test_start0_includes_empty_prefix_and_models_handle_it():
    ses = _data(); tr, te = ses[:2000], ses[2000:]
    inst = make_instances(te, [np.arange(len(s)) for s in te], start=0)
    assert (inst.n == 0).sum() == len(te) and len(inst.items[0]) == 0
    r = ranks_of_targets(LagMarkov(L=2).fit(tr, 30), inst)
    assert r.min() >= 1 and r.max() <= 30


def test_curve_improves_with_first_click_and_cohort_is_fixed():
    ses = _data(); tr, te = ses[:2000], ses[2000:]
    inst = make_instances(te, [np.arange(len(s)) for s in te], start=0)
    r = ranks_of_targets(LagMarkov(L=2).fit(tr, 30), inst)
    c, s = adaptation_curve(r, inst, K=5, N=5, n_boot=100)
    hr = c[c.metric == "HR@5"].set_index("n")["value"]
    assert hr[1] > hr[0] + 0.2                       # first click buys a lot on Markov data
    assert (c["cohort_sessions"] == len(te)).all()   # all sessions have length 9 >= N+1
    summ = s[s.metric == "HR@5"].set_index("summary")["value"]
    assert 0 < summ["A_adapt"] <= 1 and summ["n_q"] >= 1
    p = population_curve(r, inst, K=5, N=5)
    assert len(p) > 0
