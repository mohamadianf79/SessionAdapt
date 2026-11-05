import subprocess, sys
import numpy as np, pandas as pd
from pathlib import Path
from sessrec.data.prepare import build_splits, to_sessions
from sessrec.eval.metrics import per_instance, mean_ci
from sessrec.eval.protocol import make_instances, ranks_of_targets
from sessrec.models.popularity import Pop
from sessrec.models.markov import LagMarkov

ROOT = Path(__file__).resolve().parents[1]


def fake_df(n_sess=3000, seed=0):
    rng = np.random.default_rng(seed)
    rows, pos = [], 0
    for s in range(n_sess):
        t0 = s * 100 + int(rng.integers(0, 50))
        L = int(rng.integers(2, 8))
        for k in range(L):
            rows.append((s, int(rng.integers(0, 50)), t0 + k * 10, pos)); pos += 1
    return pd.DataFrame(rows, columns=["session_id", "item_id", "ts", "pos"])


def test_no_temporal_leakage_and_train_only_vocab():
    out, st = build_splits(fake_df(), val_frac=0.15, test_frac=0.15, min_item_support=3)
    assert out["train"]["ts"].max() < out["val"]["ts"].min() <= out["val"]["ts"].max() < out["test"]["ts"].min()
    s = [set(out[k]["session_id"]) for k in out]
    assert not (s[0] & s[1]) and not (s[0] & s[2]) and not (s[1] & s[2])
    assert out["val"]["item"].max() < st["n_items"] and out["test"]["item"].max() < st["n_items"]
    assert st["dropped_straddling_sessions"] >= 0


def test_rank_ties_are_pessimistic():
    class Const:
        def score(self, p, ts=None):
            return np.zeros((len(p), 10))
    inst = make_instances([np.array([1, 2, 3])], [np.array([0, 1, 2])])
    r = ranks_of_targets(Const(), inst)
    assert (r == 10).all()          # all tied -> worst rank, not 1


def test_metrics_values():
    v = per_instance(np.array([1, 2, 30]), 20)
    assert v["HR@20"].tolist() == [1, 1, 0]
    assert np.isclose(v["MRR@20"][1], 0.5) and np.isclose(v["NDCG@20"][0], 1.0)
    mu, lo, hi = mean_ci(v["HR@20"], np.array([0, 0, 1]), n_boot=200)
    assert lo <= mu <= hi


def test_markov_beats_pop_on_markov_data():
    rng = np.random.default_rng(1)
    succ = {i: rng.choice(30, 2, replace=False) for i in range(30)}
    sessions = []
    for _ in range(2000):
        it = int(rng.integers(30)); s = [it]
        for _ in range(6):
            it = int(rng.choice(succ[it])); s.append(it)
        sessions.append(np.array(s))
    tr, te = sessions[:1500], sessions[1500:]
    inst = make_instances(te, [np.arange(len(s)) for s in te])
    m = lambda mod: per_instance(ranks_of_targets(mod.fit(tr, 30), inst), 5)["MRR@5"].mean()
    assert m(LagMarkov(L=1)) > m(Pop()) + 0.2


def test_end_to_end_smoke(tmp_path):
    py = sys.executable
    run = lambda *a: subprocess.run([py, *a], cwd=ROOT, check=True, capture_output=True, text=True)
    run("scripts/make_fake_yoochoose.py")
    run("scripts/prepare_data.py", "configs/smoke.yaml")
    r = run("scripts/run_baselines.py", "configs/smoke.yaml", "--final")
    assert "[TEST] markov" in r.stdout


def test_shift_labels():
    from sessrec.analysis.shift import shift_labels
    from sessrec.eval.protocol import Instances
    cat = np.array([1, 1, 2, -1])
    inst = Instances([np.array([0]), np.array([0]), np.array([0])], [None] * 3,
                     np.array([1, 2, 3]), np.array([1, 1, 1]), np.array([0, 1, 2]))
    assert shift_labels(inst, cat).tolist() == [0, 1, -1]


def test_shift_aware_reduces_to_base_and_prior_is_distribution():
    from sessrec.models.shift_aware import ShiftAware, CategoryPrior
    rng = np.random.default_rng(0)
    sess = [rng.integers(0, 20, 6) for _ in range(300)]
    cat = np.arange(20) % 4
    pc = np.bincount(np.concatenate(sess), minlength=20).astype(float)
    cp = CategoryPrior(sess, cat, pc, n_cat=4)
    assert np.allclose(cp.C.sum(1), 1)
    S = cp.score(np.array([0, 1]))
    assert S.shape == (2, 20) and (S >= 0).all()
    base = LagMarkov(L=2).fit(sess, 20)
    m = ShiftAware(base, base, None, None, cp, w=0, mu=0)
    P = [np.array([1, 2]), np.array([3])]
    assert np.allclose(m.score(P), base.score(P))
