import numpy as np
import pytest
torch = pytest.importorskip("torch")
from sessrec.eval.protocol import make_instances, ranks_of_targets
from sessrec.eval.metrics import per_instance
from sessrec.models.neural import NeuralRec
from sessrec.models.popularity import Pop


def _data(seed=0, n_sess=2500, n_items=30):
    rng = np.random.default_rng(seed)
    succ = {i: rng.choice(n_items, 2, replace=False) for i in range(n_items)}
    out = []
    for _ in range(n_sess):
        it = int(rng.integers(n_items)); s = [it]
        for _ in range(6):
            it = int(rng.choice(succ[it])); s.append(it)
        out.append(np.array(s))
    return out


@pytest.mark.parametrize("kind,hp", [
    ("gru", dict(emb=32, hidden=64, dropout=0.0, lr=3e-3)),
    ("sasrec", dict(d=32, layers=1, heads=2, dropout=0.1, lr=3e-3)),
])
def test_neural_learns_markov_structure_and_handles_empty_prefix(kind, hp):
    ses = _data(); tr, te = ses[:2000], ses[2000:]
    inst = make_instances(te, [np.arange(len(s)) for s in te], start=0)   # includes n=0
    m = NeuralRec(kind, hp, seed=0, max_epochs=6, batch_size=128, device="cpu", verbose=False).fit(tr, 30)
    r = ranks_of_targets(m, inst)
    assert r.min() >= 1 and r.max() <= 30
    mrr = per_instance(r[inst.n >= 1], 5)["MRR@5"].mean()
    pop = per_instance(ranks_of_targets(Pop().fit(tr, 30), inst)[inst.n >= 1], 5)["MRR@5"].mean()
    assert mrr > pop + 0.2
