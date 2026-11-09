import numpy as np, pandas as pd
from sessrec.data.loaders import load_diginetica
from sessrec.data.prepare import build_splits
from sessrec.analysis.shift import item_category_map


def test_diginetica_loader_split_and_categories(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    for s in range(1, 1201):
        day = pd.Timestamp("2016-05-01") + pd.Timedelta(days=int(s // 50))
        t = 0
        for _ in range(int(rng.integers(2, 7))):
            t += int(rng.integers(1000, 60000))
            rows.append((s, "NA", int(rng.integers(1, 60)), t, day.strftime("%Y-%m-%d")))
    pd.DataFrame(rows, columns=["sessionId", "userId", "itemId", "timeframe", "eventdate"]).to_csv(tmp_path / "train-item-views.csv", sep=";", index=False)
    pd.DataFrame({"itemId": range(1, 60), "categoryId": [100 + i % 5 for i in range(1, 60)]}).to_csv(tmp_path / "product-categories.csv", sep=";", index=False)
    df = load_diginetica(str(tmp_path / "train-item-views.csv"))
    assert {"session_id", "item_id", "ts", "pos", "cat"} <= set(df.columns) and (df["cat"] >= 100).all()
    out, st = build_splits(df, val_frac=0.15, test_frac=0.15, min_item_support=3)
    assert out["train"]["ts"].max() < out["val"]["ts"].min() and out["val"]["ts"].max() < out["test"]["ts"].min()
    cat = item_category_map(out["train"], st["n_items"], None)
    assert (cat >= 0).mean() > 0.9 and cat.max() <= 4


def test_category_prior_with_many_category_ids(tmp_path):
    from sessrec.models.shift_aware import CategoryPrior
    rng = np.random.default_rng(0)
    cat = np.array([1003, 1003, 77, 5000, -1, 77])      # raw ids in the thousands
    df = pd.DataFrame({"item": np.arange(6), "cat": cat})
    m = item_category_map(df, 6, None)
    assert m.tolist() == [1, 1, 0, 2, -1, 0] and m.max() == 2
    sess = [rng.integers(0, 6, 5) for _ in range(50)]
    cp = CategoryPrior(sess, m, np.ones(6))
    assert cp.score(np.array([0, 4])).shape == (2, 6)
