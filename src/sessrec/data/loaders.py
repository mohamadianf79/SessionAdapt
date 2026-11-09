"""Raw-file loaders. Every loader returns a DataFrame with columns
   session_id (int64), item_id (int64), ts (int64, unix seconds), pos (int64, original row order)
   and optionally cat (int64, event-level category if the dataset has one). All filtering / splitting happens later in prepare.py."""
from __future__ import annotations
import pandas as pd


def load_yoochoose(path: str) -> pd.DataFrame:
    # clicks.dat: session_id, timestamp (ISO, UTC), item_id, category
    df = pd.read_csv(path, header=None, usecols=[0, 1, 2, 3],
                     names=["session_id", "ts", "item_id", "cat"],
                     dtype={"session_id": "int64", "ts": "string", "item_id": "int64", "cat": "string"})
    dt = pd.to_datetime(df["ts"], format="%Y-%m-%dT%H:%M:%S.%fZ")
    df["ts"] = (dt - pd.Timestamp("1970-01-01")).dt.total_seconds().astype("int64")
    # cat: 'S' = special offer -> -1 ; 0 = missing ; 1..12 = real category ; >12 = brand id (not a category)
    df["cat"] = pd.to_numeric(df["cat"], errors="coerce").fillna(-1).astype("int64")
    df["pos"] = range(len(df))
    return df[["session_id", "item_id", "ts", "pos", "cat"]]


def load_diginetica(path: str) -> pd.DataFrame:
    """CIKM Cup 2016 (Diginetica) train-item-views.csv  [sessionId;userId;itemId;timeframe;eventdate].
    ASSUMPTIONS (state them in the paper): `timeframe` = ms since the session start (it only orders/spaces clicks inside
    a session); `eventdate` has DAY resolution only.  => ts = day_start + timeframe/1000.  Day-of-week is valid,
    hour-of-day is NOT -> use scheme 'dow7' for the time hypothesis on this dataset.
    Category = categoryId from product-categories.csv in the same folder (item->category, static)."""
    from pathlib import Path
    df = pd.read_csv(path, sep=";", dtype=str)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=",", dtype=str)
    cols = {c.lower().replace("_", ""): c for c in df.columns}
    need = {"sessionid", "itemid", "timeframe", "eventdate"}
    if not need <= set(cols):
        raise ValueError(f"unexpected columns {list(df.columns)}; need {sorted(need)}")
    out = pd.DataFrame({
        "session_id": pd.to_numeric(df[cols["sessionid"]]).astype("int64"),
        "item_id": pd.to_numeric(df[cols["itemid"]]).astype("int64"),
    })
    day = (pd.to_datetime(df[cols["eventdate"]], format="%Y-%m-%d") - pd.Timestamp("1970-01-01")).dt.total_seconds()
    out["ts"] = (day + pd.to_numeric(df[cols["timeframe"]]) // 1000).astype("int64")
    out["pos"] = range(len(out))
    catf = Path(path).with_name("product-categories.csv")
    if catf.exists():
        c = pd.read_csv(catf, sep=";", dtype=str)
        if c.shape[1] == 1:
            c = pd.read_csv(catf, sep=",", dtype=str)
        cc = {x.lower().replace("_", ""): x for x in c.columns}
        m = dict(zip(pd.to_numeric(c[cc["itemid"]]).astype("int64"), pd.to_numeric(c[cc["categoryid"]]).astype("int64")))
        out["cat"] = out["item_id"].map(m).fillna(-1).astype("int64")
    return out


LOADERS = {"yoochoose": load_yoochoose, "diginetica": load_diginetica}


def load_raw(name: str, path: str) -> pd.DataFrame:
    if name not in LOADERS:
        raise KeyError(f"no loader '{name}' (have: {list(LOADERS)})")
    return LOADERS[name](path)
