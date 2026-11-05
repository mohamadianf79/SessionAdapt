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


LOADERS = {"yoochoose": load_yoochoose}


def load_raw(name: str, path: str) -> pd.DataFrame:
    if name not in LOADERS:
        raise KeyError(f"no loader '{name}' (have: {list(LOADERS)})")
    return LOADERS[name](path)
