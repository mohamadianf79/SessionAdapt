"""Raw clicks -> train/val/test sessions with a leakage-free chronological split.

Rules (all documented in the paper's protocol section):
 1. Sessions are ordered by START time. Optionally keep only the most recent `recent_fraction`.
 2. Split boundaries are time quantiles of session start times.
      train: session END   <  t_val
      val  : session START >= t_val  and END < t_test
      test : session START >= t_test
    Sessions straddling a boundary are DROPPED, so no session is split across sets.
 3. The item vocabulary (min_item_support) is built from TRAIN ONLY. Items unseen in train are
    removed from val/test sessions (the count is reported); sessions shorter than
    `min_session_len` after that are dropped.
 4. Item ids are re-indexed to 0..n_items-1.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd


def build_splits(df: pd.DataFrame, *, recent_fraction=1.0, val_frac=0.1, test_frac=0.1,
                 min_item_support=5, min_session_len=2, max_session_len=200):
    df = df.sort_values(["session_id", "ts", "pos"], kind="stable")
    g = df.groupby("session_id")["ts"]
    sess = pd.DataFrame({"start": g.min(), "end": g.max(), "len": g.size()})
    stats = {"raw_sessions": int(len(sess)), "raw_events": int(len(df))}
    sess = sess[(sess["len"] >= min_session_len) & (sess["len"] <= max_session_len)]
    if recent_fraction < 1.0:
        n = max(1, int(len(sess) * recent_fraction))
        sess = sess.sort_values("start", kind="stable").iloc[-n:]
    stats["sessions_after_len_and_recent_filter"] = int(len(sess))

    t_val = sess["start"].quantile(1 - val_frac - test_frac)
    t_test = sess["start"].quantile(1 - test_frac)
    ids = {
        "train": sess.index[sess["end"] < t_val],
        "val": sess.index[(sess["start"] >= t_val) & (sess["end"] < t_test)],
        "test": sess.index[sess["start"] >= t_test],
    }
    stats["dropped_straddling_sessions"] = int(len(sess) - sum(len(v) for v in ids.values()))
    stats["t_val"], stats["t_test"] = int(t_val), int(t_test)

    def sub(split):
        return df[df["session_id"].isin(ids[split])]

    tr = sub("train")
    counts = tr["item_id"].value_counts()
    keep = np.sort(counts[counts >= min_item_support].index.to_numpy())
    vocab = pd.Series(np.arange(len(keep)), index=keep)
    stats["n_items"] = int(len(keep))

    out = {}
    for split in ["train", "val", "test"]:
        d = sub(split)
        n_before = len(d)
        d = d[d["item_id"].isin(vocab.index)].copy()
        stats[f"{split}_events_removed_oov"] = int(n_before - len(d))
        d["item"] = vocab.loc[d["item_id"]].to_numpy()
        d["len"] = d.groupby("session_id")["item"].transform("size")
        d = d[d["len"] >= min_session_len]
        cols = ["session_id", "item", "ts"] + (["cat"] if "cat" in d.columns else [])
        out[split] = d[cols].reset_index(drop=True)
        stats[f"{split}_sessions"] = int(d["session_id"].nunique())
        stats[f"{split}_events"] = int(len(d))
        stats[f"{split}_time_span"] = [int(d["ts"].min()), int(d["ts"].max())] if len(d) else None
    return out, stats


def to_sessions(d: pd.DataFrame):
    """DataFrame (sorted by session, time) -> (list of item arrays, list of ts arrays)."""
    d = d.sort_values(["session_id"], kind="stable")  # within-session order already fixed
    sid = d["session_id"].to_numpy()
    cut = np.flatnonzero(np.diff(sid)) + 1
    items = np.split(d["item"].to_numpy(), cut)
    ts = np.split(d["ts"].to_numpy(), cut)
    return items, ts


def save(out, stats, outdir: str):
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    for k, v in out.items():
        v.to_pickle(p / f"{k}.pkl")
    (p / "meta.json").write_text(json.dumps(stats, indent=2))


def load_processed(outdir: str):
    p = Path(outdir)
    data = {k: pd.read_pickle(p / f"{k}.pkl") for k in ["train", "val", "test"]}
    meta = json.loads((p / "meta.json").read_text())
    return data, meta
