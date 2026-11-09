"""Collect results/*.csv into paper-ready markdown tables  ->  results/tables.md   (nothing is recomputed).
Usage: python scripts/make_tables.py yoochoose_f64 yoochoose_f8 diginetica"""
import json, sys
from pathlib import Path
import pandas as pd

R = Path("results")


def md(df, floatfmt="{:.4f}"):
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_float_dtype(df[c]):
            df[c] = df[c].map(lambda x: "" if pd.isna(x) else floatfmt.format(x))
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.astype(str).values) + "\n"


def read(name, pat):
    f = R / pat.format(name=name)
    return pd.read_csv(f) if f.exists() else None


out = []
for name in sys.argv[1:]:
    out.append(f"\n# {name}\n")
    mf = Path(f"data/processed/{name}/meta.json")
    if mf.exists():
        m = json.loads(mf.read_text())
        keep = ["raw_sessions", "raw_events", "sessions_after_len_and_recent_filter", "dropped_straddling_sessions", "n_items",
                "train_sessions", "train_events", "val_sessions", "val_events", "test_sessions", "test_events",
                "train_events_removed_oov", "val_events_removed_oov", "test_events_removed_oov"]
        out.append("## Data\n" + md(pd.DataFrame([{k: m[k] for k in keep if k in m}]).T.reset_index().rename(columns={"index": "stat", 0: "value"})))
    parts = []
    b = read(name, "{name}_baselines_test.csv")
    if b is not None:
        parts.append(b[["model", "params", "HR@20", "HR@20_lo", "HR@20_hi", "MRR@20", "MRR@20_lo", "MRR@20_hi"]])
    for nm in ("gru", "sasrec"):
        n = read(name, "{name}_" + nm + "_test.csv")
        if n is not None:
            parts.append(n[["model", "hp", "HR@20", "HR@20_lo", "HR@20_hi", "MRR@20", "MRR@20_lo", "MRR@20_hi"]].rename(columns={"hp": "params"}))
    if parts:
        out.append("## Next-item accuracy (TEST)\n" + md(pd.concat(parts, ignore_index=True)))
    c = read(name, "{name}_neural_vs_markov_test.csv")
    if c is not None:
        out.append("## Neural vs Markov (paired, TEST)\n" + md(c))
    for title, pat in [("Shift diagnosis (VAL, diagnostic)", "{name}_shift_diagnosis_val.csv"), ("Shift detector AUC (VAL, grouped CV)", "{name}_detector_auc_val.csv"),
                       ("Score-mixing adaptive rules (TEST)", "{name}_adaptive_test.csv"), ("Slot reservation grid (VAL)", "{name}_slots_val.csv"),
                       ("Category back-off ablation (TEST)", "{name}_backoff_test.csv"), ("Adaptation-curve summary (TEST)", "{name}_curve_summary_test.csv"),
                       ("Adaptation curve (TEST, cohort)", "{name}_curve_test.csv")]:
        d = read(name, pat)
        if d is not None:
            out.append(f"## {title}\n" + md(d))
    for f in sorted(R.glob(f"{name}_time_*_test.csv")):
        out.append(f"## Time hypothesis: {f.stem}\n" + md(pd.read_csv(f).drop(columns=["ci90"], errors="ignore")))
Path("results/tables.md").write_text("\n".join(out))
print("written results/tables.md")
