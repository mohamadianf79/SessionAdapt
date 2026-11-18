
"""Collect results/*.csv into paper-ready Markdown tables.

Nothing is recomputed. Reads existing CSV files and writes results/tables.md.

Usage:
    python scripts/make_tables.py yoochoose_f64 yoochoose_f8 diginetica
"""

import json
import sys
from pathlib import Path

import pandas as pd


RESULTS_DIR = Path("results")


def md(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Convert a DataFrame to a Markdown table."""
    df = df.copy()

    for column in df.columns:
        if pd.api.types.is_float_dtype(df[column]):
            df[column] = df[column].map(
                lambda value: "" if pd.isna(value) else floatfmt.format(value)
            )

    header = (
        "| " + " | ".join(map(str, df.columns)) + " |\n"
        + "|" + "---|" * len(df.columns) + "\n"
    )

    rows = [
        "| " + " | ".join(map(str, row)) + " |"
        for row in df.astype(str).values
    ]

    return header + "\n".join(rows) + "\n"


def read_csv(dataset: str, pattern: str):
    """Read a result CSV if it exists."""
    filepath = RESULTS_DIR / pattern.format(name=dataset)

    if not filepath.exists():
        return None

    return pd.read_csv(filepath)


def collect_dataset(dataset: str) -> list[str]:
    """Collect all available result tables for one dataset."""
    output = [f"\n# {dataset}\n"]

    # ---------------------------------------------------------
    # Dataset metadata
    # ---------------------------------------------------------
    metadata_file = Path(f"data/processed/{dataset}/meta.json")

    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))

        keys = [
            "raw_sessions",
            "raw_events",
            "sessions_after_len_and_recent_filter",
            "dropped_straddling_sessions",
            "n_items",
            "train_sessions",
            "train_events",
            "val_sessions",
            "val_events",
            "test_sessions",
            "test_events",
            "train_events_removed_oov",
            "val_events_removed_oov",
            "test_events_removed_oov",
        ]

        metadata_table = (
            pd.DataFrame(
                [{key: metadata[key] for key in keys if key in metadata}]
            )
            .T.reset_index()
            .rename(columns={"index": "stat", 0: "value"})
        )

        output.append("## Data\n" + md(metadata_table))

    # ---------------------------------------------------------
    # Next-item accuracy: baselines, GRU, SASRec
    # ---------------------------------------------------------
    accuracy_tables = []

    baseline = read_csv(dataset, "{name}_baselines_test.csv")

    if baseline is not None:
        columns = [
            "model",
            "params",
            "HR@20",
            "HR@20_lo",
            "HR@20_hi",
            "MRR@20",
            "MRR@20_lo",
            "MRR@20_hi",
        ]
        accuracy_tables.append(baseline[columns])

    for model_name in ("gru", "sasrec"):
        neural = read_csv(dataset, f"{{name}}_{model_name}_test.csv")

        if neural is not None:
            columns = [
                "model",
                "hp",
                "HR@20",
                "HR@20_lo",
                "HR@20_hi",
                "MRR@20",
                "MRR@20_lo",
                "MRR@20_hi",
            ]

            accuracy_tables.append(
                neural[columns].rename(columns={"hp": "params"})
            )

    if accuracy_tables:
        output.append(
            "## Next-item accuracy (TEST)\n"
            + md(pd.concat(accuracy_tables, ignore_index=True))
        )

    # ---------------------------------------------------------
    # Neural vs Markov comparison
    # ---------------------------------------------------------
    comparison = read_csv(dataset, "{name}_neural_vs_markov_test.csv")

    if comparison is not None:
        output.append(
            "## Neural vs Markov (paired, TEST)\n" + md(comparison)
        )

    # ---------------------------------------------------------
    # Other result tables
    # ---------------------------------------------------------
    other_tables = [
        (
            "Shift diagnosis (VAL, diagnostic)",
            "{name}_shift_diagnosis_val.csv",
        ),
        (
            "Shift detector AUC (VAL, grouped CV)",
            "{name}_detector_auc_val.csv",
        ),
        (
            "Score-mixing adaptive rules (TEST)",
            "{name}_adaptive_test.csv",
        ),
        (
            "Slot reservation grid (VAL)",
            "{name}_slots_val.csv",
        ),
        (
            "Category back-off ablation (TEST)",
            "{name}_backoff_test.csv",
        ),
        (
            "Adaptation curve (TEST, cohort)",
            "{name}_curve_test.csv",
        ),
    ]

    for title, pattern in other_tables:
        table = read_csv(dataset, pattern)

        if table is not None:
            output.append(f"## {title}\n" + md(table))

    # ---------------------------------------------------------
    # Adaptation-curve summaries:
    # baseline models + GRU + SASRec
    # ---------------------------------------------------------
    summary_tables = []

    for suffix in ("", "_gru", "_sasrec"):
        summary = read_csv(
            dataset,
            "{name}" + suffix + "_curve_summary_test.csv",
        )

        if summary is not None:
            summary = summary.copy()

            # Use one consistent name for this metric.
            summary["summary"] = summary["summary"].replace(
                {"A_adapt": "adapt_auc"}
            )

            summary_tables.append(summary)

    if summary_tables:
        combined_summary = pd.concat(
            summary_tables,
            ignore_index=True,
        )

        output.append(
            "## Adaptation-curve summary (TEST)\n"
            + md(combined_summary)
        )

    # ---------------------------------------------------------
    # Time hypothesis results
    # ---------------------------------------------------------
    for filepath in sorted(
        RESULTS_DIR.glob(f"{dataset}_time_*_test.csv")
    ):
        time_table = pd.read_csv(filepath).drop(
            columns=["ci90"],
            errors="ignore",
        )

        output.append(
            f"## Time hypothesis: {filepath.stem}\n"
            + md(time_table)
        )

    return output


def main():
    datasets = sys.argv[1:]

    if not datasets:
        datasets = ["yoochoose_f64", "yoochoose_f8", "diginetica"]

    output = []

    for dataset in datasets:
        output.extend(collect_dataset(dataset))

    output_file = RESULTS_DIR / "tables.md"
    output_file.write_text(
        "\n".join(output),
        encoding="utf-8",
    )

    print(f"Written: {output_file}")


if __name__ == "__main__":
    main()
