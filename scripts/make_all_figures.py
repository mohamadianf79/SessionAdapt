
from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Consistent publication style
# ---------------------------------------------------------

COLORS = {
    "Pop": "#9AA0A6",
    "SPop": "#E69F00",
    "Markov": "#0072B2",
    "GRU": "#009E73",
    "SASRec": "#CC79A7",
    "All": "#0072B2",
    "Shift": "#D55E00",
    "No shift": "#009E73",
    "base": "#555555",
    "Chance": "#777777",
}

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 400,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linewidth": 0.7,
    "axes.axisbelow": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def read(name):
    p = RESULTS / name
    if not p.exists():
        print("Missing:", name)
        return None
    return pd.read_csv(p)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("Saved:", name)


def pretty_model(value):
    value = str(value).strip().lower()
    mapping = {
        "pop": "Pop",
        "popularity": "Pop",
        "spop": "SPop",
        "s-pop": "SPop",
        "markov": "Markov",
        "lagmarkov": "LagMarkov",
        "gru": "GRU",
        "sasrec": "SASRec",
        "base_markov": "Base Markov",
    }
    return mapping.get(value, value.replace("_", " ").title())


def parse_ci(value):
    """Parse a CI string such as '[+0.0282,+0.0310]'."""
    if pd.isna(value):
        return None

    match = re.search(
        r"\[\s*([+-]?\d*\.?\d+)\s*,\s*([+-]?\d*\.?\d+)\s*\]",
        str(value)
    )
    if not match:
        return None

    return float(match.group(1)), float(match.group(2))


def add_ci_errorbars(ax, x, values, ci_values, color="black"):
    """Draw asymmetric confidence intervals around plotted values."""
    lower = []
    upper = []
    valid_x = []
    valid_y = []

    for xi, yi, ci in zip(x, values, ci_values):
        bounds = parse_ci(ci)
        if bounds is None or pd.isna(yi):
            continue

        lo, hi = bounds
        valid_x.append(xi)
        valid_y.append(yi)
        lower.append(max(0, yi - lo))
        upper.append(max(0, hi - yi))

    if valid_x:
        ax.errorbar(
            valid_x,
            valid_y,
            yerr=[lower, upper],
            fmt="none",
            ecolor=color,
            elinewidth=1.2,
            capsize=3,
            capthick=1.1,
            zorder=5,
        )


# ---------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------

def baseline_plot(dataset):
    base = read(f"{dataset}_baselines_test.csv")
    if base is None:
        return

    frames = [base[["model", "HR@20", "MRR@20", "NDCG@20"]].copy()]

    for model in ["gru", "sasrec"]:
        d = read(f"{dataset}_{model}_test.csv")
        if d is not None:
            frames.append(
                d[["model", "HR@20", "MRR@20", "NDCG@20"]]
            )

    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset="model", keep="last")
    df["display"] = df["model"].map(pretty_model)
    metrics = ["HR@20", "MRR@20", "NDCG@20"]

    fig, axes = plt.subplots(
        1, 3, figsize=(12, 4), sharey=False
    )

    for ax, metric in zip(axes, metrics):
        vals = df[["display", metric]].dropna()
        vals = vals.sort_values(metric, ascending=False)

        colors = [
            COLORS.get(label, "#777777")
            for label in vals["display"]
        ]

        bars = ax.bar(
            vals["display"],
            vals[metric],
            color=colors,
            width=0.68,
            edgecolor="white",
            linewidth=0.7,
        )

        ax.set_title(metric)
        ax.set_ylabel("Test score")
        ax.tick_params(axis="x", rotation=30)
        ax.set_ylim(0, min(1, max(vals[metric]) * 1.18))

        for bar, value in zip(bars, vals[metric]):
            ax.annotate(
                f"{value:.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    fig.suptitle(
        f"{dataset}: Test performance comparison",
        fontsize=13,
        fontweight="bold",
    )
    save(fig, f"{dataset}_baseline_comparison")


# ---------------------------------------------------------
# Cohort and population curves
# ---------------------------------------------------------

def curve_plot(dataset, population=False):
    suffix = (
        "curve_population_test.csv"
        if population else "curve_test.csv"
    )

    files = [
        f"{dataset}_{suffix}",
        f"{dataset}_gru_{suffix}",
        f"{dataset}_sasrec_{suffix}",
    ]

    frames = []
    for filename in files:
        d = read(filename)
        if d is not None:
            frames.append(d.copy())

    if not frames:
        return

    df = pd.concat(frames, ignore_index=True)

    for metric in ["HR@20", "MRR@20"]:
        part = df[df["metric"] == metric].copy()
        if part.empty:
            continue

        fig, ax = plt.subplots(figsize=(7.5, 5))

        for model, g in part.groupby("model"):
            g = g.sort_values("n")
            label = pretty_model(model)
            color = COLORS.get(label, "#777777")

            ax.plot(
                g["n"],
                g["value"],
                marker="o",
                markersize=3.5,
                linewidth=2,
                label=label,
                color=color,
            )

            if {"lo", "hi"}.issubset(g.columns):
                ax.fill_between(
                    g["n"].to_numpy(),
                    g["lo"].to_numpy(),
                    g["hi"].to_numpy(),
                    color=color,
                    alpha=0.13,
                    linewidth=0,
                )

        kind = "population" if population else "cohort"
        ax.set_title(f"{dataset}: {kind.capitalize()} adaptation")
        ax.set_xlabel("Number of observed clicks")
        ax.set_ylabel(metric)
        ax.legend(frameon=False, loc="best")
        ax.margins(x=0.03)

        save(
            fig,
            f"{dataset}_{kind}_{metric.replace('@', '_')}"
        )


# ---------------------------------------------------------
# Shift diagnosis
# ---------------------------------------------------------

def shift_plot(dataset):
    d = read(f"{dataset}_shift_diagnosis_val.csv")
    if d is None:
        return

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))

    for ax, metric in zip(axes, ["HR@20", "MRR@20"]):
        labels = d["group"].astype(str)
        colors = [
            COLORS.get(
                "No shift" if "no" in x.lower() else "Shift",
                "#777777"
            )
            for x in labels
        ]

        bars = ax.bar(
            labels,
            d[metric],
            color=colors,
            width=0.58,
            edgecolor="white",
        )

        ax.set_title(metric)
        ax.set_ylabel("Validation score")

        for bar, value in zip(bars, d[metric]):
            ax.annotate(
                f"{value:.3f}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                fontsize=8,
            )

    fig.suptitle(
        f"{dataset}: Performance under distribution shift",
        fontweight="bold",
    )
    save(fig, f"{dataset}_shift_diagnosis_val")


# ---------------------------------------------------------
# Adaptive and category-backoff comparisons with CIs
# ---------------------------------------------------------

def adaptation_plot(dataset, kind):
    filename = f"{dataset}_{kind}_test.csv"
    d = read(filename)
    if d is None:
        return

    name_col = "variant" if kind == "backoff" else "model"

    if name_col not in d.columns:
        print("Missing column:", name_col, filename)
        return

    # Base Markov is the reference, not a competing method.
    d = d[d[name_col].astype(str) != "base_markov"].copy()

    if d.empty:
        print("No non-baseline rows:", filename)
        return

    if kind == "adaptive":
        labels = [
            "Shift-aware" if str(x).startswith("shift_aware")
            else str(x).replace("_", " ").title()
            for x in d[name_col]
        ]
    else:
        labels = [
            str(x).replace("_", " ").title()
            for x in d[name_col]
        ]

    d["display"] = labels

    groups = [
        ("all", "All"),
        ("shift", "Shift"),
        ("no_shift", "No shift"),
    ]

    metrics = ["HR@20", "MRR@20"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(d))
    width = 0.23

    for ax, metric in zip(axes, metrics):
        for i, (group_key, group_label) in enumerate(groups):
            value_col = f"d{metric}[{group_key}]"
            ci_col = f"d{metric}[{group_key}]_CI"

            if value_col not in d.columns:
                continue

            values = pd.to_numeric(
                d[value_col], errors="coerce"
            ).to_numpy()

            positions = x + (i - 1) * width
            color = COLORS[group_label]

            ax.bar(
                positions,
                values,
                width=width,
                color=color,
                label=group_label,
                edgecolor="white",
                linewidth=0.6,
                zorder=3,
            )

            if ci_col in d.columns:
                add_ci_errorbars(
                    ax,
                    positions,
                    values,
                    d[ci_col].to_numpy(),
                )

        ax.axhline(
            0,
            color="#333333",
            linewidth=1,
            linestyle="--",
            zorder=2,
        )

        ax.set_title(f"Change in {metric}")
        ax.set_ylabel(f"Difference in {metric} vs. Markov")
        ax.set_xticks(x)
        ax.set_xticklabels(
            d["display"],
            rotation=25,
            ha="right",
        )
        ax.legend(frameon=False, title="Test subgroup")

    fig.suptitle(
        f"{dataset}: {kind.capitalize()} results on test set",
        fontweight="bold",
    )

    save(fig, f"{dataset}_{kind}_test_differences")


# ---------------------------------------------------------
# Slot comparisons
# ---------------------------------------------------------

def slots_plot(dataset):
    d = read(f"{dataset}_slots_test.csv")
    if d is None:
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(d))
    width = 0.34

    ax.bar(
        x - width / 2,
        d["dHR@20[all]"],
        width,
        label="HR@20",
        color=COLORS["Markov"],
    )
    ax.bar(
        x + width / 2,
        d["dMRR@20[all]"],
        width,
        label="MRR@20",
        color=COLORS["SPop"],
    )

    ax.axhline(0, color="#333333", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(
        d["model"].map(pretty_model),
        rotation=25,
        ha="right",
    )
    ax.set_ylabel("Difference vs. Markov")
    ax.set_title(f"{dataset}: Time-slot method comparison")
    ax.legend(frameon=False)

    save(fig, f"{dataset}_slots_test")


# ---------------------------------------------------------
# Detector AUC
# ---------------------------------------------------------

def detector_plot(dataset):
    d = read(f"{dataset}_detector_auc_val.csv")
    if d is None:
        return

    d = d.sort_values("AUC", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4.8))

    bars = ax.barh(
        d["detector"].astype(str),
        d["AUC"],
        color=COLORS["Markov"],
        height=0.62,
    )

    ax.axvline(
        0.5,
        color=COLORS["Chance"],
        linestyle="--",
        linewidth=1.3,
        label="Chance level",
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel("ROC AUC")
    ax.set_title(f"{dataset}: Shift detector performance")
    ax.legend(frameon=False, loc="lower right")

    for bar, value in zip(bars, d["AUC"]):
        ax.annotate(
            f"{value:.3f}",
            (bar.get_width(), bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )

    save(fig, f"{dataset}_detector_auc_val")


# ---------------------------------------------------------
# Time-aware comparisons
# ---------------------------------------------------------

def time_plot(dataset, filename):
    d = read(filename)
    if d is None:
        return

    d = d.copy()
    d["diff"] = pd.to_numeric(d["diff"], errors="coerce")
    d = d.dropna(subset=["diff"])

    labels = d["comparison"].astype(str).tolist()
    x = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(10, 4.8))

    colors = [
        COLORS["Markov"] if v >= 0 else COLORS["Shift"]
        for v in d["diff"]
    ]

    ax.bar(
        x,
        d["diff"],
        color=colors,
        width=0.62,
        edgecolor="white",
        zorder=3,
    )

    ax.axhline(
        0,
        linewidth=1,
        color="#333333",
        linestyle="--",
    )

    if "ci95" in d.columns:
        add_ci_errorbars(
            ax,
            x,
            d["diff"].to_numpy(),
            d["ci95"].to_numpy(),
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("Difference (variant - base)")
    ax.set_title(f"{dataset}: Time-aware test comparisons")

    save(fig, f"{dataset}_time_comparisons")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    for ds in ["diginetica", "yoochoose_f64", "yoochoose_f8"]:
        baseline_plot(ds)

    for ds in ["diginetica", "yoochoose_f64"]:
        curve_plot(ds, population=False)
        curve_plot(ds, population=True)
        shift_plot(ds)
        adaptation_plot(ds, "adaptive")
        adaptation_plot(ds, "backoff")
        detector_plot(ds)

    slots_plot("yoochoose_f64")

    time_plot(
        "diginetica",
        "diginetica_time_dow7_test.csv",
    )
    time_plot(
        "yoochoose_f64",
        "yoochoose_f64_time_daypart4_wkend_test.csv",
    )
    time_plot(
        "yoochoose_f8",
        "yoochoose_f8_time_daypart4_wkend_test.csv",
    )

    print("\nFinished. Figures are in:", OUT)


if __name__ == "__main__":
    main()
