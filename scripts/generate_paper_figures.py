#!/usr/bin/env python3
"""Generate publication-grade vector PDF + 400 dpi PNG data figures for PVLDB EA&B.

Strict visual rules:
  - Sans-serif Arial/Helvetica (`pdf.fonttype=42`, `mathtext.fontset='stixsans'`)
  - Minimum native font size >= 8.0 pt on ~6.6-6.8 in tight canvases so >= 7.4 pt at 0.90\\textwidth
  - Panel labels 11 pt bold (A, B, C), no suptitles, no top/right spines, no gridlines
  - Frameless legends, zero overlapping labels, CVPR-2025 Soft-Academic Dual-Tone palette
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG_DIR = ROOT / "latex" / "figures"

PAL = {
    "mint_fill": "#C4ECDA", "mint_edge": "#2F868D", "mint_mid": "#77D6C7",
    "peri_fill": "#DAE2F5", "peri_edge": "#4B6EB0", "peri_mid": "#809ED7",
    "rose_fill": "#F5B7BE", "rose_edge": "#C24B56", "rose_mid": "#E57C84",
    "iris_fill": "#E9E3EB", "iris_edge": "#77609B", "iris_mid": "#B1AFD4",
    "apri_fill": "#F8CDAC", "apri_edge": "#D47A38", "apri_mid": "#F4B183",
    "sage_fill": "#E1EFD9", "sage_edge": "#548B50", "sage_mid": "#B3D5AA",
    "gray_fill": "#D5D6DC", "gray_edge": "#6C6F78",
    "ink": "#3A3A3A", "ref": "#C8C8CE",
}

PAD_BOX = dict(boxstyle="square,pad=0.05", facecolor="#FFFFFF", edgecolor="none", alpha=0.88)


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 8.6,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.linewidth": 0.8,
        "axes.edgecolor": PAL["ink"],
        "axes.labelcolor": PAL["ink"],
        "xtick.color": PAL["ink"],
        "ytick.color": PAL["ink"],
        "text.color": PAL["ink"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "stixsans",
    })


def panel_tag(ax: plt.Axes, letter: str, x: float = -0.16, y: float = 1.05) -> None:
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11.0, fontweight="bold",
            va="bottom", ha="left", color=PAL["ink"])


def save_both(fig: plt.Figure, stem: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    pdf_p = FIG_DIR / f"{stem}.pdf"
    png_p = FIG_DIR / f"{stem}.png"
    fig.savefig(pdf_p, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(png_p, dpi=400, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Saved {pdf_p} & {png_p}")
    return png_p


# ---------------------------------------------------------------------------
# Figure 2: Cross-Engine 6-System Audit (Exp 5A)
# ---------------------------------------------------------------------------
def make_fig2() -> Path:
    df5 = pd.read_csv(RES / "exp5_multi_engine" / "exp5_multi_engine_summary.csv")
    rss_df = pd.read_csv(RES / "exp5_multi_engine" / "exp5_rss_probe.csv")
    rss_map = {(r["system"], r["param_tag"]): float(r["peak_rss_mb"]) for _, r in rss_df.iterrows()}

    sift = df5[df5["workload"] == "sift128_drift"].groupby(["system", "param_tag"], as_index=False).agg(
        query_s=("query_cpu_ms", lambda x: x.mean() / 1000.0),
        maint_s=("maint_cpu_ms", lambda x: x.mean() / 1000.0),
        total_s=("total_cpu_ms", lambda x: x.mean() / 1000.0),
        att=("attainment_pct", "mean"),
        rss=("peak_rss_mb", "mean"),
    )
    for i, r in sift.iterrows():
        if pd.isna(r["rss"]):
            sift.at[i, "rss"] = rss_map[(r["system"], r["param_tag"])]

    eng_colors = {
        "faiss_ivf": (PAL["peri_mid"], PAL["peri_edge"]),
        "quake": (PAL["mint_mid"], PAL["mint_edge"]),
        "diskann": (PAL["sage_mid"], PAL["sage_edge"]),
        "hnswlib_mut": (PAL["apri_mid"], PAL["apri_edge"]),
        "annoy_rebuild": (PAL["iris_mid"], PAL["iris_edge"]),
        "usearch_mut": (PAL["rose_mid"], PAL["rose_edge"]),
    }

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(6.80, 2.32),
        gridspec_kw={"width_ratios": [1.36, 0.94, 1.00], "wspace": 0.68},
    )

    label_pos = {
        ("faiss_ivf", "np16"): (-6, 52, "FAISS np16", "left"),
        ("faiss_ivf", "np32"): (-6, 118, "FAISS\nnp32", "left"),
        ("quake", "rt097"): (46, 70, "Quake rt097", "left"),
        ("diskann", "L64"): (86, 118, "DiskANN\nL64", "center"),
        ("diskann", "L96"): (166, 118, "DiskANN\nL96", "center"),
        ("hnswlib_mut", "ef64"): (22, 30, "hnswlib ef64", "left"),
        ("hnswlib_mut", "ef128"): (92, 84, "hnswlib ef128", "center"),
        ("usearch_mut", "ef64"): (142, 42, "USearch\nef64", "center"),
        ("usearch_mut", "ef128"): (178, 76, "USearch\nef128", "center"),
        ("annoy_rebuild", "sk8000_re4"): (-4, -9, "Annoy re4", "left"),
        ("annoy_rebuild", "sk8000_re1"): (126, 11, "Annoy re1", "left"),
    }
    for _, r in sift.iterrows():
        fc, _ = eng_colors[r["system"]]
        sz = max(24, r["rss"] / 12.5)
        ax1.scatter(r["total_s"], r["att"], s=sz, facecolor=fc, edgecolor=PAL["ink"],
                    linewidth=0.7, alpha=0.92, zorder=4)
        tx, ty, short, ha_v = label_pos[(r["system"], r["param_tag"])]
        ax1.annotate(short, xy=(r["total_s"], r["att"]), xytext=(tx, ty),
                     fontsize=8.0, color=PAL["ink"], ha=ha_v, va="center",
                     arrowprops=dict(arrowstyle="-", color=PAL["gray_edge"], lw=0.55, shrinkA=0, shrinkB=4))
    ax1.axhline(95.0, color=PAL["ref"], linestyle="--", linewidth=0.8)
    ax1.text(208, -12, "Size ~ RSS", fontsize=8.0, ha="right", color=PAL["gray_edge"], style="italic")
    ax1.set_xlabel("Total CPU Time (s)")
    ax1.set_ylabel("Attainment % (Recall \u2265 0.965)")
    ax1.set_xlim(-12, 214)
    ax1.set_ylim(-20, 136)
    panel_tag(ax1, "A", x=-0.18, y=1.10)

    rep = [
        ("quake", "rt097", "Quake rt097"),
        ("faiss_ivf", "np32", "FAISS np32"),
        ("diskann", "L64", "DiskANN L64"),
        ("hnswlib_mut", "ef64", "hnswlib ef64"),
        ("annoy_rebuild", "sk8000_re1", "Annoy re1"),
        ("usearch_mut", "ef64", "USearch ef64"),
    ]
    y_pos = np.arange(len(rep))
    q_vals = [float(sift[(sift["system"] == s) & (sift["param_tag"] == t)]["query_s"].iloc[0]) for s, t, _ in rep]
    m_vals = [float(sift[(sift["system"] == s) & (sift["param_tag"] == t)]["maint_s"].iloc[0]) for s, t, _ in rep]
    ax2.barh(y_pos, q_vals, height=0.58, facecolor=PAL["peri_fill"], edgecolor=PAL["ink"], linewidth=0.6, label="Query")
    ax2.barh(y_pos, m_vals, left=q_vals, height=0.58, facecolor=PAL["apri_fill"], edgecolor=PAL["ink"], linewidth=0.6, label="Maint.")
    for idx, (qv, mv) in enumerate(zip(q_vals, m_vals)):
        ax2.text(qv + mv + 4.0, idx, f"{qv + mv:.1f}s", va="center", ha="left", fontsize=8.0, color=PAL["ink"])
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([lbl for _, _, lbl in rep], fontsize=8.0)
    ax2.set_xlabel("SIFT-128 CPU (s)")
    ax2.set_xlim(0, 222)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.50, 1.01), ncol=2, fontsize=8.0, columnspacing=0.6, handlelength=0.9, borderpad=0.2)
    panel_tag(ax2, "B", x=-0.42, y=1.10)

    wls = ["crud30v2", "crud50v2", "runbook"]
    wl_lbls = ["CRUD\n30%", "CRUD\n50%", "Del.\nRunbook"]
    churn = df5[df5["workload"].isin(wls)].groupby(["workload", "system", "param_tag", "status"], as_index=False)["total_cpu_ms"].mean()
    c_arms = [
        ("faiss_ivf", "np16", "FAISS", PAL["peri_fill"], None),
        ("annoy_rebuild", "sk8000_re4", "Annoy", PAL["gray_fill"], "//"),
        ("usearch_mut", "ef64", "USearch", PAL["rose_fill"], None),
        ("hnswlib_mut", "ef64", "hnswlib", PAL["apri_fill"], None),
    ]
    cx = np.arange(len(wls))
    cw = 0.19
    for idx, (s, t, lbl, fc, ht) in enumerate(c_arms):
        vals = [float(churn[(churn["workload"] == w) & (churn["system"] == s) & (churn["param_tag"] == t)]["total_cpu_ms"].iloc[0]) / 1000.0 for w in wls]
        bars = ax3.bar(cx + (idx - 1.5) * cw, vals, width=cw, facecolor=fc, edgecolor=PAL["ink"], linewidth=0.6, hatch=ht, label=lbl)
        if s == "annoy_rebuild":
            for b, v in zip(bars, vals):
                ax3.text(b.get_x() + b.get_width() / 2, v + 3.5, "\u00d7", ha="center", va="bottom", fontsize=8.8, fontweight="bold", color=PAL["rose_edge"])
    ax3.set_xticks(cx)
    ax3.set_xticklabels(wl_lbls, fontsize=8.0)
    ax3.set_ylabel("Total CPU Time (s)")
    ax3.set_ylim(0, 290)
    ax3.legend(loc="upper left", ncol=2, fontsize=8.0, columnspacing=0.5, handlelength=0.9, labelspacing=0.20)
    panel_tag(ax3, "C", x=-0.24, y=1.10)

    return save_both(fig, "fig2_exp5_6engines")


# ---------------------------------------------------------------------------
# Figure 3: Trap 4 Contention, Cross-NUMA Inflation & Ranking Reversal
# ---------------------------------------------------------------------------
def make_fig3() -> Path:
    df = pd.read_csv(RES / "stage1_hygiene" / "hygiene_summary.csv")
    regimes = ["isolated_numa0", "ht_sibling_2w", "same_socket_4w", "same_socket_8w", "cross_socket_16w"]
    r_labels = ["Iso\n1w", "HT\n2w", "Skt\n4w", "Skt\n8w", "NUMA\n16w"]
    g = df.groupby(["regime", "arm"], as_index=False)[["total_cpu_ms", "mean_p95_us", "mean_p99_us"]].mean()

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(6.75, 2.24),
        gridspec_kw={"width_ratios": [1.14, 1.06, 1.02], "wspace": 0.44},
    )
    x = np.arange(len(regimes))
    w = 0.24

    arms = [
        ("noop", "No-Op", PAL["gray_fill"], PAL["gray_edge"]),
        ("fixed_c2", "Fixed c=2", PAL["peri_fill"], PAL["peri_edge"]),
        ("past_qmass", "Past-QMass", PAL["mint_fill"], PAL["mint_edge"]),
    ]
    for k, (arm, lbl, fc, ec) in enumerate(arms):
        vals = [float(g[(g["regime"] == r) & (g["arm"] == arm)]["total_cpu_ms"].iloc[0]) / 1000.0 for r in regimes]
        bars = ax1.bar(x + (k - 1) * w, vals, width=w, facecolor=fc, edgecolor=PAL["ink"], linewidth=0.6, label=lbl)
        if arm == "noop":
            iso_v = vals[0]
            ax1.text(bars[0].get_x() + 0.02, vals[0] + 7, "1.00\u00d7",
                     ha="left", va="bottom", fontsize=8.0, fontweight="bold", color=ec)
            ax1.text(bars[4].get_x() + bars[4].get_width() / 2, vals[4] + 7, f"{vals[4]/iso_v:.2f}\u00d7",
                     ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=ec)

    ax1.axhline(float(g[(g["regime"] == "isolated_numa0") & (g["arm"] == "noop")]["total_cpu_ms"].iloc[0]) / 1000.0,
                color=PAL["ref"], linestyle="--", linewidth=0.9, zorder=0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(r_labels, fontsize=8.0)
    ax1.set_ylabel("Total CPU Time (s)")
    ax1.set_xlim(-0.68, 4.55)
    ax1.set_ylim(0, 455)
    ax1.legend(loc="upper left", ncol=2, fontsize=8.0, columnspacing=0.5, handlelength=0.9, labelspacing=0.20)
    panel_tag(ax1, "A", x=-0.18, y=1.06)

    p95_noop = [float(g[(g["regime"] == r) & (g["arm"] == "noop")]["mean_p95_us"].iloc[0]) / 1000.0 for r in regimes]
    p99_noop = [float(g[(g["regime"] == r) & (g["arm"] == "noop")]["mean_p99_us"].iloc[0]) / 1000.0 for r in regimes]
    p99_pq = [float(g[(g["regime"] == r) & (g["arm"] == "past_qmass")]["mean_p99_us"].iloc[0]) / 1000.0 for r in regimes]

    ax2.plot(x, p99_noop, marker="o", markersize=4.3, linewidth=1.6, color=PAL["rose_edge"], label="No-Op p99")
    ax2.plot(x, p95_noop, marker="s", markersize=3.8, linewidth=1.3, linestyle="--", color=PAL["apri_edge"], label="No-Op p95")
    ax2.plot(x, p99_pq, marker="^", markersize=4.3, linewidth=1.5, color=PAL["mint_edge"], label="PQ p99")
    ax2.annotate(f"{p99_noop[0]:.1f} ms", xy=(0, p99_noop[0]), xytext=(0.65, 2.5),
                 fontsize=8.0, color=PAL["ink"], fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=PAL["ink"], lw=0.7))
    ax2.annotate(f"Sym. flip\n({p99_pq[3]:.1f} > {p99_noop[3]:.1f})", xy=(3, p99_pq[3]),
                 xytext=(1.75, 102.0), ha="center", va="bottom", fontsize=7.6, fontweight="bold", color=PAL["mint_edge"],
                 arrowprops=dict(arrowstyle="->", color=PAL["mint_edge"], lw=0.75))
    ax2.annotate(f"{p99_noop[4]:.1f} ms\n({p99_noop[4]/p99_noop[0]:.1f}\u00d7)", xy=(4, p99_noop[4]),
                 xytext=(3.58, 116.0), ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=PAL["rose_edge"],
                 arrowprops=dict(arrowstyle="->", color=PAL["rose_edge"], lw=0.8))
    ax2.set_xticks(x)
    ax2.set_xticklabels(r_labels, fontsize=8.0)
    ax2.set_ylabel("Query Tail Latency (ms)")
    ax2.set_xlim(-0.35, 4.45)
    ax2.set_ylim(-6, 168)
    ax2.legend(loc="upper left", fontsize=8.0, handlelength=1.1, labelspacing=0.18)
    panel_tag(ax2, "B", x=-0.18, y=1.06)

    tsws = [16, 24, 28]
    y_rows = np.arange(len(tsws))
    pq_dirty, pq_clean, gb_dirty, gb_clean = [], [], [], []
    for tsw in tsws:
        dd = pd.read_csv(RES / "abrupt_tsw" / f"abrupt_tsw{tsw}_summary.csv").groupby("arm")["total_cpu_ms"].mean() / 1000.0
        dc = pd.read_csv(RES / "abrupt_tsw_clean" / f"abrupt_tsw{tsw}_summary.csv").groupby("arm")["total_cpu_ms"].mean() / 1000.0
        pq_dirty.append(float(dd["past_qmass_sp1"]))
        pq_clean.append(float(dc["past_qmass_sp1"]))
        gb_dirty.append(float(dd["forecast_gbrt_causal"]))
        gb_clean.append(float(dc["forecast_gbrt_causal"]))

    for idx in range(len(tsws)):
        ax3.plot([pq_clean[idx], pq_dirty[idx]], [idx + 0.14, idx + 0.14], color=PAL["mint_edge"], linewidth=1.4, zorder=2)
        ax3.plot([gb_clean[idx], gb_dirty[idx]], [idx - 0.14, idx - 0.14], color=PAL["peri_edge"], linewidth=1.4, zorder=2)
    ax3.scatter(pq_dirty, y_rows + 0.14, s=40, marker="o", facecolor=PAL["rose_fill"], edgecolor=PAL["rose_edge"], linewidth=0.9, zorder=4, label="PQ (Contam. Batch)")
    ax3.scatter(pq_clean, y_rows + 0.14, s=44, marker="D", facecolor=PAL["mint_mid"], edgecolor=PAL["ink"], linewidth=0.7, zorder=5, label="PQ (Clean)")
    ax3.scatter(gb_clean, y_rows - 0.14, s=40, marker="s", facecolor=PAL["peri_mid"], edgecolor=PAL["ink"], linewidth=0.7, zorder=4, label="GBRT (Idle Batch)")

    ax3.text(pq_dirty[2] - 8, 2.36, "951.6s", fontsize=8.0, ha="right", color=PAL["rose_edge"], fontweight="bold")
    ax3.text(pq_clean[0] - 8, 0.36, "507.3s", fontsize=8.0, ha="left", color=PAL["mint_edge"], fontweight="bold")
    ax3.set_yticks(y_rows)
    ax3.set_yticklabels([f"$t_{{sw}}={t}$" for t in tsws], fontsize=8.0)
    ax3.set_xlabel("Abrupt-Switch CPU (s)")
    ax3.set_xlim(435, 1030)
    ax3.set_ylim(-1.82, 2.85)
    ax3.legend(loc="lower right", fontsize=7.8, handlelength=0.9, labelspacing=0.14)
    panel_tag(ax3, "C", x=-0.20, y=1.06)

    return save_both(fig, "fig3_trap4_contention")


# ---------------------------------------------------------------------------
# Figure 4: Hygiene Gates 1-3 (Cadence Brackets, Clock Leakage, 1w-vs-4w CF)
# ---------------------------------------------------------------------------
def make_fig4() -> Path:
    df2 = pd.read_csv(RES / "exp2_predictors_cadence" / "mimo_core8_575_summary.csv")
    diag = json.loads((RES / "exp2_predictors_cadence" / "predictor_diagnostics.json").read_text())
    ledger = json.loads((RES / "paper_tables" / "verified_numbers.json").read_text())
    s25 = df2[df2["suite"] == "sift25"].groupby("arm", as_index=False)[["total_cpu_ms", "n_actions", "maint_cpu_ms"]].mean()

    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(6.75, 2.32),
        gridspec_kw={"width_ratios": [1.08, 1.05, 1.04], "wspace": 0.66},
    )

    cad_arms = [f"fixed_c{c}_merge" for c in range(1, 9)] + ["noop"]
    cad_df = s25[s25["arm"].isin(cad_arms)].sort_values("n_actions")
    cx = cad_df["n_actions"].values
    cy = cad_df["total_cpu_ms"].values / 1000.0
    ax1.plot(cx, cy, marker="o", markersize=3.8, linewidth=1.35, color=PAL["peri_edge"], label="Fixed c=1..8")

    n0 = s25[s25["arm"] == "noop"].iloc[0]
    c1 = s25[s25["arm"] == "fixed_c1_merge"].iloc[0]
    ax1.plot([n0["n_actions"], c1["n_actions"]], [n0["total_cpu_ms"] / 1000.0, c1["total_cpu_ms"] / 1000.0],
             linestyle=":", linewidth=1.2, color=PAL["rose_edge"], label="Interp. (Trap)")

    pq = s25[s25["arm"] == "past_qmass_k5"].iloc[0]
    ora = s25[s25["arm"] == "fcst_oracle_clairvoyant_k5"].iloc[0]
    ax1.scatter([pq["n_actions"]], [pq["total_cpu_ms"] / 1000.0], s=48, marker="D",
                facecolor=PAL["mint_mid"], edgecolor=PAL["ink"], linewidth=0.8, zorder=5, label=r"PQ ($\kappa=5$)")
    ax1.scatter([ora["n_actions"]], [ora["total_cpu_ms"] / 1000.0], s=56, marker="*",
                facecolor=PAL["apri_mid"], edgecolor=PAL["ink"], linewidth=0.7, zorder=5, label=r"Oracle ($\kappa=5$)")

    ax1.annotate(f"Past-QMass\n({pq['n_actions']:.1f} acts, {pq['total_cpu_ms']/1000:.1f}s)",
                 xy=(pq["n_actions"], pq["total_cpu_ms"] / 1000.0),
                 xytext=(-0.25, 138.5), fontsize=8.0, fontweight="bold", color=PAL["mint_edge"],
                 arrowprops=dict(arrowstyle="->", color=PAL["mint_edge"], lw=0.8, shrinkB=4))
    ax1.text(0.35, 174.2, "No-Op (0)", fontsize=8.0, color=PAL["ink"])
    ax1.text(4.8, 164.0, "c=2", fontsize=8.0, color=PAL["peri_edge"])
    ax1.text(7.4, 140.5, "c=1", fontsize=8.0, color=PAL["peri_edge"])
    ax1.set_xlabel("Executed Actions (4-Block)")
    ax1.set_ylabel("Total CPU Time (s)")
    ax1.set_xlim(-0.6, 9.9)
    ax1.set_ylim(119, 185)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.55, 1.01), ncol=2, fontsize=8.0,
               columnspacing=0.5, handlelength=0.9, labelspacing=0.16)
    panel_tag(ax1, "A", x=-0.22, y=1.22)

    rhos = diag["clock_leak_audit"]["gbrt_leaky"]["all_rhos"]
    items = sorted(rhos.items(), key=lambda kv: abs(kv[1]))
    picked = [kv for kv in items if abs(kv[1]) >= 0.002 or kv[0] in ("slope", "q_lag1", "qmean8", "qstd8")][-8:]
    label_map = {
        "leak_dcum_total": "Cum. ins. (LEAK)",
        "leak_block_j": "Step j (LEAK)",
        "leak_dcum_shard": "Shard ins. (LEAK)",
        "qstd8": "Q std (8w)",
        "qmean8": "Q mean (8w)",
        "q_lag1": "Lag-1 query",
        "q_lag2": "Lag-2 query",
        "dinc4": "Delta inc. (4w)",
    }
    names = [label_map.get(k, k) for k, _ in picked]
    vals = [abs(v) for _, v in picked]
    colors = [PAL["rose_fill"] if v >= 0.70 else PAL["peri_fill"] for v in vals]
    edges = [PAL["rose_edge"] if v >= 0.70 else PAL["peri_edge"] for v in vals]

    yp = np.arange(len(names))
    bars = ax2.barh(yp, vals, height=0.62, color=colors, edgecolor=PAL["ink"], linewidth=0.6)
    ax2.axvline(0.70, color=PAL["rose_edge"], linestyle="--", linewidth=1.1)
    ax2.text(0.74, 0.75, "Gate 2\n(|\u03c1|=0.70)", ha="left", va="bottom", fontsize=8.0,
             fontweight="bold", color=PAL["rose_edge"])
    for b, v, ec in zip(bars, vals, edges):
        ax2.text(v + 0.03, b.get_y() + b.get_height() / 2, f"{v:.3f}",
                 va="center", ha="left", fontsize=8.0, color=ec if v >= 0.70 else PAL["ink"],
                 fontweight="bold" if v >= 0.70 else "normal")
    ax2.set_yticks(yp)
    ax2.set_yticklabels(names, fontsize=8.0)
    ax2.set_xlabel(r"Spearman $|\rho(f, j)|$")
    ax2.set_xlim(0, 1.54)
    panel_tag(ax2, "B", x=-0.42, y=1.22)

    cf = ledger["gate3_stats"]["track_cf_1w_vs_4w"]
    trace_disp = {
        "cf_crud30_abrupt_s05": "CRUD-30 Abrupt",
        "cf_crud50_abrupt_s05": "CRUD-50 Abrupt",
        "cf_u10_abrupt_s05": "Insert Abrupt",
        "cf_u10_gradual_s06": "Insert Gradual",
        "cf_u10_uniform_s05": "Insert Uniform",
    }
    cf_sorted = sorted(cf, key=lambda r: r["pair_flip_pct"])
    yc = np.arange(len(cf_sorted))
    flips = [r["pair_flip_pct"] for r in cf_sorted]
    ax3.hlines(yc, 0, flips, color=PAL["apri_edge"], linewidth=1.8, zorder=2)
    ax3.scatter(flips, yc, s=46, facecolor=PAL["apri_mid"], edgecolor=PAL["ink"], linewidth=0.7, zorder=4)
    for idx, r in enumerate(cf_sorted):
        ax3.text(r["pair_flip_pct"] + 5.5, idx, f"{r['pair_flip_pct']:.1f}% ({r['best_action_match']})",
                 va="center", ha="left", fontsize=8.0, color=PAL["ink"])
    ax3.text(92, 0.22, "4.4\u20134.5\u00d7\n4w/1w", fontsize=8.0, ha="right", va="bottom", color=PAL["gray_edge"], style="italic")
    ax3.set_yticks(yc)
    ax3.set_yticklabels([trace_disp[r["trace"]] for r in cf_sorted], fontsize=8.0)
    ax3.set_xlabel("1w vs. 4w Flip (%)")
    ax3.set_xlim(0, 96)
    ax3.set_ylim(-0.6, 4.6)
    panel_tag(ax3, "C", x=-0.42, y=1.22)

    return save_both(fig, "fig4_traps123_hygiene")


# ---------------------------------------------------------------------------
# Figure 5: Cross-Modality Predictability & Matched-Budget CRUD Fairness
# ---------------------------------------------------------------------------
def make_fig5() -> Path:
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.70, 2.25),
        gridspec_kw={"width_ratios": [1.06, 1.00], "wspace": 0.56},
    )

    specs = [
        ("MiniLM-384", RES / "crossmodal_rag384" / "exact_ops_and_dp.csv",
         {"fixed_c2": "fixed_c2_merge", "past_qmass": "past_qmass_sp1", "gbrt": "forecast_gbrt_causal", "oracle": "forecast_oracle"}),
        ("CLIP-512", RES / "crossmodal_clip512" / "exact_ops_and_dp.csv",
         {"fixed_c2": "fixed_c2_merge", "past_qmass": "past_qmass_sp1", "gbrt": "forecast_gbrt_causal", "oracle": "forecast_oracle"}),
        ("CLINC-150", RES / "clinc150" / "exact_ops_and_dp.csv",
         {"fixed_c2": "fixed_c2_merge", "past_qmass": "past_qmass_sp1", "gbrt": "forecast_gbrt_causal", "oracle": "forecast_oracle"}),
        ("Wiki-210d", RES / "wikimedia_windows" / "exact_ops_and_dp.csv",
         {"fixed_c2": "fixed_c2_merge", "past_qmass": "past_qmass_uncap", "gbrt": "forecast_gbrt_causal_uncap", "oracle": "forecast_oracle_clairvoyant_uncap"}),
        ("arXiv-CS", RES / "exp7_gpu_1m_arxiv" / "arxiv_minilm384_exact_ops_and_dp.csv",
         {"fixed_c2": "fixed_c2_merge", "past_qmass": "past_qmass_k5", "gbrt": "forecast_gbrt_causal_k5", "oracle": "forecast_oracle_clairvoyant_k5"}),
    ]
    x = np.arange(len(specs))
    w = 0.19
    pols = [
        ("fixed_c2", "Fixed c=2", PAL["gray_fill"]),
        ("past_qmass", "Past-QMass", PAL["mint_fill"]),
        ("gbrt", "GBRT (Causal)", PAL["peri_fill"]),
        ("oracle", "Oracle", PAL["apri_fill"]),
    ]
    for k, (pk, lbl, fc) in enumerate(pols):
        vals = []
        for _, fp, amap in specs:
            d = pd.read_csv(fp)
            vals.append(float(d[d["arm"] == amap[pk]]["global_capture_ratio"].mean()) * 100.0)
        bars = ax1.bar(x + (k - 1.5) * w, vals, width=w, facecolor=fc, edgecolor=PAL["ink"], linewidth=0.6, label=lbl)
        if pk == "fixed_c2":
            bx = bars[2].get_x() + bars[2].get_width() / 2.0
            ax1.text(bx, vals[2] + 3.0, f"{vals[2]:.1f}%", ha="center", va="bottom",
                     rotation=90, fontsize=8.0, fontweight="bold", color=PAL["rose_edge"])
    ax1.set_xticks(x)
    ax1.set_xticklabels([s[0] for s in specs], fontsize=8.0, rotation=20, ha="right")
    ax1.set_ylabel("Global Capture Ratio (%)")
    ax1.set_ylim(0, 108)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.50, 1.01), ncol=2, fontsize=8.0,
               columnspacing=0.7, handlelength=1.0, labelspacing=0.20)
    panel_tag(ax1, "A", x=-0.18, y=1.14)

    cv2 = pd.read_csv(RES / "sprint_v2_llm01" / "canon25" / "exact_ops_and_dp.csv").groupby("arm", as_index=False)["global_capture_ratio"].mean()
    c30 = pd.read_csv(RES / "sprint_v2_llm01" / "crud30v2" / "exact_ops_and_dp.csv").groupby("arm", as_index=False)["global_capture_ratio"].mean()
    c50 = pd.read_csv(RES / "sprint_v2_llm01" / "crud50v2" / "exact_ops_and_dp.csv").groupby("arm", as_index=False)["global_capture_ratio"].mean()

    arms_b = [
        ("fixed_c2_merge", "Fixed c=2 (B=4)"),
        ("max_delta_b4", "Max-Delta (B=4)"),
        ("past_qmass_b4_greedy", "Past-QMass (B=4)"),
        ("online_milp_b4", "MILP (B=4)"),
        ("past_qmass_sp1", "Past-QMass (B=9)"),
        ("gbrt_single_stream", "GBRT-SS (B=9)"),
        ("forecast_oracle", "Oracle (B=9)"),
    ]
    yb = np.arange(len(arms_b))
    h = 0.26
    vv2 = [float(cv2[cv2["arm"] == a]["global_capture_ratio"].iloc[0]) * 100.0 for a, _ in arms_b]
    v30 = [float(c30[c30["arm"] == a]["global_capture_ratio"].iloc[0]) * 100.0 for a, _ in arms_b]
    v50 = [float(c50[c50["arm"] == a]["global_capture_ratio"].iloc[0]) * 100.0 for a, _ in arms_b]

    ax2.barh(yb + h, vv2, height=h, facecolor=PAL["apri_fill"], edgecolor=PAL["ink"], linewidth=0.55, label="Canon25")
    ax2.barh(yb, v30, height=h, facecolor=PAL["peri_fill"], edgecolor=PAL["ink"], linewidth=0.55, label="CRUD 30%")
    ax2.barh(yb - h, v50, height=h, facecolor=PAL["mint_fill"], edgecolor=PAL["ink"], linewidth=0.55, label="CRUD 50%")
    ax2.axhline(3.5, color=PAL["ref"], linestyle="--", linewidth=0.9)
    ax2.set_yticks(yb)
    ax2.set_yticklabels([lbl for _, lbl in arms_b], fontsize=8.0)
    ax2.set_xlabel("Global Capture Ratio (%)")
    ax2.set_xlim(0, 102)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.48, 1.01), ncol=3, fontsize=8.0,
               columnspacing=0.45, handlelength=0.85, labelspacing=0.16)
    panel_tag(ax2, "B", x=-0.36, y=1.14)

    return save_both(fig, "fig5_predictability_spectrum")


# ---------------------------------------------------------------------------
# Figure 6: Tombstone Phase Transition, (u, Q) Phase Law, & Scale/Arch Transfer
# ---------------------------------------------------------------------------
def make_fig6() -> Path:
    fig, (ax1, ax2, ax3) = plt.subplots(
        1, 3, figsize=(6.95, 2.02),
        gridspec_kw={"width_ratios": [0.98, 1.25, 1.30], "wspace": 0.44},
    )

    rates = [0, 10, 20, 30, 50]
    m_cpus, r_cpus = [], []
    for rt in rates:
        d = pd.read_csv(RES / "tombstone_sweep" / f"crud{rt:02d}" / "exact_ops_and_dp.csv")
        g = d.groupby("arm")["total_cpu_ms"].mean() / 1000.0
        m_cpus.append(float(g["past_qmass_sp1"]))
        r_cpus.append(float(g["past_qmass_refresh"]))
    xp = np.arange(len(rates))
    ax1.plot(xp, m_cpus, marker="o", markersize=4.5, linewidth=1.6, color=PAL["peri_edge"], label="Past-QMass (Merge)")
    ax1.plot(xp, r_cpus, marker="D", markersize=4.3, linewidth=1.6, color=PAL["mint_edge"], label="Past-QMass (Refresh)")
    ax1.fill_between(xp, r_cpus, m_cpus, where=np.array(r_cpus) <= np.array(m_cpus),
                     interpolate=True, color=PAL["mint_fill"], alpha=0.55)
    ax1.annotate("-17.0%\n(245 vs 296s)", xy=(2, r_cpus[2]), xytext=(1.35, 191),
                 fontsize=8.3, fontweight="bold", color=PAL["mint_edge"], ha="center",
                 arrowprops=dict(arrowstyle="->", color=PAL["mint_edge"], lw=0.8, shrinkB=3))
    ax1.set_xticks(xp)
    ax1.set_xticklabels([f"{r}%" for r in rates], fontsize=8.3)
    ax1.set_xlabel("CRUD Deletion Rate", fontsize=8.6)
    ax1.set_ylabel("Total CPU Time (s)", fontsize=8.6)
    ax1.set_ylim(174, 338)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.50, 1.01), ncol=1, fontsize=8.2, handlelength=1.1, labelspacing=0.14)
    panel_tag(ax1, "A", x=-0.22, y=1.18)

    us = [2, 5, 10, 20]
    qs = [500, 1250, 2500, 5000]
    mat = np.zeros((len(us), len(qs)))
    for i, u in enumerate(us):
        for j, q in enumerate(qs):
            fp = RES / "phase_uxq_clean" / f"phase_u{u:02d}_q{q:04d}_v2_summary.csv"
            if fp.exists():
                d = pd.read_csv(fp)
            else:
                d = pd.read_csv(RES / "sift100k_canon25" / "exact_ops_and_dp.csv")
            if "wall_cpu_ratio" in d.columns:
                d = d[d["wall_cpu_ratio"] <= 1.18]  # Gate 4: CPU only from unflagged runs
            g = d.groupby("arm")["total_cpu_ms"].mean()
            if "noop" in g.index and "past_qmass_sp1" in g.index:
                mat[i, j] = (float(g["noop"]) - float(g["past_qmass_sp1"])) / float(g["noop"]) * 100.0
            else:
                mat[i, j] = np.nan

    cmap = LinearSegmentedColormap.from_list("mint_soft", ["#F2FAF7", "#77D6C7", "#2F868D"])
    cmap.set_bad("#E4E4E8")
    ax2.imshow(np.ma.masked_invalid(mat), cmap=cmap, aspect="auto", vmin=10.0, vmax=45.0)
    for i in range(len(us)):
        for j in range(len(qs)):
            val = mat[i, j]
            if np.isnan(val):
                ax2.text(j, i, "flag", ha="center", va="center", fontsize=7.9, style="italic", color=PAL["ink"])
                continue
            txt_col = "#FFFFFF" if val >= 34.0 else PAL["ink"]
            ax2.text(j, i, f"{val:.1f}%", ha="center", va="center", fontsize=7.9, fontweight="bold", color=txt_col)
    ax2.set_xticks(np.arange(len(qs)))
    ax2.set_xticklabels([str(q) for q in qs], fontsize=8.3)
    ax2.set_yticks(np.arange(len(us)))
    ax2.set_yticklabels([f"{u}%" for u in us], fontsize=8.3)
    ax2.set_xlabel("Queries per Window (Q)", fontsize=8.6)
    ax2.set_ylabel("Insert Rate (u)", fontsize=8.6)
    ax2.set_title("% CPU Saved vs. No-Op (Gate-4 clean)", fontsize=8.5, pad=5, color=PAL["ink"])
    ax2.spines["top"].set_visible(True)
    ax2.spines["right"].set_visible(True)
    panel_tag(ax2, "B", x=-0.22, y=1.18)

    suites = [
        ("100K", RES / "sift100k_canon25" / "exact_ops_and_dp.csv"),
        ("300K", RES / "sift300k" / "exact_ops_and_dp.csv"),
        ("600K", RES / "sift600k_s07" / "exact_ops_and_dp.csv"),
        ("1M", RES / "sprint_v2_llm01" / "sift1m_n12" / "exact_ops_and_dp.csv"),
    ]
    c2_v, pq_v, ora_v, x_lbls = [], [], [], []
    for lbl, fp in suites:
        d = pd.read_csv(fp)
        g = d.groupby("arm")["global_capture_ratio"].mean() * 100.0
        x_lbls.append(lbl)
        c2_v.append(float(g["fixed_c2_merge"]))
        pq_v.append(float(g["past_qmass_sp1"]))
        ora_v.append(float(g["forecast_oracle"]) if "forecast_oracle" in g.index else np.nan)

    s2_hr_p = RES / "exp9_rust_and_scale" / "s2_headroom_100k_to_10m.csv"
    if s2_hr_p.exists():
        s2_hr = pd.read_csv(s2_hr_p)
        for sc_tag, disp_tag in [("5M", "5M*"), ("10M", "10M*")]:
            sub_h = s2_hr[s2_hr["scale"] == sc_tag].groupby("policy")["gcr_pct"].mean()
            x_lbls.append(disp_tag)
            c2_v.append(float(sub_h["fixed_c2"]))
            pq_v.append(float(sub_h["ewma_a03"]))
            ora_v.append(float(sub_h["oracle_1blk"]))

    for lbl, fp in [
        ("Xeon", RES / "xeon_anchor" / "exact_ops_and_dp.csv"),
        ("EPYC", RES / "epyc_anchor" / "exact_ops_and_dp.csv"),
    ]:
        d = pd.read_csv(fp)
        g = d.groupby("arm")["global_capture_ratio"].mean() * 100.0
        x_lbls.append(lbl)
        c2_v.append(float(g["fixed_c2_merge"]))
        pq_v.append(float(g["past_qmass_sp1"]))
        ora_v.append(np.nan)

    xs = np.arange(len(x_lbls))
    bw = 0.24
    ax3.bar(xs - bw, c2_v, width=bw, facecolor=PAL["gray_fill"], edgecolor=PAL["ink"], linewidth=0.6, label="Fixed c=2")
    ax3.bar(xs, pq_v, width=bw, facecolor=PAL["mint_fill"], edgecolor=PAL["ink"], linewidth=0.6, label="PQ / EWMA")
    ora_xs = [i for i, v in enumerate(ora_v) if not np.isnan(v)]
    ora_vals = [v for v in ora_v if not np.isnan(v)]
    ax3.bar(np.array(ora_xs) + bw, ora_vals, width=bw, facecolor=PAL["apri_fill"], edgecolor=PAL["ink"], linewidth=0.6, label="1-blk Ora.")
    ax3.axhline(100.0, color=PAL["rose_edge"], linestyle="--", linewidth=0.9, label="Prop. 1")
    ax3.set_xticks(xs)
    ax3.set_xticklabels(x_lbls, fontsize=8.2, rotation=48, ha="right", rotation_mode="anchor")
    ax3.set_ylabel("Global Capture Ratio (%)", fontsize=8.6)
    ax3.set_ylim(0, 114)
    ax3.legend(loc="lower center", bbox_to_anchor=(0.50, 1.01), ncol=2, fontsize=8.2,
               columnspacing=0.5, handlelength=0.9, labelspacing=0.14)
    panel_tag(ax3, "C", x=-0.18, y=1.18)

    return save_both(fig, "fig6_phase_tombstone_transfer")


# ---------------------------------------------------------------------------
# Headline Figure: Unified 4-Trap & Real-Engine Ranking-Flip Matrix (fig_flip_matrix)
# ---------------------------------------------------------------------------
def make_fig_flip_matrix() -> Path:
    fp = RES / "paper_tables" / "ranking_flip_matrix.csv"
    df = pd.read_csv(fp)

    curated = [
        ("trap1_all_local_pooled", "Trap 1: Interp. vs Exec. (Local)", "Trap 1"),
        ("trap2_core_canon_leakage_exp2_ops", "Trap 2: Clock Leak (Core Pooled)", "Trap 2"),
        ("trap2_leakage_randhub_exact_scan_ops", "Trap 2: Clock Leak (Rand-Hub)", "Trap 2"),
        ("trap3_tombstone_pooled", "Trap 3: Merge vs Refresh (Pooled)", "Trap 3"),
        ("trap3_track_cf_pooled", "Trap 3: 1w vs 4w CF (Pooled)", "Trap 3"),
        ("trap3_cf_u10_uniform_s05", "Trap 3: 1w vs 4w CF (Uniform)", "Trap 3"),
        ("trap4_wikimedia_cpu_vs_ops", "Trap 4: Shared Host vs Exact Ops", "Trap 4"),
        ("trap4_dirty_vs_clean_18grids", "Trap 4: Dirty vs Clean (18 Grids)", "Trap 4"),
        ("trap4_abrupt_tsw_3grids", r"Trap 4: Abrupt $t_{sw}$ Dirty/Clean", "Trap 4"),
        ("trap4_xeon_all_regimes_mean_p99_us", "Trap 4: Xeon p99 (2w-16w)", "Trap 4"),
        ("exp7_arxiv_minilm384_uncap_ops_and_dp_exact_scan_ops", "Trap 4: arXiv Uncap vs Exact Ops", "Trap 4"),
        ("exp6_a1d_cross_numa_8w_wall_s", "Real: Cross-NUMA 8w Wall", "Real-Engine"),
        ("exp6_a1d_mt4_vs_iso1t_wall_s", "Regime: MT-4t vs 1T Wall", "Concurrency"),
        ("exp7_arxiv_5engines_summary_arxiv_cs_minilm384_cpu_vs_att", "Real: arXiv Cost vs Att.@.965", "Real-Engine"),
        ("exp9_m1_sharded_hnswlib_total_cpu_s", "Trap 2: 16-sh HNSW Bridge", "Trap 2"),
        ("colab_sift1m_drift_cpu_vs_att", "Real: 1M+F-HNSW Cost vs Att.", "Real-Engine"),
        ("exp9_s2_bigann_10m_wall_vs_floor", "Real: BigANN-10M Wall vs Flr.", "Real-Engine"),
        ("exp9_r1_a4_unmatched_cpu_vs_att", "Real: Rust A4 Cost vs Att.", "Real-Engine"),
        ("exp6_a3_crud_cpu_vs_attainment", "Real: CRUD CPU vs Att.@.965", "Real-Engine"),
        ("exp6_a4_unmatched_cpu_vs_attainment", "Real: Unmatched CPU vs Att.", "Real-Engine"),
    ]
    picked_rows = []
    for sid, lbl, tgrp in curated:
        m = df[df["suite_id"] == sid]
        if len(m):
            r = m.iloc[0].to_dict()
            r["disp"] = lbl
            r["tgrp"] = tgrp
            picked_rows.append(r)

    sub = pd.DataFrame(picked_rows).sort_values("pairwise_flip_pct", ascending=True).reset_index(drop=True)

    t_colors = {
        "Trap 1": (PAL["peri_fill"], PAL["peri_edge"]),
        "Trap 2": (PAL["iris_fill"], PAL["iris_edge"]),
        "Trap 3": (PAL["apri_fill"], PAL["apri_edge"]),
        "Trap 4": (PAL["rose_fill"], PAL["rose_edge"]),
        "Real-Engine": (PAL["mint_fill"], PAL["mint_edge"]),
        "Concurrency": (PAL["gray_fill"], PAL["gray_edge"]),
    }

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(6.85, 2.92),
        gridspec_kw={"width_ratios": [1.26, 1.12], "wspace": 0.42},
    )

    y = np.arange(len(sub))
    flips = sub["pairwise_flip_pct"].values
    for idx, r in sub.iterrows():
        fc, ec = t_colors[r["tgrp"]]
        mk = "D" if r["tgrp"] == "Real-Engine" else ("s" if r["tgrp"] == "Concurrency" else "o")
        ax1.hlines(idx, 0, r["pairwise_flip_pct"], color=ec, linewidth=1.7, zorder=2)
        ax1.scatter(r["pairwise_flip_pct"], idx, s=36, marker=mk, facecolor=fc, edgecolor=PAL["ink"], linewidth=0.7, zorder=4)
        frac = f"{r['pairwise_flip_pct']:.1f}% ({int(r['pair_flips'])}/{int(r['total_pairs'])})"
        ax1.text(r["pairwise_flip_pct"] + 4.0, idx, frac, va="center", ha="left", fontsize=8.3, color=PAL["ink"])

    ax1.set_yticks(y)
    ax1.set_yticklabels(sub["disp"].tolist(), fontsize=8.3)
    ax1.set_xlabel("Pairwise Policy/Engine Flip Rate (%)", fontsize=8.6)
    ax1.set_xlim(0, max(flips) + 52.0)
    ax1.set_ylim(-0.85, len(sub) - 0.15)
    panel_tag(ax1, "A", x=-0.56, y=1.03)

    ax2.axvline(0.85, color=PAL["ref"], linestyle="--", linewidth=0.8, zorder=1)
    for tgrp in ["Trap 1", "Trap 2", "Trap 3", "Trap 4", "Real-Engine", "Concurrency"]:
        ts = sub[sub["tgrp"] == tgrp]
        if not len(ts):
            continue
        fc, ec = t_colors[tgrp]
        mk = "D" if tgrp == "Real-Engine" else ("s" if tgrp == "Concurrency" else "o")
        ax2.scatter(ts["kendall_tau"], ts["top1_flip_pct"], s=44, marker=mk, facecolor=fc, edgecolor=ec,
                    linewidth=1.0, label=tgrp, zorder=4)

    annot_specs = {
        "trap1_all_local_pooled": ("Trap 1 (0%)", (0.80, -13.0), "center"),
        "exp6_a1d_mt4_vs_iso1t_wall_s": ("MT-4t Regime\n(0%)", (-0.22, 14.5), "center"),
        "trap2_core_canon_leakage_exp2_ops": ("Trap 2 Core\n(36.7%)", (1.10, 24.5), "center"),
        "trap4_dirty_vs_clean_18grids": ("Trap 4 18G\n(38.9%)", (0.20, 29.0), "center"),
        "exp6_a1d_cross_numa_8w_wall_s": ("Cross-NUMA\n8w (50%)", (1.14, 65.0), "center"),
        "trap4_xeon_all_regimes_mean_p99_us": ("Xeon p99\n(62.5%)", (-0.34, 56.0), "center"),
        "exp7_arxiv_minilm384_uncap_ops_and_dp_exact_scan_ops": ("arXiv Uncap\n(100%)", (0.94, 83.5), "center"),
        "exp7_arxiv_5engines_summary_arxiv_cs_minilm384_cpu_vs_att": ("arXiv Eng.\n(100%)", (0.20, 81.5), "center"),
        "exp9_r1_a4_unmatched_cpu_vs_att": ("Rust A4\n(100%)", (-0.42, 81.5), "center"),
    }
    for sid, (txt, xyt, ha_v) in annot_specs.items():
        m = sub[sub["suite_id"] == sid]
        if len(m):
            r = m.iloc[0]
            ax2.annotate(
                txt, xy=(r["kendall_tau"], r["top1_flip_pct"]), xytext=xyt,
                ha=ha_v, va="center", fontsize=8.2, color=PAL["ink"], bbox=PAD_BOX,
                arrowprops=dict(arrowstyle="-", color=PAL["gray_edge"], lw=0.6, shrinkA=0, shrinkB=4),
                zorder=6,
            )

    ax2.set_xlabel(r"Kendall's $\tau$ Rank Preservation", fontsize=8.6)
    ax2.set_ylabel("Top-1 Winner Flip Rate (%)", fontsize=8.6)
    ax2.set_xlim(-0.70, 1.44)
    ax2.set_ylim(-21, 117)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.50, 1.01), fontsize=8.2, ncol=3,
               columnspacing=0.45, handlelength=0.85, labelspacing=0.14)
    panel_tag(ax2, "B", x=-0.18, y=1.12)

    return save_both(fig, "fig_flip_matrix")


# ---------------------------------------------------------------------------
# Figure 7: Dual Xeon C++/Rust Traps, 1M Core-0 & Semantic Dim x Scale Crossover
# ---------------------------------------------------------------------------
def make_fig7() -> Path:
    a4 = pd.read_csv(RES / "exp6_real_engine_traps" / "a4_summary_by_config.csv")
    a1d = pd.read_csv(RES / "exp6_real_engine_traps" / "a1d_summary.csv")
    r1_a4 = pd.read_csv(RES / "exp9_rust_and_scale" / "r1_rust_a4_floor.csv")
    r1_a1d = pd.read_csv(RES / "exp9_rust_and_scale" / "r1_rust_a1d_numa.csv")
    s1_1m = pd.read_csv(RES / "exp9_rust_and_scale" / "s1_xeon_1m_gate4_clean.csv")
    d8_768 = pd.read_csv(RES / "exp8_colab_large_scale" / "arxiv140k_768d_engines.csv")
    d8_500 = pd.read_csv(RES / "exp8_colab_large_scale" / "arxiv500k_384d_engines.csv")
    d8_1m = pd.read_csv(RES / "exp8_colab_large_scale" / "arxiv1m_384d_engines.csv")

    r1_a4_g = r1_a4.groupby(["engine", "config"], as_index=False)[["total_cpu_s", "att_0965_pct"]].mean()
    r1_a4_g["total_cpu_ms"] = r1_a4_g["total_cpu_s"] * 1000.0
    r1_a4_g = r1_a4_g.rename(columns={"engine": "system", "config": "param_tag", "att_0965_pct": "attainment_0965_pct"})
    a4_plot = pd.concat([a4, r1_a4_g[r1_a4_g["system"] == "usearch_rust"]], ignore_index=True)

    fig = plt.figure(figsize=(7.05, 2.38))
    gs1 = fig.add_gridspec(1, 2, left=0.06, right=0.55, wspace=0.42, width_ratios=[1.12, 1.04])
    gs2 = fig.add_gridspec(1, 1, left=0.69, right=0.98)
    ax1 = fig.add_subplot(gs1[0, 0])
    ax2 = fig.add_subplot(gs1[0, 1])
    ax3 = fig.add_subplot(gs2[0, 0])

    eng_sty = {
        "faiss_ivf": (PAL["peri_mid"], "o", "FAISS-IVF", "-"),
        "diskann": (PAL["mint_mid"], "s", "FreshDiskANN", "-"),
        "hnswlib": (PAL["apri_mid"], "D", "hnswlib", "-"),
        "usearch": (PAL["rose_mid"], "^", "USearch", "-"),
        "usearch_rust": (PAL["iris_mid"], "v", "USearch-R", "--"),
    }
    for sys_name, (fc, mk, lbl, ls) in eng_sty.items():
        sub = a4_plot[a4_plot["system"] == sys_name].sort_values("total_cpu_ms")
        ax1.plot(sub["total_cpu_ms"] / 1000.0, sub["attainment_0965_pct"],
                 color=PAL["gray_edge"], linewidth=0.9, linestyle=ls, zorder=2)
        ax1.scatter(sub["total_cpu_ms"] / 1000.0, sub["attainment_0965_pct"],
                    s=34, marker=mk, facecolor=fc, edgecolor=PAL["ink"], linewidth=0.7, zorder=4, label=lbl)

    a4_ann = {
        ("faiss_ivf", "np16"): ("np16", (14, 75), "center"),
        ("faiss_ivf", "np24"): ("np24\n(1.1k)", (32, 128), "center"),
        ("diskann", "L64"): ("DiskANN\nL64 (5.4k)", (186, 129), "center"),
        ("hnswlib", "ef96"): ("hnswlib\nef96 (7.1k)", (216, 33), "center"),
        ("hnswlib", "ef64"): ("ef64\n(70.8%)", (62, 50), "center"),
        ("usearch_rust", "ef96"): ("US-R\nef96", (348, 128), "center"),
        ("usearch", "ef64"): ("ef64\n(71.7%)", (356, 52), "center"),
    }
    for _, r in a4_plot.iterrows():
        k = (r["system"], r["param_tag"])
        if k in a4_ann:
            txt, xyt, ha_v = a4_ann[k]
            ax1.annotate(txt, xy=(r["total_cpu_ms"] / 1000.0, r["attainment_0965_pct"]), xytext=xyt,
                         fontsize=8.1, ha=ha_v, va="center", color=PAL["ink"], bbox=PAD_BOX, zorder=6,
                         arrowprops=dict(arrowstyle="-", color=PAL["gray_edge"], lw=0.55, shrinkA=0, shrinkB=4))

    ax1.axhline(95.0, color=PAL["rose_edge"], linestyle="--", linewidth=0.9, zorder=3)
    ax1.text(350, 90.0, "95% Floor", fontsize=8.1, color=PAL["rose_edge"], fontweight="bold",
             ha="center", va="top", bbox=PAD_BOX, zorder=6)
    ax1.set_xlabel("Xeon Core-0 CPU (s)", fontsize=8.5)
    ax1.set_ylabel("Attainment % (@0.965)", fontsize=8.5)
    ax1.set_xlim(-24, 418)
    ax1.set_ylim(2, 148)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.50, 1.01), ncol=2, fontsize=8.1,
               columnspacing=0.35, handlelength=0.80, labelspacing=0.12)
    panel_tag(ax1, "A", x=-0.22, y=1.24)

    g1d = a1d.groupby(["regime", "system"], as_index=False)[["cpu_inflation_vs_iso", "p99_inflation_vs_iso"]].mean()
    r1_g = r1_a1d.groupby(["engine", "regime"], as_index=False)[["total_cpu_s", "p99_us"]].mean()
    r1_rows = []
    for eng in ("usearch_rust", "hnsw_rs"):
        iso = r1_g[(r1_g["engine"] == eng) & (r1_g["regime"] == "iso_numa0_1w")].iloc[0]
        for reg in ("ht_sibling_2w", "cross_numa_8w"):
            cur = r1_g[(r1_g["engine"] == eng) & (r1_g["regime"] == reg)].iloc[0]
            r1_rows.append({
                "regime": reg,
                "system": eng,
                "cpu_inflation_vs_iso": float(cur["total_cpu_s"] / iso["total_cpu_s"]),
                "p99_inflation_vs_iso": float(cur["p99_us"] / iso["p99_us"]),
            })
    g1d_all = pd.concat([g1d, pd.DataFrame(r1_rows)], ignore_index=True)

    sys_keys = ["annoy", "diskann", "faiss_ivf", "hnswlib", "usearch", "usearch_rust", "hnsw_rs"]
    sys_lbls = ["Annoy", "Disk.", "FAISS", "hnsw.", "US.", "US-R", "hnswR"]
    x2 = np.arange(len(sys_keys))
    bw = 0.25
    bars_cfg = [
        ("ht_sibling_2w", "cpu_inflation_vs_iso", "HT-2w CPU", PAL["peri_fill"]),
        ("cross_numa_8w", "cpu_inflation_vs_iso", "NUMA-8w CPU", PAL["apri_fill"]),
        ("cross_numa_8w", "p99_inflation_vs_iso", "NUMA-8w p99", PAL["rose_mid"]),
    ]
    cap_y = 2.38
    for idx, (reg, col, lbl, fc) in enumerate(bars_cfg):
        raw_vals = [float(g1d_all[(g1d_all["regime"] == reg) & (g1d_all["system"] == sk)][col].iloc[0]) for sk in sys_keys]
        plot_vals = [min(v, cap_y) for v in raw_vals]
        bs = ax2.bar(x2 + (idx - 1.0) * bw, plot_vals, width=bw, facecolor=fc, edgecolor=PAL["ink"], linewidth=0.58, label=lbl)
        for s_idx, (b, rv) in enumerate(zip(bs, raw_vals)):
            if rv > cap_y:
                bx = b.get_x() + b.get_width() / 2.0
                ax2.annotate(
                    f"{rv:.1f}\u00d7 (31.2\u00d7)",
                    xy=(bx, cap_y),
                    xytext=(3.3, 2.86),
                    ha="center", va="bottom", fontsize=8.1, fontweight="bold", color=PAL["rose_edge"],
                    arrowprops=dict(arrowstyle="->", color=PAL["rose_edge"], lw=0.75, shrinkA=0, shrinkB=2),
                )
            elif col == "cpu_inflation_vs_iso" and reg == "ht_sibling_2w" and s_idx in (0, 2, 5):
                ax2.text(b.get_x() + b.get_width() / 2.0, rv + 0.06, f"{rv:.2f}\u00d7",
                         ha="center", va="bottom", fontsize=8.0, color=PAL["ink"], rotation=90)

    ax2.axhline(1.0, color=PAL["ref"], linestyle="--", linewidth=0.8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(sys_lbls, fontsize=8.1, rotation=45, ha="right", rotation_mode="anchor")
    ax2.set_ylabel("Inflation vs. Iso-1w (\u00d7)", fontsize=8.5)
    ax2.set_ylim(0, 3.48)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.52, 1.01), ncol=2, fontsize=8.1,
               columnspacing=0.35, handlelength=0.80, labelspacing=0.12)
    panel_tag(ax2, "B", x=-0.26, y=1.24)

    s2_10m = pd.read_csv(RES / "exp9_rust_and_scale" / "s2_scale_1m_to_10m.csv")
    rows_c = []
    for wl_df, picks in [
        (d8_768, [("faiss_hnsw", "ef64", "768d F-HNSW"), ("hnswlib", "ef64", "768d hnswlib"), ("faiss_ivf", "np16", "768d IVF np16")]),
        (d8_1m, [("faiss_ivf", "np32", "1M-A IVF np32"), ("hnswlib", "ef64", "1M-A hnswlib"), ("diskann", "L64", "1M-A DiskANN")]),
    ]:
        for sy, pt, lbl in picks:
            m = wl_df[(wl_df["system"] == sy) & (wl_df["param_tag"] == pt)].iloc[0]
            rows_c.append((lbl, float(m["query_cpu_pct"]), float(m["maint_cpu_pct"]), f"{float(m['rel_cpu_vs_faiss_base']):.2f}\u00d7 ({float(m['attainment_0965_pct']):.0f}%)"))
    for df_src, tr_n, picks in [
        (s1_1m, "bigann_1m_drift", [("faiss_ivf", "np32", "1M-X IVF np32"), ("hnswlib", "ef128", "1M-X hnswlib"), ("usearch_rust", "ef128", "1M-X US-R")]),
        (s2_10m, "bigann_10m_drift", [("faiss_ivf", "np64", "10M IVF np64"), ("hnswlib", "ef96", "10M hnswlib"), ("usearch_rust", "ef96", "10M US-R")]),
    ]:
        for sy, pt, lbl in picks:
            m = df_src[(df_src["trace"] == tr_n) & (df_src["engine"] == sy) & (df_src["config"] == pt)].iloc[0]
            rows_c.append((lbl, float(m["query_pct"]), float(m["maint_pct"]), f"{float(m['total_cpu_s']):.1f}s ({float(m['att_0965_pct']):.0f}%)"))

    yc = np.arange(len(rows_c))
    q_pct = [r[1] for r in rows_c]
    m_pct = [r[2] for r in rows_c]
    ax3.barh(yc, q_pct, height=0.62, facecolor=PAL["peri_fill"], edgecolor=PAL["ink"], linewidth=0.58, label="Query %")
    ax3.barh(yc, m_pct, left=q_pct, height=0.62, facecolor=PAL["mint_fill"], edgecolor=PAL["ink"], linewidth=0.58, label="Maint %")
    for idx, (_, _, _, tag_str) in enumerate(rows_c):
        ax3.text(103.0, idx, tag_str, va="center", ha="left", fontsize=8.1, color=PAL["ink"])
    ax3.axhline(2.5, color=PAL["ref"], linestyle=":", linewidth=0.8)
    ax3.axhline(5.5, color=PAL["ref"], linestyle=":", linewidth=0.8)
    ax3.axhline(8.5, color=PAL["ref"], linestyle=":", linewidth=0.8)
    ax3.set_yticks(yc)
    ax3.set_yticklabels([r[0] for r in rows_c], fontsize=8.1)
    ax3.set_xticks([0, 25, 50, 75, 100])
    ax3.set_xticklabels(["0", "25", "50", "75", "100"], fontsize=8.1)
    ax3.set_xlabel("CPU Split (%) [Cost/Rel., Att.]", fontsize=8.5)
    ax3.set_xlim(0, 206)
    ax3.legend(loc="lower center", bbox_to_anchor=(0.48, 1.01), ncol=2, fontsize=8.1,
               columnspacing=0.35, handlelength=0.80, labelspacing=0.12)
    panel_tag(ax3, "C", x=-0.36, y=1.24)

    return save_both(fig, "fig7_real_engines_1m_arxiv")


def main() -> None:
    setup_style()
    pngs = [make_fig_flip_matrix(), make_fig2(), make_fig3(), make_fig4(), make_fig5(), make_fig6(), make_fig7()]
    print("Generated", len(pngs), "data figures.")


if __name__ == "__main__":
    main()
