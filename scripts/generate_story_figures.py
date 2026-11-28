#!/usr/bin/env python3
"""Story figures for the three-act revision (Act 1: quality floor; Act 3: pacing vs forecasting).

All numbers are computed from the result CSVs under results/; nothing is typed in by hand.
Style follows generate_paper_figures.py (Arial, >= 8 pt, no top/right spines, soft palette).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from generate_paper_figures import PAL, RES, panel_tag, save_both, setup_style

GRAPH = {"hnswlib", "hnswlib_mut", "usearch", "usearch_mut", "usearch_rust", "diskann", "faiss_hnsw", "hnsw_rs"}
IVF = {"faiss_ivf"}
FLOOR = 95.0


# ---------------------------------------------------------------------------
# Act 1 data: floor-calibrated engine comparisons
# ---------------------------------------------------------------------------
def _std(df: pd.DataFrame, sys_c: str, tag_c: str, cpu_s: pd.Series, att_c: str, rec_c: str, p99_c: str, label: str, min_att: Optional[pd.Series] = None) -> pd.DataFrame:
    return pd.DataFrame({
        "workload": label, "system": df[sys_c].astype(str), "tag": df[tag_c].astype(str),
        "cpu_s": cpu_s.astype(float), "att": df[att_c].astype(float),
        "min_att": (min_att.astype(float) if min_att is not None else df[att_c].astype(float)),
        "rec": df[rec_c].astype(float),
        "p99_us": df[p99_c].astype(float),
    })


def engine_configs() -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    e5 = pd.read_csv(RES / "paper_tables" / "tab_exp5_sift128.csv")
    parts.append(_std(e5, "system", "param_tag", e5["total_cpu_s"], "attainment_pct", "mean_recall", "p99_us", "SIFT 100K drift (Ryzen)"))
    a4 = pd.read_csv(RES / "exp6_real_engine_traps" / "a4_summary_by_config.csv")
    parts.append(_std(a4, "system", "param_tag", a4["total_cpu_ms"] / 1000.0, "attainment_0965_pct", "mean_recall", "mean_p99_us", "SIFT 100K drift (Xeon)"))
    a3 = pd.concat([pd.read_csv(RES / "exp6_real_engine_traps" / "a3_crud_engines.csv"),
                    pd.read_csv(RES / "exp6_real_engine_traps" / "a3b_crud_ivf_floor.csv")], ignore_index=True)
    a3 = a3[a3["status"] == "ok"].groupby(["workload", "system", "param_tag"], as_index=False).mean(numeric_only=True)
    for wl, lab in (("crud30v2", "SIFT 100K CRUD-30 (Xeon)"), ("crud50v2", "SIFT 100K CRUD-50 (Xeon)")):
        s = a3[a3["workload"] == wl]
        parts.append(_std(s, "system", "param_tag", s["total_cpu_ms"] / 1000.0, "attainment_0965_pct", "mean_recall", "mean_p99_us", lab))
    x = pd.read_csv(RES / "exp10a_mimo" / "exp10a_summary.csv")
    x = x[(x["status"] == "ok")].groupby(["workload", "system", "param_tag"], as_index=False).mean(numeric_only=True)
    for wl, lab in (("crud30v2", "SIFT 100K CRUD-30 (Ryzen)"), ("crud50v2", "SIFT 100K CRUD-50 (Ryzen)"), ("gist960", "GIST-960 100K (Ryzen)")):
        s = x[x["workload"] == wl]
        parts.append(_std(s, "system", "param_tag", s["total_cpu_ms"] / 1000.0, "att_0965_pct", "mean_recall", "mean_p99_us", lab))
    s1 = pd.read_csv(RES / "exp9_rust_and_scale" / "s1_xeon_1m_gate4_clean.csv")
    s1 = s1[s1["status"] == "ok"]
    for tr, lab in (("bigann_1m_drift", "BigANN 1M drift (Xeon)"), ("bigann_1m_crud20", "BigANN 1M CRUD-20 (Xeon)")):
        s = s1[s1["trace"] == tr]
        parts.append(_std(s, "engine", "config", s["total_cpu_s"], "att_0965_pct", "mean_recall", "p99_us", lab))
    s2 = pd.concat([pd.read_csv(RES / "exp9_rust_and_scale" / "s2_scale_1m_to_10m.csv"),
                    pd.read_csv(RES / "exp9_rust_and_scale" / "s2_5m_10m_ef_calibration.csv")], ignore_index=True)
    s2 = s2[s2["status"] == "ok"]
    for tr, lab in (("bigann_5m_drift", "BigANN 5M drift (8t)"), ("bigann_5m_crud20", "BigANN 5M CRUD-20 (8t)"),
                    ("bigann_10m_drift", "BigANN 10M drift (8t)"), ("bigann_10m_crud20", "BigANN 10M CRUD-20 (8t)")):
        s = s2[s2["trace"] == tr].drop_duplicates(subset=["engine", "config"], keep="last")
        parts.append(_std(s, "engine", "config", s["total_cpu_s"], "att_0965_pct", "mean_recall", "p99_us", lab))
    b1p = RES / "exp11" / "exp11b1_baremetal_real768.csv"
    b1ext = RES / "exp11" / "exp11b1_ext_graph_knobs.csv"
    if b1p.exists():
        b1_parts = [pd.read_csv(b1p)]
        if b1ext.exists():
            b1_parts.append(pd.read_csv(b1ext))
        b1 = pd.concat(b1_parts, ignore_index=True).drop_duplicates(
            subset=["workload", "seed", "system", "param_tag"], keep="last"
        )
        att_min = b1.groupby(["workload", "system", "param_tag"])["att_0965_pct"].min().rename("att_min")
        b1 = b1.groupby(["workload", "system", "param_tag"], as_index=False).mean(numeric_only=True).join(att_min, on=["workload", "system", "param_tag"])
        b1["att_0965_pct"] = b1["att_min"]  # a config counts as floor-compliant only if every seed meets the floor
        for wl in ["140K_Drift", "500K_Drift", "500K_CRUD20", "1M_Drift"]:
            s = b1[b1["workload"] == wl]
            if s.empty:
                continue
            p99c = "mean_p99_us" if "mean_p99_us" in s.columns else "p99_us"
            parts.append(_std(s, "system", "param_tag", s["total_cpu_s"], "att_0965_pct", "mean_recall", p99c, f"Wiki-768 {wl.replace('_', ' ')} (Xeon)"))
    df = pd.concat(parts, ignore_index=True)
    df["family"] = np.where(df["system"].isin(IVF), "IVF", np.where(df["system"].isin(GRAPH), "Graph", "Other"))
    return df


def calibrated_pairs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wl, s in df.groupby("workload", sort=False):
        ok = s[s["att"] >= FLOOR]
        ivf, gr = ok[ok["family"] == "IVF"], ok[ok["family"] == "Graph"]
        if ivf.empty or gr.empty:
            rows.append({"workload": wl, "ivf_ok": not ivf.empty, "graph_ok": not gr.empty})
            continue
        bi, bg = ivf.loc[ivf["cpu_s"].idxmin()], gr.loc[gr["cpu_s"].idxmin()]
        bgp = gr.loc[gr["p99_us"].idxmin()]
        rows.append({"workload": wl, "ivf_ok": True, "graph_ok": True,
                     "ivf_cfg": f"{bi['system']} {bi['tag']}", "ivf_cpu": bi["cpu_s"], "ivf_p99": bi["p99_us"],
                     "graph_cfg": f"{bg['system']} {bg['tag']}", "graph_cpu": bg["cpu_s"],
                     "graph_p99_cfg": f"{bgp['system']} {bgp['tag']}", "graph_p99": bgp["p99_us"],
                     "cpu_ratio_graph_over_ivf": bg["cpu_s"] / bi["cpu_s"], "p99_ratio_ivf_over_graph": bi["p99_us"] / bgp["p99_us"]})
    return pd.DataFrame(rows)


def floor_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Per workload: rank configurations by CPU among those with mean recall >= 0.95 (common practice) vs.
    among floor-compliant ones (others ranked after, by CPU). Reports pair flips, Kendall tau, and whether the
    winning configuration / engine / family changes, plus how much more CPU the compliant winner costs."""
    from scipy.stats import kendalltau
    rows = []
    for wl, s in df[df["family"].isin(["IVF", "Graph", "Other"])].groupby("workload", sort=False):
        r0 = (~(s["rec"] >= 0.95)).astype(int) * 1e9 + s["cpu_s"]
        r1 = (~(s["att"] >= FLOOR)).astype(int) * 1e9 + s["cpu_s"]
        idx = list(s.index)
        pairs = [(a, b) for i, a in enumerate(idx) for b in idx[i + 1:]]
        flips = sum(np.sign(r0[a] - r0[b]) != np.sign(r1[a] - r1[b]) for a, b in pairs)
        w0, w1 = r0.idxmin(), r1.idxmin()
        rows.append({"workload": wl, "n_cfg": len(s), "pair_flip_pct": 100.0 * flips / max(len(pairs), 1),
                     "kendall_tau": kendalltau(r0, r1).statistic if len(s) > 2 else np.nan,
                     "cfg_change": bool(w0 != w1), "engine_change": bool(s.loc[w0, "system"] != s.loc[w1, "system"]),
                     "family_change": bool(s.loc[w0, "family"] != s.loc[w1, "family"]),
                     "mean_winner": f"{s.loc[w0, 'system']} {s.loc[w0, 'tag']}", "mean_winner_att": s.loc[w0, "att"],
                     "floor_winner": f"{s.loc[w1, 'system']} {s.loc[w1, 'tag']}",
                     "extra_cpu_pct": 100.0 * (s.loc[w1, "cpu_s"] / s.loc[w0, "cpu_s"] - 1.0)})
    # engine level: each engine at its cheapest qualifying configuration
    eng = []
    for wl, s in df.groupby("workload", sort=False):
        best0 = s[s["rec"] >= 0.95].groupby("system")["cpu_s"].min()
        best1 = s[s["att"] >= FLOOR].groupby("system")["cpu_s"].min()
        systems = sorted(set(best0.index) | set(best1.index))
        b0 = {k: best0.get(k, np.inf) for k in systems}
        b1 = {k: best1.get(k, np.inf) for k in systems}
        for i, a in enumerate(systems):
            for b in systems[i + 1:]:
                eng.append({"workload": wl, "flip": bool(np.sign(b0[a] - b0[b]) != np.sign(b1[a] - b1[b]))})
    out = pd.DataFrame(rows)
    e = pd.DataFrame(eng)
    out.attrs["engine_pair_flips"] = (int(e["flip"].sum()), len(e))
    pd.DataFrame([{"engine_pair_flips": int(e["flip"].sum()), "engine_pairs": len(e)}]).to_csv(
        RES / "paper_tables" / "act1_floor_engine_pairs.csv", index=False)
    return out


def short_label(w: str) -> str:
    rep = [("SIFT ", ""), ("BigANN ", "BigANN "), ("Wiki-768 ", "768-d "), ("(Ryzen)", "R"), ("(Xeon)", "X"), ("(8t)", "X,8t"),
           ("CRUD-", "CRUD"), (" drift", " drift"), ("_", " ")]
    for a, b in rep:
        w = w.replace(a, b)
    return w.replace(" Drift", " drift")


def make_fig_act1() -> Path:
    df = engine_configs()
    pairs = calibrated_pairs(df)
    out_csv = RES / "paper_tables" / "act1_floor_calibrated_pairs.csv"
    pairs.to_csv(out_csv, index=False)
    df.to_csv(RES / "paper_tables" / "act1_engine_configs.csv", index=False)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7.1, 2.75), gridspec_kw={"width_ratios": [1.0, 0.72, 1.18], "wspace": 0.95})
    # (A) mean recall vs attainment
    sub = df[df["family"].isin(["IVF", "Graph"]) & df["rec"].between(0.85, 1.0)]
    hidden = sub[(sub["rec"] >= 0.95) & (sub["att"] < FLOOR)]
    ax1.axhspan(-3, FLOOR, xmin=(0.95 - 0.85) / 0.15, xmax=1.0, facecolor=PAL["rose_fill"], alpha=0.35, linewidth=0)
    for fam, fc, ec, mk in (("Graph", PAL["peri_fill"], PAL["peri_edge"], "o"), ("IVF", PAL["mint_fill"], PAL["mint_edge"], "s")):
        s = sub[sub["family"] == fam]
        ax1.scatter(s["rec"], s["att"], s=16, marker=mk, facecolor=fc, edgecolor=ec, linewidth=0.6, label=fam, zorder=3)
    ax1.axhline(FLOOR, color=PAL["ink"], linestyle="--", linewidth=0.7)
    ax1.set_xlim(0.85, 1.0)
    ax1.set_ylim(-3, 103)
    ax1.set_xlabel("Mean Recall@10")
    ax1.set_ylabel("Windows meeting 0.965 (%)")
    ax1.text(0.853, 72, f"shaded: {len(hidden)} configs\nwith mean $\\geq$0.95\nbut floor missed", fontsize=7.4, color=PAL["rose_edge"], ha="left", va="center")
    ax1.legend(loc="upper left", fontsize=7.8, handletextpad=0.2, borderaxespad=0.1)
    panel_tag(ax1, "A", x=-0.30, y=1.02)

    # (B) mean-recall-matched vs floor-matched winner
    fr = floor_rankings(df)
    fr.to_csv(RES / "paper_tables" / "act1_floor_rankings.csv", index=False)
    fb = fr.sort_values("extra_cpu_pct").reset_index(drop=True)
    y = np.arange(len(fb))
    cols = [PAL["rose_edge"] if c else PAL["gray_edge"] for c in fb["cfg_change"]]
    ax2.hlines(y, 0, fb["extra_cpu_pct"], color=PAL["gray_edge"], linewidth=0.8)
    ax2.scatter(fb["extra_cpu_pct"], y, s=18, c=cols, zorder=3)
    ax2.axvline(0, color=PAL["ref"], linewidth=0.7)
    ax2.set_yticks([])
    ax2.set_xlim(-8, 115)
    ax2.set_xlabel("Extra CPU of floor winner (%)")
    ax2.set_ylabel(f"{len(fb)} workloads (sorted)")
    n_top = int(fb["cfg_change"].sum())
    n_eng = int(fb["engine_change"].sum())
    ax2.text(0.5, -0.19, f"red: mean-recall winner\nmisses floor ({n_top}/{len(fb)})", transform=ax2.transAxes, ha="center", va="top", fontsize=7.4, color=PAL["rose_edge"])
    ax2.text(0.5, -0.315, f"engine winner same: {len(fb) - n_eng}/{len(fb)}", transform=ax2.transAxes, ha="center", va="top", fontsize=7.4, color=PAL["ink"])
    panel_tag(ax2, "B", x=-0.12, y=1.02)

    # (C) floor-calibrated ratios
    pp = pairs.dropna(subset=["cpu_ratio_graph_over_ivf"]).reset_index(drop=True)
    yy = np.arange(len(pp))[::-1]
    ax3.axvline(1.0, color=PAL["ink"], linewidth=0.7)
    ax3.scatter(pp["cpu_ratio_graph_over_ivf"], yy, s=22, marker="s", facecolor=PAL["mint_fill"], edgecolor=PAL["mint_edge"], linewidth=0.7, label="CPU: graph / IVF", zorder=3)
    ax3.scatter(pp["p99_ratio_ivf_over_graph"], yy, s=22, marker="o", facecolor=PAL["rose_fill"], edgecolor=PAL["rose_edge"], linewidth=0.7, label="$p_{99}$: IVF / graph", zorder=3)
    ax3.set_xscale("log")
    ax3.set_xlim(0.4, 40)
    ax3.set_yticks(yy)
    ax3.set_yticklabels([short_label(w) for w in pp["workload"]], fontsize=7.4)
    ax3.set_xlabel("Ratio at matched floor")
    ax3.legend(loc="upper center", bbox_to_anchor=(0.45, 1.17), ncol=2, fontsize=7.2, handletextpad=0.1, columnspacing=0.6, borderaxespad=0.1)
    panel_tag(ax3, "C", x=-0.62, y=1.02)
    return save_both(fig, "fig_act1_floor")


# ---------------------------------------------------------------------------
# Act 3: where / when / foresight
# ---------------------------------------------------------------------------
GEN_ORDER = ["stationary", "gradual", "abrupt1", "abrupt2", "abrupt3", "periodic"]
GEN_LABEL = {"stationary": "Stat.", "gradual": "Grad.", "abrupt1": "1\nshift", "abrupt2": "2\nshifts", "abrupt3": "3\nshifts", "periodic": "Period."}


def make_fig_act3() -> Path:
    r3_p = RES / "exp11" / "r3_exp1_orthogonal_decomp.csv"
    d = pd.read_csv(r3_p if r3_p.exists() else RES / "exp11" / "exp11a_pacing_generality.csv")
    val_col = "gcr_lam0_0_pct" if "gcr_lam0_0_pct" in d.columns else "gcr_pct"
    p = d.pivot_table(index=["gen", "coupling", "R", "J", "B", "seed"], columns=["policy", "mode"], values=val_col)
    rr_col = ("rr", "none") if ("rr", "none") in p.columns else ("fixed_c2", "none")
    series = {
        "RR, greedy": p[rr_col],
        "Past-QMass, greedy": p[("past_qmass", "none")],
        "Past-QMass, paced": p[("past_qmass", "uniform")],
        "1-blk oracle, paced": p[("oracle_1blk", "uniform")],
    }
    m = pd.DataFrame({k: v.groupby(level="gen").mean() for k, v in series.items()}).loc[GEN_ORDER]
    dec = pd.DataFrame({
        "Where (+PQ greedy)": (p[("past_qmass", "none")] - p[rr_col]).groupby(level="gen").mean(),
        "When (+pacing)": (p[("past_qmass", "uniform")] - p[("past_qmass", "none")]).groupby(level="gen").mean(),
        "Foresight (+oracle)": (p[("oracle_1blk", "uniform")] - p[("past_qmass", "uniform")]).groupby(level="gen").mean(),
    }).loc[GEN_ORDER]
    m.to_csv(RES / "paper_tables" / "act3_pacing_by_gen.csv")
    dec.to_csv(RES / "paper_tables" / "act3_where_when_foresight.csv")

    b_path = RES / "exp11" / "exp11b_paced_bridge.csv"
    has_c = b_path.exists()
    ncol = 3 if has_c else 2
    fig, axes = plt.subplots(1, ncol, figsize=(7.1, 2.75), gridspec_kw={"width_ratios": [1.45, 1.0, 1.0][:ncol], "wspace": 0.45})
    ax1, ax2 = axes[0], axes[1]
    colors = [(PAL["gray_fill"], PAL["gray_edge"]), (PAL["apri_fill"], PAL["apri_edge"]), (PAL["mint_fill"], PAL["mint_edge"]), (PAL["iris_fill"], PAL["iris_edge"])]
    x = np.arange(len(GEN_ORDER))
    w = 0.2
    for i, (lab, (fc, ec)) in enumerate(zip(m.columns, colors)):
        ax1.bar(x + (i - 1.5) * w, m[lab], width=w, facecolor=fc, edgecolor=ec, linewidth=0.6, label=lab)
    ax1.set_xticks(x)
    ax1.set_xticklabels([GEN_LABEL[g] for g in GEN_ORDER], fontsize=7.8)
    ax1.set_ylabel("GCR vs. Prop. 1 optimum (%)")
    ax1.set_ylim(0, 112)
    ax1.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7.3, columnspacing=0.8, handlelength=1.0)
    panel_tag(ax1, "A", x=-0.16, y=1.24)

    dcols = [(PAL["peri_fill"], PAL["peri_edge"]), (PAL["mint_fill"], PAL["mint_edge"]), (PAL["iris_fill"], PAL["iris_edge"])]
    yy = np.arange(len(GEN_ORDER))[::-1]
    h = 0.26
    for i, (lab, (fc, ec)) in enumerate(zip(dec.columns, dcols)):
        ax2.barh(yy + (1 - i) * h, dec[lab], height=h, facecolor=fc, edgecolor=ec, linewidth=0.6, label=lab)
    ax2.set_yticks(yy)
    ax2.set_yticklabels([GEN_LABEL[g].replace("\n", " ") for g in GEN_ORDER], fontsize=7.8)
    ax2.set_xlabel("GCR gained (pp)")
    ax2.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=1, fontsize=7.2, handlelength=1.0, labelspacing=0.15)
    panel_tag(ax2, "B", x=-0.38, y=1.24)

    if has_c:
        ax3 = axes[2]
        b = pd.read_csv(b_path)
        b["grp"] = np.where(b["trace"].str.startswith("abrupt") & (b["tsw"] >= 20), "Shift after\nexhaustion", np.where(b["trace"].str.startswith("abrupt"), "Shift\nbefore", "No\nshift"))
        noop = b[b["policy"] == "noop"].set_index(["engine", "trace", "tsw"])["total_cpu_s"]
        b = b.join(noop.rename("noop_cpu"), on=["engine", "trace", "tsw"])
        b["saved_pct"] = 100.0 * (b["noop_cpu"] - b["total_cpu_s"]) / b["noop_cpu"]
        pols = [("past_qmass", "Greedy", PAL["apri_fill"], PAL["apri_edge"]), ("past_qmass_paced", "Paced", PAL["mint_fill"], PAL["mint_edge"]), ("oracle_paced", "Oracle, paced", PAL["iris_fill"], PAL["iris_edge"])]
        grps = [g for g in ("Shift after\nexhaustion", "Shift\nbefore", "No\nshift") if g in set(b["grp"])]
        xg = np.arange(len(grps))
        for i, (pol, lab, fc, ec) in enumerate(pols):
            vals = [b[(b["grp"] == g) & (b["policy"] == pol)]["gcr_ops_pct"].mean() for g in grps]
            ax3.bar(xg + (i - 1) * 0.26, vals, width=0.26, facecolor=fc, edgecolor=ec, linewidth=0.6, label=lab)
        ax3.set_xticks(xg)
        ax3.set_xticklabels(grps, fontsize=7.2)
        ax3.set_ylabel("Live-engine GCR (%)")
        ax3.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=7.2, handlelength=1.0, labelspacing=0.15)
        panel_tag(ax3, "C", x=-0.34, y=1.24)
    return save_both(fig, "fig_act3_pacing")


if __name__ == "__main__":
    setup_style()
    make_fig_act1()
    make_fig_act3()


def check_family(row) -> Optional[str]:
    sid, metric, trap = str(row["suite_id"]), str(row["metric"]), str(row["trap"])
    if "att" in metric.lower() or "floor" in metric.lower() or "att" in sid or "floor" in sid:
        return "Q: quality floor"
    if sid.startswith("exp9_m3") or sid.startswith("exp9_m1") or "wikimedia_cpu_vs_ops" in sid:
        return None
    if trap.startswith("Concurrency") or "_mt_" in sid or "unpinned_16t" in sid or "_mt4_" in sid or "_mt8_" in sid:
        return "Threads (regime)"
    if sid.startswith("trap4_"):
        return "P: placement, policies"
    if trap.startswith("Trap 4"):
        return "P: placement, engines"
    if trap.startswith("Trap 1"):
        return "C: cadence"
    if trap.startswith("Trap 2"):
        return "S: split"
    if sid.startswith("trap3_tombstone"):
        return "H: refresh/merge"
    if trap.startswith("Trap 3"):
        return "H: horizon"
    return None


FAM_ORDER = ["Q: quality floor", "H: horizon", "P: placement, policies", "S: split", "H: refresh/merge", "C: cadence", "P: placement, engines", "Threads (regime)"]


def make_fig_checks() -> Path:
    fm = pd.read_csv(RES / "paper_tables" / "ranking_flip_matrix.csv")
    fm["family"] = fm.apply(check_family, axis=1)
    fm = fm.dropna(subset=["family"])
    # Q: the flip-matrix "cpu_vs_attainment" suites compare a cost ranking with a quality ranking, not a ranking
    # with and without the floor; use the mean-recall-matched vs floor-matched configuration rankings instead.
    fr = pd.read_csv(RES / "paper_tables" / "act1_floor_rankings.csv")
    fm = fm[fm["family"] != "Q: quality floor"]
    q = pd.DataFrame({"suite_id": "act1_" + fr["workload"], "family": "Q: quality floor",
                      "pairwise_flip_pct": fr["pair_flip_pct"], "top1_flips": fr["cfg_change"].astype(int)})
    fm = pd.concat([fm, q], ignore_index=True)
    summ = fm.groupby("family").agg(n=("suite_id", "count"), median=("pairwise_flip_pct", "median"),
                                     top1=("top1_flips", lambda s: int((s > 0).sum()))).reindex(FAM_ORDER)
    summ.to_csv(RES / "paper_tables" / "checks_impact_by_family.csv")
    fig, ax = plt.subplots(figsize=(3.45, 2.35))
    rng = np.random.default_rng(3)
    for i, fam in enumerate(FAM_ORDER):
        s = fm[fm["family"] == fam]
        y = len(FAM_ORDER) - 1 - i
        jit = rng.uniform(-0.18, 0.18, size=len(s))
        top = s["top1_flips"] > 0
        ax.scatter(s.loc[~top, "pairwise_flip_pct"], y + jit[~top.values], s=11, facecolor=PAL["gray_fill"], edgecolor=PAL["gray_edge"], linewidth=0.5, zorder=2)
        ax.scatter(s.loc[top, "pairwise_flip_pct"], y + jit[top.values], s=11, facecolor=PAL["rose_fill"], edgecolor=PAL["rose_edge"], linewidth=0.5, zorder=3)
        med = summ.loc[fam, "median"]
        ax.plot([med, med], [y - 0.3, y + 0.3], color=PAL["ink"], linewidth=1.2, zorder=4)
        ax.text(103, y, f"{int(summ.loc[fam, 'top1'])}/{int(summ.loc[fam, 'n'])}", va="center", ha="left", fontsize=7.2, color=PAL["rose_edge"])
    ax.set_yticks(range(len(FAM_ORDER)))
    ax.set_yticklabels(FAM_ORDER[::-1], fontsize=7.4)
    ax.set_xlim(-3, 101)
    ax.set_xlabel("Configuration or policy pairs that swap order (%)")
    ax.text(103, len(FAM_ORDER) - 0.35, "winner\nchanged", fontsize=6.9, color=PAL["rose_edge"], ha="left", va="bottom")
    return save_both(fig, "fig_checks_impact")


if __name__ == "__main__":
    make_fig_checks()
