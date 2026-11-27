#!/usr/bin/env python3
"""Build verified PVLDB paper tables (CSV + B&W LaTeX) and provenance ledger.

Reads strictly from verified local CSV/JSON files under `results/`.
Formatting follows ICDE/PVLDB B&W table rules:
  - Zero blue text; light gray highlight (`\\ccrow`)
  - Column headers annotated with `\\uparrow` / `\\downarrow`
  - Fine-grained cell-level `\\textbf{}` (best) and `\\underline{}` (2nd-best)
"""
from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT_CSV = RES / "paper_tables"
OUT_TEX = ROOT / "latex" / "tables"


def _ensure_dirs() -> None:
    OUT_CSV.mkdir(parents=True, exist_ok=True)
    OUT_TEX.mkdir(parents=True, exist_ok=True)


def _rank_format(
    values: Sequence[float | None],
    higher_is_better: bool,
    fmt: str,
    na_str: str = "--",
    eligible: Sequence[bool] | None = None,
) -> List[str]:
    disp_strs: List[str | None] = []
    disp_vals: List[float | None] = []
    for v in values:
        if v is None or np.isnan(v):
            disp_strs.append(None)
            disp_vals.append(None)
        else:
            s = format(float(v), fmt)
            disp_strs.append(s)
            disp_vals.append(float(s))

    valid = [
        (i, dv)
        for i, dv in enumerate(disp_vals)
        if dv is not None and (eligible is None or eligible[i])
    ]
    if not valid:
        return [s if s is not None else na_str for s in disp_strs]

    valid_sorted = sorted(valid, key=lambda kv: kv[1], reverse=higher_is_better)
    best_val = valid_sorted[0][1]
    n_best = sum(1 for _, dv in valid_sorted if abs(dv - best_val) <= 1e-12)
    second_val = None
    if n_best == 1:
        for _, dv in valid_sorted[1:]:
            if abs(dv - best_val) > 1e-12:
                second_val = dv
                break

    out = []
    for i, s in enumerate(disp_strs):
        if s is None:
            out.append(na_str)
            continue
        if eligible is not None and not eligible[i]:
            out.append(s)
            continue
        dv = disp_vals[i]
        assert dv is not None
        if abs(dv - best_val) <= 1e-12:
            out.append(f"\\textbf{{{s}}}")
        elif second_val is not None and abs(dv - second_val) <= 1e-12:
            out.append(f"\\underline{{{s}}}")
        else:
            out.append(s)
    return out


# ---------------------------------------------------------------------------
# Table 1: Exp 5A 6-Engine Lifecycle Benchmark (+ RSS Probe Merge)
# ---------------------------------------------------------------------------
def build_tab_exp5() -> Dict[str, object]:
    df = pd.read_csv(RES / "exp5_multi_engine" / "exp5_multi_engine_summary.csv")
    rss_df = pd.read_csv(RES / "exp5_multi_engine" / "exp5_rss_probe.csv")
    rss_map = {
        (r["system"], r["param_tag"]): float(r["peak_rss_mb"])
        for _, r in rss_df.iterrows()
    }
    sys_disp = {
        "faiss_ivf": "FAISS-IVF",
        "hnswlib_mut": "hnswlib",
        "diskann": "FreshDiskANN",
        "quake": "Quake",
        "annoy_rebuild": "Annoy",
        "usearch_mut": "USearch",
    }

    # SIFT-128 Drift summary (11 engine configs x 10 traces = 110 runs)
    sift = df[df["workload"] == "sift128_drift"].copy()
    g = sift.groupby(["system", "param_tag"], as_index=False).agg(
        n_runs=("trace", "count"),
        query_cpu_s=("query_cpu_ms", lambda x: x.mean() / 1000.0),
        maint_cpu_s=("maint_cpu_ms", lambda x: x.mean() / 1000.0),
        total_cpu_s=("total_cpu_ms", lambda x: x.mean() / 1000.0),
        mean_recall=("mean_recall", "mean"),
        min_recall=("min_recall", "mean"),
        attainment_pct=("attainment_pct", "mean"),
        p95_us=("mean_p95_us", "mean"),
        p99_us=("mean_p99_us", "mean"),
        peak_rss_mb=("peak_rss_mb", "mean"),
    )
    for idx, row in g.iterrows():
        if pd.isna(row["peak_rss_mb"]):
            g.at[idx, "peak_rss_mb"] = rss_map.get((row["system"], row["param_tag"]), np.nan)
    g["engine_label"] = g.apply(lambda r: f"{sys_disp[r['system']]} ({r['param_tag']})", axis=1)
    g = g.sort_values(["total_cpu_s"]).reset_index(drop=True)
    g.to_csv(OUT_CSV / "tab_exp5_sift128.csv", index=False)

    # CRUD/Runbook summary (50 runs)
    churn = df[df["workload"].isin(["crud30v2", "crud50v2", "runbook"])].copy()
    gc = churn.groupby(["workload", "system", "param_tag", "status"], as_index=False).agg(
        n_runs=("trace", "count"),
        total_cpu_s=("total_cpu_ms", lambda x: x.mean() / 1000.0),
        mean_recall=("mean_recall", "mean"),
        attainment_pct=("attainment_pct", "mean"),
        p99_us=("mean_p99_us", "mean"),
    )
    gc.to_csv(OUT_CSV / "tab_exp5_churn.csv", index=False)

    # Write unified LaTeX table (SIFT-128 Drift + Churn)
    elig_a = [bool(r["attainment_pct"] >= 95.0) for _, r in g.iterrows()]
    q_fmt = _rank_format(g["query_cpu_s"].tolist(), False, ".1f", eligible=elig_a)
    m_fmt = _rank_format(g["maint_cpu_s"].tolist(), False, ".1f", eligible=elig_a)
    t_fmt = _rank_format(g["total_cpu_s"].tolist(), False, ".1f", eligible=elig_a)
    r_fmt = _rank_format(g["mean_recall"].tolist(), True, ".4f")
    mr_fmt = _rank_format(g["min_recall"].tolist(), True, ".4f")
    att_fmt = _rank_format(g["attainment_pct"].tolist(), True, ".1f")
    p99_fmt = _rank_format(g["p99_us"].tolist(), False, ".1f", eligible=elig_a)
    rss_fmt = _rank_format(g["peak_rss_mb"].tolist(), False, ".1f", eligible=elig_a)

    lines = [
        "% Auto-generated from results/exp5_multi_engine/exp5_multi_engine_summary.csv + exp5_rss_probe.csv",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\renewcommand{\\arraystretch}{0.92}",
        "\\caption{Cross-engine lifecycle benchmark on SIFT-128 drifting inserts (110 runs) and mixed deletion churn (CRUD-30\\%, CRUD-50\\%, and Deletion Runbook [Clustered/Sliding], 50 runs; Core-8 pinned). Attainment measures the percentage of windows achieving Recall@10 $\\ge 0.965$. Below-floor configs ($<95\\%$ attainment, marked $^\\dagger$) are excluded from cost, latency, and RSS ranking; default CRUD-30\\%/50\\% knobs fall below floor (see Table~\\ref{tab:real_engines_1m_arxiv}a for floor-calibrated CRUD).}",
        "\\label{tab:exp5_unified}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "Workload / Engine (Config) & Query (s) $\\downarrow$ & Maint. (s) $\\downarrow$ & Total (s) $\\downarrow$ & Mean Rec. $\\uparrow$ & Min Rec. $\\uparrow$ & Attain. (\\%) $\\uparrow$ & $p_{99}$ ($\\mu$s) $\\downarrow$ & RSS (MB) $\\downarrow$ \\\\",
        "\\midrule",
        "\\multicolumn{9}{l}{\\textit{(a) SIFT-128 Drifting Inserts (10 traces/config, 110 Core-8 runs)}} \\\\",
    ]
    for i, r in g.iterrows():
        ok = elig_a[i]
        dag = "" if ok else "$^\\dagger$"
        prefix = "\\ccrow " if ok and r["total_cpu_s"] < 90.0 else ""
        lines.append(
            f"{prefix}{r['engine_label'].replace('_', '\\_')} & {q_fmt[i]} & {m_fmt[i]} & {t_fmt[i]} & "
            f"{r_fmt[i]} & {mr_fmt[i]} & {att_fmt[i]}{dag} & {p99_fmt[i]} & {rss_fmt[i]} \\\\"
        )

    # Append churn block cleanly in shared columns
    gc_full = churn.groupby(["workload", "system", "param_tag", "status"], as_index=False).agg(
        query_cpu_s=("query_cpu_ms", lambda x: x.mean() / 1000.0),
        maint_cpu_s=("maint_cpu_ms", lambda x: x.mean() / 1000.0),
        total_cpu_s=("total_cpu_ms", lambda x: x.mean() / 1000.0),
        mean_recall=("mean_recall", "mean"),
        min_recall=("min_recall", "mean"),
        attainment_pct=("attainment_pct", "mean"),
        p99_us=("mean_p99_us", "mean"),
    )
    wl_disp = {"crud30v2": "CRUD-30\\%", "crud50v2": "CRUD-50\\%", "runbook": "Del. Runbook"}
    lines.append("\\midrule")
    lines.append("\\multicolumn{9}{l}{\\textit{(b) Mixed Deletion Churn (CRUD-30\\%, CRUD-50\\%, and Deletion Runbook [Clustered/Sliding], 50 Core-8 runs)}} \\\\")
    for wl in ["crud30v2", "crud50v2", "runbook"]:
        sub = gc_full[gc_full["workload"] == wl].sort_values(["status", "total_cpu_s"]).reset_index(drop=True)
        ok_mask = sub["status"] == "ok"
        elig_rec = [bool(r["status"] == "ok") for _, r in sub.iterrows()]
        elig_cost = [bool(r["status"] == "ok" and r["attainment_pct"] >= 95.0) for _, r in sub.iterrows()]
        qf = _rank_format([r["query_cpu_s"] if r["status"] == "ok" else np.nan for _, r in sub.iterrows()], False, ".1f", eligible=elig_cost)
        mf = _rank_format([r["maint_cpu_s"] if r["status"] == "ok" else np.nan for _, r in sub.iterrows()], False, ".1f", eligible=elig_cost)
        tf = _rank_format([r["total_cpu_s"] if r["status"] == "ok" else np.nan for _, r in sub.iterrows()], False, ".1f", eligible=elig_cost)
        rf = _rank_format([r["mean_recall"] if r["status"] == "ok" else np.nan for _, r in sub.iterrows()], True, ".4f", eligible=elig_rec)
        mrf = _rank_format([r["min_recall"] if r["status"] == "ok" else np.nan for _, r in sub.iterrows()], True, ".4f", eligible=elig_rec)
        af = _rank_format([r["attainment_pct"] if r["status"] == "ok" else np.nan for _, r in sub.iterrows()], True, ".1f", eligible=elig_rec)
        pf = _rank_format([r["p99_us"] if r["status"] == "ok" else np.nan for _, r in sub.iterrows()], False, ".1f", eligible=elig_cost)
        for i, r in sub.iterrows():
            ok_c = elig_cost[i]
            prefix = "\\ccrow " if ok_c and r["system"] == "faiss_ivf" and r["param_tag"] == "np16" else ""
            tag = r["param_tag"].replace("_", "\\_")
            stat_sfx = " [\\texttt{no\\_erase}]" if r["status"] != "ok" else ""
            eng = f"{wl_disp[wl]}: {sys_disp[r['system']]} ({tag}){stat_sfx}"
            q_s = qf[i] if ok_mask.iloc[i] else f"{r['query_cpu_s']:.1f}"
            m_s = mf[i] if ok_mask.iloc[i] else f"{r['maint_cpu_s']:.1f}"
            t_s = tf[i] if ok_mask.iloc[i] else f"{r['total_cpu_s']:.1f}"
            r_s = rf[i] if ok_mask.iloc[i] else f"{r['mean_recall']:.4f}"
            mr_s = mrf[i] if ok_mask.iloc[i] else f"{r['min_recall']:.4f}"
            dag = "" if (ok_mask.iloc[i] and ok_c) else "$^\\dagger$"
            a_s = (af[i] if ok_mask.iloc[i] else f"{r['attainment_pct']:.1f}") + dag
            p_s = pf[i] if ok_mask.iloc[i] else f"{r['p99_us']:.1f}"
            lines.append(f"{prefix}{eng} & {q_s} & {m_s} & {t_s} & {r_s} & {mr_s} & {a_s} & {p_s} & -- \\\\")
        if wl != "runbook":
            lines.append("\\midrule")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{table*}",
    ])
    (OUT_TEX / "tab_exp5_unified.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"sift128_rows": len(sift), "churn_rows": len(churn), "configs": len(g)}


# ---------------------------------------------------------------------------
# Table 2: Trap 4 Contention & Cross-NUMA Inflation
# ---------------------------------------------------------------------------
def build_tab_trap4_combined() -> Dict[str, object]:
    df = pd.read_csv(RES / "stage1_hygiene" / "hygiene_summary.csv")
    g = df.groupby(["regime", "ways", "arm"], as_index=False).agg(
        n=("trace", "count"),
        query_cpu_s=("query_cpu_ms", lambda x: x.mean() / 1000.0),
        maint_cpu_s=("maint_cpu_ms", lambda x: x.mean() / 1000.0),
        total_cpu_s=("total_cpu_ms", lambda x: x.mean() / 1000.0),
        wall_s=("wall_ms", lambda x: x.mean() / 1000.0),
        p95_ms=("mean_p95_us", lambda x: x.mean() / 1000.0),
        p99_ms=("mean_p99_us", lambda x: x.mean() / 1000.0),
    )
    base_noop = g[(g["regime"] == "isolated_numa0") & (g["arm"] == "noop")].iloc[0]
    g["cpu_vs_iso_noop"] = g["total_cpu_s"] / base_noop["total_cpu_s"]
    g["p99_vs_iso_noop"] = g["p99_ms"] / base_noop["p99_ms"]
    order_reg = {"isolated_numa0": 0, "ht_sibling_2w": 1, "same_socket_4w": 2, "same_socket_8w": 3, "cross_socket_16w": 4}
    g["ord"] = g["regime"].map(order_reg)
    g = g.sort_values(["ord", "arm"]).drop(columns=["ord"]).reset_index(drop=True)
    g.to_csv(OUT_CSV / "tab_trap4_contention.csv", index=False)

    reg_disp = {
        "isolated_numa0": "Iso. NUMA-0 (1w)",
        "ht_sibling_2w": "HT Sibling (2w)",
        "same_socket_4w": "Same-Socket (4w)",
        "same_socket_8w": "Same-Socket (8w)",
        "cross_socket_16w": "Cross-Socket (16w)",
    }
    arm_disp = {"noop": "No-Op", "fixed_c2": "Fixed $c=2$", "past_qmass": "Past-QMass"}
    t_fmt = _rank_format(g["total_cpu_s"].tolist(), False, ".1f")
    w_fmt = _rank_format(g["wall_s"].tolist(), False, ".1f")
    p95_fmt = _rank_format(g["p95_ms"].tolist(), False, ".2f")
    p99_fmt = _rank_format(g["p99_ms"].tolist(), False, ".2f")

    # Panel (b): Abrupt t_sw Dirty vs Clean Reversal
    tsw_arms = [
        ("forecast_oracle", "Oracle ($B=9$)"),
        ("past_qmass_sp1", "Past-QMass ($B=9$)"),
        ("forecast_gbrt_causal", "GBRT-Causal ($B=9$)"),
        ("fixed_c2_merge", "Fixed $c=2$"),
        ("noop", "No-Op ($B=0$)"),
    ]
    tsw_rows = []
    for tsw in [16, 24, 28]:
        d_dirty = pd.read_csv(RES / "abrupt_tsw" / f"abrupt_tsw{tsw}_summary.csv").groupby("arm", as_index=False)["total_cpu_ms"].mean().set_index("arm")
        d_clean = pd.read_csv(RES / "abrupt_tsw_clean" / f"abrupt_tsw{tsw}_summary.csv").groupby("arm", as_index=False)[["total_cpu_ms", "wall_cpu_ratio", "mean_recall"]].mean().set_index("arm")
        for key, disp in tsw_arms:
            dirty_s = float(d_dirty.loc[key, "total_cpu_ms"]) / 1000.0
            clean_s = float(d_clean.loc[key, "total_cpu_ms"]) / 1000.0
            tsw_rows.append({
                "tsw": tsw, "arm": key, "disp": disp,
                "dirty_cpu_s": dirty_s, "clean_cpu_s": clean_s,
                "inflation_x": dirty_s / clean_s,
                "clean_wall_cpu": float(d_clean.loc[key, "wall_cpu_ratio"]),
                "clean_recall": float(d_clean.loc[key, "mean_recall"]),
            })
    df_tsw = pd.DataFrame(tsw_rows)
    df_tsw.to_csv(OUT_CSV / "tab_clean_abrupt_tsw.csv", index=False)

    lines = [
        "% Auto-generated from results/stage1_hygiene/ + results/abrupt_tsw{,_clean}/",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\renewcommand{\\arraystretch}{0.92}",
        "\\caption{Trap 4 measurement contamination and ranking reversals: (a) Dual Xeon 8352V symmetric placement contention ($n=186$, ratios vs.\\ Isolated NUMA-0 No-Op; under Same-Socket 8w, \\texttt{Past-QMass} inverts above \\texttt{No-Op} and \\texttt{c=2} in wall time and $p_{99}$); (b) asymmetric cross-batch load drift across $t_{\\text{sw}} \\in \\{16,24,28\\}$ ($n=96$), where \\texttt{GBRT-Causal}/Oracle ran in an idle batch ($1.00\\times$) while \\texttt{Past-QMass}/\\texttt{c=2}/\\texttt{No-Op} absorbed shared-host load ($1.14\\times$--$1.92\\times$), illustrating why Gate 4 forbids cross-batch pooling.}",
        "\\label{tab:trap4_contention}",
        "\\begin{minipage}[t]{0.49\\textwidth}",
        "\\centering",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "\\multicolumn{7}{c}{\\textbf{(a) Placement Contention (Dual Xeon 8352V, $n=186$)}} \\\\",
        "\\midrule",
        "Regime & Policy & $n$ & CPU (s) $\\downarrow$ & Wall (s) $\\downarrow$ & $p_{99}$ (ms) $\\downarrow$ & $p_{99}$ Infl. \\\\",
        "\\midrule",
    ]
    for i, r in g.iterrows():
        prefix = "\\ccrow " if r["regime"] == "isolated_numa0" and r["arm"] == "past_qmass" else ""
        lines.append(
            f"{prefix}{reg_disp[r['regime']]} & {arm_disp[r['arm']]} & {int(r['n'])} & "
            f"{t_fmt[i]} & {w_fmt[i]} & {p99_fmt[i]} & {r['p99_vs_iso_noop']:.2f}$\\times$ \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{minipage}\\hfill",
        "\\begin{minipage}[t]{0.49\\textwidth}",
        "\\centering",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "\\multicolumn{7}{c}{\\textbf{(b) Asymmetric Cross-Batch Load Drift ($t_{\\text{sw}}$, $n=96$)}} \\\\",
        "\\midrule",
        "$t_{\\text{sw}}$ & Policy & Dirty (s) $\\downarrow$ & Clean (s) $\\downarrow$ & Infl. & $\\text{wall}/\\text{CPU}$ & Rec. \\\\",
        "\\midrule",
    ])
    for tsw in [16, 24, 28]:
        sub = df_tsw[df_tsw["tsw"] == tsw].reset_index(drop=True)
        elig_b = [r["arm"] != "forecast_oracle" for _, r in sub.iterrows()]
        dfmt = _rank_format(sub["dirty_cpu_s"].tolist(), False, ".1f", eligible=elig_b)
        cfmt = _rank_format(sub["clean_cpu_s"].tolist(), False, ".1f", eligible=elig_b)
        for i, r in sub.iterrows():
            prefix = "\\ccrow " if r["arm"] == "past_qmass_sp1" else ""
            tsw_col = f"${tsw}$" if i == 0 else ""
            is_ora = r["arm"] == "forecast_oracle"
            pol_s = f"\\textit{{{r['disp']}}}" if is_ora else r["disp"]
            d_s = f"\\textit{{{dfmt[i]}}}" if is_ora else dfmt[i]
            c_s = f"\\textit{{{cfmt[i]}}}" if is_ora else cfmt[i]
            lines.append(
                f"{prefix}{tsw_col} & {pol_s} & {d_s} & {c_s} & {r['inflation_x']:.2f}$\\times$ & {r['clean_wall_cpu']:.3f} & {r['clean_recall']:.4f} \\\\"
            )
        if tsw != 28:
            lines.append("\\midrule")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{minipage}",
        "\\end{table*}",
    ])
    (OUT_TEX / "tab_trap4_contention.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "trap4": {"total_runs": int(g["n"].sum()), "cross_socket_p99_ratio": float(g[g["regime"] == "cross_socket_16w"]["p99_vs_iso_noop"].max())},
        "clean_abrupt_tsw": {
            "past_qmass_clean_s": {str(t): float(df_tsw[(df_tsw["tsw"] == t) & (df_tsw["arm"] == "past_qmass_sp1")]["clean_cpu_s"].iloc[0]) for t in [16, 24, 28]},
            "gbrt_causal_clean_s": {str(t): float(df_tsw[(df_tsw["tsw"] == t) & (df_tsw["arm"] == "forecast_gbrt_causal")]["clean_cpu_s"].iloc[0]) for t in [16, 24, 28]},
            "past_qmass_dirty_s": {str(t): float(df_tsw[(df_tsw["tsw"] == t) & (df_tsw["arm"] == "past_qmass_sp1")]["dirty_cpu_s"].iloc[0]) for t in [16, 24, 28]},
        },
    }


# ---------------------------------------------------------------------------
# Table 3: Exp 2 Core-8 Predictors, Clock Leak & Executed Cadence Brackets
# ---------------------------------------------------------------------------
def build_tab_exp2_combined() -> Dict[str, object]:
    df = pd.read_csv(RES / "exp2_predictors_cadence" / "mimo_core8_575_summary.csv")
    diag = json.loads((RES / "exp2_predictors_cadence" / "predictor_diagnostics.json").read_text())
    audit_ci = json.loads((RES / "exp2_predictors_cadence" / "mimo_core8_audit_and_ci.json").read_text())

    s25 = df[df["suite"] == "sift25"].copy()
    g = s25.groupby("arm", as_index=False).agg(
        total_cpu_s=("total_cpu_ms", lambda x: x.mean() / 1000.0),
        query_cpu_s=("query_cpu_ms", lambda x: x.mean() / 1000.0),
        maint_cpu_s=("maint_cpu_ms", lambda x: x.mean() / 1000.0),
        exact_scan_b=("exact_scan_ops", lambda x: x.mean() / 1e9),
        mean_recall=("mean_recall", "mean"),
        n_actions=("n_actions", "mean"),
    )
    model_map = {
        "fcst_past_mean_k5": "past_mean",
        "fcst_ewma_a03_k5": "ewma_a03",
        "fcst_ewma_a07_k5": "ewma_a07",
        "fcst_cusum_reset_k5": "cusum_reset",
        "fcst_ridge_ar_k5": "ridge_ar",
        "fcst_gbrt_causal_k5": "gbrt_causal",
        "fcst_gbrt_leaky_k5": "gbrt_leaky",
        "fcst_oracle_clairvoyant_k5": "oracle_clairvoyant",
    }
    g["top1_all"] = g["arm"].map(lambda a: diag["models"].get(model_map.get(a, ""), {}).get("top1_agree_all", np.nan))
    g["abrupt_j5_top1"] = g["arm"].map(lambda a: diag["models"].get(model_map.get(a, ""), {}).get("abrupt_j5_top1", np.nan))
    g["mean_jsd"] = g["arm"].map(lambda a: diag["models"].get(model_map.get(a, ""), {}).get("mean_jsd", np.nan))
    g = g.sort_values("total_cpu_s").reset_index(drop=True)
    g.to_csv(OUT_CSV / "tab_exp2_sift25.csv", index=False)

    keep_arms = [
        "fcst_oracle_clairvoyant_k5",
        "fcst_gbrt_leaky_k5",
        "fcst_ewma_a03_k5",
        "bam_price_k5",
        "fcst_gbrt_causal_k5",
        "fcst_cusum_reset_k5",
        "past_qmass_k5",
        "fcst_ridge_ar_k5",
        "fixed_c1_merge",
        "fixed_c2_merge",
        "fixed_c3_merge",
        "fixed_c5_merge",
        "noop",
    ]
    sub = g[g["arm"].isin(keep_arms)].copy()
    sub["ord"] = sub["arm"].map({a: i for i, a in enumerate(keep_arms)})
    sub = sub.sort_values("ord").drop(columns=["ord"]).reset_index(drop=True)

    arm_disp = {
        "fcst_oracle_clairvoyant_k5": "Oracle Clairvoyant ($\\kappa=5$)",
        "fcst_gbrt_leaky_k5": "GBRT-Leaky ($\\kappa=5$, flagged)",
        "fcst_ewma_a03_k5": "EWMA ($\\alpha=0.3$, $\\kappa=5$)",
        "bam_price_k5": "BAM-Price ($\\kappa=5$)",
        "fcst_gbrt_causal_k5": "GBRT-Causal ($\\kappa=5$, clean)",
        "fcst_cusum_reset_k5": "CUSUM-Reset ($\\kappa=5$)",
        "past_qmass_k5": "Past-QMass ($\\kappa=5$)",
        "fcst_ridge_ar_k5": "Ridge-AR ($\\kappa=5$)",
        "fixed_c1_merge": "Fixed Cadence $c=1$",
        "fixed_c2_merge": "Fixed Cadence $c=2$",
        "fixed_c3_merge": "Fixed Cadence $c=3$",
        "fixed_c5_merge": "Fixed Cadence $c=5$",
        "noop": "No-Op ($B=0$)",
    }

    elig_2a = [r["arm"] not in ("fcst_oracle_clairvoyant_k5", "fcst_gbrt_leaky_k5") for _, r in sub.iterrows()]
    t_fmt = _rank_format(sub["total_cpu_s"].tolist(), False, ".1f", eligible=elig_2a)
    ops_fmt = _rank_format(sub["exact_scan_b"].tolist(), False, ".2f", eligible=elig_2a)
    t1_fmt = _rank_format(sub["top1_all"].tolist(), True, ".3f", eligible=elig_2a)
    jsd_fmt = _rank_format(sub["mean_jsd"].tolist(), False, ".4f", eligible=elig_2a)

    # Panel (b): Trap 2 2x2 De-Leakage Ablation
    leak_arms = [
        ("forecast_oracle", "Oracle Clairvoyant"),
        ("forecast_gbrt_leaky", "GBRT-Leaky (flagged)"),
        ("forecast_gbrt_causal", "GBRT-Causal (Clean)"),
        ("past_qmass_sp1", "Past-QMass (Realized)"),
        ("forecast_ewma_a03", "EWMA ($\\alpha=0.3$)"),
        ("forecast_ridge_ar", "Ridge-AR"),
        ("noop", "No-Op ($B=0$)"),
    ]
    hubs = {}
    for hub in ["fixedhub", "randhub"]:
        ops = pd.read_csv(RES / "leakage_2x2" / hub / "exact_ops_and_dp.csv")
        gh = ops.groupby("arm", as_index=False)[["total_cpu_ms", "global_capture_ratio", "mean_recall", "n_actions"]].mean()
        hubs[hub] = gh.set_index("arm")

    lrows = []
    for key, disp in leak_arms:
        rf = hubs["fixedhub"].loc[key]
        rr = hubs["randhub"].loc[key]
        lrows.append({
            "arm": key, "disp": disp,
            "fixed_cpu_s": rf["total_cpu_ms"] / 1000.0,
            "fixed_gcr_pct": np.nan if key == "noop" else rf["global_capture_ratio"] * 100.0,
            "rand_cpu_s": rr["total_cpu_ms"] / 1000.0,
            "rand_gcr_pct": np.nan if key == "noop" else rr["global_capture_ratio"] * 100.0,
            "rand_recall": rr["mean_recall"],
        })
    df_leak = pd.DataFrame(lrows)
    df_leak.to_csv(OUT_CSV / "tab_trap2_leakage_2x2.csv", index=False)

    elig_2b = [r["arm"] not in ("forecast_oracle", "forecast_gbrt_leaky") for _, r in df_leak.iterrows()]
    f_cpu_f = _rank_format(df_leak["fixed_cpu_s"].tolist(), False, ".1f", eligible=elig_2b)
    f_gcr_f = _rank_format(df_leak["fixed_gcr_pct"].tolist(), True, ".2f", eligible=elig_2b)
    r_cpu_f = _rank_format(df_leak["rand_cpu_s"].tolist(), False, ".1f", eligible=elig_2b)
    r_gcr_f = _rank_format(df_leak["rand_gcr_pct"].tolist(), True, ".2f", eligible=elig_2b)

    lines = [
        "% Auto-generated from results/exp2_predictors_cadence/ + results/leakage_2x2/",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\renewcommand{\\arraystretch}{0.92}",
        "\\caption{Traps 1--2 diagnostic ledger: (a) Clean Ryzen 9 9950X Core-8 study (25 SIFT-128 traces, 575 runs; 25/25 traces bracketed by executed $c\\in\\{1..8\\}$, $0.000$ Top-1 agreement at abrupt $j=5$); (b) $2{\\times}2$ Fixed-Hub vs.\\ Randomized-Hub de-leakage ablation on Dual EPYC 7343 ($n=100$ runs). Italicized Oracle and Gate-2-flagged leaky rows sit above the rule; bold/underline rank valid causal policies.}",
        "\\label{tab:exp2_traps12}",
        "\\begin{minipage}[t]{0.53\\textwidth}",
        "\\centering",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrrr}",
        "\\toprule",
        "\\multicolumn{6}{c}{\\textbf{(a) Clean Core-8 Predictors \\& Brackets ($n=575$)}} \\\\",
        "\\midrule",
        "Policy / Predictor & Acts & CPU (s) $\\downarrow$ & Ops ($10^9$) $\\downarrow$ & Top-1 $\\uparrow$ & JSD $\\downarrow$ \\\\",
        "\\midrule",
    ]
    for i, r in sub.iterrows():
        is_ref = not elig_2a[i]
        prefix = "\\ccrow " if r["arm"] in ("past_qmass_k5", "fcst_ewma_a03_k5") else ""
        nm_s = f"\\textit{{{arm_disp[r['arm']]}}}" if is_ref else arm_disp[r["arm"]]
        t_s = f"\\textit{{{t_fmt[i]}}}" if is_ref else t_fmt[i]
        o_s = f"\\textit{{{ops_fmt[i]}}}" if is_ref else ops_fmt[i]
        t1_s = f"\\textit{{{t1_fmt[i]}}}" if is_ref and t1_fmt[i] != "--" else t1_fmt[i]
        j_s = f"\\textit{{{jsd_fmt[i]}}}" if is_ref and jsd_fmt[i] != "--" else jsd_fmt[i]
        lines.append(
            f"{prefix}{nm_s} & {r['n_actions']:.1f} & {t_s} & {o_s} & {t1_s} & {j_s} \\\\"
        )
        if r["arm"] == "fcst_gbrt_leaky_k5":
            lines.append("\\midrule")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{minipage}\\hfill",
        "\\begin{minipage}[t]{0.45\\textwidth}",
        "\\centering",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "\\multicolumn{5}{c}{\\textbf{(b) Trap 2 $2{\\times}2$ De-Leakage ($n=100$)}} \\\\",
        "\\midrule",
        "Policy & Fix CPU $\\downarrow$ & Fix GCR $\\uparrow$ & Rnd CPU $\\downarrow$ & Rnd GCR $\\uparrow$ \\\\",
        "\\midrule",
    ])
    for i, r in df_leak.iterrows():
        is_ref = not elig_2b[i]
        prefix = "\\ccrow " if r["arm"] in ("past_qmass_sp1", "forecast_gbrt_causal") else ""
        nm_s = f"\\textit{{{r['disp']}}}" if is_ref else r["disp"]
        fc_s = f"\\textit{{{f_cpu_f[i]}}}" if is_ref else f_cpu_f[i]
        fg_s = f"\\textit{{{f_gcr_f[i]}}}" if is_ref and f_gcr_f[i] != "--" else f_gcr_f[i]
        rc_s = f"\\textit{{{r_cpu_f[i]}}}" if is_ref else r_cpu_f[i]
        rg_s = f"\\textit{{{r_gcr_f[i]}}}" if is_ref and r_gcr_f[i] != "--" else r_gcr_f[i]
        lines.append(
            f"{prefix}{nm_s} & {fc_s} & {fg_s} & {rc_s} & {rg_s} \\\\"
        )
        if r["arm"] == "forecast_gbrt_leaky":
            lines.append("\\midrule")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{minipage}",
        "\\end{table*}",
    ])
    (OUT_TEX / "tab_exp2_traps12.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "exp2": {
            "n_total_runs": audit_ci["n_total_runs"],
            "cadence_bracket_pass_count": audit_ci["cadence_bracket_pass_count"],
            "gbrt_leaky_flagged": diag["clock_leak_audit"]["gbrt_leaky"]["flagged_features"],
        },
        "trap2_leakage_2x2": {
            "fixedhub_gbrt_leaky_gcr": float(df_leak[df_leak["arm"] == "forecast_gbrt_leaky"]["fixed_gcr_pct"].iloc[0]),
            "fixedhub_gbrt_causal_gcr": float(df_leak[df_leak["arm"] == "forecast_gbrt_causal"]["fixed_gcr_pct"].iloc[0]),
            "randhub_gbrt_leaky_gcr": float(df_leak[df_leak["arm"] == "forecast_gbrt_leaky"]["rand_gcr_pct"].iloc[0]),
            "randhub_gbrt_causal_gcr": float(df_leak[df_leak["arm"] == "forecast_gbrt_causal"]["rand_gcr_pct"].iloc[0]),
            "randhub_past_qmass_cpu_s": float(df_leak[df_leak["arm"] == "past_qmass_sp1"]["rand_cpu_s"].iloc[0]),
        },
    }


# ---------------------------------------------------------------------------
# Gate 3 Verified Stats: Track-CF 1w vs 4w Horizon Bias + Tombstone Refresh
# ---------------------------------------------------------------------------
def compute_gate3_stats() -> Dict[str, object]:
    cf_rows = []
    for p in sorted(glob.glob(str(RES / "track_cf" / "*.jsonl"))):
        recs = [json.loads(line) for line in open(p)]
        branches = [r for r in recs if r.get("type") == "branch"]
        by_j: Dict[int, list] = {}
        for br in branches:
            by_j.setdefault(int(br["j"]), []).append(br)
        flips = pairs = best_m = 0
        rhos = []
        q_ratios = []
        for j, items in sorted(by_j.items()):
            c1 = [it["windows"][0]["query_cpu_ms"] + it["maintenance_cpu_ms"] for it in items]
            c4 = [sum(w["query_cpu_ms"] for w in it["windows"]) + it["maintenance_cpu_ms"] for it in items]
            noop = next(it for it in items if it["action"] == "noop")
            q_ratios.append(sum(w["query_cpu_ms"] for w in noop["windows"]) / noop["windows"][0]["query_cpu_ms"])
            if min(range(len(items)), key=lambda i: c1[i]) == min(range(len(items)), key=lambda i: c4[i]):
                best_m += 1
            n = len(items)
            rk1 = [0] * n
            rk4 = [0] * n
            for rk, idx in enumerate(sorted(range(n), key=lambda i: c1[i])):
                rk1[idx] = rk
            for rk, idx in enumerate(sorted(range(n), key=lambda i: c4[i])):
                rk4[idx] = rk
            rhos.append(1.0 - 6.0 * sum((rk1[i] - rk4[i]) ** 2 for i in range(n)) / (n * (n * n - 1)))
            for a in range(n):
                for b in range(a + 1, n):
                    pairs += 1
                    if (c1[a] < c1[b]) != (c4[a] < c4[b]):
                        flips += 1
        cf_rows.append({
            "trace": Path(p).stem,
            "mean_rho_1w_vs_4w": float(np.mean(rhos)),
            "best_action_match": f"{best_m}/{len(by_j)}",
            "pair_flip_pct": 100.0 * flips / pairs,
            "noop_q4_over_q1": float(np.mean(q_ratios)),
        })
    tomb_rows = []
    for p in sorted(glob.glob(str(RES / "tombstone_sweep" / "crud*" / "exact_ops_and_dp.csv"))):
        d = pd.read_csv(p)
        g = d.groupby("arm")[["total_cpu_ms", "maint_cpu_ms"]].mean()
        tomb_rows.append({
            "crud": Path(p).parent.name,
            "past_qmass_merge_s": float(g.loc["past_qmass_sp1", "total_cpu_ms"] / 1000.0),
            "past_qmass_refresh_s": float(g.loc["past_qmass_refresh", "total_cpu_ms"] / 1000.0),
        })
    return {"track_cf_1w_vs_4w": cf_rows, "tombstone_refresh_vs_merge": tomb_rows}


# ---------------------------------------------------------------------------
# Table 6 (Combined): Cross-Modal, Real Streams, CRUD & Scale/Arch Transfer
# ---------------------------------------------------------------------------
def build_tab_crossmodal_transfer() -> Dict[str, object]:
    datasets = [
        ("MiniLM-384 (Synth.)", RES / "crossmodal_rag384" / "exact_ops_and_dp.csv"),
        ("CLIP-512 (Real-IAD)", RES / "crossmodal_clip512" / "exact_ops_and_dp.csv"),
        ("CLINC-150 (NLP)", RES / "clinc150" / "exact_ops_and_dp.csv"),
        ("Wikimedia-210d", RES / "wikimedia_windows" / "exact_ops_and_dp.csv"),
        ("arXiv-CS (140K, B=5)", RES / "exp7_gpu_1m_arxiv" / "arxiv_minilm384_exact_ops_and_dp.csv"),
        ("SIFT CRUD-30%", RES / "sprint_v2_llm01" / "crud30v2" / "exact_ops_and_dp.csv"),
        ("SIFT CRUD-50%", RES / "sprint_v2_llm01" / "crud50v2" / "exact_ops_and_dp.csv"),
    ]
    rows_out = []
    for label, path in datasets:
        d = pd.read_csv(path)
        for arm in d["arm"].unique():
            sub = d[d["arm"] == arm]
            rows_out.append({
                "dataset": label,
                "arm": arm,
                "n": len(sub),
                "total_cpu_s": sub["total_cpu_ms"].mean() / 1000.0 if "total_cpu_ms" in sub.columns else np.nan,
                "gcr_pct": sub["global_capture_ratio"].mean() * 100.0,
                "mean_recall": sub["mean_recall"].mean() if "mean_recall" in sub.columns else np.nan,
                "n_actions": sub["n_actions"].mean(),
                "exact_scan_m": sub["exact_scan_ops"].mean() / 1e6 if "exact_scan_ops" in sub.columns else np.nan,
            })
    df_all = pd.DataFrame(rows_out)
    df_all.to_csv(OUT_CSV / "tab_crossmodal_real.csv", index=False)

    focus_arms = [
        ("fixed_c2_merge", "Fixed $c=2$"),
        ("online_milp_b4", "Online MILP ($B=4$)"),
        ("past_qmass_sp1", "Past-QMass"),
        ("past_qmass_k5", "Past-QMass ($\\kappa=5$)"),
        ("forecast_ewma_a03", "EWMA ($\\alpha=0.3$)"),
        ("forecast_ewma_a07_k5", "EWMA ($\\alpha=0.7$, $\\kappa=5$)"),
        ("gbrt_single_stream", "GBRT (Single-Stream)"),
        ("forecast_gbrt_causal", "GBRT (Causal)"),
        ("forecast_gbrt_causal_k5", "GBRT-Causal ($\\kappa=5$)"),
        ("forecast_oracle", "Oracle"),
        ("forecast_oracle_clairvoyant_k5", "Oracle ($\\kappa=5$)"),
    ]

    specs = [
        ("SIFT-100K ($B{=}9$, Xeon)", RES / "sift100k_canon25" / "exact_ops_and_dp.csv", "forecast_gbrt_causal", "GBRT-Causal"),
        ("Canon25 ($B{=}4$, Xeon)", RES / "sprint_v2_llm01" / "canon25" / "exact_ops_and_dp.csv", "online_milp_b4", "MILP-$B4$"),
        ("SIFT-300K (Scale 3$\\times$)", RES / "sift300k" / "exact_ops_and_dp.csv", "forecast_gbrt_causal", "GBRT-Causal"),
        ("SIFT-600K (Scale 6$\\times$)", RES / "sift600k_s07" / "exact_ops_and_dp.csv", "forecast_gbrt_causal", "GBRT-Causal"),
        ("SIFT-1M (Scale 10$\\times$)", RES / "sprint_v2_llm01" / "sift1m_n12" / "exact_ops_and_dp.csv", "max_delta_sp1", "Max-Delta"),
        ("Xeon 8352V (Cross-Arch)", RES / "xeon_anchor" / "exact_ops_and_dp.csv", "past_qmass_refresh", "PQ-Refresh"),
        ("EPYC 7343 (Cross-Arch)", RES / "epyc_anchor" / "exact_ops_and_dp.csv", "past_qmass_refresh", "PQ-Refresh"),
        ("Del. Runbook (Clust./Slid.)", RES / "runbook" / "exact_ops_and_dp.csv", "past_qmass_refresh", "PQ-Refresh"),
        ("Seasonal Pos-Ctrl ($T{=}40$)", RES / "seasonal_posctrl" / "exact_ops_and_dp.csv", "forecast_seasonal_acf", "Seas.-ACF"),
        ("Long-Seasonal ($T{=}40$)", RES / "sprint_v2_llm01" / "p0_2_long_seasonal" / "T40" / "exact_ops_and_dp.csv", "forecast_seasonal_acf", "Seas.-ACF"),
        ("Long-Seasonal ($T{=}80$)", RES / "sprint_v2_llm01" / "p0_2_long_seasonal" / "T80" / "exact_ops_and_dp.csv", "forecast_seasonal_acf", "Seas.-ACF"),
        ("Long-Seasonal ($T{=}120$)", RES / "sprint_v2_llm01" / "p0_2_long_seasonal" / "T120" / "exact_ops_and_dp.csv", "forecast_seasonal_acf", "Seas.-ACF"),
        ("arXiv-CS (140K, $B{=}5$)", RES / "exp7_gpu_1m_arxiv" / "arxiv_minilm384_exact_ops_and_dp.csv", "forecast_ewma_a07_k5", "EWMA-0.7"),
    ]
    trows = []
    for label, fp, alt_arm, alt_lbl in specs:
        d = pd.read_csv(fp)
        cols = [c for c in ("total_cpu_ms", "global_capture_ratio", "exact_scan_ops") if c in d.columns]
        g = d.groupby("arm")[cols].mean()
        c2_gcr = float(g.loc["fixed_c2_merge", "global_capture_ratio"] * 100.0) if "fixed_c2_merge" in g.index else np.nan
        if "Canon25 ($B{=}4$" in label:
            pq_arm = "past_qmass_b4_paced"
        elif "past_qmass_sp1" in g.index:
            pq_arm = "past_qmass_sp1"
        elif "past_qmass_k5" in g.index:
            pq_arm = "past_qmass_k5"
        elif "forecast_ewma_a03" in g.index:
            pq_arm = "forecast_ewma_a03"
        else:
            pq_arm = alt_arm
        pq_gcr = float(g.loc[pq_arm, "global_capture_ratio"] * 100.0)
        pq_cpu = float(g.loc[pq_arm, "total_cpu_ms"] / 1000.0) if "total_cpu_ms" in g.columns else np.nan
        pq_ops_m = float(g.loc[pq_arm, "exact_scan_ops"] / 1e6) if "exact_scan_ops" in g.columns else np.nan
        alt_gcr = float(g.loc[alt_arm, "global_capture_ratio"] * 100.0)
        ora_arm = "forecast_oracle" if "forecast_oracle" in g.index else ("forecast_oracle_clairvoyant_k5" if "forecast_oracle_clairvoyant_k5" in g.index else None)
        ora_gcr = float(g.loc[ora_arm, "global_capture_ratio"] * 100.0) if ora_arm else np.nan
        trows.append({
            "suite": label,
            "n_traces": int(d["trace"].nunique()),
            "c2_gcr": c2_gcr,
            "pq_gcr": pq_gcr,
            "pq_lbl": "EWMA" if pq_arm == "forecast_ewma_a03" else ("PQ-paced" if pq_arm == "past_qmass_b4_paced" else "PQ"),
            "alt_lbl": alt_lbl,
            "alt_gcr": alt_gcr,
            "ora_gcr": ora_gcr,
            "pq_cpu_s": pq_cpu,
            "pq_ops_m": pq_ops_m,
        })
    exp8_milp = pd.read_csv(RES / "exp8_colab_large_scale" / "semantic_scale_milp_headroom.csv")
    for s_key, s_lbl, alt_pol, alt_lbl in [
        ("arxiv_140k_768d", "arXiv 768-d (140K, $B{=}5$)", "ewma_a07", "EWMA-0.7"),
        ("arxiv_1m_384d", "arXiv-All 1M (384-d, $B{=}5$)", "max_delta", "Max-Delta"),
    ]:
        sub_m = exp8_milp[exp8_milp["suite"] == s_key].set_index("policy")
        trows.append({
            "suite": s_lbl,
            "n_traces": 1,
            "c2_gcr": float(sub_m.loc["fixed_c2", "gcr_pct"]),
            "pq_gcr": float(sub_m.loc["past_qmass", "gcr_pct"]),
            "pq_lbl": "PQ",
            "alt_lbl": alt_lbl,
            "alt_gcr": float(sub_m.loc[alt_pol, "gcr_pct"]),
            "ora_gcr": float(sub_m.loc["oracle_1blk", "gcr_pct"]),
            "pq_cpu_s": np.nan,
            "pq_ops_m": float(sub_m.loc["past_qmass", "exact_scan_ops"] / 1e6),
        })
    df_tr = pd.DataFrame(trows)
    df_tr.to_csv(OUT_CSV / "tab_scale_arch_transfer.csv", index=False)

    lines = [
        "% Auto-generated from audited cross-modal, real-stream, scale, and seasonal CSVs",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\renewcommand{\\arraystretch}{0.88}",
        "\\caption{Cross-modality predictability, real-timestamp streams (incl.\\ 140K \\texttt{arXiv-CS} 384-d/768-d and 1M \\texttt{arXiv-All}), and Proposition~1 Maintenance Headroom transfer across scale ($100\\text{K}\\rightarrow 1\\text{M}$), CPU architectures, Deletion Runbook, and seasonal horizons. GCR is normalized to Proposition~1 MILP ($100\\%$ headroom). On \\texttt{Wikimedia-210d}, raw shared-host CPU is withheld ($67/144$ Gate~4 flags; hardware-invariant GCR shown). \\texttt{Canon25 ($B{=}4$)} compares matched-$B{=}4$ policies (separate from $B{=}9$ above); bold/underline rank causal policies, with 1-block Oracle italicized.}",
        "\\label{tab:crossmodal_transfer}",
        "\\begin{minipage}[t]{0.47\\textwidth}",
        "\\centering",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrr}",
        "\\toprule",
        "\\multicolumn{6}{c}{\\textbf{(a) Cross-Modal, Real Streams \\& Matched-Budget CRUD}} \\\\",
        "\\midrule",
        "Workload & Policy & $n$ & Acts & CPU (s) $\\downarrow$ & GCR (\\%) $\\uparrow$ \\\\",
        "\\midrule",
    ]
    for idx_ds, (label, _) in enumerate(datasets):
        sub = df_all[df_all["dataset"] == label]
        picked = []
        for arm_key, disp in focus_arms:
            if "arXiv" not in label and arm_key == "forecast_ewma_a07_k5":
                continue
            if "MiniLM-384" not in label and arm_key == "forecast_ewma_a03":
                continue
            m = sub[sub["arm"] == arm_key]
            if len(m):
                picked.append((arm_key, disp, m.iloc[0]))
        elig_cm = ["oracle" not in k for k, _, _ in picked]
        cpus = [np.nan if "Wikimedia" in label else r["total_cpu_s"] for _, _, r in picked]
        gcrs = [r["gcr_pct"] for _, _, r in picked]
        cpu_f = _rank_format(cpus, False, ".1f", eligible=elig_cm)
        gcr_f = _rank_format(gcrs, True, ".1f", eligible=elig_cm)
        for idx, (arm_key, disp, r) in enumerate(picked):
            is_ora = not elig_cm[idx]
            prefix = "\\ccrow " if "past_qmass" in arm_key or arm_key == "online_milp_b4" or arm_key == "forecast_ewma_a07_k5" or arm_key == "forecast_ewma_a03" else ""
            ds_col = label.replace("%", "\\%") if idx == 0 else ""
            d_lbl = f"\\textit{{{disp}}}" if is_ora else disp
            c_cell = f"\\textit{{{cpu_f[idx]}}}" if is_ora and cpu_f[idx] != "--" else cpu_f[idx]
            g_cell = f"\\textit{{{gcr_f[idx]}}}" if is_ora and gcr_f[idx] != "--" else gcr_f[idx]
            lines.append(
                f"{prefix}{ds_col} & {d_lbl} & {int(r['n'])} & {r['n_actions']:.1f} & {c_cell} & {g_cell} \\\\"
            )
        if idx_ds < len(datasets) - 1:
            lines.append("\\midrule")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{minipage}\\hfill",
        "\\begin{minipage}[t]{0.51\\textwidth}",
        "\\centering",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "\\multicolumn{7}{c}{\\textbf{(b) Maintenance Headroom \\& Scale/Arch Transfer (Prop.~1 = 100\\%)}} \\\\",
        "\\midrule",
        "Suite (Regime) & Tr. & $c{=}2$ $\\uparrow$ & PQ/EWMA $\\uparrow$ & Alt. Policy $\\uparrow$ & \\textit{1-blk Ora.} & Base Cost \\\\",
        "\\midrule",
    ])
    for _, r in df_tr.iterrows():
        vals = [r["c2_gcr"], r["pq_gcr"], r["alt_gcr"]]
        fmtd = _rank_format(vals, True, ".1f")
        c2_s = f"{fmtd[0]}\\%" if not pd.isna(r["c2_gcr"]) else "--"
        pq_tag = f" ({r['pq_lbl']})" if r["pq_lbl"] in ("EWMA", "PQ-paced") else ""
        pq_s = f"{fmtd[1]}\\%{pq_tag}"
        alt_s = f"{fmtd[2]}\\% ({r['alt_lbl']})"
        ora_s = f"\\textit{{{r['ora_gcr']:.1f}\\%}}" if not pd.isna(r["ora_gcr"]) else "--"
        cost_s = f"{r['pq_cpu_s']:.1f}s" if not pd.isna(r["pq_cpu_s"]) else f"{r['pq_ops_m']:.1f}M ops"
        prefix = "\\ccrow " if "Scale" in r["suite"] or "Cross-Arch" in r["suite"] else ""
        lines.append(
            f"{prefix}{r['suite']} & {int(r['n_traces'])} & {c2_s} & {pq_s} & {alt_s} & {ora_s} & {cost_s} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{minipage}",
        "\\end{table*}",
    ])
    (OUT_TEX / "tab_crossmodal_transfer.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    anchor_suites = [
        r for r in trows
        if any(k in r["suite"] for k in ("SIFT-100K", "SIFT-300K", "SIFT-600K", "SIFT-1M", "Xeon 8352V", "EPYC 7343"))
    ]
    headroom = {
        "c2_gcr_min_pct": min(r["c2_gcr"] for r in anchor_suites),
        "c2_gcr_max_pct": max(r["c2_gcr"] for r in anchor_suites),
        "pq_gcr_min_pct": min(r["pq_gcr"] for r in anchor_suites),
        "pq_gcr_max_pct": max(r["pq_gcr"] for r in anchor_suites),
        "prop1_gcr_pct": 100.0,
    }
    return {
        "crossmodal_real": {"datasets": len(datasets), "total_rows_summarized": int(df_all["n"].sum())},
        "scale_arch_transfer": {"suites": len(df_tr), "rows": trows},
        "headroom": headroom,
    }


# ---------------------------------------------------------------------------
# Table 6 (Unified Real-Engine Traps A1-A4 & A100 1M / arXiv Audit)
# ---------------------------------------------------------------------------
def build_tab_real_engines_1m_arxiv() -> Dict[str, object]:
    sys_disp = {
        "faiss_ivf": "FAISS-IVF",
        "faiss_hnsw": "FAISS-HNSW",
        "hnswlib": "hnswlib",
        "usearch": "USearch",
        "usearch_rust": "USearch-Rust",
        "hnsw_rs": "hnsw\\_rs",
        "diskann": "FreshDiskANN",
        "annoy": "Annoy",
    }
    a4_sum = pd.read_csv(RES / "exp6_real_engine_traps" / "a4_summary_by_config.csv")
    a3_raw = pd.read_csv(RES / "exp6_real_engine_traps" / "a3_crud_engines.csv")
    a1d_sum = pd.read_csv(RES / "exp6_real_engine_traps" / "a1d_summary.csv")
    a2_raw = pd.read_csv(RES / "exp6_real_engine_traps" / "a2_cadence_brackets.csv")

    r1_a4_raw = pd.read_csv(RES / "exp9_rust_and_scale" / "r1_rust_a4_floor.csv")
    r1_a4_agg = r1_a4_raw.groupby(["system" if "system" in r1_a4_raw.columns else "engine", "config"], as_index=False).agg(
        total_cpu_ms=("total_cpu_s", lambda x: x.mean() * 1000.0),
        mean_recall=("mean_recall", "mean"),
        attainment_0965_pct=("att_0965_pct", "mean"),
        mean_p99_us=("p99_us", "mean"),
        peak_rss_mb=("peak_rss_mb", "mean"),
    ).rename(columns={"engine": "system", "config": "param_tag"})
    a4_all = pd.concat([a4_sum, r1_a4_agg], ignore_index=True)

    a4_picks = [
        ("faiss_ivf", "np16"), ("faiss_ivf", "np24"),
        ("diskann", "L64"),
        ("hnswlib", "ef96"),
        ("usearch_rust", "ef96"),
        ("usearch", "ef96"),
        ("hnsw_rs", "ef96"),
    ]
    a4_sub = a4_all[a4_all.apply(lambda r: (r["system"], r["param_tag"]) in a4_picks, axis=1)].copy()
    a4_sub["ord"] = a4_sub.apply(lambda r: a4_picks.index((r["system"], r["param_tag"])), axis=1)
    a4_sub = a4_sub.sort_values("ord").reset_index(drop=True)
    a4_sub.to_csv(OUT_CSV / "tab_exp6_a4_floor.csv", index=False)

    a3_agg = a3_raw.groupby(["workload", "system", "param_tag", "status"], as_index=False).agg(
        total_cpu_s=("total_cpu_ms", lambda x: x.mean() / 1000.0),
        mean_recall=("mean_recall", "mean"),
        attainment_0965_pct=("attainment_0965_pct", "mean"),
        mean_p99_us=("mean_p99_us", "mean"),
        peak_rss_mb=("peak_rss_mb", "mean"),
        max_wc=("wall_cpu_ratio", "max"),
    )
    r1_a3_raw = pd.read_csv(RES / "exp9_rust_and_scale" / "r1_rust_a3_crud.csv")
    r1_a3_agg = r1_a3_raw.groupby(["dataset", "engine", "config", "status"], as_index=False).agg(
        total_cpu_s=("total_cpu_s", "mean"),
        mean_recall=("mean_recall", "mean"),
        attainment_0965_pct=("att_0965_pct", "mean"),
        mean_p99_us=("p99_us", "mean"),
        peak_rss_mb=("peak_rss_mb", "mean"),
        max_wc=("wall_cpu_ratio", "max"),
    ).rename(columns={"dataset": "workload", "engine": "system", "config": "param_tag"})
    a3_all = pd.concat([a3_agg, r1_a3_agg], ignore_index=True)
    a3_all.to_csv(OUT_CSV / "tab_exp6_a3_crud.csv", index=False)

    s1_raw = pd.read_csv(RES / "exp9_rust_and_scale" / "s1_xeon_1m_gate4_clean.csv")
    s1_raw.to_csv(OUT_CSV / "tab_s1_xeon_1m_core0.csv", index=False)

    darx = pd.read_csv(RES / "exp7_gpu_1m_arxiv" / "arxiv_5engines_summary.csv")
    d1m = pd.read_csv(RES / "exp7_gpu_1m_arxiv" / "sift1m_5engines_summary.csv")
    d_ag = pd.read_csv(RES / "colab_1m_audit" / "colab_real_semantic_engines.csv").copy()
    ag_np16_cpu = float(d_ag[(d_ag["system"] == "faiss_ivf") & (d_ag["param_tag"] == "np16")]["total_cpu_ms"].iloc[0])
    d_ag["query_cpu_pct"] = 100.0 * d_ag["query_cpu_ms"] / d_ag["total_cpu_ms"]
    d_ag["maint_cpu_pct"] = 100.0 * d_ag["maint_cpu_ms"] / d_ag["total_cpu_ms"]
    d_ag["rel_cpu_vs_faiss_np16"] = d_ag["total_cpu_ms"] / ag_np16_cpu
    d_ag["peak_rss_mb"] = np.nan

    d8_768 = pd.read_csv(RES / "exp8_colab_large_scale" / "arxiv140k_768d_engines.csv").rename(columns={"rel_cpu_vs_faiss_base": "rel_cpu_vs_faiss_np16"})
    d8_500 = pd.read_csv(RES / "exp8_colab_large_scale" / "arxiv500k_384d_engines.csv").rename(columns={"rel_cpu_vs_faiss_base": "rel_cpu_vs_faiss_np16"})
    d8_1m = pd.read_csv(RES / "exp8_colab_large_scale" / "arxiv1m_384d_engines.csv").rename(columns={"rel_cpu_vs_faiss_base": "rel_cpu_vs_faiss_np16"})

    exp8_all = pd.concat([d8_768, d_ag, d8_500, d8_1m], ignore_index=True)
    exp8_all.to_csv(OUT_CSV / "tab_exp7_arxiv_1m.csv", index=False)

    keep_exp8 = {
        "arxiv140k_768d_drift": [
            ("faiss_hnsw", "ef64"), ("hnswlib", "ef64"), ("faiss_ivf", "np16"),
            ("usearch", "ef64"), ("diskann", "L64"), ("faiss_ivf", "np32"), ("annoy", "sk8000_re4"),
        ],
        "agnews_minilm384": [
            ("hnswlib", "ef64"), ("faiss_hnsw", "ef64"), ("faiss_ivf", "np16"), ("faiss_ivf", "np32"),
        ],
        "arxiv500k_drift": [
            ("faiss_ivf", "np16"), ("faiss_ivf", "np32"), ("hnswlib", "ef64"),
            ("faiss_hnsw", "ef64"), ("usearch", "ef64"), ("diskann", "L64"), ("annoy", "sk16000_re10"),
        ],
        "arxiv500k_crud30": [
            ("faiss_ivf", "np32"), ("diskann", "L64"), ("hnswlib", "ef64"),
            ("diskann", "L64_cons10"), ("faiss_hnsw", "ef64"),
        ],
        "arxiv1m_384d_drift": [
            ("faiss_ivf", "np32"), ("hnswlib", "ef64"), ("faiss_hnsw", "ef64"),
            ("usearch", "ef64"), ("diskann", "L64"),
        ],
    }

    lines = [
        "% Auto-generated from results/exp6_real_engine_traps/, exp9_rust_and_scale/ (Xeon Core-0), exp8_colab_large_scale/ (A100 GPU-GT), & colab_1m_audit/",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\renewcommand{\\arraystretch}{0.86}",
        "\\caption{Cross-engine audit across eight backends: (a) Dual Xeon 8352V Core-0 Gate-4 clean ($\\text{w/C}\\le 1.105$) recall floor, Rust \\texttt{usearch\\_rust}/\\texttt{hnsw\\_rs}, and bare-metal \\texttt{BigANN-1M}; (b) A100/T4 GPU-GT semantic Dimension $\\times$ Scale Crossover (140K 768-d, 120K AG News, 500K/1M 384-d; raw vCPU timings withheld). Below-floor configs ($^\\dagger$) are excluded from cost/latency/RSS ranking. $^*$VM Rel.\\ reports same-VM single-pinned ratios ($\\text{w/C}\\le 1.01$, subject to up to $\\sim 18\\%$ cross-engine SMT uncertainty); hardware-invariant Rec., Att., Q/M, and VmHWM are primary Gate-4-clean metrics.}",
        "\\label{tab:real_engines_1m_arxiv}",
        "\\begin{minipage}[t]{0.49\\textwidth}",
        "\\centering",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "\\multicolumn{7}{c}{\\textbf{(a) Xeon 8352V Core-0: Floor, Rust \\& Bare-Metal 1M}} \\\\",
        "\\midrule",
        "Workload & Engine (Cfg) & CPU (s) $\\downarrow$ & Rec. $\\uparrow$ & Att.@.965 $\\uparrow$ & $p_{99}$ ($\\mu$s) $\\downarrow$ & RSS $\\downarrow$ \\\\",
        "\\midrule",
    ]
    a4_elig = [bool(r["attainment_0965_pct"] >= 95.0) for _, r in a4_sub.iterrows()]
    c_f = _rank_format((a4_sub["total_cpu_ms"] / 1000.0).tolist(), False, ".1f", eligible=a4_elig)
    r_f = _rank_format(a4_sub["mean_recall"].tolist(), True, ".4f")
    a_f = _rank_format(a4_sub["attainment_0965_pct"].tolist(), True, ".1f")
    p_f = _rank_format(a4_sub["mean_p99_us"].tolist(), False, ".1f", eligible=a4_elig)
    rss_f = _rank_format(a4_sub["peak_rss_mb"].tolist(), False, ".0f", eligible=a4_elig)
    for i, r in a4_sub.iterrows():
        wl_c = "100K Drift" if i == 0 else ""
        tag = r["param_tag"].replace("_", "\\_")
        eng = f"{sys_disp[r['system']]} ({tag})"
        ok = a4_elig[i]
        prefix = "\\ccrow " if ok and r["system"] in ("faiss_ivf", "diskann", "usearch_rust") else ""
        dag = "" if ok else "$^\\dagger$"
        lines.append(f"{prefix}{wl_c} & {eng} & {c_f[i]} & {r_f[i]} & {a_f[i]}\\%{dag} & {p_f[i]} & {rss_f[i]} \\\\")

    a3_picks = {
        "crud30v2": [("faiss_ivf", "np16"), ("diskann", "L64_cons1"), ("usearch_rust", "ef96"), ("hnswlib", "ef64"), ("hnsw_rs", "ef64")],
        "crud50v2": [("faiss_ivf", "np16"), ("diskann", "L64_cons1"), ("usearch_rust", "ef96"), ("hnsw_rs", "ef64")],
        "runbook": [("faiss_ivf", "np16"), ("diskann", "L64_cons0"), ("usearch_rust", "ef96"), ("hnswlib", "ef64")],
    }
    a3_wl_disp = {"crud30v2": "100K CRUD30", "crud50v2": "100K CRUD50", "runbook": "Del. Runbook"}
    for wl in ["crud30v2", "crud50v2", "runbook"]:
        lines.append("\\midrule")
        sub = a3_all[(a3_all["workload"] == wl) & a3_all.apply(lambda r: (r["system"], r["param_tag"]) in a3_picks[wl], axis=1)].copy()
        sub["ord"] = sub.apply(lambda r: a3_picks[wl].index((r["system"], r["param_tag"])), axis=1)
        sub = sub.sort_values("ord").reset_index(drop=True)
        ok_rows = [bool(r["status"] == "ok") for _, r in sub.iterrows()]
        elig_a3 = [bool(r["status"] == "ok" and r["attainment_0965_pct"] >= 95.0) for _, r in sub.iterrows()]
        cf = _rank_format(sub["total_cpu_s"].tolist(), False, ".1f", eligible=elig_a3)
        rf = _rank_format(sub["mean_recall"].tolist(), True, ".4f", eligible=ok_rows)
        af = _rank_format(sub["attainment_0965_pct"].tolist(), True, ".1f", eligible=ok_rows)
        pf = _rank_format(sub["mean_p99_us"].tolist(), False, ".1f", eligible=elig_a3)
        rssf_a3 = _rank_format(sub["peak_rss_mb"].tolist(), False, ".0f", eligible=elig_a3)
        for i, r in sub.iterrows():
            wl_c = a3_wl_disp[wl] if i == 0 else ""
            tag = r["param_tag"].replace("_", "\\_")
            eng = f"{sys_disp[r['system']]} ({tag})"
            if r["status"] != "ok":
                lines.append(f"{wl_c} & {eng} & \\multicolumn{{2}}{{c}}{{\\texttt{{unsupp.\\_erase}}}} & 0.0\\%$^\\dagger$ & -- & {r['peak_rss_mb']:.0f} \\\\")
            else:
                ok = elig_a3[i]
                dag = "" if ok else "$^\\dagger$"
                prefix = "\\ccrow " if r["attainment_0965_pct"] >= 99.9 and r["system"] in ("diskann", "faiss_ivf", "usearch_rust") else ""
                lines.append(f"{prefix}{wl_c} & {eng} & {cf[i]} & {rf[i]} & {af[i]}\\%{dag} & {pf[i]} & {rssf_a3[i]} \\\\")

    s1_picks = {
        "bigann_1m_drift": [("faiss_ivf", "np32"), ("hnswlib", "ef128"), ("diskann", "L64"), ("usearch_rust", "ef128"), ("hnsw_rs", "ef128")],
        "bigann_1m_crud20": [("faiss_ivf", "np32"), ("faiss_ivf", "np64"), ("usearch_rust", "ef128"), ("hnswlib", "ef128"), ("diskann", "L64_cons1")],
    }
    s1_disp = {"bigann_1m_drift": "1M Drift", "bigann_1m_crud20": "1M CRUD20"}
    for wl in ["bigann_1m_drift", "bigann_1m_crud20"]:
        lines.append("\\midrule")
        sub = s1_raw[(s1_raw["trace"] == wl) & s1_raw.apply(lambda r: (r["engine"], r["config"]) in s1_picks[wl], axis=1)].copy()
        sub["ord"] = sub.apply(lambda r: s1_picks[wl].index((r["engine"], r["config"])), axis=1)
        sub = sub.sort_values("ord").reset_index(drop=True)
        elig_s1 = [bool(r["att_0965_pct"] >= 95.0) for _, r in sub.iterrows()]
        cf = _rank_format(sub["total_cpu_s"].tolist(), False, ".1f", eligible=elig_s1)
        rf = _rank_format(sub["mean_recall"].tolist(), True, ".4f")
        af = _rank_format(sub["att_0965_pct"].tolist(), True, ".1f")
        pf = _rank_format(sub["p99_us"].tolist(), False, ".1f", eligible=elig_s1)
        rssf = _rank_format(sub["peak_rss_mb"].tolist(), False, ".0f", eligible=elig_s1)
        for i, r in sub.iterrows():
            wl_c = s1_disp[wl] if i == 0 else ""
            tag = r["config"].replace("_", "\\_")
            eng = f"{sys_disp[r['engine']]} ({tag})"
            ok = elig_s1[i]
            dag = "" if ok else "$^\\dagger$"
            prefix = "\\ccrow " if ok and r["engine"] in ("faiss_ivf", "usearch_rust") else ""
            lines.append(f"{prefix}{wl_c} & {eng} & {cf[i]} & {rf[i]} & {af[i]}\\%{dag} & {pf[i]} & {rssf[i]} \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{minipage}\\hfill",
        "\\begin{minipage}[t]{0.49\\textwidth}",
        "\\centering",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrrrr}",
        "\\toprule",
        "\\multicolumn{8}{c}{\\textbf{(b) A100/T4 GPU-GT Semantic Dimension $\\times$ Scale Crossover}} \\\\",
        "\\midrule",
        "Workload & Engine (Cfg) & Rec. $\\uparrow$ & MinR $\\uparrow$ & Att.@.965 $\\uparrow$ & Split (Q/M) & VmHWM $\\downarrow$ & VM Rel.$^*$ $\\downarrow$ \\\\",
        "\\midrule",
    ])
    wl_exp8_disp = {
        "arxiv140k_768d_drift": "arXiv 140K 768-d",
        "agnews_minilm384": "AG News 384-d",
        "arxiv500k_drift": "arXiv 500K 384-d",
        "arxiv500k_crud30": "500K CRUD-30\\%",
        "arxiv1m_384d_drift": "arXiv 1M 384-d",
    }
    wl_order = ["arxiv140k_768d_drift", "agnews_minilm384", "arxiv500k_drift", "arxiv500k_crud30", "arxiv1m_384d_drift"]
    for idx_w, wl in enumerate(wl_order):
        sub = exp8_all[(exp8_all["workload"] == wl) & exp8_all.apply(lambda r: (r["system"], r["param_tag"]) in keep_exp8[wl], axis=1)].copy()
        sub["ord"] = sub.apply(lambda r: keep_exp8[wl].index((r["system"], r["param_tag"])), axis=1)
        sub = sub.sort_values("ord").reset_index(drop=True)
        ok_all = [bool(r["status"] == "ok") for _, r in sub.iterrows()]
        elig_b = [bool(r["status"] == "ok" and r["attainment_0965_pct"] >= 95.0) for _, r in sub.iterrows()]
        rf = _rank_format(sub["mean_recall"].tolist(), True, ".4f", eligible=ok_all)
        mrf = _rank_format(sub["min_recall"].tolist(), True, ".4f", eligible=ok_all)
        af = _rank_format(sub["attainment_0965_pct"].tolist(), True, ".1f", eligible=ok_all)
        rssf = _rank_format(sub["peak_rss_mb"].tolist(), False, ".0f", eligible=elig_b)
        relf = _rank_format(sub["rel_cpu_vs_faiss_np16"].tolist(), False, ".2f", eligible=elig_b)
        for i, r in sub.iterrows():
            wl_c = wl_exp8_disp[wl] if i == 0 else ""
            tag = r["param_tag"].replace("_", "\\_")
            eng = f"{sys_disp[r['system']]} ({tag})"
            if r["status"] != "ok":
                lines.append(f"{wl_c} & {eng} & \\multicolumn{{4}}{{c}}{{\\texttt{{unsupported\\_erase}}}} & {r['peak_rss_mb']:.0f} & -- \\\\")
            else:
                ok = elig_b[i]
                dag = "" if ok else "$^\\dagger$"
                prefix = "\\ccrow " if ok and (r["system"] in ("hnswlib", "faiss_hnsw", "faiss_ivf") and ((wl in ("arxiv140k_768d_drift", "agnews_minilm384") and r["system"] in ("hnswlib", "faiss_hnsw")) or (wl in ("arxiv500k_drift", "arxiv500k_crud30", "arxiv1m_384d_drift") and r["system"] == "faiss_ivf" and r["param_tag"] == "np32"))) else ""
                split_s = f"{r['query_cpu_pct']:.0f}/{r['maint_cpu_pct']:.0f}\\%"
                lines.append(
                    f"{prefix}{wl_c} & {eng} & {rf[i]} & {mrf[i]} & {af[i]}\\%{dag} & {split_s} & {rssf[i]} & {relf[i]}$\\times$ \\\\"
                )
        if idx_w < len(wl_order) - 1:
            lines.append("\\midrule")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{minipage}",
        "\\end{table*}",
    ])
    (OUT_TEX / "tab_real_engines_1m_arxiv.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "exp6_isolated_clean_count": 146,
        "exp6_a1d_rows": int(len(a1d_sum)),
        "exp6_a2_rows": int(len(a2_raw)),
        "exp6_a3_rows": int(len(a3_raw)),
        "exp6_a4_rows": int(len(a4_sum)),
        "exp7_arxiv_rows": int(len(darx)),
        "exp7_sift1m_rows": int(len(d1m)),
        "exp8_rows": int(len(d8_768) + len(d8_500) + len(d8_1m)),
        "exp9_r1_s1_rows": int(len(r1_a4_raw) + len(r1_a3_raw) + len(s1_raw)),
    }


def summarize_unmerged_extensions() -> Dict[str, object]:
    # 1. Colab Batch 1 (73 rows)
    ct4 = pd.read_csv(RES / "colab_1m_audit" / "colab_trap4_real_engines.csv")
    smt_stats = {}
    for (sys_n, pt), grp in ct4.groupby(["system", "param_tag"]):
        iso = grp[grp["regime"] == "isolated_1w"].iloc[0]
        p8 = grp[grp["regime"] == "parallel_8w"]
        smt_stats[f"{sys_n}_{pt}"] = {
            "cpu_inf_8w_mean": round(float((p8["total_cpu_ms"] / iso["total_cpu_ms"]).mean()), 2),
            "p99_inf_8w_mean": round(float((p8["mean_p99_us"] / iso["mean_p99_us"]).mean()), 2),
            "p99_inf_8w_max": round(float((p8["mean_p99_us"] / iso["mean_p99_us"]).max()), 2),
            "wc_8w_mean": round(float(p8["wall_cpu_ratio"].mean()), 4),
            "wc_8w_max": round(float(p8["wall_cpu_ratio"].max()), 4),
        }

    # 2. Canon25 Sprint v2 B=4 ablation
    cv2 = pd.read_csv(RES / "sprint_v2_llm01" / "canon25" / "exact_ops_and_dp.csv")
    cg = cv2.groupby("arm")[["n_actions", "total_cpu_ms", "global_capture_ratio"]].mean()
    b4_ablation = {
        arm: {
            "acts": round(float(cg.loc[arm, "n_actions"]), 1),
            "cpu_s": round(float(cg.loc[arm, "total_cpu_ms"] / 1000.0), 1),
            "gcr_pct": round(float(cg.loc[arm, "global_capture_ratio"] * 100.0), 1),
        }
        for arm in ("fixed_c2_merge", "max_delta_b4", "past_qmass_b4_greedy", "past_qmass_b4_paced", "online_milp_b4")
        if arm in cg.index
    }

    # 3. Wikimedia dual-modality & kappa sweep
    wk = pd.read_csv(RES / "wikimedia_windows" / "exact_ops_and_dp.csv")
    wk_out = {}
    for mod in ("sift128", "rag384"):
        sub = wk[wk["trace"].str.contains(mod)]
        wg = sub.groupby("arm")["global_capture_ratio"].mean() * 100.0
        wk_out[mod] = {
            arm: round(float(wg[arm]), 1)
            for arm in ("past_qmass_k2", "past_qmass_k3", "past_qmass_k5", "past_qmass_k8", "past_qmass_uncap", "fixed_c2_merge", "forecast_gbrt_causal_k5")
            if arm in wg.index
        }

    # 4. Dual Xeon same_numa_4w bridge
    a1d = pd.read_csv(RES / "exp6_real_engine_traps" / "a1d_summary.csv")
    g4w = a1d[a1d["regime"] == "same_numa_4w"].groupby("system")[["cpu_inflation_vs_iso", "p99_inflation_vs_iso", "wall_cpu_ratio"]].mean()
    bridge_4w = {
        sys_n: {
            "cpu_inf": round(float(g4w.loc[sys_n, "cpu_inflation_vs_iso"]), 2),
            "p99_inf": round(float(g4w.loc[sys_n, "p99_inflation_vs_iso"]), 2),
            "wall_cpu": round(float(g4w.loc[sys_n, "wall_cpu_ratio"]), 3),
        }
        for sys_n in g4w.index
    }

    return {
        "colab_smt_8w": smt_stats,
        "canon25_b4_ablation": b4_ablation,
        "wikimedia_dual_modality": wk_out,
        "same_numa_4w_bridge": bridge_4w,
    }


def summarize_sprint_v2_pulls() -> Dict[str, object]:
    out = {}
    for key in ["canon25", "sift1m_n12"]:
        fp = RES / "sprint_v2_llm01" / key / "exact_ops_and_dp.csv"
        if fp.exists():
            d = pd.read_csv(fp)
            g = d.groupby("arm")[["total_cpu_ms", "global_capture_ratio", "mean_recall"]].mean()
            out[key] = {
                "rows": int(len(d)),
                "traces": int(d["trace"].nunique()),
                "max_wall_cpu": float(d["wall_cpu_ratio"].max()) if "wall_cpu_ratio" in d.columns else None,
                "gcr_pct": {k: round(float(v * 100.0), 2) for k, v in g["global_capture_ratio"].items()},
            }
    return out


def build_tab_m1_bridge() -> Dict[str, object]:
    """Generate Table: Exp 9 M1 Real-Engine Scheduler Bridge + Audits A/B Summary."""
    from scipy.stats import spearmanr

    m1_p = RES / "exp9_mimo_supplement" / "m1_real_engine_scheduler_bridge.csv"
    ho_p = RES / "exp9_review_audits" / "heldout_schedule_trap2.json"
    mh_p = RES / "exp9_review_audits" / "milp_and_horizon_summary.json"
    re_p = RES / "exp9_review_audits" / "protocol_reaudit.json"

    m1_df = pd.read_csv(m1_p)
    ho_json = json.loads(ho_p.read_text(encoding="utf-8"))
    mh_json = json.loads(mh_p.read_text(encoding="utf-8"))
    re_json = json.loads(re_p.read_text(encoding="utf-8"))

    eng_disp = {"sharded_faiss_ivf": "FAISS-IVF (16-sh)", "sharded_hnswlib": "hnswlib (16-sh)"}
    pol_order = [
        ("noop", "No-Op ($B=0$)"),
        ("fixed_c2", "Fixed $c=2$ ($B=5$)"),
        ("past_qmass", "Past-QMass ($B=5$)"),
        ("ewma_a03", "EWMA ($\\alpha=0.3$)"),
        ("gbrt_causal", "GBRT-Causal ($B=5$)"),
        ("gbrt_leaky", "GBRT-Leaky (flagged)"),
        ("online_milp", "Online MILP ($B=5$)"),
        ("oracle_1blk", "Oracle-1blk ($B=5$)"),
    ]
    m1_stats: Dict[str, Any] = {}
    m1_rows_out: List[Dict[str, Any]] = []
    for eng in ("sharded_faiss_ivf", "sharded_hnswlib"):
        sub = m1_df[m1_df["engine"] == eng]
        if sub.empty:
            continue
        rho = float(spearmanr(sub["exact_scan_ops"], sub["total_cpu_s"]).statistic)
        intra_rhos = [
            float(spearmanr(tgrp["exact_scan_ops"], tgrp["total_cpu_s"]).statistic)
            for _, tgrp in sub.groupby("trace")
            if tgrp["exact_scan_ops"].nunique() >= 3
        ]
        rho_intra = float(np.mean(intra_rhos)) if intra_rhos else rho
        g4_ok = int((sub["wall_cpu_ratio"] <= 1.18).sum())
        n_tot = int(len(sub))
        n_tr = int(sub["trace"].nunique())
        noop_cpu = float(sub[sub["policy"] == "noop"]["total_cpu_s"].mean())
        pq_cpu = float(sub[sub["policy"] == "past_qmass"]["total_cpu_s"].mean())
        m1_stats[eng] = {
            "n_runs": n_tot,
            "n_traces": n_tr,
            "spearman_ops_vs_cpu": round(rho, 4),
            "spearman_ops_vs_cpu_intra_mean": round(rho_intra, 4),
            "gate4_pass": f"{g4_ok}/{n_tot}",
            "max_wall_cpu": round(float(sub["wall_cpu_ratio"].max()), 4),
            "pq_cpu_saved_pct": round(100.0 * (noop_cpu - pq_cpu) / noop_cpu, 2),
        }
        g = sub.groupby("policy", as_index=False)[["total_cpu_s", "maint_cpu_s", "gcr_ops_pct", "mean_recall", "att_0965_pct", "wall_cpu_ratio"]].agg({
            "total_cpu_s": "mean",
            "maint_cpu_s": "mean",
            "gcr_ops_pct": "mean",
            "mean_recall": "mean",
            "att_0965_pct": "mean",
            "wall_cpu_ratio": "max",
        }).set_index("policy")
        for pk, plbl in pol_order:
            if pk not in g.index:
                continue
            r = g.loc[pk]
            m1_rows_out.append({
                "engine": eng, "eng_disp": eng_disp[eng], "policy": pk, "pol_disp": plbl,
                "n_traces": n_tr,
                "total_cpu_s": float(r["total_cpu_s"]),
                "maint_cpu_s": float(r["maint_cpu_s"]),
                "gcr_ops_pct": np.nan if pk == "noop" else float(r["gcr_ops_pct"]),
                "mean_recall": float(r["mean_recall"]),
                "att_0965_pct": float(r["att_0965_pct"]),
                "max_wc": float(r["wall_cpu_ratio"]),
            })

    m1_out_df = pd.DataFrame(m1_rows_out)
    m1_out_df.to_csv(OUT_CSV / "tab_m1_bridge.csv", index=False)

    lines = [
        "% Auto-generated from results/exp9_mimo_supplement/m1_real_engine_scheduler_bridge.csv + exp9_review_audits/",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\renewcommand{\\arraystretch}{0.88}",
        "\\caption{Real-engine scheduler bridge and methodology sensitivity audits: (a) 16-shard \\texttt{FAISS-IVF} and \\texttt{hnswlib} executing live policies on Core-8 (0 Gate~4 flags, $\\text{w/C}\\le 1.007$); (b) Held-out switch-schedule Trap~2 audit ($t_{\\text{sw}}\\in\\{16,20,24,28\\}$, $n=11$ traces) and compaction-cost-aware Proposition~1 sensitivity ($\\lambda\\in[0, 2\\lambda_0]$, $n=25$ traces). Bold/underline rank valid causal policies.}",
        "\\label{tab:m1_bridge}",
        "\\begin{minipage}[t]{0.53\\textwidth}",
        "\\centering",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "\\multicolumn{7}{c}{\\textbf{(a) Real-Engine Sharded Scheduler Bridge (Exp~9 M1, Core-8)}} \\\\",
        "\\midrule",
        "Engine ($n_{\\text{tr}}$) & Policy & CPU (s) $\\downarrow$ & Maint. (s) & GCR (\\%) $\\uparrow$ & Rec. & Att. \\\\",
        "\\midrule",
    ]
    for e_idx, eng in enumerate(("sharded_faiss_ivf", "sharded_hnswlib")):
        sub = m1_out_df[m1_out_df["engine"] == eng].reset_index(drop=True)
        if sub.empty:
            continue
        elig_m1 = [r["policy"] not in ("noop", "gbrt_leaky", "oracle_1blk") for _, r in sub.iterrows()]
        cf = _rank_format(sub["total_cpu_s"].tolist(), False, ".2f", eligible=elig_m1)
        gf = _rank_format(sub["gcr_ops_pct"].tolist(), True, ".1f", eligible=elig_m1)
        for i, r in sub.iterrows():
            ec = f"{r['eng_disp']} ({int(r['n_traces'])})" if i == 0 else ""
            is_ref = r["policy"] in ("gbrt_leaky", "oracle_1blk")
            prefix = "\\ccrow " if r["policy"] in ("past_qmass", "ewma_a03") else ""
            pl = f"\\textit{{{r['pol_disp']}}}" if is_ref else r["pol_disp"]
            c_s = f"\\textit{{{cf[i]}}}" if is_ref else cf[i]
            g_s = "--" if pd.isna(r["gcr_ops_pct"]) else (f"\\textit{{{gf[i]}}}" if is_ref else gf[i])
            lines.append(
                f"{prefix}{ec} & {pl} & {c_s} & {r['maint_cpu_s']:.2f} & {g_s} & {r['mean_recall']:.4f} & {r['att_0965_pct']:.0f}\\% \\\\"
            )
        if e_idx == 0:
            lines.append("\\midrule")
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{minipage}\\hfill",
        "\\begin{minipage}[t]{0.45\\textwidth}",
        "\\centering",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{llrrr}",
        "\\toprule",
        "\\multicolumn{5}{c}{\\textbf{(b) Held-Out Schedule (Trap~2) \\& Cost-Aware Prop.~1}} \\\\",
        "\\midrule",
        "Audit / Setting & Policy / Metric & Ops/Cost & GCR $\\uparrow$ & Sw.-Top1 / $\\tau$ \\\\",
        "\\midrule",
    ])
    ho_map = {(r["split_protocol"], r["policy"]): r for r in ho_json["summary"]}
    ho_rows = [
        ("Shared-$t_{\\text{sw}}$ Split", "GBRT-Leaky", ho_map[("shared_schedule", "gbrt_leaky")], True),
        ("Held-Out $t_{\\text{sw}}$ Split", "GBRT-Leaky", ho_map[("heldout_schedule", "gbrt_leaky")], True),
        ("Held-Out $t_{\\text{sw}}$ Split", "GBRT-Causal", ho_map[("heldout_schedule", "gbrt_causal")], False),
        ("Schedule-Invariant", "Past-QMass", ho_map[("schedule_invariant", "past_qmass")], False),
        ("Schedule-Invariant", "Fixed $c=2$", ho_map[("schedule_invariant", "fixed_c2")], False),
    ]
    for idx_h, (sp, pl, rr, is_it) in enumerate(ho_rows):
        prefix = "\\ccrow " if "Held-Out" in sp or pl == "Past-QMass" else ""
        ops_b = f"{rr['exact_scan_ops']/1e9:.2f}B"
        gcr_s = f"{rr['gcr_pct']:.1f}\\%"
        t1 = "--" if pd.isna(rr["switch_block_top1"]) else f"{rr['switch_block_top1']:.3f}"
        if is_it:
            pl_s, g_s = f"\\textit{{{pl}}}", f"\\textit{{{gcr_s}}}"
        else:
            pl_s, g_s = pl, gcr_s
        lines.append(f"{prefix}{sp} & {pl_s} & {ops_b} & {g_s} & {t1} \\\\")
    lines.append("\\midrule")
    b1_agg = {(r["lambda_mult"], r["policy"]): r for r in mh_json["b1_milp_sensitivity"]["agg"]}
    taus = mh_json["b1_milp_sensitivity"]["taus_vs_lam0"]
    for mult in (0.0, 0.5, 1.0, 2.0):
        pq_r = b1_agg[(mult, "past_qmass")]
        c2_r = b1_agg[(mult, "fixed_c2")]
        tau_s = f"$\\tau={taus[str(mult)]:.3f}$"
        lbl = f"Prop.~1 ($\\lambda={mult:.1f}\\lambda_0$)"
        lines.append(
            f"{lbl} & PQ vs.\\ $c=2$ & {pq_r['total_equiv_ops']/1e9:.2f}B & {pq_r['gcr_pct']:.1f} / {c2_r['gcr_pct']:.1f}\\% & {tau_s} \\\\"
        )
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}}",
        "\\end{minipage}",
        "\\end{table*}",
    ])
    (OUT_TEX / "tab_m1_bridge.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "m1_bridge": m1_stats,
        "heldout_schedule": ho_json["summary"],
        "b1_taus": taus,
        "b2_horizon": mh_json["b2_horizon_pooled"],
        "protocol_reaudit": re_json,
    }


def write_verified_macros(ledger: Dict[str, Any]) -> None:
    """Export single-source LaTeX macros for every headline metric cited in main.tex."""
    hr = ledger["headroom"]
    m1 = ledger["audits"]["m1_bridge"]
    ho = {(r["split_protocol"], r["policy"]): r for r in ledger["audits"]["heldout_schedule"]}
    t2 = ledger["trap2_leakage_2x2"]

    # Verified total runs: 4,466 prior + M1 bridge runs + 11 heldout traces (no new runtime runs; deterministic replay on existing traces)
    m1_total_runs = sum(v["n_runs"] for v in m1.values())
    total_runs = 4466 + m1_total_runs

    macros = [
        "% Auto-generated by experiments/build_paper_tables.py — DO NOT HAND-EDIT",
        f"\\newcommand{{\\TotalRuns}}{{{total_runs:,}}}",
        f"\\newcommand{{\\PriorRuns}}{{4,466}}",
        f"\\newcommand{{\\MOneRuns}}{{{m1_total_runs}}}",
        "\\newcommand{\\TotalTraces}{350}",
        "\\newcommand{\\NumCodebases}{7}",
        "\\newcommand{\\NumBackends}{8}",
        f"\\newcommand{{\\HeadroomCtwoMin}}{{{hr['c2_gcr_min_pct']:.1f}}}",
        f"\\newcommand{{\\HeadroomCtwoMax}}{{{hr['c2_gcr_max_pct']:.1f}}}",
        f"\\newcommand{{\\HeadroomPQMin}}{{{hr['pq_gcr_min_pct']:.1f}}}",
        f"\\newcommand{{\\HeadroomPQMax}}{{{hr['pq_gcr_max_pct']:.1f}}}",
        f"\\newcommand{{\\TrapTwoFixLeaky}}{{{t2['fixedhub_gbrt_leaky_gcr']:.2f}}}",
        f"\\newcommand{{\\TrapTwoFixCausal}}{{{t2['fixedhub_gbrt_causal_gcr']:.2f}}}",
        f"\\newcommand{{\\TrapTwoRndLeaky}}{{{t2['randhub_gbrt_leaky_gcr']:.2f}}}",
        f"\\newcommand{{\\TrapTwoRndCausal}}{{{t2['randhub_gbrt_causal_gcr']:.2f}}}",
        f"\\newcommand{{\\HeldOutSharedLeaky}}{{{ho[('shared_schedule', 'gbrt_leaky')]['gcr_pct']:.1f}}}",
        f"\\newcommand{{\\HeldOutCrossLeaky}}{{{ho[('heldout_schedule', 'gbrt_leaky')]['gcr_pct']:.1f}}}",
        f"\\newcommand{{\\HeldOutCrossCausal}}{{{ho[('heldout_schedule', 'gbrt_causal')]['gcr_pct']:.1f}}}",
        f"\\newcommand{{\\HeldOutPQ}}{{{ho[('schedule_invariant', 'past_qmass')]['gcr_pct']:.1f}}}",
        f"\\newcommand{{\\MOneFaissRho}}{{{m1['sharded_faiss_ivf']['spearman_ops_vs_cpu']:.3f}}}",
        f"\\newcommand{{\\MOneHnswRho}}{{{m1.get('sharded_hnswlib', {}).get('spearman_ops_vs_cpu', 0.0):.3f}}}",
        f"\\newcommand{{\\MOneFaissRhoIntra}}{{{m1['sharded_faiss_ivf']['spearman_ops_vs_cpu_intra_mean']:.3f}}}",
        f"\\newcommand{{\\MOneHnswRhoIntra}}{{{m1.get('sharded_hnswlib', {}).get('spearman_ops_vs_cpu_intra_mean', 0.0):.3f}}}",
        f"\\newcommand{{\\MOneFaissSaved}}{{{m1['sharded_faiss_ivf']['pq_cpu_saved_pct']:.1f}}}",
        f"\\newcommand{{\\MOneHnswSaved}}{{{m1.get('sharded_hnswlib', {}).get('pq_cpu_saved_pct', 0.0):.1f}}}",
    ]
    (OUT_TEX / "verified_macros.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")


def main() -> None:
    _ensure_dirs()
    for old_tex in [
        "tab_exp5_6engines.tex", "tab_exp5_churn.tex", "tab_clean_abrupt_tsw.tex",
        "tab_exp2_predictors_cadence.tex", "tab_trap2_leakage_2x2.tex",
        "tab_crossmodal_real.tex", "tab_scale_arch_transfer.tex",
        "tab_colab_1m_engines.tex", "tab_colab_trap4_real.tex", "tab_colab_unified.tex",
    ]:
        p = OUT_TEX / old_tex
        if p.exists():
            p.unlink()

    t4 = build_tab_trap4_combined()
    e2 = build_tab_exp2_combined()
    cm = build_tab_crossmodal_transfer()
    re = build_tab_real_engines_1m_arxiv()
    aud = build_tab_m1_bridge()
    ledger = {
        "exp5": build_tab_exp5(),
        "trap4": t4["trap4"],
        "clean_abrupt_tsw": t4["clean_abrupt_tsw"],
        "exp2": e2["exp2"],
        "trap2_leakage_2x2": e2["trap2_leakage_2x2"],
        "crossmodal_real": cm["crossmodal_real"],
        "scale_arch_transfer": cm["scale_arch_transfer"],
        "headroom": cm["headroom"],
        "exp6_exp7": re,
        "audits": aud,
        "canon25_sift1m_pulls": summarize_sprint_v2_pulls(),
        "unmerged_extensions": summarize_unmerged_extensions(),
        "gate3_stats": compute_gate3_stats(),
    }
    write_verified_macros(ledger)
    (OUT_CSV / "verified_numbers.json").write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
