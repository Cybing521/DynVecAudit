#!/usr/bin/env python3
"""Offline Gate Threshold Sensitivity Audit for DynVecAudit (PVLDB Vol. 20 EA&B).

Sweeps thresholds across archived experimental outputs (zero re-runs needed):
  1. Gate 4 tau_wc in [1.05, 1.10, 1.15, 1.18, 1.25, 1.50]
  2. Gate 2 tau_rho in [0.20, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95]
  3. Recall-attainment floor in [0.90, 0.95, 0.965, 0.98]

Outputs:
  - results/paper_tables/gate_sensitivity_audit.csv
  - results/paper_tables/gate_sensitivity_audit.json
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
RES = ROOT / "results"
OUT_DIR = RES / "paper_tables"

sys.path.insert(0, str(THIS_DIR))
from compute_flip_matrix import compare_rankings  # noqa: E402


def _safe_ratio_series(df: pd.DataFrame) -> np.ndarray:
    if "wall_cpu_ratio" in df.columns:
        return pd.to_numeric(df["wall_cpu_ratio"], errors="coerce").dropna().to_numpy(dtype=float)
    if "wall_ms" in df.columns and "total_cpu_ms" in df.columns:
        w = pd.to_numeric(df["wall_ms"], errors="coerce")
        c = pd.to_numeric(df["total_cpu_ms"], errors="coerce")
        return (w / np.maximum(c, 1e-9)).dropna().to_numpy(dtype=float)
    if "wall_s" in df.columns and "total_cpu_ms" in df.columns:
        w = pd.to_numeric(df["wall_s"], errors="coerce") * 1000.0
        c = pd.to_numeric(df["total_cpu_ms"], errors="coerce")
        return (w / np.maximum(c, 1e-9)).dropna().to_numpy(dtype=float)
    return np.array([], dtype=float)


def sweep_gate4_tau_wc() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    taus = [1.05, 1.10, 1.15, 1.18, 1.25, 1.50]

    clean_ratios: List[float] = []
    # Isolated Dual Xeon Core-0 real-engine runs (Exp 6 A4 + A3 + isolated A1D)
    a4 = pd.read_csv(RES / "exp6_real_engine_traps" / "a4_attainment_sweep.csv")
    a3 = pd.read_csv(RES / "exp6_real_engine_traps" / "a3_crud_engines.csv")
    a1d = pd.read_csv(RES / "exp6_real_engine_traps" / "a1d_summary.csv")
    a1d_iso = a1d[a1d["regime"].isin(["iso_numa0_1w", "iso_numa0_1t_batch"])]
    for df in (a4, a3, a1d_iso):
        clean_ratios.extend(_safe_ratio_series(df).tolist())

    # Clean Core-8 / Pinned replay suites (abrupt_tsw_clean, phase_uxq_clean)
    tsw_clean_dfs = [pd.read_csv(p) for p in sorted((RES / "abrupt_tsw_clean").glob("*.csv"))]
    uxq_clean_dfs = [pd.read_csv(p) for p in sorted((RES / "phase_uxq_clean").glob("*.csv"))]
    for df in tsw_clean_dfs + uxq_clean_dfs:
        clean_ratios.extend(_safe_ratio_series(df).tolist())

    # Isolated Xeon NUMA-0 in stage1_hygiene
    s1 = pd.read_csv(RES / "stage1_hygiene" / "hygiene_summary.csv")
    s1_iso = s1[s1["regime"] == "isolated_numa0"]
    clean_ratios.extend(_safe_ratio_series(s1_iso).tolist())

    # Contended runs: stage1_hygiene non-isolated regimes + exp6 contended A1D
    s1_cont = s1[s1["regime"] != "isolated_numa0"]
    a1d_cont = a1d[a1d["regime"].isin(["ht_sibling_2w", "same_socket_4w", "cross_socket_8w", "mt_same_numa_4t", "mt_unpinned_8t"])]
    cont_ratios = np.concatenate([_safe_ratio_series(s1_cont), _safe_ratio_series(a1d_cont)])
    clean_arr = np.asarray(clean_ratios, dtype=float)

    # Check dirty-vs-clean ranking flip stability across tau_wc on abrupt_tsw_clean + phase_uxq_clean
    paired_grids: List[Tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for p in sorted((RES / "abrupt_tsw_clean").glob("*.csv")):
        dp = RES / "abrupt_tsw" / p.name
        if dp.exists():
            paired_grids.append((p.stem, pd.read_csv(dp), pd.read_csv(p)))
    for p in sorted((RES / "phase_uxq_clean").glob("*.csv")):
        dp = RES / "phase_uxq" / p.name
        if dp.exists():
            paired_grids.append((p.stem, pd.read_csv(dp), pd.read_csv(p)))

    rows: List[Dict[str, Any]] = []
    for tau in taus:
        clean_pass = int(np.sum(clean_arr <= tau))
        clean_flag = int(np.sum(clean_arr > tau))
        cont_flag = int(np.sum(cont_ratios > tau))
        s1_cont_r = _safe_ratio_series(s1_cont)
        s1_cont_flag = int(np.sum(s1_cont_r > tau))

        # Recompute dirty-vs-clean flip rate keeping clean grids whose median wall/cpu <= tau
        kept_units = 0
        flips = 0
        pairs = 0
        top1_flips = 0
        for _gname, ddf, cdf in paired_grids:
            c_med_ratio = float(np.median(_safe_ratio_series(cdf)))
            if c_med_ratio > tau:
                continue
            kept_units += 1
            da = ddf.groupby("arm")["total_cpu_ms"].mean()
            ca = cdf.groupby("arm")["total_cpu_ms"].mean()
            common = sorted(set(da.index) & set(ca.index))
            cmp = compare_rankings(common, [da[a] for a in common], [ca[a] for a in common], higher_is_better=False)
            flips += cmp["pair_flips"]
            pairs += cmp["total_pairs"]
            if cmp["top1_flipped"]:
                top1_flips += 1

        rows.append({
            "gate": "Gate 4 (Contention)",
            "parameter": "tau_wc",
            "threshold": tau,
            "n_clean_total": int(len(clean_arr)),
            "clean_pass_count": clean_pass,
            "clean_pass_pct": round(100.0 * clean_pass / max(len(clean_arr), 1), 2),
            "clean_flag_pct": round(100.0 * clean_flag / max(len(clean_arr), 1), 2),
            "n_contended_stage1": int(len(s1_cont_r)),
            "stage1_contended_caught_pct": round(100.0 * s1_cont_flag / max(len(s1_cont_r), 1), 2),
            "n_contended_all": int(len(cont_ratios)),
            "all_contended_caught_pct": round(100.0 * cont_flag / max(len(cont_ratios), 1), 2),
            "kept_clean_grids": kept_units,
            "dirty_vs_clean_pair_flip_pct": round(100.0 * flips / max(pairs, 1), 2) if pairs else None,
            "dirty_vs_clean_top1_flip_pct": round(100.0 * top1_flips / max(kept_units, 1), 2) if kept_units else None,
        })

    dist_stats = {
        "clean_min": round(float(np.min(clean_arr)), 4),
        "clean_median": round(float(np.median(clean_arr)), 4),
        "clean_p95": round(float(np.percentile(clean_arr, 95)), 4),
        "clean_max": round(float(np.max(clean_arr)), 4),
        "stage1_contended_min": round(float(np.min(_safe_ratio_series(s1_cont))), 4),
        "stage1_contended_median": round(float(np.median(_safe_ratio_series(s1_cont))), 4),
        "stage1_contended_max": round(float(np.max(_safe_ratio_series(s1_cont))), 4),
    }
    return rows, dist_stats


def sweep_gate2_tau_rho() -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    taus = [0.20, 0.50, 0.65, 0.75, 0.85, 0.90, 0.95]
    feats: List[Tuple[str, str, float, bool]] = []  # (source, name, abs_rho, is_leaky_ground_truth)

    exp2_pd = json.loads((RES / "exp2_predictors_cadence" / "predictor_diagnostics.json").read_text())
    for k, v in exp2_pd["clock_leak_audit"]["gbrt_leaky"]["all_rhos"].items():
        feats.append(("Exp2_SIFT25", k, abs(float(v)), k.startswith("leak_")))

    exp7_pd = json.loads((RES / "exp7_gpu_1m_arxiv" / "arxiv_minilm384_gate2_spearman.json").read_text())
    for k, v in exp7_pd["clock_leak_audit"]["gbrt_leaky"]["all_rhos"].items():
        feats.append(("Exp7_arXivCS_384d", k, abs(float(v)), k.startswith("leak_")))

    exp8_pd = json.loads((RES / "exp8_colab_large_scale" / "semantic_scale_gate2_spearman.json").read_text())
    for suite_name, rdict in exp8_pd.items():
        for k, v in rdict.items():
            feats.append((f"Exp8_{suite_name}", k, abs(float(v)), k.startswith("leak_")))

    causal_vals = [r for _, _, r, lk in feats if not lk]
    leaky_vals = [r for _, _, r, lk in feats if lk]

    rows: List[Dict[str, Any]] = []
    for tau in taus:
        tp = sum(1 for _, _, r, lk in feats if lk and r >= tau)
        fp = sum(1 for _, _, r, lk in feats if (not lk) and r >= tau)
        fn = sum(1 for _, _, r, lk in feats if lk and r < tau)
        tn = sum(1 for _, _, r, lk in feats if (not lk) and r < tau)
        prec = tp / max(tp + fp, 1) if (tp + fp) > 0 else 1.0
        rec = tp / max(tp + fn, 1)
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        rows.append({
            "gate": "Gate 2 (Clock Leak)",
            "parameter": "tau_rho",
            "threshold": tau,
            "n_causal_features": len(causal_vals),
            "n_leaky_features": len(leaky_vals),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "all_suites_separated": bool(fp == 0 and fn == 0),
        })

    margin_stats = {
        "max_causal_abs_rho": round(max(causal_vals), 4),
        "min_leaky_abs_rho": round(min(leaky_vals), 4),
        "empty_margin_width": round(min(leaky_vals) - max(causal_vals), 4),
        "leaky_values_below_085": [
            {"source": s, "feature": f, "abs_rho": round(r, 4)}
            for s, f, r, lk in feats if lk and r < 0.85
        ],
    }
    return rows, margin_stats


def _eval_floor_suite(
    suite_name: str,
    df: pd.DataFrame,
    sys_col: str,
    cfg_col: str,
    cost_col: str,
    recall_col: str,
    criterion_label: str,
    floors: Sequence[float],
    group_col: str | None = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    sub_dfs = [(None, df)] if group_col is None else list(df.groupby(group_col))

    for fl in floors:
        total_flips = 0
        total_pairs = 0
        top1_flips = 0
        n_units = 0
        taus: List[float] = []
        n_qual_total = 0
        n_total_arms = 0
        ex_unmatched = ""
        ex_matched = ""

        for gval, gdf in sub_dfs:
            gdf = gdf.copy()
            if "status" in gdf.columns:
                gdf = gdf[gdf["status"] == "ok"].copy()
            gdf[cost_col] = pd.to_numeric(gdf[cost_col], errors="coerce")
            gdf[recall_col] = pd.to_numeric(gdf[recall_col], errors="coerce")
            gdf = gdf.dropna(subset=[cost_col, recall_col])
            if len(gdf) < 2:
                continue

            agg = gdf.groupby([sys_col, cfg_col], as_index=False).agg(
                cost=(cost_col, "mean"),
                rec=(recall_col, "mean"),
            )
            agg["arm"] = agg[sys_col].astype(str) + "_" + agg[cfg_col].astype(str)
            n_arms = len(agg)
            if n_arms < 2:
                continue
            n_total_arms += n_arms
            qual = agg["rec"] >= fl
            n_qual_total += int(qual.sum())

            # Unmatched rank: raw cost ascending. Matched rank: qualified first, then cost ascending.
            unmatched_scores = agg["cost"].to_numpy(dtype=float)
            penalty = max(float(agg["cost"].max()) * 100.0, 1e9)
            matched_scores = np.where(qual, agg["cost"], agg["cost"] + penalty)
            cmp = compare_rankings(agg["arm"].tolist(), unmatched_scores, matched_scores, higher_is_better=False)
            total_flips += cmp["pair_flips"]
            total_pairs += cmp["total_pairs"]
            taus.append(cmp["kendall_tau"])
            n_units += 1
            if cmp["top1_flipped"]:
                top1_flips += 1
                if not ex_unmatched:
                    ex_unmatched = cmp["top1_dirty"]
                    ex_matched = cmp["top1_clean"]

        if n_units == 0:
            continue
        rows.append({
            "gate": "Recall Floor",
            "parameter": "recall_floor",
            "threshold": fl,
            "suite": suite_name,
            "criterion": criterion_label,
            "n_units": n_units,
            "qualified_arms_frac": f"{n_qual_total}/{n_total_arms}",
            "pair_flips": total_flips,
            "total_pairs": total_pairs,
            "pairwise_flip_pct": round(100.0 * total_flips / max(total_pairs, 1), 2),
            "kendall_tau": round(float(np.mean(taus)), 4),
            "top1_flip_pct": round(100.0 * top1_flips / max(n_units, 1), 2),
            "top1_unmatched_ex": ex_unmatched,
            "top1_matched_ex": ex_matched,
        })
    return rows


def sweep_recall_floor() -> List[Dict[str, Any]]:
    floors = [0.90, 0.95, 0.965, 0.98]
    rows: List[Dict[str, Any]] = []

    # 1. Exp 6 A4 Dual Xeon Core-0 Floor Sweep (17 configs across 3 traces)
    a4 = pd.read_csv(RES / "exp6_real_engine_traps" / "a4_attainment_sweep.csv")
    rows.extend(_eval_floor_suite("Exp6_Xeon_A4_Drift", a4, "system", "param_tag", "total_cpu_ms", "min_recall", "min_recall>=floor", floors))
    rows.extend(_eval_floor_suite("Exp6_Xeon_A4_Drift", a4, "system", "param_tag", "total_cpu_ms", "mean_recall", "mean_recall>=floor", floors))

    # 2. Exp 6 A3 CRUD & Runbook
    a3 = pd.read_csv(RES / "exp6_real_engine_traps" / "a3_crud_engines.csv")
    rows.extend(_eval_floor_suite("Exp6_Xeon_A3_CRUD", a3, "system", "param_tag", "total_cpu_ms", "min_recall", "min_recall>=floor", floors, group_col="workload"))

    # 3. Exp 5A 6-Engine Ryzen Core-8
    e5 = pd.read_csv(RES / "exp5_multi_engine" / "exp5_multi_engine_summary.csv")
    e5_drift = e5[e5["workload"] == "sift128_drift"]
    rows.extend(_eval_floor_suite("Exp5A_Ryzen_Drift", e5_drift, "system", "param_tag", "total_cpu_ms", "min_recall", "min_recall>=floor", floors))

    # 4. Exp 8 Colab Large-Scale Real Embeddings
    for fname, sname in [
        ("arxiv140k_768d_engines.csv", "Exp8_arXiv140K_768d"),
        ("arxiv500k_384d_engines.csv", "Exp8_arXiv500K_384d"),
        ("arxiv1m_384d_engines.csv", "Exp8_arXiv1M_384d"),
    ]:
        df8 = pd.read_csv(RES / "exp8_colab_large_scale" / fname)
        df8_d = df8[df8["workload"].str.contains("drift")]
        rows.extend(_eval_floor_suite(sname, df8_d, "system", "param_tag", "rel_cpu_vs_faiss_base", "min_recall", "min_recall>=floor", floors))
    return rows


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g4_rows, g4_dist = sweep_gate4_tau_wc()
    g2_rows, g2_margin = sweep_gate2_tau_rho()
    rf_rows = sweep_recall_floor()

    all_rows = g4_rows + g2_rows + rf_rows
    out_csv = OUT_DIR / "gate_sensitivity_audit.csv"
    keys: List[str] = []
    seen = set()
    for r in all_rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(all_rows)

    out_json = OUT_DIR / "gate_sensitivity_audit.json"
    payload = {
        "gate4_tau_wc_sweep": g4_rows,
        "gate4_ratio_distribution": g4_dist,
        "gate2_tau_rho_sweep": g2_rows,
        "gate2_margin_stats": g2_margin,
        "recall_floor_sweep": rf_rows,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print("=== GATE 4 tau_wc SWEEP ===")
    print("Distribution:", g4_dist)
    for r in g4_rows:
        print(f"  tau_wc={r['threshold']:<4} | clean_pass={r['clean_pass_pct']:>6}% ({r['clean_pass_count']}/{r['n_clean_total']}) | stage1_cont_caught={r['stage1_contended_caught_pct']:>6}% | dirty_vs_clean_flip={r['dirty_vs_clean_pair_flip_pct']}%")

    print("\n=== GATE 2 tau_rho SWEEP ===")
    print("Margin:", g2_margin)
    for r in g2_rows:
        print(f"  tau_rho={r['threshold']:<4} | TP={r['tp']} FP={r['fp']} FN={r['fn']} | P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f}")

    print("\n=== RECALL FLOOR SWEEP (min_recall>=floor) ===")
    for r in rf_rows:
        if r["criterion"] == "min_recall>=floor":
            print(f"  {r['suite']:<22} floor={r['threshold']:<5} | qual={r['qualified_arms_frac']:<6} | flip={r['pairwise_flip_pct']:>5}% | tau={r['kendall_tau']:>7} | top1_flip={r['top1_flip_pct']:>5}% ({r['top1_unmatched_ex']} -> {r['top1_matched_ex']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
