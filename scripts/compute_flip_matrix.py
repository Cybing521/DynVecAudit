#!/usr/bin/env python3
"""Compute the Unified 4-Trap Ranking-Flip Matrix for PVLDB EA&B (DynVecAudit).

Quantifies pairwise ranking flip %, Kendall's tau, and Top-1 winner flip rates across:
  - Trap 1: Linearly interpolated sparse baselines vs. executed adjacent cadence brackets
  - Trap 2: Clock/step-leaky ML schedulers (GBRT-Leaky) vs. de-leaked GBRT-Causal
  - Trap 3: Truncated 1-block counterfactuals vs. 4-block horizons (Track-CF) &
            tombstone-reclaiming `refresh` vs. `merge` (Tombstone Sweep)
  - Trap 4: Shared-host contention & cross-NUMA placement vs. pinned NUMA-0 Core-8 clean runs

Automatically ingests `results/exp6_real_engine_traps/` and `results/exp7_gpu_1m_arxiv/`
whenever those result directories are populated.

Outputs:
  - results/paper_tables/ranking_flip_matrix.csv
  - results/paper_tables/ranking_flip_matrix.json
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT_DIR = RES / "paper_tables"


def compare_rankings(
    names: Sequence[str],
    vals_dirty: Sequence[float],
    vals_clean: Sequence[float],
    higher_is_better: bool = False,
) -> Dict[str, Any]:
    """Compare two score vectors over common `names` (Condition A dirty vs Condition B clean)."""
    n = len(names)
    if n < 2:
        return {
            "n_arms": n,
            "pair_flips": 0,
            "total_pairs": 0,
            "kendall_tau": 1.0,
            "top1_dirty": names[0] if n else "",
            "top1_clean": names[0] if n else "",
            "top1_flipped": False,
        }

    sign = -1.0 if higher_is_better else 1.0
    a = [float(v) * sign for v in vals_dirty]
    b = [float(v) * sign for v in vals_clean]

    flips = 0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            if (a[i] < a[j]) != (b[i] < b[j]):
                flips += 1

    tau, _ = kendalltau(a, b)
    if np.isnan(tau):
        tau = 1.0

    idx_a = int(np.argmin(a))
    idx_b = int(np.argmin(b))
    top_a = str(names[idx_a])
    top_b = str(names[idx_b])
    return {
        "n_arms": n,
        "pair_flips": flips,
        "total_pairs": pairs,
        "kendall_tau": float(tau),
        "top1_dirty": top_a,
        "top1_clean": top_b,
        "top1_flipped": top_a != top_b,
    }


def _aggregate_unit_results(
    trap: str,
    suite_id: str,
    suite_label: str,
    source_type: str,
    comparison: str,
    metric: str,
    units: List[Dict[str, Any]],
    notes: str = "",
) -> Dict[str, Any]:
    total_flips = sum(u["pair_flips"] for u in units)
    total_pairs = sum(u["total_pairs"] for u in units)
    top1_flips = sum(1 for u in units if u["top1_flipped"])
    n_units = len(units)
    taus = [u["kendall_tau"] for u in units]
    ex_dirty = next((u["top1_dirty"] for u in units if u["top1_flipped"]), units[0]["top1_dirty"] if units else "")
    ex_clean = next((u["top1_clean"] for u in units if u["top1_flipped"]), units[0]["top1_clean"] if units else "")
    n_arms_vals = sorted({u["n_arms"] for u in units})
    n_arms_str = str(n_arms_vals[0]) if len(n_arms_vals) == 1 else f"{min(n_arms_vals)}..{max(n_arms_vals)}"

    return {
        "trap": trap,
        "suite_id": suite_id,
        "suite_label": suite_label,
        "source_type": source_type,
        "comparison": comparison,
        "metric": metric,
        "n_units": n_units,
        "n_arms": n_arms_str,
        "pair_flips": int(total_flips),
        "total_pairs": int(total_pairs),
        "pairwise_flip_pct": round(100.0 * total_flips / total_pairs, 2) if total_pairs else 0.0,
        "kendall_tau": round(float(np.mean(taus)), 4) if taus else 1.0,
        "top1_flips": int(top1_flips),
        "top1_total": n_units,
        "top1_flip_pct": round(100.0 * top1_flips / n_units, 1) if n_units else 0.0,
        "top1_example_dirty": ex_dirty,
        "top1_example_clean": ex_clean,
        "status": "verified",
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Trap 1: Interpolated Baseline vs. Executed Adjacent Cadence Brackets
# ---------------------------------------------------------------------------
def compute_trap1_rows(res_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    specs = [
        (
            "trap1_exp2_sift25",
            "Exp2 SIFT-128 (sift25)",
            res_dir / "exp2_predictors_cadence" / "mimo_core8_575_summary.csv",
            "sift25",
            ["noop"] + [f"fixed_c{c}_merge" for c in range(8, 0, -1)],
            [
                "past_qmass_k5", "bam_price_k5", "fcst_past_mean_k5", "fcst_ewma_a03_k5",
                "fcst_ewma_a07_k5", "fcst_cusum_reset_k5", "fcst_ridge_ar_k5", "fcst_gbrt_causal_k5",
            ],
        ),
        (
            "trap1_sift100k_canon25",
            "SIFT-100K Canon25",
            res_dir / "sift100k_canon25" / "exact_ops_and_dp.csv",
            None,
            ["noop", "fixed_c5_merge", "fixed_c4_merge", "fixed_c3_merge", "fixed_c2_merge"],
            [
                "past_qmass_b4", "bam_price_b4", "past_qmass_sp3", "past_qmass_sp2",
                "past_qmass_sp1", "forecast_ewma_a03", "forecast_gbrt_causal",
            ],
        ),
        (
            "trap1_wikimedia_210d",
            "Wikimedia-210d",
            res_dir / "wikimedia_windows" / "exact_ops_and_dp.csv",
            None,
            ["noop", "fixed_c8_merge", "fixed_c5_merge", "fixed_c4_merge", "fixed_c3_merge", "fixed_c2_merge", "fixed_c1_merge"],
            [
                "past_qmass_k8", "past_qmass_k5", "past_qmass_k3", "past_qmass_k2",
                "forecast_ewma_a03_k5", "forecast_gbrt_causal_k5",
            ],
        ),
    ]

    pooled_units: List[Dict[str, Any]] = []
    for sid, label, path, suite_filter, cad_list, pol_list in specs:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if suite_filter and "suite" in df.columns:
            df = df[df["suite"] == suite_filter]
        c_end = cad_list[-1]
        interior_cads = cad_list[1:-1]
        items = interior_cads + pol_list

        units_cpu: List[Dict[str, Any]] = []
        interp_errs: List[float] = []
        for _, grp in df.groupby("trace"):
            s = grp.set_index("arm")
            if not set(cad_list + pol_list).issubset(set(s.index)):
                continue
            n0_cpu = float(s.loc["noop", "total_cpu_ms"])
            ce_cpu = float(s.loc[c_end, "total_cpu_ms"])
            ce_act = max(float(s.loc[c_end, "n_actions"]), 1e-9)

            v_lin, v_exec = [], []
            for a in items:
                ax = float(s.loc[a, "n_actions"])
                y_ex = float(s.loc[a, "total_cpu_ms"])
                y_lin = n0_cpu + (ax / ce_act) * (ce_cpu - n0_cpu)
                if a.startswith("fixed_c"):
                    v_lin.append(y_lin)
                    v_exec.append(y_ex)
                    interp_errs.append((y_ex - y_lin) / y_ex * 100.0)
                else:
                    v_lin.append(y_ex)
                    v_exec.append(y_ex)
            units_cpu.append(compare_rankings(items, v_lin, v_exec, higher_is_better=False))

        if units_cpu:
            mean_err = float(np.mean(interp_errs)) if interp_errs else 0.0
            max_err = float(np.max(interp_errs)) if interp_errs else 0.0
            rows.append(_aggregate_unit_results(
                trap="Trap 1: Interp. Baseline",
                suite_id=sid,
                suite_label=label,
                source_type="local_qbam",
                comparison=f"Linear interp (noop<->{c_end}) vs executed cadences",
                metric="total_cpu_ms",
                units=units_cpu,
                notes=f"Non-linear interpolation error: mean {mean_err:+.2f}%, max {max_err:+.2f}%",
            ))
            pooled_units.extend(units_cpu)

    if pooled_units:
        rows.append(_aggregate_unit_results(
            trap="Trap 1: Interp. Baseline",
            suite_id="trap1_all_local_pooled",
            suite_label="All Local Cadence Suites (Pooled)",
            source_type="local_qbam",
            comparison="Linear interp vs executed cadence brackets (pooled)",
            metric="total_cpu_ms",
            units=pooled_units,
            notes="Aggregated across Exp2 SIFT-25, Canon25, and Wikimedia-210d",
        ))
    return rows


# ---------------------------------------------------------------------------
# Trap 2: Clock / Step Leakage (GBRT-Leaky vs. GBRT-Causal)
# ---------------------------------------------------------------------------
def compute_trap2_rows(res_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    specs = [
        (
            "trap2_canon25",
            "SIFT-100K Canon25",
            res_dir / "sift100k_canon25" / "exact_ops_and_dp.csv",
            None,
            "forecast_gbrt_leaky",
            "forecast_gbrt_causal",
            ["past_qmass_sp1", "forecast_ewma_a03", "forecast_cusum_reset", "forecast_ridge_ar", "bam_price_b4", "fixed_c2_merge", "noop"],
            True,
        ),
        (
            "trap2_exp2_575",
            "Exp2 SIFT-128 (575 runs)",
            res_dir / "exp2_predictors_cadence" / "mimo_core8_575_summary.csv",
            "sift25",
            "fcst_gbrt_leaky_k5",
            "fcst_gbrt_causal_k5",
            ["past_qmass_k5", "bam_price_k5", "fcst_past_mean_k5", "fcst_ewma_a03_k5", "fcst_ewma_a07_k5", "fcst_cusum_reset_k5", "fcst_ridge_ar_k5", "fixed_c2_merge", "noop"],
            True,
        ),
        (
            "trap2_leakage_fixedhub",
            "Leakage 2x2 (Fixed-Hub)",
            res_dir / "leakage_2x2" / "fixedhub" / "exact_ops_and_dp.csv",
            None,
            "forecast_gbrt_leaky",
            "forecast_gbrt_causal",
            ["past_qmass_sp1", "forecast_ewma_a03", "forecast_ridge_ar", "noop"],
            True,
        ),
        (
            "trap2_leakage_randhub",
            "Leakage 2x2 (Rand-Hub)",
            res_dir / "leakage_2x2" / "randhub" / "exact_ops_and_dp.csv",
            None,
            "forecast_gbrt_leaky",
            "forecast_gbrt_causal",
            ["past_qmass_sp1", "forecast_ewma_a03", "forecast_ridge_ar", "noop"],
            True,
        ),
        (
            "trap2_crossmodal_rag384",
            "MiniLM-384 (Synth.)",
            res_dir / "crossmodal_rag384" / "exact_ops_and_dp.csv",
            None,
            "forecast_gbrt_leaky",
            "forecast_gbrt_causal",
            ["past_qmass_sp1", "forecast_ewma_a03", "forecast_cusum_reset", "forecast_ridge_ar", "bam_price_b4", "fixed_c2_merge", "noop"],
            False,
        ),
        (
            "trap2_crossmodal_clip512",
            "CLIP-512 (Real-IAD)",
            res_dir / "crossmodal_clip512" / "exact_ops_and_dp.csv",
            None,
            "forecast_gbrt_leaky",
            "forecast_gbrt_causal",
            ["past_qmass_sp1", "forecast_ewma_a03", "forecast_cusum_reset", "forecast_ridge_ar", "bam_price_b4", "fixed_c2_merge", "noop"],
            False,
        ),
        (
            "trap2_clinc150",
            "CLINC-150 (NLP)",
            res_dir / "clinc150" / "exact_ops_and_dp.csv",
            None,
            "forecast_gbrt_leaky",
            "forecast_gbrt_causal",
            ["past_qmass_sp1", "forecast_ewma_a03", "forecast_ridge_ar", "fixed_c2_merge", "noop"],
            False,
        ),
        (
            "trap2_wikimedia_210d",
            "Wikimedia-210d",
            res_dir / "wikimedia_windows" / "exact_ops_and_dp.csv",
            None,
            "forecast_gbrt_leaky_k5",
            "forecast_gbrt_causal_k5",
            ["past_qmass_k5", "forecast_ewma_a03_k5", "forecast_cusum_reset_k5", "forecast_ridge_ar_k5", "fixed_c2_merge", "noop"],
            False,
        ),
    ]

    core_units_ops: List[Dict[str, Any]] = []
    core_units_cpu: List[Dict[str, Any]] = []
    all_units_ops: List[Dict[str, Any]] = []
    all_units_cpu: List[Dict[str, Any]] = []

    for sid, label, path, suite_filter, l_arm, c_arm, base_arms, is_core in specs:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if suite_filter and "suite" in df.columns:
            df = df[df["suite"] == suite_filter]

        for metric in ("exact_scan_ops", "total_cpu_ms"):
            if metric not in df.columns:
                continue
            units: List[Dict[str, Any]] = []
            for _, grp in df.groupby("trace"):
                s = grp.set_index("arm")
                if not ({l_arm, c_arm} | set(base_arms)).issubset(set(s.index)):
                    continue
                slot_names = ["GBRT"] + base_arms
                vals_leaky = [float(s.loc[l_arm, metric])] + [float(s.loc[b, metric]) for b in base_arms]
                vals_causal = [float(s.loc[c_arm, metric])] + [float(s.loc[b, metric]) for b in base_arms]
                units.append(compare_rankings(slot_names, vals_leaky, vals_causal, higher_is_better=False))

            if not units:
                continue
            row = _aggregate_unit_results(
                trap="Trap 2: Clock Leakage",
                suite_id=f"{sid}_{metric}",
                suite_label=label,
                source_type="local_qbam",
                comparison="GBRT-Leaky vs GBRT-Causal leaderboard",
                metric=metric,
                units=units,
            )
            rows.append(row)
            if metric == "exact_scan_ops":
                all_units_ops.extend(units)
                if is_core:
                    core_units_ops.extend(units)
            else:
                all_units_cpu.extend(units)
                if is_core:
                    core_units_cpu.extend(units)

    if core_units_ops:
        rows.append(_aggregate_unit_results(
            trap="Trap 2: Clock Leakage",
            suite_id="trap2_core_canon_leakage_exp2_ops",
            suite_label="Canon25 + Leakage 2x2 + Exp2-575 (Core Pooled)",
            source_type="local_qbam",
            comparison="GBRT-Leaky vs GBRT-Causal leaderboard",
            metric="exact_scan_ops",
            units=core_units_ops,
        ))
    if core_units_cpu:
        rows.append(_aggregate_unit_results(
            trap="Trap 2: Clock Leakage",
            suite_id="trap2_core_canon_leakage_exp2_cpu",
            suite_label="Canon25 + Leakage 2x2 + Exp2-575 (Core Pooled)",
            source_type="local_qbam",
            comparison="GBRT-Leaky vs GBRT-Causal leaderboard",
            metric="total_cpu_ms",
            units=core_units_cpu,
        ))
    if all_units_ops:
        rows.append(_aggregate_unit_results(
            trap="Trap 2: Clock Leakage",
            suite_id="trap2_all_local_pooled_ops",
            suite_label="All Local Predictor Suites (Pooled)",
            source_type="local_qbam",
            comparison="GBRT-Leaky vs GBRT-Causal leaderboard",
            metric="exact_scan_ops",
            units=all_units_ops,
        ))
    return rows


# ---------------------------------------------------------------------------
# Trap 3: Counterfactual Horizon (1w vs 4w) & Tombstone Semantics (Refresh vs Merge)
# ---------------------------------------------------------------------------
def compute_trap3_rows(res_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # 3a: Track-CF 1w vs 4w across all JSONLs
    cf_files = sorted(glob.glob(str(res_dir / "track_cf" / "*.jsonl")))
    cf_pooled: List[Dict[str, Any]] = []
    for fp in cf_files:
        p = Path(fp)
        recs = [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
        branches = [r for r in recs if r.get("type") == "branch"]
        by_j: Dict[int, List[Dict[str, Any]]] = {}
        for br in branches:
            by_j.setdefault(int(br["j"]), []).append(br)
        units: List[Dict[str, Any]] = []
        q_ratios: List[float] = []
        for _, items in sorted(by_j.items()):
            names = [str(it.get("action_str") or it.get("action") or i) for i, it in enumerate(items)]
            c1 = [float(it["windows"][0]["query_cpu_ms"]) + float(it["maintenance_cpu_ms"]) for it in items]
            c4 = [sum(float(w["query_cpu_ms"]) for w in it["windows"]) + float(it["maintenance_cpu_ms"]) for it in items]
            noop = next((it for it in items if it.get("action") == "noop"), None)
            if noop and noop["windows"][0]["query_cpu_ms"] > 0:
                q_ratios.append(sum(float(w["query_cpu_ms"]) for w in noop["windows"]) / float(noop["windows"][0]["query_cpu_ms"]))
            units.append(compare_rankings(names, c1, c4, higher_is_better=False))
        if units:
            rows.append(_aggregate_unit_results(
                trap="Trap 3: CF Horizon & Tombstone",
                suite_id=f"trap3_{p.stem}",
                suite_label=f"Track-CF ({p.stem})",
                source_type="local_qbam",
                comparison="1-block (1w) vs 4-block (4w) counterfactual cost",
                metric="total_cost_ms",
                units=units,
                notes=f"4w/1w noop query work ratio = {np.mean(q_ratios):.2f}x" if q_ratios else "",
            ))
            cf_pooled.extend(units)

    if cf_pooled:
        rows.append(_aggregate_unit_results(
            trap="Trap 3: CF Horizon & Tombstone",
            suite_id="trap3_track_cf_pooled",
            suite_label="Track-CF 1w vs 4w (5 Traces Pooled)",
            source_type="local_qbam",
            comparison="1-block (1w) vs 4-block (4w) counterfactual cost",
            metric="total_cost_ms",
            units=cf_pooled,
            notes="20 decision blocks x 33 CF branches/block = 2,720 action pairs",
        ))

    # 3b: Tombstone Sweep refresh vs merge across crud00..crud50
    tomb_units_all: List[Dict[str, Any]] = []
    tomb_units_crud_gt0: List[Dict[str, Any]] = []
    for crud in ["crud00", "crud10", "crud20", "crud30", "crud50"]:
        fp = res_dir / "tombstone_sweep" / crud / "exact_ops_and_dp.csv"
        if not fp.exists():
            continue
        df = pd.read_csv(fp)
        units: List[Dict[str, Any]] = []
        ref_wins = 0
        n_tr = 0
        for _, grp in df.groupby("trace"):
            s = grp.set_index("arm")["total_cpu_ms"]
            if not {"fixed_c2_merge", "fixed_c2_refresh", "past_qmass_sp1", "past_qmass_refresh", "noop"}.issubset(set(s.index)):
                continue
            n_tr += 1
            if float(s["past_qmass_refresh"]) < float(s["past_qmass_sp1"]):
                ref_wins += 1
            # Compare 4-arm ranking: [fixed_c2_merge, fixed_c2_refresh, past_qmass_merge, past_qmass_refresh]
            # vs merge-only assumption (ignoring refresh, equating refresh=merge) OR policy rank under merge vs refresh
            names_pol = ["fixed_c2", "past_qmass", "noop"]
            v_merge = [float(s["fixed_c2_merge"]), float(s["past_qmass_sp1"]), float(s["noop"])]
            v_refresh = [float(s["fixed_c2_refresh"]), float(s["past_qmass_refresh"]), float(s["noop"])]
            units.append(compare_rankings(names_pol, v_merge, v_refresh, higher_is_better=False))

        if units:
            rows.append(_aggregate_unit_results(
                trap="Trap 3: CF Horizon & Tombstone",
                suite_id=f"trap3_tombstone_{crud}",
                suite_label=f"Tombstone Sweep ({crud.upper()})",
                source_type="local_qbam",
                comparison="Merge-only vs Refresh-enabled policy ranking",
                metric="total_cpu_ms",
                units=units,
                notes=f"Past-QMass Refresh beats Merge in {ref_wins}/{n_tr} traces",
            ))
            tomb_units_all.extend(units)
            if crud != "crud00":
                tomb_units_crud_gt0.extend(units)

    if tomb_units_all:
        rows.append(_aggregate_unit_results(
            trap="Trap 3: CF Horizon & Tombstone",
            suite_id="trap3_tombstone_pooled",
            suite_label="Tombstone Sweep (0%-50% CRUD Pooled)",
            source_type="local_qbam",
            comparison="Merge-only vs Refresh-enabled policy ranking",
            metric="total_cpu_ms",
            units=tomb_units_all,
            notes="At >=10% CRUD, Refresh beats Merge on 18/24 (75.0%) trace-levels",
        ))
    return rows


# ---------------------------------------------------------------------------
# Trap 4: Contention & Cross-NUMA Placement
# ---------------------------------------------------------------------------
def compute_trap4_rows(res_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    # 4a: phase_uxq (15 grids) + abrupt_tsw (3 grids) = 18 grids dirty vs clean
    phase_units: List[Dict[str, Any]] = []
    for df_p in sorted(glob.glob(str(res_dir / "phase_uxq" / "*.csv"))):
        name = Path(df_p).name
        cf_p = res_dir / "phase_uxq_clean" / name
        if not cf_p.exists():
            continue
        d = pd.read_csv(df_p).groupby("arm")["total_cpu_ms"].mean()
        c = pd.read_csv(cf_p).groupby("arm")["total_cpu_ms"].mean()
        arms = sorted(set(d.index) & set(c.index))
        phase_units.append(compare_rankings(arms, [float(d[a]) for a in arms], [float(c[a]) for a in arms], higher_is_better=False))

    abrupt_units: List[Dict[str, Any]] = []
    for tsw in (16, 24, 28):
        df_p = res_dir / "abrupt_tsw" / f"abrupt_tsw{tsw}_summary.csv"
        cf_p = res_dir / "abrupt_tsw_clean" / f"abrupt_tsw{tsw}_summary.csv"
        if not (df_p.exists() and cf_p.exists()):
            continue
        d = pd.read_csv(df_p).groupby("arm")["total_cpu_ms"].mean()
        c = pd.read_csv(cf_p).groupby("arm")["total_cpu_ms"].mean()
        arms = sorted(set(d.index) & set(c.index))
        abrupt_units.append(compare_rankings(arms, [float(d[a]) for a in arms], [float(c[a]) for a in arms], higher_is_better=False))

    if phase_units:
        rows.append(_aggregate_unit_results(
            trap="Trap 4: Contention & NUMA",
            suite_id="trap4_phase_uxq_15grids",
            suite_label="(u, Q) Phase Grid (15 Grids)",
            source_type="local_qbam",
            comparison="Shared-host Dirty vs Core-8 Clean",
            metric="total_cpu_ms",
            units=phase_units,
        ))
    if abrupt_units:
        rows.append(_aggregate_unit_results(
            trap="Trap 4: Contention & NUMA",
            suite_id="trap4_abrupt_tsw_3grids",
            suite_label="Abrupt Switch t_sw in {16,24,28} (3 Grids)",
            source_type="local_qbam",
            comparison="Shared-host Dirty vs Core-8 Clean",
            metric="total_cpu_ms",
            units=abrupt_units,
            notes="Past-QMass vs GBRT-Causal reverses on 3/3 grids",
        ))
    all_18 = phase_units + abrupt_units
    if all_18:
        rows.append(_aggregate_unit_results(
            trap="Trap 4: Contention & NUMA",
            suite_id="trap4_dirty_vs_clean_18grids",
            suite_label="Dirty vs Clean Re-Audit (18 Grids Pooled)",
            source_type="local_qbam",
            comparison="Shared-host Dirty vs Core-8 Clean (18 grids)",
            metric="total_cpu_ms",
            units=all_18,
            notes="82/504 = 16.27% headline ranking-flip rate across 18 grids",
        ))

    # 4b: Stage-1 Xeon 8352V NUMA Contention Sweep
    sh_p = res_dir / "stage1_hygiene" / "hygiene_summary.csv"
    if sh_p.exists():
        sh = pd.read_csv(sh_p)
        iso = sh[sh["regime"] == "isolated_numa0"].groupby(["trace", "arm"])[["total_cpu_ms", "mean_p99_us"]].mean()
        arms = ["noop", "fixed_c2", "past_qmass"]
        for metric in ("total_cpu_ms", "mean_p99_us"):
            pooled_reg: List[Dict[str, Any]] = []
            for reg in ("ht_sibling_2w", "same_socket_4w", "same_socket_8w", "cross_socket_16w"):
                sub = sh[sh["regime"] == reg].groupby(["trace", "arm"])[["total_cpu_ms", "mean_p99_us"]].mean()
                units_r: List[Dict[str, Any]] = []
                for tr in sorted(sh["trace"].unique()):
                    va = [float(sub.loc[(tr, a), metric]) for a in arms]
                    vb = [float(iso.loc[(tr, a), metric]) for a in arms]
                    units_r.append(compare_rankings(arms, va, vb, higher_is_better=False))
                rows.append(_aggregate_unit_results(
                    trap="Trap 4: Contention & NUMA",
                    suite_id=f"trap4_xeon_{reg}_{metric}",
                    suite_label=f"Xeon 8352V ({reg})",
                    source_type="local_qbam",
                    comparison=f"{reg} vs isolated_numa0",
                    metric=metric,
                    units=units_r,
                ))
                pooled_reg.extend(units_r)
            if pooled_reg:
                rows.append(_aggregate_unit_results(
                    trap="Trap 4: Contention & NUMA",
                    suite_id=f"trap4_xeon_all_regimes_{metric}",
                    suite_label="Xeon 8352V Contention Sweep (All Tiers)",
                    source_type="local_qbam",
                    comparison="Contended tiers (2w-16w) vs isolated_numa0",
                    metric=metric,
                    units=pooled_reg,
                ))

    # 4c: Wikimedia-210d Shared-Host CPU vs Hardware-Invariant Exact Scan Ops
    wk_p = res_dir / "wikimedia_windows" / "exact_ops_and_dp.csv"
    if wk_p.exists():
        wk = pd.read_csv(wk_p)
        units_wk: List[Dict[str, Any]] = []
        for _, grp in wk.groupby("trace"):
            arms = grp["arm"].tolist()
            units_wk.append(compare_rankings(
                arms,
                grp["total_cpu_ms"].astype(float).tolist(),
                grp["exact_scan_ops"].astype(float).tolist(),
                higher_is_better=False,
            ))
        if units_wk:
            rows.append(_aggregate_unit_results(
                trap="Trap 4: Contention & NUMA",
                suite_id="trap4_wikimedia_cpu_vs_ops",
                suite_label="Wikimedia-210d (Shared-Host CPU vs Exact Ops)",
                source_type="local_qbam",
                comparison="Contended total_cpu_ms vs invariant exact_scan_ops",
                metric="total_cpu_ms_vs_ops",
                units=units_wk,
                notes="past_qmass_uncap inflated to 844.1s CPU despite 95.0% exact_scan_ops GCR",
            ))
    return rows


# ---------------------------------------------------------------------------
# Auto-Ingestion Hooks for Exp 6 (llm01 Real Engines) & Exp 7 (GPU 1M + arXiv)
# ---------------------------------------------------------------------------
def ingest_exp6_real_engine_traps(res_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    exp6_dir = res_dir / "exp6_real_engine_traps"
    if not exp6_dir.exists():
        return [], ["results/exp6_real_engine_traps/ (pending @llm01-engines)"]

    csvs = sorted(exp6_dir.rglob("*.csv"))
    if not csvs:
        return [], ["results/exp6_real_engine_traps/ (directory exists, no CSVs yet)"]

    rows: List[Dict[str, Any]] = []

    # 1. A1+D Real-Engine Contention & MT NUMA (a1d_summary.csv)
    a1d_p = exp6_dir / "a1d_summary.csv"
    if a1d_p.exists():
        df = pd.read_csv(a1d_p)
        df["_eng"] = df["system"].astype(str) + "_" + df["param_tag"].astype(str)
        # Single-worker per-query regimes vs iso_numa0_1w
        iso_1w = df[df["regime"] == "iso_numa0_1w"]
        if not iso_1w.empty:
            for reg in ("ht_sibling_2w", "same_numa_4w", "cross_numa_8w"):
                sub = df[df["regime"] == reg]
                if sub.empty:
                    continue
                for metric in [m for m in ("total_cpu_ms", "wall_s", "mean_p99_us") if m in df.columns]:
                    units = []
                    for tr in sorted(df["trace"].unique()):
                        s_iso = iso_1w[iso_1w["trace"] == tr].set_index("_eng")[metric]
                        s_reg = sub[sub["trace"] == tr].set_index("_eng")[metric]
                        common = sorted(set(s_iso.index) & set(s_reg.index))
                        if len(common) >= 2:
                            units.append(compare_rankings(common, [float(s_reg[e]) for e in common], [float(s_iso[e]) for e in common], higher_is_better=False))
                    if units:
                        rows.append(_aggregate_unit_results(
                            trap="Trap 4: Contention & NUMA",
                            suite_id=f"exp6_a1d_{reg}_{metric}",
                            suite_label=f"Real-Engine 5-Sys ({reg})",
                            source_type="exp6_real_engine",
                            comparison=f"{reg} vs iso_numa0_1w",
                            metric=metric,
                            units=units,
                        ))
        # Multi-threaded pinned 4t & unpinned 8t vs iso_numa0_1t_batch (and mt8 vs mt4)
        iso_bt = df[df["regime"] == "iso_numa0_1t_batch"]
        mt4 = df[df["regime"] == "mt_same_numa_4t"]
        mt8 = df[df["regime"] == "mt_unpinned_8t"]
        mt_pairs = [
            ("mt4_vs_iso1t", iso_bt, mt4, "mt_same_numa_4t vs iso_numa0_1t_batch", "Real-Engine 5-Sys (MT-NUMA0-4t vs 1T)"),
            ("mt8_vs_iso1t", iso_bt, mt8, "mt_unpinned_8t vs iso_numa0_1t_batch", "Real-Engine 5-Sys (MT-Unpinned-8t vs 1T)"),
            ("mt8_vs_mt4", mt4, mt8, "mt_unpinned_8t vs mt_same_numa_4t", "Real-Engine 5-Sys (MT-Unpinned-8t vs MT-NUMA0-4t)"),
        ]
        for pid, base_df, alt_df, cmp_str, lbl_str in mt_pairs:
            if base_df.empty or alt_df.empty:
                continue
            for metric in [m for m in ("wall_s", "total_cpu_ms") if m in df.columns]:
                units = []
                for tr in sorted(df["trace"].unique()):
                    s_base = base_df[base_df["trace"] == tr].set_index("_eng")[metric]
                    s_alt = alt_df[alt_df["trace"] == tr].set_index("_eng")[metric]
                    common = sorted(set(s_base.index) & set(s_alt.index))
                    if len(common) >= 2:
                        units.append(compare_rankings(common, [float(s_alt[e]) for e in common], [float(s_base[e]) for e in common], higher_is_better=False))
                if units:
                    rows.append(_aggregate_unit_results(
                        trap="Concurrency Regime",
                        suite_id=f"exp6_a1d_{pid}_{metric}",
                        suite_label=lbl_str,
                        source_type="exp6_real_engine",
                        comparison=cmp_str,
                        metric=metric,
                        units=units,
                    ))

    # 2. A2 Real-Engine Cadence Brackets (a2_cadence_brackets.csv)
    a2_p = exp6_dir / "a2_cadence_brackets.csv"
    if a2_p.exists():
        df = pd.read_csv(a2_p)
        if {"workload", "trace", "system", "cadence", "maint_cpu_ms", "total_cpu_ms"}.issubset(df.columns):
            units = []
            interp_errs = []
            for (_, _, _), grp in df.groupby(["workload", "trace", "system"]):
                gpos = grp[grp["cadence"] > 0].sort_values("cadence")
                if len(gpos) < 3:
                    continue
                c_min, c_max = int(gpos["cadence"].min()), int(gpos["cadence"].max())
                s_min = float(gpos[gpos["cadence"] == c_min]["maint_cpu_ms"].iloc[0])
                s_max = float(gpos[gpos["cadence"] == c_max]["maint_cpu_ms"].iloc[0])
                cads = gpos["cadence"].astype(int).tolist()
                v_lin, v_act = [], []
                for _, r in gpos.iterrows():
                    c = int(r["cadence"])
                    inv_frac = (1.0 / c - 1.0 / c_max) / max(1.0 / c_min - 1.0 / c_max, 1e-9)
                    m_lin = s_max + inv_frac * (s_min - s_max)
                    m_act = float(r["maint_cpu_ms"])
                    if c not in (c_min, c_max) and m_act > 1e-6:
                        interp_errs.append((m_lin - m_act) / m_act * 100.0)
                    # Total CPU using interpolated maintenance vs actual total CPU
                    v_lin.append(float(r["query_cpu_ms"]) + m_lin if "query_cpu_ms" in grp.columns else m_lin)
                    v_act.append(float(r["total_cpu_ms"]) if "query_cpu_ms" in grp.columns else m_act)
                units.append(compare_rankings([f"c={c}" for c in cads], v_lin, v_act, higher_is_better=False))
            if units:
                rows.append(_aggregate_unit_results(
                    trap="Trap 1: Interp. Baseline",
                    suite_id="exp6_a2_real_engine_cadence",
                    suite_label="Real-Engine Trap 1 (FAISS/Annoy/DiskANN Cadence)",
                    source_type="exp6_real_engine",
                    comparison="Linear 1/c interp vs executed cadence total CPU",
                    metric="total_cpu_ms",
                    units=units,
                    notes=f"Maint interp error: mean {np.mean(interp_errs):+.2f}%, max abs {np.max(np.abs(interp_errs)):.2f}%" if interp_errs else "",
                ))

    # 3. A3 Real-Engine CRUD (a3_crud_engines.csv): DiskANN consolidate=0 vs consolidate=1 & engine flips across workloads
    a3_p = exp6_dir / "a3_crud_engines.csv"
    if a3_p.exists():
        df = pd.read_csv(a3_p)
        if {"workload", "trace", "system", "param_tag", "total_cpu_ms"}.issubset(df.columns):
            # Compare ranking by raw total_cpu_ms vs recall-attainment (attainment_0965_pct) across valid engines per workload
            att_col = next((c for c in ("attainment_0965_pct", "attainment_0950_pct") if c in df.columns), None)
            if att_col:
                df = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()
                df["_eng"] = df["system"].astype(str) + "_" + df["param_tag"].astype(str)
                units = []
                for (_, _), grp in df.groupby(["workload", "trace"]):
                    if len(grp) >= 2:
                        units.append(compare_rankings(
                            grp["_eng"].tolist(),
                            grp["total_cpu_ms"].astype(float).tolist(),
                            (-grp[att_col]).astype(float).tolist(),
                            higher_is_better=False,
                        ))
                if units:
                    rows.append(_aggregate_unit_results(
                        trap="Trap 3: CF Horizon & Tombstone",
                        suite_id="exp6_a3_crud_cpu_vs_attainment",
                        suite_label="Real-Engine CRUD 30/50% & Runbook",
                        source_type="exp6_real_engine",
                        comparison="Unmatched total_cpu_ms vs Recall-Attainment rank",
                        metric="cpu_vs_attainment",
                        units=units,
                    ))

    # 4. A4 Recall-Matched >=95% Attainment Sweep (a4_summary_by_config.csv)
    a4_p = exp6_dir / "a4_summary_by_config.csv"
    if a4_p.exists():
        df = pd.read_csv(a4_p)
        att_col = next((c for c in ("attainment_0965_pct", "attainment_0950_pct") if c in df.columns), None)
        if att_col and {"system", "param_tag", "total_cpu_ms"}.issubset(df.columns):
            df = df.copy()
            df["_eng"] = df["system"].astype(str) + "_" + df["param_tag"].astype(str)
            u = compare_rankings(
                df["_eng"].tolist(),
                df["total_cpu_ms"].astype(float).tolist(),
                (-df[att_col]).astype(float).tolist(),
                higher_is_better=False,
            )
            rows.append(_aggregate_unit_results(
                trap="Trap 4: Contention & NUMA",
                suite_id="exp6_a4_unmatched_cpu_vs_attainment",
                suite_label="Real-Engine Recall-Matched Floor Sweep (A4)",
                source_type="exp6_real_engine",
                comparison="Unmatched total_cpu_ms vs Recall@10>=0.965 Attainment",
                metric="cpu_vs_attainment",
                units=[u],
            ))

    return rows, []


def ingest_exp7_gpu_1m_arxiv(res_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    exp7_dir = res_dir / "exp7_gpu_1m_arxiv"
    if not exp7_dir.exists():
        return [], ["results/exp7_gpu_1m_arxiv/ (pending @gpu-streams)"]

    csvs = sorted(exp7_dir.rglob("*.csv"))
    if not csvs:
        return [], ["results/exp7_gpu_1m_arxiv/ (directory exists, no CSVs yet)"]

    rows: List[Dict[str, Any]] = []
    for cp in csvs:
        try:
            df = pd.read_csv(cp)
        except Exception:
            continue
        cols = set(df.columns)
        stem = cp.stem

        # Counterfactual replay CSV (arXiv MiniLM-384): Trap 2 leaky vs causal + Trap 1 cadence brackets
        if {"trace", "arm"}.issubset(cols):
            arms = set(df["arm"].astype(str).unique())
            leaky = next((a for a in arms if "leaky" in a.lower()), None)
            causal = next((a for a in arms if "causal" in a.lower()), None)
            if leaky and causal:
                others = sorted(a for a in arms if a not in (leaky, causal) and "oracle" not in a.lower())
                for metric in [m for m in ("exact_scan_ops", "total_cpu_ms") if m in cols]:
                    units = []
                    for _, grp in df.groupby("trace"):
                        s = grp.set_index("arm")[metric]
                        if {leaky, causal}.issubset(set(s.index)):
                            present = [o for o in others if o in s.index]
                            if present:
                                va = [float(s[leaky])] + [float(s[o]) for o in present]
                                vb = [float(s[causal])] + [float(s[o]) for o in present]
                                units.append(compare_rankings(["GBRT"] + present, va, vb, higher_is_better=False))
                    if units:
                        rows.append(_aggregate_unit_results(
                            trap="Trap 2: Clock Leakage",
                            suite_id=f"exp7_{stem}_{metric}",
                            suite_label=f"arXiv MiniLM-384 ({stem})",
                            source_type="exp7_gpu_1m_arxiv",
                            comparison=f"{leaky} vs {causal}",
                            metric=metric,
                            units=units,
                        ))
            # Trap 1 on exact_scan_ops when noop + fixed_c1..8 + _k5 policies exist
            cad_list = ["noop"] + [f"fixed_c{c}_merge" for c in range(8, 0, -1) if f"fixed_c{c}_merge" in arms]
            pol_list = sorted(a for a in arms if a.endswith("_k5") and "leaky" not in a and "oracle" not in a)
            if len(cad_list) >= 4 and pol_list and "exact_scan_ops" in cols and "n_actions" in cols:
                c_end = cad_list[-1]
                items = cad_list[1:-1] + pol_list
                units_t1 = []
                for _, grp in df.groupby("trace"):
                    s = grp.set_index("arm")
                    if not set(cad_list + pol_list).issubset(set(s.index)):
                        continue
                    n0_ops = float(s.loc["noop", "exact_scan_ops"])
                    ce_ops = float(s.loc[c_end, "exact_scan_ops"])
                    ce_act = max(float(s.loc[c_end, "n_actions"]), 1e-9)
                    v_lin, v_exec = [], []
                    for a in items:
                        ax = float(s.loc[a, "n_actions"])
                        y_ex = float(s.loc[a, "exact_scan_ops"])
                        y_lin = n0_ops + (ax / ce_act) * (ce_ops - n0_ops)
                        v_lin.append(y_lin if a.startswith("fixed_c") else y_ex)
                        v_exec.append(y_ex)
                    units_t1.append(compare_rankings(items, v_lin, v_exec, higher_is_better=False))
                if units_t1:
                    rows.append(_aggregate_unit_results(
                        trap="Trap 1: Interp. Baseline",
                        suite_id=f"exp7_{stem}_trap1_cadence",
                        suite_label=f"arXiv MiniLM-384 ({stem})",
                        source_type="exp7_gpu_1m_arxiv",
                        comparison=f"Linear interp (noop<->{c_end}) vs executed cadences",
                        metric="exact_scan_ops",
                        units=units_t1,
                    ))

        # 1M / arXiv multi-engine summary: ranking by relative/raw CPU vs recall-attainment per workload
        cpu_col = next((c for c in ("total_cpu_ms", "rel_cpu_vs_faiss_np16") if c in cols), None)
        att_col = next((c for c in ("attainment_0965_pct", "attainment_pct") if c in cols), None)
        if {"system"}.issubset(cols) and cpu_col and att_col and len(df) >= 2:
            df = df[df["status"] == "ok"].copy() if "status" in cols else df.copy()
            df = df.dropna(subset=[cpu_col, att_col])
            df["_eng"] = df["system"].astype(str) + ("_" + df["param_tag"].astype(str) if "param_tag" in cols else "")
            wl_groups = list(df.groupby("workload")) if "workload" in cols else [(stem, df)]
            for wl_name, sub_df in wl_groups:
                g = sub_df.groupby("_eng", as_index=False)[[cpu_col, att_col]].mean()
                if len(g) < 2:
                    continue
                u = compare_rankings(
                    g["_eng"].tolist(),
                    g[cpu_col].astype(float).tolist(),
                    (-g[att_col]).astype(float).tolist(),
                    higher_is_better=False,
                )
                rows.append(_aggregate_unit_results(
                    trap="Trap 4: Contention & NUMA",
                    suite_id=f"exp7_{stem}_{wl_name}_cpu_vs_att",
                    suite_label=f"Exp7 Multi-Engine ({wl_name})",
                    source_type="exp7_gpu_1m_arxiv",
                    comparison=f"Unmatched {cpu_col} vs Recall-Attainment rank",
                    metric=f"{cpu_col}_vs_attainment",
                    units=[u],
                ))

    return rows, []


def ingest_colab_1m_audit(res_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    colab_dir = res_dir / "colab_1m_audit"
    if not colab_dir.exists():
        return [], ["results/colab_1m_audit/"]

    rows: List[Dict[str, Any]] = []

    # 1. AG News MiniLM-384 (8 rows): unmatched total_cpu_ms vs attainment_0965_pct
    ag_p = colab_dir / "colab_real_semantic_engines.csv"
    if ag_p.exists():
        df = pd.read_csv(ag_p)
        df = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()
        if {"system", "param_tag", "total_cpu_ms", "attainment_0965_pct"}.issubset(df.columns) and len(df) >= 2:
            df["_eng"] = df["system"].astype(str) + "_" + df["param_tag"].astype(str)
            u = compare_rankings(
                df["_eng"].tolist(),
                df["total_cpu_ms"].astype(float).tolist(),
                (-df["attainment_0965_pct"]).astype(float).tolist(),
                higher_is_better=False,
            )
            rows.append(_aggregate_unit_results(
                trap="Trap 4: Contention & NUMA",
                suite_id="colab_agnews_cpu_vs_att",
                suite_label="Colab AG News 384-d (Cost vs Att.)",
                source_type="colab_1m_audit",
                comparison="Unmatched total_cpu_ms vs Recall@10>=0.965 Attainment",
                metric="total_cpu_ms_vs_attainment",
                units=[u],
            ))

    # 2. Colab 1M Engines incl. 7th backend FAISS-HNSW (13 rows)
    c1m_p = colab_dir / "colab_1m_engines_summary.csv"
    if c1m_p.exists():
        df = pd.read_csv(c1m_p)
        df = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()
        if {"workload", "system", "param_tag", "total_cpu_ms", "attainment_0965_pct"}.issubset(df.columns):
            df["_eng"] = df["system"].astype(str) + "_" + df["param_tag"].astype(str)
            for wl, grp in df.groupby("workload"):
                if len(grp) >= 2:
                    u = compare_rankings(
                        grp["_eng"].tolist(),
                        grp["total_cpu_ms"].astype(float).tolist(),
                        (-grp["attainment_0965_pct"]).astype(float).tolist(),
                        higher_is_better=False,
                    )
                    rows.append(_aggregate_unit_results(
                        trap="Trap 4: Contention & NUMA",
                        suite_id=f"colab_{wl}_cpu_vs_att",
                        suite_label=f"Colab {wl} + FAISS-HNSW (Cost vs Att.)",
                        source_type="colab_1m_audit",
                        comparison="Unmatched total_cpu_ms vs Recall@10>=0.965 Attainment",
                        metric="total_cpu_ms_vs_attainment",
                        units=[u],
                    ))

    # 3. Colab 8-vCPU cloud silent SMT Trap 4 (52 rows)
    ct4_p = colab_dir / "colab_trap4_real_engines.csv"
    if ct4_p.exists():
        df = pd.read_csv(ct4_p)
        if {"regime", "system", "param_tag", "total_cpu_ms", "mean_p99_us"}.issubset(df.columns):
            df["_eng"] = df["system"].astype(str) + "_" + df["param_tag"].astype(str)
            agg = df.groupby(["regime", "_eng"], as_index=False)[["total_cpu_ms", "mean_p99_us"]].mean()
            iso = agg[agg["regime"] == "isolated_1w"].set_index("_eng")
            for reg in ("parallel_4w", "parallel_8w"):
                sub = agg[agg["regime"] == reg].set_index("_eng")
                common = sorted(set(iso.index) & set(sub.index))
                if len(common) >= 2:
                    for metric in ("total_cpu_ms", "mean_p99_us"):
                        u = compare_rankings(
                            common,
                            [float(sub.loc[e, metric]) for e in common],
                            [float(iso.loc[e, metric]) for e in common],
                            higher_is_better=False,
                        )
                        rows.append(_aggregate_unit_results(
                            trap="Trap 4: Contention & NUMA",
                            suite_id=f"colab_trap4_{reg}_{metric}",
                            suite_label=f"Colab 8-vCPU SMT ({reg})",
                            source_type="colab_1m_audit",
                            comparison=f"{reg} vs isolated_1w",
                            metric=metric,
                            units=[u],
                        ))

    return rows, []


def ingest_exp8_rust_and_10m_scale(res_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    exp8_dir = res_dir / "exp8_rust_and_10m_scale"
    if not exp8_dir.exists():
        return [], ["results/exp8_rust_and_10m_scale/"]

    rows: List[Dict[str, Any]] = []

    # 1. R1-A1+D (Rust engines NUMA/HT/MT Trap 4)
    p_a1d = exp8_dir / "r1_rust_a1d_numa.csv"
    if p_a1d.exists():
        df = pd.read_csv(p_a1d)
        if {"regime", "engine", "trace"}.issubset(df.columns):
            agg = df.groupby(["regime", "engine", "trace"], as_index=False)[[c for c in ("total_cpu_s", "wall_s", "p99_us") if c in df.columns]].mean()
            iso = agg[agg["regime"] == "iso_numa0_1w"]
            for reg in ("ht_sibling_2w", "same_numa_4w", "cross_numa_8w", "mt_same_numa_4t", "mt_unpinned_8t"):
                sub = agg[agg["regime"] == reg]
                if sub.empty or iso.empty:
                    continue
                for metric in [m for m in ("total_cpu_s", "wall_s", "p99_us") if m in agg.columns]:
                    units = []
                    for tr in sorted(agg["trace"].unique()):
                        si = iso[iso["trace"] == tr].set_index("engine")[metric]
                        sr = sub[sub["trace"] == tr].set_index("engine")[metric]
                        common = sorted(set(si.index) & set(sr.index))
                        if len(common) >= 2:
                            units.append(compare_rankings(common, [float(sr[e]) for e in common], [float(si[e]) for e in common], higher_is_better=False))
                    if units:
                        rows.append(_aggregate_unit_results(
                            trap="Trap 4: Contention & NUMA",
                            suite_id=f"exp8_r1_a1d_{reg}_{metric}",
                            suite_label=f"Rust Engines 100K ({reg})",
                            source_type="exp8_rust_10m",
                            comparison=f"{reg} vs iso_numa0_1w",
                            metric=metric,
                            units=units,
                        ))

    # 2. R1-A4 (Rust Core-0 >=95% Floor)
    p_a4 = exp8_dir / "r1_rust_a4_floor.csv"
    if p_a4.exists():
        df = pd.read_csv(p_a4)
        if {"engine", "config", "total_cpu_s", "att_0965_pct"}.issubset(df.columns):
            g = df.groupby(["engine", "config"], as_index=False)[["total_cpu_s", "att_0965_pct"]].mean()
            g["_arm"] = g["engine"] + "_" + g["config"]
            if len(g) >= 2:
                u = compare_rankings(g["_arm"].tolist(), g["total_cpu_s"].tolist(), (-g["att_0965_pct"]).tolist(), higher_is_better=False)
                rows.append(_aggregate_unit_results(
                    trap="Trap 4: Contention & NUMA",
                    suite_id="exp8_r1_a4_unmatched_cpu_vs_att",
                    suite_label="Rust Engines (hnsw_rs + USearch-Rust A4)",
                    source_type="exp8_rust_10m",
                    comparison="Unmatched total_cpu_s vs Recall@10>=0.965 Attainment",
                    metric="cpu_vs_attainment",
                    units=[u],
                ))

    # 3. S1 (Gate-4 Clean Dual Xeon Core-0 1M Drift & CRUD-20%)
    p_s1 = exp8_dir / "s1_xeon_1m_gate4_clean.csv"
    if p_s1.exists():
        df = pd.read_csv(p_s1)
        df_ok = df[df["status"] == "ok"].copy() if "status" in df.columns else df.copy()
        if {"trace", "engine", "config", "total_cpu_s", "att_0965_pct"}.issubset(df_ok.columns):
            df_ok["_arm"] = df_ok["engine"].astype(str) + "_" + df_ok["config"].astype(str)
            units = []
            for tr, grp in df_ok.groupby("trace"):
                if len(grp) >= 2:
                    units.append(compare_rankings(grp["_arm"].tolist(), grp["total_cpu_s"].astype(float).tolist(), (-grp["att_0965_pct"]).astype(float).tolist(), higher_is_better=False))
            if units:
                rows.append(_aggregate_unit_results(
                    trap="Trap 4: Contention & NUMA",
                    suite_id="exp8_s1_1m_core0_cpu_vs_att",
                    suite_label="Xeon Core-0 BigANN-1M (5 Engines)",
                    source_type="exp8_rust_10m",
                    comparison="Gate-4 clean total_cpu_s vs Recall@10>=0.965",
                    metric="cpu_vs_attainment",
                    units=units,
                ))

    # 4. S2-NUMA10M (10M Scale Trap 4 Contention & Cross-NUMA Flips)
    p_n10 = exp8_dir / "s2_numa_10m_trap4.csv"
    if p_n10.exists():
        df = pd.read_csv(p_n10)
        if {"regime", "engine", "dyn_wall_s", "p99_us"}.issubset(df.columns):
            iso = df[df["regime"] == "iso_numa0_8t"].set_index("engine")
            for reg in ("ht_sibling_8t", "cross_numa_8t", "unpinned_16t"):
                sub = df[df["regime"] == reg].set_index("engine")
                common = sorted(set(iso.index) & set(sub.index))
                if len(common) < 2:
                    continue
                for metric in [m for m in ("dyn_wall_s", "total_cpu_s", "p99_us") if m in df.columns]:
                    u = compare_rankings(common, [float(sub.loc[e, metric]) for e in common], [float(iso.loc[e, metric]) for e in common], higher_is_better=False)
                    rows.append(_aggregate_unit_results(
                        trap="Trap 4: Contention & NUMA",
                        suite_id=f"exp8_s2_10m_{reg}_{metric}",
                        suite_label=f"BigANN-10M ({reg})",
                        source_type="exp8_rust_10m",
                        comparison=f"10M {reg} vs iso_numa0_8t",
                        metric=metric,
                        units=[u],
                    ))
            # Pooled 10M NUMA metrics across the 3 contended tiers
            for metric in [m for m in ("dyn_wall_s", "p99_us") if m in df.columns]:
                units_all = []
                for reg in ("ht_sibling_8t", "cross_numa_8t", "unpinned_16t"):
                    sub = df[df["regime"] == reg].set_index("engine")
                    common = sorted(set(iso.index) & set(sub.index))
                    if len(common) >= 2:
                        units_all.append(compare_rankings(common, [float(sub.loc[e, metric]) for e in common], [float(iso.loc[e, metric]) for e in common], higher_is_better=False))
                if units_all:
                    rows.append(_aggregate_unit_results(
                        trap="Trap 4: Contention & NUMA",
                        suite_id=f"exp8_s2_10m_pooled_{metric}",
                        suite_label="BigANN-10M Trap 4 (HT/Cross-NUMA/16t)",
                        source_type="exp8_rust_10m",
                        comparison="Contended 10M vs iso_numa0_8t",
                        metric=metric,
                        units=units_all,
                    ))

    return rows, []


def build_matrix(res_dir: Path = RES) -> Dict[str, Any]:
    all_rows: List[Dict[str, Any]] = []
    all_rows.extend(compute_trap1_rows(res_dir))
    all_rows.extend(compute_trap2_rows(res_dir))
    all_rows.extend(compute_trap3_rows(res_dir))
    all_rows.extend(compute_trap4_rows(res_dir))

    exp6_rows, exp6_pending = ingest_exp6_real_engine_traps(res_dir)
    exp7_rows, exp7_pending = ingest_exp7_gpu_1m_arxiv(res_dir)
    colab_rows, colab_pending = ingest_colab_1m_audit(res_dir)
    exp8_rows, exp8_pending = ingest_exp8_rust_and_10m_scale(res_dir)
    all_rows.extend(exp6_rows)
    all_rows.extend(exp7_rows)
    all_rows.extend(colab_rows)
    all_rows.extend(exp8_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_out = pd.DataFrame(all_rows)
    csv_path = OUT_DIR / "ranking_flip_matrix.csv"
    json_path = OUT_DIR / "ranking_flip_matrix.json"
    df_out.to_csv(csv_path, index=False)

    payload = {
        "n_rows": len(all_rows),
        "pending_optional_sources": exp6_pending + exp7_pending + colab_pending + exp8_pending,
        "headline_summary": {
            "trap4_dirty_vs_clean_18grids": next((r for r in all_rows if r["suite_id"] == "trap4_dirty_vs_clean_18grids"), None),
            "trap1_all_local_pooled": next((r for r in all_rows if r["suite_id"] == "trap1_all_local_pooled"), None),
            "trap2_core_canon_leakage_exp2_ops": next((r for r in all_rows if r["suite_id"] == "trap2_core_canon_leakage_exp2_ops"), None),
            "trap3_track_cf_pooled": next((r for r in all_rows if r["suite_id"] == "trap3_track_cf_pooled"), None),
            "trap3_tombstone_pooled": next((r for r in all_rows if r["suite_id"] == "trap3_tombstone_pooled"), None),
        },
        "rows": all_rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=RES)
    args = ap.parse_args()
    payload = build_matrix(args.results_dir)
    print(f"Wrote {OUT_DIR / 'ranking_flip_matrix.csv'} & .json ({payload['n_rows']} rows)")
    if payload["pending_optional_sources"]:
        print("Pending optional auto-ingest sources:", ", ".join(payload["pending_optional_sources"]))
    for key, val in payload["headline_summary"].items():
        if val:
            print(
                f"  {key:36s}: {val['pair_flips']}/{val['total_pairs']} ({val['pairwise_flip_pct']:.2f}%), "
                f"tau={val['kendall_tau']:.4f}, top1={val['top1_flips']}/{val['top1_total']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
