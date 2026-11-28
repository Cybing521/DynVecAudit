#!/usr/bin/env python3
"""R3-Exp 2: Budget pacing on real-timestamp streams (Wikimedia-210d & CLINC-150)
and rebuild-cost (lambda / lambda_0) phase boundary sweep.

Outputs:
  results/exp11/r3_exp2_real_streams_pacing.csv
  results/exp11/r3_exp2_cost_phase.csv
  results/exp11/r3_exp2_summary.json
"""
from __future__ import annotations

import json
import math
import zlib
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import run_exp11a_pacing_generality as e11a
import run_r3_exp1_orthogonal_decomp as r3e1

K = 4
LAMBDA_0 = r3e1.LAMBDA_0
QD_DIR = Path("results/exp11/real_stream_qd")
OUT_DIR = Path("results/exp11")


def eval_with_modes(
    Q: np.ndarray, D: np.ndarray, B: int, policy: str, mode: str, lam: float = 0.0, base_n: float | np.ndarray = 0.0
) -> Tuple[float, float]:
    if mode != "uniform_shift":
        return r3e1.eval_policy_cost(Q, D, B, policy, mode, lam=lam, base_n=base_n)

    T, R = Q.shape
    J = T // K
    d_reset = np.zeros(R, dtype=np.float64)
    ewma_q = np.zeros(R, dtype=np.float64)
    spent = 0
    scan_tot = 0.0
    reb_tot = 0.0
    prev_hot = -1
    for j in range(J):
        if j >= 1:
            q_blk = Q[(j - 1) * K : j * K].sum(axis=0)
            hot = int(np.argmax(q_blk))
            shifted = (prev_hot >= 0) and (hot != prev_hot)
            prev_hot = hot
            cap = min(B, math.floor(B * j / max(J - 1, 1)) + (1 if shifted else 0))
            if spent < min(B, cap):
                cur_d = np.maximum(0.0, D[j * K - 1] - d_reset)
                if policy == "past_qmass":
                    score = q_blk * cur_d
                elif policy == "ewma_a03":
                    score = ewma_q * cur_d
                elif policy == "oracle_1blk":
                    score = Q[j * K : min((j + 1) * K, T)].sum(axis=0) * cur_d
                else:
                    raise ValueError(policy)
                sel = int(np.argmax(score))
                if cur_d[sel] > 0:
                    d_reset[sel] = D[j * K - 1, sel]
                    bn_s = float(base_n[sel]) if np.ndim(base_n) > 0 else float(base_n)
                    reb_tot += lam * (bn_s + float(D[j * K - 1, sel]))
                    spent += 1
        for t in range(j * K, min((j + 1) * K, T)):
            scan_tot += float((Q[t] * np.maximum(0.0, D[t] - d_reset)).sum())
            ewma_q = Q[t] if t == 0 else (0.3 * Q[t] + 0.7 * ewma_q)
    return scan_tot, scan_tot + reb_tot


def run_real_streams() -> pd.DataFrame:
    manifest = json.loads((QD_DIR / "manifest.json").read_text(encoding="utf-8"))
    rows: List[Dict[str, object]] = []
    for item in manifest:
        z = np.load(QD_DIR / item["file"])
        Q = z["Q"].astype(np.float64)
        D = z["D_delta"].astype(np.float64)
        T, R = Q.shape
        J = T // K
        Q = Q[: J * K]
        D = D[: J * K]
        w0 = float((Q * D).sum())
        ds_family = "wikimedia" if item["dataset"].startswith("wiki_") else "clinc150"
        for B in (3, 5, 7):
            if B >= J:
                continue
            ws = r3e1.solve_prop1_milp_with_cost(Q, D, K, B, lam=0.0, base_n=0.0)
            denom = max(w0 - ws, 1e-9)
            for pol, mode in [
                ("rr", "none"),
                ("rr", "uniform"),
                ("past_qmass", "none"),
                ("past_qmass", "uniform"),
                ("past_qmass", "uniform_shift"),
                ("ewma_a03", "none"),
                ("ewma_a03", "uniform"),
                ("oracle_1blk", "none"),
                ("oracle_1blk", "uniform"),
            ]:
                s0, _ = eval_with_modes(Q, D, B, pol, mode, lam=0.0, base_n=0.0)
                rows.append({
                    "family": ds_family,
                    "dataset": item["dataset"],
                    "trace": item["trace"],
                    "T": J * K,
                    "J": J,
                    "R": R,
                    "B": B,
                    "policy": pol,
                    "mode": mode,
                    "gcr_pct": 100.0 * (w0 - s0) / denom,
                })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "r3_exp2_real_streams_pacing.csv", index=False)
    return df


def _cost_cell(args: Tuple[str, str, int, int, int, float]) -> List[Dict[str, object]]:
    gen, coupling, J, B, seed, mult = args
    R = 16
    lam = mult * LAMBDA_0
    base_n = 100_000.0 / float(R)
    Q, D = e11a.gen_trace(gen, coupling, R, J, seed=zlib.crc32(f"{gen}|{coupling}|{R}|{J}|{B}|{seed}".encode()))
    w0 = float((Q * D).sum())
    ws = r3e1.solve_prop1_milp_with_cost(Q, D, K, B, lam=lam, base_n=base_n)
    denom = w0 - ws
    rows: List[Dict[str, object]] = []
    for pol, mode in [
        ("past_qmass", "none"),
        ("past_qmass", "uniform"),
        ("past_qmass", "uniform_shift"),
        ("oracle_1blk", "none"),
        ("oracle_1blk", "uniform"),
    ]:
        _, tot = eval_with_modes(Q, D, B, pol, mode, lam=lam, base_n=base_n)
        rows.append({
            "source": "synth_240",
            "gen": gen,
            "coupling": coupling,
            "J": J,
            "B": B,
            "seed": seed,
            "lam_mult": mult,
            "lam": lam,
            "policy": pol,
            "mode": mode,
            "gcr_pct": (100.0 * (w0 - tot) / denom) if denom > 1e-6 else np.nan,
            "net_saved_pct_vs_noop": 100.0 * (w0 - tot) / max(w0, 1e-9),
        })
    return rows


def run_cost_phase() -> pd.DataFrame:
    mults = [0.0, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]
    gens = ("stationary", "abrupt1", "abrupt2", "abrupt3", "gradual", "periodic")
    cells = [
        (g, c, J, 5, s, m)
        for g in gens
        for c in ("decoupled", "colocated")
        for J in (10, 20)
        for s in range(10)
        for m in mults
    ]
    with Pool(8) as pool:
        rows = [r for rs in pool.imap_unordered(_cost_cell, cells, chunksize=12) for r in rs]

    # Also evaluate real streams across the same lambda multipliers at B=5
    manifest = json.loads((QD_DIR / "manifest.json").read_text(encoding="utf-8"))
    for item in manifest:
        z = np.load(QD_DIR / item["file"])
        Q = z["Q"].astype(np.float64)
        D = z["D_delta"].astype(np.float64)
        T, R = Q.shape
        J = T // K
        Q, D = Q[: J * K], D[: J * K]
        w0 = float((Q * D).sum())
        base_n = z["base_n"].astype(np.float64) if "base_n" in z else (100_000.0 / float(R))
        fam = "wikimedia" if item["dataset"].startswith("wiki_") else "clinc150"
        for mult in mults:
            lam = mult * LAMBDA_0
            ws = r3e1.solve_prop1_milp_with_cost(Q, D, K, 5, lam=lam, base_n=base_n)
            denom = w0 - ws
            for pol, mode in [
                ("past_qmass", "none"),
                ("past_qmass", "uniform"),
                ("past_qmass", "uniform_shift"),
                ("oracle_1blk", "none"),
                ("oracle_1blk", "uniform"),
            ]:
                _, tot = eval_with_modes(Q, D, 5, pol, mode, lam=lam, base_n=base_n)
                rows.append({
                    "source": fam,
                    "gen": item["dataset"],
                    "coupling": "real",
                    "J": J,
                    "B": 5,
                    "seed": item["trace"],
                    "lam_mult": mult,
                    "lam": lam,
                    "policy": pol,
                    "mode": mode,
                    "gcr_pct": (100.0 * (w0 - tot) / denom) if denom > 1e-6 else np.nan,
                    "net_saved_pct_vs_noop": 100.0 * (w0 - tot) / max(w0, 1e-9),
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "r3_exp2_cost_phase.csv", index=False)
    return df


def main() -> int:
    df_real = run_real_streams()
    df_cost = run_cost_phase()

    piv_r = df_real.pivot_table(index=["family", "dataset", "trace", "B"], columns=["policy", "mode"], values="gcr_pct")
    real_sum = {}
    for fam in ("wikimedia", "clinc150", None):
        sub = piv_r.loc[fam] if fam is not None else piv_r
        key = fam or "all_real"
        real_sum[key] = {
            "n": int(len(sub)),
            "rr_greedy": round(float(sub[("rr", "none")].mean()), 2),
            "pq_greedy": round(float(sub[("past_qmass", "none")].mean()), 2),
            "pq_uniform": round(float(sub[("past_qmass", "uniform")].mean()), 2),
            "pq_shift": round(float(sub[("past_qmass", "uniform_shift")].mean()), 2),
            "ewma_greedy": round(float(sub[("ewma_a03", "none")].mean()), 2),
            "ewma_uniform": round(float(sub[("ewma_a03", "uniform")].mean()), 2),
            "oracle_uniform": round(float(sub[("oracle_1blk", "uniform")].mean()), 2),
            "when_gain": round(float((sub[("past_qmass", "uniform")] - sub[("past_qmass", "none")]).mean()), 2),
            "when_shift_gain": round(float((sub[("past_qmass", "uniform_shift")] - sub[("past_qmass", "none")]).mean()), 2),
            "when_win_pct": round(float(100.0 * ((sub[("past_qmass", "uniform")] - sub[("past_qmass", "none")]) > 0).mean()), 1),
        }

    piv_c = df_cost.pivot_table(index=["source", "gen", "coupling", "J", "B", "seed", "lam_mult"], columns=["policy", "mode"], values="gcr_pct", dropna=False)
    piv_n = df_cost.pivot_table(index=["source", "gen", "coupling", "J", "B", "seed", "lam_mult"], columns=["policy", "mode"], values="net_saved_pct_vs_noop")
    cost_sum = {}
    for src in ("synth_240", "wikimedia", "clinc150"):
        s = piv_c.loc[src]
        sn = piv_n.loc[src]
        tbl = []
        for mult in sorted(sn.index.get_level_values("lam_mult").unique()):
            sub = s.xs(mult, level="lam_mult")
            sub_n = sn.xs(mult, level="lam_mult")
            valid_n = int(sub[("past_qmass", "uniform")].notna().sum())
            def _m(frame: pd.DataFrame, col: Tuple[str, str]) -> float | None:
                val = frame[col].mean(skipna=True)
                return None if pd.isna(val) else round(float(val), 2)
            wg_s = (sub[("past_qmass", "uniform")] - sub[("past_qmass", "none")]).dropna()
            wn_s = (sub_n[("past_qmass", "uniform")] - sub_n[("past_qmass", "none")]).dropna()
            tbl.append({
                "lam_mult": float(mult),
                "n_total": int(len(sub_n)),
                "n_gcr_valid": valid_n,
                "pq_greedy_gcr": _m(sub, ("past_qmass", "none")),
                "pq_uniform_gcr": _m(sub, ("past_qmass", "uniform")),
                "pq_shift_gcr": _m(sub, ("past_qmass", "uniform_shift")),
                "oracle_uniform_gcr": _m(sub, ("oracle_1blk", "uniform")),
                "when_gain_gcr": None if wg_s.empty else round(float(wg_s.mean()), 2),
                "pq_greedy_net": _m(sub_n, ("past_qmass", "none")),
                "pq_uniform_net": _m(sub_n, ("past_qmass", "uniform")),
                "pq_shift_net": _m(sub_n, ("past_qmass", "uniform_shift")),
                "oracle_uniform_net": _m(sub_n, ("oracle_1blk", "uniform")),
                "when_gain_net": None if wn_s.empty else round(float(wn_s.mean()), 2),
            })
        cost_sum[src] = tbl

    summary = {"real_streams": real_sum, "cost_phase": cost_sum}
    (OUT_DIR / "r3_exp2_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
