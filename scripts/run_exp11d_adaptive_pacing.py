#!/usr/bin/env python3
"""Exp 11-D: does adaptive pacing beat the uniform cap? Pure replay, same 2,400 traces as Exp 11-A.

Shard choice is fixed to Past-QMass (or the one-block oracle for reference); only the timing rule changes.
All rules are causal: they see query mass and deltas up to the current boundary only.
  none           : greedy, act at every boundary while budget remains
  uniform        : cumulative spend at boundary j capped at floor(B * j / (J - 1))   (Exp 11-A)
  uniform_shift  : uniform cap, but may borrow one action ahead of the cap when the shard with the
                   largest recent query mass differs from the one in the previous block (shift detected)
  threshold      : act when the best score reaches the running median of best scores seen so far,
                   and always act when the remaining boundaries equal the remaining budget
  thr_capped     : threshold rule, never more than one action ahead of the uniform cap
Every trace is scored against its own exact Prop. 1 MILP optimum (GCR, %). No timing is measured.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

import run_exp11a_pacing_generality as a11
import run_xww_exp10f_milp_scale_and_multiR as f2

K = a11.K
POLICIES = ("past_qmass", "oracle_1blk")
MODES = ("none", "uniform", "uniform_shift", "threshold", "thr_capped")


def eval_policy(Q, D, B, policy, mode):
    T, R = Q.shape
    J = T // K
    d_reset = np.zeros(R)
    spent, tot = 0, 0.0
    hist = []
    prev_top = None
    for j in range(J):
        allow = j >= 1 and spent < B
        score = None
        if allow:
            cur_d = np.maximum(0.0, D[j * K - 1] - d_reset)
            qwin = Q[(j - 1) * K: j * K] if policy == "past_qmass" else Q[j * K: (j + 1) * K]
            score = qwin.sum(axis=0) * cur_d
            top = int(np.argmax(Q[(j - 1) * K: j * K].sum(axis=0)))
            cap = math.floor(B * j / (J - 1))
            best = float(score.max())
            remaining_bounds = J - j  # boundaries j..J-1
            forced = remaining_bounds <= (B - spent)
            if mode == "uniform":
                allow = spent < cap
            elif mode == "uniform_shift":
                shifted = prev_top is not None and top != prev_top
                allow = spent < cap or (shifted and spent < cap + 1)
            elif mode in ("threshold", "thr_capped"):
                med = float(np.median(hist)) if hist else 0.0
                allow = forced or (hist and best >= med)
                if mode == "thr_capped":
                    allow = allow and (forced or spent < cap + 1)
            hist.append(best)
            prev_top = top
        elif j >= 1:
            prev_top = int(np.argmax(Q[(j - 1) * K: j * K].sum(axis=0)))
        if allow and score is not None:
            sel = int(np.argmax(score))
            if score[sel] > 0 or np.maximum(0.0, D[j * K - 1] - d_reset)[sel] > 0:
                d_reset[sel] = D[j * K - 1, sel]
                spent += 1
        for t in range(j * K, min((j + 1) * K, T)):
            tot += float((Q[t] * np.maximum(0.0, D[t] - d_reset)).sum())
    return tot


def run_cell(args):
    gen, coupling, R, J, B, seed = args
    Q, D = a11.gen_trace(gen, coupling, R, J, seed=zlib.crc32(f"{gen}|{coupling}|{R}|{J}|{B}|{seed}".encode()))
    w0 = float((Q * D).sum())
    ws, _, _, _, _ = f2.solve_prop1_milp(Q, D, K, B)
    denom = max(w0 - ws, 1e-9)
    rows = []
    for pol, mode in itertools.product(POLICIES, MODES):
        wp = eval_policy(Q, D, B, pol, mode)
        rows.append(dict(gen=gen, coupling=coupling, R=R, J=J, B=B, seed=seed, policy=pol, mode=mode,
                         gcr_pct=100 * (w0 - wp) / denom))
    # consistency check against Exp 11-A implementation for none/uniform
    for mode in ("none", "uniform"):
        ref = a11.eval_policy(Q, D, B, "past_qmass", mode)
        mine = eval_policy(Q, D, B, "past_qmass", mode)
        rows.append(dict(gen=gen, coupling=coupling, R=R, J=J, B=B, seed=seed, policy="check_vs_11a", mode=mode,
                         gcr_pct=100 * (mine - ref) / denom))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=20)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gens = ("stationary", "abrupt1", "abrupt2", "abrupt3", "gradual", "periodic")
    jb = ((10, 3), (10, 5), (10, 7), (20, 5), (20, 10))
    cells = [(g, c, R, J, B, s) for g in gens for c in ("decoupled", "colocated") for R in (16, 64)
             for (J, B) in jb for s in range(args.seeds)]
    with Pool(args.procs) as pool:
        rows = [r for rs in pool.imap_unordered(run_cell, cells, chunksize=4) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(out / "exp11d_adaptive_pacing.csv", index=False)
    chk = df[df.policy == "check_vs_11a"]["gcr_pct"].abs().max()
    d = df[df.policy != "check_vs_11a"]
    piv = d.pivot_table(index=["gen", "coupling", "R", "J", "B", "seed"], columns=["policy", "mode"], values="gcr_pct")
    summ = {"max_abs_diff_vs_11a_pp": float(chk), "n_traces": int(len(piv))}
    for pol in POLICIES:
        for m in MODES:
            summ[f"{pol}_{m}_mean"] = round(float(piv[(pol, m)].mean()), 2)
        for m in MODES[2:]:
            dd = piv[(pol, m)] - piv[(pol, "uniform")]
            summ[f"{pol}_{m}_minus_uniform"] = {"mean": round(float(dd.mean()), 2),
                                                 "frac_gain_gt1": round(float((dd > 1).mean()), 3),
                                                 "frac_loss_gt1": round(float((dd < -1).mean()), 3)}
    by = d.groupby(["gen", "policy", "mode"])["gcr_pct"].mean().round(2).unstack(["policy", "mode"])
    summ["by_gen"] = json.loads(by.to_json(orient="index"))
    byjb = d[d.policy == "past_qmass"].groupby(["J", "B", "mode"])["gcr_pct"].mean().round(2).unstack("mode")
    summ["by_JB"] = json.loads(byjb.to_json(orient="index"))
    (out / "exp11d_summary.json").write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8")
    (out / "EXP11D_DONE.flag").write_text("ok\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k not in ("by_gen", "by_JB")}, indent=1))
    print(by.to_string())
    print(byjb.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
