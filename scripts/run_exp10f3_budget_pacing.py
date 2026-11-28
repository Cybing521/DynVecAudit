#!/usr/bin/env python3
"""Exp 10-F3: does budget pacing close the unpaced headroom gap of Exp 10-F2?

Exp 10-F2 found that, with B=5 over J=10 blocks and no pacing, every greedy policy spends
its whole budget in blocks 1-5 while the hotspot moves at block 5, so Past-QMass captures
22-29% of Prop. 1 headroom versus 84-87% for a 1-block oracle. This replay re-uses the
identical Exp 10-F2 traces (same generator, seeds, MILP bound) and adds a uniform pacing cap
(spent <= floor(B * j / (J - 1)) at boundary j) to each greedy policy. Pure replay, no timing.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_xww_exp10f_milp_scale_and_multiR as f2


def eval_policy(Q: np.ndarray, D: np.ndarray, K: int, B: int, policy: str, paced: bool) -> float:
    T, R = Q.shape
    J = T // K
    d_reset = np.zeros(R)
    ewma_q = np.zeros(R)
    spent, rr_ptr, tot = 0, 0, 0.0
    for j in range(J):
        cap = B if not paced else int(np.floor(B * j / (J - 1)))
        if j >= 1 and spent < min(B, cap) and policy != "noop":
            cur_d = np.maximum(0.0, D[j * K - 1] - d_reset)
            sel = -1
            if policy == "fixed_c2":
                if (j - 1) % 2 == 0:
                    sel = rr_ptr % R
                    rr_ptr += 1
            elif policy == "max_delta":
                sel = int(np.argmax(cur_d))
            elif policy == "past_qmass":
                sel = int(np.argmax(Q[(j - 1) * K: j * K].sum(axis=0) * cur_d))
            elif policy == "ewma_a03":
                sel = int(np.argmax(ewma_q * cur_d))
            elif policy == "oracle_1blk":
                sel = int(np.argmax(Q[j * K: (j + 1) * K].sum(axis=0) * cur_d))
            if sel >= 0 and cur_d[sel] > 0:
                d_reset[sel] = D[j * K - 1, sel]
                spent += 1
        for t in range(j * K, min((j + 1) * K, T)):
            cur_d = np.maximum(0.0, D[t] - d_reset)
            tot += float((Q[t] * cur_d).sum())
            ewma_q = Q[t] if t == 0 else (0.3 * Q[t] + 0.7 * ewma_q)
    return tot


def gen_trace_rand_tsw(R: int, J: int, K: int, seed: int, coupling: str, n_ins_win: int = 2500):
    """Exp 10-F2 generator with the hotspot switch drawn uniformly from blocks 2..J-2 per seed."""
    rng = np.random.default_rng(seed)
    T = J * K
    Q = np.zeros((T, R))
    ins = np.zeros((T, R))
    h0, h1 = rng.choice(R, size=2, replace=False)
    t_sw = int(rng.integers(2, J - 1)) * K + int(rng.integers(0, K))
    if coupling == "decoupled_uniform_writes":
        base_w = 1.0 / (np.arange(1, R + 1) ** 0.4)
        rng.shuffle(base_w)
        ins_p = base_w / base_w.sum()
    else:
        ins_p = np.full(R, 0.35 / R)
        ins_p[h0] += 0.35
        ins_p[h1] += 0.30
        ins_p /= ins_p.sum()
    for t in range(T):
        qw = np.full(R, 0.25 / R)
        qw[h0 if t < t_sw else h1] += 0.75
        Q[t] = rng.multinomial(2500, qw)
        ins[t] = rng.multinomial(n_ins_win, ins_p)
    return Q, np.cumsum(ins, axis=0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rand-tsw", action="store_true", help="draw the switch time per seed instead of T//2")
    args = ap.parse_args()
    gen = gen_trace_rand_tsw if args.rand_tsw else f2.gen_trace
    tag = "_randtsw" if args.rand_tsw else ""
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = [("100K", 16, 10, 5, 2500), ("1M", 32, 10, 5, 5000), ("5M", 64, 10, 5, 25000), ("10M", 128, 10, 5, 50000)]
    rows = []
    for coupling in ("decoupled_uniform_writes", "colocated_hotspot_writes"):
        for lbl, R, J, B, n_ins in specs:
            for sd in range(20 if args.rand_tsw else 10):
                Q, D = gen(R, J, 4, seed=(9000 if args.rand_tsw else 7000) + R * 31 + sd, coupling=coupling, n_ins_win=n_ins)
                w0 = f2.eval_policy_exact(Q, D, 4, B, "noop")
                ws, _, _, _, _ = f2.solve_prop1_milp(Q, D, 4, B)
                for pol in ("past_qmass", "ewma_a03", "max_delta", "oracle_1blk"):
                    for paced in (False, True):
                        wp = eval_policy(Q, D, 4, B, pol, paced)
                        rows.append({"coupling": coupling, "scale": lbl, "R": R, "seed": sd, "policy": pol,
                                     "paced": paced, "gcr_pct": 100.0 * (w0 - wp) / max(w0 - ws, 1e-9)})
    df = pd.DataFrame(rows)
    df.to_csv(out / f"exp10f3_pacing_vs_unpaced{tag}.csv", index=False)
    # sanity: unpaced replay must reproduce Exp 10-F2 values exactly
    agg = df.groupby(["coupling", "scale", "R", "policy", "paced"], as_index=False)["gcr_pct"].mean().round(2)
    summ = {"cells": agg.to_dict(orient="records")}
    (out / f"exp10f3_summary{tag}.json").write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8")
    (out / f"EXP10F3{tag}_DONE.flag").write_text("ok\n", encoding="utf-8")
    print(agg.pivot_table(index=["coupling", "R"], columns=["policy", "paced"], values="gcr_pct").to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
