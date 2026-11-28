#!/usr/bin/env python3
"""Exp 11-A: generality of budget pacing vs. forecasting (storyline S7), pure replay.

Extends Exp 10-F3 (single switch) to six drift families, two insert couplings, R in {16, 64},
(J, B) in {(10,3), (10,5), (10,7), (20,5), (20,10)}, 20 seeds per cell. Every trace is scored
against its own exact Prop. 1 MILP optimum. Pacing modes per greedy policy:
  none     : act at every boundary while budget remains (greedy)
  uniform  : cumulative spend at boundary j capped at floor(B * j / (J - 1))
  spacing  : at least ceil((J - 1) / B) blocks between consecutive actions
No timing is measured, so the job can share a host with other work.
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

import run_xww_exp10f_milp_scale_and_multiR as f2

K = 4
POLICIES = ("past_qmass", "ewma_a03", "max_delta", "oracle_1blk")
MODES = ("none", "uniform", "spacing")


def hotspot_schedule(gen: str, J: int, rng: np.random.Generator, R: int):
    """Return per-window (h_from, h_to, mix) where query weight on h_to is `mix`."""
    T = J * K
    hubs = rng.choice(R, size=4, replace=False)
    sched = np.zeros((T, 2), dtype=int)
    mix = np.zeros(T)
    if gen == "stationary":
        sched[:] = hubs[0]
        return sched, mix
    if gen.startswith("abrupt"):
        n_sw = int(gen[-1])
        sw_blocks = np.sort(rng.choice(np.arange(2, J - 1), size=n_sw, replace=False))
        sw_win = sw_blocks * K + rng.integers(0, K, size=n_sw)
        cur = 0
        for t in range(T):
            while cur < n_sw and t >= sw_win[cur]:
                cur += 1
            sched[t] = hubs[cur]
        return sched, mix
    if gen == "gradual":
        s = int(rng.integers(1, J - 5)) * K
        L = 4 * K
        for t in range(T):
            sched[t] = (hubs[0], hubs[1])
            mix[t] = float(np.clip((t - s) / L, 0.0, 1.0))
        return sched, mix
    if gen == "periodic":
        P = int(rng.integers(2, 4)) * K
        for t in range(T):
            sched[t] = hubs[(t // P) % 2]
        return sched, mix
    raise ValueError(gen)


def gen_trace(gen: str, coupling: str, R: int, J: int, seed: int, n_ins: int = 2500):
    rng = np.random.default_rng(seed)
    T = J * K
    sched, mix = hotspot_schedule(gen, J, rng, R)
    Q = np.zeros((T, R))
    ins = np.zeros((T, R))
    base_w = 1.0 / (np.arange(1, R + 1) ** 0.4)
    rng.shuffle(base_w)
    for t in range(T):
        qw = np.full(R, 0.25 / R)
        h_from, h_to = sched[t]
        qw[h_from] += 0.75 * (1.0 - mix[t])
        qw[h_to] += 0.75 * mix[t]
        Q[t] = rng.multinomial(2500, qw / qw.sum())
        if coupling == "decoupled":
            ip = base_w / base_w.sum()
        else:  # co-located: 65% of inserts follow the current query hotspot
            ip = np.full(R, 0.35 / R)
            ip[h_from] += 0.65 * (1.0 - mix[t])
            ip[h_to] += 0.65 * mix[t]
            ip /= ip.sum()
        ins[t] = rng.multinomial(n_ins, ip)
    return Q, np.cumsum(ins, axis=0)


def eval_policy(Q, D, B, policy, mode):
    T, R = Q.shape
    J = T // K
    gap = math.ceil((J - 1) / max(B, 1))
    d_reset = np.zeros(R)
    ewma_q = np.zeros(R)
    spent, last_j, tot = 0, -10 ** 9, 0.0
    for j in range(J):
        allow = j >= 1 and spent < B
        if mode == "uniform":
            allow = allow and spent < math.floor(B * j / (J - 1))
        elif mode == "spacing":
            allow = allow and (j - last_j) >= gap
        if allow:
            cur_d = np.maximum(0.0, D[j * K - 1] - d_reset)
            if policy == "max_delta":
                score = cur_d
            elif policy == "past_qmass":
                score = Q[(j - 1) * K: j * K].sum(axis=0) * cur_d
            elif policy == "ewma_a03":
                score = ewma_q * cur_d
            elif policy == "oracle_1blk":
                score = Q[j * K: (j + 1) * K].sum(axis=0) * cur_d
            sel = int(np.argmax(score))
            if cur_d[sel] > 0:
                d_reset[sel] = D[j * K - 1, sel]
                spent += 1
                last_j = j
        for t in range(j * K, min((j + 1) * K, T)):
            tot += float((Q[t] * np.maximum(0.0, D[t] - d_reset)).sum())
            ewma_q = Q[t] if t == 0 else 0.3 * Q[t] + 0.7 * ewma_q
    return tot


def run_cell(args):
    gen, coupling, R, J, B, seed = args
    Q, D = gen_trace(gen, coupling, R, J, seed=zlib.crc32(f"{gen}|{coupling}|{R}|{J}|{B}|{seed}".encode()))
    w0 = float((Q * D).sum())
    ws, _, _, _, _ = f2.solve_prop1_milp(Q, D, K, B)
    rows = []
    denom = max(w0 - ws, 1e-9)
    wc2 = f2.eval_policy_exact(Q, D, K, B, "fixed_c2")
    rows.append(dict(gen=gen, coupling=coupling, R=R, J=J, B=B, seed=seed, policy="fixed_c2", mode="none", gcr_pct=100 * (w0 - wc2) / denom))
    for pol, mode in itertools.product(POLICIES, MODES):
        wp = eval_policy(Q, D, B, pol, mode)
        rows.append(dict(gen=gen, coupling=coupling, R=R, J=J, B=B, seed=seed, policy=pol, mode=mode, gcr_pct=100 * (w0 - wp) / denom))
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
    cells = [(g, c, R, J, B, s) for g in gens for c in ("decoupled", "colocated") for R in (16, 64) for (J, B) in jb for s in range(args.seeds)]
    with Pool(args.procs) as pool:
        rows = [r for rs in pool.imap_unordered(run_cell, cells, chunksize=4) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(out / "exp11a_pacing_generality.csv", index=False)
    piv = df.pivot_table(index=["gen", "coupling", "R", "J", "B", "seed"], columns=["policy", "mode"], values="gcr_pct")
    summ = {}
    for pol in POLICIES:
        for m in ("uniform", "spacing"):
            d = piv[(pol, m)] - piv[(pol, "none")]
            summ[f"{pol}_{m}_minus_none"] = {"mean": round(float(d.mean()), 2), "frac_gain": round(float((d > 0).mean()), 3),
                                              "frac_loss_gt1pp": round(float((d < -1).mean()), 3)}
    summ["oracle_uniform_minus_pq_uniform"] = round(float((piv[("oracle_1blk", "uniform")] - piv[("past_qmass", "uniform")]).mean()), 2)
    summ["pq_uniform_beats_oracle_none_frac"] = round(float((piv[("past_qmass", "uniform")] > piv[("oracle_1blk", "none")]).mean()), 3)
    by_gen = df.groupby(["gen", "policy", "mode"])["gcr_pct"].mean().round(2).unstack(["policy", "mode"])
    summ["by_gen"] = json.loads(by_gen.to_json(orient="index"))
    (out / "exp11a_summary.json").write_text(json.dumps(summ, indent=1) + "\n", encoding="utf-8")
    (out / "EXP11A_DONE.flag").write_text("ok\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summ.items() if k != "by_gen"}, indent=1))
    print(by_gen.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
