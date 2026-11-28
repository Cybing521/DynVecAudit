#!/usr/bin/env python3
"""Floor-definition sensitivity for Act 1: does "IVF wins CPU at matched floor" depend on 0.965 / 95%?

Re-uses the config table of generate_story_figures.engine_configs() and adds, where the source CSV has them,
the share of windows at Recall@10 >= 0.95 and the minimum window recall. For each floor definition we redo the
matched-floor comparison (cheapest compliant IVF vs cheapest compliant graph per workload).
Output: results/paper_tables/act1_floor_sensitivity.csv and a printed summary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import generate_story_figures as g

_orig_std = g._std


def _std_ext(df, sys_c, tag_c, cpu_s, att_c, rec_c, p99_c, label, min_att=None):
    out = _orig_std(df, sys_c, tag_c, cpu_s, att_c, rec_c, p99_c, label, min_att)
    a95 = next((c for c in ("att_0950_pct", "attainment_0950_pct") if c in df.columns), None)
    out["att95"] = df[a95].astype(float).values if a95 else np.nan
    out["min_rec"] = df["min_recall"].astype(float).values if "min_recall" in df.columns else np.nan
    return out


g._std = _std_ext

FLOORS = [
    ("95% windows >= 0.965 (main)", lambda s: s["att"] >= 95.0),
    ("90% windows >= 0.965", lambda s: s["att"] >= 90.0),
    ("99% windows >= 0.965", lambda s: s["att"] >= 99.0),
    ("all windows >= 0.965", lambda s: s["att"] >= 100.0),
    ("95% windows >= 0.95", lambda s: s["att95"] >= 95.0),
    ("all windows >= 0.95", lambda s: s["att95"] >= 100.0),
    ("min window recall >= 0.90", lambda s: s["min_rec"] >= 0.90),
    ("mean recall >= 0.95 (no floor)", lambda s: s["rec"] >= 0.95),
]


def main() -> int:
    df = g.engine_configs()
    df = df[df["family"].isin(["IVF", "Graph"])]
    rows = []
    for name, f in FLOORS:
        for wl, s in df.groupby("workload", sort=False):
            mask = f(s).fillna(False)
            ok = s[mask]
            ivf, gr = ok[ok["family"] == "IVF"], ok[ok["family"] == "Graph"]
            r = {"floor": name, "workload": wl, "ivf_ok": not ivf.empty, "graph_ok": not gr.empty,
                 "defined": bool(f(s).notna().any())}
            if not ivf.empty and not gr.empty:
                bi, bg = ivf.loc[ivf["cpu_s"].idxmin()], gr.loc[gr["cpu_s"].idxmin()]
                bgp = gr.loc[gr["p99_us"].idxmin()]
                r.update(ivf_cfg=f"{bi['system']} {bi['tag']}", graph_cfg=f"{bg['system']} {bg['tag']}",
                         cpu_ratio=bg["cpu_s"] / bi["cpu_s"], p99_ratio=bi["p99_us"] / bgp["p99_us"])
            rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(g.RES / "paper_tables" / "act1_floor_sensitivity.csv", index=False)
    summ = []
    for name, _ in FLOORS:
        s = out[out["floor"] == name]
        both = s.dropna(subset=["cpu_ratio"]) if "cpu_ratio" in s else s.iloc[0:0]
        summ.append({"floor": name, "workloads_defined": int(s["defined"].sum()), "pairs": len(both),
                     "ivf_cpu_wins": int((both["cpu_ratio"] > 1).sum()),
                     "cpu_ratio_min": round(both["cpu_ratio"].min(), 2) if len(both) else None,
                     "cpu_ratio_max": round(both["cpu_ratio"].max(), 2) if len(both) else None,
                     "graph_p99_wins": int((both["p99_ratio"] > 1).sum()),
                     "no_graph": int((s["ivf_ok"] & ~s["graph_ok"]).sum()),
                     "no_ivf": int((~s["ivf_ok"] & s["graph_ok"]).sum())})
    sm = pd.DataFrame(summ)
    sm.to_csv(g.RES / "paper_tables" / "act1_floor_sensitivity_summary.csv", index=False)
    print(sm.to_string(index=False))
    lose = out[(out.get("cpu_ratio", pd.Series(dtype=float)) <= 1)]
    if len(lose):
        print("\nIVF loses CPU:\n", lose.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
