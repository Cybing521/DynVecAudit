# harness/backends — external-baseline plugins (E3 protocol v1.9)

Each file here adds one **external baseline** to `window_harness.py` without touching the harness:

```bash
python3 engines/window_harness.py --system hnswlib_mut --trace uniform_s05 \
    --data-root data/sift100k_u10_q2500_v2 \
    --bl-args '{"ef":64}' --out results/harness/hnswlib_mut_ef64_uniform_s05.jsonl
```

## Contract (must match exactly; the analysis pipeline depends on it)

`make_backend(args, data) -> backend` where `backend` provides:

| attribute / method | meaning |
|---|---|
| `name` | system id used in the JSONL header and in the curve file name |
| `n_threads` | number of threads the backend actually uses (recorded in every row) |
| `params` | dict of the knobs used for this run (recorded in every row) |
| `build_cpu_ms`, `build_wall_ms` | initial build (0 if the system has no build step) |
| `run_window(data, w, crud, qe) -> dict` | replay one window |

`data` (built by `load_common`): `root, trdir, pool (np.float32 [nvec,dim]), init (structured: id, vec_ref),
ev (trace events), gt_ids [nq,k], n_windows, q_off (window -> query row offset), k, dim`.

`crud`: structured array of this window's non-query events (`type` in {EV_INSERT=0, EV_UPDATE=1, EV_ERASE=2},
`id`, `vec_ref`). `qe`: structured array of this window's query events (`id`, `vec_ref`).

Returned dict must contain: `n_insert`, `n_queries`, `query_cpu_ms` (process CPU for queries),
`maint_cpu_ms` (process CPU for update/maintenance work), `p50_us`, `p95_us`, `p99_us` (per-query wall),
`mean_recall`, `window_status` (`"ok"` or reason). Optional: `insert_cpu_ms`,
`explicit_maintenance_cpu_ms`, `rebuild_triggered`, `index_n`.

## Rules (protocol v1.8 §6, non-negotiable)

1. **Timing**: `time.process_time()` for CPU (single-threaded work); `time.perf_counter()` per query for wall.
   Never subtract other-process load, never "correct" the numbers.
2. **One thread**: single-threaded unless the published design *is* multi-threaded (Quake runs 2 online threads,
   documented in its params). Set `OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`,
   `faiss.omp_set_num_threads(1)`, etc.
3. **Rebuild systems**: a full rebuild is *maintenance work* -> charge it to `maint_cpu_ms` in the window where
   it happens. The initial build goes to `build_cpu_ms` and is **excluded** from total CPU.
4. **No cheating on recall**: return exactly what the system returns; `mean_recall` is computed by the harness
   against the trace's exact GT. Never post-filter by GT.
5. **Deletes**: if the system cannot delete, report it in `window_status` instead of silently ignoring events.
   The q2500 main grid is insert-only (u=10%), so this only matters for future CRUD grids.
6. Record every knob in `params` so the curve file can be attributed unambiguously.
