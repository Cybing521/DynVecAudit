use anyhow::Result;
use clap::{Parser, ValueEnum};
use dynvec_core::{
    mean_recall_at_k, parse_core_list, peak_rss_mb, percentile_sorted, pin_process_to_cores,
    process_cpu_ms, read_loadavg_1m, BigannWorkload, QbamTrace, EV_ERASE, EV_INSERT, EV_UPDATE,
};
use hnsw_rs::prelude::{DistL2, Hnsw};
use serde::Serialize;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind};

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Mode {
    Qbam,
    Bigann,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum EngineKind {
    UsearchRust,
    HnswRs,
}

#[derive(Parser)]
#[command(name = "dynvec-bench", version, about = "Zero-FFI native Rust vector benchmark harness")]
struct Cli {
    #[arg(long, value_enum)]
    mode: Mode,
    #[arg(long, value_enum)]
    engine: EngineKind,
    #[arg(long)]
    data_dir: PathBuf,
    #[arg(long)]
    src_dir: Option<PathBuf>,
    #[arg(long, default_value = "")]
    trace: String,
    #[arg(long, default_value_t = 16)]
    m: usize,
    #[arg(long, default_value_t = 200)]
    ef_construction: usize,
    #[arg(long, default_value_t = 64)]
    ef: usize,
    #[arg(long)]
    ef_list: Option<String>,
    #[arg(long, default_value_t = 1)]
    threads: usize,
    #[arg(long, default_value = "0")]
    cores: String,
    #[arg(long, default_value_t = false)]
    batch_query: bool,
    #[arg(long)]
    max_windows: Option<usize>,
    #[arg(long)]
    out_json: Option<PathBuf>,
}

#[derive(Serialize)]
struct BenchResult {
    engine: String,
    mode: String,
    trace: String,
    threads: usize,
    cores: String,
    m: usize,
    ef_construction: usize,
    ef: usize,
    batch_query: bool,
    n_windows: usize,
    status: String,
    build_cpu_s: f64,
    build_wall_s: f64,
    query_cpu_s: f64,
    maint_cpu_s: f64,
    total_cpu_s: f64,
    dyn_wall_s: f64,
    wall_s: f64,
    wall_cpu_ratio: f64,
    query_pct: f64,
    maint_pct: f64,
    mean_recall: f64,
    min_recall: f64,
    att_0965_pct: f64,
    att_0950_pct: f64,
    p50_us: Option<f64>,
    p95_us: Option<f64>,
    p99_us: Option<f64>,
    peak_rss_mb: f64,
    load_start: f64,
    load_end: f64,
}

struct UsearchState {
    idx: Index,
    label_of: HashMap<u64, u64>,
    synth_to_orig: Vec<u64>,
    label_base: u64,
    next_label: u64,
}

impl UsearchState {
    fn new(dim: usize, m: usize, ef_c: usize, ef_s: usize, cap: usize, threads: usize, max_id: i64) -> Result<Self> {
        let opts = IndexOptions {
            dimensions: dim,
            metric: MetricKind::L2sq,
            quantization: ScalarKind::F32,
            connectivity: m,
            expansion_add: ef_c,
            expansion_search: ef_s,
            multi: false,
        };
        let idx = Index::new(&opts)?;
        idx.reserve_capacity_and_threads(cap, threads.max(1))?;
        let base = (max_id + 1).max(0) as u64;
        Ok(Self {
            idx,
            label_of: HashMap::new(),
            synth_to_orig: Vec::new(),
            label_base: base,
            next_label: base,
        })
    }

    #[inline]
    fn decode_key(&self, raw: u64) -> i64 {
        if !self.synth_to_orig.is_empty() && raw >= self.label_base {
            let off = (raw - self.label_base) as usize;
            if let Some(&orig) = self.synth_to_orig.get(off) {
                return orig as i64;
            }
        }
        raw as i64
    }
}

fn parse_ef_list(cli: &Cli) -> Result<Vec<usize>> {
    if let Some(ref spec) = cli.ef_list {
        let mut out = Vec::new();
        for tok in spec.split(',') {
            let s = tok.trim();
            if !s.is_empty() {
                out.push(s.parse()?);
            }
        }
        if !out.is_empty() {
            return Ok(out);
        }
    }
    Ok(vec![cli.ef])
}

fn configure_threads(threads: usize, cores_spec: &str) -> Result<rayon::ThreadPool> {
    unsafe {
        std::env::set_var("OMP_NUM_THREADS", threads.to_string());
    }
    let cores = parse_core_list(cores_spec)?;
    pin_process_to_cores(&cores)?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads.max(1)).build()?;
    Ok(pool)
}

#[allow(clippy::too_many_arguments)]
fn build_results(
    cli: &Cli,
    mode_str: &str,
    trace_name: &str,
    n_win: usize,
    unsupported_erase: bool,
    efs: &[usize],
    build_cpu_s: f64,
    build_wall_s: f64,
    maint_cpu_ms: f64,
    maint_wall_s: f64,
    query_cpu_ms: &[f64],
    query_wall_s: &[f64],
    recalls: &[Vec<f64>],
    p50s: &[Vec<f64>],
    p95s: &[Vec<f64>],
    p99s: &[Vec<f64>],
    load_start: f64,
) -> Vec<BenchResult> {
    let rss = peak_rss_mb();
    let load_end = read_loadavg_1m();
    let m_s = maint_cpu_ms / 1000.0;
    let avg = |xs: &[f64]| -> Option<f64> {
        if xs.is_empty() {
            None
        } else {
            Some(((xs.iter().sum::<f64>() / xs.len() as f64) * 100.0).round() / 100.0)
        }
    };

    efs.iter()
        .enumerate()
        .map(|(i, &ef_val)| {
            let q_s = query_cpu_ms[i] / 1000.0;
            let tot_cpu_s = q_s + m_s;
            let dyn_wall = maint_wall_s + query_wall_s[i];
            let wall_s = build_wall_s + dyn_wall;
            let mean_r = recalls[i].iter().sum::<f64>() / (recalls[i].len().max(1) as f64);
            let min_r = recalls[i].iter().cloned().fold(f64::INFINITY, f64::min);
            let att965 =
                100.0 * (recalls[i].iter().filter(|&&r| r >= 0.965).count() as f64) / (recalls[i].len().max(1) as f64);
            let att950 =
                100.0 * (recalls[i].iter().filter(|&&r| r >= 0.950).count() as f64) / (recalls[i].len().max(1) as f64);
            BenchResult {
                engine: match cli.engine {
                    EngineKind::UsearchRust => "usearch_rust".into(),
                    EngineKind::HnswRs => "hnsw_rs".into(),
                },
                mode: mode_str.into(),
                trace: trace_name.into(),
                threads: cli.threads,
                cores: cli.cores.clone(),
                m: cli.m,
                ef_construction: cli.ef_construction,
                ef: ef_val,
                batch_query: cli.batch_query,
                n_windows: n_win,
                status: if unsupported_erase {
                    "unsupported_erase".into()
                } else {
                    "ok".into()
                },
                build_cpu_s: (build_cpu_s * 10000.0).round() / 10000.0,
                build_wall_s: (build_wall_s * 10000.0).round() / 10000.0,
                query_cpu_s: (q_s * 10000.0).round() / 10000.0,
                maint_cpu_s: (m_s * 10000.0).round() / 10000.0,
                total_cpu_s: (tot_cpu_s * 10000.0).round() / 10000.0,
                dyn_wall_s: (dyn_wall * 10000.0).round() / 10000.0,
                wall_s: (wall_s * 10000.0).round() / 10000.0,
                wall_cpu_ratio: if tot_cpu_s + build_cpu_s > 0.0 {
                    ((wall_s / (tot_cpu_s + build_cpu_s)) * 10000.0).round() / 10000.0
                } else {
                    1.0
                },
                query_pct: if tot_cpu_s > 0.0 {
                    ((100.0 * q_s / tot_cpu_s) * 100.0).round() / 100.0
                } else {
                    0.0
                },
                maint_pct: if tot_cpu_s > 0.0 {
                    ((100.0 * m_s / tot_cpu_s) * 100.0).round() / 100.0
                } else {
                    0.0
                },
                mean_recall: (mean_r * 100000.0).round() / 100000.0,
                min_recall: (min_r * 100000.0).round() / 100000.0,
                att_0965_pct: (att965 * 10.0).round() / 10.0,
                att_0950_pct: (att950 * 10.0).round() / 10.0,
                p50_us: avg(&p50s[i]),
                p95_us: avg(&p95s[i]),
                p99_us: avg(&p99s[i]),
                peak_rss_mb: rss,
                load_start,
                load_end,
            }
        })
        .collect()
}

fn run_qbam(cli: &Cli, pool_rt: &rayon::ThreadPool, efs: &[usize]) -> Result<Vec<BenchResult>> {
    let load_start = read_loadavg_1m();
    let t_total_w0 = Instant::now();
    let data = QbamTrace::load(&cli.data_dir, &cli.trace)?;
    let n_win = cli.max_windows.map(|m| m.min(data.n_windows)).unwrap_or(data.n_windows);

    let t_build_c0 = process_cpu_ms();
    let mut us_state: Option<UsearchState> = None;
    let mut hn_state: Option<Hnsw<'static, f32, DistL2>> = None;

    match cli.engine {
        EngineKind::UsearchRust => {
            let cap = data.init.len() + data.n_inserts_total + data.n_updates_total + 50_000;
            let st = UsearchState::new(data.dim, cli.m, cli.ef_construction, efs[0], cap, cli.threads, data.max_nonquery_id)?;
            if cli.threads > 1 {
                pool_rt.install(|| -> Result<()> {
                    use rayon::prelude::*;
                    data.init.par_iter().try_for_each(|rec| -> Result<()> {
                        st.idx.add(rec.id, data.vec_slice(rec.vec_ref))?;
                        Ok(())
                    })
                })?;
            } else {
                for rec in &data.init {
                    st.idx.add(rec.id, data.vec_slice(rec.vec_ref))?;
                }
            }
            us_state = Some(st);
        }
        EngineKind::HnswRs => {
            let cap = data.init.len() + data.n_inserts_total + data.n_updates_total + 50_000;
            let hn = Hnsw::<f32, DistL2>::new(cli.m, cap, 16, cli.ef_construction, DistL2 {});
            if cli.threads > 1 {
                let items: Vec<(Vec<f32>, usize)> = data
                    .init
                    .iter()
                    .map(|r| (data.vec_slice(r.vec_ref).to_vec(), r.id as usize))
                    .collect();
                let refs: Vec<(&Vec<f32>, usize)> = items.iter().map(|(v, id)| (v, *id)).collect();
                pool_rt.install(|| {
                    hn.parallel_insert(&refs);
                });
            } else {
                for rec in &data.init {
                    hn.insert((data.vec_slice(rec.vec_ref), rec.id as usize));
                }
            }
            hn_state = Some(hn);
        }
    }

    let build_cpu_s = (process_cpu_ms() - t_build_c0) / 1000.0;
    let build_wall_s = t_total_w0.elapsed().as_secs_f64();

    let n_ef = efs.len();
    let mut maint_cpu_ms = 0.0f64;
    let mut maint_wall_s = 0.0f64;
    let mut query_cpu_ms = vec![0.0f64; n_ef];
    let mut query_wall_s = vec![0.0f64; n_ef];
    let mut recalls = vec![Vec::with_capacity(n_win); n_ef];
    let mut p50s = vec![Vec::with_capacity(n_win); n_ef];
    let mut p95s = vec![Vec::with_capacity(n_win); n_ef];
    let mut p99s = vec![Vec::with_capacity(n_win); n_ef];
    let mut unsupported_erase = false;

    for w in 0..n_win {
        let crud = &data.win_crud[w];
        let queries = &data.win_queries[w];

        let tm_w0 = Instant::now();
        let tm0 = process_cpu_ms();
        match cli.engine {
            EngineKind::UsearchRust => {
                let st = us_state.as_mut().unwrap();
                for ev in crud {
                    match ev.ev_type {
                        EV_INSERT => {
                            st.idx.add(ev.id, data.vec_slice(ev.vec_ref))?;
                        }
                        EV_ERASE => {
                            let cur = st.label_of.remove(&ev.id).unwrap_or(ev.id);
                            let _ = st.idx.remove(cur);
                        }
                        EV_UPDATE => {
                            let cur = *st.label_of.get(&ev.id).unwrap_or(&ev.id);
                            let _ = st.idx.remove(cur);
                            let nl = st.next_label;
                            st.next_label += 1;
                            st.synth_to_orig.push(ev.id);
                            st.label_of.insert(ev.id, nl);
                            st.idx.add(nl, data.vec_slice(ev.vec_ref))?;
                        }
                        _ => {}
                    }
                }
            }
            EngineKind::HnswRs => {
                let hn = hn_state.as_mut().unwrap();
                hn.set_searching_mode(false);
                let mut ins_items = Vec::new();
                for ev in crud {
                    match ev.ev_type {
                        EV_INSERT => {
                            if cli.threads > 1 {
                                ins_items.push((data.vec_slice(ev.vec_ref).to_vec(), ev.id as usize));
                            } else {
                                hn.insert((data.vec_slice(ev.vec_ref), ev.id as usize));
                            }
                        }
                        EV_ERASE | EV_UPDATE => {
                            unsupported_erase = true;
                        }
                        _ => {}
                    }
                }
                if cli.threads > 1 && !ins_items.is_empty() {
                    let refs: Vec<(&Vec<f32>, usize)> = ins_items.iter().map(|(v, id)| (v, *id)).collect();
                    pool_rt.install(|| hn.parallel_insert(&refs));
                }
            }
        }
        maint_cpu_ms += process_cpu_ms() - tm0;
        maint_wall_s += tm_w0.elapsed().as_secs_f64();

        let nq = queries.len();
        let q_off = data.q_offsets[w] * data.k;
        let gt_w = &data.gt_ids[q_off..q_off + nq * data.k];

        if let EngineKind::HnswRs = cli.engine {
            hn_state.as_mut().unwrap().set_searching_mode(true);
        }

        for (ei, &ef_val) in efs.iter().enumerate() {
            let tq_w0 = Instant::now();
            let tq0 = process_cpu_ms();
            let mut preds: Vec<Vec<i64>> = vec![Vec::new(); nq];
            let mut lat_us = Vec::with_capacity(nq);

            match cli.engine {
                EngineKind::UsearchRust => {
                    let st = us_state.as_ref().unwrap();
                    st.idx.change_expansion_search(ef_val);
                    if cli.batch_query && cli.threads > 1 {
                        let par_res: Vec<(f64, Vec<i64>)> = pool_rt.install(|| {
                            use rayon::prelude::*;
                            queries
                                .par_iter()
                                .map(|qe| {
                                    let t_q = Instant::now();
                                    let m = st.idx.search(data.vec_slice(qe.vec_ref), data.k).expect("search");
                                    let us = t_q.elapsed().as_secs_f64() * 1e6;
                                    let row: Vec<i64> = m.keys.into_iter().take(data.k).map(|k| st.decode_key(k)).collect();
                                    (us, row)
                                })
                                .collect()
                        });
                        for (qi, (us, row)) in par_res.into_iter().enumerate() {
                            lat_us.push(us);
                            preds[qi] = row;
                        }
                    } else {
                        for (qi, qe) in queries.iter().enumerate() {
                            let tq_s = Instant::now();
                            let m = st.idx.search(data.vec_slice(qe.vec_ref), data.k)?;
                            lat_us.push(tq_s.elapsed().as_secs_f64() * 1e6);
                            preds[qi] = m.keys.into_iter().take(data.k).map(|k| st.decode_key(k)).collect();
                        }
                    }
                }
                EngineKind::HnswRs => {
                    let hn_ref: &Hnsw<'static, f32, DistL2> = hn_state.as_ref().unwrap();
                    if cli.batch_query && cli.threads > 1 {
                        let par_res: Vec<(f64, Vec<i64>)> = pool_rt.install(|| {
                            use rayon::prelude::*;
                            queries
                                .par_iter()
                                .map(|qe| {
                                    let t_q = Instant::now();
                                    let nbrs = hn_ref.search(data.vec_slice(qe.vec_ref), data.k, ef_val);
                                    let us = t_q.elapsed().as_secs_f64() * 1e6;
                                    let row: Vec<i64> = nbrs.into_iter().take(data.k).map(|n| n.d_id as i64).collect();
                                    (us, row)
                                })
                                .collect()
                        });
                        for (qi, (us, row)) in par_res.into_iter().enumerate() {
                            lat_us.push(us);
                            preds[qi] = row;
                        }
                    } else {
                        for (qi, qe) in queries.iter().enumerate() {
                            let tq_s = Instant::now();
                            let nbrs = hn_ref.search(data.vec_slice(qe.vec_ref), data.k, ef_val);
                            lat_us.push(tq_s.elapsed().as_secs_f64() * 1e6);
                            preds[qi] = nbrs.into_iter().take(data.k).map(|n| n.d_id as i64).collect();
                        }
                    }
                }
            }
            query_cpu_ms[ei] += process_cpu_ms() - tq0;
            query_wall_s[ei] += tq_w0.elapsed().as_secs_f64();

            recalls[ei].push(mean_recall_at_k(&preds, gt_w, data.k));
            lat_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if let Some(v) = percentile_sorted(&lat_us, 50.0) {
                p50s[ei].push(v);
            }
            if let Some(v) = percentile_sorted(&lat_us, 95.0) {
                p95s[ei].push(v);
            }
            if let Some(v) = percentile_sorted(&lat_us, 99.0) {
                p99s[ei].push(v);
            }
        }
    }

    Ok(build_results(
        cli,
        "qbam",
        &cli.trace,
        n_win,
        unsupported_erase,
        efs,
        build_cpu_s,
        build_wall_s,
        maint_cpu_ms,
        maint_wall_s,
        &query_cpu_ms,
        &query_wall_s,
        &recalls,
        &p50s,
        &p95s,
        &p99s,
        load_start,
    ))
}

fn run_bigann(cli: &Cli, pool_rt: &rayon::ThreadPool, efs: &[usize]) -> Result<Vec<BenchResult>> {
    let src_dir = cli.src_dir.as_ref().expect("--src-dir required in bigann mode");
    let load_start = read_loadavg_1m();
    let t_total_w0 = Instant::now();
    let data = BigannWorkload::load(src_dir, &cli.data_dir)?;
    let n_win = cli.max_windows.map(|m| m.min(data.n_windows)).unwrap_or(data.n_windows);

    let t_build_c0 = process_cpu_ms();
    let mut us_state: Option<UsearchState> = None;
    let mut hn_state: Option<Hnsw<'static, f32, DistL2>> = None;

    let cap = data.init.len() + data.n_inserts_total + 100_000;
    match cli.engine {
        EngineKind::UsearchRust => {
            let st = UsearchState::new(data.dim, cli.m, cli.ef_construction, efs[0], cap, cli.threads, data.max_nonquery_id)?;
            if cli.threads > 1 {
                pool_rt.install(|| -> Result<()> {
                    use rayon::prelude::*;
                    data.init.par_iter().try_for_each_init(
                        || vec![0.0f32; data.dim],
                        |buf, rec| -> Result<()> {
                            data.fill_vec_f32(rec.src, rec.row, buf);
                            st.idx.add(rec.id, buf)?;
                            Ok(())
                        },
                    )
                })?;
            } else {
                let mut buf = vec![0.0f32; data.dim];
                for rec in &data.init {
                    data.fill_vec_f32(rec.src, rec.row, &mut buf);
                    st.idx.add(rec.id, &buf)?;
                }
            }
            us_state = Some(st);
        }
        EngineKind::HnswRs => {
            let hn = Hnsw::<f32, DistL2>::new(cli.m, cap, 16, cli.ef_construction, DistL2 {});
            let chunk = 250_000usize;
            if cli.threads > 1 {
                for start in (0..data.init.len()).step_by(chunk) {
                    let end = (start + chunk).min(data.init.len());
                    let items: Vec<(Vec<f32>, usize)> = data.init[start..end]
                        .iter()
                        .map(|r| {
                            let mut v = vec![0.0f32; data.dim];
                            data.fill_vec_f32(r.src, r.row, &mut v);
                            (v, r.id as usize)
                        })
                        .collect();
                    let refs: Vec<(&Vec<f32>, usize)> = items.iter().map(|(v, id)| (v, *id)).collect();
                    pool_rt.install(|| hn.parallel_insert(&refs));
                }
            } else {
                let mut buf = vec![0.0f32; data.dim];
                for rec in &data.init {
                    data.fill_vec_f32(rec.src, rec.row, &mut buf);
                    hn.insert((&buf, rec.id as usize));
                }
            }
            hn_state = Some(hn);
        }
    }

    let build_cpu_s = (process_cpu_ms() - t_build_c0) / 1000.0;
    let build_wall_s = t_total_w0.elapsed().as_secs_f64();

    let n_ef = efs.len();
    let mut maint_cpu_ms = 0.0f64;
    let mut maint_wall_s = 0.0f64;
    let mut query_cpu_ms = vec![0.0f64; n_ef];
    let mut query_wall_s = vec![0.0f64; n_ef];
    let mut recalls = vec![Vec::with_capacity(n_win); n_ef];
    let mut p50s = vec![Vec::with_capacity(n_win); n_ef];
    let mut p95s = vec![Vec::with_capacity(n_win); n_ef];
    let mut p99s = vec![Vec::with_capacity(n_win); n_ef];
    let mut unsupported_erase = false;
    let mut q_buf = vec![0.0f32; data.dim];

    for w in 0..n_win {
        let crud = &data.win_crud[w];
        let tm_w0 = Instant::now();
        let tm0 = process_cpu_ms();
        match cli.engine {
            EngineKind::UsearchRust => {
                let st = us_state.as_mut().unwrap();
                let mut buf = vec![0.0f32; data.dim];
                let mut ins_ev = Vec::new();
                for ev in crud {
                    match ev.ev_type {
                        EV_ERASE => {
                            let cur = st.label_of.remove(&ev.id).unwrap_or(ev.id);
                            let _ = st.idx.remove(cur);
                        }
                        EV_INSERT => ins_ev.push(*ev),
                        _ => {}
                    }
                }
                if cli.threads > 1 && !ins_ev.is_empty() {
                    pool_rt.install(|| -> Result<()> {
                        use rayon::prelude::*;
                        ins_ev.par_iter().try_for_each_init(
                            || vec![0.0f32; data.dim],
                            |b, ev| -> Result<()> {
                                data.fill_vec_f32(ev.src, ev.row, b);
                                st.idx.add(ev.id, b)?;
                                Ok(())
                            },
                        )
                    })?;
                } else {
                    for ev in &ins_ev {
                        data.fill_vec_f32(ev.src, ev.row, &mut buf);
                        st.idx.add(ev.id, &buf)?;
                    }
                }
            }
            EngineKind::HnswRs => {
                let hn = hn_state.as_mut().unwrap();
                hn.set_searching_mode(false);
                let mut ins_items = Vec::new();
                for ev in crud {
                    match ev.ev_type {
                        EV_INSERT => {
                            let mut v = vec![0.0f32; data.dim];
                            data.fill_vec_f32(ev.src, ev.row, &mut v);
                            if cli.threads > 1 {
                                ins_items.push((v, ev.id as usize));
                            } else {
                                hn.insert((&v, ev.id as usize));
                            }
                        }
                        EV_ERASE | EV_UPDATE => {
                            unsupported_erase = true;
                        }
                        _ => {}
                    }
                }
                if cli.threads > 1 && !ins_items.is_empty() {
                    let refs: Vec<(&Vec<f32>, usize)> = ins_items.iter().map(|(v, id)| (v, *id)).collect();
                    pool_rt.install(|| hn.parallel_insert(&refs));
                }
            }
        }
        maint_cpu_ms += process_cpu_ms() - tm0;
        maint_wall_s += tm_w0.elapsed().as_secs_f64();

        let nq = data.nq_per_win;
        let q_row_slice = &data.q_rows[w * nq..(w + 1) * nq];
        let g_off = w * nq * data.k;
        let gt_w = &data.gt_ids[g_off..g_off + nq * data.k];

        if let EngineKind::HnswRs = cli.engine {
            hn_state.as_mut().unwrap().set_searching_mode(true);
        }

        for (ei, &ef_val) in efs.iter().enumerate() {
            let tq_w0 = Instant::now();
            let tq0 = process_cpu_ms();
            let mut preds: Vec<Vec<i64>> = vec![Vec::new(); nq];
            let mut lat_us = Vec::with_capacity(nq);

            match cli.engine {
                EngineKind::UsearchRust => {
                    let st = us_state.as_ref().unwrap();
                    st.idx.change_expansion_search(ef_val);
                    if cli.batch_query && cli.threads > 1 {
                        let par_res: Vec<(f64, Vec<i64>)> = pool_rt.install(|| {
                            use rayon::prelude::*;
                            q_row_slice
                                .par_iter()
                                .map_init(
                                    || vec![0.0f32; data.dim],
                                    |qb, &qr| {
                                        data.fill_query_f32(qr, qb);
                                        let t_q = Instant::now();
                                        let m = st.idx.search(qb, data.k).expect("search");
                                        let us = t_q.elapsed().as_secs_f64() * 1e6;
                                        let row: Vec<i64> = m.keys.into_iter().take(data.k).map(|k| st.decode_key(k)).collect();
                                        (us, row)
                                    },
                                )
                                .collect()
                        });
                        for (qi, (us, row)) in par_res.into_iter().enumerate() {
                            lat_us.push(us);
                            preds[qi] = row;
                        }
                    } else {
                        for (qi, &qr) in q_row_slice.iter().enumerate() {
                            data.fill_query_f32(qr, &mut q_buf);
                            let tq_s = Instant::now();
                            let m = st.idx.search(&q_buf, data.k)?;
                            lat_us.push(tq_s.elapsed().as_secs_f64() * 1e6);
                            preds[qi] = m.keys.into_iter().take(data.k).map(|k| st.decode_key(k)).collect();
                        }
                    }
                }
                EngineKind::HnswRs => {
                    let hn_ref: &Hnsw<'static, f32, DistL2> = hn_state.as_ref().unwrap();
                    if cli.batch_query && cli.threads > 1 {
                        let par_res: Vec<(f64, Vec<i64>)> = pool_rt.install(|| {
                            use rayon::prelude::*;
                            q_row_slice
                                .par_iter()
                                .map_init(
                                    || vec![0.0f32; data.dim],
                                    |qb, &qr| {
                                        data.fill_query_f32(qr, qb);
                                        let t_q = Instant::now();
                                        let nbrs = hn_ref.search(qb, data.k, ef_val);
                                        let us = t_q.elapsed().as_secs_f64() * 1e6;
                                        let row: Vec<i64> = nbrs.into_iter().take(data.k).map(|n| n.d_id as i64).collect();
                                        (us, row)
                                    },
                                )
                                .collect()
                        });
                        for (qi, (us, row)) in par_res.into_iter().enumerate() {
                            lat_us.push(us);
                            preds[qi] = row;
                        }
                    } else {
                        for (qi, &qr) in q_row_slice.iter().enumerate() {
                            data.fill_query_f32(qr, &mut q_buf);
                            let tq_s = Instant::now();
                            let nbrs = hn_ref.search(&q_buf, data.k, ef_val);
                            lat_us.push(tq_s.elapsed().as_secs_f64() * 1e6);
                            preds[qi] = nbrs.into_iter().take(data.k).map(|n| n.d_id as i64).collect();
                        }
                    }
                }
            }
            query_cpu_ms[ei] += process_cpu_ms() - tq0;
            query_wall_s[ei] += tq_w0.elapsed().as_secs_f64();

            recalls[ei].push(mean_recall_at_k(&preds, gt_w, data.k));
            lat_us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            if let Some(v) = percentile_sorted(&lat_us, 50.0) {
                p50s[ei].push(v);
            }
            if let Some(v) = percentile_sorted(&lat_us, 95.0) {
                p95s[ei].push(v);
            }
            if let Some(v) = percentile_sorted(&lat_us, 99.0) {
                p99s[ei].push(v);
            }
        }
    }

    let wl_name = cli
        .data_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("bigann")
        .to_string();

    Ok(build_results(
        cli,
        "bigann",
        &wl_name,
        n_win,
        unsupported_erase,
        efs,
        build_cpu_s,
        build_wall_s,
        maint_cpu_ms,
        maint_wall_s,
        &query_cpu_ms,
        &query_wall_s,
        &recalls,
        &p50s,
        &p95s,
        &p99s,
        load_start,
    ))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let efs = parse_ef_list(&cli)?;
    let pool_rt = configure_threads(cli.threads, &cli.cores)?;
    let results = match cli.mode {
        Mode::Qbam => run_qbam(&cli, &pool_rt, &efs)?,
        Mode::Bigann => run_bigann(&cli, &pool_rt, &efs)?,
    };
    let s = if cli.ef_list.is_some() {
        serde_json::to_string_pretty(&results)?
    } else {
        serde_json::to_string_pretty(&results[0])?
    };
    if let Some(op) = &cli.out_json {
        if let Some(p) = op.parent() {
            std::fs::create_dir_all(p)?;
        }
        std::fs::write(op, &s)?;
    }
    println!("{s}");
    Ok(())
}
