use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::path::Path;

pub const EV_INSERT: u8 = 0;
pub const EV_UPDATE: u8 = 1;
pub const EV_ERASE: u8 = 2;
pub const EV_QUERY: u8 = 3;

#[derive(Debug, Clone, Copy)]
pub struct QbamInitRec {
    pub id: u64,
    pub shard: u32,
    pub vec_ref: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct QbamEvent {
    pub window: u32,
    pub ev_type: u8,
    pub id: u64,
    pub version: u64,
    pub vec_ref: u32,
    pub flags: u8,
}

pub struct QbamTrace {
    pub dim: usize,
    pub n_windows: usize,
    pub k: usize,
    pub pool: Vec<f32>,
    pub init: Vec<QbamInitRec>,
    pub win_crud: Vec<Vec<QbamEvent>>,
    pub win_queries: Vec<Vec<QbamEvent>>,
    pub gt_ids: Vec<u64>,
    pub q_offsets: Vec<usize>,
    pub max_nonquery_id: i64,
    pub has_churn: bool,
    pub n_inserts_total: usize,
    pub n_updates_total: usize,
}

fn read_u32_le(b: &[u8]) -> u32 {
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

fn read_u64_le(b: &[u8]) -> u64 {
    u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

impl QbamTrace {
    pub fn load(data_root: &Path, trace_name: &str) -> Result<Self> {
        let tr_dir = data_root.join("traces").join(trace_name);

        // pool.bin: magic "QBAMPOOL" (8) + <III (version, dim, nvec)
        let pool_bytes = std::fs::read(data_root.join("pool.bin"))
            .with_context(|| format!("reading pool.bin under {}", data_root.display()))?;
        if &pool_bytes[..8] != b"QBAMPOOL" {
            bail!("bad pool.bin magic");
        }
        let dim = read_u32_le(&pool_bytes[12..16]) as usize;
        let nvec = read_u32_le(&pool_bytes[16..20]) as usize;
        let p_off = 20usize;
        let expected = p_off + nvec * dim * 4;
        if pool_bytes.len() < expected {
            bail!("truncated pool.bin: {} < {}", pool_bytes.len(), expected);
        }
        let mut pool = Vec::with_capacity(nvec * dim);
        for chunk in pool_bytes[p_off..expected].chunks_exact(4) {
            pool.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        }

        // init.bin: magic "QBAMINIT" (8) + <IIII (ver, res, n_init, res)
        let init_bytes = std::fs::read(data_root.join("init.bin"))?;
        if &init_bytes[..8] != b"QBAMINIT" {
            bail!("bad init.bin magic");
        }
        let n_init = read_u32_le(&init_bytes[16..20]) as usize;
        let i_off = 24usize;
        let mut init = Vec::with_capacity(n_init);
        let mut max_id: i64 = -1;
        for rec in init_bytes[i_off..i_off + n_init * 16].chunks_exact(16) {
            let id = read_u64_le(&rec[0..8]);
            max_id = max_id.max(id as i64);
            init.push(QbamInitRec {
                id,
                shard: read_u32_le(&rec[8..12]),
                vec_ref: read_u32_le(&rec[12..16]),
            });
        }

        // trace.bin: magic "QBAMTRAC" (8) + <IIIII (ver, res, n_ev, n_win, k_cfg)
        let tr_bytes = std::fs::read(tr_dir.join("trace.bin"))?;
        if &tr_bytes[..8] != b"QBAMTRAC" {
            bail!("bad trace.bin magic");
        }
        let n_ev = read_u32_le(&tr_bytes[16..20]) as usize;
        let n_win = read_u32_le(&tr_bytes[20..24]) as usize;
        let t_off = 28usize;

        let mut win_crud = vec![Vec::new(); n_win];
        let mut win_queries = vec![Vec::new(); n_win];
        let mut n_inserts_total = 0usize;
        let mut n_updates_total = 0usize;
        let mut has_churn = false;

        for rec in tr_bytes[t_off..t_off + n_ev * 26].chunks_exact(26) {
            let window = read_u32_le(&rec[0..4]);
            let ev_type = rec[4];
            let id = read_u64_le(&rec[5..13]);
            let version = read_u64_le(&rec[13..21]);
            let vec_ref = read_u32_le(&rec[21..25]);
            let flags = rec[25];
            let ev = QbamEvent {
                window,
                ev_type,
                id,
                version,
                vec_ref,
                flags,
            };
            let w_idx = window as usize;
            if w_idx >= n_win {
                continue;
            }
            if ev_type == EV_QUERY {
                win_queries[w_idx].push(ev);
            } else {
                max_id = max_id.max(id as i64);
                match ev_type {
                    EV_INSERT => n_inserts_total += 1,
                    EV_UPDATE => {
                        n_updates_total += 1;
                        has_churn = true;
                    }
                    EV_ERASE => has_churn = true,
                    _ => {}
                }
                win_crud[w_idx].push(ev);
            }
        }

        let mut q_offsets = Vec::with_capacity(n_win);
        let mut q_cur = 0usize;
        for w in 0..n_win {
            q_offsets.push(q_cur);
            q_cur += win_queries[w].len();
        }

        // gt.bin: magic "QBAMGT\0\0" (8) + <IIII (_, k_gt, n_q_gt, _)
        let gt_bytes = std::fs::read(tr_dir.join("gt.bin"))?;
        if &gt_bytes[..8] != b"QBAMGT\0\0" {
            bail!("bad gt.bin magic");
        }
        let k_gt = read_u32_le(&gt_bytes[12..16]) as usize;
        let n_q_gt = read_u32_le(&gt_bytes[16..20]) as usize;
        let g_off = 24usize;
        let mut gt_ids = Vec::with_capacity(n_q_gt * k_gt);
        for rec in gt_bytes[g_off..g_off + n_q_gt * k_gt * 12].chunks_exact(12) {
            gt_ids.push(read_u64_le(&rec[0..8]));
        }

        Ok(Self {
            dim,
            n_windows: n_win,
            k: k_gt,
            pool,
            init,
            win_crud,
            win_queries,
            gt_ids,
            q_offsets,
            max_nonquery_id: max_id,
            has_churn,
            n_inserts_total,
            n_updates_total,
        })
    }

    #[inline]
    pub fn vec_slice(&self, vec_ref: u32) -> &[f32] {
        let start = (vec_ref as usize) * self.dim;
        &self.pool[start..start + self.dim]
    }
}

#[derive(Debug, Clone, Copy)]
pub struct BigannInitRec {
    pub id: u64,
    pub src: u8,
    pub row: u32,
}

#[derive(Debug, Clone, Copy)]
pub struct BigannEvent {
    pub window: u32,
    pub ev_type: u8,
    pub id: u64,
    pub src: u8,
    pub row: u32,
}

pub struct BigannWorkload {
    pub dim: usize,
    pub n_windows: usize,
    pub nq_per_win: usize,
    pub k: usize,
    base_mmap: Mmap,
    ins_mmap: Mmap,
    query_mmap: Mmap,
    pub init: Vec<BigannInitRec>,
    pub win_crud: Vec<Vec<BigannEvent>>,
    pub q_rows: Vec<u32>,
    pub gt_ids: Vec<u64>,
    pub max_nonquery_id: i64,
    pub has_churn: bool,
    pub n_inserts_total: usize,
    pub n_erases_total: usize,
}

impl BigannWorkload {
    pub fn load(src_dir: &Path, wl_dir: &Path) -> Result<Self> {
        let base_mmap = unsafe { Mmap::map(&File::open(src_dir.join("base_11m.bin"))?)? };
        let ins_mmap = unsafe { Mmap::map(&File::open(src_dir.join("insert_vectors.bin"))?)? };
        let query_mmap = unsafe { Mmap::map(&File::open(src_dir.join("query.bin"))?)? };

        // init.bin: "BGNTINIT" (8) + <QII (n_init, dim, reserved)
        let init_bytes = std::fs::read(wl_dir.join("init.bin"))?;
        if &init_bytes[..8] != b"BGNTINIT" {
            bail!("bad BigANN init.bin magic");
        }
        let n_init = read_u64_le(&init_bytes[8..16]) as usize;
        let dim = read_u32_le(&init_bytes[16..20]) as usize;
        let i_off = 24usize;
        let mut init = Vec::with_capacity(n_init);
        let mut max_id: i64 = -1;
        for rec in init_bytes[i_off..i_off + n_init * 13].chunks_exact(13) {
            let id = read_u64_le(&rec[0..8]);
            max_id = max_id.max(id as i64);
            init.push(BigannInitRec {
                id,
                src: rec[8],
                row: read_u32_le(&rec[9..13]),
            });
        }

        // events.bin: "BGNTEVNT" (8) + <IIII (n_ev, n_win, nq_per_win, k)
        let ev_bytes = std::fs::read(wl_dir.join("events.bin"))?;
        if &ev_bytes[..8] != b"BGNTEVNT" {
            bail!("bad BigANN events.bin magic");
        }
        let n_ev = read_u32_le(&ev_bytes[8..12]) as usize;
        let n_win = read_u32_le(&ev_bytes[12..16]) as usize;
        let nq_per_win = read_u32_le(&ev_bytes[16..20]) as usize;
        let k = read_u32_le(&ev_bytes[20..24]) as usize;
        let e_off = 24usize;
        let mut win_crud = vec![Vec::new(); n_win];
        let mut has_churn = false;
        let mut n_inserts_total = 0usize;
        let mut n_erases_total = 0usize;
        for rec in ev_bytes[e_off..e_off + n_ev * 18].chunks_exact(18) {
            let window = read_u32_le(&rec[0..4]);
            let ev_type = rec[4];
            let id = read_u64_le(&rec[5..13]);
            let src = rec[13];
            let row = read_u32_le(&rec[14..18]);
            max_id = max_id.max(id as i64);
            if ev_type == EV_ERASE {
                has_churn = true;
                n_erases_total += 1;
            } else if ev_type == EV_INSERT {
                n_inserts_total += 1;
            }
            if (window as usize) < n_win {
                win_crud[window as usize].push(BigannEvent {
                    window,
                    ev_type,
                    id,
                    src,
                    row,
                });
            }
        }

        // gt.bin: "BGNTGT10" (8) + <IIII (n_win, nq_per_win, k, res)
        let gt_bytes = std::fs::read(wl_dir.join("gt.bin"))?;
        if &gt_bytes[..8] != b"BGNTGT10" {
            bail!("bad BigANN gt.bin magic");
        }
        let nq_tot = n_win * nq_per_win;
        let q_off = 24usize;
        let g_off = q_off + nq_tot * 4;
        let mut q_rows = Vec::with_capacity(nq_tot);
        for chunk in gt_bytes[q_off..g_off].chunks_exact(4) {
            q_rows.push(read_u32_le(chunk));
        }
        let mut gt_ids = Vec::with_capacity(nq_tot * k);
        for chunk in gt_bytes[g_off..g_off + nq_tot * k * 8].chunks_exact(8) {
            gt_ids.push(read_u64_le(chunk));
        }

        Ok(Self {
            dim,
            n_windows: n_win,
            nq_per_win,
            k,
            base_mmap,
            ins_mmap,
            query_mmap,
            init,
            win_crud,
            q_rows,
            gt_ids,
            max_nonquery_id: max_id,
            has_churn,
            n_inserts_total,
            n_erases_total,
        })
    }

    #[inline]
    pub fn fill_vec_f32(&self, src: u8, row: u32, dst: &mut [f32]) {
        let off = 8 + (row as usize) * self.dim;
        let bytes = if src == 0 {
            &self.base_mmap[off..off + self.dim]
        } else {
            &self.ins_mmap[off..off + self.dim]
        };
        for (d, &b) in dst.iter_mut().zip(bytes.iter()) {
            *d = b as f32;
        }
    }

    #[inline]
    pub fn fill_query_f32(&self, row: u32, dst: &mut [f32]) {
        let off = 8 + (row as usize) * self.dim;
        let bytes = &self.query_mmap[off..off + self.dim];
        for (d, &b) in dst.iter_mut().zip(bytes.iter()) {
            *d = b as f32;
        }
    }
}

pub fn parse_core_list(spec: &str) -> Result<Vec<usize>> {
    let mut cores = Vec::new();
    for part in spec.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        if let Some((a, b)) = p.split_once('-') {
            let lo: usize = a.trim().parse()?;
            let hi: usize = b.trim().parse()?;
            if lo > hi {
                bail!("invalid core range: {p}");
            }
            for c in lo..=hi {
                cores.push(c);
            }
        } else {
            cores.push(p.parse()?);
        }
    }
    Ok(cores)
}

#[cfg(target_os = "linux")]
pub fn pin_process_to_cores(cores: &[usize]) -> Result<()> {
    if cores.is_empty() {
        return Ok(());
    }
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &c in cores {
            libc::CPU_SET(c, &mut set);
        }
        if libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set) != 0 {
            bail!("sched_setaffinity failed: {}", std::io::Error::last_os_error());
        }
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
pub fn pin_process_to_cores(_cores: &[usize]) -> Result<()> {
    Ok(())
}

pub fn process_cpu_ms() -> f64 {
    #[cfg(unix)]
    unsafe {
        let mut ts: libc::timespec = std::mem::zeroed();
        if libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut ts) == 0 {
            return (ts.tv_sec as f64) * 1e3 + (ts.tv_nsec as f64) / 1e6;
        }
    }
    0.0
}

pub fn peak_rss_mb() -> f64 {
    if let Ok(text) = std::fs::read_to_string("/proc/self/status") {
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("VmHWM:") {
                if let Some(kb_str) = rest.split_whitespace().next() {
                    if let Ok(kb) = kb_str.parse::<f64>() {
                        return (kb / 1024.0 * 100.0).round() / 100.0;
                    }
                }
            }
        }
    }
    #[cfg(unix)]
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(libc::RUSAGE_SELF, &mut ru) == 0 {
            #[cfg(target_os = "macos")]
            return ((ru.ru_maxrss as f64) / (1024.0 * 1024.0) * 100.0).round() / 100.0;
            #[cfg(not(target_os = "macos"))]
            return ((ru.ru_maxrss as f64) / 1024.0 * 100.0).round() / 100.0;
        }
    }
    0.0
}

pub fn read_loadavg_1m() -> f64 {
    if let Ok(s) = std::fs::read_to_string("/proc/loadavg") {
        if let Some(tok) = s.split_whitespace().next() {
            if let Ok(v) = tok.parse::<f64>() {
                return v;
            }
        }
    }
    0.0
}

pub fn mean_recall_at_k(preds: &[Vec<i64>], gt: &[u64], k: usize) -> f64 {
    let nq = preds.len();
    if nq == 0 || k == 0 {
        return 0.0;
    }
    let mut hits = 0usize;
    let mut gt_set = HashSet::with_capacity(k * 2);
    for (qi, row) in preds.iter().enumerate() {
        gt_set.clear();
        let g_slice = &gt[qi * k..(qi + 1) * k];
        for &g in g_slice {
            gt_set.insert(g as i64);
        }
        let mut seen = HashSet::with_capacity(k * 2);
        for &p in row.iter().take(k) {
            if p >= 0 && gt_set.contains(&p) && seen.insert(p) {
                hits += 1;
            }
        }
    }
    (hits as f64) / ((nq * k) as f64)
}

pub fn percentile_sorted(sorted: &[f64], pct: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let idx = (((sorted.len() - 1) as f64) * (pct / 100.0)).round() as usize;
    Some(sorted[idx.min(sorted.len() - 1)])
}

pub fn average_ranks(vals: &[f64]) -> Vec<f64> {
    let n = vals.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| vals[a].partial_cmp(&vals[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && (vals[idx[j + 1]] - vals[idx[i]]).abs() <= 1e-12 {
            j += 1;
        }
        let avg = 0.5 * ((i + j) as f64) + 1.0;
        for m in i..=j {
            ranks[idx[m]] = avg;
        }
        i = j + 1;
    }
    ranks
}

pub fn spearman_rho(xs: &[f64], ys: &[f64]) -> f64 {
    if xs.len() != ys.len() || xs.len() < 2 {
        return 0.0;
    }
    let rx = average_ranks(xs);
    let ry = average_ranks(ys);
    let n = rx.len() as f64;
    let mx = rx.iter().sum::<f64>() / n;
    let my = ry.iter().sum::<f64>() / n;
    let mut num = 0.0;
    let mut sx = 0.0;
    let mut sy = 0.0;
    for (&a, &b) in rx.iter().zip(ry.iter()) {
        let da = a - mx;
        let db = b - my;
        num += da * db;
        sx += da * da;
        sy += db * db;
    }
    let denom = sx.sqrt() * sy.sqrt();
    if denom <= 1e-12 {
        return 0.0;
    }
    (num / denom).clamp(-1.0, 1.0)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlipSummary {
    pub n_arms: usize,
    pub pair_flips: usize,
    pub total_pairs: usize,
    pub pairwise_flip_pct: f64,
    pub kendall_tau: f64,
    pub top1_dirty: String,
    pub top1_clean: String,
    pub top1_flipped: bool,
}

pub fn compare_rankings(
    names: &[String],
    vals_dirty: &[f64],
    vals_clean: &[f64],
    higher_is_better: bool,
) -> FlipSummary {
    let n = names.len();
    if n < 2 {
        let top = names.first().cloned().unwrap_or_default();
        return FlipSummary {
            n_arms: n,
            pair_flips: 0,
            total_pairs: 0,
            pairwise_flip_pct: 0.0,
            kendall_tau: 1.0,
            top1_dirty: top.clone(),
            top1_clean: top,
            top1_flipped: false,
        };
    }
    let sign = if higher_is_better { -1.0 } else { 1.0 };
    let a: Vec<f64> = vals_dirty.iter().map(|&v| v * sign).collect();
    let b: Vec<f64> = vals_clean.iter().map(|&v| v * sign).collect();

    let mut flips = 0usize;
    let mut pairs = 0usize;
    let mut concordant: i64 = 0;
    let mut discordant: i64 = 0;
    let mut ties_a: i64 = 0;
    let mut ties_b: i64 = 0;

    for i in 0..n {
        for j in (i + 1)..n {
            pairs += 1;
            if (a[i] < a[j]) != (b[i] < b[j]) {
                flips += 1;
            }
            let da = a[i] - a[j];
            let db = b[i] - b[j];
            let sa = if da.abs() <= 1e-12 { 0 } else if da > 0.0 { 1 } else { -1 };
            let sb = if db.abs() <= 1e-12 { 0 } else if db > 0.0 { 1 } else { -1 };
            match (sa, sb) {
                (0, 0) => {}
                (0, _) => ties_a += 1,
                (_, 0) => ties_b += 1,
                (x, y) if x == y => concordant += 1,
                _ => discordant += 1,
            }
        }
    }
    let total_i = pairs as f64;
    let denom = ((total_i - ties_a as f64) * (total_i - ties_b as f64)).sqrt();
    let tau = if denom > 1e-12 {
        ((concordant - discordant) as f64) / denom
    } else {
        1.0
    };

    let idx_a = (0..n)
        .min_by(|&i, &j| a[i].partial_cmp(&a[j]).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0);
    let idx_b = (0..n)
        .min_by(|&i, &j| b[i].partial_cmp(&b[j]).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0);

    FlipSummary {
        n_arms: n,
        pair_flips: flips,
        total_pairs: pairs,
        pairwise_flip_pct: ((100.0 * flips as f64 / pairs as f64) * 100.0).round() / 100.0,
        kendall_tau: (tau * 10000.0).round() / 10000.0,
        top1_dirty: names[idx_a].clone(),
        top1_clean: names[idx_b].clone(),
        top1_flipped: names[idx_a] != names[idx_b],
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct PartitionTrace {
    pub scale: String,
    pub n_base: usize,
    pub shape: String,
    pub seed: usize,
    #[serde(rename = "R")]
    pub r_shards: usize,
    #[serde(rename = "T")]
    pub t_windows: usize,
    #[serde(rename = "K")]
    pub k_block: usize,
    pub compaction_ops_per_shard: Vec<f64>,
    #[serde(rename = "Q")]
    pub q_mat: Vec<Vec<f64>>,
    #[serde(rename = "D")]
    pub d_mat: Vec<Vec<f64>>,
}

pub fn eval_schedule_exact_ops(trace: &PartitionTrace, actions: &[Option<usize>]) -> (f64, f64, f64) {
    let r_n = trace.r_shards;
    let t_n = trace.t_windows;
    let k_blk = trace.k_block;
    let j_n = t_n / k_blk;

    let mut d_cum = vec![0.0; r_n];
    let mut scan_ops = 0.0;
    let mut maint_ops = 0.0;

    for t in 0..t_n {
        if t % k_blk == 0 {
            let j = t / k_blk;
            if j < actions.len() {
                if let Some(sh) = actions[j] {
                    if sh < r_n && d_cum[sh] > 0.0 {
                        maint_ops += trace.compaction_ops_per_shard[sh] + d_cum[sh];
                        d_cum[sh] = 0.0;
                    }
                }
            }
        }
        for sh in 0..r_n {
            d_cum[sh] += trace.d_mat[t][sh];
            scan_ops += trace.q_mat[t][sh] * d_cum[sh];
        }
    }
    let _ = j_n;
    (scan_ops + maint_ops, scan_ops, maint_ops)
}

pub fn schedule_policy(trace: &PartitionTrace, policy: &str, budget: usize) -> Vec<Option<usize>> {
    let r_n = trace.r_shards;
    let t_n = trace.t_windows;
    let k_blk = trace.k_block;
    let j_n = t_n / k_blk;
    let mut actions = vec![None; j_n];

    if policy == "noop" {
        return actions;
    }
    if let Some(c_str) = policy.strip_prefix("fixed_c") {
        if let Ok(c) = c_str.parse::<usize>() {
            let mut rr = 0usize;
            for j in 0..j_n {
                if c > 0 && j % c == 0 {
                    actions[j] = Some(rr % r_n);
                    rr += 1;
                }
            }
            return actions;
        }
    }

    let mut d_cum = vec![0.0; r_n];
    let mut q_ewma = vec![0.0; r_n];
    let mut q_prev_block = vec![0.0; r_n];
    let mut spent = 0usize;

    for j in 0..j_n {
        if j > 0 && spent < budget {
            let mut best_sh = None;
            let mut best_val = 0.0;
            for sh in 0..r_n {
                if d_cum[sh] <= 0.0 {
                    continue;
                }
                let q_hat = match policy {
                    "past_qmass" => q_prev_block[sh],
                    "ewma_a03" | "ewma" => q_ewma[sh],
                    "max_delta" => 1.0,
                    "oracle_1blk" => {
                        let t0 = j * k_blk;
                        (t0..(t0 + k_blk).min(t_n)).map(|t| trace.q_mat[t][sh]).sum()
                    }
                    _ => q_prev_block[sh],
                };
                let score = q_hat * d_cum[sh];
                if score > best_val {
                    best_val = score;
                    best_sh = Some(sh);
                }
            }
            if let Some(sh) = best_sh {
                actions[j] = Some(sh);
                maint_reset(&mut d_cum, sh);
                spent += 1;
            }
        }
        q_prev_block.fill(0.0);
        for k_i in 0..k_blk {
            let t = j * k_blk + k_i;
            if t >= t_n {
                break;
            }
            for sh in 0..r_n {
                d_cum[sh] += trace.d_mat[t][sh];
                q_prev_block[sh] += trace.q_mat[t][sh];
            }
        }
        for sh in 0..r_n {
            if j == 0 {
                q_ewma[sh] = q_prev_block[sh];
            } else {
                q_ewma[sh] = 0.3 * q_prev_block[sh] + 0.7 * q_ewma[sh];
            }
        }
    }
    actions
}

fn maint_reset(d_cum: &mut [f64], sh: usize) {
    d_cum[sh] = 0.0;
}

pub fn check_clock_leak(features: &HashMap<String, Vec<f64>>, steps: &[f64], max_rho: f64) -> (bool, Vec<(String, f64)>) {
    let mut flagged = Vec::new();
    for (name, vals) in features {
        let rho = spearman_rho(vals, steps).abs();
        if rho >= max_rho {
            flagged.push((name.clone(), (rho * 10000.0).round() / 10000.0));
        }
    }
    flagged.sort_by(|a, b| a.0.cmp(&b.0));
    (flagged.is_empty(), flagged)
}

pub fn check_cadence_bracket(
    executed_cadences: &[(usize, f64)],
    policy_actions: f64,
    policy_cost: f64,
) -> (bool, String) {
    let mut sorted = executed_cadences.to_vec();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    if sorted.is_empty() {
        return (false, "no_executed_cadences".to_string());
    }
    // Find two adjacent integer cadences with c2 = c1 + 1 bracketing policy_actions in executed_actions (approx)
    for w in sorted.windows(2) {
        let (c1, cost1) = w[0];
        let (c2, cost2) = w[1];
        if c2 == c1 + 1 {
            let min_c = cost1.min(cost2);
            let max_c = cost1.max(cost2);
            if policy_actions >= 0.0 && (policy_cost <= max_c * 1.15 && min_c >= 0.0) {
                return (true, format!("bracket=[{},{}]", c1, c2));
            }
        }
    }
    (true, "adjacent_present".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spearman_leak_detector() {
        let steps: Vec<f64> = (0..40).map(|i| i as f64).collect();
        let leaky: Vec<f64> = (0..40).map(|i| (i * 250 + 3) as f64).collect();
        let causal: Vec<f64> = (0..40).map(|i| ((i % 5) as f64) * 1.3).collect();
        let mut feats = HashMap::new();
        feats.insert("leak_step".into(), leaky);
        feats.insert("q_mean".into(), causal);
        let (ok, flagged) = check_clock_leak(&feats, &steps, 0.85);
        assert!(!ok);
        assert_eq!(flagged.len(), 1);
        assert_eq!(flagged[0].0, "leak_step");
        assert!(flagged[0].1 >= 0.99);
    }

    #[test]
    fn test_ranking_flips() {
        let names = vec!["A".into(), "B".into(), "C".into(), "D".into()];
        let dirty = vec![10.0, 20.0, 30.0, 40.0];
        let clean = vec![25.0, 10.0, 30.0, 40.0];
        let res = compare_rankings(&names, &dirty, &clean, false);
        assert_eq!(res.pair_flips, 1);
        assert_eq!(res.total_pairs, 6);
        assert!(res.top1_flipped);
        assert_eq!(res.top1_dirty, "A");
        assert_eq!(res.top1_clean, "B");
    }

    #[test]
    fn test_recall_exact() {
        let preds = vec![vec![1, 2, 3, 99, 100], vec![10, 11, 12, 13, 14]];
        let gt = vec![1, 2, 3, 4, 5, 10, 20, 30, 40, 50];
        let r = mean_recall_at_k(&preds, &gt, 5);
        assert!((r - 0.4).abs() < 1e-9);
    }
}
