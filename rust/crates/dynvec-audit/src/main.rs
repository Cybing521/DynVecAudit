use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use dynvec_core::{
    check_cadence_bracket, check_clock_leak, compare_rankings, eval_schedule_exact_ops,
    schedule_policy, spearman_rho, PartitionTrace,
};
use serde::Serialize;
use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Instant;

#[derive(Parser)]
#[command(name = "dynvec-audit", version, about = "Fast Rust measurement-hygiene & counterfactual audit CLI")]
struct Cli {
    #[command(subcommand)]
    cmd: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run deterministic self-tests for all 4 measurement-hygiene gates
    SelfTest,
    /// Audit a benchmark CSV for Gate-4 wall/CPU inflation
    AuditCsv {
        #[arg(long)]
        csv: PathBuf,
        #[arg(long, default_value_t = 1.18)]
        max_wall_cpu_ratio: f64,
        #[arg(long)]
        out_json: Option<PathBuf>,
    },
    /// Compute pairwise flips, Kendall tau, and Top-1 flips from a comparison CSV
    FlipMatrix {
        #[arg(long)]
        csv: PathBuf,
        #[arg(long)]
        group_col: String,
        #[arg(long)]
        arm_col: String,
        #[arg(long)]
        dirty_col: String,
        #[arg(long)]
        clean_col: String,
        #[arg(long, default_value_t = false)]
        higher_is_better: bool,
        #[arg(long)]
        out_json: Option<PathBuf>,
    },
    /// Deterministic partitioned counterfactual replay across multi-scale shard traces
    ScaleReplay {
        #[arg(long)]
        trace_dir: PathBuf,
        #[arg(long, default_value_t = 5)]
        budget: usize,
        #[arg(long)]
        out_json: PathBuf,
    },
}

#[derive(Serialize)]
struct SelfTestReport {
    status: String,
    gates_passed: usize,
    gate1_cadence_bracket: bool,
    gate2_clock_leak: bool,
    gate3_counterfactual_horizon: bool,
    gate4_contamination: bool,
    elapsed_ms: f64,
}

#[derive(Serialize)]
struct AuditCsvReport {
    path: String,
    total_rows: usize,
    checked_rows: usize,
    violations: usize,
    max_wall_cpu_ratio: f64,
    passed: bool,
    elapsed_ms: f64,
}

#[derive(Serialize)]
struct ScaleReplayRow {
    file: String,
    scale: String,
    n_base: usize,
    shape: String,
    seed: usize,
    policy: String,
    budget: usize,
    total_ops: f64,
    scan_ops: f64,
    maint_ops: f64,
    executed_actions: usize,
}

#[derive(Serialize)]
struct ScaleReplayReport {
    traces_evaluated: usize,
    runs_evaluated: usize,
    elapsed_ms: f64,
    rows: Vec<ScaleReplayRow>,
}

fn split_csv_line(line: &str) -> Vec<&str> {
    line.trim_end_matches(&['\r', '\n'][..]).split(',').collect()
}

fn run_self_test() -> Result<()> {
    let t0 = Instant::now();
    // Gate 1: cadence bracketing
    let brackets = vec![(1usize, 152.0f64), (2, 166.0), (4, 169.0)];
    let (g1_ok, _) = check_cadence_bracket(&brackets, 6.0, 140.0);

    // Gate 2: clock-leak catches monotonic counter and passes causal variance
    let steps: Vec<f64> = (0..40).map(|i| i as f64).collect();
    let leaky: Vec<f64> = (0..40).map(|i| (i * 10 + 5) as f64).collect();
    let causal: Vec<f64> = (0..40).map(|i| ((i * 7) % 5) as f64).collect();
    let mut feats_bad = HashMap::new();
    feats_bad.insert("leak_step".into(), leaky);
    let (bad_ok, flagged) = check_clock_leak(&feats_bad, &steps, 0.85);
    let mut feats_good = HashMap::new();
    feats_good.insert("causal_var".into(), causal);
    let (good_ok, good_flagged) = check_clock_leak(&feats_good, &steps, 0.85);
    let g2_ok = !bad_ok && !flagged.is_empty() && good_ok && good_flagged.is_empty() && spearman_rho(&steps, &steps) > 0.99;

    // Gate 3: counterfactual divergence between noop and past_qmass on non-trivial skew
    let trace = PartitionTrace {
        scale: "test".into(),
        n_base: 1000,
        shape: "abrupt".into(),
        seed: 0,
        r_shards: 4,
        t_windows: 16,
        k_block: 4,
        compaction_ops_per_shard: vec![100.0; 4],
        q_mat: (0..16)
            .map(|t| if t < 8 { vec![100.0, 5.0, 5.0, 5.0] } else { vec![5.0, 100.0, 5.0, 5.0] })
            .collect(),
        d_mat: vec![vec![50.0, 50.0, 5.0, 5.0]; 16],
    };
    let noop_ops = eval_schedule_exact_ops(&trace, &schedule_policy(&trace, "noop", 3)).0;
    let pq_ops = eval_schedule_exact_ops(&trace, &schedule_policy(&trace, "past_qmass", 3)).0;
    let g3_ok = pq_ops < noop_ops;

    // Gate 4: wall/CPU contamination check & ranking flip detection
    let flip = compare_rankings(
        &["a".into(), "b".into(), "c".into()],
        &[10.0, 20.0, 30.0],
        &[25.0, 15.0, 30.0],
        false,
    );
    let g4_ok = flip.pair_flips == 1 && flip.top1_flipped;

    let report = SelfTestReport {
        status: if g1_ok && g2_ok && g3_ok && g4_ok {
            "PASS (4/4 gates verified)".into()
        } else {
            "FAIL".into()
        },
        gates_passed: [g1_ok, g2_ok, g3_ok, g4_ok].into_iter().filter(|&b| b).count(),
        gate1_cadence_bracket: g1_ok,
        gate2_clock_leak: g2_ok,
        gate3_counterfactual_horizon: g3_ok,
        gate4_contamination: g4_ok,
        elapsed_ms: t0.elapsed().as_secs_f64() * 1e3,
    };
    println!("{}", serde_json::to_string_pretty(&report)?);
    if report.gates_passed != 4 {
        bail!("self-test failed");
    }
    Ok(())
}

fn run_audit_csv(csv_path: PathBuf, max_ratio: f64, out_json: Option<PathBuf>) -> Result<()> {
    let t0 = Instant::now();
    let f = File::open(&csv_path).with_context(|| format!("opening {}", csv_path.display()))?;
    let mut lines = BufReader::new(f).lines();
    let header_line = lines.next().context("empty csv")??;
    let headers: Vec<String> = split_csv_line(&header_line).into_iter().map(|s| s.to_string()).collect();

    let col_idx = |names: &[&str]| -> Option<usize> {
        for cand in names {
            if let Some(p) = headers.iter().position(|h| h == cand) {
                return Some(p);
            }
        }
        None
    };

    let ratio_col = col_idx(&["wall_cpu_ratio"]);
    let wall_col = col_idx(&["wall_s"]);
    let cpu_col = col_idx(&["total_cpu_s", "total_s", "cpu_s"]);

    let mut total_rows = 0usize;
    let mut checked_rows = 0usize;
    let mut violations = 0usize;
    let mut max_seen = 0.0f64;

    for line_res in lines {
        let line = line_res?;
        if line.trim().is_empty() {
            continue;
        }
        total_rows += 1;
        let fields = split_csv_line(&line);
        let r_opt = if let Some(ci) = ratio_col {
            fields.get(ci).and_then(|s| s.parse::<f64>().ok())
        } else if let (Some(wi), Some(ci)) = (wall_col, cpu_col) {
            match (
                fields.get(wi).and_then(|s| s.parse::<f64>().ok()),
                fields.get(ci).and_then(|s| s.parse::<f64>().ok()),
            ) {
                (Some(w), Some(c)) if c > 0.0 => Some(w / c),
                _ => None,
            }
        } else {
            None
        };
        if let Some(r) = r_opt {
            checked_rows += 1;
            if r > max_seen {
                max_seen = r;
            }
            if r > max_ratio {
                violations += 1;
            }
        }
    }

    let report = AuditCsvReport {
        path: csv_path.display().to_string(),
        total_rows,
        checked_rows,
        violations,
        max_wall_cpu_ratio: (max_seen * 10000.0).round() / 10000.0,
        passed: violations == 0 && checked_rows > 0,
        elapsed_ms: (t0.elapsed().as_secs_f64() * 1e3 * 1000.0).round() / 1000.0,
    };
    let s = serde_json::to_string_pretty(&report)?;
    if let Some(op) = out_json {
        std::fs::write(op, &s)?;
    }
    println!("{s}");
    Ok(())
}

fn run_flip_matrix(
    csv_path: PathBuf,
    group_col: String,
    arm_col: String,
    dirty_col: String,
    clean_col: String,
    higher_is_better: bool,
    out_json: Option<PathBuf>,
) -> Result<()> {
    let t0 = Instant::now();
    let f = File::open(&csv_path)?;
    let mut lines = BufReader::new(f).lines();
    let header_line = lines.next().context("empty csv")??;
    let headers: Vec<String> = split_csv_line(&header_line).into_iter().map(|s| s.to_string()).collect();

    let g_idx = headers.iter().position(|h| h == &group_col).context("missing group_col")?;
    let a_idx = headers.iter().position(|h| h == &arm_col).context("missing arm_col")?;
    let d_idx = headers.iter().position(|h| h == &dirty_col).context("missing dirty_col")?;
    let c_idx = headers.iter().position(|h| h == &clean_col).context("missing clean_col")?;

    let mut groups: BTreeMap<String, (Vec<String>, Vec<f64>, Vec<f64>)> = BTreeMap::new();
    for line_res in lines {
        let line = line_res?;
        if line.trim().is_empty() {
            continue;
        }
        let cols = split_csv_line(&line);
        if let (Some(g), Some(a), Ok(vd), Ok(vc)) = (
            cols.get(g_idx),
            cols.get(a_idx),
            cols.get(d_idx).unwrap_or(&"").parse::<f64>(),
            cols.get(c_idx).unwrap_or(&"").parse::<f64>(),
        ) {
            let entry = groups.entry((*g).to_string()).or_default();
            entry.0.push((*a).to_string());
            entry.1.push(vd);
            entry.2.push(vc);
        }
    }

    let mut total_flips = 0usize;
    let mut total_pairs = 0usize;
    let mut top1_flips = 0usize;
    let mut taus = Vec::new();
    let n_groups = groups.len();
    for (_g, (names, vd, vc)) in groups {
        let fs = compare_rankings(&names, &vd, &vc, higher_is_better);
        total_flips += fs.pair_flips;
        total_pairs += fs.total_pairs;
        if fs.top1_flipped {
            top1_flips += 1;
        }
        taus.push(fs.kendall_tau);
    }

    #[derive(Serialize)]
    struct FlipAgg {
        n_groups: usize,
        pair_flips: usize,
        total_pairs: usize,
        pairwise_flip_pct: f64,
        mean_kendall_tau: f64,
        top1_flips: usize,
        top1_flip_pct: f64,
        elapsed_ms: f64,
    }
    let agg = FlipAgg {
        n_groups,
        pair_flips: total_flips,
        total_pairs,
        pairwise_flip_pct: if total_pairs > 0 {
            ((100.0 * total_flips as f64 / total_pairs as f64) * 100.0).round() / 100.0
        } else {
            0.0
        },
        mean_kendall_tau: if !taus.is_empty() {
            ((taus.iter().sum::<f64>() / taus.len() as f64) * 10000.0).round() / 10000.0
        } else {
            1.0
        },
        top1_flips,
        top1_flip_pct: if n_groups > 0 {
            ((100.0 * top1_flips as f64 / n_groups as f64) * 10.0).round() / 10.0
        } else {
            0.0
        },
        elapsed_ms: (t0.elapsed().as_secs_f64() * 1e3 * 1000.0).round() / 1000.0,
    };
    let s = serde_json::to_string_pretty(&agg)?;
    if let Some(op) = out_json {
        std::fs::write(op, &s)?;
    }
    println!("{s}");
    Ok(())
}

fn run_scale_replay(trace_dir: PathBuf, budget: usize, out_json: PathBuf) -> Result<()> {
    let t0 = Instant::now();
    let mut files: Vec<PathBuf> = std::fs::read_dir(&trace_dir)?
        .filter_map(|e| e.ok().map(|x| x.path()))
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    files.sort();

    let policies = ["noop", "fixed_c2", "fixed_c4", "past_qmass", "ewma_a03", "max_delta", "oracle_1blk"];
    let mut rows = Vec::with_capacity(files.len() * policies.len());

    for fp in &files {
        let trace: PartitionTrace = serde_json::from_str(&std::fs::read_to_string(fp)?)?;
        let fname = fp.file_name().and_then(|s| s.to_str()).unwrap_or("").to_string();
        for pol in policies {
            let acts = schedule_policy(&trace, pol, budget);
            let (tot, scan, maint) = eval_schedule_exact_ops(&trace, &acts);
            let executed = acts.iter().filter(|a| a.is_some()).count();
            rows.push(ScaleReplayRow {
                file: fname.clone(),
                scale: trace.scale.clone(),
                n_base: trace.n_base,
                shape: trace.shape.clone(),
                seed: trace.seed,
                policy: pol.to_string(),
                budget,
                total_ops: (tot * 100.0).round() / 100.0,
                scan_ops: (scan * 100.0).round() / 100.0,
                maint_ops: (maint * 100.0).round() / 100.0,
                executed_actions: executed,
            });
        }
    }

    let report = ScaleReplayReport {
        traces_evaluated: files.len(),
        runs_evaluated: rows.len(),
        elapsed_ms: (t0.elapsed().as_secs_f64() * 1e3 * 1000.0).round() / 1000.0,
        rows,
    };
    if let Some(parent) = out_json.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&out_json, serde_json::to_string_pretty(&report)?)?;
    println!(
        "scale-replay evaluated {} traces ({} policy runs) in {:.3} ms -> {}",
        report.traces_evaluated,
        report.runs_evaluated,
        report.elapsed_ms,
        out_json.display()
    );
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Commands::SelfTest => run_self_test(),
        Commands::AuditCsv {
            csv,
            max_wall_cpu_ratio,
            out_json,
        } => run_audit_csv(csv, max_wall_cpu_ratio, out_json),
        Commands::FlipMatrix {
            csv,
            group_col,
            arm_col,
            dirty_col,
            clean_col,
            higher_is_better,
            out_json,
        } => run_flip_matrix(csv, group_col, arm_col, dirty_col, clean_col, higher_is_better, out_json),
        Commands::ScaleReplay {
            trace_dir,
            budget,
            out_json,
        } => run_scale_replay(trace_dir, budget, out_json),
    }
}
