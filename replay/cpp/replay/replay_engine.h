// Q-BAM M1 engine -- deterministic replay engine: shard state, K-window blocks, policies,
// JSONL record emission (engine_contract.md §6-§8).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/output.h"
#include "core/live_store.h"
#include "core/router.h"
#include "core/shard_index.h"
#include "eval/cpu_meter.h"
#include "eval/quality_attribution.h"
#include "replay/policy_forecast.h"
#include "io/events.h"

namespace qbam {

// ---------------------------------------------------------------- config / actions
struct EngineConfig {
    uint32_t R = 0;
    uint32_t dim = 0;
    uint32_t k = 10;
    uint32_t nprobe = 0;
    size_t M = 16;
    size_t ef_construction = 200;
    size_t ef_search = 64;
    uint64_t seed = 0;
    double r_min = 0.95;  // quality-gate threshold for recall_violation_rate (plan v1.4 C2.1)
    double kappa_RE = 0.0;
    double c_ref_cpu_ms = 0.0;
    std::string run_id;
    int pin_cpu = -1;           // sched_setaffinity target; -1 = do not pin
    uint32_t qmass_window = 8;
    // 预算节奏（E4b）：>1 时 past_qmass/forecast_qmass/bam_price 只在 j=1,1+sp,1+2sp... 允许动作
    int spacing_blocks = 1;  // query_delta heuristic lookback (plan §10)
    uint32_t overfetch_start = 0;  // 0 => contract formula max(4k, k+32)
    // Delta representation: "scan" (exact-scan counterfactual replay) or "hnsw" (per-shard HNSW).
    std::string delta_mode = "scan";
    size_t delta_ef_search = 0;
    size_t delta_initial_capacity = 4096;
};

struct Action {
    enum class Kind { Noop = 0, Refresh = 1, Merge = 2, RefreshPair = 3 };
    Kind kind = Kind::Noop;
    int shard = -1;
    int shard2 = -1;

    std::string name() const;          // contract §7 action value: noop|refresh|merge|refresh_pair
    // Config-style action string (inverse of the cf.actions parser): "noop", "refresh:r",
    // "merge:r", "refresh:r1+r2". Emitted as the optional `action_str` branch-row key so that
    // Python analysis can key branches exactly like the config actions.
    std::string config_string() const;
    static Action noop();
    static Action refresh(int r);
    static Action merge(int r);
    static Action refresh_pair(int r, int s);
    bool is_noop() const { return kind == Kind::Noop; }
    bool is_refresh() const { return kind == Kind::Refresh; }
    bool is_merge() const { return kind == Kind::Merge; }
};

// ---------------------------------------------------------------- policy
class ReplayEngine;  // defined below

struct PolicyParams {
    int cadence_blocks = 4;
    int phase = 0;
    std::string shard_order = "round_robin";
    uint32_t qmass_window = 8;
    // 预算节奏（E4b）：>1 时 past_qmass/forecast_qmass/bam_price 只在 j=1,1+sp,1+2sp... 允许动作
    int spacing_blocks = 1;
    // ---- E3-stage2 budget-allocation policies (past_qmass / forecast_qmass / bam_price) ----
    std::string action_mode = "refresh";  // refresh | merge | auto
    double c_scan = 2.83e-5;              // ms / (query * vector), E1-lite calibration
    double c_insert = 0.04573;            // ms / vector, E1-lite calibration
    double lambda0 = 0.0;                 // initial budget price
    double eta = 1.0;                     // integral controller gain
    double lambda_max = 1e3;              // price clip
    std::string pred_path;                // predictions.csv (forecast_qmass / bam_price)
    std::string pred_model;               // e.g. "lgbm"
    std::string trace_name;               // optional override; default = trace dir basename
};

struct PolicyRuntime {
    std::string policy = "noop";
    PolicyParams params;
    uint64_t fixed_counter = 0;
    std::vector<double> observed_refresh_ms;  // per shard, 0.0 = never observed
    double observed_global_ms = 0.0;
    uint64_t observed_count = 0;
    // ---- E3-stage2 state ----
    ForecastTable forecast;          // loaded once by policy_prepare()
    double budget_total_ms = 0.0;    // B = kappa_RE * c_ref_cpu_ms; <= 0 => unlimited
    double cum_spend_ms = 0.0;       // actual maintenance + policy CPU so far
    double lambda = 0.0;             // current budget price
    uint64_t forecast_fallback_rows = 0;  // (j,shard) misses that fell back to past qmass
};

// One decision returned by choose_action(); the E3-stage2 fields are written to block rows.
struct ActionDecision {
    Action action;
    bool budget_rejected = false;
    double score_chosen = 0.0;    // raw/ net score of the selected shard
    double cost_pred_ms = 0.0;    // cost_r = (base_n + delta_n) * c_insert
    double lambda = 0.0;          // price used for this decision
    uint64_t delta_chosen = 0;    // delta_n of the selected shard (0 for noop)
    bool forecast_fallback = false;  // selected shard used past qmass due to missing forecast
};

void policy_runtime_init(PolicyRuntime* rt, const std::string& policy, const PolicyParams& params);
// Resolves trace_name, loads predictions.csv when needed, sets B = kappa*c_ref (0 => unlimited)
// and lambda = lambda0. Logs a warning and falls back to past qmass on forecast load failure.
void policy_prepare(PolicyRuntime* rt, const std::string& trace_path, uint32_t n_windows,
                    uint32_t K, double kappa_RE, double c_ref_cpu_ms);
// Integral budget-price controller: lambda <- clip(lambda + eta*(cum_spend - B*(j+1)/J)/max(B,eps)).
double budget_lambda_update(const PolicyRuntime& rt, uint32_t j, uint32_t J);
// Chooses the action at decision block j (decision point = start of window j*K).
ActionDecision choose_action(PolicyRuntime* rt, uint32_t j, const ReplayEngine* engine);
void record_refresh_cost(PolicyRuntime* rt, int shard, double cpu_ms);

// ---------------------------------------------------------------- query / window / block metrics
struct QueryExecResult {
    std::vector<uint64_t> ids;
    std::vector<uint64_t> versions;
    std::vector<float> dists;
    std::vector<bool> from_delta;
    std::vector<uint32_t> visited_shards;
    int base_candidates_requested = 0;
    int stale_candidates_filtered = 0;
    int overfetch_expansions = 0;
    bool underfilled = false;
    double wall_us = 0.0;
    double cpu_us = 0.0;
    // Component breakdown (wall-clock microseconds; diagnostics only, NOT part of query_cpu_ms).
    double base_search_us = 0.0;    // sum over visited shards
    double delta_scan_us = 0.0;     // sum over visited shards
    double selection_us = 0.0;      // bounded top-k copy-out + sort + dedupe
    uint32_t valid_candidates = 0;  // candidates that passed filters in the last scan pass
    // evaluation-only (computed after the timed query section; not part of query_cpu_ms)
    double recall = 0.0;
    double route_upper = 0.0;
    double index_loss = 0.0;
    double returned_delta_fraction = 0.0;
};

// Structured per-shard decision record (engineering: decisions are data, not stdout text).
struct DecisionEvent {
    uint32_t shard = 0;
    std::string kind;      // "promote" | "demote" | "gc"
    double rent_qvec = 0.0;   // realized rent that justified the decision (q x vec)
    double price_qvec = 0.0;  // price threshold at decision time (q x vec)
    double cost_ms = 0.0;     // measured CPU of the action (timing: not deterministic)
    uint64_t vectors = 0;
};

struct WindowMetrics {
    uint32_t window = 0;
    uint32_t query_count = 0;
    double query_cpu_ms = 0.0;
    double update_cpu_ms = 0.0;
    Percentiles p;
    double recall = 0.0;
    double route_upper = 0.0;
    double index_loss = 0.0;
    double returned_delta_fraction = 0.0;
    uint32_t recall_violations = 0;
    uint32_t overfetch_count = 0;
    uint32_t underfilled_queries = 0;
    std::vector<uint32_t> shard_delta_n;
    std::vector<uint32_t> shard_base_n;
    std::vector<uint32_t> shard_q_accesses;
    std::vector<double> wall_us_all;  // per-query wall samples (for block percentiles)
    // Per-window diagnostics (wall-clock based, same as the per-shard search/scan counters).
    double base_search_cpu_ms = 0.0;
    double delta_scan_cpu_ms = 0.0;
    double selection_cpu_ms = 0.0;
    uint32_t candidate_count_max = 0;
};

struct ShardSnapshot {
    uint32_t r = 0;
    uint64_t base_n = 0;
    uint64_t delta_n = 0;
    uint64_t tombstone_n = 0;
    uint64_t q_accesses = 0;
    double base_search_wall_us = 0.0;
    double delta_scan_wall_us = 0.0;
};

struct BlockMetrics {
    uint32_t j = 0;
    uint32_t window_start = 0;
    uint32_t n_windows = 0;
    Action action;
    std::string state_hash;  // state at the decision point, BEFORE the action
    double policy_cpu_ms = 0.0;
    double maintenance_cpu_ms = 0.0;
    double budget_nominal_cpu_ms = 0.0;
    double budget_actual_cpu_ms = 0.0;
    double budget_unused_cpu_ms = 0.0;
    bool overshoot = false;
    bool budget_rejected = false;
    double query_cpu_ms = 0.0;
    double update_cpu_ms = 0.0;
    Percentiles p;
    double recall = 0.0;
    double route_upper = 0.0;
    double index_loss = 0.0;
    double returned_delta_fraction = 0.0;
    double recall_violation_rate = 0.0;
    uint64_t overfetch_count = 0;
    uint64_t underfilled_queries = 0;
    // E3-stage2 decision diagnostics (0 for noop/fixed/query_delta).
    double score_chosen = 0.0;
    double cost_pred_ms = 0.0;
    double lambda = 0.0;
    uint64_t delta_chosen = 0;
    std::vector<ShardSnapshot> shards;  // counts after action; counters over the block
};

struct ShardRecentStats {
    uint32_t r = 0;
    uint64_t q_accesses_recent = 0;
    double base_search_us_recent = 0.0;
    double delta_scan_us_recent = 0.0;
};

struct BranchRecord {
    uint32_t j = 0;
    Action action;
    std::string state_hash;  // pre-action state hash (shared by all branches of a block)
    // For merge:r this includes the Base deep copy; copy_cpu_ms isolates that helper cost so
    // the aggregation can use (maintenance_cpu_ms - copy_cpu_ms) as the incremental-merge cost.
    double maintenance_cpu_ms = 0.0;
    double copy_cpu_ms = 0.0;
    double policy_cpu_ms = 0.0;
    std::vector<WindowMetrics> windows;
};

// ---------------------------------------------------------------- engine
class ReplayEngine {
public:
    ReplayEngine() = default;

    // Builds the initial state from init.bin (Base per shard, empty Delta, LiveStore v=1).
    void init(const EngineConfig& cfg, const PoolData* pool, const TraceData* trace,
              const GtData* gt, const InitData& init, const Centroids& centroids);

    // Deep state copy for counterfactual branches. Base HNSW snapshots are immutable and are
    // shared by pointer, so no graph is copied (engine_contract.md §3.2).
    ReplayEngine clone_for_branch() const;

    const EngineConfig& cfg() const { return cfg_; }
    const PoolData& pool() const { return *pool_; }
    const TraceData& trace() const { return *trace_; }
    const GtData& gt() const { return *gt_; }
    const LiveStore& live() const { return live_; }
    const std::vector<ShardIndex>& shards() const { return shards_; }
    uint32_t R() const { return cfg_.R; }
    uint32_t K() const { return trace_->K; }
    uint32_t n_windows() const { return trace_->n_windows; }
    uint32_t J() const { return (trace_->n_windows + trace_->K - 1) / trace_->K; }
    uint32_t window_block(uint32_t w) const { return w / trace_->K; }
    uint32_t window_h(uint32_t w) const { return w - window_block(w) * trace_->K + 1; }

    // Applies one CRUD event; `cpu_ms` receives the measured thread CPU cost.
    void apply_crud(const TraceEvent& ev, double* cpu_ms);
    // Applies an action; `maintenance_cpu_ms` receives the measured thread CPU cost.
    RefreshResult apply_action(const Action& action, double* maintenance_cpu_ms);

    // Executes one query at query-index `gt_index` (global order). Quality attribution is
    // computed after the timed section (evaluation-only data, not part of query CPU).
    QueryExecResult execute_query(uint32_t window, uint32_t gt_index, const float* query);

    // Applies the window's CRUD and queries in trace order and collects window metrics.
    // When `query_sink` is non-null, per-query records stream to it (--events-out).
    WindowMetrics process_window(uint32_t window, LineSink* query_sink, const std::string& run_id);

    BlockMetrics aggregate_block(uint32_t j, const Action& action, double policy_cpu_ms,
                                        double maintenance_cpu_ms, const std::string& state_hash,
                                        double budget_nominal_cpu_ms,
                                        const std::vector<WindowMetrics>& windows,
                                        const std::vector<ShardSnapshot>& shard_snapshot);

    // Counts (base_n/delta_n/tombstone_n) after the action; counters start at zero.
    void capture_shard_counts(std::vector<ShardSnapshot>* out) const;
    // Fills block-scoped counters (q_accesses, search wall us) and leaves them untouched.
    void fill_shard_counters(std::vector<ShardSnapshot>* io) const;
    void reset_shard_block_counters();

    // sha256 over (sorted live id -> (version, shard)) + per-shard (base_n, delta_n,
    // tombstone_n), truncated to 16 hex chars (engine_contract.md §8.9).
    std::string state_hash() const;

    // Decision-point visible per-shard statistics over the last `lookback` base windows
    // strictly before `window` (W=8 for the G0 Query x Delta heuristic).
    void recent_shard_stats(uint32_t before_window, uint32_t lookback,
                            std::vector<ShardRecentStats>* out) const;

    // Query-mass of shard r over the `lookback` base windows strictly before `window`.
    uint64_t qmass_before(uint32_t window, uint32_t shard, uint32_t lookback) const;
    uint64_t window_q_accesses(uint32_t window, uint32_t shard) const;
    uint64_t base_n(uint32_t r) const { return shards_[r].base_n(); }
    uint64_t delta_n(uint32_t r) const { return shards_[r].delta_n(); }

private:
    uint64_t build_seed(uint32_t shard) const;

    EngineConfig cfg_;
    const PoolData* pool_ = nullptr;
    const TraceData* trace_ = nullptr;
    const GtData* gt_ = nullptr;
    Router router_;
    LiveStore live_;
    std::vector<ShardIndex> shards_;
    HnswParams hnsw_;
    // Per-query scratch (single-threaded engine: reuse to avoid 4-5 allocations per query).
    std::vector<uint32_t> scratch_visited_;
    std::vector<char> scratch_visited_mask_;
    std::vector<std::pair<float, uint64_t>> scratch_base_hits_;
    std::vector<std::pair<float, uint64_t>> scratch_delta_hits_;  // delta_mode=hnsw only
    std::vector<std::vector<uint32_t>> window_q_accesses_;      // n_windows x R
    std::vector<std::vector<double>> window_base_search_us_;    // n_windows x R
    std::vector<std::vector<double>> window_delta_scan_us_;     // n_windows x R
};

// ---------------------------------------------------------------- JSONL emission (contract §7)
struct RunManifest {
    std::string run_id;
    std::string mode;
    std::string policy;
    std::string git_sha;
    std::string config_sha256;
    std::string dataset_sha256;
    std::string trace_sha256;
    std::string gt_sha256;
    uint32_t R = 0;
    uint32_t dim = 0;
    uint32_t k = 0;
    size_t M = 0;
    size_t ef_construction = 0;
    size_t ef_search = 0;
    std::string delta_mode = "scan";
    size_t delta_ef_search = 0;
    uint32_t nprobe = 0;
    uint64_t seed = 0;
    int threads = 1;
    int pin_cpu = -1;
    uint32_t n_events = 0;
    uint32_t n_windows = 0;
    uint32_t K = 0;
    uint32_t J = 0;
    double wall_s = 0.0;
    std::string platform;
    std::string build;
    std::string policy_params_json = "{}";
    int return_code = 0;
};

std::string json_run_record(const RunManifest& m);
std::string json_block_record(const std::string& run_id, const BlockMetrics& b);
std::string json_window_record(const std::string& run_id, const WindowMetrics& w, uint32_t j,
                               const Action& action, int action_index, uint32_t h);
std::string json_branch_record(const std::string& run_id, const BranchRecord& br);
std::string json_state_record(const std::string& run_id, uint32_t j, uint32_t window_start,
                              const std::string& state_hash,
                              const std::vector<ShardSnapshot>& counts,
                              const std::vector<ShardRecentStats>& recent);
std::string json_query_record(const std::string& run_id, uint32_t window, uint32_t q_index,
                              const QueryExecResult& r);

}  // namespace qbam
