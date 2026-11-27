// Q-BAM M1 engine -- deterministic replay engine implementation.
#include "replay/replay_engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <queue>
#include <unordered_set>

#include "common/logging.h"

namespace qbam {

namespace {

std::string u64s(uint64_t v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%llu", static_cast<unsigned long long>(v));
    return std::string(buf);
}

std::string u32s(uint32_t v) {
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%u", v);
    return std::string(buf);
}

std::string jbool(bool v) { return v ? std::string("true") : std::string("false"); }

void append_u32_array(std::string* out, const std::vector<uint32_t>& v) {
    *out += '[';
    for (size_t i = 0; i < v.size(); ++i) {
        if (i > 0) *out += ',';
        *out += u32s(v[i]);
    }
    *out += ']';
}

// Candidate for the bounded top-k heap. Strict order (dist, id, shard) == contract tie-breaking.
struct QueryCand {
    float dist = 0.0f;
    uint64_t id = 0;
    uint32_t shard = 0;
    bool from_delta = false;
};

struct QueryCandLess {
    bool operator()(const QueryCand& a, const QueryCand& b) const {
        if (a.dist != b.dist) return a.dist < b.dist;
        if (a.id != b.id) return a.id < b.id;
        return a.shard < b.shard;
    }
};

void append_block_shards(std::string* out, const std::vector<ShardSnapshot>& shards) {
    *out += '[';
    for (size_t i = 0; i < shards.size(); ++i) {
        const ShardSnapshot& s = shards[i];
        if (i > 0) *out += ',';
        *out += "{\"r\":" + u32s(s.r) + ",\"base_n\":" + u64s(s.base_n) +
                ",\"delta_n\":" + u64s(s.delta_n) + ",\"tombstone_n\":" + u64s(s.tombstone_n) +
                ",\"q_accesses\":" + u64s(s.q_accesses) +
                ",\"delta_scan_cpu_us\":" + fmt6(s.delta_scan_wall_us) +
                ",\"base_search_cpu_us\":" + fmt6(s.base_search_wall_us) + "}";
    }
    *out += ']';
}

}  // namespace

// ---------------------------------------------------------------- Action
std::string Action::name() const {
    switch (kind) {
        case Kind::Noop: return "noop";
        case Kind::Refresh: return "refresh";
        case Kind::Merge: return "merge";
        case Kind::RefreshPair: return "refresh_pair";
    }
    return "noop";
}

std::string Action::config_string() const {
    switch (kind) {
        case Kind::Noop: return "noop";
        case Kind::Refresh: return "refresh:" + std::to_string(shard);
        case Kind::Merge: return "merge:" + std::to_string(shard);
        case Kind::RefreshPair:
            return "refresh:" + std::to_string(shard) + "+" + std::to_string(shard2);
    }
    return "noop";
}

Action Action::noop() { return Action(); }

Action Action::refresh(int r) {
    Action a;
    a.kind = Kind::Refresh;
    a.shard = r;
    return a;
}

Action Action::merge(int r) {
    Action a;
    a.kind = Kind::Merge;
    a.shard = r;
    return a;
}

Action Action::refresh_pair(int r, int s) {
    Action a;
    a.kind = Kind::RefreshPair;
    a.shard = r;
    a.shard2 = s;
    return a;
}

// ---------------------------------------------------------------- policy
void policy_runtime_init(PolicyRuntime* rt, const std::string& policy, const PolicyParams& params) {
    QBAM_CHECK(rt != nullptr, "policy_runtime_init: null");
    rt->policy = policy;
    rt->params = params;
    rt->fixed_counter = 0;
    rt->observed_refresh_ms.assign(64, 0.0);
    rt->observed_global_ms = 0.0;
    rt->observed_count = 0;
    rt->lambda = params.lambda0;
    rt->cum_spend_ms = 0.0;
    rt->budget_total_ms = 0.0;
    rt->forecast_fallback_rows = 0;
    rt->forecast = ForecastTable();
}

void record_refresh_cost(PolicyRuntime* rt, int shard, double cpu_ms) {
    if (rt == nullptr || shard < 0) return;
    if (static_cast<size_t>(shard) >= rt->observed_refresh_ms.size()) {
        rt->observed_refresh_ms.resize(static_cast<size_t>(shard) + 1, 0.0);
    }
    rt->observed_refresh_ms[shard] = cpu_ms;
    rt->observed_global_ms += cpu_ms;
    rt->observed_count += 1;
}

namespace {
std::string path_basename_dir(const std::string& path) {
    // ".../traces/uniform_s00/trace.bin" -> "uniform_s00"
    const size_t last = path.find_last_of('/');
    if (last == std::string::npos) return std::string();
    const std::string parent = path.substr(0, last);
    const size_t prev = parent.find_last_of('/');
    return prev == std::string::npos ? parent : parent.substr(prev + 1);
}
}  // namespace

void policy_prepare(PolicyRuntime* rt, const std::string& trace_path, uint32_t n_windows,
                    uint32_t K, double kappa_RE, double c_ref_cpu_ms) {
    QBAM_CHECK(rt != nullptr, "policy_prepare: null runtime");
    rt->budget_total_ms = kappa_RE * c_ref_cpu_ms;
    if (rt->budget_total_ms <= 0.0) rt->budget_total_ms = 0.0;  // 0 => unlimited (uncalibrated)
    rt->cum_spend_ms = 0.0;
    rt->lambda = rt->params.lambda0;
    rt->forecast = ForecastTable();
    rt->forecast_fallback_rows = 0;
    const bool needs_forecast = rt->policy == "forecast_qmass" || rt->policy == "bam_price";
    if (!needs_forecast) return;
    std::string trace_name = rt->params.trace_name;
    if (trace_name.empty()) trace_name = path_basename_dir(trace_path);
    if (trace_name.empty()) trace_name = "trace";
    if (rt->params.pred_path.empty()) {
        QBAM_LOG("[policy] WARNING: %s requested but policy_params.pred_path is empty; "
                 "falling back to past qmass for every (j, shard)",
                 rt->policy.c_str());
        return;
    }
    std::string error;
    if (!rt->forecast.load(rt->params.pred_path, trace_name, rt->params.pred_model, &error)) {
        QBAM_LOG("[policy] WARNING: forecast load failed (%s); falling back to past qmass",
                 error.c_str());
        return;
    }
    QBAM_LOG("[policy] %s: loaded %zu q_hat rows (trace=%s model=%s) from %s; B=%.2f ms",
             rt->policy.c_str(), rt->forecast.rows_kept, trace_name.c_str(),
             rt->params.pred_model.c_str(), rt->params.pred_path.c_str(), rt->budget_total_ms);
}

double budget_lambda_update(const PolicyRuntime& rt, uint32_t j, uint32_t J) {
    if (J == 0) return rt.lambda;
    const double B = rt.budget_total_ms;
    if (B <= 0.0) return rt.lambda;  // uncalibrated => price stays put
    const double schedule = B * static_cast<double>(j + 1) / static_cast<double>(J);
    const double err = rt.cum_spend_ms - schedule;
    double lambda = rt.lambda + rt.params.eta * err / std::max(B, 1e-9);
    if (lambda < 0.0) lambda = 0.0;
    if (lambda > rt.params.lambda_max) lambda = rt.params.lambda_max;
    return lambda;
}

ActionDecision choose_action(PolicyRuntime* rt, uint32_t j, const ReplayEngine* engine) {
    QBAM_CHECK(rt != nullptr && engine != nullptr, "choose_action: null argument");
    ActionDecision out;
    if (rt->policy == "noop") {
        return out;
    }
    if (rt->policy == "fixed") {
        const int cadence = rt->params.cadence_blocks;
        QBAM_CHECK(cadence >= 1, "policy=fixed requires cadence_blocks >= 1");
        int phase = rt->params.phase;
        phase = ((phase % cadence) + cadence) % cadence;
        const bool at_cadence =
            (static_cast<int>(j) >= phase) && ((static_cast<int>(j) - phase) % cadence == 0);
        if (!at_cadence) return out;
        if (rt->params.shard_order != "round_robin") {
            fail("policy=fixed: unsupported shard_order \"" + rt->params.shard_order +
                 "\" (M1 supports round_robin)");
        }
        const uint32_t r = static_cast<uint32_t>(rt->fixed_counter % engine->R());
        ++rt->fixed_counter;
        // E3-stage2 baseline: honour policy_params.action_mode so the periodic (no-selection)
        // baseline can be compared against merge-based budgeted policies at equal action type.
        const std::string& mode = rt->params.action_mode;
        QBAM_CHECK(mode == "refresh" || mode == "merge" || mode == "auto",
                   "unknown policy_params.action_mode \"" + mode + "\"");
        if (engine->delta_n(r) == 0 && mode != "refresh") {
            // nothing to merge on an empty Delta (merge of nothing is a no-op that costs copy)
            return out;
        }
        if (mode == "merge") {
            out.action = Action::merge(static_cast<int>(r));
        } else if (mode == "auto") {
            const uint64_t dn = engine->delta_n(r);
            const double merge_cost = static_cast<double>(dn) * rt->params.c_insert;
            const double refresh_cost = (static_cast<double>(engine->base_n(r)) +
                                         static_cast<double>(dn)) * rt->params.c_insert;
            out.action = (merge_cost < refresh_cost) ? Action::merge(static_cast<int>(r))
                                                     : Action::refresh(static_cast<int>(r));
        } else {
            out.action = Action::refresh(static_cast<int>(r));
        }
        return out;
    }
    if (rt->policy == "query_delta") {
        const uint32_t start_window = std::min<uint32_t>(j * engine->K(), engine->n_windows());
        const uint64_t lookback = rt->params.qmass_window;
        const double global_cost =
            rt->observed_count > 0 ? rt->observed_global_ms / static_cast<double>(rt->observed_count)
                                   : 0.0;
        double best_score = -1.0;
        int best_r = -1;
        for (uint32_t r = 0; r < engine->R(); ++r) {
            const uint64_t dn = engine->delta_n(r);
            if (dn == 0) continue;
            const uint64_t qmass = engine->qmass_before(start_window, r, static_cast<uint32_t>(lookback));
            double cost = 0.0;
            if (static_cast<size_t>(r) < rt->observed_refresh_ms.size()) {
                cost = rt->observed_refresh_ms[r];
            }
            if (cost <= 0.0) {
                cost = global_cost > 0.0 ? global_cost : static_cast<double>(engine->base_n(r) + dn);
            }
            const double score = static_cast<double>(qmass) * static_cast<double>(dn) /
                                 std::max(cost, 1e-6);
            if (score > best_score) {
                best_score = score;
                best_r = static_cast<int>(r);
            }
        }
        if (best_r < 0 || !(best_score > 0.0)) return out;
        out.action = Action::refresh(best_r);
        out.score_chosen = best_score;
        return out;
    }
    if (rt->policy == "past_qmass" || rt->policy == "forecast_qmass" ||
        rt->policy == "bam_price") {
        // E4b 预算节奏：spacing_blocks>1 时只在 j=1, 1+sp, ... 允许动作（j=0 无 Delta，固定跳过）
        if (rt->params.spacing_blocks > 1) {
            const int sp = rt->params.spacing_blocks;
            const int jj = static_cast<int>(j);
            if (jj < 1 || (jj - 1) % sp != 0) return out;
        }
        const uint32_t R = engine->R();
        const uint32_t start_window = std::min<uint32_t>(j * engine->K(), engine->n_windows());
        const bool use_forecast = rt->policy != "past_qmass";
        const uint32_t W = rt->params.qmass_window;
        const uint32_t hist_windows =
            start_window == 0 ? 0 : std::min<uint32_t>(W, start_window);
        const std::string& mode = rt->params.action_mode;
        QBAM_CHECK(mode == "refresh" || mode == "merge" || mode == "auto",
                   "unknown policy_params.action_mode \"" + mode + "\"");
        double best_value = -std::numeric_limits<double>::infinity();
        int best_r = -1;
        double best_score = 0.0;
        double best_cost = 0.0;
        uint64_t best_delta = 0;
        bool best_fallback = false;
        for (uint32_t r = 0; r < R; ++r) {
            const uint64_t dn = engine->delta_n(r);
            if (dn == 0) continue;  // E1-lite: maintaining an empty Delta is wasted budget
            const uint64_t qmass =
                hist_windows == 0 ? 0 : engine->qmass_before(start_window, r, W);
            const double q_past =
                hist_windows == 0 ? 0.0 : static_cast<double>(qmass) / hist_windows;
            double q_hat = q_past;
            bool fallback = false;
            if (use_forecast) {
                if (rt->forecast.has(j, r)) {
                    q_hat = rt->forecast.get(j, r);
                } else {
                    fallback = true;
                    rt->forecast_fallback_rows += 1;
                }
            }
            const double score = q_hat * static_cast<double>(dn);
            const double cost =
                (static_cast<double>(engine->base_n(r)) + static_cast<double>(dn)) * rt->params.c_insert;
            double value = score;
            if (rt->policy == "bam_price") {
                value = score * rt->params.c_scan - rt->lambda * cost;
            }
            if (value > best_value + 1e-18) {
                best_value = value;
                best_r = static_cast<int>(r);
                best_score = score;
                best_cost = cost;
                best_delta = dn;
                best_fallback = fallback;
            }
        }
        out.lambda = rt->lambda;
        if (best_r < 0 || !(best_value > 0.0)) return out;
        Action action;
        if (mode == "refresh") {
            action = Action::refresh(best_r);
        } else if (mode == "merge") {
            action = Action::merge(best_r);
        } else {  // auto: choose the cheaper action type for the selected shard
            const double merge_cost = static_cast<double>(best_delta) * rt->params.c_insert;
            action = (merge_cost < best_cost) ? Action::merge(best_r) : Action::refresh(best_r);
        }
        out.action = action;
        out.score_chosen = best_score;
        out.cost_pred_ms = best_cost;
        out.delta_chosen = best_delta;
        out.forecast_fallback = best_fallback;
        if (rt->budget_total_ms > 0.0) {
            const double remaining = rt->budget_total_ms - rt->cum_spend_ms;
            if (best_cost > remaining) {
                out.action = Action::noop();  // soft hard-cap: reject over-budget maintenance
                out.budget_rejected = true;
            }
        }
        return out;
    }
    fail("unknown policy: \"" + rt->policy + "\"");
}

// ---------------------------------------------------------------- engine init
uint64_t ReplayEngine::build_seed(uint32_t shard) const {
    // Deterministic per-shard HNSW seed; never depends on iteration order.
    return cfg_.seed + 7919ULL * (static_cast<uint64_t>(shard) + 1ULL);
}

void ReplayEngine::init(const EngineConfig& cfg, const PoolData* pool, const TraceData* trace,
                        const GtData* gt, const InitData& init, const Centroids& centroids) {
    QBAM_CHECK(pool != nullptr && trace != nullptr, "engine init: null dataset");
    QBAM_CHECK(gt != nullptr, "engine init: null gt (pass an empty GtData when GT is unavailable)");
    cfg_ = cfg;
    pool_ = pool;
    trace_ = trace;
    gt_ = gt;
    router_.init(cfg_.R, cfg_.dim, centroids.data, cfg_.nprobe);
    hnsw_.M = cfg_.M;
    hnsw_.ef_construction = cfg_.ef_construction;
    hnsw_.ef_search = cfg_.ef_search;
    hnsw_.seed = cfg_.seed;
    QBAM_CHECK(cfg_.delta_mode == "scan" || cfg_.delta_mode == "hnsw",
               "engine init: delta_mode must be \"scan\" or \"hnsw\"");
    const bool delta_indexed = (cfg_.delta_mode == "hnsw");
    HnswParams delta_params = hnsw_;
    if (cfg_.delta_ef_search > 0) delta_params.ef_search = cfg_.delta_ef_search;

    live_ = LiveStore();
    shards_.assign(cfg_.R, ShardIndex());
    std::vector<std::vector<LiveRef>> per_shard(cfg_.R);
    for (const InitEntry& e : init.entries) {
        per_shard[e.shard].push_back(LiveRef{e.id, 1, e.vec_ref});
    }
    for (uint32_t r = 0; r < cfg_.R; ++r) {
        std::sort(per_shard[r].begin(), per_shard[r].end(),
                  [](const LiveRef& a, const LiveRef& b) { return a.id < b.id; });
        shards_[r].r = r;
        shards_[r].dim = cfg_.dim;
        shards_[r].base = ImmutableBaseHNSW::build(*pool_, per_shard[r], hnsw_);
        shards_[r].delta.clear();
        shards_[r].tombstone_n = 0;
        if (delta_indexed) {
            shards_[r].enable_delta_graph(delta_params, cfg_.delta_initial_capacity);
        }
        shards_[r].reset_block_counters();
    }
    for (const InitEntry& e : init.entries) {
        live_.assign(e.id, 1, e.shard, e.vec_ref);
    }
    window_q_accesses_.assign(trace_->n_windows, std::vector<uint32_t>(cfg_.R, 0));
    window_base_search_us_.assign(trace_->n_windows, std::vector<double>(cfg_.R, 0.0));
    window_delta_scan_us_.assign(trace_->n_windows, std::vector<double>(cfg_.R, 0.0));
    QBAM_LOG("[engine] init: R=%u dim=%u live=%llu windows=%u K=%u J=%u nprobe=%u", cfg_.R, cfg_.dim,
             static_cast<unsigned long long>(live_.alive_count()), trace_->n_windows, trace_->K, J(),
             cfg_.nprobe);
}

ReplayEngine ReplayEngine::clone_for_branch() const {
    // delta_mode=hnsw keeps a mutable per-shard Delta graph; sharing it across counterfactual
    // branches would alias writes, so cf mode is restricted to the frozen scan arm.
    for (const ShardIndex& s : shards_) {
        QBAM_CHECK(!s.delta_indexed,
                   "clone_for_branch: delta_mode=hnsw is not supported in cf mode");
    }
    ReplayEngine copy = *this;  // Base HNSW snapshots are shared immutable pointers.
    copy.reset_shard_block_counters();
    return copy;
}

void ReplayEngine::reset_shard_block_counters() {
    for (ShardIndex& s : shards_) s.reset_block_counters();
}

// ---------------------------------------------------------------- state mutation
void ReplayEngine::apply_crud(const TraceEvent& ev, double* cpu_ms) {
    QBAM_CHECK(cpu_ms != nullptr, "apply_crud: null cpu_ms");
    CpuWallTimer timer;
    timer.start();
    const float* vec = pool_->vec(ev.vec_ref);
    if (ev.type == EventType::Insert) {
        QBAM_CHECK(!live_.alive(ev.id), "insert of already-live id " + u64s(ev.id));
        const uint32_t shard = router_.assign_shard(vec);
        live_.assign(ev.id, ev.version, shard, ev.vec_ref);
        shards_[shard].insert_delta(*pool_, ev.id, ev.version, ev.vec_ref);
    } else if (ev.type == EventType::Update) {
        const LiveEntry* e = live_.find(ev.id);
        QBAM_CHECK(e != nullptr && e->alive, "update of non-live id " + u64s(ev.id));
        shards_[e->shard].invalidate_delta(ev.id);
        shards_[e->shard].base_has_stale = true;
        const uint32_t shard = router_.assign_shard(vec);
        live_.assign(ev.id, ev.version, shard, ev.vec_ref);
        shards_[shard].insert_delta(*pool_, ev.id, ev.version, ev.vec_ref);
    } else if (ev.type == EventType::Erase) {
        const LiveEntry* e = live_.find(ev.id);
        QBAM_CHECK(e != nullptr && e->alive, "erase of non-live id " + u64s(ev.id));
        shards_[e->shard].invalidate_delta(ev.id);
        shards_[e->shard].base_has_stale = true;
        shards_[e->shard].tombstone_n += 1;
        live_.erase(ev.id, ev.version);
    } else {
        fail("apply_crud called with a query event");
    }
    *cpu_ms = timer.cpu_ms();
}

RefreshResult ReplayEngine::apply_action(const Action& action, double* maintenance_cpu_ms) {
    QBAM_CHECK(maintenance_cpu_ms != nullptr, "apply_action: null maintenance_cpu_ms");
    *maintenance_cpu_ms = 0.0;
    RefreshResult rr;
    rr.shard = action.shard;
    if (action.kind == Action::Kind::Noop) return rr;

    if (action.kind == Action::Kind::Refresh) {
        QBAM_CHECK(action.shard >= 0 && static_cast<uint32_t>(action.shard) < cfg_.R,
                   "refresh shard out of range");
        rr = refresh_shard(&shards_[action.shard], *pool_, live_, hnsw_, cfg_.dim,
                           build_seed(static_cast<uint32_t>(action.shard)));
        *maintenance_cpu_ms = rr.build_cpu_ms;
        return rr;
    }
    if (action.kind == Action::Kind::RefreshPair) {
        QBAM_CHECK(action.shard >= 0 && static_cast<uint32_t>(action.shard) < cfg_.R,
                   "refresh_pair shard out of range");
        QBAM_CHECK(action.shard2 >= 0 && static_cast<uint32_t>(action.shard2) < cfg_.R,
                   "refresh_pair shard2 out of range");
        double total = 0.0;
        RefreshResult first = refresh_shard(&shards_[action.shard], *pool_, live_, hnsw_, cfg_.dim,
                                            build_seed(static_cast<uint32_t>(action.shard)));
        RefreshResult second = refresh_shard(&shards_[action.shard2], *pool_, live_, hnsw_, cfg_.dim,
                                             build_seed(static_cast<uint32_t>(action.shard2)));
        total = first.build_cpu_ms + second.build_cpu_ms;
        rr.build_cpu_ms = total;
        rr.wall_ms = first.wall_ms + second.wall_ms;
        rr.live_vectors_built = first.live_vectors_built + second.live_vectors_built;
        rr.base_before = second.base_before;
        rr.delta_before = second.delta_before;
        *maintenance_cpu_ms = total;
        return rr;
    }
    if (action.kind == Action::Kind::Merge) {
        // Diagnostic action (lead, Phase 2): incrementally merge live Delta vectors into Base.
        // The deep copy of the immutable Base is offline CF bookkeeping and is NOT charged;
        // only hnswlib resizeIndex + addPoint CPU is charged as maintenance.
        const uint32_t r = static_cast<uint32_t>(action.shard);
        QBAM_CHECK(action.shard >= 0 && r < cfg_.R, "merge shard out of range");
        ShardIndex& sh = shards_[r];
        std::vector<DeltaEntry> to_merge;
        for (const DeltaEntry& de : sh.delta) {
            if (!de.valid) continue;
            if (sh.base == nullptr || sh.base->versions().find(de.id) == sh.base->versions().end()) {
                // hnswlib labels are unique; updates of ids already in the Base stay in Delta
                // (documented limitation of the M1 merge diagnostic).
                to_merge.push_back(de);
            }
        }
        std::sort(to_merge.begin(), to_merge.end(), [](const DeltaEntry& a, const DeltaEntry& b) {
            if (a.id != b.id) return a.id < b.id;
            return a.version < b.version;
        });
        std::unordered_set<uint64_t> merged_ids;
        merged_ids.reserve(to_merge.size() * 2 + 1);
        for (const DeltaEntry& de : to_merge) merged_ids.insert(de.id);
        rr.base_before = sh.base_n();
        rr.delta_before = sh.delta_n();
        if (to_merge.empty()) {
            // Nothing new to insert; physically drop invalidated slots (query-invisible) and keep
            // the live entries whose label already exists in the Base (not mergeable).
            std::vector<DeltaEntry> remaining;
            remaining.reserve(sh.delta_index.size());
            for (const DeltaEntry& de : sh.delta) {
                if (de.valid) remaining.push_back(de);
            }
            sh.delta.swap(remaining);
            sh.delta.shrink_to_fit();
            sh.rebuild_delta_index();
            sh.rebuild_delta_graph(*pool_);
            rr.build_cpu_ms = 0.0;
            *maintenance_cpu_ms = 0.0;
            return rr;
        }
        // Deep copy of the immutable Base: reported separately as copy_cpu_ms (lead's G0
        // diagnostic wants maintenance_cpu_ms - copy_cpu_ms as the incremental-merge cost).
        std::shared_ptr<ImmutableBaseHNSW> new_base;
        {
            CpuWallTimer copy_timer;
            copy_timer.start();
            if (sh.base != nullptr) new_base = sh.base->deep_copy(*pool_);
            rr.copy_cpu_ms = copy_timer.cpu_ms();
        }
        CpuWallTimer timer;
        timer.start();
        if (new_base == nullptr) {
            new_base = ImmutableBaseHNSW::create(*pool_, to_merge.size(), hnsw_);
        } else {
            new_base->reserve(new_base->n() + to_merge.size());
        }
        for (const DeltaEntry& de : to_merge) {
            new_base->add_point(*pool_, de.vec_ref, de.id, de.version);
        }
        rr.build_cpu_ms = timer.cpu_ms();
        rr.wall_ms = timer.wall_ms();
        rr.live_vectors_built = to_merge.size();
        sh.base = new_base;
        // Keep only live Delta entries that could not be merged (label already in the Base:
        // in-place version updates). Dead/superseded entries are dropped: query-invisible.
        std::vector<DeltaEntry> remaining;
        remaining.reserve(sh.delta_index.size());
        for (const DeltaEntry& de : sh.delta) {
            if (!de.valid) continue;
            if (merged_ids.find(de.id) != merged_ids.end()) continue;
            remaining.push_back(de);
        }
        sh.delta.swap(remaining);
        sh.delta.shrink_to_fit();
        sh.rebuild_delta_index();
        sh.rebuild_delta_graph(*pool_);
        *maintenance_cpu_ms = rr.build_cpu_ms + rr.copy_cpu_ms;
        return rr;
    }
    fail("apply_action: unknown action kind");
}

// ---------------------------------------------------------------- query
QueryExecResult ReplayEngine::execute_query(uint32_t window, uint32_t gt_index,
                                            const float* query) {
    QueryExecResult res;
    CpuWallTimer query_timer;
    query_timer.start();

    const size_t k = cfg_.k;

    std::vector<uint32_t>& visited = scratch_visited_;
    router_.select(query, &visited);
    std::vector<char>& visited_mask = scratch_visited_mask_;
    visited_mask.assign(cfg_.R, 0);
    for (uint32_t r : visited) {
        visited_mask[r] = 1;
        shards_[r].q_accesses += 1;
        window_q_accesses_[window][r] += 1;
    }
    res.visited_shards = visited;

    const size_t contract_start = std::max<size_t>(4 * k, k + 32);
    const size_t k0 = cfg_.overfetch_start > 0
                          ? std::max<size_t>(cfg_.overfetch_start, k)
                          : contract_start;
    size_t total_possible = 0;
    for (uint32_t r : visited) {
        total_possible += shards_[r].base_n() + shards_[r].delta_n();
    }

    size_t kk = k0;
    int stale_total = 0;
    int requested_total = 0;
    bool underfilled = false;
    size_t expansions = 0;
    size_t valid_total = 0;
    double base_us = 0.0;
    double delta_us = 0.0;
    std::vector<std::pair<float, uint64_t>>& base_hits = scratch_base_hits_;
    std::vector<QueryCand> picked;
    std::priority_queue<QueryCand, std::vector<QueryCand>, QueryCandLess> topk;

    // Bounded max-heap top-k while scanning: O(log k) per candidate instead of sorting every
    // candidate (full sort of ~46K candidates alone was ~2.4 ms/query at w36 on the 100K trace).
    // Overfetch retries re-scan from scratch; they only happen for tombstone-heavy shards.
    while (true) {
        topk = decltype(topk)();
        valid_total = 0;
        stale_total = 0;
        requested_total = 0;
        base_us = 0.0;
        delta_us = 0.0;
        auto consider = [&](const QueryCand& cand) {
            ++valid_total;
            if (topk.size() < k) {
                topk.push(cand);
            } else if (QueryCandLess()(cand, topk.top())) {
                topk.pop();
                topk.push(cand);
            }
        };
        for (uint32_t r : visited) {
            ShardIndex& sh = shards_[r];
            if (sh.base != nullptr && sh.base->n() > 0) {
                CpuWallTimer base_timer;
                base_timer.start();
                sh.base->search(query, kk, &base_hits);
                const double base_wall = base_timer.wall_us();
                base_us += base_wall;
                sh.base_search_wall_us += base_wall;
                requested_total += static_cast<int>(kk);
                if (!sh.base_has_stale) {
                    // Clean Base snapshot: all labels are live/current, no hash lookups needed.
                    for (const auto& hit : base_hits) {
                        consider(QueryCand{hit.first, hit.second, r, false});
                    }
                } else {
                    const std::unordered_map<uint64_t, uint64_t>& versions = sh.base->versions();
                    for (const auto& hit : base_hits) {
                        const uint64_t id = hit.second;
                        auto vit = versions.find(id);
                        const LiveEntry* e = live_.find(id);
                        if (vit == versions.end() || e == nullptr || !e->alive || e->shard != r ||
                            e->version != vit->second) {
                            ++stale_total;
                            continue;
                        }
                        consider(QueryCand{hit.first, id, r, false});
                    }
                }
            }
            if (sh.delta_graph != nullptr) {
                // Probe Delta HNSW graph.
                CpuWallTimer delta_timer;
                delta_timer.start();
                std::vector<std::pair<float, uint64_t>>& delta_hits = scratch_delta_hits_;
                sh.delta_graph->search(query, kk, &delta_hits);
                const double delta_wall = delta_timer.wall_us();
                delta_us += delta_wall;
                sh.delta_scan_wall_us += delta_wall;
                requested_total += static_cast<int>(kk);
                for (const auto& hit : delta_hits) {
                    consider(QueryCand{hit.first, hit.second, r, true});
                }
            }
            if (!sh.delta_indexed) {
                CpuWallTimer delta_timer;
                delta_timer.start();
                const DeltaEntry* entries = sh.delta.data();
                const size_t n_entries = sh.delta.size();
                for (size_t i = 0; i < n_entries; ++i) {
                    const DeltaEntry& de = entries[i];
                    if (!de.valid) continue;
                    const float d = l2_squared(query, sh.delta_vec(de), cfg_.dim);
                    consider(QueryCand{d, de.id, r, true});
                }
                const double delta_wall = delta_timer.wall_us();
                delta_us += delta_wall;
                sh.delta_scan_wall_us += delta_wall;
            }
        }
        if (valid_total >= k || kk >= total_possible) break;
        size_t next = std::max<size_t>(kk * 2, kk + 64);
        next = std::min(next, total_possible);
        if (next <= kk) break;
        kk = next;
        ++expansions;
    }

    CpuWallTimer selection_timer;
    selection_timer.start();
    picked.clear();
    picked.reserve(topk.size());
    while (!topk.empty()) {
        picked.push_back(topk.top());
        topk.pop();
    }
    std::sort(picked.begin(), picked.end(), QueryCandLess());
    for (const QueryCand& cand : picked) {
        bool duplicate = false;
        for (uint64_t rid : res.ids) {
            if (rid == cand.id) {
                duplicate = true;
                break;
            }
        }
        if (duplicate) continue;
        res.ids.push_back(cand.id);
        res.dists.push_back(cand.dist);
        res.from_delta.push_back(cand.from_delta);
        const LiveEntry* e = live_.find(cand.id);
        res.versions.push_back(e != nullptr ? e->version : 0);
    }
    underfilled = res.ids.size() < k;

    res.base_candidates_requested = requested_total;
    res.stale_candidates_filtered = stale_total;
    res.overfetch_expansions = static_cast<int>(expansions);
    res.underfilled = underfilled;
    res.base_search_us = base_us;
    res.delta_scan_us = delta_us;
    res.selection_us = selection_timer.wall_us();
    res.valid_candidates = static_cast<uint32_t>(valid_total);
    res.cpu_us = query_timer.cpu_ms() * 1000.0;
    res.wall_us = query_timer.wall_us();

    // ---- evaluation-only section (not charged to query_cpu_ms / wall_us) ----
    // GT-independent observable: fraction of returned ids served from Delta.
    if (!res.from_delta.empty()) {
        size_t from_delta_count = 0;
        for (bool b : res.from_delta) {
            if (b) ++from_delta_count;
        }
        res.returned_delta_fraction =
            static_cast<double>(from_delta_count) / static_cast<double>(res.from_delta.size());
    }
    if (gt_ != nullptr && gt_->n_queries > 0) {
        const QueryQuality q = attribute_query(cfg_.k, *gt_, gt_index, res.ids, res.from_delta,
                                               visited_mask, live_);
        res.recall = q.recall;
        res.route_upper = q.route_upper;
        res.index_loss = q.index_loss;
    }
    return res;
}

WindowMetrics ReplayEngine::process_window(uint32_t window, LineSink* query_sink,
                                           const std::string& run_id) {
    WindowMetrics m;
    m.window = window;
    const uint32_t R = cfg_.R;
    std::vector<double> base_us_baseline(R, 0.0);
    std::vector<double> delta_us_baseline(R, 0.0);
    for (uint32_t r = 0; r < R; ++r) {
        base_us_baseline[r] = shards_[r].base_search_wall_us;
        delta_us_baseline[r] = shards_[r].delta_scan_wall_us;
    }
    m.shard_delta_n.assign(R, 0);
    m.shard_base_n.assign(R, 0);
    m.shard_q_accesses.assign(R, 0);
    std::vector<double> walls;
    bool counts_captured = false;
    double recall_sum = 0.0, route_sum = 0.0, loss_sum = 0.0, delta_frac_sum = 0.0;
    uint32_t gt_index = trace_->window_query_start[window];
    const uint32_t begin = trace_->window_begin[window];
    const uint32_t end = trace_->window_begin[window + 1];
    for (uint32_t i = begin; i < end; ++i) {
        const TraceEvent& ev = trace_->events[i];
        if (ev.type == EventType::Query) {
            if (!counts_captured) {
                for (uint32_t r = 0; r < R; ++r) {
                    m.shard_delta_n[r] = static_cast<uint32_t>(shards_[r].delta_n());
                    m.shard_base_n[r] = static_cast<uint32_t>(shards_[r].base_n());
                }
                counts_captured = true;
            }
            const QueryExecResult r = execute_query(window, gt_index, pool_->vec(ev.vec_ref));
            m.query_cpu_ms += r.cpu_us / 1000.0;
            m.base_search_cpu_ms += r.base_search_us / 1000.0;
            m.delta_scan_cpu_ms += r.delta_scan_us / 1000.0;
            m.selection_cpu_ms += r.selection_us / 1000.0;
            if (r.valid_candidates > m.candidate_count_max) m.candidate_count_max = r.valid_candidates;
            walls.push_back(r.wall_us);
            recall_sum += r.recall;
            route_sum += r.route_upper;
            loss_sum += r.index_loss;
            delta_frac_sum += r.returned_delta_fraction;
            if (r.recall < cfg_.r_min) ++m.recall_violations;
            if (r.underfilled) ++m.underfilled_queries;
            if (r.overfetch_expansions > 0) ++m.overfetch_count;
            if (query_sink != nullptr) {
                query_sink->write_line(json_query_record(run_id, window, gt_index, r));
            }
            ++m.query_count;
            ++gt_index;
        } else {
            double ucpu = 0.0;
            apply_crud(ev, &ucpu);
            m.update_cpu_ms += ucpu;
        }
    }
    if (!counts_captured) {
        for (uint32_t r = 0; r < R; ++r) {
            m.shard_delta_n[r] = static_cast<uint32_t>(shards_[r].delta_n());
            m.shard_base_n[r] = static_cast<uint32_t>(shards_[r].base_n());
        }
    }
    m.shard_q_accesses = window_q_accesses_[window];
    for (uint32_t r = 0; r < R; ++r) {
        window_base_search_us_[window][r] = shards_[r].base_search_wall_us - base_us_baseline[r];
        window_delta_scan_us_[window][r] = shards_[r].delta_scan_wall_us - delta_us_baseline[r];
    }
    m.wall_us_all = walls;
    std::vector<double> walls_sorted = walls;
    m.p = compute_percentiles(&walls_sorted);
    if (m.query_count > 0) {
        const double n = static_cast<double>(m.query_count);
        m.recall = recall_sum / n;
        m.route_upper = route_sum / n;
        m.index_loss = loss_sum / n;
        m.returned_delta_fraction = delta_frac_sum / n;
    }
    return m;
}

// ---------------------------------------------------------------- block aggregation
BlockMetrics ReplayEngine::aggregate_block(uint32_t j, const Action& action, double policy_cpu_ms,
                                           double maintenance_cpu_ms, const std::string& state_hash,
                                           double budget_nominal_cpu_ms,
                                           const std::vector<WindowMetrics>& windows,
                                           const std::vector<ShardSnapshot>& shard_snapshot) {
    BlockMetrics b;
    b.j = j;
    b.window_start = j * trace_->K;
    b.n_windows = static_cast<uint32_t>(windows.size());
    b.action = action;
    b.state_hash = state_hash;
    b.policy_cpu_ms = policy_cpu_ms;
    b.maintenance_cpu_ms = maintenance_cpu_ms;
    b.budget_nominal_cpu_ms = budget_nominal_cpu_ms;
    b.budget_actual_cpu_ms = maintenance_cpu_ms + policy_cpu_ms;
    if (budget_nominal_cpu_ms > 0.0) {
        b.budget_unused_cpu_ms = budget_nominal_cpu_ms - b.budget_actual_cpu_ms;
        if (b.budget_unused_cpu_ms < 0.0) b.budget_unused_cpu_ms = 0.0;
        b.overshoot = b.budget_actual_cpu_ms > budget_nominal_cpu_ms;
    } else {
        // Uncalibrated budget (config budget.c_ref_cpu_ms = null/0): RE accounting is not
        // available yet, so do not emit a spurious overshoot flag (documented in NOTES §2.4).
        b.budget_unused_cpu_ms = 0.0;
        b.overshoot = false;
    }
    b.budget_rejected = false;
    double recall_sum = 0.0, route_sum = 0.0, loss_sum = 0.0, delta_frac_sum = 0.0;
    uint64_t qcount = 0;
    uint32_t violations = 0;
    std::vector<double> walls;
    for (const WindowMetrics& w : windows) {
        b.query_cpu_ms += w.query_cpu_ms;
        b.update_cpu_ms += w.update_cpu_ms;
        recall_sum += w.recall * static_cast<double>(w.query_count);
        route_sum += w.route_upper * static_cast<double>(w.query_count);
        loss_sum += w.index_loss * static_cast<double>(w.query_count);
        delta_frac_sum += w.returned_delta_fraction * static_cast<double>(w.query_count);
        qcount += w.query_count;
        violations += w.recall_violations;
        b.overfetch_count += w.overfetch_count;
        b.underfilled_queries += w.underfilled_queries;
        walls.insert(walls.end(), w.wall_us_all.begin(), w.wall_us_all.end());
    }
    if (qcount > 0) {
        const double n = static_cast<double>(qcount);
        b.recall = recall_sum / n;
        b.route_upper = route_sum / n;
        b.index_loss = loss_sum / n;
        b.returned_delta_fraction = delta_frac_sum / n;
        b.recall_violation_rate = static_cast<double>(violations) / n;
    }
    b.p = compute_percentiles(&walls);
    b.shards = shard_snapshot;
    return b;
}

void ReplayEngine::capture_shard_counts(std::vector<ShardSnapshot>* out) const {
    QBAM_CHECK(out != nullptr, "capture_shard_counts: null");
    out->clear();
    out->resize(cfg_.R);
    for (uint32_t r = 0; r < cfg_.R; ++r) {
        ShardSnapshot& s = (*out)[r];
        s.r = r;
        s.base_n = shards_[r].base_n();
        s.delta_n = shards_[r].delta_n();
        s.tombstone_n = shards_[r].tombstone_n;
        s.q_accesses = 0;
        s.base_search_wall_us = 0.0;
        s.delta_scan_wall_us = 0.0;
    }
}

void ReplayEngine::fill_shard_counters(std::vector<ShardSnapshot>* io) const {
    QBAM_CHECK(io != nullptr, "fill_shard_counters: null");
    QBAM_CHECK(io->size() == cfg_.R, "fill_shard_counters: size mismatch");
    for (uint32_t r = 0; r < cfg_.R; ++r) {
        ShardSnapshot& s = (*io)[r];
        s.q_accesses = shards_[r].q_accesses;
        s.base_search_wall_us = shards_[r].base_search_wall_us;
        s.delta_scan_wall_us = shards_[r].delta_scan_wall_us;
    }
}

// ---------------------------------------------------------------- state hash
std::string ReplayEngine::state_hash() const {
    Sha256 hasher;
    std::string header = "QBAMSTATE1\nlive_n " + u64s(live_.alive_count()) + "\n";
    hasher.update(header.data(), header.size());
    char line[96];
    const std::vector<uint64_t>& ids = live_.alive_ids_sorted();
    for (uint64_t id : ids) {
        const LiveEntry* e = live_.find(id);
        const int n = std::snprintf(line, sizeof(line), "%llu %llu %u\n",
                                    static_cast<unsigned long long>(id),
                                    static_cast<unsigned long long>(e->version), e->shard);
        hasher.update(line, static_cast<size_t>(n));
    }
    for (uint32_t r = 0; r < cfg_.R; ++r) {
        const int n = std::snprintf(line, sizeof(line), "s %u %llu %llu %llu\n", r,
                                    static_cast<unsigned long long>(shards_[r].base_n()),
                                    static_cast<unsigned long long>(shards_[r].delta_n()),
                                    static_cast<unsigned long long>(shards_[r].tombstone_n));
        hasher.update(line, static_cast<size_t>(n));
    }
    const std::string hex = hasher.hexdigest();
    return hex.substr(0, 16);
}

void ReplayEngine::recent_shard_stats(uint32_t before_window, uint32_t lookback,
                                      std::vector<ShardRecentStats>* out) const {
    QBAM_CHECK(out != nullptr, "recent_shard_stats: null");
    out->clear();
    out->resize(cfg_.R);
    const uint32_t begin = before_window > lookback ? before_window - lookback : 0;
    const uint32_t end = std::min<uint32_t>(before_window, trace_->n_windows);
    for (uint32_t r = 0; r < cfg_.R; ++r) {
        ShardRecentStats s;
        s.r = r;
        for (uint32_t w = begin; w < end; ++w) {
            s.q_accesses_recent += window_q_accesses_[w][r];
            s.base_search_us_recent += window_base_search_us_[w][r];
            s.delta_scan_us_recent += window_delta_scan_us_[w][r];
        }
        (*out)[r] = s;
    }
}

uint64_t ReplayEngine::qmass_before(uint32_t window, uint32_t shard, uint32_t lookback) const {
    if (window == 0 || lookback == 0 || shard >= cfg_.R) return 0;
    const uint32_t begin = window > lookback ? window - lookback : 0;
    uint64_t sum = 0;
    for (uint32_t w = begin; w < window; ++w) sum += window_q_accesses_[w][shard];
    return sum;
}

uint64_t ReplayEngine::window_q_accesses(uint32_t window, uint32_t shard) const {
    QBAM_CHECK(window < window_q_accesses_.size() && shard < cfg_.R,
               "window_q_accesses out of range");
    return window_q_accesses_[window][shard];
}

// ---------------------------------------------------------------- JSONL emission
std::string json_run_record(const RunManifest& m) {
    std::string o;
    o.reserve(1024);
    o += "{\"type\":\"run\"";
    o += ",\"run_id\":\"" + escape_json_string(m.run_id) + "\"";
    o += ",\"mode\":\"" + escape_json_string(m.mode) + "\"";
    o += ",\"policy\":\"" + escape_json_string(m.policy) + "\"";
    o += ",\"git_sha\":\"" + escape_json_string(m.git_sha) + "\"";
    o += ",\"config_sha256\":\"" + escape_json_string(m.config_sha256) + "\"";
    o += ",\"dataset_sha256\":\"" + escape_json_string(m.dataset_sha256) + "\"";
    o += ",\"trace_sha256\":\"" + escape_json_string(m.trace_sha256) + "\"";
    o += ",\"gt_sha256\":\"" + escape_json_string(m.gt_sha256) + "\"";
    o += ",\"R\":" + u32s(m.R);
    o += ",\"dim\":" + u32s(m.dim);
    o += ",\"k\":" + u32s(m.k);
    o += ",\"M\":" + u64s(m.M);
    o += ",\"efConstruction\":" + u64s(m.ef_construction);
    o += ",\"efSearch\":" + u64s(m.ef_search);
    o += ",\"delta_mode\":\"" + escape_json_string(m.delta_mode) + "\"";
    o += ",\"deltaEfSearch\":" + u64s(m.delta_ef_search);
    o += ",\"nprobe\":" + u32s(m.nprobe);
    o += ",\"seed\":" + u64s(m.seed);
    o += ",\"threads\":" + u32s(static_cast<uint32_t>(m.threads));
    o += ",\"pin_cpu\":" + u32s(static_cast<uint32_t>(m.pin_cpu));
    o += ",\"n_events\":" + u32s(m.n_events);
    o += ",\"n_windows\":" + u32s(m.n_windows);
    o += ",\"K\":" + u32s(m.K);
    o += ",\"J\":" + u32s(m.J);
    o += ",\"wall_s\":" + fmt6(m.wall_s);
    o += ",\"platform\":\"" + escape_json_string(m.platform) + "\"";
    o += ",\"build\":\"" + escape_json_string(m.build) + "\"";
    o += ",\"policy_params\":" + (m.policy_params_json.empty() ? std::string("{}") : m.policy_params_json);
    o += ",\"return_code\":" + u32s(static_cast<uint32_t>(m.return_code));
    o += "}";
    return o;
}

std::string json_block_record(const std::string& run_id, const BlockMetrics& b) {
    std::string o;
    o.reserve(2048);
    o += "{\"type\":\"block\"";
    o += ",\"run_id\":\"" + escape_json_string(run_id) + "\"";
    o += ",\"j\":" + u32s(b.j);
    o += ",\"window_start\":" + u32s(b.window_start);
    o += ",\"action\":\"" + b.action.name() + "\"";
    o += ",\"shard\":" + std::to_string(b.action.shard);
    o += ",\"policy_cpu_ms\":" + fmt6(b.policy_cpu_ms);
    o += ",\"maintenance_cpu_ms\":" + fmt6(b.maintenance_cpu_ms);
    o += ",\"budget_nominal_cpu_ms\":" + fmt6(b.budget_nominal_cpu_ms);
    o += ",\"budget_actual_cpu_ms\":" + fmt6(b.budget_actual_cpu_ms);
    o += ",\"budget_unused_cpu_ms\":" + fmt6(b.budget_unused_cpu_ms);
    o += ",\"overshoot\":" + jbool(b.overshoot);
    o += ",\"budget_rejected\":" + jbool(b.budget_rejected);
    o += ",\"query_cpu_ms\":" + fmt6(b.query_cpu_ms);
    o += ",\"update_cpu_ms\":" + fmt6(b.update_cpu_ms);
    o += ",\"p50_wall_us\":" + fmt6(b.p.p50);
    o += ",\"p95_wall_us\":" + fmt6(b.p.p95);
    o += ",\"p99_wall_us\":" + fmt6(b.p.p99);
    o += ",\"recall_at_10\":" + fmt6(b.recall);
    o += ",\"recall_violation_rate\":" + fmt6(b.recall_violation_rate);
    o += ",\"returned_delta_fraction\":" + fmt6(b.returned_delta_fraction);
    o += ",\"gt_delta_fraction\":null";
    o += ",\"route_upper\":" + fmt6(b.route_upper);
    o += ",\"index_loss\":" + fmt6(b.index_loss);
    o += ",\"overfetch_count\":" + u64s(b.overfetch_count);
    o += ",\"underfilled_queries\":" + u64s(b.underfilled_queries);
    o += ",\"state_hash\":\"" + escape_json_string(b.state_hash) + "\"";
    o += ",\"score_chosen\":" + fmt6(b.score_chosen);
    o += ",\"cost_pred_ms\":" + fmt6(b.cost_pred_ms);
    o += ",\"lambda\":" + fmt6(b.lambda);
    o += ",\"delta_chosen\":" + u64s(b.delta_chosen);
    o += ",\"shards\":";
    append_block_shards(&o, b.shards);
    o += "}";
    return o;
}

std::string json_window_record(const std::string& run_id, const WindowMetrics& w, uint32_t j,
                               const Action& action, int action_index, uint32_t h) {
    std::string o;
    o.reserve(1024);
    o += "{\"type\":\"window\"";
    o += ",\"run_id\":\"" + escape_json_string(run_id) + "\"";
    o += ",\"j\":" + u32s(j);
    o += ",\"action\":\"" + action.name() + "\"";
    o += ",\"shard\":" + std::to_string(action.shard);
    o += ",\"h\":" + u32s(h);
    o += ",\"action_index\":" + std::to_string(action_index);
    o += ",\"window\":" + u32s(w.window);
    o += ",\"query_count\":" + u32s(w.query_count);
    o += ",\"query_cpu_ms\":" + fmt6(w.query_cpu_ms);
    o += ",\"update_cpu_ms\":" + fmt6(w.update_cpu_ms);
    o += ",\"p50_wall_us\":" + fmt6(w.p.p50);
    o += ",\"p95_wall_us\":" + fmt6(w.p.p95);
    o += ",\"p99_wall_us\":" + fmt6(w.p.p99);
    o += ",\"recall_at_10\":" + fmt6(w.recall);
    o += ",\"route_upper\":" + fmt6(w.route_upper);
    o += ",\"index_loss\":" + fmt6(w.index_loss);
    o += ",\"returned_delta_fraction\":" + fmt6(w.returned_delta_fraction);
    o += ",\"underfilled_queries\":" + u32s(w.underfilled_queries);
    o += ",\"base_search_cpu_ms\":" + fmt6(w.base_search_cpu_ms);
    o += ",\"delta_scan_cpu_ms\":" + fmt6(w.delta_scan_cpu_ms);
    o += ",\"selection_cpu_ms\":" + fmt6(w.selection_cpu_ms);
    o += ",\"candidate_count_max\":" + u32s(w.candidate_count_max);
    o += ",\"shard_delta_n\":";
    append_u32_array(&o, w.shard_delta_n);
    o += ",\"shard_base_n\":";
    append_u32_array(&o, w.shard_base_n);
    o += ",\"shard_q_accesses\":";
    append_u32_array(&o, w.shard_q_accesses);
    o += "}";
    return o;
}

std::string json_branch_record(const std::string& run_id, const BranchRecord& br) {
    std::string o;
    o.reserve(2048);
    o += "{\"type\":\"branch\"";
    o += ",\"run_id\":\"" + escape_json_string(run_id) + "\"";
    o += ",\"j\":" + u32s(br.j);
    o += ",\"action\":\"" + br.action.name() + "\"";
    o += ",\"action_str\":\"" + escape_json_string(br.action.config_string()) + "\"";
    o += ",\"shard\":" + std::to_string(br.action.shard);
    o += ",\"shard2\":" + std::to_string(br.action.shard2);
    o += ",\"state_hash\":\"" + escape_json_string(br.state_hash) + "\"";
    o += ",\"maintenance_cpu_ms\":" + fmt6(br.maintenance_cpu_ms);
    o += ",\"copy_cpu_ms\":" + fmt6(br.copy_cpu_ms);
    o += ",\"policy_cpu_ms\":" + fmt6(br.policy_cpu_ms);
    o += ",\"windows\":[";
    for (size_t i = 0; i < br.windows.size(); ++i) {
        const WindowMetrics& w = br.windows[i];
        if (i > 0) o += ',';
        o += "{\"h\":" + u32s(static_cast<uint32_t>(i + 1));
        o += ",\"query_cpu_ms\":" + fmt6(w.query_cpu_ms);
        o += ",\"recall_at_10\":" + fmt6(w.recall);
        o += ",\"p95_wall_us\":" + fmt6(w.p.p95);
        o += ",\"returned_delta_fraction\":" + fmt6(w.returned_delta_fraction);
        o += ",\"route_upper\":" + fmt6(w.route_upper);
        o += ",\"index_loss\":" + fmt6(w.index_loss);
        o += "}";
    }
    o += "]}";
    return o;
}

std::string json_state_record(const std::string& run_id, uint32_t j, uint32_t window_start,
                              const std::string& state_hash,
                              const std::vector<ShardSnapshot>& counts,
                              const std::vector<ShardRecentStats>& recent) {
    std::string o;
    o.reserve(1536);
    o += "{\"type\":\"state\"";
    o += ",\"run_id\":\"" + escape_json_string(run_id) + "\"";
    o += ",\"j\":" + u32s(j);
    o += ",\"window_start\":" + u32s(window_start);
    o += ",\"state_hash\":\"" + escape_json_string(state_hash) + "\"";
    o += ",\"shards\":[";
    for (size_t i = 0; i < counts.size(); ++i) {
        const ShardSnapshot& c = counts[i];
        if (i > 0) o += ',';
        o += "{\"r\":" + u32s(c.r) + ",\"base_n\":" + u64s(c.base_n) +
             ",\"delta_n\":" + u64s(c.delta_n) + ",\"tombstone_n\":" + u64s(c.tombstone_n);
        if (i < recent.size()) {
            o += ",\"q_accesses_recent8\":" + u64s(recent[i].q_accesses_recent);
            o += ",\"delta_scan_cpu_us_recent\":" + fmt6(recent[i].delta_scan_us_recent);
            o += ",\"base_search_cpu_us_recent\":" + fmt6(recent[i].base_search_us_recent);
        } else {
            o += ",\"q_accesses_recent8\":0,\"delta_scan_cpu_us_recent\":0.000000"
                 ",\"base_search_cpu_us_recent\":0.000000";
        }
        o += "}";
    }
    o += "]}";
    return o;
}

std::string json_query_record(const std::string& run_id, uint32_t window, uint32_t q_index,
                              const QueryExecResult& r) {
    std::string o;
    o.reserve(1024);
    o += "{\"type\":\"query\"";
    o += ",\"run_id\":\"" + escape_json_string(run_id) + "\"";
    o += ",\"window\":" + u32s(window);
    o += ",\"q_index\":" + u32s(q_index);
    o += ",\"returned\":[";
    for (size_t i = 0; i < r.ids.size(); ++i) {
        if (i > 0) o += ',';
        o += u64s(r.ids[i]);
    }
    o += "],\"distances\":[";
    for (size_t i = 0; i < r.dists.size(); ++i) {
        if (i > 0) o += ',';
        o += fmt6(static_cast<double>(r.dists[i]));
    }
    o += "],\"from_delta\":[";
    for (size_t i = 0; i < r.from_delta.size(); ++i) {
        if (i > 0) o += ',';
        o += jbool(r.from_delta[i]);
    }
    o += "]";
    o += ",\"wall_us\":" + fmt6(r.wall_us);
    o += ",\"cpu_us\":" + fmt6(r.cpu_us);
    o += ",\"base_candidates_requested\":" + std::to_string(r.base_candidates_requested);
    o += ",\"stale_candidates_filtered\":" + std::to_string(r.stale_candidates_filtered);
    o += ",\"underfilled\":" + jbool(r.underfilled);
    o += "}";
    return o;
}

}  // namespace qbam
