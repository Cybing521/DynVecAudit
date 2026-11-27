// Q-BAM M1 engine -- shared helpers for unit tests (in-memory mini fixtures, no file I/O).
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/live_store.h"
#include "core/router.h"
#include "eval/cpu_meter.h"
#include "eval/exact_gt_check.h"
#include "io/events.h"
#include "replay/cf_runner.h"
#include "replay/replay_engine.h"

namespace qbam {
namespace test {

inline void check(bool cond, const std::string& msg, const char* file, int line) {
    if (!cond) {
        std::fprintf(stderr, "TEST FAILED at %s:%d: %s\n", file, line, msg.c_str());
        std::exit(1);
    }
}

inline void check_near(double a, double b, double tol, const std::string& what) {
    if (!(std::fabs(a - b) <= tol)) {
        std::fprintf(stderr, "TEST FAILED: %s: %.9f != %.9f (tol %.3g)\n", what.c_str(), a, b,
                     tol);
        std::exit(1);
    }
}

}  // namespace test
}  // namespace qbam

#define TCHECK(cond) ::qbam::test::check((cond), #cond, __FILE__, __LINE__)
#define TCHECK_NEAR(a, b, tol) ::qbam::test::check_near((a), (b), (tol), #a " ~= " #b)
#define TCHECK_MSG(cond, msg) ::qbam::test::check((cond), (msg), __FILE__, __LINE__)

namespace qbam {
namespace test {

// In-memory equivalent of the five contract files.
struct MiniDataset {
    PoolData pool;
    InitData init;
    Centroids centroids;
    TraceData trace;
    GtData gt;
};

// Small in-memory dataset builder. Vectors live in one PoolData; every event refers to it.
class MiniBuilder {
public:
    MiniBuilder(uint32_t dim, uint32_t R) : dim_(dim), R_(R) {
        centroids_.assign(static_cast<size_t>(R) * dim, 0.0f);
        // Default centroids: shard r sits at 10*r on dimension 0 (enough for routing tests).
        for (uint32_t r = 0; r < R; ++r) centroids_[static_cast<size_t>(r) * dim] = 10.0f * r;
        pool_.dim = dim;
        pool_.nvec = 0;
    }

    void set_centroid(uint32_t r, const std::vector<float>& c) {
        TCHECK(c.size() == dim_);
        for (uint32_t d = 0; d < dim_; ++d) centroids_[static_cast<size_t>(r) * dim_ + d] = c[d];
    }

    uint32_t add_vec(const std::vector<float>& pos) {
        TCHECK(pos.size() == dim_);
        const uint32_t ref = pool_.nvec;
        pool_.data.insert(pool_.data.end(), pos.begin(), pos.end());
        pool_.nvec += 1;
        return ref;
    }

    uint64_t add_init(uint32_t shard, const std::vector<float>& pos) {
        const uint64_t id = next_id_++;
        const uint32_t ref = add_vec(pos);
        init_entries_.push_back(InitEntry{id, shard, ref});
        return id;
    }

    void event_insert(uint32_t window, uint64_t id, uint64_t version, uint32_t vec_ref) {
        events_.push_back(TraceEvent{window, EventType::Insert, id, version, vec_ref});
    }
    void event_update(uint32_t window, uint64_t id, uint64_t version, uint32_t vec_ref) {
        events_.push_back(TraceEvent{window, EventType::Update, id, version, vec_ref});
    }
    void event_erase(uint32_t window, uint64_t id, uint64_t version) {
        events_.push_back(TraceEvent{window, EventType::Erase, id, version, 0xFFFFFFFF});
    }
    void event_query(uint32_t window, const std::vector<float>& qpos) {
        const uint32_t ref = add_vec(qpos);
        events_.push_back(TraceEvent{window, EventType::Query, 0, 0, ref});
    }

    uint32_t dim() const { return dim_; }
    uint32_t R() const { return R_; }

    // Builds TraceData + exact gt.bin content (brute force over the live set at each query).
    MiniDataset finalize(uint32_t n_windows, uint32_t K, uint32_t k) const {
        MiniDataset ds;
        ds.pool = pool_;
        ds.init.dim = dim_;
        ds.init.n = static_cast<uint32_t>(init_entries_.size());
        ds.init.R = R_;
        ds.init.entries = init_entries_;
        ds.centroids.R = R_;
        ds.centroids.dim = dim_;
        ds.centroids.data = centroids_;
        ds.trace.dim = dim_;
        ds.trace.n_events = static_cast<uint32_t>(events_.size());
        ds.trace.n_windows = n_windows;
        ds.trace.K = K;
        ds.trace.events = events_;
        build_window_index(&ds.trace);

        // Exact gt by simulating the live set in event order.
        LiveStore live;
        for (const InitEntry& e : init_entries_) live.assign(e.id, 1, e.shard, e.vec_ref);
        ds.gt.k = k;
        ds.gt.n_queries = ds.trace.total_queries;
        ds.gt.dim = dim_;
        ds.gt.ids.reserve(static_cast<size_t>(ds.gt.n_queries) * k);
        ds.gt.squared_l2.reserve(static_cast<size_t>(ds.gt.n_queries) * k);
        for (const TraceEvent& ev : events_) {
            if (ev.type == EventType::Insert) {
                live.assign(ev.id, ev.version, 0, ev.vec_ref);
            } else if (ev.type == EventType::Update) {
                live.assign(ev.id, ev.version, 0, ev.vec_ref);
            } else if (ev.type == EventType::Erase) {
                live.erase(ev.id, ev.version);
            } else {
                ExactTopK top = compute_exact_topk(ds.pool, live, ds.pool.vec(ev.vec_ref), k);
                for (size_t i = 0; i < k; ++i) {
                    if (i < top.ids.size()) {
                        ds.gt.ids.push_back(top.ids[i]);
                        ds.gt.squared_l2.push_back(top.squared_l2[i]);
                    } else {
                        ds.gt.ids.push_back(0);
                        ds.gt.squared_l2.push_back(0.0f);
                    }
                }
            }
        }
        TCHECK(ds.gt.ids.size() == static_cast<size_t>(ds.gt.n_queries) * k);
        return ds;
    }

private:
    uint32_t dim_;
    uint32_t R_;
    PoolData pool_;
    std::vector<InitEntry> init_entries_;
    std::vector<TraceEvent> events_;
    std::vector<float> centroids_;
    uint64_t next_id_ = 1;
};

inline EngineConfig make_config(uint32_t R, uint32_t dim, uint32_t k = 3, uint32_t nprobe = 0) {
    EngineConfig c;
    c.R = R;
    c.dim = dim;
    c.k = k;
    c.nprobe = nprobe == 0 ? R : nprobe;
    c.M = 8;
    c.ef_construction = 64;
    c.ef_search = 64;
    c.seed = 7;
    c.run_id = "unit_test";
    c.kappa_RE = 2.0;
    c.c_ref_cpu_ms = 5.0;
    return c;
}

inline ReplayEngine make_engine(const MiniDataset& ds, const EngineConfig& cfg) {
    ReplayEngine e;
    e.init(cfg, &ds.pool, &ds.trace, &ds.gt, ds.init, ds.centroids);
    return e;
}

// Applies the CRUD events of one window (queries untouched).
inline void apply_window_crud(ReplayEngine* engine, const TraceData& trace, uint32_t window) {
    for (uint32_t i = trace.window_begin_of(window); i < trace.window_end_of(window); ++i) {
        const TraceEvent& ev = trace.events[i];
        if (ev.type == EventType::Query) continue;
        double cpu = 0.0;
        engine->apply_crud(ev, &cpu);
    }
}

inline QueryExecResult run_query(ReplayEngine* engine, uint32_t window, uint32_t gt_index,
                                 const std::vector<float>& qpos) {
    return engine->execute_query(window, gt_index, qpos.data());
}

inline bool contains_id(const QueryExecResult& r, uint64_t id) {
    for (uint64_t x : r.ids) {
        if (x == id) return true;
    }
    return false;
}

inline std::string first_mismatch(const std::string& line, const std::vector<std::string>& keys) {
    for (const std::string& key : keys) {
        if (line.find(key) == std::string::npos) return key;
    }
    return std::string();
}

}  // namespace test
}  // namespace qbam
