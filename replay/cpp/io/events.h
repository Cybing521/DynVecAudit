// Q-BAM M1 engine -- binary contract loaders (engine_contract.md §1-§5).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "common/util.h"

namespace qbam {

// ---------------------------------------------------------------- contract structs
enum class EventType : uint8_t { Insert = 0, Update = 1, Erase = 2, Query = 3 };

struct PoolData {
    uint32_t dim = 0;
    uint32_t nvec = 0;
    std::vector<float> data;  // nvec * dim, row major

    const float* vec(uint32_t vec_ref) const { return data.data() + static_cast<size_t>(vec_ref) * dim; }
};

struct InitEntry {
    uint64_t id;
    uint32_t shard;
    uint32_t vec_ref;
};

struct InitData {
    uint32_t dim = 0;
    uint32_t n = 0;
    uint32_t R = 0;
    std::vector<InitEntry> entries;
};

struct Centroids {
    uint32_t R = 0;
    uint32_t dim = 0;
    std::vector<float> data;  // R * dim, row major

    const float* centroid(uint32_t r) const { return data.data() + static_cast<size_t>(r) * dim; }
};

struct TraceEvent {
    uint32_t window = 0;
    EventType type = EventType::Insert;
    uint64_t id = 0;
    uint64_t version = 0;
    uint32_t vec_ref = 0;
};

struct TraceData {
    uint32_t dim = 0;
    uint32_t n_events = 0;
    uint32_t n_windows = 0;
    uint32_t K = 0;
    std::vector<TraceEvent> events;
    // window_begin[w] = index of first event with window >= w (size n_windows + 1)
    std::vector<uint32_t> window_begin;
    std::vector<uint32_t> window_query_count;
    std::vector<uint32_t> window_query_start;  // gt row index of first query in window
    uint32_t total_queries = 0;

    uint32_t window_begin_of(uint32_t w) const { return window_begin[w]; }
    uint32_t window_end_of(uint32_t w) const { return window_begin[w + 1]; }
};

struct GtData {
    uint32_t k = 0;
    uint32_t n_queries = 0;
    uint32_t dim = 0;
    std::vector<uint64_t> ids;     // n_queries * k
    std::vector<float> squared_l2;  // n_queries * k
};

// ---------------------------------------------------------------- loaders
PoolData load_pool(const std::string& path);
InitData load_init(const std::string& path);
Centroids load_centroids(const std::string& path, uint32_t R, uint32_t dim);
TraceData load_trace(const std::string& path);
GtData load_gt(const std::string& path);

// Semantic validation across files (dims, refs, CRUD-before-query, version monotonicity,
// alive-state consistency, gt alignment). Throws std::runtime_error with a precise message.
// `has_gt=false` skips GT-file checks (used by no-GT smoke runs while GT is being computed).
void validate_dataset(const PoolData& pool, const InitData& init, const Centroids& centroids,
                      const TraceData& trace, const GtData& gt, uint32_t expected_dim,
                      uint32_t expected_R, uint32_t expected_k, bool has_gt = true);

// Recomputes window_begin / window_query_count / window_query_start from `events`.
// Requires events already sorted by non-decreasing window.
void build_window_index(TraceData* trace);

}  // namespace qbam
