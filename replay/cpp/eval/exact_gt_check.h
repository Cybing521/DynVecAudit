// Q-BAM M1 engine -- offline exact global top-k (evaluation / tests only).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/live_store.h"
#include "core/router.h"
#include "io/events.h"

namespace qbam {

struct ExactTopK {
    std::vector<uint64_t> ids;     // sorted ascending by (squared L2, id)
    std::vector<float> squared_l2;
};

// Brute force over the entire global live set. Never used inside mode=replay queries.
ExactTopK compute_exact_topk(const PoolData& pool, const LiveStore& live, const float* query,
                             uint32_t k);

struct GtVerifyResult {
    uint32_t checked = 0;
    uint32_t mismatches = 0;
    std::vector<std::string> errors;
};

// Independent dataset-level check of the first `n_queries` gt.bin rows: replays init + trace CRUD
// through a plain LiveStore (no HNSW involved) and compares each stored row against an exact
// brute-force global top-k. Optional C++ validation of Python-generated GT (contract §0/§5).
GtVerifyResult verify_gt_sample(const PoolData& pool, const InitData& init, const Centroids& centroids,
                                const TraceData& trace, const GtData& gt, uint32_t k,
                                uint32_t n_queries);

// Checks one gt.bin row (at the state matching that query) against the brute-force result.
// `query` points at the same pool vector used to build the gt row.
// Returns true when ids and distances agree; fills *message otherwise.
bool verify_gt_row(const PoolData& pool, const LiveStore& live, const GtData& gt,
                   uint32_t gt_index, const float* query, uint32_t k, std::string* message);

}  // namespace qbam
