// Q-BAM M1 engine -- quality attribution: recall, route_upper, index_loss, source fractions.
#pragma once

#include <cstdint>
#include <vector>

#include "core/live_store.h"
#include "io/events.h"

namespace qbam {

struct QueryQuality {
    double recall = 0.0;                    // |G(q) ∩ Returned(q)| / k          (online metric)
    double route_upper = 0.0;               // |G(q) ∩ VisitedShardVectors| / k  (evaluation)
    double index_loss = 0.0;                // route_upper - recall (clamped at 0)
    double returned_delta_fraction = 0.0;   // returned ids sourced from Delta   (online metric)
};

// `gt_index` is the global query index in trace/gt order. `visited_mask` has one entry per
// shard (non-zero = shard visited for this query). `gt` is evaluation-only data: never used
// as a policy feature in mode=replay (engine_contract.md §8.5).
QueryQuality attribute_query(uint32_t k, const GtData& gt, uint32_t gt_index,
                             const std::vector<uint64_t>& returned_ids,
                             const std::vector<bool>& from_delta,
                             const std::vector<char>& visited_mask, const LiveStore& live);

}  // namespace qbam
