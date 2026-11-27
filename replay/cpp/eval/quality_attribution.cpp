// Q-BAM M1 engine -- quality attribution implementation.
#include "eval/quality_attribution.h"

#include <algorithm>

namespace qbam {

QueryQuality attribute_query(uint32_t k, const GtData& gt, uint32_t gt_index,
                             const std::vector<uint64_t>& returned_ids,
                             const std::vector<bool>& from_delta,
                             const std::vector<char>& visited_mask, const LiveStore& live) {
    QueryQuality q;
    if (k == 0) return q;
    const size_t gt_k = std::min<size_t>(gt.k, k);
    const uint64_t* gt_row = gt.ids.data() + static_cast<size_t>(gt_index) * gt.k + 0;

    size_t hits = 0;
    size_t routed_hits = 0;
    for (size_t i = 0; i < gt_k; ++i) {
        const uint64_t id = gt_row[i];
        for (uint64_t rid : returned_ids) {
            if (rid == id) {
                ++hits;
                break;
            }
        }
        const LiveEntry* e = live.find(id);
        if (e != nullptr && e->alive && e->shard < visited_mask.size() && visited_mask[e->shard]) {
            ++routed_hits;
        }
    }
    const double denom = static_cast<double>(k);
    q.recall = static_cast<double>(hits) / denom;
    q.route_upper = static_cast<double>(routed_hits) / denom;
    q.index_loss = q.route_upper - q.recall;
    if (q.index_loss < 0.0) q.index_loss = 0.0;  // float guards; R <= R_route by construction

    if (!returned_ids.empty()) {
        size_t from_delta_count = 0;
        for (size_t i = 0; i < returned_ids.size(); ++i) {
            if (i < from_delta.size() && from_delta[i]) ++from_delta_count;
        }
        q.returned_delta_fraction =
            static_cast<double>(from_delta_count) / static_cast<double>(returned_ids.size());
    }
    return q;
}

}  // namespace qbam
