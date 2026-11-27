// Q-BAM M1 engine -- fixed centroid routing (all-shard and fixed nprobe).
#pragma once

#include <cstdint>
#include <vector>

namespace qbam {

// Router is frozen for a run: same centroids, same nprobe for every policy
// (engine_contract.md §3, §8.6). Distances are squared L2; ties break on smaller shard id.
class Router {
public:
    Router() = default;

    void init(uint32_t R, uint32_t dim, const std::vector<float>& centroids, uint32_t nprobe);

    // Fills `out` with the selected shard ids, nearest centroid first (ties: smaller r first).
    void select(const float* query, std::vector<uint32_t>* out) const;

    // Nearest-centroid assignment used for insert/update events (contract §4 has no shard
    // field; plan v1.2 §5.1 fixes this rule). Ties: smaller shard id.
    uint32_t assign_shard(const float* vec) const;

    uint32_t R() const { return R_; }
    uint32_t dim() const { return dim_; }
    uint32_t nprobe() const { return nprobe_; }

private:
    uint32_t R_ = 0;
    uint32_t dim_ = 0;
    uint32_t nprobe_ = 0;
    std::vector<float> centroids_;  // R * dim
    mutable std::vector<std::pair<float, uint32_t>> scratch_;  // per-call reuse (single thread)
};

// Squared L2 distance between two dim-dimensional float vectors.
inline float l2_squared(const float* a, const float* b, uint32_t dim) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        const float d = a[i] - b[i];
        sum += d * d;
    }
    return sum;
}

}  // namespace qbam
