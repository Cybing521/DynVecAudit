// Q-BAM M1 engine -- fixed centroid routing implementation.
#include "core/router.h"

#include <algorithm>
#include <utility>

#include "common/logging.h"

namespace qbam {

void Router::init(uint32_t R, uint32_t dim, const std::vector<float>& centroids,
                  uint32_t nprobe) {
    QBAM_CHECK(R > 0, "router R must be > 0");
    QBAM_CHECK(dim > 0, "router dim must be > 0");
    QBAM_CHECK(nprobe >= 1 && nprobe <= R, "router nprobe must be in [1, R]");
    QBAM_CHECK(centroids.size() == static_cast<size_t>(R) * dim,
               "router centroid buffer shape mismatch");
    R_ = R;
    dim_ = dim;
    nprobe_ = nprobe;
    centroids_ = centroids;
}

void Router::select(const float* query, std::vector<uint32_t>* out) const {
    QBAM_CHECK(out != nullptr, "router select: null output");
    std::vector<std::pair<float, uint32_t>>& cand = scratch_;
    cand.clear();
    cand.reserve(R_);
    for (uint32_t r = 0; r < R_; ++r) {
        cand.emplace_back(l2_squared(query, centroids_.data() + static_cast<size_t>(r) * dim_, dim_),
                          r);
    }
    const uint32_t take = std::min(nprobe_, R_);
    std::partial_sort(cand.begin(), cand.begin() + take, cand.end(),
                      [](const std::pair<float, uint32_t>& a,
                         const std::pair<float, uint32_t>& b) {
                          if (a.first != b.first) return a.first < b.first;
                          return a.second < b.second;
                      });
    out->clear();
    out->reserve(take);
    for (uint32_t i = 0; i < take; ++i) out->push_back(cand[i].second);
}

uint32_t Router::assign_shard(const float* vec) const {
    uint32_t best_r = 0;
    float best_d = l2_squared(vec, centroids_.data(), dim_);
    for (uint32_t r = 1; r < R_; ++r) {
        const float d = l2_squared(vec, centroids_.data() + static_cast<size_t>(r) * dim_, dim_);
        if (d < best_d) {
            best_d = d;
            best_r = r;
        }
    }
    return best_r;
}

}  // namespace qbam
