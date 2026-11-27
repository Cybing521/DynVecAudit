// Q-BAM M1 engine -- offline exact global top-k implementation.
#include "eval/exact_gt_check.h"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "core/router.h"

namespace qbam {

ExactTopK compute_exact_topk(const PoolData& pool, const LiveStore& live, const float* query,
                             uint32_t k) {
    std::vector<std::pair<float, uint64_t>> all;
    const std::vector<uint64_t>& ids = live.alive_ids_sorted();
    all.reserve(ids.size());
    for (uint64_t id : ids) {
        const LiveEntry* e = live.find(id);
        if (e == nullptr || !e->alive) continue;
        all.emplace_back(l2_squared(query, pool.vec(e->vec_ref), pool.dim), id);
    }
    const size_t take = std::min<size_t>(k, all.size());
    std::partial_sort(all.begin(), all.begin() + take, all.end(),
                      [](const std::pair<float, uint64_t>& a,
                         const std::pair<float, uint64_t>& b) {
                          if (a.first != b.first) return a.first < b.first;
                          return a.second < b.second;
                      });
    ExactTopK out;
    out.ids.reserve(take);
    out.squared_l2.reserve(take);
    for (size_t i = 0; i < take; ++i) {
        out.ids.push_back(all[i].second);
        out.squared_l2.push_back(all[i].first);
    }
    return out;
}

GtVerifyResult verify_gt_sample(const PoolData& pool, const InitData& init,
                                const Centroids& centroids, const TraceData& trace,
                                const GtData& gt, uint32_t k, uint32_t n_queries) {
    GtVerifyResult out;
    if (gt.k < k) {
        out.errors.push_back("gt.k < config k");
        out.mismatches = 1;
        return out;
    }
    Router router;
    router.init(init.R, init.dim, centroids.data, init.R);
    LiveStore live;
    for (const InitEntry& e : init.entries) live.assign(e.id, 1, e.shard, e.vec_ref);
    const uint32_t limit = std::min(n_queries, gt.n_queries);
    for (const TraceEvent& ev : trace.events) {
        if (ev.type == EventType::Insert || ev.type == EventType::Update) {
            const float* vec = pool.vec(ev.vec_ref);
            live.assign(ev.id, ev.version, router.assign_shard(vec), ev.vec_ref);
        } else if (ev.type == EventType::Erase) {
            live.erase(ev.id, ev.version);
        } else {
            if (out.checked >= limit) break;
            const float* q = pool.vec(ev.vec_ref);
            std::string msg;
            if (!verify_gt_row(pool, live, gt, out.checked, q, k, &msg)) {
                ++out.mismatches;
                if (out.errors.size() < 10) {
                    out.errors.push_back("query " + std::to_string(out.checked) + ": " + msg);
                }
            }
            ++out.checked;
        }
    }
    return out;
}

bool verify_gt_row(const PoolData& pool, const LiveStore& live, const GtData& gt,
                   uint32_t gt_index, const float* query, uint32_t k, std::string* message) {
    if (gt_index >= gt.n_queries) {
        if (message != nullptr) *message = "gt_index out of range";
        return false;
    }
    if (k > gt.k) {
        if (message != nullptr) *message = "k > gt.k";
        return false;
    }
    ExactTopK ref = compute_exact_topk(pool, live, query, k);
    const uint64_t* gt_ids = gt.ids.data() + static_cast<size_t>(gt_index) * gt.k;
    const float* gt_d = gt.squared_l2.data() + static_cast<size_t>(gt_index) * gt.k;
    if (ref.ids.size() != k) {
        if (message != nullptr) *message = "brute force returned fewer than k ids";
        return false;
    }
    for (uint32_t i = 0; i < k; ++i) {
        if (ref.ids[i] != gt_ids[i]) {
            if (message != nullptr) {
                *message = "id mismatch at rank " + std::to_string(i) + ": gt=" +
                           std::to_string(gt_ids[i]) + " ref=" + std::to_string(ref.ids[i]);
            }
            return false;
        }
        const double tol = 1e-3 * std::max(1.0, static_cast<double>(gt_d[i]));
        if (std::fabs(static_cast<double>(gt_d[i]) - static_cast<double>(ref.squared_l2[i])) > tol) {
            if (message != nullptr) {
                char buf[160];
                std::snprintf(buf, sizeof(buf),
                              "distance mismatch at rank %u: gt=%.9f ref=%.9f", i,
                              static_cast<double>(gt_d[i]),
                              static_cast<double>(ref.squared_l2[i]));
                *message = buf;
            }
            return false;
        }
    }
    return true;
}

}  // namespace qbam
