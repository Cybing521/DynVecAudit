// Q-BAM M1 engine -- immutable HNSW base + exact delta implementation.
#include "core/shard_index.h"

#include <algorithm>
#include <string>

#include "common/logging.h"
#include "eval/cpu_meter.h"

namespace qbam {

namespace {
void init_space_and_index(const PoolData& pool, const HnswParams& params, size_t capacity,
                          std::shared_ptr<hnswlib::SpaceInterface<float>>* space,
                          std::unique_ptr<hnswlib::HierarchicalNSW<float>>* index) {
    QBAM_CHECK(params.M >= 2, "HNSW M must be >= 2");
    QBAM_CHECK(params.ef_construction >= params.M,
               "HNSW efConstruction must be >= M (hnswlib clamps it silently otherwise)");
    *space = std::make_shared<hnswlib::L2Space>(static_cast<size_t>(pool.dim));
    const size_t cap = std::max<size_t>(capacity, 1);
    *index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
        space->get(), cap, params.M, params.ef_construction, params.seed,
        /*allow_replace_deleted=*/true);
    (*index)->setEf(params.ef_search);
}
}  // namespace

std::shared_ptr<ImmutableBaseHNSW> ImmutableBaseHNSW::build_impl(const PoolData& pool,
                                                                  const std::vector<LiveRef>& items,
                                                                  const HnswParams& params) {
    std::shared_ptr<ImmutableBaseHNSW> out(new ImmutableBaseHNSW());
    out->dim_ = pool.dim;
    out->params_ = params;
    out->items_ = items;
    out->n_ = items.size();
    if (items.empty()) {
        init_space_and_index(pool, params, 1, &out->space_, &out->index_);
        out->versions_.clear();
        return out;
    }
    init_space_and_index(pool, params, items.size(), &out->space_, &out->index_);
    out->versions_.reserve(items.size() * 2 + 1);
    for (const LiveRef& item : items) {
        out->index_->addPoint(pool.vec(item.vec_ref), static_cast<hnswlib::labeltype>(item.id));
        out->versions_.emplace(item.id, item.version);
    }
    QBAM_CHECK(out->index_->getCurrentElementCount() == items.size(),
               "HNSW base build element count mismatch");
    return out;
}

std::shared_ptr<const ImmutableBaseHNSW> ImmutableBaseHNSW::build(const PoolData& pool,
                                                                  const std::vector<LiveRef>& items,
                                                                  const HnswParams& params) {
    if (items.empty()) return nullptr;
    return build_impl(pool, items, params);
}

std::shared_ptr<ImmutableBaseHNSW> ImmutableBaseHNSW::create(const PoolData& pool, size_t capacity,
                                                             const HnswParams& params) {
    std::shared_ptr<ImmutableBaseHNSW> out(new ImmutableBaseHNSW());
    out->dim_ = pool.dim;
    out->params_ = params;
    out->n_ = 0;
    init_space_and_index(pool, params, capacity, &out->space_, &out->index_);
    out->versions_.reserve(capacity * 2 + 1);
    return out;
}

std::shared_ptr<ImmutableBaseHNSW> ImmutableBaseHNSW::deep_copy(const PoolData& pool) const {
    return build_impl(pool, items_, params_);
}

void ImmutableBaseHNSW::reserve(size_t new_capacity) {
    QBAM_CHECK(index_ != nullptr, "reserve on a null index");
    const size_t current = index_->getMaxElements();
    if (new_capacity <= current) return;
    index_->resizeIndex(new_capacity);
}

void ImmutableBaseHNSW::append_label(uint64_t id, uint64_t version, uint32_t vec_ref) {
    items_.push_back(LiveRef{id, version, vec_ref});
    versions_.emplace(id, version);
    ++n_;
}

void ImmutableBaseHNSW::add_point(const PoolData& pool, uint32_t vec_ref, uint64_t id,
                                  uint64_t version) {
    if (index_ == nullptr) {
        // Lazily materialize an empty Base (used by merge:r when the shard had no Base).
        dim_ = pool.dim;
        init_space_and_index(pool, params_, 1, &space_, &index_);
        versions_.reserve(16);
    }
    auto v_it = versions_.find(id);
    const bool replacing = (v_it != versions_.end() && deleted_labels_.count(id) > 0);
    QBAM_CHECK(v_it == versions_.end() || replacing,
               "add_point: duplicate HNSW label " + std::to_string(id) +
                   " (label exists and was not marked deleted)");
    const size_t count = index_->getCurrentElementCount();
    if (count >= index_->getMaxElements()) {
        reserve(std::max<size_t>(count * 2, count + 16));
    }
    if (replacing) {
        // CRUD update path: hnswlib replace_deleted swaps the deleted slot with this insertion.
        index_->addPoint(pool.vec(vec_ref), static_cast<hnswlib::labeltype>(id),
                         /*replace_deleted=*/true);
        deleted_labels_.erase(id);
        v_it->second = version;
        return;
    }
    index_->addPoint(pool.vec(vec_ref), static_cast<hnswlib::labeltype>(id));
    append_label(id, version, vec_ref);
}

void ImmutableBaseHNSW::search(const float* query, size_t k,
                               std::vector<std::pair<float, uint64_t>>* out) const {
    out->clear();
    if (n_ == 0 || k == 0 || index_ == nullptr) return;
    const size_t kk = std::min(k, n_);
    auto pq = index_->searchKnn(query, kk);
    out->reserve(pq.size());
    while (!pq.empty()) {
        out->emplace_back(pq.top().first, static_cast<uint64_t>(pq.top().second));
        pq.pop();
    }
    // Deliberately unordered: the query path feeds these into a total-order bounded top-k heap,
    // so sorting here (4x40 elements per query) would only add cost without changing results.
}

void ImmutableBaseHNSW::mark_deleted(uint64_t label) {
    QBAM_CHECK(index_ != nullptr, "mark_deleted on a null index");
    QBAM_CHECK(versions_.find(label) != versions_.end(),
               "mark_deleted: label not present in this graph");
    if (deleted_labels_.count(label) > 0) return;  // idempotent: already marked deleted
    // With allow_replace_deleted=true, hnswlib may have consumed the slot of a previously deleted
    // label (its entry is removed from the internal lookup), so markDelete can fail for a label we
    // still track in versions_. Treat that as "already gone" rather than crashing the run.
    try {
        index_->markDelete(static_cast<hnswlib::labeltype>(label));
    } catch (const std::exception&) {
        deleted_labels_.insert(label);
        return;
    }
    deleted_labels_.insert(label);
}

RefreshResult refresh_shard(ShardIndex* shard, const PoolData& pool, const LiveStore& live,
                            const HnswParams& params, uint32_t dim, uint64_t build_seed) {
    QBAM_CHECK(shard != nullptr, "refresh_shard: null shard");
    RefreshResult rr;
    rr.shard = static_cast<int>(shard->r);
    rr.base_before = shard->base_n();
    rr.delta_before = shard->delta_n();

    CpuWallTimer timer;
    timer.start();
    std::vector<LiveRef> items;
    const std::vector<uint64_t>& alive_ids = live.alive_ids_sorted();
    items.reserve(alive_ids.size() / 8 + 16);
    for (uint64_t id : alive_ids) {
        const LiveEntry* e = live.find(id);
        if (e != nullptr && e->shard == shard->r && e->alive) {
            items.push_back(LiveRef{id, e->version, e->vec_ref});
        }
    }
    rr.live_vectors_built = items.size();

    HnswParams params_with_seed = params;
    params_with_seed.seed = build_seed;
    std::shared_ptr<const ImmutableBaseHNSW> new_base =
        ImmutableBaseHNSW::build(pool, items, params_with_seed);

    shard->base = std::move(new_base);
    shard->clear_delta();
    shard->delta_arena.shrink_to_fit();  // release the per-shard Delta arena (~100 MB at 100K/u05)
    shard->tombstone_n = 0;
    shard->base_has_stale = false;  // rebuilt from the live set: every label is current
    rr.build_cpu_ms = timer.cpu_ms();
    rr.wall_ms = timer.wall_ms();
    (void)dim;
    return rr;
}

}  // namespace qbam
