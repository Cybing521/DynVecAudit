// Q-BAM M1 engine -- authoritative live set implementation.
#include "core/live_store.h"

#include <algorithm>

#include "common/logging.h"

namespace qbam {

void LiveStore::invalidate_cache() const {
    alive_cache_dirty_ = true;
    all_cache_dirty_ = true;
}

void LiveStore::assign(uint64_t id, uint64_t version, uint32_t shard, uint32_t vec_ref) {
    auto it = entries_.find(id);
    if (it == entries_.end()) {
        LiveEntry e;
        e.version = version;
        e.shard = shard;
        e.vec_ref = vec_ref;
        e.alive = true;
        entries_.emplace(id, e);
        ++alive_count_;
    } else {
        if (!it->second.alive) ++alive_count_;
        it->second.version = version;
        it->second.shard = shard;
        it->second.vec_ref = vec_ref;
        it->second.alive = true;
    }
    invalidate_cache();
}

void LiveStore::erase(uint64_t id, uint64_t version) {
    auto it = entries_.find(id);
    QBAM_CHECK(it != entries_.end() && it->second.alive, "erase of non-live id " + std::to_string(id));
    it->second.alive = false;
    it->second.version = version;
    --alive_count_;
    invalidate_cache();
}

const LiveEntry* LiveStore::find(uint64_t id) const {
    auto it = entries_.find(id);
    return it == entries_.end() ? nullptr : &it->second;
}

bool LiveStore::alive(uint64_t id) const {
    auto it = entries_.find(id);
    return it != entries_.end() && it->second.alive;
}

const std::vector<uint64_t>& LiveStore::alive_ids_sorted() const {
    if (alive_cache_dirty_) {
        alive_cache_.clear();
        alive_cache_.reserve(alive_count_);
        for (const auto& kv : entries_) {
            if (kv.second.alive) alive_cache_.push_back(kv.first);
        }
        std::sort(alive_cache_.begin(), alive_cache_.end());
        alive_cache_dirty_ = false;
    }
    return alive_cache_;
}

const std::vector<uint64_t>& LiveStore::all_ids_sorted() const {
    if (all_cache_dirty_) {
        all_cache_.clear();
        all_cache_.reserve(entries_.size());
        for (const auto& kv : entries_) all_cache_.push_back(kv.first);
        std::sort(all_cache_.begin(), all_cache_.end());
        all_cache_dirty_ = false;
    }
    return all_cache_;
}

}  // namespace qbam
