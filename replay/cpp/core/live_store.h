// Q-BAM M1 engine -- authoritative live set (id -> version/shard/vec_ref/alive).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace qbam {

struct LiveEntry {
    uint64_t version = 0;
    uint32_t shard = 0;
    uint32_t vec_ref = 0;
    bool alive = false;
};

// Authoritative global state. Iteration order of the hash map is NEVER used for decisions:
// anything order-sensitive goes through alive_ids_sorted() / all_ids_sorted().
class LiveStore {
public:
    LiveStore() = default;

    // Insert or update: id becomes alive with the given version/shard/vec_ref.
    void assign(uint64_t id, uint64_t version, uint32_t shard, uint32_t vec_ref);
    // Erase: id must be currently alive. Keeps the tombstone entry (version is retained for
    // the state hash and for monotonicity checks; it is not alive).
    void erase(uint64_t id, uint64_t version);

    const LiveEntry* find(uint64_t id) const;
    bool alive(uint64_t id) const;

    size_t size() const { return entries_.size(); }
    size_t alive_count() const { return alive_count_; }

    const std::unordered_map<uint64_t, LiveEntry>& entries() const { return entries_; }

    // Ascending id order; cached until the next mutation.
    const std::vector<uint64_t>& alive_ids_sorted() const;
    const std::vector<uint64_t>& all_ids_sorted() const;

private:
    void invalidate_cache() const;

    std::unordered_map<uint64_t, LiveEntry> entries_;
    size_t alive_count_ = 0;
    mutable bool alive_cache_dirty_ = true;
    mutable bool all_cache_dirty_ = true;
    mutable std::vector<uint64_t> alive_cache_;
    mutable std::vector<uint64_t> all_cache_;
};

}  // namespace qbam
