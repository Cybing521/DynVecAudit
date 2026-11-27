// Q-BAM M1 engine -- immutable HNSW base + exact delta + tombstone counters per shard.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "hnswlib/hnswlib.h"

#include "core/live_store.h"
#include "io/events.h"

namespace qbam {

struct HnswParams {
    size_t M = 16;
    size_t ef_construction = 200;
    size_t ef_search = 64;
    uint64_t seed = 0;
};

struct LiveRef {
    uint64_t id = 0;
    uint64_t version = 0;
    uint32_t vec_ref = 0;
};

// A shard Base snapshot. Immutable once published: only refresh(r) replaces the whole object,
// so counterfactual branches share it by pointer without copying (engine_contract.md §8.1).
//
// `merge:r` (diagnostic Phase-2 action) is the one exception: it deep-copies the Base into a new
// object and mutates that copy, leaving the published snapshot untouched.
class ImmutableBaseHNSW {
public:
    ImmutableBaseHNSW(const ImmutableBaseHNSW&) = delete;
    ImmutableBaseHNSW& operator=(const ImmutableBaseHNSW&) = delete;

    // Builds from `items` (insertion order = deterministic caller order, normally ascending id).
    // Returns nullptr when items is empty.
    static std::shared_ptr<const ImmutableBaseHNSW> build(const PoolData& pool,
                                                          const std::vector<LiveRef>& items,
                                                          const HnswParams& params);
    // Empty mutable Base with the given initial capacity (used as the target of merge:r on a
    // shard without a Base).
    static std::shared_ptr<ImmutableBaseHNSW> create(const PoolData& pool, size_t capacity,
                                                     const HnswParams& params);

    // Deterministic rebuild of this snapshot: same vectors, same order, same seed => identical
    // graph. Used to deep-copy a Base for merge:r branches (cost excluded from maintenance).
    std::shared_ptr<ImmutableBaseHNSW> deep_copy(const PoolData& pool) const;

    // hnswlib resizeIndex wrapper; new_capacity must be >= current capacity.
    void reserve(size_t new_capacity);
    // Appends one vector with a label not already present in this Base.
    void add_point(const PoolData& pool, uint32_t vec_ref, uint64_t id, uint64_t version);

    size_t n() const { return n_; }
    uint32_t dim() const { return dim_; }
    const HnswParams& params() const { return params_; }

    // id -> version of the vector stored in this Base snapshot.
    const std::unordered_map<uint64_t, uint64_t>& versions() const { return versions_; }

    // Sorted ascending by (squared L2, id); at most min(k, n) entries.
    void search(const float* query, size_t k,
                std::vector<std::pair<float, uint64_t>>* out) const;

    // hnswlib markDelete wrapper (delta_mode=hnsw): the label stays in the graph
    // as a traversal waypoint but is filtered out of search results.
    void mark_deleted(uint64_t label);

private:
    ImmutableBaseHNSW() = default;
    static std::shared_ptr<ImmutableBaseHNSW> build_impl(const PoolData& pool,
                                                         const std::vector<LiveRef>& items,
                                                         const HnswParams& params);
    void append_label(uint64_t id, uint64_t version, uint32_t vec_ref);

    std::shared_ptr<hnswlib::SpaceInterface<float>> space_;
    std::unique_ptr<hnswlib::HierarchicalNSW<float>> index_;
    std::unordered_map<uint64_t, uint64_t> versions_;  // id -> version of the vector in this snapshot
    std::unordered_set<uint64_t> deleted_labels_;       // labels marked deleted (replace_deleted support)
    std::vector<LiveRef> items_;  // insertion order, kept for deep_copy/merge
    HnswParams params_;
    size_t n_ = 0;
    uint32_t dim_ = 0;
};

struct DeltaEntry {
    uint64_t id = 0;
    uint64_t version = 0;
    uint32_t vec_ref = 0;
    uint32_t offset = 0;     // float offset into ShardIndex::delta_arena
    bool valid = true;       // false once the id is updated/erased/refreshed/merged
    bool in_graph = false;   // true if this entry resides in delta_graph
};

struct ShardIndex {
    uint32_t r = 0;
    std::shared_ptr<const ImmutableBaseHNSW> base;  // may be null (empty shard)
    std::vector<DeltaEntry> delta;
    std::vector<float> delta_arena;                       // dim floats per Delta vector, insertion order
    std::unordered_map<uint64_t, uint32_t> delta_index;   // id -> index of its live DeltaEntry
    uint32_t dim = 0;
    uint64_t tombstone_n = 0;
    bool base_has_stale = false;

    bool delta_indexed = false;
    HnswParams delta_params;
    size_t delta_initial_capacity = 4096;
    std::shared_ptr<ImmutableBaseHNSW> delta_graph;  // null in scan mode, or before first insert

    bool delta_has_graph() const { return delta_graph != nullptr; }
    void enable_delta_graph(const HnswParams& params, size_t initial_capacity) {
        delta_indexed = true;
        delta_params = params;
        delta_initial_capacity = std::max<size_t>(initial_capacity, 1);
    }
    void ensure_delta_graph(const PoolData& pool) {
        if (!delta_indexed || delta_graph != nullptr) return;
        delta_graph = ImmutableBaseHNSW::create(pool, delta_initial_capacity, delta_params);
    }
    void rebuild_delta_graph(const PoolData& pool) {
        if (!delta_indexed) return;
        delta_graph.reset();
        delta_arena.clear();
        if (delta.empty()) return;
        ensure_delta_graph(pool);
        for (DeltaEntry& de : delta) {
            if (!de.valid) continue;
            delta_graph->add_point(pool, de.vec_ref, de.id, de.version);
            de.in_graph = true;
        }
    }

    // Block-scoped counters (reset after every emitted block record).
    uint64_t q_accesses = 0;
    double base_search_wall_us = 0.0;
    double delta_scan_wall_us = 0.0;

    size_t base_n() const { return base ? base->n() : 0; }
    size_t delta_n() const { return delta_index.size(); }

    void insert_delta(const PoolData& pool, uint64_t id, uint64_t version, uint32_t vec_ref) {
        if (dim == 0) dim = pool.dim;
        auto it = delta_index.find(id);
        if (it != delta_index.end()) {
            if (delta_graph != nullptr && delta_graph->versions().count(id) > 0) {
                delta_graph->mark_deleted(id);
            }
            delta[it->second].valid = false;
        }
        uint32_t offset = 0;
        bool in_graph = false;
        if (delta_indexed) {
            ensure_delta_graph(pool);
            if (delta_graph->versions().count(id) > 0) delta_graph->mark_deleted(id);
            delta_graph->add_point(pool, vec_ref, id, version);
            in_graph = true;
        } else {
            offset = static_cast<uint32_t>(delta_arena.size());
            const float* src = pool.vec(vec_ref);
            delta_arena.insert(delta_arena.end(), src, src + dim);
        }
        const uint32_t idx = static_cast<uint32_t>(delta.size());
        delta.push_back(DeltaEntry{id, version, vec_ref, offset, true, in_graph});
        delta_index[id] = idx;
        if (!delta_indexed) maybe_compact_delta();
    }

    const float* delta_vec(const DeltaEntry& entry) const { return delta_arena.data() + entry.offset; }

    void invalidate_delta(uint64_t id) {
        auto it = delta_index.find(id);
        if (it == delta_index.end()) return;
        if (delta_graph != nullptr && delta_graph->versions().count(id) > 0) {
            delta_graph->mark_deleted(id);
        }
        delta[it->second].valid = false;
        delta_index.erase(it);
    }

    void clear_delta() {
        delta.clear();
        delta_arena.clear();
        delta_index.clear();
        delta_graph.reset();
    }

    void rebuild_delta_index() {
        delta_index.clear();
        delta_index.reserve(delta.size() * 2 + 1);
        for (uint32_t i = 0; i < delta.size(); ++i) {
            if (delta[i].valid) delta_index[delta[i].id] = i;
        }
    }

    void maybe_compact_delta() {
        if (delta.size() <= 4096) return;
        if (delta_index.size() * 2 >= delta.size()) return;
        std::vector<DeltaEntry> compacted;
        std::vector<float> arena;
        compacted.reserve(delta_index.size());
        arena.reserve(delta_index.size() * static_cast<size_t>(dim));
        for (const DeltaEntry& de : delta) {
            if (!de.valid) continue;
            DeltaEntry moved = de;
            if (!de.in_graph) {
                moved.offset = static_cast<uint32_t>(arena.size());
                arena.insert(arena.end(), delta_arena.begin() + de.offset,
                             delta_arena.begin() + de.offset + dim);
            } else {
                moved.offset = 0;
            }
            compacted.push_back(moved);
        }
        delta.swap(compacted);
        delta_arena.swap(arena);
        delta.shrink_to_fit();
        delta_arena.shrink_to_fit();
        rebuild_delta_index();
    }

    void reset_block_counters() {
        q_accesses = 0;
        base_search_wall_us = 0.0;
        delta_scan_wall_us = 0.0;
    }
};

struct RefreshResult {
    int shard = -1;
    double build_cpu_ms = 0.0;   // action work (for refresh: gather+build+swap; for merge: addPoint loop)
    double copy_cpu_ms = 0.0;    // merge:r only -- deep copy of the immutable Base (offline helper)
    double wall_ms = 0.0;
    uint64_t live_vectors_built = 0;
    uint64_t base_before = 0;
    uint64_t delta_before = 0;
};

// Full rebuild of shard r from the current global live set: new HNSW over every live vector
// assigned to r, Base replaced, Delta cleared, tombstones cleared (engine_contract.md §8.2).
// CPU/wall include the live-set gathering, the build and the swap.
RefreshResult refresh_shard(ShardIndex* shard, const PoolData& pool, const LiveStore& live,
                            const HnswParams& params, uint32_t dim, uint64_t build_seed);

}  // namespace qbam
