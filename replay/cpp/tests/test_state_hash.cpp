// Q-BAM M1 engine -- test 6/7: state hash semantics (contract §8.9). The hash must be
// independent of unordered_map iteration order, must change with any live-set/index change,
// must be deterministic across identical runs, and CF clones must share the decision-point hash.
#include <algorithm>
#include <vector>

#include "tests/test_util.h"

using namespace qbam;
using namespace qbam::test;

namespace {
MiniDataset make_dataset() {
    MiniBuilder b(4, 2);
    for (uint32_t i = 0; i < 6; ++i) {
        b.add_init(0, {0.5f * static_cast<float>(i), 0.0f, 0.0f, 0.0f});
        b.add_init(1, {10.0f + 0.5f * static_cast<float>(i), 0.0f, 0.0f, 0.0f});
    }
    for (uint32_t w = 0; w < 2; ++w) {
        const uint32_t ref = b.add_vec({1.0f + 0.1f * static_cast<float>(w), 0.0f, 0.0f, 0.0f});
        b.event_update(w, 1, 2 + w, ref);
        b.event_query(w, {0.5f, 0.0f, 0.0f, 0.0f});
    }
    return b.finalize(2, 2, 3);
}
}  // namespace

int main() {
    const uint32_t dim = 4;
    const uint32_t k = 3;
    EngineConfig cfg = make_config(2, dim, k);

    // ---- 1) hash is independent of hash-map / init-file entry order ----
    MiniDataset ds = make_dataset();
    MiniDataset ds_reordered = ds;
    std::reverse(ds_reordered.init.entries.begin(), ds_reordered.init.entries.end());
    {
        ReplayEngine e1 = make_engine(ds, cfg);
        ReplayEngine e2 = make_engine(ds_reordered, cfg);
        TCHECK_MSG(e1.state_hash() == e2.state_hash(),
                   "state hash must not depend on init entry insertion order");
        apply_window_crud(&e1, ds.trace, 0);
        apply_window_crud(&e2, ds_reordered.trace, 0);
        TCHECK_NEAR(e1.live().alive_count(), e2.live().alive_count(), 0.0);
        TCHECK(e1.state_hash() == e2.state_hash());
        TCHECK(e1.state_hash().size() == 16);
        for (char c : e1.state_hash()) {
            TCHECK((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'));
        }
    }

    // ---- 2) identical replay from scratch => identical hash sequence ----
    {
        ReplayEngine a = make_engine(ds, cfg);
        ReplayEngine b = make_engine(ds, cfg);
        TCHECK(a.state_hash() == b.state_hash());
        for (uint32_t w = 0; w < ds.trace.n_windows; ++w) {
            apply_window_crud(&a, ds.trace, w);
            apply_window_crud(&b, ds.trace, w);
            TCHECK(a.state_hash() == b.state_hash());
        }
    }

    // ---- 3) clone_for_branch shares the decision-point state without aliasing mutations ----
    {
        ReplayEngine parent = make_engine(ds, cfg);
        apply_window_crud(&parent, ds.trace, 0);
        const std::string parent_hash = parent.state_hash();
        ReplayEngine clone = parent.clone_for_branch();
        TCHECK(clone.state_hash() == parent_hash);
        double maint = 0.0;
        clone.apply_action(Action::refresh(0), &maint);
        TCHECK_MSG(parent.state_hash() == parent_hash, "branch action must not mutate the parent");
        TCHECK_MSG(clone.state_hash() != parent_hash,
                   "refresh with a non-empty Delta must change the state");
        TCHECK(clone.shards()[0].delta_n() == 0);
        TCHECK(clone.shards()[0].base_n() >= parent.shards()[0].base_n());
        TCHECK(parent.shards()[0].delta_n() > 0);
        // applying an unrelated insert to the clone must not leak into the parent
        const uint64_t before = parent.live().alive_count();
        const uint32_t ref = ds.pool.nvec;  // not used; build a copy event below instead
        (void)ref;
        TraceEvent ev{0, EventType::Insert, 7777, 1, 0};
        double cpu = 0.0;
        clone.apply_crud(ev, &cpu);
        TCHECK(clone.live().alive_count() == before + 1);
        TCHECK(parent.live().alive_count() == before);
    }

    // ---- 4) refresh is state-changing once and idempotent afterwards ----
    {
        ReplayEngine e = make_engine(ds, cfg);
        apply_window_crud(&e, ds.trace, 0);
        const std::string h0 = e.state_hash();
        double maint = 0.0;
        e.apply_action(Action::refresh(0), &maint);
        const std::string h1 = e.state_hash();
        TCHECK(h1 != h0);
        e.apply_action(Action::refresh(0), &maint);
        TCHECK(e.state_hash() == h1);
        // any CRUD change (live set and/or Delta counters) changes the hash
        double cpu = 0.0;
        e.apply_crud(TraceEvent{0, EventType::Insert, 9999, 1, 0}, &cpu);
        const std::string h_ins = e.state_hash();
        TCHECK(h_ins != h1);
        e.apply_crud(TraceEvent{0, EventType::Erase, 9999, 2, 0}, &cpu);
        const std::string h2 = e.state_hash();
        TCHECK(h2 != h_ins);
        TCHECK(h2 != h1);
        TCHECK(h2 != h0);
    }

    std::printf("test_state_hash OK (16 hex chars, order-independent, clone-isolated)\n");
    return 0;
}
