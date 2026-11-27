// Q-BAM M1 engine -- test 2/7: no-op CRUD path, delete-then-refresh, refresh idempotence,
// cross-shard update filtering, and live-set preservation across refresh.
#include <algorithm>
#include <vector>

#include "tests/test_util.h"

using namespace qbam;
using namespace qbam::test;

namespace {
bool same_id_set(const std::vector<uint64_t>& a, std::vector<uint64_t> b) {
    std::vector<uint64_t> x = a;
    std::sort(x.begin(), x.end());
    std::sort(b.begin(), b.end());
    return x == b;
}
}  // namespace

int main() {
    const uint32_t dim = 4;
    MiniBuilder b(dim, 2);
    const uint64_t id1 = b.add_init(0, {0.0f, 0.0f, 0.0f, 0.0f});
    const uint64_t id2 = b.add_init(0, {1.0f, 0.0f, 0.0f, 0.0f});
    const uint64_t id3 = b.add_init(0, {2.0f, 0.0f, 0.0f, 0.0f});
    const uint64_t id4 = b.add_init(1, {10.0f, 0.0f, 0.0f, 0.0f});
    const uint64_t id5 = b.add_init(1, {11.0f, 0.0f, 0.0f, 0.0f});
    (void)id4;
    (void)id5;

    b.event_query(0, {0.0f, 0.0f, 0.0f, 0.0f});                 // gt row 0
    b.event_erase(1, id1, 2);                                    // tombstone in shard 0
    const uint32_t v_moved = b.add_vec({10.0f, 0.0f, 0.0f, 0.0f});
    b.event_update(1, id2, 2, v_moved);                          // shard 0 -> shard 1
    const uint32_t v_new = b.add_vec({0.5f, 0.0f, 0.0f, 0.0f});
    b.event_insert(1, 1006, 1, v_new);                           // new vector in shard 0 delta
    b.event_query(1, {0.0f, 0.0f, 0.0f, 0.0f});                 // gt row 1
    b.event_query(1, {10.0f, 0.0f, 0.0f, 0.0f});                // gt row 2
    b.event_query(2, {0.0f, 0.0f, 0.0f, 0.0f});                 // gt row 3
    b.event_query(3, {10.0f, 0.0f, 0.0f, 0.0f});                // gt row 4

    const uint32_t k = 3;
    MiniDataset ds = b.finalize(4, 1, k);
    EngineConfig cfg = make_config(2, dim, k);
    ReplayEngine engine = make_engine(ds, cfg);

    // window 0: initial state
    apply_window_crud(&engine, ds.trace, 0);
    QueryExecResult q0 = run_query(&engine, 0, 0, {0.0f, 0.0f, 0.0f, 0.0f});
    TCHECK(same_id_set(q0.ids, {id1, id2, id3}));
    TCHECK(engine.live().alive_count() == 5);
    TCHECK(engine.shards()[0].tombstone_n == 0);
    TCHECK(engine.shards()[0].delta_n() == 0);
    const std::string hash_initial = engine.state_hash();
    TCHECK(hash_initial.size() == 16);

    // window 1: erase + cross-shard update + insert
    apply_window_crud(&engine, ds.trace, 1);
    TCHECK(engine.live().alive_count() == 5);  // -1 erase +1 insert +1 update = 5
    TCHECK(engine.shards()[0].base_n() == 3);
    TCHECK(engine.shards()[0].delta_n() == 1);
    TCHECK(engine.shards()[0].tombstone_n == 1);
    TCHECK(engine.shards()[1].base_n() == 2);
    TCHECK(engine.shards()[1].delta_n() == 1);

    QueryExecResult q1 = run_query(&engine, 1, 1, {0.0f, 0.0f, 0.0f, 0.0f});
    TCHECK(same_id_set(q1.ids, {1006, id3, id2}));
    TCHECK_MSG(!contains_id(q1, id1), "erased id leaked into results");
    for (size_t i = 0; i < q1.ids.size(); ++i) {
        if (q1.ids[i] == id2) TCHECK_MSG(q1.versions[i] == 2, "stale cross-shard version returned");
    }

    QueryExecResult q1b = run_query(&engine, 1, 2, {10.0f, 0.0f, 0.0f, 0.0f});
    TCHECK(contains_id(q1b, id2));
    for (size_t i = 0; i < q1b.ids.size(); ++i) {
        if (q1b.ids[i] == id2) {
            TCHECK(q1b.versions[i] == 2);
            TCHECK_MSG(q1b.from_delta[i], "cross-shard update should be served from Delta");
        }
    }

    // refresh shard 0: Base rebuilt from live set, Delta cleared, tombstones cleared
    const std::vector<uint64_t> live_before = engine.live().alive_ids_sorted();
    double maint = 0.0;
    RefreshResult rr = engine.apply_action(Action::refresh(0), &maint);
    TCHECK(rr.shard == 0);
    TCHECK(maint >= 0.0);
    TCHECK(rr.live_vectors_built == 2);  // id3 + 1006
    TCHECK(engine.shards()[0].base_n() == 2);
    TCHECK(engine.shards()[0].delta_n() == 0);
    TCHECK(engine.shards()[0].tombstone_n == 0);
    TCHECK(engine.live().alive_ids_sorted() == live_before);
    const std::string hash_after_refresh = engine.state_hash();
    TCHECK(hash_after_refresh != hash_initial);

    // refresh again with no intervening change: deterministic rebuild => identical state hash
    engine.apply_action(Action::refresh(0), &maint);
    TCHECK(engine.state_hash() == hash_after_refresh);

    // live-set behaviour unchanged by refresh; still no resurrected id
    QueryExecResult q2 = run_query(&engine, 2, 3, {0.0f, 0.0f, 0.0f, 0.0f});
    TCHECK(same_id_set(q2.ids, q1.ids));
    TCHECK_MSG(!contains_id(q2, id1), "refresh resurrected an erased id");

    // refresh shard 1: id2 (updated version) must move from Delta to Base, never v1
    const std::string hash_before_shard1 = engine.state_hash();
    engine.apply_action(Action::refresh(1), &maint);
    TCHECK(engine.shards()[1].base_n() == 3);  // id2 v2 + id4 + id5
    TCHECK(engine.shards()[1].delta_n() == 0);
    TCHECK(engine.state_hash() != hash_before_shard1);
    QueryExecResult q3 = run_query(&engine, 3, 4, {10.0f, 0.0f, 0.0f, 0.0f});
    TCHECK(contains_id(q3, id2));
    for (size_t i = 0; i < q3.ids.size(); ++i) {
        if (q3.ids[i] == id2) {
            TCHECK(q3.versions[i] == 2);
            TCHECK_MSG(!q3.from_delta[i], "after refresh, updated vector must be served from Base");
        }
    }
    // id1 is not alive anywhere: its old Base label must stay filtered
    TCHECK(!engine.live().alive(id1));
    QueryExecResult q4 = run_query(&engine, 3, 4, {0.0f, 0.0f, 0.0f, 0.0f});
    TCHECK_MSG(!contains_id(q4, id1), "erased id resurrected after shard refresh");

    // empty-shard refresh leaves a null Base; merge:r must recreate it from Delta
    {
        MiniBuilder eb(dim, 2);
        const uint64_t a = eb.add_init(0, {0.0f, 0.0f, 0.0f, 0.0f});
        const uint64_t b2 = eb.add_init(0, {1.0f, 0.0f, 0.0f, 0.0f});
        eb.add_init(1, {10.0f, 0.0f, 0.0f, 0.0f});
        eb.add_init(1, {11.0f, 0.0f, 0.0f, 0.0f});
        eb.event_erase(0, a, 2);
        eb.event_erase(0, b2, 3);
        eb.event_query(0, {0.0f, 0.0f, 0.0f, 0.0f});
        MiniDataset eds = eb.finalize(1, 1, 2);
        ReplayEngine e = make_engine(eds, make_config(2, dim, 2));
        apply_window_crud(&e, eds.trace, 0);
        double m = 0.0;
        e.apply_action(Action::refresh(0), &m);
        TCHECK_MSG(e.shards()[0].base_n() == 0, "empty shard must have no Base after refresh");
        TCHECK(e.shards()[0].delta_n() == 0);
        // insert into the now-empty shard: goes to Delta only
        const uint32_t ref = eds.init.entries[0].vec_ref;  // reuse a pool vector
        e.apply_crud(TraceEvent{0, EventType::Insert, 50, 1, ref}, &m);
        TCHECK(e.shards()[0].delta_n() == 1);
        QueryExecResult q = run_query(&e, 0, 0, {0.0f, 0.0f, 0.0f, 0.0f});
        TCHECK(contains_id(q, 50));
        for (size_t i = 0; i < q.ids.size(); ++i) {
            if (q.ids[i] == 50) {
                TCHECK(q.from_delta[i]);
                TCHECK_NEAR(q.dists[i], 0.0, 1e-12);
            }
        }
        // merge:r on a shard without Base must create one from the live Delta
        e.apply_action(Action::merge(0), &m);
        TCHECK(e.shards()[0].base_n() == 1);
        TCHECK(e.shards()[0].delta_n() == 0);
        QueryExecResult q_after = run_query(&e, 0, 0, {0.0f, 0.0f, 0.0f, 0.0f});
        TCHECK(contains_id(q_after, 50));
        for (size_t i = 0; i < q_after.ids.size(); ++i) {
            if (q_after.ids[i] == 50) TCHECK_MSG(!q_after.from_delta[i], "merged vector must be in Base");
        }
    }

    std::printf("test_refresh OK (maint=%.3f ms/shard0, live=%zu)\n", maint,
                engine.live().alive_count());
    return 0;
}
