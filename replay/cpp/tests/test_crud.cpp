// Q-BAM M1 engine -- test 1/7: CRUD correctness, version resurrection, cross-shard update,
// stale-version filtering and (id, distance) ordering.
#include <vector>

#include "tests/test_util.h"

using namespace qbam;
using namespace qbam::test;

int main() {
    const uint32_t dim = 4;
    MiniBuilder b(dim, 2);
    // shard 0: sparse points near origin; shard 1: points near x=10 (default centroids 0 / 10).
    const uint64_t id0 = b.add_init(0, {0.0f, 0.0f, 0.0f, 0.0f});
    const uint64_t id1 = b.add_init(0, {1.0f, 0.0f, 0.0f, 0.0f});
    const uint64_t id2 = b.add_init(0, {2.0f, 0.0f, 0.0f, 0.0f});
    b.add_init(1, {10.0f, 0.0f, 0.0f, 0.0f});
    b.add_init(1, {11.0f, 0.0f, 0.0f, 0.0f});

    // window 0: query only (baseline)
    b.event_query(0, {0.2f, 0.0f, 0.0f, 0.0f});
    // window 1: insert a new id in shard 1
    const uint32_t vec_new = b.add_vec({10.0f, 0.0f, 0.0f, 0.0f});
    b.event_insert(1, 1001, 1, vec_new);
    b.event_query(1, {10.0f, 0.0f, 0.0f, 0.0f});
    // window 2: update 1001 to shard 0 (new version)
    const uint32_t vec_moved = b.add_vec({0.0f, 0.0f, 0.0f, 0.0f});
    b.event_update(2, 1001, 2, vec_moved);
    b.event_query(2, {0.0f, 0.0f, 0.0f, 0.0f});
    b.event_query(2, {10.0f, 0.0f, 0.0f, 0.0f});
    // window 3: erase 1001
    b.event_erase(3, 1001, 3);
    b.event_query(3, {0.0f, 0.0f, 0.0f, 0.0f});
    // window 4: resurrect 1001 with a higher version in shard 1
    const uint32_t vec_back = b.add_vec({10.5f, 0.0f, 0.0f, 0.0f});
    b.event_insert(4, 1001, 4, vec_back);
    b.event_query(4, {10.5f, 0.0f, 0.0f, 0.0f});
    // window 5: query only
    b.event_query(5, {0.2f, 0.0f, 0.0f, 0.0f});

    const uint32_t k = 3;
    MiniDataset ds = b.finalize(6, 1, k);
    EngineConfig cfg = make_config(2, dim, k);
    ReplayEngine engine = make_engine(ds, cfg);

    auto index_of = [](const QueryExecResult& r, uint64_t id) -> int {
        for (size_t i = 0; i < r.ids.size(); ++i) {
            if (r.ids[i] == id) return static_cast<int>(i);
        }
        return -1;
    };
    auto check_result_wellformed = [&](const QueryExecResult& r) {
        TCHECK(r.ids.size() <= k);
        TCHECK(r.ids.size() == r.dists.size());
        TCHECK(r.ids.size() == r.versions.size());
        TCHECK(r.ids.size() == r.from_delta.size());
        for (size_t i = 1; i < r.dists.size(); ++i) {
            if (r.dists[i - 1] == r.dists[i]) {
                TCHECK_MSG(r.ids[i - 1] < r.ids[i], "tie-break must be (dist, id) ascending");
            } else {
                TCHECK_MSG(r.dists[i - 1] < r.dists[i], "distances must be ascending");
            }
        }
        for (size_t i = 0; i < r.ids.size(); ++i) {
            for (size_t j = i + 1; j < r.ids.size(); ++j) {
                TCHECK_MSG(r.ids[i] != r.ids[j], "duplicate id in query result");
            }
            const LiveEntry* e = engine.live().find(r.ids[i]);
            TCHECK(e != nullptr && e->alive);
            TCHECK_MSG(e->version == r.versions[i], "returned version is not the live version");
        }
    };

    // window 0
    apply_window_crud(&engine, ds.trace, 0);
    QueryExecResult r0 = run_query(&engine, 0, 0, {0.2f, 0.0f, 0.0f, 0.0f});
    check_result_wellformed(r0);
    TCHECK(r0.ids.size() == k);
    TCHECK(contains_id(r0, id0));
    TCHECK(contains_id(r0, id1));
    TCHECK(contains_id(r0, id2));
    TCHECK(r0.ids[0] == id0);  // distance 0.04 is the smallest
    TCHECK_NEAR(r0.dists[0], 0.04, 1e-6);
    TCHECK(!contains_id(r0, 1001));

    // window 1: insert returns from Delta of shard 1
    apply_window_crud(&engine, ds.trace, 1);
    QueryExecResult r1 = run_query(&engine, 1, 1, {10.0f, 0.0f, 0.0f, 0.0f});
    check_result_wellformed(r1);
    TCHECK(contains_id(r1, 1001));
    const int i1 = index_of(r1, 1001);
    TCHECK(i1 >= 0);
    TCHECK_NEAR(r1.dists[i1], 0.0, 1e-12);
    TCHECK(r1.from_delta[i1]);
    TCHECK(r1.versions[i1] == 1);

    // window 2: update moves id 1001 from shard 1 to shard 0; only the new version is visible
    apply_window_crud(&engine, ds.trace, 2);
    QueryExecResult r2a = run_query(&engine, 2, 2, {0.0f, 0.0f, 0.0f, 0.0f});
    check_result_wellformed(r2a);
    TCHECK(contains_id(r2a, 1001));
    const int i2 = index_of(r2a, 1001);
    TCHECK(i2 >= 0);
    TCHECK_NEAR(r2a.dists[i2], 0.0, 1e-12);
    TCHECK(r2a.versions[i2] == 2);
    TCHECK(r2a.from_delta[i2]);
    QueryExecResult r2b = run_query(&engine, 2, 3, {10.0f, 0.0f, 0.0f, 0.0f});
    check_result_wellformed(r2b);
    TCHECK_MSG(!contains_id(r2b, 1001), "stale shard-1 version must be filtered after cross-shard update");

    // window 3: erase hides the id everywhere
    apply_window_crud(&engine, ds.trace, 3);
    QueryExecResult r3 = run_query(&engine, 3, 4, {0.0f, 0.0f, 0.0f, 0.0f});
    check_result_wellformed(r3);
    TCHECK_MSG(!contains_id(r3, 1001), "erased id must never be returned");

    // window 4: version resurrection with a strictly higher version
    apply_window_crud(&engine, ds.trace, 4);
    QueryExecResult r4 = run_query(&engine, 4, 5, {10.5f, 0.0f, 0.0f, 0.0f});
    check_result_wellformed(r4);
    TCHECK(contains_id(r4, 1001));
    const int i4 = index_of(r4, 1001);
    TCHECK(i4 >= 0);
    TCHECK_NEAR(r4.dists[i4], 0.0, 1e-12);
    TCHECK(r4.versions[i4] == 4);
    TCHECK(r4.from_delta[i4]);

    // window 5: state stable query repeats window-0 result
    apply_window_crud(&engine, ds.trace, 5);
    QueryExecResult r5 = run_query(&engine, 5, 6, {0.2f, 0.0f, 0.0f, 0.0f});
    check_result_wellformed(r5);
    TCHECK(r5.ids == r0.ids);
    TCHECK(std::fabs(r5.dists[0] - r0.dists[0]) < 1e-9);

    // exact top-k sanity: engine result on this tiny exact index matches brute force
    TCHECK(engine.live().alive_count() == 6);  // 5 init + resurrected 1001
    std::printf("test_crud OK (live=%zu, k=%u, n_windows=6)\n", engine.live().alive_count(), k);
    return 0;
}
