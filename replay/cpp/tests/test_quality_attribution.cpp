// Q-BAM M1 engine -- test 5/7: quality attribution (recall_at_k, route_upper, index_loss,
// returned_delta_fraction) on a hand-computable dataset, plus gt.bin verification.
#include <vector>

#include "tests/test_util.h"

using namespace qbam;
using namespace qbam::test;

namespace {
const std::vector<float> kC0 = {100.0f, 0.0f};
const std::vector<float> kC1 = {0.0f, 0.1f};
}  // namespace

int main() {
    const uint32_t dim = 2;
    const uint32_t k = 5;

    // ---- scenario A/B: routing upper bound and index loss, no Delta ----
    MiniBuilder b(dim, 2);
    b.set_centroid(0, kC0);
    b.set_centroid(1, kC1);
    // shard 0: distances from q=[0,0] are 1, 4, 9, 16, 25, 36, 49
    for (int i = 1; i <= 7; ++i) b.add_init(0, {static_cast<float>(i), 0.0f});
    // shard 1: distances 0.25, 6.25, 12.25
    b.add_init(1, {0.5f, 0.0f});
    b.add_init(1, {2.5f, 0.0f});
    b.add_init(1, {3.5f, 0.0f});
    b.event_query(0, {0.0f, 0.0f});
    MiniDataset ds = b.finalize(1, 1, k);

    // gt top-5: shard1 ids (d 0.25, 6.25) + shard0 ids (d 1, 4, 9) => 3 from shard0, 2 from shard1.
    {
        EngineConfig cfg_all = make_config(2, dim, k, /*nprobe=*/2);
        ReplayEngine e = make_engine(ds, cfg_all);
        apply_window_crud(&e, ds.trace, 0);
        QueryExecResult r = run_query(&e, 0, 0, {0.0f, 0.0f});
        TCHECK(r.ids.size() == k);
        TCHECK_NEAR(r.recall, 1.0, 0.0);
        TCHECK_NEAR(r.route_upper, 1.0, 0.0);
        TCHECK_NEAR(r.index_loss, 0.0, 0.0);
        TCHECK_NEAR(r.returned_delta_fraction, 0.0, 0.0);
        TCHECK(r.visited_shards.size() == 2);
    }
    {
        EngineConfig cfg_1 = make_config(2, dim, k, /*nprobe=*/1);
        ReplayEngine e = make_engine(ds, cfg_1);
        apply_window_crud(&e, ds.trace, 0);
        QueryExecResult r = run_query(&e, 0, 0, {0.0f, 0.0f});
        TCHECK(r.visited_shards.size() == 1);
        TCHECK(r.visited_shards[0] == 1);  // query [0,0] is nearest to centroid 1
        TCHECK(r.ids.size() == 3);          // shard 1 only has 3 vectors => underfilled
        TCHECK(r.underfilled);
        TCHECK_NEAR(r.route_upper, 0.4, 0.0);  // {id8,id9} of gt top-5 live in shard 1
        TCHECK_NEAR(r.recall, 0.4, 0.0);
        TCHECK_NEAR(r.index_loss, 0.0, 0.0);  // everything visited was returned exactly
        TCHECK_NEAR(r.returned_delta_fraction, 0.0, 0.0);
    }

    // ---- scenario C: returned_delta_fraction and Delta-attributed recall ----
    {
        MiniBuilder bc(dim, 2);
        bc.set_centroid(0, kC0);
        bc.set_centroid(1, kC1);
        for (int i = 1; i <= 7; ++i) bc.add_init(0, {static_cast<float>(i), 0.0f});
        bc.add_init(1, {0.5f, 0.0f});
        bc.add_init(1, {2.5f, 0.0f});
        bc.add_init(1, {3.5f, 0.0f});
        bc.event_query(0, {0.0f, 0.0f});                          // gt row 0
        const uint32_t ref = bc.add_vec({0.1f, 0.0f});
        bc.event_insert(1, 11, 1, ref);                            // nearest shard is 1
        bc.event_query(1, {0.0f, 0.0f});                          // gt row 1
        MiniDataset dsc = bc.finalize(2, 1, k);
        EngineConfig cfg_1 = make_config(2, dim, k, /*nprobe=*/1);
        ReplayEngine e = make_engine(dsc, cfg_1);
        apply_window_crud(&e, dsc.trace, 0);
        QueryExecResult r0 = run_query(&e, 0, 0, {0.0f, 0.0f});
        TCHECK_NEAR(r0.returned_delta_fraction, 0.0, 0.0);
        apply_window_crud(&e, dsc.trace, 1);
        QueryExecResult r1 = run_query(&e, 1, 1, {0.0f, 0.0f});
        TCHECK(contains_id(r1, 11));
        TCHECK(r1.ids.size() == 4);  // 3 base + 1 delta (k=5 => underfilled)
        TCHECK(r1.underfilled);
        TCHECK_NEAR(r1.returned_delta_fraction, 1.0 / 4.0, 0.0);
        // gt top-5 at window 1: id11(0.01), 8(0.25), 1(1), 2(4), 9(6.25); shard-1 members: 11,8,9
        TCHECK_NEAR(r1.route_upper, 3.0 / 5.0, 0.0);
        TCHECK_NEAR(r1.recall, 3.0 / 5.0, 0.0);
        TCHECK_NEAR(r1.index_loss, 0.0, 0.0);
        for (size_t i = 0; i < r1.ids.size(); ++i) {
            if (r1.ids[i] == 11) TCHECK(r1.from_delta[i]);
        }
    }

    // ---- gt.bin verification helper: brute force must reproduce the stored gt row ----
    {
        LiveStore live;
        for (const InitEntry& e : ds.init.entries) live.assign(e.id, 1, e.shard, e.vec_ref);
        const float* q = ds.pool.vec(ds.trace.events[0].vec_ref);
        std::string msg;
        TCHECK_MSG(verify_gt_row(ds.pool, live, ds.gt, 0, q, k, &msg), msg);
        GtData corrupted = ds.gt;
        corrupted.ids[0] = 999999;
        TCHECK(!verify_gt_row(ds.pool, live, corrupted, 0, q, k, &msg));
        TCHECK(live.alive_count() == 10);
        // dataset-level verifier: replays CRUD through a plain LiveStore and brute-forces all
        // checked rows (no HNSW); must accept the built gt ...
        GtVerifyResult vr = verify_gt_sample(ds.pool, ds.init, ds.centroids, ds.trace, ds.gt, k, 1);
        TCHECK(vr.checked == 1);
        TCHECK(vr.mismatches == 0);
        // ... and reject a corrupted row
        GtVerifyResult vbad =
            verify_gt_sample(ds.pool, ds.init, ds.centroids, ds.trace, corrupted, k, 1);
        TCHECK(vbad.mismatches == 1);
    }

    std::printf("test_quality_attribution OK (route_upper 1.0 vs 0.4, delta frac 0.25)\n");
    return 0;
}
