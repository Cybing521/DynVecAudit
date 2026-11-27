// Q-BAM M1 engine -- test 4/7: replay determinism. Two independent runs of the same config
// must produce identical action sequences, returned ids, quality metrics and state hashes
// (timing floats are explicitly excluded; engine_contract.md §8.10).
#include <vector>

#include "tests/test_util.h"

using namespace qbam;
using namespace qbam::test;

namespace {

void compare_block(const BlockMetrics& a, const BlockMetrics& b) {
    TCHECK(a.action.name() == b.action.name());
    TCHECK(a.action.shard == b.action.shard);
    TCHECK(a.state_hash == b.state_hash);
    TCHECK_NEAR(a.recall, b.recall, 0.0);
    TCHECK_NEAR(a.route_upper, b.route_upper, 0.0);
    TCHECK_NEAR(a.index_loss, b.index_loss, 0.0);
    TCHECK_NEAR(a.returned_delta_fraction, b.returned_delta_fraction, 0.0);
    TCHECK(a.overfetch_count == b.overfetch_count);
    TCHECK(a.underfilled_queries == b.underfilled_queries);
    TCHECK(a.window_start == b.window_start);
    TCHECK(a.shards.size() == b.shards.size());
    for (size_t r = 0; r < a.shards.size(); ++r) {
        TCHECK(a.shards[r].base_n == b.shards[r].base_n);
        TCHECK(a.shards[r].delta_n == b.shards[r].delta_n);
        TCHECK(a.shards[r].tombstone_n == b.shards[r].tombstone_n);
        TCHECK(a.shards[r].q_accesses == b.shards[r].q_accesses);
    }
}

void compare_window(const WindowMetrics& a, const WindowMetrics& b) {
    TCHECK(a.window == b.window);
    TCHECK(a.query_count == b.query_count);
    TCHECK_NEAR(a.recall, b.recall, 0.0);
    TCHECK_NEAR(a.route_upper, b.route_upper, 0.0);
    TCHECK_NEAR(a.index_loss, b.index_loss, 0.0);
    TCHECK_NEAR(a.returned_delta_fraction, b.returned_delta_fraction, 0.0);
    TCHECK(a.overfetch_count == b.overfetch_count);
    TCHECK(a.underfilled_queries == b.underfilled_queries);
    TCHECK(a.shard_delta_n == b.shard_delta_n);
    TCHECK(a.shard_base_n == b.shard_base_n);
    TCHECK(a.shard_q_accesses == b.shard_q_accesses);
}

}  // namespace

int main() {
    const uint32_t dim = 4;
    MiniBuilder b(dim, 2);
    for (uint32_t i = 0; i < 5; ++i) {
        b.add_init(0, {static_cast<float>(i) * 0.5f, 0.0f, 0.0f, 0.0f});
        b.add_init(1, {10.0f + static_cast<float>(i) * 0.5f, 0.0f, 0.0f, 0.0f});
    }
    for (uint32_t w = 0; w < 4; ++w) {
        const uint32_t ref_update = b.add_vec({1.0f + 0.1f * static_cast<float>(w), 0.0f, 0.0f, 0.0f});
        b.event_update(w, 1 + (w % 5), 2 + w, ref_update);
        const uint32_t ref_insert = b.add_vec({10.1f + 0.1f * static_cast<float>(w), 0.0f, 0.0f, 0.0f});
        b.event_insert(w, 900 + w, 1, ref_insert);
        b.event_query(w, {0.5f + 0.1f * static_cast<float>(w), 0.0f, 0.0f, 0.0f});
        b.event_query(w, {10.5f + 0.1f * static_cast<float>(w), 0.0f, 0.0f, 0.0f});
    }
    const uint32_t k = 3;
    MiniDataset ds = b.finalize(4, 2, k);  // K=2 => J=2 decision blocks
    EngineConfig cfg = make_config(2, dim, k);

    // ---- 1) replay(noop) twice: metrics and state hashes must be identical
    {
        PolicyRuntime rt1;
        policy_runtime_init(&rt1, "noop", PolicyParams{});
        PolicyRuntime rt2 = rt1;
        ReplayEngine e1 = make_engine(ds, cfg);
        ReplayEngine e2 = make_engine(ds, cfg);
        std::vector<std::string> lines1, lines2;
        LineSink* null_sink = nullptr;
        ReplayModeResult r1 = run_replay_mode(&e1, rt1, &lines1, null_sink);
        ReplayModeResult r2 = run_replay_mode(&e2, rt2, &lines2, null_sink);
        TCHECK(r1.blocks.size() == 2 && r2.blocks.size() == 2);
        TCHECK(r1.blocks[0].action.is_noop());
        for (size_t i = 0; i < r1.blocks.size(); ++i) compare_block(r1.blocks[i], r2.blocks[i]);
        TCHECK(r1.windows.size() == r2.windows.size());
        for (size_t i = 0; i < r1.windows.size(); ++i) compare_window(r1.windows[i], r2.windows[i]);
        TCHECK(lines1.size() == lines2.size());  // window + block records
    }

    // ---- 2) replay(fixed cadence=1) twice: refresh actions deterministic, state hashes equal
    {
        PolicyParams params;
        params.cadence_blocks = 1;
        params.phase = 0;
        PolicyRuntime rt1;
        policy_runtime_init(&rt1, "fixed", params);
        PolicyRuntime rt2 = rt1;
        ReplayEngine e1 = make_engine(ds, cfg);
        ReplayEngine e2 = make_engine(ds, cfg);
        std::vector<std::string> lines1, lines2;
        ReplayModeResult r1 = run_replay_mode(&e1, rt1, &lines1, nullptr);
        ReplayModeResult r2 = run_replay_mode(&e2, rt2, &lines2, nullptr);
        TCHECK(r1.blocks.size() == 2);
        TCHECK(r1.blocks[0].action.is_refresh());
        TCHECK(r1.blocks[0].action.shard == 0);
        TCHECK(r1.blocks[1].action.is_refresh());
        TCHECK(r1.blocks[1].action.shard == 1);
        TCHECK(r1.blocks[0].maintenance_cpu_ms >= 0.0);
        for (size_t i = 0; i < r1.blocks.size(); ++i) compare_block(r1.blocks[i], r2.blocks[i]);
        for (size_t i = 0; i < r1.windows.size(); ++i) compare_window(r1.windows[i], r2.windows[i]);
        // refresh happened inside the run: first block's pre-action state hash must differ from
        // the post-refresh entering state of the window records is not tracked, but the second
        // block hash must differ from the first (shard 1 rebuilt).
        TCHECK(r1.blocks[0].state_hash != r1.blocks[1].state_hash);
    }

    // ---- 3) per-query determinism: returned ids/dists/from_delta are pointer-identical
    {
        ReplayEngine e1 = make_engine(ds, cfg);
        ReplayEngine e2 = make_engine(ds, cfg);
        uint32_t query_count = 0;
        for (uint32_t w = 0; w < ds.trace.n_windows; ++w) {
            apply_window_crud(&e1, ds.trace, w);
            apply_window_crud(&e2, ds.trace, w);
            TCHECK(e1.state_hash() == e2.state_hash());
            uint32_t gt_index = ds.trace.window_query_start[w];
            for (uint32_t i = ds.trace.window_begin_of(w); i < ds.trace.window_end_of(w); ++i) {
                const TraceEvent& ev = ds.trace.events[i];
                if (ev.type != EventType::Query) continue;
                const float* q = ds.pool.vec(ev.vec_ref);
                QueryExecResult r1 = e1.execute_query(w, gt_index, q);
                QueryExecResult r2 = e2.execute_query(w, gt_index, q);
                TCHECK(r1.ids == r2.ids);
                TCHECK(r1.versions == r2.versions);
                TCHECK(r1.from_delta == r2.from_delta);
                TCHECK(r1.dists.size() == r2.dists.size());
                for (size_t j = 0; j < r1.dists.size(); ++j) {
                    TCHECK(r1.dists[j] == r2.dists[j]);
                }
                TCHECK(r1.stale_candidates_filtered == r2.stale_candidates_filtered);
                TCHECK(r1.base_candidates_requested == r2.base_candidates_requested);
                ++gt_index;
                ++query_count;
            }
        }
        TCHECK(query_count == 8);
    }

    // ---- 4) query_delta heuristic policy: deterministic actions, partial last block ----
    {
        MiniBuilder bq(dim, 2);
        const uint32_t windows = 5;  // K=2 => J=3 with a 1-window tail block
        for (uint32_t i = 0; i < 4; ++i) {
            bq.add_init(0, {0.5f * static_cast<float>(i), 0.0f, 0.0f, 0.0f});
            bq.add_init(1, {10.0f + 0.5f * static_cast<float>(i), 0.0f, 0.0f, 0.0f});
        }
        for (uint32_t w = 0; w < windows; ++w) {
            const uint32_t ref = bq.add_vec({0.2f + 0.05f * static_cast<float>(w), 0.0f, 0.0f, 0.0f});
            bq.event_insert(w, 500 + w, 1, ref);
            bq.event_query(w, {0.2f, 0.0f, 0.0f, 0.0f});
        }
        MiniDataset qds = bq.finalize(windows, 2, 3);
        EngineConfig qcfg = make_config(2, dim, 3);
        PolicyParams qp;
        qp.qmass_window = 3;
        PolicyRuntime rt1;
        policy_runtime_init(&rt1, "query_delta", qp);
        PolicyRuntime rt2 = rt1;
        ReplayEngine qe1 = make_engine(qds, qcfg);
        ReplayEngine qe2 = make_engine(qds, qcfg);
        std::vector<std::string> ql1, ql2;
        ReplayModeResult qr1 = run_replay_mode(&qe1, rt1, &ql1, nullptr);
        ReplayModeResult qr2 = run_replay_mode(&qe2, rt2, &ql2, nullptr);
        TCHECK(qr1.blocks.size() == 3);          // ceil(5/2)
        TCHECK(qr1.blocks[2].n_windows == 1);    // partial tail block
        TCHECK(qr1.windows.size() == windows);
        for (size_t i = 0; i < qr1.blocks.size(); ++i) {
            compare_block(qr1.blocks[i], qr2.blocks[i]);
            if (!qr1.blocks[i].action.is_noop()) {
                TCHECK(qr1.blocks[i].action.is_refresh());
                TCHECK(qr1.blocks[i].action.shard >= 0 && qr1.blocks[i].action.shard < 2);
            }
        }
        for (size_t i = 0; i < qr1.windows.size(); ++i) compare_window(qr1.windows[i], qr2.windows[i]);
        TCHECK(ql1.size() == ql2.size());
    }

    std::printf("test_replay_determinism OK (blocks=2/3, queries=8, two runs bit-identical)\n");
    return 0;
}
