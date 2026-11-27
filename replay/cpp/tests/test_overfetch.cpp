// Q-BAM M1 engine -- test 3/7: tombstone overfetch (base candidates are filtered after ANN
// search; k' must grow until enough live candidates are collected) and underfill accounting.
#include <vector>

#include "tests/test_util.h"

using namespace qbam;
using namespace qbam::test;

namespace {

// Single-shard dataset: n vectors on the x axis at i*0.01, ids 1..n (add_init assigns 1..n).
MiniDataset make_line_dataset(uint32_t n, uint32_t n_erase, uint32_t k) {
    MiniBuilder b(2, 1);
    std::vector<uint64_t> ids;
    for (uint32_t i = 0; i < n; ++i) {
        ids.push_back(b.add_init(0, {0.01f * static_cast<float>(i), 0.0f}));
    }
    for (uint32_t i = 0; i < n_erase; ++i) {
        b.event_erase(0, ids[i], 2);
    }
    b.event_query(0, {0.0f, 0.0f});
    return b.finalize(1, 1, k);
}

}  // namespace

int main() {
    const uint32_t k = 10;

    // ---- A: only 5 live vectors survive => unavoidable underfill, k' must expand to n ----
    MiniDataset dsA = make_line_dataset(60, 55, k);
    EngineConfig cfgA = make_config(1, 2, k);
    ReplayEngine engineA = make_engine(dsA, cfgA);
    apply_window_crud(&engineA, dsA.trace, 0);
    QueryExecResult qA = run_query(&engineA, 0, 0, {0.0f, 0.0f});
    TCHECK(qA.ids.size() == 5);
    TCHECK(qA.underfilled);
    TCHECK_MSG(qA.overfetch_expansions >= 1, "tombstone filtering must trigger k' expansion");
    TCHECK_MSG(qA.stale_candidates_filtered >= 50, "most ANN candidates should be stale");
    TCHECK(engineA.shards()[0].base_n() == 60);
    TCHECK(engineA.shards()[0].tombstone_n == 55);
    TCHECK_MSG(engineA.live().alive_count() == 5, "live set after erases");

    // window metrics must account for the expansion and the underfilled query
    MiniDataset dsA2 = make_line_dataset(60, 55, k);
    ReplayEngine engineA2 = make_engine(dsA2, cfgA);
    WindowMetrics wmA = engineA2.process_window(0, nullptr, "unit_test");
    TCHECK(wmA.query_count == 1);
    TCHECK(wmA.underfilled_queries == 1);
    TCHECK(wmA.overfetch_count == 1);
    TCHECK(wmA.candidate_count_max == 5);
    TCHECK(wmA.base_search_cpu_ms >= 0.0 && wmA.delta_scan_cpu_ms >= 0.0 &&
           wmA.selection_cpu_ms >= 0.0);

    // ---- B: enough live vectors remain => expansion succeeds and fills k ----
    MiniDataset dsB = make_line_dataset(100, 45, k);
    EngineConfig cfgB = make_config(1, 2, k);
    ReplayEngine engineB = make_engine(dsB, cfgB);
    apply_window_crud(&engineB, dsB.trace, 0);
    QueryExecResult qB = run_query(&engineB, 0, 0, {0.0f, 0.0f});
    TCHECK(qB.ids.size() == k);
    TCHECK(!qB.underfilled);
    TCHECK_MSG(qB.overfetch_expansions >= 1, "45 stale nearest neighbours must trigger expansion");
    TCHECK_MSG(qB.stale_candidates_filtered >= 40, "stale candidate accounting");
    TCHECK(qB.base_candidates_requested > static_cast<int>(k));

    // ---- C: no tombstones => no overfetch at all ----
    MiniDataset dsC = make_line_dataset(100, 0, k);
    ReplayEngine engineC = make_engine(dsC, cfgB);
    apply_window_crud(&engineC, dsC.trace, 0);
    QueryExecResult qC = run_query(&engineC, 0, 0, {0.0f, 0.0f});
    TCHECK(qC.ids.size() == k);
    TCHECK(!qC.underfilled);
    TCHECK_MSG(qC.overfetch_expansions == 0, "no overfetch expected when no tombstones exist");
    TCHECK(qC.stale_candidates_filtered == 0);

    std::printf(
        "test_overfetch OK (A: %zu/%u underfilled, B: %zu ids, expansions A/B/C=%d/%d/%d)\n",
        qA.ids.size(), k, qB.ids.size(), qA.overfetch_expansions, qB.overfetch_expansions,
        qC.overfetch_expansions);
    return 0;
}
