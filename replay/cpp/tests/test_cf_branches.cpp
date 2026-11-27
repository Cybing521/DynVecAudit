// Q-BAM M1 engine -- test 7/7: counterfactual branch mechanics. All branches of one decision
// block must start from the same state hash; single actions emit window/block/branch rows;
// pair actions emit a refresh_pair branch row; merge:r is supported as a diagnostic action;
// noop branches reproduce the no-op replay trajectory; repeated cf runs are deterministic.
#include <algorithm>
#include <string>
#include <vector>

#include "tests/test_util.h"

using namespace qbam;
using namespace qbam::test;

namespace {

size_t count_with(const std::vector<std::string>& lines, const std::string& needle) {
    size_t n = 0;
    for (const std::string& line : lines) {
        if (line.find(needle) != std::string::npos) ++n;
    }
    return n;
}

void compare_branches(const std::vector<BranchRecord>& a, const std::vector<BranchRecord>& b) {
    TCHECK(a.size() == b.size());
    for (size_t i = 0; i < a.size(); ++i) {
        TCHECK(a[i].action.name() == b[i].action.name());
        TCHECK(a[i].action.shard == b[i].action.shard);
        TCHECK(a[i].action.shard2 == b[i].action.shard2);
        TCHECK(a[i].state_hash == b[i].state_hash);
        TCHECK(a[i].copy_cpu_ms >= 0.0 && b[i].copy_cpu_ms >= 0.0);
        TCHECK(a[i].j == b[i].j);
        TCHECK(a[i].windows.size() == b[i].windows.size());
        for (size_t h = 0; h < a[i].windows.size(); ++h) {
            TCHECK_NEAR(a[i].windows[h].recall, b[i].windows[h].recall, 0.0);
            TCHECK_NEAR(a[i].windows[h].route_upper, b[i].windows[h].route_upper, 0.0);
            TCHECK_NEAR(a[i].windows[h].index_loss, b[i].windows[h].index_loss, 0.0);
            TCHECK_NEAR(a[i].windows[h].returned_delta_fraction,
                        b[i].windows[h].returned_delta_fraction, 0.0);
            TCHECK(a[i].windows[h].query_count == b[i].windows[h].query_count);
        }
    }
}

}  // namespace

int main() {
    const uint32_t dim = 4;
    const uint32_t R = 4;
    const uint32_t k = 3;
    MiniBuilder b(dim, R);
    for (uint32_t r = 0; r < R; ++r) {
        const float base = 10.0f * static_cast<float>(r);
        for (uint32_t i = 0; i < 4; ++i) {
            b.add_init(r, {base + 0.5f * static_cast<float>(i), 0.0f, 0.0f, 0.0f});
        }
    }
    for (uint32_t w = 0; w < 8; ++w) {
        const uint64_t updated_id = 1 + (w % 16);
        const uint32_t ref_upd =
            b.add_vec({10.0f + 0.2f * static_cast<float>(w), 0.0f, 0.0f, 0.0f});
        b.event_update(w, updated_id, 2 + w, ref_upd);
        const uint32_t ref_ins =
            b.add_vec({20.0f + 0.1f * static_cast<float>(w), 0.0f, 0.0f, 0.0f});
        b.event_insert(w, 900 + w, 1, ref_ins);
        b.event_query(w, {0.2f + 0.1f * static_cast<float>(w), 0.0f, 0.0f, 0.0f});
        b.event_query(w, {10.2f + 0.1f * static_cast<float>(w), 0.0f, 0.0f, 0.0f});
    }
    MiniDataset ds = b.finalize(8, 4, k);  // J=2 decision blocks
    EngineConfig cfg = make_config(R, dim, k);

    // ---- a separate no-op replay gives the reference trajectory for noop branches ----
    PolicyRuntime noop_rt;
    policy_runtime_init(&noop_rt, "noop", PolicyParams{});
    std::vector<std::string> replay_lines;
    ReplayModeResult reference;
    {
        ReplayEngine e = make_engine(ds, cfg);
        reference = run_replay_mode(&e, noop_rt, &replay_lines, nullptr);
        TCHECK(reference.blocks.size() == 2);
    }

    // ---- recent-window state statistics used by the G0 Query x Delta heuristic ----
    {
        ReplayEngine e = make_engine(ds, cfg);
        for (uint32_t w = 0; w < 4; ++w) {
            e.process_window(w, nullptr, "unit_test");
        }
        std::vector<ShardRecentStats> recent;
        e.recent_shard_stats(4, 8, &recent);
        TCHECK(recent.size() == R);
        for (uint32_t r = 0; r < R; ++r) {
            TCHECK_MSG(recent[r].q_accesses_recent == 8,
                       "2 queries x 4 windows x all-shard routing = 8 accesses per shard");
            TCHECK(recent[r].base_search_us_recent > 0.0);
        }
        // Window 0 has no history.
        std::vector<ShardRecentStats> empty;
        e.recent_shard_stats(0, 8, &empty);
        for (uint32_t r = 0; r < R; ++r) TCHECK(empty[r].q_accesses_recent == 0);
    }

    // ---- run cf with single actions, pair actions and a merge action ----
    CfParams cf;
    policy_runtime_init(&cf.parent, "noop", PolicyParams{});
    cf.sample_blocks = {0, 1};
    cf.actions = {Action::noop(), Action::refresh(0), Action::refresh(1), Action::merge(2)};
    cf.pair_actions = {{0, 1}};

    std::vector<std::string> lines1;
    CfModeResult r1;
    {
        ReplayEngine e = make_engine(ds, cfg);
        r1 = run_cf_mode(&e, cf, &lines1);
    }
    TCHECK(r1.branches.size() == 10);  // 2 blocks x (4 actions + 1 pair)
    TCHECK(r1.parent_blocks.size() == 2);
    TCHECK(count_with(lines1, "\"type\":\"state\"") == 2);
    TCHECK(count_with(lines1, "\"type\":\"branch\"") == 10);
    TCHECK(count_with(lines1, "\"type\":\"block\"") == 8);
    TCHECK(count_with(lines1, "\"type\":\"window\"") == 32);  // 2 x 4 actions x K=4

    // every branch of one block starts from the same pre-action hash; state rows carry it
    for (uint32_t j = 0; j < 2; ++j) {
        std::string hash;
        for (const BranchRecord& br : r1.branches) {
            if (br.j != j) continue;
            TCHECK(br.windows.size() == 4);
            if (hash.empty()) {
                hash = br.state_hash;
            } else {
                TCHECK_MSG(br.state_hash == hash, "cf branches must share the decision state hash");
            }
        }
        TCHECK_MSG(!hash.empty(), "missing branches for block");
        TCHECK_MSG(count_with(lines1, "\"state_hash\":\"" + hash + "\"") >= 1,
                   "state row must carry the decision state hash");
    }
    TCHECK(r1.branches[0].state_hash != r1.branches[5].state_hash);

    // action semantics (no timing assertions on this tiny fixture: refresh CPU is far below the
    // clock-accounting granularity; cost scaling is validated on the real 100K workload)
    for (const BranchRecord& br : r1.branches) {
        if (br.action.is_noop()) {
            TCHECK(br.maintenance_cpu_ms == 0.0);
        } else {
            TCHECK(br.maintenance_cpu_ms >= 0.0);
        }
        if (br.action.name() == "merge") {
            if (br.j == 1) {
                // shard 2 has live Delta inserts at block 1 => Base deep copy is charged
                // separately in copy_cpu_ms (maintenance includes it).
                TCHECK_MSG(br.copy_cpu_ms > 0.0, "merge must report its Base deep-copy cost");
                TCHECK(br.maintenance_cpu_ms >= br.copy_cpu_ms);
            }
        } else {
            TCHECK_MSG(br.copy_cpu_ms == 0.0, "only merge:r has a Base deep-copy cost");
        }
        if (br.action.name() == "refresh_pair") {
            TCHECK(br.action.shard == 0 && br.action.shard2 == 1);
        }
    }
    // pair action must be exactly "refresh(r) then refresh(s)" (state-hash equivalence)
    {
        ReplayEngine e = make_engine(ds, cfg);
        for (uint32_t w = 0; w < 4; ++w) e.process_window(w, nullptr, "unit_test");
        double m = 0.0;
        ReplayEngine seq = e.clone_for_branch();
        seq.apply_action(Action::refresh(0), &m);
        seq.apply_action(Action::refresh(1), &m);
        ReplayEngine pair = e.clone_for_branch();
        pair.apply_action(Action::refresh_pair(0, 1), &m);
        TCHECK(seq.state_hash() == pair.state_hash());
        // merge semantic: live Delta labels move into Base, live set unchanged, state changes
        ReplayEngine merged = e.clone_for_branch();
        const std::vector<uint64_t> live_before = merged.live().alive_ids_sorted();
        const size_t delta_before = merged.shards()[2].delta_n();
        const size_t base_before = merged.shards()[2].base_n();
        merged.apply_action(Action::merge(2), &m);
        TCHECK(merged.shards()[2].delta_n() < delta_before);
        TCHECK(merged.shards()[2].base_n() > base_before);
        TCHECK(merged.live().alive_ids_sorted() == live_before);
        TCHECK(merged.state_hash() != e.state_hash());
    }
    // window rows carry the cf branch identifiers required for (j, action, shard, h) grouping
    for (const std::string& line : lines1) {
        if (line.find("\"type\":\"window\"") == std::string::npos) continue;
        for (const char* key : {"\"j\":", "\"action\":", "\"shard\":", "\"h\":", "\"shard_q_accesses\":"}) {
            TCHECK_MSG(line.find(key) != std::string::npos, std::string("window row missing ") + key);
        }
    }
    TCHECK(count_with(lines1, "\"action\":\"refresh_pair\"") >= 2);
    TCHECK(count_with(lines1, "\"shard2\":1") >= 2);
    // config-style action string (inverse of cf.actions parsing) for every branch kind
    TCHECK(count_with(lines1, "\"action_str\":\"refresh:0+1\"") >= 2);
    TCHECK(count_with(lines1, "\"action_str\":\"refresh:0\"") >= 2);
    TCHECK(count_with(lines1, "\"action_str\":\"noop\"") >= 2);
    TCHECK(Action::refresh_pair(2, 7).config_string() == "refresh:2+7");
    TCHECK(Action::refresh(3).config_string() == "refresh:3");
    TCHECK(Action::merge(5).config_string() == "merge:5");
    TCHECK(Action::noop().config_string() == "noop");
    TCHECK(count_with(lines1, "\"action\":\"merge\"") >= 2);

    // noop branch reproduces the plain no-op replay trajectory (same state, no maintenance)
    for (uint32_t j = 0; j < 2; ++j) {
        const BranchRecord* noop_branch = nullptr;
        for (const BranchRecord& br : r1.branches) {
            if (br.j == j && br.action.is_noop()) noop_branch = &br;
        }
        TCHECK(noop_branch != nullptr);
        for (uint32_t h = 0; h < 4; ++h) {
            const WindowMetrics& bw = noop_branch->windows[h];
            const WindowMetrics& rw = reference.windows[j * 4 + h];
            TCHECK_NEAR(bw.recall, rw.recall, 0.0);
            TCHECK_NEAR(bw.route_upper, rw.route_upper, 0.0);
            TCHECK_NEAR(bw.index_loss, rw.index_loss, 0.0);
            TCHECK(bw.query_count == rw.query_count);
        }
    }

    // ---- repeated cf runs are deterministic (excluding timing) ----
    std::vector<std::string> lines2;
    CfModeResult r2;
    {
        ReplayEngine e = make_engine(ds, cfg);
        r2 = run_cf_mode(&e, cf, &lines2);
    }
    compare_branches(r1.branches, r2.branches);
    TCHECK(lines1.size() == lines2.size());

    std::printf("test_cf_branches OK (branches=%zu, lines=%zu, state rows=2)\n",
                r1.branches.size(), lines1.size());
    return 0;
}
