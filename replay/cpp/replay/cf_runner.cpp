// Q-BAM M1 engine -- replay-mode driver and counterfactual branch runner.
#include "replay/cf_runner.h"

#include <algorithm>

#include "common/logging.h"

namespace qbam {

namespace {

double per_block_budget(const ReplayEngine& engine) {
    const EngineConfig& cfg = engine.cfg();
    const uint32_t J = engine.J();
    if (J == 0) return 0.0;
    return (cfg.kappa_RE * cfg.c_ref_cpu_ms) / static_cast<double>(J);
}

// Runs one CF branch from a clone of `base_engine` (decision-point state) and emits its rows.
BranchRecord run_branch(const ReplayEngine& base_engine, uint32_t j, const Action& action,
                        int action_index, bool emit_block_and_windows,
                        const std::string& decision_state_hash, double budget_nominal,
                        std::vector<std::string>* lines) {
    const std::string& run_id = base_engine.cfg().run_id;
    const uint32_t K = base_engine.K();
    const uint32_t n_windows = base_engine.n_windows();

    ReplayEngine branch = base_engine.clone_for_branch();
    const std::string clone_hash = branch.state_hash();
    QBAM_CHECK(clone_hash == decision_state_hash,
               "cf branch clone state hash mismatch at block " + std::to_string(j) +
                   " (clone=" + clone_hash + " expected=" + decision_state_hash + ")");

    double maintenance_cpu_ms = 0.0;
    const RefreshResult action_result = branch.apply_action(action, &maintenance_cpu_ms);

    std::vector<ShardSnapshot> snapshots;
    branch.capture_shard_counts(&snapshots);
    branch.reset_shard_block_counters();

    const uint32_t w0 = j * K;
    const uint32_t w1 = std::min<uint32_t>(w0 + K, n_windows);
    std::vector<WindowMetrics> windows;
    windows.reserve(w1 - w0);
    for (uint32_t w = w0; w < w1; ++w) {
        WindowMetrics wm = branch.process_window(w, nullptr, run_id);
        if (emit_block_and_windows) {
            lines->push_back(json_window_record(run_id, wm, j, action, action_index, w - w0 + 1));
        }
        windows.push_back(wm);
    }
    branch.fill_shard_counters(&snapshots);

    if (emit_block_and_windows) {
        BlockMetrics bm = branch.aggregate_block(j, action, 0.0, maintenance_cpu_ms,
                                                  decision_state_hash, budget_nominal, windows,
                                                  snapshots);
        lines->push_back(json_block_record(run_id, bm));
    }

    BranchRecord br;
    br.j = j;
    br.action = action;
    br.state_hash = decision_state_hash;
    br.maintenance_cpu_ms = maintenance_cpu_ms;
    br.copy_cpu_ms = action_result.copy_cpu_ms;
    br.policy_cpu_ms = 0.0;
    br.windows = std::move(windows);
    lines->push_back(json_branch_record(run_id, br));
    return br;
}

// Parent-trajectory block: choose policy action, apply it, replay K windows WITHOUT emitting
// per-window/per-block records (cf mode only outputs branch diagnostics).
BlockMetrics run_parent_block(ReplayEngine* engine, PolicyRuntime* policy_rt, uint32_t j,
                              bool record_policy_cpu) {
    const std::string& run_id = engine->cfg().run_id;
    CpuWallTimer policy_timer;
    policy_timer.start();
    const ActionDecision decision = choose_action(policy_rt, j, engine);
    const Action action = decision.action;
    const double policy_cpu_ms = record_policy_cpu ? policy_timer.cpu_ms() : 0.0;

    const std::string pre_hash = engine->state_hash();
    double maintenance_cpu_ms = 0.0;
    engine->apply_action(action, &maintenance_cpu_ms);
    if (action.is_refresh()) {
        record_refresh_cost(policy_rt, action.shard, maintenance_cpu_ms);
    }

    std::vector<ShardSnapshot> snapshots;
    engine->capture_shard_counts(&snapshots);
    engine->reset_shard_block_counters();

    const uint32_t K = engine->K();
    const uint32_t w0 = j * K;
    const uint32_t w1 = std::min<uint32_t>(w0 + K, engine->n_windows());
    std::vector<WindowMetrics> windows;
    windows.reserve(w1 - w0);
    for (uint32_t w = w0; w < w1; ++w) {
        windows.push_back(engine->process_window(w, nullptr, run_id));
    }
    engine->fill_shard_counters(&snapshots);
    BlockMetrics bm = engine->aggregate_block(j, action, policy_cpu_ms, maintenance_cpu_ms,
                                              pre_hash, per_block_budget(*engine), windows,
                                              snapshots);
    bm.budget_rejected = decision.budget_rejected;
    bm.score_chosen = decision.score_chosen;
    bm.cost_pred_ms = decision.cost_pred_ms;
    bm.lambda = decision.lambda;
    bm.delta_chosen = decision.delta_chosen;
    policy_rt->cum_spend_ms += maintenance_cpu_ms + policy_cpu_ms;
    if (policy_rt->policy == "bam_price") {
        policy_rt->lambda = budget_lambda_update(*policy_rt, j, engine->J());
    }
    return bm;
}

}  // namespace

ReplayModeResult run_replay_mode(ReplayEngine* engine, const PolicyRuntime& policy,
                                 std::vector<std::string>* lines, LineSink* query_sink) {
    QBAM_CHECK(engine != nullptr && lines != nullptr, "run_replay_mode: null argument");
    const std::string& run_id = engine->cfg().run_id;
    const uint32_t J = engine->J();
    const uint32_t K = engine->K();
    const uint32_t n_windows = engine->n_windows();
    const bool e3_policy = policy.policy == "past_qmass" || policy.policy == "forecast_qmass" ||
                           policy.policy == "bam_price";
    // E3 policies use the cumulative budget B=kappa*c_ref set by policy_prepare(); the legacy
    // policies keep the per-block nominal derived from the engine config.
    const double budget_nominal =
        e3_policy ? (J == 0 ? 0.0 : policy.budget_total_ms / static_cast<double>(J))
                  : per_block_budget(*engine);
    PolicyRuntime rt = policy;
    ReplayModeResult out;

    for (uint32_t j = 0; j < J; ++j) {
        CpuWallTimer policy_timer;
        policy_timer.start();
        const ActionDecision decision = choose_action(&rt, j, engine);
        const Action action = decision.action;
        const double policy_cpu_ms = policy_timer.cpu_ms();
        const std::string pre_hash = engine->state_hash();

        double maintenance_cpu_ms = 0.0;
        engine->apply_action(action, &maintenance_cpu_ms);
        if (action.is_refresh()) record_refresh_cost(&rt, action.shard, maintenance_cpu_ms);

        std::vector<ShardSnapshot> snapshots;
        engine->capture_shard_counts(&snapshots);
        engine->reset_shard_block_counters();

        const uint32_t w0 = j * K;
        const uint32_t w1 = std::min<uint32_t>(w0 + K, n_windows);
        std::vector<WindowMetrics> windows;
        windows.reserve(w1 - w0);
        for (uint32_t w = w0; w < w1; ++w) {
            WindowMetrics wm = engine->process_window(w, query_sink, run_id);
            lines->push_back(json_window_record(run_id, wm, j, action, -1, w - w0 + 1));
            windows.push_back(wm);
        }
        engine->fill_shard_counters(&snapshots);
        BlockMetrics bm = engine->aggregate_block(j, action, policy_cpu_ms, maintenance_cpu_ms,
                                                  pre_hash, budget_nominal, windows, snapshots);
        bm.budget_rejected = decision.budget_rejected;
        bm.score_chosen = decision.score_chosen;
        bm.cost_pred_ms = decision.cost_pred_ms;
        bm.lambda = decision.lambda;
        bm.delta_chosen = decision.delta_chosen;
        lines->push_back(json_block_record(run_id, bm));
        out.blocks.push_back(std::move(bm));
        for (const WindowMetrics& wm : windows) out.windows.push_back(wm);
        // budget accounting + integral price controller update for the next block
        rt.cum_spend_ms += maintenance_cpu_ms + policy_cpu_ms;
        if (rt.policy == "bam_price") rt.lambda = budget_lambda_update(rt, j, J);
    }
    uint32_t rejections = 0;
    for (const BlockMetrics& b : out.blocks) rejections += b.budget_rejected ? 1u : 0u;
    QBAM_LOG("[policy] %s: blocks=%u cum_spend=%.2f ms budget=%.2f ms rejections=%u "
             "lambda_final=%.6f forecast_fallback_rows=%llu",
             rt.policy.c_str(), J, rt.cum_spend_ms, rt.budget_total_ms, rejections, rt.lambda,
             static_cast<unsigned long long>(rt.forecast_fallback_rows));
    return out;
}

CfModeResult run_cf_mode(ReplayEngine* engine, const CfParams& cf, std::vector<std::string>* lines) {
    QBAM_CHECK(engine != nullptr && lines != nullptr, "run_cf_mode: null argument");
    const std::string& run_id = engine->cfg().run_id;
    const uint32_t J = engine->J();
    const uint32_t K = engine->K();
    const double budget_nominal = per_block_budget(*engine);
    QBAM_CHECK(!cf.sample_blocks.empty(), "mode=cf requires cf.sample_blocks");
    QBAM_CHECK(!cf.actions.empty(), "mode=cf requires cf.actions");
    std::vector<char> sampled(J, 0);
    for (uint32_t j : cf.sample_blocks) {
        QBAM_CHECK(j < J, "cf.sample_blocks contains out-of-range block " + std::to_string(j));
        sampled[j] = 1;
    }

    PolicyRuntime parent = cf.parent;
    CfModeResult out;
    for (uint32_t j = 0; j < J; ++j) {
        if (sampled[j]) {
            const std::string decision_hash = engine->state_hash();
            std::vector<ShardSnapshot> pre_counts;
            engine->capture_shard_counts(&pre_counts);
            std::vector<ShardRecentStats> recent;
            engine->recent_shard_stats(j * K, 8, &recent);
            lines->push_back(
                json_state_record(run_id, j, j * K, decision_hash, pre_counts, recent));

            for (size_t ai = 0; ai < cf.actions.size(); ++ai) {
                out.branches.push_back(run_branch(*engine, j, cf.actions[ai], static_cast<int>(ai),
                                                  true, decision_hash, budget_nominal, lines));
            }
            for (const std::pair<int, int>& pair : cf.pair_actions) {
                QBAM_CHECK(pair.first >= 0 && static_cast<uint32_t>(pair.first) < engine->R(),
                           "cf.pair_actions first shard out of range");
                QBAM_CHECK(pair.second >= 0 && static_cast<uint32_t>(pair.second) < engine->R(),
                           "cf.pair_actions second shard out of range");
                out.branches.push_back(
                    run_branch(*engine, j, Action::refresh_pair(pair.first, pair.second), -1, false,
                               decision_hash, budget_nominal, lines));
            }
            (void)K;
        }
        out.parent_blocks.push_back(run_parent_block(engine, &parent, j, false));
    }
    return out;
}

}  // namespace qbam
