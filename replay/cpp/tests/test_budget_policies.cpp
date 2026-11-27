// Q-BAM engine -- E3-stage2 budget-allocation policies: lambda controller directionality,
// budget audit, forecast fallback, action_mode legality, and JSONL decision fields.
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "tests/test_util.h"

using namespace qbam;
using namespace qbam::test;

namespace {

PolicyRuntime make_policy(const std::string& name, const PolicyParams& params) {
    PolicyRuntime rt;
    policy_runtime_init(&rt, name, params);
    return rt;
}

// Mini dataset: R=2, K=2, 4 windows => J=2. Inserts land near shard 0 (shard 1 keeps delta=0).
MiniDataset make_policy_dataset() {
    MiniBuilder b(4, 2);
    for (uint32_t i = 0; i < 4; ++i) {
        b.add_init(0, {0.5f * static_cast<float>(i), 0.0f, 0.0f, 0.0f});
        b.add_init(1, {10.0f + 0.5f * static_cast<float>(i), 0.0f, 0.0f, 0.0f});
    }
    for (uint32_t w = 0; w < 4; ++w) {
        const uint32_t ref = b.add_vec({0.25f + 0.1f * static_cast<float>(w), 0.0f, 0.0f, 0.0f});
        b.event_insert(w, 500 + w, 1, ref);
        b.event_query(w, {0.25f, 0.0f, 0.0f, 0.0f});   // routed to shard 0
        b.event_query(w, {10.25f, 0.0f, 0.0f, 0.0f});  // routed to shard 1
    }
    return b.finalize(4, 2, 3);
}

}  // namespace

int main() {
    const uint32_t dim = 4;
    const uint32_t k = 3;
    EngineConfig cfg = make_config(2, dim, k);
    MiniDataset ds = make_policy_dataset();

    // ---- 1) integral lambda controller: spending ahead of schedule raises the price ----
    {
        PolicyParams params;
        params.lambda0 = 10.0;
        params.eta = 1.0;
        params.lambda_max = 100.0;
        PolicyRuntime rt = make_policy("bam_price", params);
        rt.budget_total_ms = 1000.0;
        rt.cum_spend_ms = 500.0;  // j=0 schedule is 100 => 400 ms ahead
        const double ahead = budget_lambda_update(rt, 0, 10);
        TCHECK_MSG(ahead > rt.lambda, "spending ahead of schedule must raise lambda");
        rt.cum_spend_ms = 0.0;  // j=9 schedule is 1000 => 1000 ms behind
        const double behind = budget_lambda_update(rt, 9, 10);
        TCHECK_MSG(behind < rt.lambda, "spending behind schedule must lower lambda");
        TCHECK(behind >= 0.0);
        rt.lambda = 99.5;
        rt.cum_spend_ms = 5000.0;
        TCHECK_NEAR(budget_lambda_update(rt, 0, 10), 100.0, 0.0);  // clipped at lambda_max
        rt.lambda = 0.5;
        rt.cum_spend_ms = 0.0;
        TCHECK_NEAR(budget_lambda_update(rt, 0, 10), 0.4, 1e-12);  // 0.5 - 0.1
        rt.lambda = 0.05;
        TCHECK_NEAR(budget_lambda_update(rt, 0, 10), 0.0, 0.0);  // clipped at 0
        rt.budget_total_ms = 0.0;  // uncalibrated => price unchanged
        rt.lambda = 7.0;
        TCHECK_NEAR(budget_lambda_update(rt, 0, 10), 7.0, 0.0);
    }

    // ---- 2) budget audit + rejection (tiny B => every maintenance action rejected) ----
    {
        PolicyParams params;
        params.action_mode = "refresh";
        PolicyRuntime rt = make_policy("bam_price", params);
        const double kappa = 1.0;
        // B = 0.05 ms：低于任何 refresh 的预测成本（(base_n+delta)*c_insert ≈ 0.27 ms），
        // 但高于 policy 自身 overhead（~0.005 ms），否则积分控制器会把 λ 推高、先抑制动作而不是触发预算拒绝。
        const double c_ref = 0.05;
        policy_prepare(&rt, "/tmp/qbam_trace_s00/trace.bin", ds.trace.n_windows, ds.trace.K, kappa,
                       c_ref);
        TCHECK(rt.budget_total_ms > 0.0);
        ReplayEngine engine = make_engine(ds, cfg);
        std::vector<std::string> lines;
        ReplayModeResult res = run_replay_mode(&engine, rt, &lines, nullptr);
        TCHECK(res.blocks.size() == 2);
        double sum_actual = 0.0;
        uint32_t rejections = 0;
        for (const BlockMetrics& b : res.blocks) {
            const double actual = b.maintenance_cpu_ms + b.policy_cpu_ms;
            TCHECK_NEAR(b.budget_actual_cpu_ms, actual, 1e-9);
            TCHECK_NEAR(b.budget_nominal_cpu_ms, rt.budget_total_ms / 2.0, 1e-9);
            TCHECK_NEAR(b.budget_unused_cpu_ms, std::max(0.0, b.budget_nominal_cpu_ms - actual), 1e-9);
            TCHECK(b.overshoot == (actual > b.budget_nominal_cpu_ms));
            sum_actual += actual;
            if (b.budget_rejected) ++rejections;
            TCHECK(b.action.is_noop() || b.budget_rejected == false);
        }
        // 决策点 j=0 的 Delta 恒为 0（trace 开始前没有 insert），策略按设计跳过而不算"拒绝"；
        // 因此可被拒绝的只有其后 delta>0 的块。这里显式断言该语义，而不是要求所有块都 rejected。
        TCHECK_MSG(rejections + 1 == res.blocks.size(),
                   "all blocks with non-empty Delta must be rejected under a tiny budget "
                   "(j=0 has delta=0 by construction and is skipped, not rejected)");
        TCHECK_MSG(res.blocks.empty() || res.blocks.front().action.is_noop(),
                   "j=0 must be noop (empty Delta)");
        TCHECK_MSG(sum_actual <= rt.budget_total_ms + 0.01, "soft cap exceeded by more than policy noise");
    }
    // uncalibrated budget (B=0) must not reject
    {
        PolicyParams params;
        params.action_mode = "refresh";
        PolicyRuntime rt = make_policy("past_qmass", params);
        policy_prepare(&rt, "/tmp/qbam_trace_s00/trace.bin", ds.trace.n_windows, ds.trace.K, 10.0,
                       0.0);
        TCHECK(rt.budget_total_ms == 0.0);
        ReplayEngine engine = make_engine(ds, cfg);
        std::vector<std::string> lines;
        ReplayModeResult res = run_replay_mode(&engine, rt, &lines, nullptr);
        for (const BlockMetrics& b : res.blocks) TCHECK(!b.budget_rejected);
    }

    // ---- 3) forecast fallback: missing file and missing (j, shard) both equal past_qmass ----
    {
        PolicyParams past_params;
        past_params.action_mode = "refresh";
        PolicyRuntime past_rt = make_policy("past_qmass", past_params);
        policy_prepare(&past_rt, "/tmp/qbam_trace_s00/trace.bin", ds.trace.n_windows, ds.trace.K,
                       10.0, 0.0);
        PolicyParams fc_params = past_params;
        fc_params.pred_path = "/nonexistent/predictions.csv";
        fc_params.pred_model = "lgbm";
        fc_params.trace_name = "qbam_trace_s00";
        PolicyRuntime fc_rt = make_policy("forecast_qmass", fc_params);
        policy_prepare(&fc_rt, "/tmp/qbam_trace_s00/trace.bin", ds.trace.n_windows, ds.trace.K,
                       10.0, 0.0);
        TCHECK_MSG(!fc_rt.forecast.loaded, "missing predictions file must fall back, not fail");
        ReplayEngine e1 = make_engine(ds, cfg);
        ReplayEngine e2 = make_engine(ds, cfg);
        std::vector<std::string> l1, l2;
        ReplayModeResult r1 = run_replay_mode(&e1, past_rt, &l1, nullptr);
        ReplayModeResult r2 = run_replay_mode(&e2, fc_rt, &l2, nullptr);
        TCHECK(r1.blocks.size() == r2.blocks.size());
        for (size_t b = 0; b < r1.blocks.size(); ++b) {
            TCHECK(r1.blocks[b].action.name() == r2.blocks[b].action.name());
            TCHECK(r1.blocks[b].action.shard == r2.blocks[b].action.shard);
        }
        // CSV with only j=0 rows: block 1 has no forecast -> fallback flag on the decision
        const char* csv_path = "qbam_test_forecast_tmp.csv";
        {
            std::ofstream f(csv_path);
            f << "trace,j,shard,model,q_pred\n";
            f << "qbam_trace_s00,0,0,lgbm,100.0\n";
            f << "qbam_trace_s00,0,1,lgbm,0.0\n";
        }
        PolicyRuntime fc2 = make_policy("forecast_qmass", fc_params);
        fc2.params.pred_path = csv_path;
        policy_prepare(&fc2, "/tmp/qbam_trace_s00/trace.bin", ds.trace.n_windows, ds.trace.K, 10.0,
                       0.0);
        TCHECK(fc2.forecast.loaded);
        TCHECK(fc2.forecast.has(0, 0));
        TCHECK_NEAR(fc2.forecast.get(0, 0), 100.0, 0.0);
        TCHECK(!fc2.forecast.has(1, 0));
        ReplayEngine e3 = make_engine(ds, cfg);
        for (uint32_t w = 0; w < 2; ++w) apply_window_crud(&e3, ds.trace, w);  // advance to block 1
        const uint64_t fallback_before = fc2.forecast_fallback_rows;
        const ActionDecision d1 = choose_action(&fc2, 1, &e3);
        // j=1 在 CSV 里没有任何行：策略必须对每个 shard 走 past-qmass 回退（计数增加）。
        // 该 fixture 未执行查询窗口（apply_window_crud 只做 CRUD），qmass 历史为 0，
        // 因而不存在正收益候选、合法地返回 noop——不能要求 d1.action 非空。
        TCHECK_MSG(fc2.forecast_fallback_rows > fallback_before,
                   "missing (j,shard) must fall back to past qmass");
        TCHECK(d1.forecast_fallback || d1.action.is_noop());
        std::remove(csv_path);
    }

    // ---- 4) action_mode legality: auto never maintains a shard with delta_n == 0 ----
    {
        for (const char* mode : {"refresh", "merge", "auto"}) {
            PolicyParams params;
            params.action_mode = mode;
            PolicyRuntime rt = make_policy("past_qmass", params);
            policy_prepare(&rt, "/tmp/qbam_trace_s00/trace.bin", ds.trace.n_windows, ds.trace.K,
                           10.0, 0.0);
            ReplayEngine engine = make_engine(ds, cfg);
            std::vector<std::string> lines;
            ReplayModeResult res = run_replay_mode(&engine, rt, &lines, nullptr);
            uint32_t actions = 0;
            for (const BlockMetrics& b : res.blocks) {
                if (b.action.is_noop()) continue;
                ++actions;
                TCHECK_MSG(b.delta_chosen > 0, "delta_n == 0 shard must never be selected");
                TCHECK(b.action.shard >= 0);
                if (std::string(mode) == "refresh") TCHECK(b.action.is_refresh());
                if (std::string(mode) == "merge") TCHECK(b.action.is_merge());
                if (std::string(mode) == "auto") {
                    TCHECK(b.action.is_refresh() || b.action.is_merge());
                }
            }
            TCHECK(actions > 0);
        }
    }

    // ---- 5) JSONL decision fields + run-row policy_params summary ----
    {
        PolicyParams params;
        params.action_mode = "auto";
        params.lambda0 = 3.0;
        params.eta = 0.5;
        PolicyRuntime rt = make_policy("bam_price", params);
        policy_prepare(&rt, "/tmp/qbam_trace_s00/trace.bin", ds.trace.n_windows, ds.trace.K, 10.0,
                       0.0);
        ReplayEngine engine = make_engine(ds, cfg);
        std::vector<std::string> lines;
        ReplayModeResult res = run_replay_mode(&engine, rt, &lines, nullptr);
        TCHECK(res.blocks.size() == 2);
        size_t blocks = 0;
        for (const std::string& line : lines) {
            if (line.find("\"type\":\"block\"") == std::string::npos) continue;
            ++blocks;
            for (const char* key : {"\"score_chosen\":", "\"cost_pred_ms\":", "\"lambda\":",
                                    "\"delta_chosen\":"}) {
                TCHECK_MSG(line.find(key) != std::string::npos,
                           std::string("block row missing ") + key);
            }
        }
        TCHECK(blocks == res.blocks.size());
        RunManifest manifest;
        manifest.policy_params_json = "{\"action_mode\":\"auto\"}";
        const std::string run_line = json_run_record(manifest);
        TCHECK(run_line.find("\"policy_params\":{\"action_mode\":\"auto\"}") != std::string::npos);
    }

    std::printf("test_budget_policies OK (lambda controller, budget audit, forecast fallback, "
                "action_mode, JSONL fields)\n");
    return 0;
}
