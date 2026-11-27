// Q-BAM M1 engine -- CLI entry point (engine_contract.md §6).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "common/json.h"
#include "common/logging.h"
#include "common/output.h"
#include "common/util.h"
#include "eval/cpu_meter.h"
#include "eval/exact_gt_check.h"
#include "io/events.h"
#include "replay/cf_runner.h"
#include "replay/replay_engine.h"

#ifndef QBAM_GIT_SHA
#define QBAM_GIT_SHA "nogit"
#endif
#ifndef QBAM_BUILD_DESC
#define QBAM_BUILD_DESC "Release (flags unknown)"
#endif

namespace qbam {
namespace {

struct CliOptions {
    std::string config_path;
    std::string out_path;         // empty => stdout
    std::string events_out_path;  // empty => no per-query records
    bool validate_only = false;   // load + contract validation (+ gt check) and exit
    uint32_t verify_gt = 0;       // verify first N gt.bin rows against brute force
};

struct AppConfig {
    std::string pool_path;
    std::string init_path;
    std::string centroids_path;
    std::string trace_path;
    std::string gt_path;
    EngineConfig engine;
    std::string mode;
    std::string policy;
    PolicyParams policy_params;
    CfParams cf;
    bool has_gt = true;
    std::string policy_params_json = "{}";
    std::string config_text;
};

void print_usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s --config <config.json> [--out <out.jsonl>] [--events-out <events.jsonl>]\n"
                 "          [--validate-only] [--verify-gt N]\n",
                 argv0);
}

CliOptions parse_args(int argc, char** argv) {
    CliOptions o;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need_next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) fail(std::string("missing value for ") + name);
            return std::string(argv[++i]);
        };
        if (arg == "--config") {
            o.config_path = need_next("--config");
        } else if (arg == "--out") {
            o.out_path = need_next("--out");
        } else if (arg == "--events-out") {
            o.events_out_path = need_next("--events-out");
        } else if (arg == "--validate-only") {
            o.validate_only = true;
        } else if (arg == "--verify-gt") {
            const std::string value = need_next("--verify-gt");
            o.verify_gt = static_cast<uint32_t>(std::strtoul(value.c_str(), nullptr, 10));
        } else if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            fail("unknown command line argument: " + arg);
        }
    }
    if (o.config_path.empty()) fail("--config <config.json> is required");
    return o;
}

std::string resolve_input_path(const std::string& config_dir, const std::string& value) {
    if (file_exists(value)) return value;
    const std::string joined = join_path(config_dir, value);
    if (file_exists(joined)) return joined;
    fail("input file not found: " + value + " (also tried " + joined + ")");
    return std::string();
}

int64_t require_int(const JsonValue& root, const char* key) {
    return root.at(key).as_int(std::numeric_limits<int64_t>::min());
}

std::string optional_string(const JsonValue& root, const char* key, const std::string& dflt) {
    const JsonValue* v = root.find(key);
    if (v == nullptr || v->is_null()) return dflt;
    return v->as_string();
}

PolicyParams parse_policy_params(const JsonValue* v) {
    PolicyParams p;
    if (v == nullptr || !v->is_object()) return p;
    if (const JsonValue* x = v->find("cadence_blocks")) p.cadence_blocks = static_cast<int>(x->as_int(p.cadence_blocks));
    if (const JsonValue* x = v->find("phase")) p.phase = static_cast<int>(x->as_int(p.phase));
    if (const JsonValue* x = v->find("shard_order")) p.shard_order = x->as_string();
    if (const JsonValue* x = v->find("qmass_window")) p.qmass_window = static_cast<uint32_t>(x->as_int(p.qmass_window));
    if (const JsonValue* x = v->find("spacing_blocks")) p.spacing_blocks = static_cast<int>(x->as_int(p.spacing_blocks));
    // E3-stage2 policy parameters
    if (const JsonValue* x = v->find("action_mode")) p.action_mode = x->as_string();
    if (const JsonValue* x = v->find("c_scan")) p.c_scan = x->as_double(p.c_scan);
    if (const JsonValue* x = v->find("c_insert")) p.c_insert = x->as_double(p.c_insert);
    if (const JsonValue* x = v->find("lambda0")) p.lambda0 = x->as_double(p.lambda0);
    if (const JsonValue* x = v->find("eta")) p.eta = x->as_double(p.eta);
    if (const JsonValue* x = v->find("lambda_max")) p.lambda_max = x->as_double(p.lambda_max);
    if (const JsonValue* x = v->find("pred_path")) p.pred_path = x->as_string();
    if (const JsonValue* x = v->find("pred_model")) p.pred_model = x->as_string();
    if (const JsonValue* x = v->find("trace_name")) p.trace_name = x->as_string();
    return p;
}

Action parse_action_string(const std::string& s) {
    if (s == "noop") return Action::noop();
    auto parse_index = [&](const std::string& body) -> int {
        if (body.empty()) fail("action \"" + s + "\" is missing a shard index");
        int value = 0;
        for (char c : body) {
            if (c < '0' || c > '9') fail("action \"" + s + "\" has a non-numeric shard index");
            value = value * 10 + (c - '0');
        }
        return value;
    };
    if (starts_with(s, "refresh:")) {
        const std::string spec = s.substr(8);
        const std::vector<std::string> parts = split(spec, '+');
        if (parts.size() == 1) return Action::refresh(parse_index(parts[0]));
        if (parts.size() == 2) {
            // Joint action for the C8 additivity diagnostic: refresh(r1) then refresh(r2).
            return Action::refresh_pair(parse_index(parts[0]), parse_index(parts[1]));
        }
        fail("action \"" + s + "\": only refresh:<r> or refresh:<r1>+<r2> are supported");
    }
    if (starts_with(s, "merge:")) return Action::merge(parse_index(s.substr(6)));
    fail("unknown cf action string: \"" + s +
         "\" (expected noop|refresh:<r>|refresh:<r1>+<r2>|merge:<r>)");
    return Action::noop();
}

AppConfig load_app_config(const std::string& path) {
    AppConfig app;
    app.config_text = read_file_text(path);
    const JsonValue root = JsonValue::parse(app.config_text);
    if (!root.is_object()) fail("config file must contain a JSON object: " + path);
    const std::string cfg_dir = dirname_of(path);

    EngineConfig& e = app.engine;
    app.pool_path = resolve_input_path(cfg_dir, root.at("pool").as_string());
    app.init_path = resolve_input_path(cfg_dir, root.at("init").as_string());
    app.centroids_path = resolve_input_path(cfg_dir, root.at("centroids").as_string());
    app.trace_path = resolve_input_path(cfg_dir, root.at("trace").as_string());
    const JsonValue* gt_value = root.find("gt");
    app.has_gt = gt_value != nullptr && !gt_value->is_null();
    if (app.has_gt) {
        app.gt_path = resolve_input_path(cfg_dir, gt_value->as_string());
    }

    e.R = static_cast<uint32_t>(require_int(root, "R"));
    e.dim = static_cast<uint32_t>(require_int(root, "dim"));
    e.k = static_cast<uint32_t>(require_int(root, "k"));
    e.M = static_cast<size_t>(require_int(root, "M"));
    e.ef_construction = static_cast<size_t>(require_int(root, "efConstruction"));
    e.ef_search = static_cast<size_t>(require_int(root, "efSearch"));
    e.nprobe = static_cast<uint32_t>(require_int(root, "nprobe"));
    e.seed = static_cast<uint64_t>(require_int(root, "seed"));
    e.run_id = root.at("run_id").as_string();
    app.mode = root.at("mode").as_string();
    app.policy = optional_string(root, "policy", "noop");
    app.policy_params = parse_policy_params(root.find("policy_params"));
    if (const JsonValue* pp = root.find("policy_params")) app.policy_params_json = pp->dump();
    if (!app.policy_params.pred_path.empty()) {
        // Resolve predictions.csv relative to the config dir when it is not found as given;
        // a genuinely missing file is NOT fatal: policy_prepare() warns and falls back.
        if (!file_exists(app.policy_params.pred_path)) {
            const std::string joined = join_path(cfg_dir, app.policy_params.pred_path);
            if (file_exists(joined)) app.policy_params.pred_path = joined;
        }
    }
    e.qmass_window = app.policy_params.qmass_window;
    e.pin_cpu = -1;
    if (const JsonValue* x = root.find("pin_cpu")) e.pin_cpu = static_cast<int>(x->as_int(-1));
    e.r_min = 0.95;
    if (const JsonValue* x = root.find("r_min")) e.r_min = x->as_double(0.95);
    e.overfetch_start = 0;
    if (const JsonValue* x = root.find("overfetch_start")) {
        e.overfetch_start = static_cast<uint32_t>(x->as_int(0));
    }
    // Delta representation: "scan" (default counterfactual exact-scan) or "hnsw" (per-shard HNSW).
    e.delta_mode = optional_string(root, "delta_mode", "scan");
    e.delta_ef_search = 0;
    if (const JsonValue* x = root.find("deltaEfSearch")) {
        e.delta_ef_search = static_cast<size_t>(x->as_int(0));
    }
    e.delta_initial_capacity = 4096;
    if (const JsonValue* x = root.find("deltaInitialCapacity")) {
        e.delta_initial_capacity = static_cast<size_t>(x->as_int(4096));
    }

    if (const JsonValue* budget = root.find("budget")) {
        if (budget->is_object()) {
            if (const JsonValue* x = budget->find("kappa_RE")) e.kappa_RE = x->as_double(0.0);
            if (const JsonValue* x = budget->find("c_ref_cpu_ms")) e.c_ref_cpu_ms = x->as_double(0.0);
        }
    }

    if (app.mode == "cf") {
        const JsonValue& cfv = root.at("cf");
        if (!cfv.is_object()) fail("config: \"cf\" must be an object when mode=cf");
        const std::string parent_policy = cfv.at("parent_policy").as_string();
        PolicyParams parent_params = parse_policy_params(cfv.find("parent_params"));
        policy_runtime_init(&app.cf.parent, parent_policy, parent_params);
        for (const JsonValue& v : cfv.at("sample_blocks").as_array()) {
            const int64_t b = v.as_int(-1);
            if (b < 0) fail("config: cf.sample_blocks entries must be non-negative integers");
            app.cf.sample_blocks.push_back(static_cast<uint32_t>(b));
        }
        for (const JsonValue& v : cfv.at("actions").as_array()) {
            app.cf.actions.push_back(parse_action_string(v.as_string()));
        }
        if (const JsonValue* pairs = cfv.find("pair_actions")) {
            for (const JsonValue& v : pairs->as_array()) {
                const std::vector<JsonValue>& items = v.as_array();
                if (items.size() != 2) fail("config: cf.pair_actions entries must be [r, s]");
                const int r = static_cast<int>(items[0].as_int(-1));
                const int s = static_cast<int>(items[1].as_int(-1));
                if (r < 0 || s < 0) fail("config: cf.pair_actions entries must be non-negative");
                app.cf.pair_actions.emplace_back(r, s);
            }
        }
    }
    return app;
}

void validate_config(const AppConfig& app) {
    const EngineConfig& e = app.engine;
    QBAM_CHECK(e.R > 0, "config: R must be > 0");
    QBAM_CHECK(e.dim > 0, "config: dim must be > 0");
    QBAM_CHECK(e.k > 0, "config: k must be > 0");
    QBAM_CHECK(e.nprobe >= 1 && e.nprobe <= e.R, "config: nprobe must be in [1, R]");
    QBAM_CHECK(e.M >= 2, "config: M must be >= 2");
    QBAM_CHECK(e.ef_construction >= e.M, "config: efConstruction must be >= M");
    QBAM_CHECK(e.ef_search >= 1, "config: efSearch must be >= 1");
    QBAM_CHECK(e.delta_mode == "scan" || e.delta_mode == "hnsw",
               "config: delta_mode must be \"scan\" or \"hnsw\"");
    QBAM_CHECK(e.delta_ef_search == 0 || e.delta_ef_search >= 1,
               "config: deltaEfSearch must be >= 1 (omit or 0 to reuse efSearch)");
    QBAM_CHECK(e.delta_initial_capacity >= 1, "config: deltaInitialCapacity must be >= 1");
    if (e.delta_mode == "hnsw") {
        QBAM_CHECK(app.mode != "cf",
                   "config: delta_mode=hnsw is not supported in cf mode (mutable "
                   "graphs cannot be shared by counterfactual branches)");
    }
    QBAM_CHECK(e.r_min > 0.0 && e.r_min <= 1.0, "config: r_min must be in (0, 1]");
    QBAM_CHECK(e.kappa_RE >= 0.0, "config: budget.kappa_RE must be >= 0");
    QBAM_CHECK(e.c_ref_cpu_ms >= 0.0, "config: budget.c_ref_cpu_ms must be >= 0");
    QBAM_CHECK(app.mode == "replay" || app.mode == "cf",
               "config: mode must be \"replay\" or \"cf\"");
    const std::string policy = app.mode == "replay" ? app.policy : app.cf.parent.policy;
    QBAM_CHECK(policy == "noop" || policy == "fixed" || policy == "query_delta" ||
                   policy == "past_qmass" || policy == "forecast_qmass" || policy == "bam_price",
               "config: unsupported policy \"" + policy + "\"");
    if (policy == "past_qmass" || policy == "forecast_qmass" || policy == "bam_price") {
        const std::string& mode = app.mode == "replay" ? app.policy_params.action_mode
                                                       : app.cf.parent.params.action_mode;
        QBAM_CHECK(mode == "refresh" || mode == "merge" || mode == "auto",
                   "config: policy_params.action_mode must be refresh|merge|auto");
    }
    if (app.policy_params.c_insert <= 0.0) fail("config: policy_params.c_insert must be > 0");
    if (app.policy_params.c_scan < 0.0) fail("config: policy_params.c_scan must be >= 0");
    if (app.mode == "cf") {
        QBAM_CHECK(!app.cf.sample_blocks.empty(), "config: cf.sample_blocks is empty");
        QBAM_CHECK(!app.cf.actions.empty(), "config: cf.actions is empty");
        for (const Action& a : app.cf.actions) {
            if (a.is_refresh() || a.is_merge()) {
                QBAM_CHECK(a.shard >= 0 && static_cast<uint32_t>(a.shard) < e.R,
                           "config: cf action shard out of range");
            }
        }
        for (const auto& p : app.cf.pair_actions) {
            QBAM_CHECK(p.first >= 0 && static_cast<uint32_t>(p.first) < e.R,
                       "config: cf pair_actions shard out of range");
            QBAM_CHECK(p.second >= 0 && static_cast<uint32_t>(p.second) < e.R,
                       "config: cf pair_actions shard out of range");
        }
    }
}

int run_pipeline(const CliOptions& opts) {
    const uint64_t run_start_ns = now_ns();
    AppConfig app = load_app_config(opts.config_path);
    validate_config(app);
    if (!opts.out_path.empty() && file_exists(opts.out_path)) {
        fail("output file already exists (append-only rule): " + opts.out_path);
    }
    if (!opts.events_out_path.empty() && file_exists(opts.events_out_path)) {
        fail("events output file already exists (append-only rule): " + opts.events_out_path);
    }
    if (app.engine.pin_cpu >= 0) {
        pin_current_thread_to_cpu(app.engine.pin_cpu);
    }

    QBAM_LOG("[main] loading dataset ...");
    PoolData pool = load_pool(app.pool_path);
    InitData init = load_init(app.init_path);
    Centroids centroids = load_centroids(app.centroids_path, app.engine.R, app.engine.dim);
    TraceData trace = load_trace(app.trace_path);
    GtData gt;
    if (app.has_gt) {
        gt = load_gt(app.gt_path);
    } else {
        QBAM_LOG("[main] WARNING: no gt.bin configured (\"gt\": null/absent): recall/route metrics "
                 "will be 0. This is a logic-only smoke mode.");
        gt.k = app.engine.k;
        gt.n_queries = 0;
        gt.dim = app.engine.dim;
    }
    validate_dataset(pool, init, centroids, trace, gt, app.engine.dim, app.engine.R, app.engine.k,
                     app.has_gt);
    if (opts.verify_gt > 0) {
        const uint32_t want = std::min<uint32_t>(opts.verify_gt, gt.n_queries);
        QBAM_LOG("[main] verifying first %u gt.bin rows against brute force ...", want);
        const GtVerifyResult vr = verify_gt_sample(pool, init, centroids, trace, gt, app.engine.k, want);
        QBAM_LOG("[verify-gt] checked=%u mismatches=%u", vr.checked, vr.mismatches);
        for (const std::string& e : vr.errors) QBAM_LOG("[verify-gt] %s", e.c_str());
        if (vr.mismatches != 0) {
            fail("gt.bin verification failed (" + std::to_string(vr.mismatches) + " mismatches)");
        }
    }
    if (opts.validate_only) {
        QBAM_LOG("[main] validate-only: dataset conforms to engine_contract.md (no replay)");
        return 0;
    }

    QBAM_LOG("[main] hashing dataset files ...");
    const std::string config_sha = sha256_hex(app.config_text.data(), app.config_text.size());
    const std::string dataset_sha = sha256_file_hex(app.pool_path);
    const std::string trace_sha = sha256_file_hex(app.trace_path);
    const std::string gt_sha = app.has_gt ? sha256_file_hex(app.gt_path) : std::string("none");

    ReplayEngine engine;
    engine.init(app.engine, &pool, &trace, &gt, init, centroids);

    VectorLineSink metric_lines;
    FileLineSink event_lines;
    if (!opts.events_out_path.empty()) {
        event_lines.open_exclusive(opts.events_out_path);
    }
    LineSink* query_sink = opts.events_out_path.empty() ? nullptr : &event_lines;

    if (app.mode == "replay") {
        PolicyRuntime rt;
        policy_runtime_init(&rt, app.policy, app.policy_params);
        policy_prepare(&rt, app.trace_path, trace.n_windows, trace.K, app.engine.kappa_RE,
                       app.engine.c_ref_cpu_ms);
        ReplayModeResult res = run_replay_mode(&engine, rt, &metric_lines.lines, query_sink);
        QBAM_LOG("[main] replay mode: policy=%s blocks=%zu windows=%zu", app.policy.c_str(),
                 res.blocks.size(), res.windows.size());
    } else {
        policy_prepare(&app.cf.parent, app.trace_path, trace.n_windows, trace.K,
                       app.engine.kappa_RE, app.engine.c_ref_cpu_ms);
        CfModeResult res = run_cf_mode(&engine, app.cf, &metric_lines.lines);
        QBAM_LOG("[main] cf mode: parent_policy=%s branches=%zu parent_blocks=%zu",
                 app.cf.parent.policy.c_str(), res.branches.size(), res.parent_blocks.size());
    }
    if (event_lines.is_open()) event_lines.close();

    RunManifest manifest;
    manifest.run_id = app.engine.run_id;
    manifest.mode = app.mode;
    manifest.policy = app.mode == "replay" ? app.policy : app.cf.parent.policy;
    manifest.git_sha = QBAM_GIT_SHA;
    manifest.config_sha256 = config_sha;
    manifest.dataset_sha256 = dataset_sha;
    manifest.trace_sha256 = trace_sha;
    manifest.gt_sha256 = gt_sha;
    manifest.R = app.engine.R;
    manifest.dim = app.engine.dim;
    manifest.k = app.engine.k;
    manifest.M = app.engine.M;
    manifest.ef_construction = app.engine.ef_construction;
    manifest.ef_search = app.engine.ef_search;
    manifest.delta_mode = app.engine.delta_mode;
    manifest.delta_ef_search = app.engine.delta_ef_search > 0 ? app.engine.delta_ef_search
                                                              : app.engine.ef_search;
    manifest.nprobe = app.engine.nprobe;
    manifest.seed = app.engine.seed;
    manifest.threads = 1;
    manifest.pin_cpu = app.engine.pin_cpu;
    manifest.n_events = trace.n_events;
    manifest.n_windows = trace.n_windows;
    manifest.K = trace.K;
    manifest.J = engine.J();
    manifest.wall_s = static_cast<double>(now_ns() - run_start_ns) / 1e9;
    manifest.platform = platform_string();
    manifest.build = QBAM_BUILD_DESC;
    manifest.policy_params_json = app.policy_params_json;
    manifest.return_code = 0;

    FileLineSink out;
    out.open_exclusive(opts.out_path);
    out.write_line(json_run_record(manifest));
    for (const std::string& line : metric_lines.lines) out.write_line(line);
    out.close();

    QBAM_LOG("[main] done: run_id=%s lines=%zu wall_s=%.3f", manifest.run_id.c_str(),
             metric_lines.lines.size() + 1, manifest.wall_s);
    return 0;
}

}  // namespace
}  // namespace qbam

int main(int argc, char** argv) {
    try {
        qbam::CliOptions opts = qbam::parse_args(argc, argv);
        return qbam::run_pipeline(opts);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FATAL: %s\n", e.what());
        return 1;
    }
}
