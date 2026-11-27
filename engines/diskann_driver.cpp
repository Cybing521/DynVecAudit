// qbam/external/drivers/diskann_driver.cpp
//
// Windowed driver for Microsoft DiskANN 0.7.0 (FreshDiskANN in-memory dynamic index).
// Purpose: replay Q-BAM's frozen window event stream (docs/engine_contract.md §4) against
// DiskANN so the v3 protocol can compare CPU/latency/recall on identical windows.
//
// Protocol mapping:
//   * base   = Q-BAM init.bin vectors (head of the trace), built with Q-BAM ids as DiskANN tags;
//   * per window: insert that window's insert events (tag = Q-BAM id), then answer that window's
//     query events (the trace guarantees CRUD before queries within a window);
//   * DiskANN has no separate maintenance action for an insert-only stream: the insert work IS
//     the maintenance cost, reported as insert_cpu_ms / maintenance_cpu_ms by the Python side;
//   * per-query CPU is CLOCK_THREAD_CPUTIME_ID, wall is steady_clock; the process is expected to
//     be pinned with taskset(1) and nice(19) by the caller.
//
// Outputs (all JSONL, one object per line):
//   --out          : {"type":"run",...} once, {"type":"window",...} per window
//   query_out files: {"window":w,"q":i,"ids":[k ids...],"wall_us":..,"cpu_us":..} per query
// The Python harness computes recall@k against gt.bin from the query_out files.
//
// Build: qbam/external/build_diskann_driver.sh

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include <index.h>
#include <index_factory.h>
#include <abstract_index.h>
#include <parameters.h>
#include <defaults.h>

namespace {

double now_wall_us()
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

double now_thread_cpu_us()
{
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

double now_proc_cpu_us()
{
    struct timespec ts;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

long vm_hwm_kb()
{
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
    {
        if (line.rfind("VmHWM:", 0) == 0)
        {
            std::istringstream is(line.substr(6));
            long kb = 0;
            is >> kb;
            return kb;
        }
    }
    return -1;
}

std::string json_escape(const std::string &s)
{
    std::string out;
    for (char c : s)
    {
        if (c == '"' || c == '\\') { out.push_back('\\'); out.push_back(c); }
        else if (c == '\n') out += "\\n";
        else out.push_back(c);
    }
    return out;
}

// ---- minimal flat-json object parser (values: string or number only) -----------------------
struct JsonObj
{
    std::map<std::string, std::string> kv;   // numbers stored as their literal text
    bool has(const std::string &k) const { return kv.count(k) > 0; }
    std::string str(const std::string &k, const std::string &def = "") const
    {
        auto it = kv.find(k);
        return it == kv.end() ? def : it->second;
    }
    long long num(const std::string &k, long long def = 0) const
    {
        auto it = kv.find(k);
        return it == kv.end() ? def : std::strtoll(it->second.c_str(), nullptr, 10);
    }
};

std::string trim(const std::string &s)
{
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

JsonObj parse_flat_json(const std::string &line)
{
    JsonObj o;
    size_t i = 0, n = line.size();
    while (i < n && line[i] != '{') ++i;
    ++i;
    while (i < n)
    {
        while (i < n && (line[i] == ' ' || line[i] == ',' || line[i] == '\t' || line[i] == '\n')) ++i;
        if (i >= n || line[i] == '}') break;
        if (line[i] != '"') { ++i; continue; }
        size_t ks = ++i;
        while (i < n && line[i] != '"') ++i;
        std::string key = line.substr(ks, i - ks);
        ++i;
        while (i < n && (line[i] == ' ' || line[i] == ':')) ++i;
        std::string val;
        if (i < n && line[i] == '"')
        {
            size_t vs = ++i;
            while (i < n && line[i] != '"') ++i;
            val = line.substr(vs, i - vs);
            ++i;
        }
        else
        {
            size_t vs = i;
            while (i < n && line[i] != ',' && line[i] != '}') ++i;
            val = trim(line.substr(vs, i - vs));
        }
        o.kv[key] = val;
    }
    return o;
}

struct PlanRow
{
    long long window = -1;
    std::string insert_file;
    std::string insert_tags;
    std::string query_file;
    std::string query_out;
    long long search_L = 0;   // 0 -> use global
};

struct Args
{
    std::string data_type = "float";
    std::string dist_fn = "l2";
    long long dim = 128;
    std::string base_file;
    std::string base_tags_file;
    std::string plan_file;
    std::string out_file;
    long long R = 32;
    long long Lbuild = 64;
    double alpha = 1.2;
    long long search_L = 64;
    long long K = 10;
    long long build_threads = 8;
    long long max_points = 0;   // 0 -> computed from plan
    long long frozen_pts = 1;
    std::string tag = "diskann";
    std::string run_id = "diskann_smoke";
};

void usage()
{
    std::cout <<
        "diskann_driver --base <base.bin> --base_tags <base_tags.u32> --plan <plan.jsonl> --out <out.jsonl>\n"
        "  [--dim 128] [--R 32] [--Lbuild 64] [--alpha 1.2] [--search_L 64] [--K 10]\n"
        "  [--build_threads 8] [--max_points N] [--run_id NAME] [--tag NAME]\n"
        "plan.jsonl line: {\"window\":w,\"insert_file\":f,\"insert_tags\":f,\"query_file\":f,\"query_out\":f,\"search_L\":L}\n";
}

// Load a DiskANN .bin (int32 npts, int32 dim, then row-major T) into an aligned buffer padded to
// ROUND_UP(dim,8) as DiskANN's build(data,...)/insert paths expect.
template <typename T>
T *load_bin_aligned(const std::string &path, size_t &npts, size_t &dim, size_t &aligned_dim)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) { std::cerr << "FATAL: cannot open " << path << std::endl; std::exit(2); }
    size_t fsize = (size_t)in.tellg();
    in.seekg(0, std::ios::beg);
    int32_t n32 = 0, d32 = 0;
    in.read((char *)&n32, 4);
    in.read((char *)&d32, 4);
    if (!in || n32 < 0 || d32 <= 0) { std::cerr << "FATAL: bad .bin header " << path << std::endl; std::exit(2); }
    npts = (size_t)n32;
    dim = (size_t)d32;
    aligned_dim = (dim + 7) / 8 * 8;
    size_t expect = 8 + npts * dim * sizeof(T);
    if (fsize != expect)
    {
        std::cerr << "FATAL: " << path << " size " << fsize << " != expected " << expect << std::endl;
        std::exit(2);
    }
    if (npts == 0)   // static (u=0) traces have windows with zero inserts
        return nullptr;
    T *buf = (T *)aligned_alloc(64, ((npts * aligned_dim * sizeof(T)) + 63) / 64 * 64);
    if (!buf) { std::cerr << "FATAL: aligned_alloc failed" << std::endl; std::exit(2); }
    std::vector<T> row(dim);
    for (size_t i = 0; i < npts; ++i)
    {
        in.read((char *)row.data(), dim * sizeof(T));
        std::memcpy(buf + i * aligned_dim, row.data(), dim * sizeof(T));
        std::memset(buf + i * aligned_dim + dim, 0, (aligned_dim - dim) * sizeof(T));
    }
    return buf;
}

std::vector<uint32_t> load_tags(const std::string &path, size_t expect_n)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) { std::cerr << "FATAL: cannot open tags " << path << std::endl; std::exit(2); }
    size_t fsize = (size_t)in.tellg();
    in.seekg(0, std::ios::beg);
    if (fsize != expect_n * 4)
    {
        std::cerr << "FATAL: tags file " << path << " has " << fsize << " bytes, expected " << expect_n * 4 << std::endl;
        std::exit(2);
    }
    std::vector<uint32_t> tags(expect_n);
    in.read((char *)tags.data(), fsize);
    return tags;
}

double percentile(std::vector<double> v, double p)
{
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    double idx = (p / 100.0) * (double)(v.size() - 1);
    size_t lo = (size_t)idx, hi = std::min(lo + 1, v.size() - 1);
    double frac = idx - (double)lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

} // namespace

int main(int argc, char **argv)
{
    Args a;
    for (int i = 1; i < argc; ++i)
    {
        std::string k = argv[i];
        auto next = [&](const char *name) -> std::string {
            if (i + 1 >= argc) { std::cerr << "FATAL: missing value for " << name << std::endl; std::exit(2); }
            return argv[++i];
        };
        if (k == "-h" || k == "--help") { usage(); return 0; }
        else if (k == "--base") a.base_file = next("--base");
        else if (k == "--base_tags") a.base_tags_file = next("--base_tags");
        else if (k == "--plan") a.plan_file = next("--plan");
        else if (k == "--out") a.out_file = next("--out");
        else if (k == "--dim") a.dim = std::atoll(next("--dim").c_str());
        else if (k == "--R") a.R = std::atoll(next("--R").c_str());
        else if (k == "--Lbuild") a.Lbuild = std::atoll(next("--Lbuild").c_str());
        else if (k == "--alpha") a.alpha = std::atof(next("--alpha").c_str());
        else if (k == "--search_L") a.search_L = std::atoll(next("--search_L").c_str());
        else if (k == "--K") a.K = std::atoll(next("--K").c_str());
        else if (k == "--build_threads") a.build_threads = std::atoll(next("--build_threads").c_str());
        else if (k == "--max_points") a.max_points = std::atoll(next("--max_points").c_str());
        else if (k == "--run_id") a.run_id = next("--run_id");
        else if (k == "--tag") a.tag = next("--tag");
        else { std::cerr << "FATAL: unknown arg " << k << std::endl; usage(); return 2; }
    }
    if (a.base_file.empty() || a.base_tags_file.empty() || a.plan_file.empty() || a.out_file.empty())
    {
        usage();
        return 2;
    }
    omp_set_num_threads(1);   // the harness measures single-thread CPU; callers may still use
                              // --build_threads for build-time parallelism via IndexWriteParameters.

    // ---- read plan ----
    std::vector<PlanRow> plan;
    {
        std::ifstream f(a.plan_file);
        if (!f) { std::cerr << "FATAL: cannot open plan " << a.plan_file << std::endl; return 2; }
        std::string line;
        while (std::getline(f, line))
        {
            line = trim(line);
            if (line.empty() || line[0] == '#') continue;
            JsonObj o = parse_flat_json(line);
            PlanRow r;
            r.window = o.num("window", -1);
            r.insert_file = o.str("insert_file");
            r.insert_tags = o.str("insert_tags");
            r.query_file = o.str("query_file");
            r.query_out = o.str("query_out");
            r.search_L = o.num("search_L", 0);
            plan.push_back(r);
        }
    }
    if (plan.empty()) { std::cerr << "FATAL: empty plan" << std::endl; return 2; }

    // ---- load base + tags ----
    size_t base_n = 0, dim = 0, aligned_dim = 0;
    float *base = load_bin_aligned<float>(a.base_file, base_n, dim, aligned_dim);
    if ((long long)dim != a.dim)
    {
        std::cerr << "FATAL: base dim " << dim << " != --dim " << a.dim << std::endl;
        return 2;
    }
    std::vector<uint32_t> base_tags = load_tags(a.base_tags_file, base_n);

    long long total_points = (long long)base_n;
    for (auto &r : plan)
    {
        // cheap metadata probe: read header only
        std::ifstream f(r.insert_file, std::ios::binary);
        int32_t n32 = 0, d32 = 0;
        f.read((char *)&n32, 4);
        f.read((char *)&d32, 4);
        total_points += n32;
    }
    long long max_points = a.max_points > 0 ? a.max_points : total_points + 16;

    // ---- create dynamic index (same IndexConfig as upstream test_insert_deletes_consolidate) ----
    diskann::IndexWriteParameters wp = diskann::IndexWriteParametersBuilder((uint32_t)a.Lbuild, (uint32_t)a.R)
                                            .with_max_occlusion_size(500)
                                            .with_alpha((float)a.alpha)
                                            .with_num_threads((uint32_t)a.build_threads)
                                            .with_filter_list_size((uint32_t)a.Lbuild)
                                            .build();
    auto search_params = diskann::IndexSearchParams((uint32_t)a.search_L, 1);
    auto cfg = diskann::IndexConfigBuilder()
                   .with_metric(diskann::L2)
                   .with_dimension(dim)
                   .with_max_points((size_t)max_points)
                   .is_dynamic_index(true)
                   .is_enable_tags(true)
                   .is_use_opq(false)
                   .is_filtered(false)
                   .with_num_pq_chunks(0)
                   .is_pq_dist_build(false)
                   .with_num_frozen_pts((size_t)a.frozen_pts)
                   .with_tag_type(diskann_type_to_name<uint32_t>())
                   .with_label_type(diskann_type_to_name<uint32_t>())
                   .with_data_type(diskann_type_to_name<float>())
                   .with_index_write_params(wp)
                   .with_index_search_params(search_params)
                   .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                   .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                   .build();
    diskann::IndexFactory factory(cfg);
    auto index = factory.create_instance();

    double t0 = now_proc_cpu_us(), w0 = now_wall_us();
    index->build(base, base_n, base_tags);
    double build_cpu_ms = (now_proc_cpu_us() - t0) / 1e3;
    double build_wall_ms = (now_wall_us() - w0) / 1e3;

    std::ofstream out(a.out_file, std::ios::app);
    if (!out) { std::cerr << "FATAL: cannot open out " << a.out_file << std::endl; return 2; }
    out << "{\"type\":\"run\",\"run_id\":\"" << json_escape(a.run_id) << "\",\"system\":\"" << json_escape(a.tag)
        << "\",\"data_type\":\"float\",\"dist_fn\":\"" << json_escape(a.dist_fn) << "\",\"dim\":" << dim
        << ",\"R\":" << a.R << ",\"Lbuild\":" << a.Lbuild << ",\"alpha\":" << a.alpha << ",\"search_L\":" << a.search_L
        << ",\"K\":" << a.K << ",\"build_threads\":" << a.build_threads << ",\"base_n\":" << base_n
        << ",\"max_points\":" << max_points << ",\"n_windows\":" << plan.size() << ",\"build_cpu_ms\":"
        << build_cpu_ms << ",\"build_wall_ms\":" << build_wall_ms << ",\"threads\":1" << ",\"rss_hwm_kb\":"
        << vm_hwm_kb() << "}" << std::endl;

    std::cout << "[diskann] build " << base_n << " pts: wall " << build_wall_ms << " ms, cpu " << build_cpu_ms
              << " ms; max_points " << max_points << std::endl;

    // AbstractIndex does not expose get_num_points(); track the live count ourselves (insert-only here).
    long long live_n = (long long)base_n;

    for (const auto &r : plan)
    {
        size_t ins_n = 0, ins_dim = 0, ins_aligned = 0;
        double ins_cpu_ms = 0, ins_wall_ms = 0;
        long long failed = 0;
        if (!r.insert_file.empty())
        {
            float *ins = load_bin_aligned<float>(r.insert_file, ins_n, ins_dim, ins_aligned);
            if (ins_dim != dim) { std::cerr << "FATAL: insert dim mismatch" << std::endl; return 2; }
            std::vector<uint32_t> ins_tags = load_tags(r.insert_tags, ins_n);
            double c0 = now_thread_cpu_us(), uw0 = now_wall_us();
            for (size_t i = 0; i < ins_n; ++i)
            {
                int rc = index->insert_point(ins + i * ins_aligned, ins_tags[i]);
                if (rc != 0) ++failed;
            }
            ins_cpu_ms = (now_thread_cpu_us() - c0) / 1e3;
            ins_wall_ms = (now_wall_us() - uw0) / 1e3;
            free(ins);
        }

        size_t qn = 0, qdim = 0, qaligned = 0;
        float *queries = nullptr;
        std::vector<double> wall_us, cpu_us;
        std::ofstream qout;
        if (!r.query_file.empty())
        {
            queries = load_bin_aligned<float>(r.query_file, qn, qdim, qaligned);
            if (qdim != dim) { std::cerr << "FATAL: query dim mismatch" << std::endl; return 2; }
            qout.open(r.query_out, std::ios::trunc);
            if (!qout) { std::cerr << "FATAL: cannot open query_out " << r.query_out << std::endl; return 2; }
        }
        uint32_t L = (uint32_t)(r.search_L > 0 ? r.search_L : a.search_L);
        double qcpu_total = 0.0;
        // Dynamic (tag-enabled) indexes return *internal locations* from search(); the Q-BAM ids we
        // built/inserted with are only available through search_with_tags(). res_vectors must be
        // K caller-allocated aligned_dim buffers; we do not use the reconstructed vectors.
        std::vector<float> resbuf((size_t)a.K * aligned_dim, 0.0f);
        std::vector<float *> resvec((size_t)a.K);
        for (long long j = 0; j < a.K; ++j) resvec[(size_t)j] = resbuf.data() + (size_t)j * aligned_dim;
        std::vector<uint32_t> tags_out((size_t)a.K, 0u);
        std::vector<float> dists((size_t)a.K, 0.0f);
        for (size_t qi = 0; qi < qn; ++qi)
        {
            double wq = now_wall_us(), cq = now_thread_cpu_us();
            size_t nres = index->search_with_tags(queries + qi * qaligned, (uint64_t)a.K, L, tags_out.data(),
                                                 dists.data(), resvec);
            double dt_w = now_wall_us() - wq, dt_c = now_thread_cpu_us() - cq;
            wall_us.push_back(dt_w);
            cpu_us.push_back(dt_c);
            qcpu_total += dt_c;
            qout << "{\"window\":" << r.window << ",\"q\":" << qi << ",\"n_res\":" << nres << ",\"ids\":[";
            for (long long j = 0; j < a.K; ++j) qout << (j ? "," : "") << tags_out[(size_t)j];
            qout << "]}"
                 << "\n";
        }
        if (queries) free(queries);
        double p50 = percentile(wall_us, 50), p95 = percentile(wall_us, 95), p99 = percentile(wall_us, 99);
        out << "{\"type\":\"window\",\"window\":" << r.window << ",\"n_insert\":" << ins_n << ",\"n_queries\":" << qn
            << ",\"insert_cpu_ms\":" << ins_cpu_ms << ",\"insert_wall_ms\":" << ins_wall_ms
            << ",\"query_cpu_ms\":" << (qcpu_total / 1e3) << ",\"p50_wall_us\":" << p50 << ",\"p95_wall_us\":" << p95
            << ",\"p99_wall_us\":" << p99 << ",\"search_L\":" << L << ",\"k\":" << a.K
            << ",\"failed_inserts\":" << failed << ",\"index_n\":" << (live_n += (long long)ins_n)
            << ",\"rss_hwm_kb\":" << vm_hwm_kb() << "}" << std::endl;
        std::cout << "[diskann] window " << r.window << ": ins " << ins_n << " (" << ins_wall_ms << " ms wall)"
                  << " q " << qn << " (cpu " << qcpu_total / 1e3 << " ms, p50 " << p50 << " us)" << std::endl;
    }
    out.close();
    free(base);
    return 0;
}
