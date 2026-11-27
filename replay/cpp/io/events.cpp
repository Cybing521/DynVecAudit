// Q-BAM M1 engine -- binary contract loaders (engine_contract.md §1-§5).
#include "io/events.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "common/logging.h"

namespace qbam {

namespace {

constexpr char kPoolMagic[8] = {'Q', 'B', 'A', 'M', 'P', 'O', 'O', 'L'};
constexpr char kInitMagic[8] = {'Q', 'B', 'A', 'M', 'I', 'N', 'I', 'T'};
constexpr char kTraceMagic[8] = {'Q', 'B', 'A', 'M', 'T', 'R', 'A', 'C'};
constexpr char kGtMagic[8] = {'Q', 'B', 'A', 'M', 'G', 'T', '\0', '\0'};

constexpr uint32_t kContractVersion = 1;

void check_version(uint32_t v, const char* what) {
    if (v != kContractVersion) {
        fail(std::string("unsupported ") + what + " version: " + std::to_string(v) +
             " (contract expects 1)");
    }
}

}  // namespace

PoolData load_pool(const std::string& path) {
    std::vector<char> raw = read_file_bytes(path);
    ByteReader rd(raw.data(), raw.size());
    rd.expect_magic(kPoolMagic, sizeof(kPoolMagic));
    check_version(rd.u32(), "pool.bin");
    PoolData pool;
    pool.dim = rd.u32();
    pool.nvec = rd.u32();
    QBAM_CHECK(pool.dim > 0, "pool.bin: dim must be > 0");
    const uint64_t want = static_cast<uint64_t>(pool.nvec) * pool.dim;
    const uint64_t have = rd.remaining() / sizeof(float);
    if (have != want) {
        fail("pool.bin: payload mismatch: header says " + std::to_string(want) + " floats but file has " +
             std::to_string(have));
    }
    rd.f32_array(&pool.data, static_cast<size_t>(want));
    return pool;
}

InitData load_init(const std::string& path) {
    std::vector<char> raw = read_file_bytes(path);
    ByteReader rd(raw.data(), raw.size());
    rd.expect_magic(kInitMagic, sizeof(kInitMagic));
    check_version(rd.u32(), "init.bin");
    InitData init;
    init.dim = rd.u32();
    init.n = rd.u32();
    init.R = rd.u32();
    const uint64_t want_bytes = static_cast<uint64_t>(init.n) * 16u;
    if (rd.remaining() != want_bytes) {
        fail("init.bin: payload mismatch: header says " + std::to_string(init.n) + " entries (" +
             std::to_string(want_bytes) + " bytes) but file has " + std::to_string(rd.remaining()));
    }
    init.entries.resize(init.n);
    for (uint32_t i = 0; i < init.n; ++i) {
        InitEntry& e = init.entries[i];
        e.id = rd.u64();
        e.shard = rd.u32();
        e.vec_ref = rd.u32();
    }
    return init;
}

Centroids load_centroids(const std::string& path, uint32_t R, uint32_t dim) {
    std::vector<char> raw = read_file_bytes(path);
    const uint64_t want = static_cast<uint64_t>(R) * dim * sizeof(float);
    if (raw.size() != want) {
        fail("centroids.f32: expected " + std::to_string(want) + " bytes for R=" + std::to_string(R) +
             " dim=" + std::to_string(dim) + ", got " + std::to_string(raw.size()));
    }
    Centroids c;
    c.R = R;
    c.dim = dim;
    ByteReader rd(raw.data(), raw.size());
    rd.f32_array(&c.data, static_cast<size_t>(R) * dim);
    return c;
}

TraceData load_trace(const std::string& path) {
    std::vector<char> raw = read_file_bytes(path);
    ByteReader rd(raw.data(), raw.size());
    rd.expect_magic(kTraceMagic, sizeof(kTraceMagic));
    check_version(rd.u32(), "trace.bin");
    TraceData trace;
    trace.dim = rd.u32();
    trace.n_events = rd.u32();
    trace.n_windows = rd.u32();
    trace.K = rd.u32();
    QBAM_CHECK(trace.dim > 0, "trace.bin: dim must be > 0");
    QBAM_CHECK(trace.n_windows > 0, "trace.bin: n_windows must be > 0");
    QBAM_CHECK(trace.K > 0, "trace.bin: K must be > 0");
    const uint64_t want_bytes = static_cast<uint64_t>(trace.n_events) * 26u;
    if (rd.remaining() != want_bytes) {
        fail("trace.bin: payload mismatch: header says " + std::to_string(trace.n_events) +
             " events (" + std::to_string(want_bytes) + " bytes) but file has " +
             std::to_string(rd.remaining()));
    }
    trace.events.resize(trace.n_events);
    uint32_t prev_window = 0;
    for (uint32_t i = 0; i < trace.n_events; ++i) {
        TraceEvent& e = trace.events[i];
        e.window = rd.u32();
        uint8_t type = rd.u8();
        e.id = rd.u64();
        e.version = rd.u64();
        e.vec_ref = rd.u32();
        uint8_t flags = rd.u8();
        if (flags != 0) {
            fail("trace.bin: event " + std::to_string(i) + " has non-zero reserved flags");
        }
        if (type > 3) {
            fail("trace.bin: event " + std::to_string(i) + " has unknown type " + std::to_string(type));
        }
        e.type = static_cast<EventType>(type);
        if (e.window >= trace.n_windows) {
            fail("trace.bin: event " + std::to_string(i) + " window " + std::to_string(e.window) +
                 " out of range [0," + std::to_string(trace.n_windows) + ")");
        }
        if (i > 0 && e.window < prev_window) {
            fail("trace.bin: events are not grouped by non-decreasing window (event " +
                 std::to_string(i) + ")");
        }
        prev_window = e.window;
        if (e.type == EventType::Query) {
            if (e.id != 0 || e.version != 0) {
                fail("trace.bin: query event " + std::to_string(i) +
                     " must have id=0 and version=0 (contract §4)");
            }
        }
    }
    build_window_index(&trace);
    return trace;
}

GtData load_gt(const std::string& path) {
    std::vector<char> raw = read_file_bytes(path);
    ByteReader rd(raw.data(), raw.size());
    rd.expect_magic(kGtMagic, sizeof(kGtMagic));
    check_version(rd.u32(), "gt.bin");
    GtData gt;
    gt.k = rd.u32();
    gt.n_queries = rd.u32();
    gt.dim = rd.u32();
    QBAM_CHECK(gt.k > 0, "gt.bin: k must be > 0");
    const uint64_t want = static_cast<uint64_t>(gt.n_queries) * gt.k * 12u;
    if (rd.remaining() != want) {
        fail("gt.bin: payload mismatch: expected " + std::to_string(want) + " bytes, got " +
             std::to_string(rd.remaining()));
    }
    const size_t total = static_cast<size_t>(gt.n_queries) * gt.k;
    gt.ids.resize(total);
    gt.squared_l2.resize(total);
    for (size_t i = 0; i < total; ++i) {
        gt.ids[i] = rd.u64();
        gt.squared_l2[i] = rd.f32();
    }
    return gt;
}

void build_window_index(TraceData* trace) {
    const uint32_t nw = trace->n_windows;
    trace->window_begin.assign(nw + 1, trace->n_events);
    trace->window_query_count.assign(nw, 0);
    trace->window_query_start.assign(nw, 0);
    // window_begin[w] = first event index with window == w.
    uint32_t ev = 0;
    for (uint32_t w = 0; w < nw; ++w) {
        while (ev < trace->n_events && trace->events[ev].window < w) ++ev;
        trace->window_begin[w] = ev;
    }
    trace->window_begin[nw] = trace->n_events;
    uint32_t q = 0;
    for (uint32_t w = 0; w < nw; ++w) {
        trace->window_query_start[w] = q;
        for (uint32_t i = trace->window_begin[w]; i < trace->window_begin[w + 1]; ++i) {
            if (trace->events[i].type == EventType::Query) ++trace->window_query_count[w];
        }
        q += trace->window_query_count[w];
    }
    trace->total_queries = q;
}

void validate_dataset(const PoolData& pool, const InitData& init, const Centroids& centroids,
                      const TraceData& trace, const GtData& gt, uint32_t expected_dim,
                      uint32_t expected_R, uint32_t expected_k, bool has_gt) {
    if (pool.dim != expected_dim) {
        fail("pool.bin dim " + std::to_string(pool.dim) + " != config dim " +
             std::to_string(expected_dim));
    }
    if (init.dim != expected_dim) {
        fail("init.bin dim " + std::to_string(init.dim) + " != config dim " +
             std::to_string(expected_dim));
    }
    if (trace.dim != expected_dim) {
        fail("trace.bin dim " + std::to_string(trace.dim) + " != config dim " +
             std::to_string(expected_dim));
    }
    if (has_gt && gt.dim != expected_dim) {
        fail("gt.bin dim " + std::to_string(gt.dim) + " != config dim " +
             std::to_string(expected_dim));
    }
    if (init.R != expected_R) {
        fail("init.bin R " + std::to_string(init.R) + " != config R " + std::to_string(expected_R));
    }
    if (centroids.R != expected_R || centroids.dim != expected_dim) {
        fail("centroids.f32 shape mismatch with config R/dim");
    }
    if (has_gt && gt.k < expected_k) {
        fail("gt.bin k " + std::to_string(gt.k) + " < config k " + std::to_string(expected_k));
    }
    if (has_gt && gt.n_queries != trace.total_queries) {
        fail("gt.bin n_queries " + std::to_string(gt.n_queries) + " != trace query events " +
             std::to_string(trace.total_queries));
    }

    // Init entries: unique ids, in-range shard/vec_ref; every id globally alive at t=0.
    std::unordered_map<uint64_t, uint32_t> init_id_shard;
    init_id_shard.reserve(init.n * 2 + 1);
    for (uint32_t i = 0; i < init.n; ++i) {
        const InitEntry& e = init.entries[i];
        if (e.id == 0) fail("init.bin: id 0 is not allowed (contract §2)");
        if (e.shard >= expected_R) {
            fail("init.bin: shard out of range for id " + std::to_string(e.id));
        }
        if (e.vec_ref >= pool.nvec) {
            fail("init.bin: vec_ref out of range for id " + std::to_string(e.id));
        }
        auto ins = init_id_shard.emplace(e.id, e.shard);
        if (!ins.second) {
            fail("init.bin: duplicate id " + std::to_string(e.id));
        }
    }

    // Query vec refs + per-window CRUD-before-query + version monotonicity + alive semantics.
    std::unordered_map<uint64_t, uint64_t> last_version;
    last_version.reserve(trace.n_events * 2 + 1);
    for (uint32_t i = 0; i < trace.n_events; ++i) {
        const TraceEvent& e = trace.events[i];
        if (e.type != EventType::Erase && e.vec_ref >= pool.nvec) {
            // Erase uses 0xFFFFFFFF as the unused vec_ref (contract §4).
            fail("trace.bin: event " + std::to_string(i) + " vec_ref out of pool range");
        }
        if (e.type == EventType::Query) continue;
        last_version[e.id] = e.version;  // monotonicity checked in the second pass below
    }
    // Second pass: strict per-id version monotonicity in event order.
    {
        std::unordered_map<uint64_t, uint64_t> seen;
        seen.reserve(trace.n_events * 2 + 1);
        for (uint32_t i = 0; i < trace.n_events; ++i) {
            const TraceEvent& e = trace.events[i];
            if (e.type == EventType::Query) continue;
            auto it = seen.find(e.id);
            if (it != seen.end() && !(e.version > it->second)) {
                fail("trace.bin: per-id version not strictly increasing for id " +
                     std::to_string(e.id) + " at event " + std::to_string(i));
            }
            seen[e.id] = e.version;
        }
    }
    // Third pass: CRUD-before-query per window, plus alive semantics against init.
    {
        std::unordered_map<uint64_t, bool> alive;
        alive.reserve(init.n * 2 + 1);
        for (const InitEntry& e : init.entries) alive[e.id] = true;
        std::vector<char> query_seen(trace.n_windows, 0);
        for (uint32_t i = 0; i < trace.n_events; ++i) {
            const TraceEvent& e = trace.events[i];
            if (e.type == EventType::Query) {
                query_seen[e.window] = 1;
                continue;
            }
            if (query_seen[e.window]) {
                fail("trace.bin: contract §4 violated: CRUD event after a query in window " +
                     std::to_string(e.window) + " (event " + std::to_string(i) + ")");
            }
            if (e.type == EventType::Insert) {
                auto it = alive.find(e.id);
                if (it != alive.end() && it->second) {
                    fail("trace.bin: insert of already-live id " + std::to_string(e.id) +
                         " at event " + std::to_string(i));
                }
                alive[e.id] = true;
            } else if (e.type == EventType::Update) {
                auto it = alive.find(e.id);
                if (it == alive.end() || !it->second) {
                    fail("trace.bin: update of non-live id " + std::to_string(e.id) + " at event " +
                         std::to_string(i));
                }
            } else if (e.type == EventType::Erase) {
                auto it = alive.find(e.id);
                if (it == alive.end() || !it->second) {
                    fail("trace.bin: erase of non-live id " + std::to_string(e.id) + " at event " +
                         std::to_string(i));
                }
                alive[e.id] = false;
            }
        }
        uint64_t n_alive = 0;
        for (const auto& kv : alive) {
            if (kv.second) ++n_alive;
        }
        QBAM_LOG("[validate] init=%u events=%u windows=%u K=%u queries=%u final_live=%llu", init.n,
                 trace.n_events, trace.n_windows, trace.K, trace.total_queries,
                 static_cast<unsigned long long>(n_alive));
    }
    // GT sanity: ids alive-checked only structurally here (live state is time dependent).
    if (has_gt) {
        for (size_t i = 0; i < gt.ids.size(); ++i) {
            if (gt.ids[i] == 0) fail("gt.bin: id 0 in ground truth is not allowed");
            if (!(gt.squared_l2[i] >= 0.0f) || !std::isfinite(gt.squared_l2[i])) {
                fail("gt.bin: non-finite or negative squared L2 distance");
            }
        }
    }
}

}  // namespace qbam
