// Q-BAM M1 engine -- CPU/wall timing implementation.
#include "eval/cpu_meter.h"

#ifdef __linux__
#include <sched.h>
#endif
#include <time.h>

#include <algorithm>
#include <cerrno>
#include <cstring>

#include "common/logging.h"

namespace qbam {

uint64_t now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

uint64_t thread_cpu_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

double thread_cpu_ms() { return static_cast<double>(thread_cpu_ns()) / 1e6; }

namespace {
// Nearest-rank percentile: index = ceil(p/100 * n) - 1, clamped to [0, n-1].
double nearest_rank(const std::vector<double>& sorted, double p) {
    if (sorted.empty()) return 0.0;
    const size_t n = sorted.size();
    double raw = (p / 100.0) * static_cast<double>(n);
    size_t idx = static_cast<size_t>(raw);
    if (raw > static_cast<double>(idx)) idx += 1;  // ceil
    if (idx == 0) idx = 1;
    if (idx > n) idx = n;
    return sorted[idx - 1];
}
}  // namespace

Percentiles compute_percentiles(std::vector<double>* values_us) {
    Percentiles out;
    if (values_us == nullptr || values_us->empty()) return out;
    std::vector<double>& v = *values_us;
    std::sort(v.begin(), v.end());
    out.count = v.size();
    out.p50 = nearest_rank(v, 50.0);
    out.p95 = nearest_rank(v, 95.0);
    out.p99 = nearest_rank(v, 99.0);
    double sum = 0.0;
    for (double x : v) sum += x;
    out.mean = sum / static_cast<double>(v.size());
    return out;
}

void pin_current_thread_to_cpu(int cpu) {
    if (cpu < 0) return;
#ifdef __linux__
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(cpu_set_t), &set) != 0) {
        fail("sched_setaffinity(cpu=" + std::to_string(cpu) + ") failed: " + std::strerror(errno));
    }
#else
    QBAM_LOG("[cpu_meter] pin_cpu=%d ignored on non-Linux host", cpu);
#endif
}

}  // namespace qbam
