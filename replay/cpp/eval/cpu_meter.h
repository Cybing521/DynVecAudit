// Q-BAM M1 engine -- CPU/wall timing separated (engine_contract.md §8.8).
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace qbam {

// Monotonic wall clock, nanoseconds.
uint64_t now_ns();
// CLOCK_THREAD_CPUTIME_ID in milliseconds (contract allows process clock; thread clock is
// the strictest: single-thread engine, no background threads).
double thread_cpu_ms();
// CLOCK_THREAD_CPUTIME_ID in nanoseconds.
uint64_t thread_cpu_ns();

struct CpuWallTimer {
    void start() {
        wall_start_ns = now_ns();
        cpu_start_ns = thread_cpu_ns();
    }
    double cpu_ms() const { return static_cast<double>(thread_cpu_ns() - cpu_start_ns) / 1e6; }
    double wall_us() const { return static_cast<double>(now_ns() - wall_start_ns) / 1e3; }
    double wall_ms() const { return static_cast<double>(now_ns() - wall_start_ns) / 1e6; }

    uint64_t wall_start_ns = 0;
    uint64_t cpu_start_ns = 0;
};

struct Percentiles {
    double p50 = 0.0;
    double p95 = 0.0;
    double p99 = 0.0;
    double mean = 0.0;
    size_t count = 0;
};

// Nearest-rank percentile over a copy-free (mutating) sort of `values_us`.
Percentiles compute_percentiles(std::vector<double>* values_us);

// Pins the calling thread to `cpu` (sched_setaffinity). Throws on failure.
void pin_current_thread_to_cpu(int cpu);

}  // namespace qbam
