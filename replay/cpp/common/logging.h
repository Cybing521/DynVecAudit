// Q-BAM M1 engine -- tiny logging / assertion helpers.
#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>

namespace qbam {

[[noreturn]] inline void fail(const std::string& msg) {
    throw std::runtime_error(msg);
}

inline void check_impl(bool ok, const char* file, int line, const std::string& msg) {
    if (!ok) {
        fail(std::string(file) + ":" + std::to_string(line) + ": CHECK failed: " + msg);
    }
}

}  // namespace qbam

// QBAM_CHECK(cond, "message") -- throws std::runtime_error with file:line.
#define QBAM_CHECK(cond, msg) \
    ::qbam::check_impl(static_cast<bool>(cond), __FILE__, __LINE__, (msg))

// QBAM_LOG("fmt", args...) -- diagnostics to stderr (never to the JSONL stream).
#define QBAM_LOG(...)                         \
    do {                                      \
        std::fprintf(stderr, __VA_ARGS__);    \
        std::fputc('\n', stderr);             \
        std::fflush(stderr);                  \
    } while (0)
