// Q-BAM M1 engine -- byte/file/hash/string utilities (C++17 only, no deps).
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace qbam {

// ---------------------------------------------------------------- file I/O
std::vector<char> read_file_bytes(const std::string& path);
std::string read_file_text(const std::string& path);
bool file_exists(const std::string& path);
uint64_t file_size_bytes(const std::string& path);

// ---------------------------------------------------------------- paths / strings
std::string dirname_of(const std::string& path);
std::string join_path(const std::string& a, const std::string& b);
bool is_absolute_path(const std::string& path);
std::vector<std::string> split(const std::string& s, char sep);
std::string trim(const std::string& s);
bool ends_with(const std::string& s, const std::string& suffix);
bool starts_with(const std::string& s, const std::string& prefix);

// JSON string escaping (no surrounding quotes).
std::string escape_json_string(const std::string& s);
// Fixed 6-decimal formatting, deterministic across runs.
std::string fmt6(double v);
std::string fmt3(double v);

// ---------------------------------------------------------------- SHA-256
class Sha256 {
public:
    Sha256();
    void update(const void* data, size_t len);
    // Finalizes and returns lowercase hex. Do not call update() afterwards.
    std::string hexdigest();

private:
    void transform(const uint8_t* block);
    uint32_t h_[8];
    uint64_t bitlen_;
    uint8_t buf_[64];
    size_t buflen_;
};

std::string sha256_hex(const void* data, size_t len);
std::string sha256_file_hex(const std::string& path);

// ---------------------------------------------------------------- byte reader
// Bounds-checked little-endian reader over a byte buffer.
class ByteReader {
public:
    ByteReader(const void* data, size_t size);

    size_t pos() const { return static_cast<size_t>(p_ - begin_); }
    size_t remaining() const { return static_cast<size_t>(end_ - p_); }
    const uint8_t* ptr() const { return p_; }
    void skip(size_t n);

    uint8_t u8();
    uint32_t u32();
    uint64_t u64();
    float f32();
    // Reads a little-endian f32 array of `count` elements into out (resized).
    void f32_array(std::vector<float>* out, size_t count);
    void expect_magic(const char* magic, size_t n);

private:
    void need(size_t n) const;
    const uint8_t* begin_;
    const uint8_t* p_;
    const uint8_t* end_;
};

// ---------------------------------------------------------------- platform
std::string platform_string();

}  // namespace qbam
