// Q-BAM M1 engine -- utility implementation.
#include "common/util.h"

#include <sys/utsname.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <stdexcept>

#include "common/logging.h"

namespace qbam {

// ---------------------------------------------------------------- file I/O
bool file_exists(const std::string& path) {
    return ::access(path.c_str(), F_OK) == 0;
}

uint64_t file_size_bytes(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        fail("cannot open file: " + path + ": " + std::strerror(errno));
    }
    if (std::fseek(f, 0, SEEK_END) != 0) {
        std::fclose(f);
        fail("cannot seek file: " + path);
    }
    long sz = std::ftell(f);
    std::fclose(f);
    if (sz < 0) {
        fail("cannot determine size of file: " + path);
    }
    return static_cast<uint64_t>(sz);
}

std::vector<char> read_file_bytes(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        fail("cannot open file: " + path + ": " + std::strerror(errno));
    }
    if (std::fseek(f, 0, SEEK_END) != 0) {
        std::fclose(f);
        fail("cannot seek file: " + path);
    }
    long sz = std::ftell(f);
    if (sz < 0) {
        std::fclose(f);
        fail("cannot determine size of file: " + path);
    }
    std::rewind(f);
    std::vector<char> buf(static_cast<size_t>(sz));
    if (sz > 0) {
        size_t got = std::fread(buf.data(), 1, static_cast<size_t>(sz), f);
        if (got != static_cast<size_t>(sz)) {
            std::fclose(f);
            fail("short read on file: " + path);
        }
    }
    std::fclose(f);
    return buf;
}

std::string read_file_text(const std::string& path) {
    std::vector<char> bytes = read_file_bytes(path);
    return std::string(bytes.begin(), bytes.end());
}

// ---------------------------------------------------------------- paths / strings
std::string dirname_of(const std::string& path) {
    size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) return std::string(".");
    if (pos == 0) return std::string("/");
    return path.substr(0, pos);
}

bool is_absolute_path(const std::string& path) {
    return !path.empty() && path[0] == '/';
}

std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (is_absolute_path(b)) return b;
    if (a.back() == '/') return a + b;
    return a + "/" + b;
}

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    size_t start = 0;
    while (true) {
        size_t pos = s.find(sep, start);
        if (pos == std::string::npos) {
            out.push_back(s.substr(start));
            break;
        }
        out.push_back(s.substr(start, pos - start));
        start = pos + 1;
    }
    return out;
}

std::string trim(const std::string& s) {
    size_t b = 0;
    size_t e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n')) ++b;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r' || s[e - 1] == '\n')) --e;
    return s.substr(b, e - b);
}

bool ends_with(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool starts_with(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

std::string escape_json_string(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(c));
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

std::string fmt6(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.6f", v);
    return std::string(buf);
}

std::string fmt3(double v) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.3f", v);
    return std::string(buf);
}

// ---------------------------------------------------------------- SHA-256 (FIPS 180-4)
namespace {
inline uint32_t rotr(uint32_t x, uint32_t n) { return (x >> n) | (x << (32 - n)); }
}  // namespace

Sha256::Sha256() : bitlen_(0), buflen_(0) {
    h_[0] = 0x6a09e667u;
    h_[1] = 0xbb67ae85u;
    h_[2] = 0x3c6ef372u;
    h_[3] = 0xa54ff53au;
    h_[4] = 0x510e527fu;
    h_[5] = 0x9b05688cu;
    h_[6] = 0x1f83d9abu;
    h_[7] = 0x5be0cd19u;
}

void Sha256::update(const void* data, size_t len) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    bitlen_ += static_cast<uint64_t>(len) * 8u;
    while (len > 0) {
        size_t take = 64 - buflen_;
        if (take > len) take = len;
        std::memcpy(buf_ + buflen_, p, take);
        buflen_ += take;
        p += take;
        len -= take;
        if (buflen_ == 64) {
            transform(buf_);
            buflen_ = 0;
        }
    }
}

void Sha256::transform(const uint8_t* block) {
    static const uint32_t K[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
        0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
        0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
        0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
        0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
        0xc67178f2u};

    uint32_t w[64];
    for (int i = 0; i < 16; ++i) {
        w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
               (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
               (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
               static_cast<uint32_t>(block[i * 4 + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
    uint32_t e = h_[4], f = h_[5], g = h_[6], h = h_[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t t1 = h + S1 + ch + K[i] + w[i];
        uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }
    h_[0] += a;
    h_[1] += b;
    h_[2] += c;
    h_[3] += d;
    h_[4] += e;
    h_[5] += f;
    h_[6] += g;
    h_[7] += h;
}

std::string Sha256::hexdigest() {
    uint64_t bitlen = bitlen_;
    uint8_t pad[72];
    size_t padlen = 0;
    pad[padlen++] = 0x80;
    size_t rem = static_cast<size_t>((bitlen / 8u) % 64u);  // bytes already buffered
    size_t zeros = (rem + 1 <= 56) ? (56 - rem - 1) : (120 - rem - 1);
    for (size_t i = 0; i < zeros; ++i) pad[padlen++] = 0;
    for (int i = 0; i < 8; ++i) {
        pad[padlen + i] = static_cast<uint8_t>((bitlen >> (56 - 8 * i)) & 0xffu);
    }
    padlen += 8;
    // At most 2 blocks of padding.
    update(pad, padlen);

    static const char* hex = "0123456789abcdef";
    std::string out(64, '0');
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 4; ++j) {
            uint32_t byte = (h_[i] >> (24 - 8 * j)) & 0xffu;
            out[(i * 4 + j) * 2] = hex[byte >> 4];
            out[(i * 4 + j) * 2 + 1] = hex[byte & 0x0f];
        }
    }
    return out;
}

std::string sha256_hex(const void* data, size_t len) {
    Sha256 h;
    h.update(data, len);
    return h.hexdigest();
}

std::string sha256_file_hex(const std::string& path) {
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (f == nullptr) {
        fail("cannot open file for sha256: " + path + ": " + std::strerror(errno));
    }
    Sha256 h;
    std::vector<unsigned char> buf(1u << 20);
    while (true) {
        size_t got = std::fread(buf.data(), 1, buf.size(), f);
        if (got > 0) h.update(buf.data(), got);
        if (got < buf.size()) {
            if (std::ferror(f)) {
                std::fclose(f);
                fail("read error while hashing: " + path);
            }
            break;
        }
    }
    std::fclose(f);
    return h.hexdigest();
}

// ---------------------------------------------------------------- ByteReader
ByteReader::ByteReader(const void* data, size_t size)
    : begin_(static_cast<const uint8_t*>(data)),
      p_(static_cast<const uint8_t*>(data)),
      end_(static_cast<const uint8_t*>(data) + size) {}

void ByteReader::need(size_t n) const {
    if (remaining() < n) {
        fail("unexpected end of file while parsing binary data (want " + std::to_string(n) +
             " bytes, have " + std::to_string(remaining()) + ")");
    }
}

void ByteReader::skip(size_t n) {
    need(n);
    p_ += n;
}

uint8_t ByteReader::u8() {
    need(1);
    return *p_++;
}

uint32_t ByteReader::u32() {
    need(4);
    uint32_t v = static_cast<uint32_t>(p_[0]) | (static_cast<uint32_t>(p_[1]) << 8) |
                 (static_cast<uint32_t>(p_[2]) << 16) | (static_cast<uint32_t>(p_[3]) << 24);
    p_ += 4;
    return v;
}

uint64_t ByteReader::u64() {
    need(8);
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) {
        v = (v << 8) | static_cast<uint64_t>(p_[i]);
    }
    p_ += 8;
    return v;
}

float ByteReader::f32() {
    uint32_t bits = u32();
    float v;
    std::memcpy(&v, &bits, sizeof(v));
    return v;
}

void ByteReader::f32_array(std::vector<float>* out, size_t count) {
    need(count * sizeof(float));
    out->resize(count);
#if defined(__BYTE_ORDER__) && (__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__)
    if (count > 0) {
        std::memcpy(out->data(), p_, count * sizeof(float));
    }
    p_ += count * sizeof(float);
#else
    for (size_t i = 0; i < count; ++i) {
        (*out)[i] = f32();
    }
#endif
}

void ByteReader::expect_magic(const char* magic, size_t n) {
    need(n);
    if (std::memcmp(p_, magic, n) != 0) {
        fail("bad magic in binary file (expected " + std::to_string(n) +
             "-byte contract magic)");
    }
    p_ += n;
}

// ---------------------------------------------------------------- platform
std::string platform_string() {
    struct utsname u;
    std::string sysname = "unknown";
    std::string release = "unknown";
    std::string machine = "unknown";
    if (::uname(&u) == 0) {
        sysname = u.sysname;
        release = u.release;
        machine = u.machine;
    }
    std::string compiler = "gcc?";
#ifdef __VERSION__
    compiler = __VERSION__;
#endif
    return sysname + " " + release + " " + machine + " (" + compiler + ")";
}

}  // namespace qbam
