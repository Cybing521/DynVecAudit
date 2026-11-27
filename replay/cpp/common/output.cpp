// Q-BAM M1 engine -- JSONL sinks implementation.
#include "common/output.h"

#include <cerrno>
#include <cstring>

#include "common/logging.h"
#include "common/util.h"

namespace qbam {

FileLineSink::~FileLineSink() { close(); }

void FileLineSink::open_exclusive(const std::string& path) {
    QBAM_CHECK(file_ == nullptr, "FileLineSink: already open");
    if (path.empty()) {
        file_ = stdout;
        is_stdout_ = true;
        return;
    }
    if (file_exists(path)) {
        fail("output file already exists (append-only rule): " + path);
    }
    file_ = std::fopen(path.c_str(), "wbx");
    if (file_ == nullptr) {
        fail("cannot create output file: " + path + ": " + std::strerror(errno));
    }
    char buf[1 << 20];
    std::setvbuf(file_, buf, _IOFBF, sizeof(buf));
}

void FileLineSink::write_line(const std::string& line) {
    QBAM_CHECK(file_ != nullptr, "FileLineSink: write before open");
    if (!line.empty()) {
        if (std::fwrite(line.data(), 1, line.size(), file_) != line.size()) {
            fail("short write to output stream");
        }
    }
    if (std::fputc('\n', file_) == EOF) {
        fail("write error on output stream");
    }
}

void FileLineSink::close() {
    if (file_ == nullptr) return;
    if (std::fflush(file_) != 0) {
        if (!is_stdout_) std::fclose(file_);
        file_ = nullptr;
        fail("flush error on output stream");
    }
    if (!is_stdout_ && std::fclose(file_) != 0) {
        file_ = nullptr;
        fail("close error on output file");
    }
    file_ = nullptr;
}

}  // namespace qbam
