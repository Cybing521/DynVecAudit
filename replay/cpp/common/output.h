// Q-BAM M1 engine -- JSONL sinks (append-only file output, contract §6).
#pragma once

#include <cstdio>
#include <string>
#include <vector>

namespace qbam {

class LineSink {
public:
    virtual ~LineSink() = default;
    virtual void write_line(const std::string& line) = 0;
};

// Collects lines in memory (metric records are small; query records stream instead).
class VectorLineSink : public LineSink {
public:
    void write_line(const std::string& line) override { lines.push_back(line); }
    std::vector<std::string> lines;
};

// Append-only JSONL file. open_exclusive() fails if the path already exists
// (contract §6: results are append-only, never overwrite).
class FileLineSink : public LineSink {
public:
    FileLineSink() = default;
    ~FileLineSink() override;
    FileLineSink(const FileLineSink&) = delete;
    FileLineSink& operator=(const FileLineSink&) = delete;

    // Empty path => stdout.
    void open_exclusive(const std::string& path);
    void write_line(const std::string& line) override;
    void close();
    bool is_open() const { return file_ != nullptr; }

private:
    std::FILE* file_ = nullptr;
    bool is_stdout_ = false;
};

}  // namespace qbam
