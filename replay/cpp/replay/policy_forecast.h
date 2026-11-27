// Q-BAM engine -- E2 forecast table loader (predictions.csv) for the E3-stage2 policies.
#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace qbam {

// predictions.csv schema (E2 output):
//   trace,drift,seed,j,shard,model,q_pred,q_true,delta_n,split,in_sample
// An optional `h` column is tolerated: rows are then summed per (j, shard) (multi-horizon score).
struct ForecastTable {
    bool loaded = false;
    std::string source;
    std::string model;
    std::string trace_name;
    std::unordered_map<uint64_t, double> qhat;  // key(j, shard) -> q_hat
    size_t rows_total = 0;
    size_t rows_kept = 0;

    static uint64_t key(uint32_t j, uint32_t shard) {
        return (static_cast<uint64_t>(j) << 32) | static_cast<uint64_t>(shard);
    }
    bool has(uint32_t j, uint32_t shard) const { return qhat.find(key(j, shard)) != qhat.end(); }
    // Returns q_hat or NaN when (j, shard) is absent.
    double get(uint32_t j, uint32_t shard) const;
    bool load(const std::string& path, const std::string& trace_name, const std::string& model,
              std::string* error);
};

}  // namespace qbam
