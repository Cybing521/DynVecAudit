// Q-BAM engine -- E2 forecast table loader implementation (minimal CSV parser, no deps).
#include "replay/policy_forecast.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace qbam {

namespace {
std::string trim_copy(const std::string& s) {
    size_t b = 0, e = s.size();
    while (b < e && (s[b] == ' ' || s[b] == '\t' || s[b] == '\r' || s[b] == '\n' || s[b] == '"')) ++b;
    while (e > b && (s[e - 1] == ' ' || s[e - 1] == '\t' || s[e - 1] == '\r' || s[e - 1] == '\n' ||
                     s[e - 1] == '"')) {
        --e;
    }
    return s.substr(b, e - b);
}

std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : line) {
        if (c == ',') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(c);
        }
    }
    out.push_back(cur);
    return out;
}
}  // namespace

double ForecastTable::get(uint32_t j, uint32_t shard) const {
    auto it = qhat.find(key(j, shard));
    if (it == qhat.end()) return std::numeric_limits<double>::quiet_NaN();
    return it->second;
}

bool ForecastTable::load(const std::string& path, const std::string& trace_name_in,
                         const std::string& model_in, std::string* error) {
    loaded = false;
    qhat.clear();
    rows_total = 0;
    rows_kept = 0;
    source = path;
    trace_name = trace_name_in;
    model = model_in;
    std::ifstream in(path);
    if (!in) {
        if (error) *error = "cannot open predictions file: " + path;
        return false;
    }
    std::string header;
    if (!std::getline(in, header)) {
        if (error) *error = "empty predictions file: " + path;
        return false;
    }
    const std::vector<std::string> cols = split_csv(header);
    int i_trace = -1, i_j = -1, i_shard = -1, i_model = -1, i_q = -1, i_h = -1;
    for (size_t c = 0; c < cols.size(); ++c) {
        const std::string name = trim_copy(cols[c]);
        if (name == "trace") i_trace = static_cast<int>(c);
        else if (name == "j") i_j = static_cast<int>(c);
        else if (name == "shard") i_shard = static_cast<int>(c);
        else if (name == "model") i_model = static_cast<int>(c);
        else if (name == "q_pred") i_q = static_cast<int>(c);
        else if (name == "h") i_h = static_cast<int>(c);
    }
    if (i_trace < 0 || i_j < 0 || i_shard < 0 || i_model < 0 || i_q < 0) {
        if (error) *error = "predictions.csv is missing required columns (trace,j,shard,model,q_pred)";
        return false;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const std::vector<std::string> f = split_csv(line);
        const size_t need = static_cast<size_t>(std::max({i_trace, i_j, i_shard, i_model, i_q, i_h})) + 1;
        if (f.size() < need) continue;
        ++rows_total;
        if (trim_copy(f[i_trace]) != trace_name_in) continue;
        if (trim_copy(f[i_model]) != model_in) continue;
        char* endp = nullptr;
        const long j = std::strtol(trim_copy(f[i_j]).c_str(), &endp, 10);
        if (endp == nullptr) continue;
        const long shard = std::strtol(trim_copy(f[i_shard]).c_str(), &endp, 10);
        const double q = std::strtod(trim_copy(f[i_q]).c_str(), &endp);
        if (j < 0 || shard < 0) continue;
        const uint64_t k = key(static_cast<uint32_t>(j), static_cast<uint32_t>(shard));
        qhat[k] += q;  // multi-horizon rows (optional h column) are summed
        ++rows_kept;
    }
    loaded = rows_kept > 0;
    if (!loaded && error) {
        *error = "no rows matched trace=\"" + trace_name_in + "\" model=\"" + model_in +
                 "\" in " + path;
    }
    return loaded;
}

}  // namespace qbam
