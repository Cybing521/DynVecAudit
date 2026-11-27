// Q-BAM M1 engine -- replay-mode driver and counterfactual (cf) branch runner.
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "replay/replay_engine.h"

namespace qbam {

struct CfParams {
    PolicyRuntime parent;                          // parent trajectory policy + parameters
    std::vector<uint32_t> sample_blocks;           // decision blocks to branch at
    std::vector<Action> actions;                   // noop / refresh:r / merge:r
    std::vector<std::pair<int, int>> pair_actions;  // refresh_pair [[r,s], ...]
};

struct ReplayModeResult {
    std::vector<BlockMetrics> blocks;
    std::vector<WindowMetrics> windows;
};

struct CfModeResult {
    std::vector<BranchRecord> branches;
    std::vector<BlockMetrics> parent_blocks;
};

// mode=replay: applies `policy` at every decision block, replays all base windows in order,
// emits type="window" and type="block" lines, and streams type="query" lines when query_sink
// is non-null (--events-out).
ReplayModeResult run_replay_mode(ReplayEngine* engine, const PolicyRuntime& policy,
                                 std::vector<std::string>* lines, LineSink* query_sink);

// mode=cf: runs the parent trajectory over the whole trace without emitting its per-block
// records, then for every sampled block clones the decision-point state and replays each
// action's following K windows with no further maintenance. Branches for a block MUST start
// from the same state; the engine asserts that via state_hash.
// Emission: one type="state" line per sampled block; for every single action K type="window"
// lines + one type="block" line + one type="branch" line; for every pair_action one
// type="branch" line (refresh_pair; block schema has no shard2).
CfModeResult run_cf_mode(ReplayEngine* engine, const CfParams& cf, std::vector<std::string>* lines);

}  // namespace qbam
