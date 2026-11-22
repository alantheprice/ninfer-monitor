#pragma once

// The monitoring contract of the alantheprice fleet fork (ninfer-monitor port/fleet-features),
// which link agents and the gateway read: /usage (tokens, throughput window, speculative
// acceptance by position, lanes, NVML energy with per-day cost), /slots (lane occupancy) and the
// ninfer_* Prometheus families (engine gauges and counters, speculative decoding, request
// latency histograms, per-rank GPU telemetry). Built from this line's runtime statistics and
// request outcomes; the same names, labels and JSON shapes as the fleet fork. Energy and the
// lifetime token baseline persist in "<request log>.metrics-state.json" (schema 2), the fleet
// fork's file, so a deployment keeps its history across the switch.

#include "ninfer/observability/snapshot.h"

#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace ninfer::serve {

struct GenerationOutcome;
struct ThroughputReport;

struct PortContractConfig {
    std::string model_id;
    std::uint32_t max_concurrency = 0;
    std::uint32_t max_context     = 0;
    std::string speculative_backend; // the fleet fork's backend names: none, mtp, dspark, ...
    std::uint32_t draft_window = 0;
    double electricity_usd_per_kwh = 0.0;
    std::string state_path; // empty: no persistence
    unsigned int gpu_ordinal = 0;
};

class PortContract {
public:
    void configure(PortContractConfig config);

    // One finished request (the request-done funnel), a failed one, a refused one.
    void record_done(const GenerationOutcome& outcome);
    void record_failure();
    void record_rejection();

    // One stats-reporter interval: the throughput window /usage reports and the energy the GPU
    // drew over it, with the interval's computed tokens attributed to the same UTC day.
    void tick(const ThroughputReport& report);
    // Writes the energy and token baseline (every 12 ticks, and at shutdown).
    void persist() const;

    [[nodiscard]] std::string usage_json(const ninfer::RuntimeStats& stats) const;
    [[nodiscard]] std::string slots_json(const ninfer::RuntimeStats& stats) const;
    // The ninfer_* families, appended after this line's own metrics.
    [[nodiscard]] std::string metrics_text(const ninfer::RuntimeStats& stats, bool ready) const;

private:
    void restore();

    PortContractConfig config_;
    std::chrono::steady_clock::time_point start_ = std::chrono::steady_clock::now();
    double start_unix_seconds_ = 0.0;

    mutable std::mutex mutex_;
    // Requests and their speculative rounds.
    std::uint64_t completed_ = 0, failed_ = 0, rejected_ = 0, cancelled_ = 0;
    std::uint64_t spec_rounds_ = 0, spec_drafted_ = 0, spec_accepted_ = 0, spec_fallback_ = 0;
    std::vector<std::uint64_t> spec_attempted_, spec_position_accepted_;
    std::vector<double> ttft_buckets_, duration_buckets_;
    double ttft_sum_ = 0.0, duration_sum_ = 0.0;
    std::uint64_t ttft_count_ = 0, duration_count_ = 0;
    // The latest reporter interval.
    double window_seconds_ = 0.0;
    std::uint64_t window_prefill_ = 0, window_decode_ = 0;
    // Energy: watt-seconds and computed tokens per UTC day (31 retained).
    std::map<std::string, double> daily_ws_;
    std::map<std::string, std::uint64_t> daily_tokens_;
    std::uint64_t tokens_today_ = 0;
    std::string tokens_today_key_;
    double last_energy_mj_ = -1.0;
    std::chrono::steady_clock::time_point last_tick_ = std::chrono::steady_clock::now();
    unsigned int last_power_mw_ = 0;
    // Lifetime baseline from earlier runs.
    std::uint64_t persisted_prompt_ = 0, persisted_cached_ = 0, persisted_output_ = 0;
    // This run's lifetime counters at the last tick (persisted as part of the baseline).
    std::uint64_t run_prompt_ = 0, run_cached_ = 0, run_output_ = 0;
    std::uint64_t ticks_ = 0;
};

} // namespace ninfer::serve
