#include "serve/port_contract.h"

#include "serve/generation_service.h"
#include "serve/request_events.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdio>
#include <ctime>
#include <dlfcn.h>
#include <iomanip>
#include <locale>
#include <sstream>

namespace ninfer::serve {
namespace {

// Histogram bounds of the fleet fork's latency families.
constexpr double kBuckets[] = {0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0,
                               2.5,   5.0,   10.0,  20.0, 40.0,  80.0, 160.0, 320.0, 640.0, 1280.0};
constexpr std::size_t kBucketCount = sizeof(kBuckets) / sizeof(kBuckets[0]);

std::string escape_label(const std::string& value) {
    std::string out;
    for (const char c : value) {
        if (c == '\\' || c == '"') { out.push_back('\\'); }
        if (c == '\n') {
            out += "\\n";
            continue;
        }
        out.push_back(c);
    }
    return out;
}

std::string utc_day() {
    const std::time_t tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
    gmtime_r(&tt, &utc);
    char key[16];
    std::snprintf(key, sizeof(key), "%04d-%02d-%02d", utc.tm_year + 1900, utc.tm_mon + 1,
                  utc.tm_mday);
    return key;
}

// GPU telemetry through NVML, loaded lazily and once (the fleet fork's sampler). Every field is
// optional: a missing symbol, library or unsupported query omits that metric. The ordinal is the
// CUDA device, which matches the NVML index under CUDA_DEVICE_ORDER=PCI_BUS_ID.
struct GpuTelemetry {
    bool utilization = false;
    unsigned int gpu_percent = 0, memory_percent = 0;
    bool memory = false;
    unsigned long long memory_total = 0, memory_free = 0, memory_used = 0;
    bool temperature = false;
    unsigned int temperature_celsius = 0;
    bool power = false;
    unsigned int power_mw = 0;
    bool power_limit = false;
    unsigned int power_limit_mw = 0;
    bool sm_clock = false;
    unsigned int sm_clock_mhz = 0;
    bool fan = false;
    unsigned int fan_percent = 0;
    bool energy = false;
    unsigned long long energy_mj = 0;
    std::string name;
};

class Nvml {
public:
    static const Nvml& instance() {
        static const Nvml nvml;
        return nvml;
    }
    [[nodiscard]] bool ready() const { return ready_; }

    GpuTelemetry sample(unsigned int ordinal) const {
        GpuTelemetry out;
        if (!ready_) { return out; }
        void* device = nullptr;
        if (handle_(ordinal, &device) != 0 || device == nullptr) { return out; }
        if (utilization_ != nullptr) {
            struct {
                unsigned int gpu, memory;
            } rates{};
            if (utilization_(device, &rates) == 0) {
                out.utilization    = true;
                out.gpu_percent    = rates.gpu;
                out.memory_percent = rates.memory;
            }
        }
        if (memory_ != nullptr) {
            struct {
                unsigned long long total, free, used;
            } info{};
            if (memory_(device, &info) == 0) {
                out.memory       = true;
                out.memory_total = info.total;
                out.memory_free  = info.free;
                out.memory_used  = info.used;
            }
        }
        unsigned int value = 0;
        if (temperature_ != nullptr && temperature_(device, 0, &value) == 0) {
            out.temperature         = true;
            out.temperature_celsius = value;
        }
        if (power_ != nullptr && power_(device, &value) == 0) {
            out.power    = true;
            out.power_mw = value;
        }
        if (power_limit_ != nullptr && power_limit_(device, &value) == 0) {
            out.power_limit    = true;
            out.power_limit_mw = value;
        }
        if (clock_ != nullptr && clock_(device, 1 /* NVML_CLOCK_SM */, &value) == 0) {
            out.sm_clock     = true;
            out.sm_clock_mhz = value;
        }
        if (fan_ != nullptr && fan_(device, &value) == 0) {
            out.fan         = true;
            out.fan_percent = value;
        }
        unsigned long long mj = 0;
        if (energy_ != nullptr && energy_(device, &mj) == 0) {
            out.energy    = true;
            out.energy_mj = mj;
        }
        if (name_ != nullptr) {
            char buffer[96]{};
            if (name_(device, buffer, sizeof(buffer)) == 0) { out.name = buffer; }
        }
        return out;
    }

private:
    using HandleFn   = int (*)(unsigned int, void**);
    using StructFn   = int (*)(void*, void*);
    using TempFn     = int (*)(void*, unsigned int, unsigned int*);
    using UnsignedFn = int (*)(void*, unsigned int*);
    using ClockFn    = int (*)(void*, unsigned int, unsigned int*);
    using EnergyFn   = int (*)(void*, unsigned long long*);
    using NameFn     = int (*)(void*, char*, unsigned int);

    Nvml() {
        void* library = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
        if (library == nullptr) { return; }
        const auto init = reinterpret_cast<int (*)()>(dlsym(library, "nvmlInit_v2"));
        handle_      = reinterpret_cast<HandleFn>(dlsym(library, "nvmlDeviceGetHandleByIndex_v2"));
        utilization_ = reinterpret_cast<StructFn>(dlsym(library, "nvmlDeviceGetUtilizationRates"));
        memory_      = reinterpret_cast<StructFn>(dlsym(library, "nvmlDeviceGetMemoryInfo"));
        temperature_ = reinterpret_cast<TempFn>(dlsym(library, "nvmlDeviceGetTemperature"));
        power_       = reinterpret_cast<UnsignedFn>(dlsym(library, "nvmlDeviceGetPowerUsage"));
        power_limit_ = reinterpret_cast<UnsignedFn>(dlsym(library, "nvmlDeviceGetPowerManagementLimit"));
        clock_       = reinterpret_cast<ClockFn>(dlsym(library, "nvmlDeviceGetClockInfo"));
        fan_         = reinterpret_cast<UnsignedFn>(dlsym(library, "nvmlDeviceGetFanSpeed"));
        energy_      = reinterpret_cast<EnergyFn>(dlsym(library, "nvmlDeviceGetTotalEnergyConsumption"));
        name_        = reinterpret_cast<NameFn>(dlsym(library, "nvmlDeviceGetName"));
        if (init == nullptr || handle_ == nullptr || init() != 0) { return; }
        ready_ = true;
    }

    HandleFn handle_        = nullptr;
    StructFn utilization_   = nullptr;
    StructFn memory_        = nullptr;
    TempFn temperature_     = nullptr;
    UnsignedFn power_       = nullptr;
    UnsignedFn power_limit_ = nullptr;
    ClockFn clock_          = nullptr;
    UnsignedFn fan_         = nullptr;
    EnergyFn energy_        = nullptr;
    NameFn name_            = nullptr;
    bool ready_             = false;
};

void observe(std::vector<double>& buckets, double value) {
    if (buckets.size() != kBucketCount) { buckets.assign(kBucketCount, 0.0); }
    for (std::size_t i = 0; i < kBucketCount; ++i) {
        if (value <= kBuckets[i]) { buckets[i] += 1.0; }
    }
}

} // namespace

void PortContract::configure(PortContractConfig config) {
    std::lock_guard lock(mutex_);
    config_             = std::move(config);
    start_              = std::chrono::steady_clock::now();
    start_unix_seconds_ = std::chrono::duration<double>(
                              std::chrono::system_clock::now().time_since_epoch())
                              .count();
    ttft_buckets_.assign(kBucketCount, 0.0);
    duration_buckets_.assign(kBucketCount, 0.0);
    last_tick_ = start_;
    restore();
}

void PortContract::record_done(const GenerationOutcome& outcome) {
    const GenerationMetrics& m = outcome.metrics;
    std::lock_guard lock(mutex_);
    if (outcome.finish_reason == ninfer::FinishReason::Cancelled) {
        ++cancelled_;
    } else {
        ++completed_;
    }
    spec_rounds_ += m.speculative_rounds;
    spec_drafted_ += m.speculative_draft_tokens;
    spec_accepted_ += m.speculative_accepted_tokens;
    spec_fallback_ += m.speculative_fallback_steps;
    const std::size_t positions = std::max<std::size_t>(m.speculative_accepted_per_position.size(),
                                                        m.speculative_draft_window);
    if (spec_attempted_.size() < positions) {
        spec_attempted_.resize(positions, 0);
        spec_position_accepted_.resize(positions, 0);
    }
    for (std::size_t i = 0; i < m.speculative_accepted_per_position.size(); ++i) {
        spec_position_accepted_[i] += m.speculative_accepted_per_position[i];
    }
    // A round attempts position i when its window is wider than i: the adaptive rounds by
    // window, else every round up to the configured window.
    if (!m.speculative_rounds_per_window.empty()) {
        for (std::size_t w = 0; w < m.speculative_rounds_per_window.size(); ++w) {
            const std::uint64_t rounds = m.speculative_rounds_per_window[w];
            for (std::size_t i = 0; i <= w && i < spec_attempted_.size(); ++i) {
                spec_attempted_[i] += rounds;
            }
        }
    } else {
        for (std::size_t i = 0; i < std::min<std::size_t>(m.speculative_draft_window, spec_attempted_.size());
             ++i) {
            spec_attempted_[i] += m.speculative_rounds;
        }
    }
    observe(ttft_buckets_, m.ttft_seconds);
    ttft_sum_ += m.ttft_seconds;
    ++ttft_count_;
    observe(duration_buckets_, m.total_seconds);
    duration_sum_ += m.total_seconds;
    ++duration_count_;
}

void PortContract::record_failure() {
    std::lock_guard lock(mutex_);
    ++failed_;
}

void PortContract::record_rejection() {
    std::lock_guard lock(mutex_);
    ++rejected_;
}

void PortContract::tick(const ThroughputReport& report) {
    const GpuTelemetry gpu = Nvml::instance().sample(config_.gpu_ordinal);
    const auto now         = std::chrono::steady_clock::now();
    bool persist_now       = false;
    {
        std::lock_guard lock(mutex_);
        const double seconds = std::chrono::duration<double>(now - last_tick_).count();
        last_tick_           = now;
        window_seconds_      = report.interval_seconds;
        window_prefill_      = report.computed_prefill_tokens;
        window_decode_       = report.committed_decode_tokens;
        // Watt-seconds over the interval: the energy counter's difference where the GPU keeps
        // one, else the interval's mean of the power samples at its ends.
        double ws = 0.0;
        if (gpu.energy) {
            if (last_energy_mj_ >= 0.0 && double(gpu.energy_mj) >= last_energy_mj_) {
                ws = (double(gpu.energy_mj) - last_energy_mj_) / 1000.0;
            }
            last_energy_mj_ = double(gpu.energy_mj);
        } else if (gpu.power) {
            ws = (double(last_power_mw_ + gpu.power_mw) / 2.0 / 1000.0) * seconds;
        }
        if (gpu.power) { last_power_mw_ = gpu.power_mw; }
        const std::string day    = utc_day();
        const std::uint64_t toks = report.computed_prefill_tokens + report.committed_decode_tokens;
        daily_ws_[day] += ws;
        daily_tokens_[day] += toks;
        if (tokens_today_key_ != day) {
            tokens_today_     = 0;
            tokens_today_key_ = day;
        }
        tokens_today_ += toks;
        while (daily_ws_.size() > 31) { daily_ws_.erase(daily_ws_.begin()); }
        while (daily_tokens_.size() > 31) { daily_tokens_.erase(daily_tokens_.begin()); }
        run_prompt_ = report.current.computed_prefill_tokens + report.current.reused_prompt_tokens;
        run_cached_ = report.current.reused_prompt_tokens;
        run_output_ = report.current.committed_decode_tokens;
        persist_now = ++ticks_ % 12 == 0;
    }
    if (persist_now) { persist(); }
}

void PortContract::persist() const {
    if (config_.state_path.empty()) { return; }
    nlohmann::json j;
    {
        std::lock_guard lock(mutex_);
        j = nlohmann::json{
            {"schema", 2},
            {"daily_ws", daily_ws_},
            {"daily_tokens", daily_tokens_},
            {"persisted_prompt_tokens", persisted_prompt_ + run_prompt_ - run_cached_},
            {"persisted_cached_tokens", persisted_cached_ + run_cached_},
            {"persisted_output_tokens", persisted_output_ + run_output_},
        };
    }
    const std::string tmp = config_.state_path + ".tmp";
    if (FILE* f = std::fopen(tmp.c_str(), "w")) {
        std::fputs(j.dump().c_str(), f);
        std::fclose(f);
        std::rename(tmp.c_str(), config_.state_path.c_str());
    }
}

void PortContract::restore() {
    if (config_.state_path.empty()) { return; }
    FILE* f = std::fopen(config_.state_path.c_str(), "r");
    if (f == nullptr) { return; }
    std::string text;
    char chunk[8192];
    std::size_t n = 0;
    while ((n = std::fread(chunk, 1, sizeof(chunk), f)) > 0) { text.append(chunk, n); }
    std::fclose(f);
    try {
        const auto j = nlohmann::json::parse(text);
        if (j.contains("daily_ws")) {
            for (auto it = j["daily_ws"].begin(); it != j["daily_ws"].end(); ++it) {
                daily_ws_[it.key()] = it.value().get<double>();
            }
        }
        if (j.contains("daily_tokens")) {
            for (auto it = j["daily_tokens"].begin(); it != j["daily_tokens"].end(); ++it) {
                daily_tokens_[it.key()] = it.value().get<std::uint64_t>();
            }
        }
        // The fleet fork stores prompt tokens without the cached ones (its total adds both).
        persisted_prompt_ = j.value("persisted_prompt_tokens", 0ULL);
        persisted_cached_ = j.value("persisted_cached_tokens", 0ULL);
        persisted_output_ = j.value("persisted_output_tokens", 0ULL);
        std::fprintf(stderr, "[metrics] state restored from %s\n", config_.state_path.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[metrics] state restore failed: %s\n", e.what());
    }
}

std::string PortContract::slots_json(const ninfer::RuntimeStats& stats) const {
    std::lock_guard lock(mutex_);
    nlohmann::json lanes = nlohmann::json::array();
    for (std::uint32_t lane = 0; lane < config_.max_concurrency; ++lane) {
        lanes.push_back({{"id", lane}, {"busy", lane < stats.running_requests}});
    }
    const nlohmann::json out{
        {"model", config_.model_id},
        {"max_concurrency", config_.max_concurrency},
        {"requests_processing", stats.running_requests},
        {"requests_waiting", stats.waiting_requests},
        {"prefilling", stats.prefilling_requests},
        {"decode_ready", stats.decode_ready_requests},
        {"materializing", stats.materializing_requests},
        {"paused", 0},
        {"replaying", 0},
        {"kv_pools", nlohmann::json::array()},
        {"slots", std::move(lanes)},
    };
    return out.dump();
}

std::string PortContract::usage_json(const ninfer::RuntimeStats& stats) const {
    std::lock_guard lock(mutex_);
    const double uptime = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_).count();
    const std::uint64_t prompt = persisted_prompt_ + persisted_cached_ +
                                 stats.computed_prefill_tokens + stats.reused_prompt_tokens;
    const std::uint64_t cached = persisted_cached_ + stats.reused_prompt_tokens;
    const std::uint64_t output = persisted_output_ + stats.committed_decode_tokens;
    const double rate          = config_.electricity_usd_per_kwh;
    const auto bucket = [&](double ws, std::uint64_t tokens) {
        const double cost = ws / 3.6e6 * rate;
        return nlohmann::json{{"wh", ws / 3600.0},
                              {"kwh", ws / 3.6e6},
                              {"cost_usd", cost},
                              {"tokens", tokens},
                              {"cost_per_m_tokens_usd", tokens != 0 ? cost / double(tokens) * 1e6 : 0.0}};
    };
    nlohmann::json days = nlohmann::json::object();
    double month_ws     = 0.0;
    std::uint64_t month_tokens = 0;
    for (const auto& [key, ws] : daily_ws_) {
        const auto found         = daily_tokens_.find(key);
        const std::uint64_t toks = found != daily_tokens_.end() ? found->second : 0;
        days[key]                = bucket(ws, toks);
        month_ws += ws;
    }
    for (const auto& [key, toks] : daily_tokens_) { month_tokens += toks; }
    const std::string today = utc_day();
    const auto today_ws     = daily_ws_.find(today);
    nlohmann::json per_position = nlohmann::json::array();
    for (std::size_t i = 0; i < spec_attempted_.size(); ++i) {
        const std::uint64_t attempted = spec_attempted_[i];
        const std::uint64_t accepted  = i < spec_position_accepted_.size() ? spec_position_accepted_[i] : 0;
        per_position.push_back({{"position", i},
                                {"attempted", attempted},
                                {"accepted", accepted},
                                {"acceptance_pct", attempted != 0 ? 100.0 * double(accepted) / double(attempted) : 0.0}});
    }
    const bool window_active = window_seconds_ > 0.0;
    const nlohmann::json out{
        {"model", config_.model_id},
        {"uptime",
         {{"seconds", uptime},
          {"human", std::to_string(int(uptime / 3600)) + "h " + std::to_string(int(uptime / 60) % 60) + "m"}}},
        {"tokens",
         {{"input",
           {{"total", prompt},
            {"cache_hits", cached},
            {"cache_computed", prompt - cached},
            {"cache_hit_rate_pct", prompt != 0 ? 100.0 * double(cached) / double(prompt) : 0.0}}},
          {"output", {{"total", output}}},
          {"total", prompt + output}}},
        {"throughput",
         {{"window_seconds", window_seconds_},
          {"window_active", window_active},
          {"prefill_tok_per_s", window_active ? double(window_prefill_) / window_seconds_ : 0.0},
          {"decode_tok_per_s", window_active ? double(window_decode_) / window_seconds_ : 0.0},
          {"prefill_tokens_last_window", window_prefill_},
          {"decode_tokens_last_window", window_decode_}}},
        {"speculative",
         {{"enabled", !spec_attempted_.empty()},
          {"rounds", spec_rounds_},
          {"draft_tokens", spec_drafted_},
          {"accepted_tokens", spec_accepted_},
          {"accepted_pct", spec_drafted_ != 0 ? 100.0 * double(spec_accepted_) / double(spec_drafted_) : 0.0},
          {"mean_accepted_per_round", spec_rounds_ != 0 ? double(spec_accepted_) / double(spec_rounds_) : 0.0},
          {"committed_draft_tokens", spec_accepted_},
          {"fallback_steps", spec_fallback_},
          {"per_position", std::move(per_position)}}},
        {"requests", {{"total", completed_ + failed_}, {"completed", completed_}, {"failed", failed_}}},
        {"lanes",
         {{"capacity", config_.max_concurrency},
          {"processing", stats.running_requests},
          {"waiting", stats.waiting_requests},
          {"decoding", stats.decode_ready_requests},
          {"prefilling", stats.prefilling_requests},
          {"restoring_checkpoints", stats.materializing_requests}}},
        {"health",
         {{"checkpoint_captures_completed", stats.active_captures_completed},
          {"checkpoint_captures_aborted", stats.active_captures_aborted},
          {"preemptions", 0},
          {"snapshot_restores", stats.state_restores},
          {"replay_restores", 0}}},
        {"energy",
         {{"rate_usd_per_kwh", rate},
          {"nvml_available", Nvml::instance().ready()},
          {"today", bucket(today_ws != daily_ws_.end() ? today_ws->second : 0.0,
                           tokens_today_key_ == today ? tokens_today_ : 0)},
          {"rolling_30d", bucket(month_ws, month_tokens)},
          {"daily", std::move(days)},
          {"note",
           "Measures the NVML device this process runs on (device " +
               std::to_string(config_.gpu_ordinal) +
               "). Excludes host components (CPU, RAM, cooling) and any other GPUs. Sampled via "
               "NVML at stats-reporter cadence. cost_per_m_tokens_usd is blended (input+output "
               "combined); the true split depends on cache hit rate and output share."}}},
    };
    return out.dump(2);
}

std::string PortContract::metrics_text(const ninfer::RuntimeStats& stats, bool ready) const {
    std::lock_guard lock(mutex_);
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::setprecision(17);
    const auto header = [&](const char* name, const char* type, const char* help) {
        out << "# HELP ninfer_" << name << ' ' << help << '\n' << "# TYPE ninfer_" << name << ' ' << type << '\n';
    };
    const auto sample = [&](const char* name, const char* type, auto value, const char* help) {
        header(name, type, help);
        out << "ninfer_" << name << ' ' << value << '\n';
    };
    sample("engine_ready", "gauge", ready ? 1 : 0, "Whether Engine can accept work.");
    sample("server_start_time_seconds", "gauge", start_unix_seconds_,
           "Unix time at service attachment after warmup.");
    header("model_info", "gauge", "Resident model and speculative backend.");
    out << "ninfer_model_info{model_name=\"" << escape_label(config_.model_id)
        << "\",speculative_backend=\"" << escape_label(config_.speculative_backend) << "\"} 1\n";
    sample("max_concurrency", "gauge", config_.max_concurrency, "Maximum resident execution lanes.");
    sample("max_context_tokens", "gauge", config_.max_context, "Per-request logical context limit.");
    sample("spec_decode_draft_window", "gauge", config_.draft_window, "Configured speculative draft window.");
    sample("requests_running", "gauge", stats.running_requests, "Resident requests, including prefill and decode.");
    sample("requests_waiting", "gauge", stats.waiting_requests, "Requests waiting for initial admission.");
    sample("requests_paused", "gauge", 0, "Paused requests waiting for recovery.");
    sample("requests_prefilling", "gauge", stats.prefilling_requests, "Resident requests in initial prefill.");
    sample("requests_decode_ready", "gauge", stats.decode_ready_requests, "Resident requests ready for decode.");
    sample("requests_replaying", "gauge", 0, "Resident requests rebuilding committed history.");
    sample("requests_materializing", "gauge", stats.materializing_requests,
           "Requests with a binding transaction in progress.");
    sample("prompt_tokens_total", "counter", stats.computed_prefill_tokens + stats.reused_prompt_tokens,
           "Full input tokens counted once on initial binding.");
    sample("prompt_tokens_cached_total", "counter", stats.reused_prompt_tokens,
           "Input tokens supplied by exact checkpoints.");
    sample("prefill_tokens_total", "counter", stats.computed_prefill_tokens,
           "Initial input tokens actually computed, excluding Replay.");
    sample("generation_tokens_total", "counter", stats.committed_decode_tokens,
           "Committed output tokens, including first and control tokens.");
    sample("decode_tokens_total", "counter", stats.committed_decode_tokens,
           "Committed decode/control tokens, excluding the first token.");
    sample("decode_rounds_total", "counter", stats.decode_rounds, "Decode rounds.");
    sample("decode_row_rounds_total", "counter", stats.decode_row_rounds, "Decode row rounds.");
    sample("preemptions_total", "counter", 0, "Resource-pressure pauses.");
    sample("snapshot_restores_total", "counter", stats.state_restores, "Restores from complete paused snapshots.");
    sample("replay_restores_total", "counter", 0, "Recovery operations that rebuild history.");
    sample("spec_decode_rounds_total", "counter", spec_rounds_, "Native speculative verification rounds.");
    sample("spec_decode_draft_tokens_total", "counter", spec_drafted_, "Draft tokens evaluated by native verification.");
    sample("spec_decode_accepted_tokens_total", "counter", spec_accepted_, "Draft tokens accepted by native verification.");
    sample("spec_decode_committed_draft_tokens_total", "counter", spec_accepted_,
           "Accepted draft tokens committed to the output.");
    sample("spec_decode_fallback_steps_total", "counter", spec_fallback_, "Speculative fallback steps without drafts.");
    header("spec_decode_position_tokens_total", "counter", "Draft tokens by position, attempted and accepted.");
    for (std::size_t i = 0; i < spec_attempted_.size(); ++i) {
        out << "ninfer_spec_decode_position_tokens_total{position=\"" << i << "\",kind=\"attempted\"} "
            << spec_attempted_[i] << '\n';
        out << "ninfer_spec_decode_position_tokens_total{position=\"" << i << "\",kind=\"accepted\"} "
            << (i < spec_position_accepted_.size() ? spec_position_accepted_[i] : 0) << '\n';
    }
    header("requests_total", "counter", "Generation attempts entering preparation, by terminal outcome.");
    out << "ninfer_requests_total{outcome=\"completed\"} " << completed_ << '\n'
        << "ninfer_requests_total{outcome=\"cancelled\"} " << cancelled_ << '\n'
        << "ninfer_requests_total{outcome=\"failed\"} " << failed_ << '\n'
        << "ninfer_requests_total{outcome=\"rejected\"} " << rejected_ << '\n';
    sample("requests_submitted_total", "counter", completed_ + cancelled_ + failed_, "Requests submitted to the Engine.");
    header("engine_requests_total", "counter", "Engine requests by outcome.");
    out << "ninfer_engine_requests_total{outcome=\"submitted\"} " << completed_ + cancelled_ + failed_ << '\n'
        << "ninfer_engine_requests_total{outcome=\"completed\"} " << completed_ << '\n'
        << "ninfer_engine_requests_total{outcome=\"cancelled\"} " << cancelled_ << '\n'
        << "ninfer_engine_requests_total{outcome=\"failed\"} " << failed_ << '\n';
    const auto histogram = [&](const char* name, const char* help, const std::vector<double>& buckets,
                               double sum, std::uint64_t count) {
        header(name, "histogram", help);
        for (std::size_t i = 0; i < kBucketCount; ++i) {
            out << "ninfer_" << name << "_bucket{le=\"" << kBuckets[i] << "\"} "
                << (i < buckets.size() ? std::uint64_t(buckets[i]) : 0) << '\n';
        }
        out << "ninfer_" << name << "_bucket{le=\"+Inf\"} " << count << '\n'
            << "ninfer_" << name << "_sum " << sum << '\n'
            << "ninfer_" << name << "_count " << count << '\n';
    };
    histogram("time_to_first_token_seconds",
              "Preparation through first committed token; observed once, before completion.",
              ttft_buckets_, ttft_sum_, ttft_count_);
    histogram("request_duration_seconds", "Preparation through terminal outcome.", duration_buckets_,
              duration_sum_, duration_count_);
    // Per-rank GPU telemetry.
    const GpuTelemetry gpu = Nvml::instance().sample(config_.gpu_ordinal);
    const std::string labels = "{rank=\"0\",node=\"0\"}";
    const auto gpu_metric = [&](const char* name, const char* help, bool present, auto value) {
        header(name, "gauge", help);
        if (present) { out << "ninfer_" << name << labels << ' ' << value << '\n'; }
    };
    gpu_metric("cuda_memory_free_bytes", "Sampled CUDA free memory, not process-owned bytes.", gpu.memory,
               gpu.memory_free);
    gpu_metric("gpu_utilization_percent", "Sampled GPU SM utilization, 0-100.", gpu.utilization, gpu.gpu_percent);
    gpu_metric("gpu_memory_utilization_percent", "Sampled GPU memory-controller utilization, 0-100.",
               gpu.utilization, gpu.memory_percent);
    gpu_metric("gpu_temperature_celsius", "Sampled GPU die temperature.", gpu.temperature, gpu.temperature_celsius);
    gpu_metric("gpu_power_watts", "Sampled GPU board power draw.", gpu.power, double(gpu.power_mw) / 1000.0);
    gpu_metric("gpu_power_limit_watts", "Configured GPU power management limit.", gpu.power_limit,
               double(gpu.power_limit_mw) / 1000.0);
    gpu_metric("gpu_sm_clock_mhz", "Sampled GPU SM clock.", gpu.sm_clock, gpu.sm_clock_mhz);
    gpu_metric("gpu_memory_total_bytes", "Sampled GPU total memory.", gpu.memory, gpu.memory_total);
    gpu_metric("gpu_memory_used_bytes", "Sampled GPU memory in use.", gpu.memory, gpu.memory_used);
    gpu_metric("gpu_memory_free_bytes", "Sampled GPU free memory.", gpu.memory, gpu.memory_free);
    gpu_metric("gpu_fan_speed_percent", "Sampled GPU fan speed, 0-100.", gpu.fan, gpu.fan_percent);
    header("gpu_info", "gauge", "Resident GPU identity; always 1.");
    if (!gpu.name.empty()) {
        out << "ninfer_gpu_info{rank=\"0\",node=\"0\",name=\"" << escape_label(gpu.name) << "\"} 1\n";
    }
    return out.str();
}

} // namespace ninfer::serve
