#include "serve/http_server.h"

#include "serve/anthropic_messages.h"
#include "serve/http_transport.h"
#include "serve/openai_common.h"
#include "serve/request_log.h"
#include "product/speculative_options.h"

#include <nlohmann/json.hpp>

#include <cuda_runtime.h>
#include <dlfcn.h>
#include <sys/utsname.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace ninfer::serve {
namespace {

// NVML dynamic loading for GPU power sampling. Loaded once; sampling is a no-op
// returning 0 if the library or device is unavailable (energy accounting then
// reports unavailable rather than wrong numbers).
struct GpuPowerSampler {
    // nvmlDeviceGetPowerUsage(nvmlDevice_t, unsigned int*) — milliwatts.
    using PowerFn = int (*)(void*, unsigned int*);
    // nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t*) — note arg order.
    using HandleFn = int (*)(unsigned int, void*);
    PowerFn query_power = nullptr;
    HandleFn handle_fn  = nullptr;
    void* nvml_handle = nullptr;
    void* device = nullptr;
    bool initialized = false;

    GpuPowerSampler() {
        nvml_handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
        if (nvml_handle == nullptr) {
            std::fprintf(stderr, "[nvml] dlopen failed\n");
            return;
        }
        auto init_fn     = (int (*)())dlsym(nvml_handle, "nvmlInit_v2");
        handle_fn        = (HandleFn)dlsym(nvml_handle, "nvmlDeviceGetHandleByIndex_v2");
        query_power      = (PowerFn)dlsym(nvml_handle, "nvmlDeviceGetPowerUsage");
        if (init_fn == nullptr || handle_fn == nullptr || query_power == nullptr) {
            std::fprintf(stderr, "[nvml] symbol lookup failed\n");
            return;
        }
        if (init_fn() != 0) {
            std::fprintf(stderr, "[nvml] init failed\n");
            return;
        }
        // Prefer the CUDA device order match is unavailable at this layer; probe 0..3.
        int rc = 1;
        for (unsigned int idx = 0; idx < 4 && rc != 0; ++idx) {
            rc = handle_fn(idx, &device);
        }
        if (rc != 0) {
            std::fprintf(stderr, "[nvml] no device handle (rc=%d)\n", rc);
            return;
        }
        initialized = true;
    }
    ~GpuPowerSampler() = default;
    void sample() {
        if (!initialized) { return; }
        unsigned int mw = 0;
        if (query_power(device, &mw) == 0) { last_milliwatts = mw; }
    }
    unsigned int last_milliwatts = 0;
};

void write_exception(httplib::Response& res, const std::exception& ex) {
    ApiError error;
    error.status  = 500;
    error.type    = "internal_error";
    error.message = ex.what();
    write_openai_error(res, error);
}

bool is_anthropic_path(std::string_view path) { return path.starts_with("/v1/messages"); }

bool is_openai_path(std::string_view path) {
    return path.starts_with("/v1/") && !is_anthropic_path(path);
}

void ensure_openai_request_id(const httplib::Request& request, httplib::Response& response) {
    if (is_openai_path(request.path) && !response.has_header("x-request-id")) {
        response.set_header("x-request-id", new_openai_request_id());
    }
}

ThroughputReport make_throughput_report(const ninfer::RuntimeStats& previous,
                                        const ninfer::RuntimeStats& current,
                                        double interval_seconds) {
    return ThroughputReport{
        .interval_seconds = interval_seconds,
        .computed_prefill_tokens =
            current.computed_prefill_tokens - previous.computed_prefill_tokens,
        .committed_decode_tokens =
            current.committed_decode_tokens - previous.committed_decode_tokens,
        .decode_rounds     = current.decode_rounds - previous.decode_rounds,
        .decode_row_rounds = current.decode_row_rounds - previous.decode_row_rounds,
        .previous          = previous,
        .current           = current,
    };
}

bool report_has_activity(const ThroughputReport& report) {
    return report.computed_prefill_tokens != 0 || report.committed_decode_tokens != 0 ||
           report.decode_rounds != 0 || report.current.running_requests != 0 ||
           report.current.waiting_requests != 0 || report.current.materializing_requests != 0 ||
           report.current.capture_pending_requests != 0 ||
           report.current.terminal_pending_requests != 0 ||
           report.current.active_captures_completed != report.previous.active_captures_completed ||
           report.current.active_captures_aborted != report.previous.active_captures_aborted ||
           report.current.root_selections != report.previous.root_selections ||
           report.current.private_endpoint_selections !=
               report.previous.private_endpoint_selections ||
           report.current.private_turn_closure_selections !=
               report.previous.private_turn_closure_selections ||
           report.current.private_response_replay_selections !=
               report.previous.private_response_replay_selections ||
           report.current.private_long_anchor_selections !=
               report.previous.private_long_anchor_selections ||
           report.current.shared_stable_prefix_selections !=
               report.previous.shared_stable_prefix_selections ||
           report.current.state_moves != report.previous.state_moves ||
           report.current.state_forks != report.previous.state_forks ||
           report.current.state_restores != report.previous.state_restores ||
           report.current.state_d2h_count != report.previous.state_d2h_count ||
           report.current.state_h2d_count != report.previous.state_h2d_count ||
           report.current.state_d2d_count != report.previous.state_d2d_count ||
           report.current.main_kv_d2h_pages != report.previous.main_kv_d2h_pages ||
           report.current.main_kv_h2d_pages != report.previous.main_kv_h2d_pages ||
           report.current.main_kv_d2d_pages != report.previous.main_kv_d2d_pages ||
           report.current.backend_kv_d2h_pages != report.previous.backend_kv_d2h_pages ||
           report.current.backend_kv_h2d_pages != report.previous.backend_kv_h2d_pages ||
           report.current.backend_kv_d2d_pages != report.previous.backend_kv_d2d_pages ||
           report.current.pressure_spill_pages != report.previous.pressure_spill_pages ||
           report.current.partial_tail_cow_pages != report.previous.partial_tail_cow_pages ||
           report.current.pressure_private_owners_degraded !=
               report.previous.pressure_private_owners_degraded ||
           report.current.pressure_private_owners_evicted !=
               report.previous.pressure_private_owners_evicted ||
           report.current.pressure_shared_owners_degraded !=
               report.previous.pressure_shared_owners_degraded ||
           report.current.pressure_shared_owners_evicted !=
               report.previous.pressure_shared_owners_evicted ||
           report.current.pressure_checkpoints_dropped !=
               report.previous.pressure_checkpoints_dropped ||
           report.current.pressure_searches != report.previous.pressure_searches ||
           report.current.pressure_search_budget_exhaustions !=
               report.previous.pressure_search_budget_exhaustions ||
           report.current.pressure_maximal_fallback_selections !=
               report.previous.pressure_maximal_fallback_selections ||
           report.current.historical_fork_hits != report.previous.historical_fork_hits ||
           report.current.device_state_occupied_slots !=
               report.previous.device_state_occupied_slots ||
           report.current.host_state_occupied_slots != report.previous.host_state_occupied_slots ||
           report.current.device_main_kv_occupied_pages !=
               report.previous.device_main_kv_occupied_pages ||
           report.current.device_backend_kv_occupied_pages !=
               report.previous.device_backend_kv_occupied_pages ||
           report.current.host_kv_occupied_bytes != report.previous.host_kv_occupied_bytes ||
           report.current.disk_restores != report.previous.disk_restores ||
           report.current.disk_saves != report.previous.disk_saves ||
           report.current.shared_active_references != report.previous.shared_active_references ||
           report.current.host_work.engine_boundary_ns !=
               report.previous.host_work.engine_boundary_ns ||
           report.current.host_work.program_submit_ns !=
               report.previous.host_work.program_submit_ns ||
           report.current.host_work.program_post_ns != report.previous.host_work.program_post_ns ||
           report.current.host_work.engine_commit_output_ns !=
               report.previous.host_work.engine_commit_output_ns ||
           report.current.host_work.engine_maintenance_ns !=
               report.previous.host_work.engine_maintenance_ns ||
           report.current.host_work.device_wait_ns != report.previous.host_work.device_wait_ns;
}

const char* endpoint_name(std::string_view path) noexcept {
    if (path == "/v1/chat/completions") { return "openai_chat_completions"; }
    if (path == "/v1/responses") { return "openai_responses"; }
    if (path == "/v1/responses/input_tokens") { return "openai_responses_input_tokens"; }
    if (path == "/v1/messages") { return "anthropic_messages"; }
    if (path == "/v1/messages/count_tokens") { return "anthropic_count_tokens"; }
    return "http_route";
}

std::string response_request_id(const httplib::Response& response) {
    if (response.has_header("x-request-id")) { return response.get_header_value("x-request-id"); }
    if (response.has_header("request-id")) { return response.get_header_value("request-id"); }
    return {};
}

} // namespace

void write_openai_error(httplib::Response& response, const ApiError& error) {
    response.status = error.status;
    response.set_content(make_error_body(error), "application/json");
}

void write_anthropic_error(httplib::Response& response, const ApiError& api_error,
                           const std::string& request_id) {
    const ApiError error = normalize_anthropic_error(api_error);
    response.status      = error.status;
    response.headers.erase("request-id");
    response.set_header("request-id", request_id);
    response.set_content(make_anthropic_error_body(error, request_id), "application/json");
}

httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response) {
    ensure_openai_request_id(request, response);
    if (!response.body.empty()) { return httplib::Server::HandlerResponse::Unhandled; }

    ApiError error;
    if (response.status == 413) {
        error.status  = 413;
        error.type    = "invalid_request_error";
        error.code    = "request_too_large";
        error.message = "request body exceeds the configured payload limit of " +
                        std::to_string(options.max_request_bytes) + " bytes";
    } else if (response.status == 404 && request.path.rfind("/v1/messages", 0) == 0) {
        error.status  = 404;
        error.code    = "not_found";
        error.message = "requested Anthropic resource was not found";
    } else {
        return httplib::Server::HandlerResponse::Unhandled;
    }
    if (request.path.rfind("/v1/messages", 0) == 0) {
        write_anthropic_error(response, error, new_anthropic_request_id());
    } else {
        write_openai_error(response, error);
    }
    return httplib::Server::HandlerResponse::Handled;
}

bool matches_bearer_credential(std::string_view authorization, std::string_view api_key) noexcept {
    if (api_key.empty()) { return false; }
    const auto is_whitespace = [](char value) { return value == ' ' || value == '\t'; };
    const auto ascii_equal   = [](char lhs, char rhs) {
        if (lhs >= 'A' && lhs <= 'Z') { lhs = static_cast<char>(lhs - 'A' + 'a'); }
        if (rhs >= 'A' && rhs <= 'Z') { rhs = static_cast<char>(rhs - 'A' + 'a'); }
        return lhs == rhs;
    };

    std::size_t position = 0;
    while (position < authorization.size() && is_whitespace(authorization[position])) {
        ++position;
    }
    constexpr std::string_view scheme = "Bearer";
    if (authorization.size() - position < scheme.size()) { return false; }
    for (std::size_t index = 0; index < scheme.size(); ++index) {
        if (!ascii_equal(authorization[position + index], scheme[index])) { return false; }
    }
    position += scheme.size();
    if (position == authorization.size() || !is_whitespace(authorization[position])) {
        return false;
    }
    while (position < authorization.size() && is_whitespace(authorization[position])) {
        ++position;
    }
    std::size_t end = authorization.size();
    while (end > position && is_whitespace(authorization[end - 1])) { --end; }
    return authorization.substr(position, end - position) == api_key;
}

HttpServer::HttpServer(ServeOptions options, std::shared_ptr<spdlog::logger> logger)
    : options_(std::move(options)), openai_responses_store_(options_.response_store_max_records,
                                                            options_.response_store_max_bytes),
      operational_log_(logger),
      request_jsonl_(options_.request_log_jsonl, options_.artifact_path, std::move(logger)) {
    const std::size_t queued_requests =
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests;
    const std::size_t worker_count = queued_requests + 1;
    server_.new_task_queue         = [queued_requests, worker_count] {
        return new httplib::ThreadPool(worker_count, worker_count, queued_requests);
    };
    server_.set_socket_options(configure_http_server_socket);
    server_.set_payload_max_length(options_.max_request_bytes);
    register_routes();
}

HttpServer::RequestLifecycle::RequestLifecycle(HttpServer& owner, RequestLogContext context)
    : owner_(&owner), context_(std::move(context)) {
    owner_->record_request_start(context_);
}

bool HttpServer::RequestLifecycle::claim(State terminal) noexcept {
    State expected = State::Pending;
    return state_.compare_exchange_strong(expected, terminal, std::memory_order_acq_rel);
}

void HttpServer::RequestLifecycle::done(const GenerationOutcome& outcome) {
    if (claim(State::Done)) { owner_->record_request_done(context_, outcome); }
}

void HttpServer::RequestLifecycle::failure(const RequestFailure& failure) {
    if (claim(State::Error)) { owner_->record_request_failure(context_, failure); }
}

void HttpServer::RequestLifecycle::response_failure(const RequestFailure& failure) {
    owner_->record_response_failure(context_.id, failure);
}

std::shared_ptr<HttpServer::RequestLifecycle> HttpServer::begin_request(RequestLogContext context) {
    return std::make_shared<RequestLifecycle>(*this, std::move(context));
}

void HttpServer::record_request_start(const RequestLogContext& context) {
    request_jsonl_.write_request_start(context);
    operational_log_.request_start(context);
}

void HttpServer::record_request_rejected(const RequestRejectionLogContext& context) {
    request_jsonl_.write_request_rejected(context);
    operational_log_.request_rejected(context);
}

void HttpServer::record_request_done(const RequestLogContext& context,
                                     const GenerationOutcome& outcome) {
    request_jsonl_.write_request_done(context, outcome);
    operational_log_.request_done(context, outcome);
}

void HttpServer::record_request_failure(const RequestLogContext& context,
                                        const RequestFailure& failure) {
    request_jsonl_.write_request_error(context, failure.machine_message);
    operational_log_.request_failure(context, failure);
}

void HttpServer::record_response_failure(std::uint64_t request_id, const RequestFailure& failure) {
    operational_log_.response_failure(request_id, failure);
}

void HttpServer::record_throughput(const ThroughputReport& report) {
    request_jsonl_.write_throughput(report);
    operational_log_.throughput(report);
    {
        std::lock_guard lock(last_throughput_mutex_);
        last_throughput_        = report;
        last_throughput_at_     = std::chrono::steady_clock::now();
    }
    // Lifetime token counters (survive restarts via persisted baseline).
    persisted_output_tokens += report.committed_decode_tokens;
    persisted_prompt_tokens += report.computed_prefill_tokens;
    persisted_cached_tokens +=
        report.current.reused_prompt_tokens - report.previous.reused_prompt_tokens;
}

void HttpServer::persist_metrics_state() {
    const std::string path = !options_.metrics_state_path.empty()
                                 ? options_.metrics_state_path
                                 : (!options_.request_log_jsonl.empty()
                                        ? options_.request_log_jsonl + ".metrics-state.json"
                                        : std::string());
    if (path.empty()) { return; }
    std::lock_guard lock(energy_mutex_);
    nlohmann::json daily_tokens_json = nlohmann::json::object();
    for (const auto& [key, count] : energy.daily_tokens) {
        daily_tokens_json[key] = count;
    }
    nlohmann::json j{
        {"schema", 2},
        {"daily_ws", energy.daily_ws},
        {"daily_tokens", daily_tokens_json},
        {"persisted_prompt_tokens", persisted_prompt_tokens},
        {"persisted_cached_tokens", persisted_cached_tokens},
        {"persisted_output_tokens", persisted_output_tokens},
    };
    const std::string tmp = path + ".tmp";
    if (FILE* f = fopen(tmp.c_str(), "w")) {
        fputs(j.dump().c_str(), f);
        fclose(f);
        rename(tmp.c_str(), path.c_str());
    }
}

void HttpServer::restore_metrics_state() {
    const std::string path = !options_.metrics_state_path.empty()
                                 ? options_.metrics_state_path
                                 : (!options_.request_log_jsonl.empty()
                                        ? options_.request_log_jsonl + ".metrics-state.json"
                                        : std::string());
    if (path.empty()) { return; }
    FILE* f = fopen(path.c_str(), "r");
    if (f == nullptr) { return; }
    std::string buf;
    char chunk[8192];
    std::size_t n = 0;
    while ((n = fread(chunk, 1, sizeof(chunk), f)) > 0) { buf.append(chunk, n); }
    fclose(f);
    try {
        const auto j = nlohmann::json::parse(buf);
        {
            std::lock_guard lock(energy_mutex_);
            for (auto it = j["daily_ws"].begin(); it != j["daily_ws"].end(); ++it) {
                energy.daily_ws[it.key()] = it.value().get<double>();
            }
            if (j.contains("daily_tokens")) {
                for (auto it = j["daily_tokens"].begin(); it != j["daily_tokens"].end(); ++it) {
                    energy.daily_tokens[it.key()] = it.value().get<std::uint64_t>();
                }
            }
            persisted_prompt_tokens = j.value("persisted_prompt_tokens", 0ULL);
            persisted_cached_tokens = j.value("persisted_cached_tokens", 0ULL);
            persisted_output_tokens = j.value("persisted_output_tokens", 0ULL);
        }
        std::fprintf(stderr, "[metrics] state restored from %s\n", path.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[metrics] state restore failed: %s\n", e.what());
    }
}

void HttpServer::record_energy(double tokens, double interval_s, double avg_watts) {
    if (interval_s <= 0.0) { return; }

    const std::chrono::time_point now_utc =
        std::chrono::time_point_cast<std::chrono::seconds>(
            std::chrono::system_clock::now());
    const std::time_t tt = std::chrono::system_clock::to_time_t(now_utc);
    std::tm utc{};
    gmtime_r(&tt, &utc);
    char day_key[16];
    std::snprintf(day_key, sizeof(day_key), "%04d-%02d-%02d", utc.tm_year + 1900,
                  utc.tm_mon + 1, utc.tm_mday);
    const double ws = avg_watts * interval_s;
    std::lock_guard lock(energy_mutex_);
    energy.daily_ws[day_key] += ws;
    energy.daily_tokens[day_key] += static_cast<std::uint64_t>(tokens);
    energy.tokens_total += static_cast<std::uint64_t>(tokens);
    if (energy.tokens_today_key != day_key) { energy.tokens_today = 0; energy.tokens_today_key = day_key; }
    energy.tokens_today += static_cast<std::uint64_t>(tokens);
    // rolling 30-day = sum of retained daily buckets (prune older than 31 days)
    static constexpr int kRetainDays = 31;
    if (energy.daily_ws.size() > static_cast<std::size_t>(kRetainDays)) {
        for (auto it = energy.daily_ws.begin();
             it != energy.daily_ws.end() &&
             static_cast<int>(energy.daily_ws.size()) > kRetainDays;) {
            it = energy.daily_ws.erase(it);
        }
    }
    double month = 0.0;
    for (const auto& [key, ws_val] : energy.daily_ws) { month += ws_val; }
    energy.month_ws = month;
}

void HttpServer::run_stats_reporter() {
    using Clock                     = std::chrono::steady_clock;
    ninfer::RuntimeStats previous   = service_->runtime_stats();
    Clock::time_point previous_time = Clock::now();
    const auto interval             = std::chrono::milliseconds(options_.log_stats_interval_ms);
    Clock::time_point next_deadline = previous_time + interval;
    GpuPowerSampler power_sampler;
    double window_watt_seconds = 0.0;
    double window_seconds_sum  = 0.0;
    int    window_samples      = 0;
    long   reporter_ticks      = 0;
    auto    window_started     = Clock::now();

    for (;;) {
        {
            std::unique_lock lock(stats_mutex_);
            if (stats_cv_.wait_until(lock, next_deadline, [this] { return stats_stopping_; })) {
                break;
            }
        }
        // sub-sample power twice per window (start and end); trapezoid over the pair
        power_sampler.sample();
        const unsigned int mw_start = power_sampler.last_milliwatts;

        const ninfer::RuntimeStats current = service_->runtime_stats();
        const Clock::time_point now        = Clock::now();
        power_sampler.sample();
        const unsigned int mw_end = power_sampler.last_milliwatts;
        const double window_s     = std::chrono::duration<double>(now - window_started).count();
        window_seconds_sum += window_s;
        window_watt_seconds += ((mw_start + mw_end) / 2.0 / 1000.0) * window_s;
        ++window_samples;
        window_started = now;
        const ThroughputReport report      = make_throughput_report(
            previous, current, std::chrono::duration<double>(now - previous_time).count());
        if (report_has_activity(report)) { record_throughput(report); }
        const double window_avg_watts =
            window_seconds_sum > 0.0 ? window_watt_seconds / window_seconds_sum : 0.0;
        record_energy(static_cast<double>(report.computed_prefill_tokens +
                                          report.committed_decode_tokens),
                      window_s, window_avg_watts);
        if (++reporter_ticks % 12 == 0) { persist_metrics_state(); }
        previous      = current;
        previous_time = now;
        next_deadline += interval;
        const Clock::time_point after_write = Clock::now();
        if (next_deadline <= after_write) { next_deadline = after_write + interval; }
    }

    const ninfer::RuntimeStats current = service_->runtime_stats();
    const Clock::time_point now        = Clock::now();
    const ThroughputReport tail        = make_throughput_report(
        previous, current, std::chrono::duration<double>(now - previous_time).count());
    // The exact partial interval remains useful to measurement consumers. Pretty throughput is a
    // fixed-cadence operational record and deliberately has no irregular shutdown tail.
    if (report_has_activity(tail)) { request_jsonl_.write_throughput(tail); }
}

void HttpServer::stop_stats_reporter() {
    if (!stats_thread_.joinable()) { return; }
    {
        std::lock_guard lock(stats_mutex_);
        stats_stopping_ = true;
    }
    stats_cv_.notify_one();
    stats_thread_.join();
    persist_metrics_state();
}

void HttpServer::register_routes() {
    server_.set_error_handler([this](const httplib::Request& request, httplib::Response& response) {
        return handle_unrendered_http_error(options_, request, response);
    });
    if (options_.enable_cors) {
        server_.set_default_headers(
            {{"Access-Control-Allow-Origin", "*"},
             {"Access-Control-Expose-Headers", "x-request-id, request-id"},
             {"Access-Control-Allow-Headers",
              "Authorization, Content-Type, X-API-Key, anthropic-version, anthropic-beta, "
              "anthropic-user-profile-id"},
             {"Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS"}});
        // CORS preflight: browsers send OPTIONS with no credentials before the real
        // request; answer it without auth so the actual GET/POST can carry the key.
        server_.Options(R"(.*)",
                        [](const httplib::Request&, httplib::Response& res) { res.status = 204; });
    }

    server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        ensure_openai_request_id(req, res);
        if (options_.api_key.empty() || req.path == "/health" || req.method == "OPTIONS") {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        // Accept both the OpenAI-style bearer token and the Anthropic-style
        // x-api-key header so OpenAI clients and Claude Code (ANTHROPIC_API_KEY
        // -> x-api-key, ANTHROPIC_AUTH_TOKEN -> Authorization: Bearer) both work.
        const bool bearer_ok =
            matches_bearer_credential(req.get_header_value("Authorization"), options_.api_key);
        const bool x_api_key_ok = req.get_header_value("x-api-key") == options_.api_key;
        if (!bearer_ok && !x_api_key_ok) {
            ApiError error;
            error.status  = 401;
            error.type    = "invalid_request_error";
            error.code    = "invalid_api_key";
            error.message = "missing or invalid API key";
            // Render the 401 in the shape the target endpoint speaks.
            if (req.path.rfind("/v1/messages", 0) == 0) {
                write_anthropic_error(res, error, new_anthropic_request_id());
            } else {
                write_openai_error(res, error);
            }
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    server_.set_exception_handler(
        [this](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
            ensure_openai_request_id(req, res);
            try {
                std::rethrow_exception(ep);
            } catch (const ApiException& e) {
                if (e.error().status >= 500) {
                    operational_log_.http_failure(
                        endpoint_name(req.path),
                        make_request_failure(RequestFailurePhase::Http, e.error()),
                        response_request_id(res));
                }
                if (req.path.rfind("/v1/messages", 0) == 0) {
                    write_anthropic_error(res, e.error(), new_anthropic_request_id());
                } else {
                    write_openai_error(res, e.error());
                }
            } catch (const std::exception& e) {
                operational_log_.http_failure(
                    endpoint_name(req.path),
                    make_internal_request_failure(RequestFailurePhase::Http, e.what()),
                    response_request_id(res));
                if (req.path.rfind("/v1/messages", 0) == 0) {
                    ApiError error;
                    error.status  = 500;
                    error.message = e.what();
                    write_anthropic_error(res, error, new_anthropic_request_id());
                } else {
                    write_exception(res, e);
                }
            } catch (...) {
                operational_log_.http_failure(
                    endpoint_name(req.path),
                    make_internal_request_failure(RequestFailurePhase::Http, "unknown error"),
                    response_request_id(res));
                ApiError error;
                error.status  = 500;
                error.type    = "internal_error";
                error.message = "unknown error";
                if (req.path.rfind("/v1/messages", 0) == 0) {
                    write_anthropic_error(res, error, new_anthropic_request_id());
                } else {
                    write_openai_error(res, error);
                }
            }
        });

    server_.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        const bool available = service_ != nullptr && service_->is_available();
        res.status           = available ? 200 : 503;
        res.set_content(nlohmann::json{{"status", available ? "ok" : "unavailable"}}.dump(),
                        "application/json");
    });
    server_.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
        handle_metrics(res);
    });
    server_.Get("/slots", [this](const httplib::Request&, httplib::Response& res) {
        handle_slots(res);
    });
    server_.Get("/usage", [this](const httplib::Request&, httplib::Response& res) {
        handle_usage(res);
    });
    server_.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res);
    });
    server_.Get(R"(/v1/models/(.+)/details)",
                [this](const httplib::Request& req, httplib::Response& res) {
                    const std::string id =
                        req.matches.size() > 1 ? req.matches[1].str() : std::string();
                    handle_model_details(id, res);
                });
    server_.Get(R"(/v1/models/(.+)/card)",
                [this](const httplib::Request& req, httplib::Response& res) {
                    const std::string id =
                        req.matches.size() > 1 ? req.matches[1].str() : std::string();
                    handle_model_card(id, res);
                });
    server_.Get(R"(/v1/models/(.+))", [this](const httplib::Request& req, httplib::Response& res) {
        handle_model(req, res);
    });
    server_.Post("/v1/chat/completions",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_chat_completions(req, res);
                 });
    server_.Post("/v1/responses", [this](const httplib::Request& req, httplib::Response& res) {
        handle_responses(req, res);
    });
    server_.Post("/v1/responses/input_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_input_tokens(req, res);
                 });
    server_.Post("/v1/responses/compact",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_compact(req, res);
                 });
    server_.Post(R"(/v1/responses/([^/]+)/cancel)",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_response_cancel(req, res);
                 });
    server_.Get(R"(/v1/responses/([^/]+)/input_items)",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_input_items(req, res);
                });
    server_.Get(R"(/v1/responses/([^/]+))",
                [this](const httplib::Request& req, httplib::Response& res) {
                    handle_response_get(req, res);
                });
    server_.Delete(R"(/v1/responses/([^/]+))",
                   [this](const httplib::Request& req, httplib::Response& res) {
                       handle_response_delete(req, res);
                   });
    server_.Post("/v1/messages/count_tokens",
                 [this](const httplib::Request& req, httplib::Response& res) {
                     handle_count_tokens(req, res);
                 });
    server_.Post("/v1/messages", [this](const httplib::Request& req, httplib::Response& res) {
        handle_messages(req, res);
    });
}

void HttpServer::handle_models(const httplib::Request&, httplib::Response& res) const {
    res.set_content(make_models_list(public_model_id_, unix_time_now(), options_.max_context),
                    "application/json");
}

void HttpServer::handle_model(const httplib::Request& req, httplib::Response& res) const {
    const std::string id = req.matches.size() > 1 ? req.matches[1].str() : std::string();
    if (id != public_model_id_) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.code    = "model_not_found";
        error.message = "model '" + id + "' not found";
        write_openai_error(res, error);
        return;
    }
    nlohmann::json payload = nlohmann::json::parse(make_model_object(
        public_model_id_, unix_time_now(), options_.max_context));
    // Non-spec extension: attach the configured model card inline (vLLM precedent — clients
    // ignore unknown members on the Model object).
    if (service_ != nullptr) {
        if (const nlohmann::json* card = service_->model_card();
            card != nullptr && !card->is_null()) {
            payload["model_card"] = *card;
        }
    }
    res.set_content(payload.dump(), "application/json");
}

void HttpServer::handle_model_card(const std::string& model_id, httplib::Response& res) const {
    if (model_id != public_model_id_) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.code    = "model_not_found";
        error.message = "model '" + model_id + "' not found";
        write_openai_error(res, error);
        return;
    }
    const nlohmann::json* card = service_ != nullptr ? service_->model_card() : nullptr;
    if (card == nullptr || card->is_null()) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.code    = "model_card_not_found";
        error.message = "no model card configured for this server (start with --model-card FILE)";
        write_openai_error(res, error);
        return;
    }
    res.set_content(card->dump(), "application/json");
}

namespace {

using Json = nlohmann::json;

const char* kv_cache_storage_name(ninfer::KvCacheStorage storage) noexcept {
    switch (storage) {
    case ninfer::KvCacheStorage::BFloat16:
        return "bf16";
    case ninfer::KvCacheStorage::Int8Group64:
        return "int8";
    case ninfer::KvCacheStorage::Fp8E4M3Row256:
        return "fp8";
    case ninfer::KvCacheStorage::Nvfp4Group16:
        return "nvfp4";
    case ninfer::KvCacheStorage::Fp8KeyNvfp4Value:
        return "k8v4";
    }
    return "unknown";
}

std::uint64_t meminfo_kib_field(const char* label) {
    std::FILE* file = std::fopen("/proc/meminfo", "r");
    if (file == nullptr) { return 0; }
    std::uint64_t value = 0;
    char line[256];
    while (std::fgets(line, sizeof(line), file) != nullptr) {
        if (std::strncmp(line, label, std::strlen(label)) == 0) {
            value = std::strtoull(line + std::strlen(label), nullptr, 10);
            break;
        }
    }
    std::fclose(file);
    return value * 1024U; // kB -> bytes
}

// Lazily dlopen'd NVML introspection for /details. Every field degrades to null when the
// library, the driver, or the requested device is unavailable — hardware reporting must never
// take the server down.
struct DeviceIntrospection {
    using HandleFn = int (*)(unsigned int, void**);
    using NameFn = int (*)(void*, char*, unsigned int);
    using ComputeFn = int (*)(void*, int*, int*);
    using MemoryFn = int (*)(void*, void*); // nvmlDeviceGetMemoryInfo(device, nvmlMemory_t*)
    using DriverFn = int (*)(char*, unsigned int);
    static constexpr unsigned int kNameBufferSize = 256;
    static constexpr unsigned int kVersionBufferSize = 96;

    void* nvml_handle    = nullptr;
    HandleFn handle_fn   = nullptr;
    NameFn name_fn       = nullptr;
    ComputeFn compute_fn = nullptr;
    MemoryFn memory_fn   = nullptr;
    DriverFn driver_fn   = nullptr;

    DeviceIntrospection() {
        nvml_handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
        if (nvml_handle == nullptr) { return; }
        auto init_fn = (int (*)())dlsym(nvml_handle, "nvmlInit_v2");
        handle_fn    = (HandleFn)dlsym(nvml_handle, "nvmlDeviceGetHandleByIndex_v2");
        name_fn      = (NameFn)dlsym(nvml_handle, "nvmlDeviceGetName");
        compute_fn   = (ComputeFn)dlsym(nvml_handle, "nvmlDeviceGetCudaComputeCapability");
        memory_fn    = (MemoryFn)dlsym(nvml_handle, "nvmlDeviceGetMemoryInfo");
        driver_fn    = (DriverFn)dlsym(nvml_handle, "nvmlSystemGetDriverVersion");
        if (init_fn == nullptr || handle_fn == nullptr) { return; }
        if (init_fn() != 0) { return; }
        initialized = true;
    }
    bool initialized = false;
};

Json gpu_report(int device_index) {
    static const DeviceIntrospection introspection;
    if (!introspection.initialized) { return Json(nullptr); }
    void* device = nullptr;
    if (introspection.handle_fn(static_cast<unsigned int>(device_index), &device) != 0) {
        return Json(nullptr);
    }
    Json report = Json::object();
    if (introspection.name_fn != nullptr) {
        char name[DeviceIntrospection::kNameBufferSize] = {};
        if (introspection.name_fn(device, name, sizeof(name)) == 0) {
            report["name"] = std::string(name);
        }
    }
    if (introspection.compute_fn != nullptr) {
        int major = 0, minor = 0;
        if (introspection.compute_fn(device, &major, &minor) == 0) {
            report["compute_capability"] = std::to_string(major) + "." + std::to_string(minor);
        }
    }
    if (introspection.memory_fn != nullptr) {
        // nvmlMemory_t layout (v1): unsigned long long total, free, used — stable ABI.
        struct NvmlMemory {
            unsigned long long total;
            unsigned long long free;
            unsigned long long used;
        } memory {};
        if (introspection.memory_fn(device, &memory) == 0) {
            report["vram_total_bytes"] = memory.total;
            report["vram_free_bytes"]  = memory.free;
            report["vram_used_bytes"]  = memory.used;
        }
    }
    return report.empty() ? Json(nullptr) : Json(report);
}

std::string nvml_driver_version() {
    static const DeviceIntrospection introspection;
    if (!introspection.initialized || introspection.driver_fn == nullptr) { return {}; }
    char version[DeviceIntrospection::kVersionBufferSize] = {};
    if (introspection.driver_fn(version, sizeof(version)) == 0) { return std::string(version); }
    return {};
}

Json hardware_report(int device_index) {
    Json host = Json::object();
    struct utsname system_info {};
    if (uname(&system_info) == 0) {
        host["kernel"]       = std::string(system_info.release);
        host["architecture"] = std::string(system_info.machine);
    }
    host["cpu_cores"]           = static_cast<std::int64_t>(sysconf(_SC_NPROCESSORS_ONLN));
    host["ram_total_bytes"]     = meminfo_kib_field("MemTotal:");
    host["ram_available_bytes"] = meminfo_kib_field("MemAvailable:");

    std::string driver = nvml_driver_version();
    Json hardware = {{"gpu", gpu_report(device_index)}, {"host", std::move(host)}};
    hardware["driver_version"] = driver.empty() ? Json(nullptr) : Json(driver);
    return hardware;
}

} // namespace

void HttpServer::handle_model_details(const std::string& model_id, httplib::Response& res) const {
    if (service_ == nullptr) {
        res.status = 503;
        res.set_content(R"({"error":"service not attached"})", "application/json");
        return;
    }
    if (model_id != public_model_id_) {
        ApiError error;
        error.status  = 404;
        error.type    = "invalid_request_error";
        error.code    = "model_not_found";
        error.message = "model '" + model_id + "' not found";
        write_openai_error(res, error);
        return;
    }

    Json payload = {{"id", public_model_id_},
                    {"object", "model"},
                    {"created", unix_time_now()},
                    {"owned_by", "ninfer"},
                    {"status", service_->is_available() ? Json("ready") : Json("loading")}};
    if (const nlohmann::json* card = service_->model_card();
        card != nullptr && !card->is_null()) {
        payload["model_card"] = *card;
    }

    Json server = {{"engine", "ninfer"},
                   {"max_context_len", options_.max_context},
                   {"kv_cache_dtype", kv_cache_storage_name(options_.kv_cache)},
                   {"max_concurrency", options_.max_concurrency},
                   {"max_pending_requests", options_.max_pending_requests},
                   {"prefill_chunk", options_.prefill_chunk},
                   {"enable_vision", options_.enable_vision},
                   {"cuda_graphs", options_.use_cuda_graph},
                   {"prefix_reuse", options_.allow_prefix_reuse},
                   {"auto_long_anchors", options_.auto_long_anchors},
                   {"electricity_rate_usd_per_kwh", options_.electricity_rate_usd_per_kwh}};
    if (options_.enable_thinking.has_value()) {
        server["enable_thinking"] = *options_.enable_thinking;
    }
    if (options_.preserve_thinking.has_value()) {
        server["preserve_thinking"] = *options_.preserve_thinking;
    }
    if (options_.default_thinking_budget.has_value()) {
        server["default_thinking_budget"] = *options_.default_thinking_budget;
    }
    if (options_.speculative.backend != SpeculativeBackend::None) {
        server["speculative"] = {
            {"backend", product::speculative_backend_name(options_.speculative.backend)},
            {"draft_tokens", options_.speculative.draft_tokens}};
    }
    server["context_cache"] = {
        {"device_state_slots", service_->engine_options().context_cache.device_state_slots},
        {"host_state_slots", service_->engine_options().context_cache.host_state_slots},
        {"host_kv_bytes", service_->engine_options().context_cache.host_kv_capacity_bytes},
        {"max_private_continuations",
         service_->engine_options().context_cache.max_private_continuations},
        {"max_shared_prefixes", service_->engine_options().context_cache.max_shared_prefixes}};
    int driver_version = 0, runtime_version = 0;
    (void)cudaDriverGetVersion(&driver_version);
    (void)cudaRuntimeGetVersion(&runtime_version);
    server["cuda_driver_version"]  = driver_version;
    server["cuda_runtime_version"] = runtime_version;
    payload["server"] = std::move(server);

    const ninfer::LoadSummary load = service_->load_summary();
    Json load_json = {{"architecture", load.architecture},
                      {"model_name", load.model_name},
                      {"load_seconds", load.load_seconds},
                      {"upload_seconds", load.upload_seconds},
                      {"artifact_bytes_read", load.artifact_bytes_read},
                      {"host_to_device_bytes", load.host_to_device_bytes},
                      {"device_object_count", load.device_object_count},
                      {"prefill_signature", load.prefill_signature}};
    load_json["weight_formats"] = load.weight_formats;
    payload["load"] = std::move(load_json);

    payload["hardware"] = hardware_report(options_.device);
    res.set_content(payload.dump(), "application/json");
}

void HttpServer::handle_metrics(httplib::Response& res) const {
    using Clock  = std::chrono::steady_clock;
    static const Clock::time_point process_start = Clock::now();
    struct CounterDef {
        const char* name;
        const char* help;
        std::uint64_t value;
    };
    std::string out;
    auto emit_u64 = [&](const char* name, const char* help, std::uint64_t value,
                        const char* labels = "") {
        out += std::string("# HELP ") + name + ' ' + help + "\n# TYPE " + name + " counter\n" +
               name + labels + ' ' + std::to_string(value) + "\n";
    };
    auto emit_gauge = [&](const char* name, const char* help, std::uint64_t value) {
        out += std::string("# HELP ") + name + ' ' + help + "\n# TYPE " + name + " gauge\n" +
               name + ' ' + std::to_string(value) + "\n";
    };
    auto emit_dbl = [&](const char* name, const char* help, double value) {
        out += std::string("# HELP ") + name + ' ' + help + "\n# TYPE " + name + " gauge\n" +
               name + ' ' + std::to_string(value) + "\n";
    };

    if (service_ == nullptr) {
        res.status = 503;
        res.set_content(out, "text/plain; version=0.0.4");
        return;
    }
    const ninfer::RuntimeStats stats = service_->runtime_stats();
    const ninfer::LoadSummary load   = service_->load_summary();
    // Fresh 5s-window rates from the stats reporter (0 when idle — lifetime averages
    // over full uptime are meaningless on a bursty server).
    double window_prefill_tps = 0.0;
    double window_decode_tps  = 0.0;
    bool   window_fresh       = false;
    {
        std::lock_guard lock(last_throughput_mutex_);
        const double age_s = std::chrono::duration<double>(
                                 Clock::now() - last_throughput_at_).count();
        window_fresh = age_s < 10.0 && last_throughput_.interval_seconds > 0.0;
        if (window_fresh) {
            window_prefill_tps = static_cast<double>(last_throughput_.computed_prefill_tokens) /
                                 last_throughput_.interval_seconds;
            window_decode_tps = static_cast<double>(last_throughput_.committed_decode_tokens) /
                                last_throughput_.interval_seconds;
        }
    }

    emit_u64("llamacpp:prompt_tokens_total",
             "Total prompt tokens computed by prefill (cache hits excluded).",
             stats.computed_prefill_tokens);
    emit_u64("llamacpp:prompt_seconds_total", "Total wall seconds spent computing prefill.",
             0);
    emit_u64("llamacpp:tokens_predicted_total",
             "Total output tokens committed by decode.", stats.committed_decode_tokens);
    emit_u64("llamacpp:tokens_predicted_seconds_total",
             "Total wall seconds spent decoding.", 0);
    emit_gauge("llamacpp:requests_processing", "Requests currently running or prefilling.",
               static_cast<std::uint64_t>(stats.running_requests));
    emit_gauge("llamacpp:prompt_tokens_seconds", "Recent-window prefill throughput (tok/s).",
               window_prefill_tps);
    emit_gauge("llamacpp:predicted_tokens_seconds", "Recent-window decode throughput (tok/s).",
               window_decode_tps);
    emit_gauge("ninfer:throughput_window_fresh", "1 if the throughput window saw activity.",
               window_fresh ? 1 : 0);
    emit_gauge("llamacpp:requests_deferred", "Requests waiting in the admission queue.",
               static_cast<std::uint64_t>(stats.waiting_requests));
    emit_gauge("llamacpp:prompt_cache_tokens_total",
               "Prompt tokens served from checkpoint reuse (all reuse paths).",
               stats.reused_prompt_tokens);

    emit_u64("ninfer:requests_total", "Total generation requests admitted.",
             stats.root_selections + stats.private_endpoint_selections +
                 stats.private_turn_closure_selections +
                 stats.private_response_replay_selections +
                 stats.private_long_anchor_selections +
                 stats.shared_stable_prefix_selections);
    emit_u64("ninfer:decode_rounds_total", "Decode batch executions.", stats.decode_rounds);
    emit_u64("ninfer:decode_row_rounds_total",
             "Sum of per-row decode executions across batches.", stats.decode_row_rounds);
    emit_u64("ninfer:prefill_units_total", "Prefill compact-batch units executed.",
             stats.host_work.prefill_units);
    emit_u64("ninfer:prompt_cache_root_selections_total",
             "Requests whose reuse candidate was the shared root (cold prefill).",
             stats.root_selections);
    emit_u64("ninfer:prompt_cache_private_endpoint_selections_total",
             "Requests restored from their session endpoint (pure appends).",
             stats.private_endpoint_selections);
    emit_u64("ninfer:prompt_cache_turn_closure_selections_total",
             "Requests restored from a turn-closure checkpoint.",
             stats.private_turn_closure_selections);
    emit_u64("ninfer:prompt_cache_response_replay_selections_total",
             "Requests restored by response replay.", stats.private_response_replay_selections);
    emit_u64("ninfer:prompt_cache_long_anchor_selections_total",
             "Requests restored from a private long anchor (history rewrite).",
             stats.private_long_anchor_selections);
    emit_u64("ninfer:prompt_cache_shared_prefix_selections_total",
             "Requests reading a published shared prefix.",
             stats.shared_stable_prefix_selections);
    emit_gauge("ninfer:prefilling_requests", "Requests currently computing prefill.",
               stats.prefilling_requests);
    emit_gauge("ninfer:decode_ready_requests", "Requests holding a decode lane.",
               stats.decode_ready_requests);
    emit_gauge("ninfer:materializing_requests", "Requests restoring checkpoints.",
               stats.materializing_requests);
    emit_gauge("ninfer:capture_pending_requests", "Requests waiting on checkpoint capture.",
               stats.capture_pending_requests);
    emit_gauge("ninfer:device_state_occupied_slots", "Occupied device state slots.",
               stats.device_state_occupied_slots);
    emit_gauge("ninfer:host_state_occupied_slots", "Occupied host state slots.",
               stats.host_state_occupied_slots);
    emit_gauge("ninfer:device_main_kv_occupied_pages", "Occupied main KV pages on device.",
               stats.device_main_kv_occupied_pages);
    emit_u64("ninfer:active_captures_completed_total",
             "Checkpoint captures completed.", stats.active_captures_completed);
    emit_u64("ninfer:active_captures_aborted_total",
             "Checkpoint captures aborted.", stats.active_captures_aborted);
    emit_u64("ninfer:pressure_spill_pages_total", "KV pages spilled under pressure.",
             stats.pressure_spill_pages);
    emit_u64("ninfer:pressure_private_owners_evicted_total",
             "Private continuations evicted under pressure.",
             stats.pressure_private_owners_evicted);
    emit_u64("ninfer:disk_restores_total", "Prefills resumed from the NVMe context archive.",
             stats.disk_restores);
    emit_u64("ninfer:disk_restored_tokens_total",
             "Prompt tokens restored from the NVMe context archive instead of computed.",
             stats.disk_restored_tokens);
    emit_u64("ninfer:disk_restore_failures_total",
             "NVMe archive candidates that could not be read or verified.",
             stats.disk_restore_failures);
    emit_dbl("ninfer:disk_read_seconds_total",
             "Wall seconds spent reading and verifying archived checkpoints.",
             stats.disk_read_seconds);
    emit_dbl("ninfer:disk_restore_copy_seconds_total",
             "Worker seconds spent copying restored checkpoints to the device.",
             stats.disk_restore_seconds);
    emit_u64("ninfer:disk_saves_total", "Checkpoints archived to NVMe.", stats.disk_saves);
    emit_u64("ninfer:disk_saves_skipped_total",
             "Checkpoint archives skipped by the write budget or staging capacity.",
             stats.disk_saves_skipped);
    emit_u64("ninfer:disk_bytes_written_total", "Bytes written to the NVMe context archive.",
             stats.disk_bytes_written);
    emit_u64("ninfer:disk_bytes_read_total", "Bytes read from the NVMe context archive.",
             stats.disk_bytes_read);
    emit_gauge("ninfer:disk_stored_bytes", "Bytes held by the NVMe context archive.",
               stats.disk_stored_bytes);
    emit_gauge("ninfer:disk_checkpoints", "Checkpoints held by the NVMe context archive.",
               stats.disk_manifests);
    emit_dbl("ninfer:process_uptime_seconds", "Seconds since the stats endpoint was initialized.",
             std::chrono::duration<double>(Clock::now() - process_start).count());

    res.set_header("Cache-Control", "no-store");
    res.set_content(out, "text/plain; version=0.0.4");
}

void HttpServer::handle_slots(httplib::Response& res) const {
    if (service_ == nullptr) {
        res.status = 503;
        res.set_content(R"({"error":"service not attached"})", "application/json");
        return;
    }
    const ninfer::RuntimeStats stats = service_->runtime_stats();
    const std::uint32_t max_lanes    = service_->engine_options().max_concurrency;
    nlohmann::json j                 = nlohmann::json::array();
    for (std::uint32_t lane = 0; lane < max_lanes; ++lane) {
        j.push_back(nlohmann::json{
            {"id", lane},
            {"busy", lane < stats.running_requests},
        });
    }
    nlohmann::json out{
        {"model", public_model_id_},
        {"max_concurrency", max_lanes},
        {"requests_processing", stats.running_requests},
        {"requests_waiting", stats.waiting_requests},
        {"prefilling", stats.prefilling_requests},
        {"decode_ready", stats.decode_ready_requests},
        {"materializing", stats.materializing_requests},
        {"device_state_slots", {{"occupied", stats.device_state_occupied_slots}}},
        {"host_state_slots", {{"occupied", stats.host_state_occupied_slots}}},
        {"kv_pages",
         {{"main_occupied", stats.device_main_kv_occupied_pages},
          {"backend_occupied", stats.device_backend_kv_occupied_pages}}},
        {"slots", std::move(j)},
    };
    res.set_content(out.dump(), "application/json");
}

void HttpServer::handle_usage(httplib::Response& res) const {
    using Clock  = std::chrono::steady_clock;
    static const Clock::time_point process_start = Clock::now();
    if (service_ == nullptr) {
        res.status = 503;
        res.set_content(R"({"error":"service not attached"})", "application/json");
        return;
    }
    const ninfer::RuntimeStats stats = service_->runtime_stats();
    const double uptime_s =
        std::chrono::duration<double>(Clock::now() - process_start).count();

    // Copy everything under short lock scopes; build the response afterwards.
    std::uint64_t pt_total = 0, pt_cache = 0, pt_out = 0;
    std::map<std::string, double> daily_ws_copy;
    double month_ws = 0.0;
    {
        std::lock_guard lock(energy_mutex_);
        pt_total  = persisted_prompt_tokens + persisted_cached_tokens +
                    stats.computed_prefill_tokens + stats.reused_prompt_tokens;
        pt_cache  = persisted_cached_tokens + stats.reused_prompt_tokens;
        pt_out    = persisted_output_tokens + stats.committed_decode_tokens;
        daily_ws_copy = energy.daily_ws;
        month_ws  = energy.month_ws;
    }

    const double hit_rate = pt_total ? 100.0 * pt_cache / pt_total : 0.0;
    double cache_efficiency = 0.0;
    {
        const double denom = static_cast<double>(stats.root_selections +
            stats.private_endpoint_selections + stats.private_turn_closure_selections +
            stats.private_response_replay_selections + stats.private_long_anchor_selections +
            stats.shared_stable_prefix_selections);
        if (denom > 0) {
            cache_efficiency = 100.0 * (denom - stats.root_selections) / denom;
        }
    }

    // Most recent 5s stats-reporter window.
    double window_prefill_tps = 0.0, window_decode_tps = 0.0, window_seconds = 0.0;
    std::uint64_t window_prefill_tokens = 0, window_decode_tokens = 0;
    bool window_active = false;
    {
        std::lock_guard lock(last_throughput_mutex_);
        window_seconds       = last_throughput_.interval_seconds;
        window_prefill_tokens = last_throughput_.computed_prefill_tokens;
        window_decode_tokens  = last_throughput_.committed_decode_tokens;
        window_active        = window_seconds > 0.0;
        if (window_active) {
            window_prefill_tps = window_prefill_tokens / window_seconds;
            window_decode_tps  = window_decode_tokens / window_seconds;
        }
    }

    const int hours   = static_cast<int>(uptime_s / 3600);
    const int minutes = static_cast<int>(uptime_s / 60) % 60;

    std::uint64_t tokens_today = 0;
    double tokens_30d = 0.0;
    std::map<std::string, std::uint64_t> daily_tokens_copy;
    {
        std::lock_guard lock(energy_mutex_);
        tokens_today = energy.tokens_today;
        daily_tokens_copy = energy.daily_tokens;
        for (const auto& [key, count] : energy.daily_tokens) {
            tokens_30d += static_cast<double>(count);
        }
    }
    nlohmann::json days = nlohmann::json::object();
    for (const auto& [key, ws] : daily_ws_copy) {
        const std::uint64_t day_tokens =
            daily_tokens_copy.count(key) ? daily_tokens_copy.at(key) : 0;
        const double day_cost = ws / 3.6e6 * options_.electricity_rate_usd_per_kwh;
        days[key] = {{"wh", ws / 3600.0},
                     {"kwh", ws / 3.6e6},
                     {"cost_usd", day_cost},
                     {"tokens", day_tokens},
                     {"cost_per_m_tokens_usd",
                      day_tokens ? day_cost / static_cast<double>(day_tokens) * 1e6 : 0.0}};
    }
    const double now_epoch = static_cast<double>(std::time(nullptr));
    const std::time_t tt = static_cast<std::time_t>(now_epoch);
    std::tm utc{};
    gmtime_r(&tt, &utc);
    char today_key[16];
    std::snprintf(today_key, sizeof(today_key), "%04d-%02d-%02d",
                  utc.tm_year + 1900, utc.tm_mon + 1, utc.tm_mday);
    const double today_ws = daily_ws_copy.count(today_key) ? daily_ws_copy.at(today_key) : 0.0;
    const double today_cost = today_ws / 3.6e6 * options_.electricity_rate_usd_per_kwh;
    const double month_cost = month_ws / 3.6e6 * options_.electricity_rate_usd_per_kwh;

    nlohmann::json out{
        {"model", public_model_id_},
        {"uptime", {{"seconds", uptime_s},
                    {"human", std::to_string(hours) + "h " + std::to_string(minutes) + "m"}}},
        {"tokens",
         {{"input",
           {{"total", pt_total},
            {"cache_hits", pt_cache},
            {"cache_computed", pt_total - pt_cache},
            {"cache_hit_rate_pct", hit_rate}}},
          {"output", {{"total", pt_out}}},
          {"total", pt_total + pt_out}}},
        {"throughput",
         {{"window_seconds", window_seconds},
          {"window_active", window_active},
          {"prefill_tok_per_s", window_prefill_tps},
          {"decode_tok_per_s", window_decode_tps},
          {"prefill_tokens_last_window", window_prefill_tokens},
          {"decode_tokens_last_window", window_decode_tokens}}},
        {"requests",
         {{"total", stats.root_selections + stats.private_endpoint_selections +
                        stats.private_turn_closure_selections +
                        stats.private_response_replay_selections +
                        stats.private_long_anchor_selections +
                        stats.shared_stable_prefix_selections},
          {"cache_miss_cold_starts", stats.root_selections},
          {"cache_hit_endpoint", stats.private_endpoint_selections},
          {"cache_hit_turn_closure", stats.private_turn_closure_selections},
          {"cache_hit_response_replay", stats.private_response_replay_selections},
          {"cache_hit_long_anchor", stats.private_long_anchor_selections},
          {"cache_hit_shared_prefix", stats.shared_stable_prefix_selections},
          {"cache_efficiency_pct", cache_efficiency}}},
        {"lanes",
         {{"capacity", service_->engine_options().max_concurrency},
          {"processing", stats.running_requests},
          {"waiting", stats.waiting_requests},
          {"decoding", stats.decode_ready_requests},
          {"prefilling", stats.prefilling_requests},
          {"restoring_checkpoints", stats.materializing_requests}}},
        {"health",
         {{"checkpoint_captures_completed", stats.active_captures_completed},
          {"checkpoint_captures_aborted", stats.active_captures_aborted},
          {"kv_pressure_spills", stats.pressure_spill_pages},
          {"sessions_evicted", stats.pressure_private_owners_evicted},
          {"kv_pages_occupied", stats.device_main_kv_occupied_pages}}},
        {"energy",
         {{"rate_usd_per_kwh", options_.electricity_rate_usd_per_kwh},
          {"today",
           {{"wh", today_ws / 3600.0},
            {"kwh", today_ws / 3.6e6},
            {"cost_usd", today_cost},
            {"tokens", tokens_today},
            {"cost_per_m_tokens_usd",
             tokens_today ? today_cost / tokens_today * 1e6 : 0.0}}},
          {"rolling_30d",
           {{"wh", month_ws / 3600.0},
            {"kwh", month_ws / 3.6e6},
            {"cost_usd", month_cost},
            {"tokens", static_cast<std::uint64_t>(tokens_30d)},
            {"cost_per_m_tokens_usd",
             tokens_30d ? month_cost / tokens_30d * 1e6 : 0.0}}},
          {"daily", std::move(days)},
          {"note", "Measures the NVML device this process runs on (device 0). "
                   "Excludes host components (CPU, RAM, cooling) and any other "
                   "GPUs. Sampled via NVML at stats-reporter cadence. "
                   "cost_per_m_tokens_usd is blended (input+output combined); "
                   "the true split depends on cache hit rate and output share."}}},
    };
    res.set_header("Cache-Control", "no-store");
    res.set_content(out.dump(2), "application/json");
}

bool HttpServer::bind() { return server_.bind_to_port(options_.host, options_.port); }

void HttpServer::attach(GenerationService& service) {
    if (service_ != nullptr) {
        throw std::logic_error("HTTP generation service is already attached");
    }
    const ninfer::LoadSummary load = service.load_summary();
    public_model_id_               = resolve_public_model_id(options_, load.model_name);
    service_                       = &service;
    request_jsonl_.write_server_start(options_, service.engine_options(),
                                      service.sampling_defaults(), public_model_id_, load,
                                      service.memory_summary());
    restore_metrics_state();
}

bool HttpServer::listen() {
    if (service_ == nullptr) { throw std::logic_error("HTTP generation service is not attached"); }
    if (public_model_id_.empty()) {
        throw std::logic_error("HTTP public model id is not resolved");
    }
    if (options_.log_stats_interval_ms != 0) {
        stats_stopping_ = false;
        stats_thread_   = std::thread([this] { run_stats_reporter(); });
    }
    try {
        const bool result = server_.listen_after_bind();
        stop_stats_reporter();
        return result;
    } catch (...) {
        stop_stats_reporter();
        throw;
    }
}

void HttpServer::stop() { server_.stop(); }

} // namespace ninfer::serve
