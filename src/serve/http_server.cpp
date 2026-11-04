#include "serve/http_server.h"

#include "serve/anthropic_messages.h"
#include "serve/http_transport.h"
#include "serve/openai_common.h"
#include "serve/request_log.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <iterator>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <dlfcn.h>
#include <time.h>

namespace ninfer::serve {
namespace {

// NVML dynamic loading for GPU power sampling. Loaded once; sampling is a no-op returning 0
// when the library or device is unavailable, so energy accounting reports unavailable rather
// than wrong numbers. Probes handles 0..3 because the CUDA-order match is unavailable here.
struct GpuPowerSampler {
    using PowerFn = int (*)(void*, unsigned int*); // nvmlDeviceGetPowerUsage — milliwatts
    using HandleFn = int (*)(unsigned int, void*); // nvmlDeviceGetHandleByIndex_v2
    PowerFn query_power = nullptr;
    HandleFn handle_fn = nullptr;
    void* nvml_handle = nullptr;
    void* device = nullptr;
    bool initialized = false;
    unsigned int last_milliwatts = 0;

    GpuPowerSampler() {
        nvml_handle = dlopen("libnvidia-ml.so.1", RTLD_LAZY);
        if (nvml_handle == nullptr) { return; }
        init_fn = reinterpret_cast<int (*)()>(dlsym(nvml_handle, "nvmlInit_v2"));
        handle_fn = reinterpret_cast<HandleFn>(dlsym(nvml_handle, "nvmlDeviceGetHandleByIndex_v2"));
        query_power = reinterpret_cast<PowerFn>(dlsym(nvml_handle, "nvmlDeviceGetPowerUsage"));
        if (init_fn == nullptr || handle_fn == nullptr || query_power == nullptr) { return; }
        if (init_fn() != 0) { return; }
        int rc = 1;
        for (unsigned int idx = 0; idx < 4 && rc != 0; ++idx) {
            rc = handle_fn(idx, &device);
        }
        initialized = rc == 0;
    }
    ~GpuPowerSampler() = default;
    void sample() {
        if (!initialized) { return; }
        unsigned int mw = 0;
        if (query_power(device, &mw) == 0) { last_milliwatts = mw; }
    }

private:
    int (*init_fn)() = nullptr;
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

void ensure_http_request_id(const httplib::Request& request, httplib::Response& response) {
    if (is_anthropic_path(request.path) && !response.has_header("request-id")) {
        response.set_header("request-id", new_anthropic_request_id());
    } else if (is_openai_path(request.path) && !response.has_header("x-request-id")) {
        response.set_header("x-request-id", new_openai_request_id());
    }
}

ThroughputReport make_throughput_report(const ninfer::EngineSnapshot& before,
                                        const ninfer::EngineSnapshot& after,
                                        double interval_seconds) {
    const auto& previous = before.runtime;
    const auto& current  = after.runtime;
    return ThroughputReport{
        .previous_execution = before.executions,
        .current_execution  = after.executions,
        .interval_seconds   = interval_seconds,
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
           report.current.waiting_requests != 0 || report.current.paused_requests != 0 ||
           report.current.replaying_requests != 0 || report.current.materializing_requests != 0 ||
           report.current.capture_pending_requests != 0 ||
           report.current.terminal_pending_requests != 0 ||
           report.current.active_captures_completed != report.previous.active_captures_completed ||
           report.current.active_captures_aborted != report.previous.active_captures_aborted ||
           report.current.preemptions != report.previous.preemptions ||
           report.current.snapshot_restores != report.previous.snapshot_restores ||
           report.current.replay_restores != report.previous.replay_restores ||
           report.current.replayed_tokens != report.previous.replayed_tokens ||
           report.current.root_selections != report.previous.root_selections ||
           report.current.checkpoint_selections != report.previous.checkpoint_selections ||
           report.current.state_moves != report.previous.state_moves ||
           report.current.state_forks != report.previous.state_forks ||
           report.current.materialization_state_forks !=
               report.previous.materialization_state_forks ||
           report.current.state_restores != report.previous.state_restores ||
           report.current.context_transfers != report.previous.context_transfers ||
           report.current.pressure_spill_pages != report.previous.pressure_spill_pages ||
           report.current.partial_tail_cow_pages != report.previous.partial_tail_cow_pages ||
           report.current.state_objects != report.previous.state_objects ||
           report.current.history_objects != report.previous.history_objects ||
           report.current.kv_pools != report.previous.kv_pools ||
           report.current.context_pools != report.previous.context_pools ||
           report.current.context_metadata != report.previous.context_metadata ||
           report.current.host_work.engine_boundary_ns !=
               report.previous.host_work.engine_boundary_ns ||
           report.current.host_work.session_call_ns != report.previous.host_work.session_call_ns ||
           report.current.host_work.executor_poll_ns !=
               report.previous.host_work.executor_poll_ns ||
           report.current.host_work.engine_commit_output_ns !=
               report.previous.host_work.engine_commit_output_ns ||
           report.current.host_work.engine_maintenance_ns !=
               report.previous.host_work.engine_maintenance_ns ||
           report.current.host_work.scheduler_yield_ns !=
               report.previous.host_work.scheduler_yield_ns;
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

httplib::Server::HandlerResponse handle_unrendered_http_error(const product::ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response) {
    ensure_http_request_id(request, response);
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
        write_anthropic_error(response, error, response.get_header_value("request-id"));
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

HttpServer::HttpServer(product::ServeOptions options, std::shared_ptr<spdlog::logger> logger)
    : options_(std::move(options)), openai_responses_store_(options_.response_store_max_records,
                                                            options_.response_store_max_bytes),
      operational_log_(logger),
      request_jsonl_(observations_.instance_id(), options_.launch.logging.request_jsonl.string(),
                     options_.launch.engine.artifact_path.string(), std::move(logger)) {
    request_jsonl_.write_startup_config(options_);
    const std::size_t queued_requests =
        static_cast<std::size_t>(options_.launch.engine.max_concurrency) +
        options_.launch.engine.max_pending_requests;
    const std::size_t worker_count = queued_requests + 1 + MonitorEvents::kMaxSubscribers;
    server_.new_task_queue         = [queued_requests, worker_count] {
        return make_http_task_queue(worker_count, queued_requests).release();
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
    observations_.prepared(context);
    request_jsonl_.write_request_start(context);
    operational_log_.request_start(context);
}

void HttpServer::record_request_rejected(const RequestRejectionLogContext& context) {
    observations_.rejected(context.id);
    request_jsonl_.write_request_rejected(context);
    operational_log_.request_rejected(context);
}

ninfer::GenerationFirstTokenObserver HttpServer::first_token_observer(std::uint64_t request_id) {
    return [this, request_id](const ninfer::GenerationFirstTokenObservation& observation) {
        observations_.first_token(request_id, observation);
    };
}

ninfer::GenerationSchedulingObserver HttpServer::scheduling_observer(std::uint64_t request_id,
                                                                     std::string http_request_id) {
    return [this, request_id, http_request_id = std::move(http_request_id)](
               const ninfer::GenerationSchedulingObservation& observation) {
        observations_.scheduling(request_id, observation);
        request_jsonl_.write_request_scheduling(request_id, http_request_id, observation);
    };
}

void HttpServer::record_request_done(const RequestLogContext& context,
                                     const GenerationOutcome& outcome) {
    observations_.done(context.id, outcome);
    request_jsonl_.write_request_done(context, outcome);
    operational_log_.request_done(context, outcome);
}

void HttpServer::record_request_failure(const RequestLogContext& context,
                                        const RequestFailure& failure) {
    observations_.failed(context.id,
                         failure.classification == RequestFailureClass::ClientDisconnected);
    request_jsonl_.write_request_error(context, failure.machine_message);
    operational_log_.request_failure(context, failure);
}

void HttpServer::record_response_failure(std::uint64_t request_id, const RequestFailure& failure) {
    observations_.response_failed(request_id);
    operational_log_.response_failure(request_id, failure);
}

void HttpServer::record_throughput(const ThroughputReport& report) {
    request_jsonl_.write_throughput(report);
    operational_log_.throughput(report);
}

std::string metrics_state_path_for(const product::ServeOptions& options) {
    if (!options.launch.logging.request_jsonl.empty()) {
        return options.launch.logging.request_jsonl.string() + ".metrics-state.json";
    }
    return {};
}

void HttpServer::persist_metrics_state() {
    const std::string path = metrics_state_path_for(options_);
    if (path.empty()) { return; }
    nlohmann::json daily_tokens_json = nlohmann::json::object();
    {
        std::lock_guard lock(energy_mutex_);
        for (const auto& [key, count] : energy.daily_tokens) {
            daily_tokens_json[key] = count;
        }
        const nlohmann::json j{
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
}

void HttpServer::restore_metrics_state() {
    const std::string path = metrics_state_path_for(options_);
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
        std::fprintf(stderr, "[metrics] state restored from %s\n", path.c_str());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[metrics] state restore failed: %s\n", e.what());
    }
}

void HttpServer::record_energy(double tokens, double interval_s, double avg_watts) {
    if (interval_s <= 0.0) { return; }
    const std::time_t tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
    gmtime_r(&tt, &utc);
    char day_key[16];
    std::snprintf(day_key, sizeof(day_key), "%04d-%02d-%02d", utc.tm_year + 1900, utc.tm_mon + 1,
                  utc.tm_mday);
    std::lock_guard lock(energy_mutex_);
    energy.daily_ws[day_key] += avg_watts * interval_s;
    energy.daily_tokens[day_key] += static_cast<std::uint64_t>(tokens);
    energy.tokens_total += static_cast<std::uint64_t>(tokens);
    if (energy.tokens_today_key != day_key) {
        energy.tokens_today = 0;
        energy.tokens_today_key = day_key;
    }
    energy.tokens_today += static_cast<std::uint64_t>(tokens);
    // Rolling 30-day total = retained daily buckets; prune older than 31 days.
    static constexpr std::size_t kRetainDays = 31;
    while (energy.daily_ws.size() > kRetainDays) {
        energy.daily_ws.erase(energy.daily_ws.begin());
    }
    double month = 0.0;
    for (const auto& [key, ws] : energy.daily_ws) { month += ws; }
    energy.month_ws = month;
}

void HttpServer::run_stats_reporter() {
    using Clock                     = std::chrono::steady_clock;
    ninfer::EngineSnapshot previous = service_->snapshot();
    Clock::time_point previous_time = Clock::now();
    const auto interval = std::chrono::milliseconds(options_.launch.logging.stats_interval_ms);
    Clock::time_point next_deadline = previous_time + interval;
    restore_metrics_state();

    GpuPowerSampler power_sampler;
    double window_watt_seconds = 0.0;
    double window_seconds_sum  = 0.0;
    long   reporter_ticks      = 0;
    auto   window_started      = Clock::now();

    for (;;) {
        {
            std::unique_lock lock(stats_mutex_);
            if (stats_cv_.wait_until(lock, next_deadline, [this] { return stats_stopping_; })) {
                break;
            }
        }

        const ninfer::EngineSnapshot current = service_->snapshot();
        const Clock::time_point now          = Clock::now();
        // Trapezoid power over the window: sample at both ends, average the pair.
        power_sampler.sample();
        const unsigned int mw_start = power_sampler.last_milliwatts;
        const ThroughputReport report        = make_throughput_report(
            previous, current, std::chrono::duration<double>(now - previous_time).count());
        if (report_has_activity(report)) { record_throughput(report); }
        power_sampler.sample();
        const unsigned int mw_end = power_sampler.last_milliwatts;
        const double window_s =
            std::chrono::duration<double>(now - window_started).count();
        window_seconds_sum += window_s;
        window_watt_seconds += ((mw_start + mw_end) / 2.0 / 1000.0) * window_s;
        ++reporter_ticks;
        window_started = now;
        const double window_avg_watts =
            window_seconds_sum > 0.0 ? window_watt_seconds / window_seconds_sum : 0.0;
        record_energy(
            static_cast<double>(report.computed_prefill_tokens + report.committed_decode_tokens),
            window_s, window_avg_watts);
        {
            std::lock_guard lock(throughput_window_mutex_);
            throughput_window_ = report;
        }
        if (reporter_ticks % 12 == 0) { persist_metrics_state(); }
        previous      = current;
        previous_time = now;
        next_deadline += interval;
        const Clock::time_point after_write = Clock::now();
        if (next_deadline <= after_write) { next_deadline = after_write + interval; }
    }

    const ninfer::EngineSnapshot current = service_->snapshot();
    const Clock::time_point now          = Clock::now();
    ThroughputReport tail                = make_throughput_report(
        previous, current, std::chrono::duration<double>(now - previous_time).count());
    // The exact partial interval remains useful to measurement consumers. Pretty throughput is a
    // fixed-cadence operational record and deliberately has no irregular shutdown tail.
    tail.final_interval = true;
    request_jsonl_.write_throughput(tail);
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
        ensure_http_request_id(req, res);
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
                write_anthropic_error(res, error, res.get_header_value("request-id"));
            } else {
                write_openai_error(res, error);
            }
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    server_.set_exception_handler(
        [this](const httplib::Request& req, httplib::Response& res, std::exception_ptr ep) {
            ensure_http_request_id(req, res);
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
                    write_anthropic_error(res, e.error(), res.get_header_value("request-id"));
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
                    write_anthropic_error(res, error, res.get_header_value("request-id"));
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
                    write_anthropic_error(res, error, res.get_header_value("request-id"));
                } else {
                    write_openai_error(res, error);
                }
            }
        });

    register_monitor_routes();
    server_.Get("/health", [this](const httplib::Request&, httplib::Response& res) {
        const bool available = service_ != nullptr && service_->is_available();
        res.status           = available ? 200 : 503;
        res.set_content(nlohmann::json{{"status", available ? "ok" : "unavailable"}}.dump(),
                        "application/json");
    });
    server_.Get("/v1/models", [this](const httplib::Request& req, httplib::Response& res) {
        handle_models(req, res);
    });
    server_.Get("/metrics", [this](const httplib::Request&, httplib::Response& res) {
        const auto stats = observations_.normalize(service_->snapshot());
        res.set_header("Cache-Control", "no-store");
        res.set_content(metrics_.render(stats, observations_.snapshot(), service_->is_available()),
                        "text/plain; version=0.0.4; charset=utf-8");
    });
    server_.Get("/slots", [this](const httplib::Request&, httplib::Response& res) {
        handle_slots(res);
    });
    server_.Get("/usage", [this](const httplib::Request&, httplib::Response& res) {
        handle_usage(res);
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
    res.set_content(
        make_models_list(public_model_id_, unix_time_now(), options_.launch.engine.max_context),
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
    // Non-spec extension: attach the configured model card inline (vLLM precedent — clients
    // ignore unknown members on the Model object). Thin object stays byte-identical without.
    nlohmann::json payload = nlohmann::json::parse(make_model_object(
        public_model_id_, unix_time_now(), options_.launch.engine.max_context));
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

// Fleet operators watch materializing/prefilling/decode_ready to tell a healthy engine
// from one that has latched (see 2026-10-30 wedge: /health stayed 200 while wedged).
void HttpServer::handle_slots(httplib::Response& res) const {
    if (service_ == nullptr) {
        res.status = 503;
        res.set_content(R"({"error":"service not attached"})", "application/json");
        return;
    }
    const auto snapshot = observations_.normalize(service_->snapshot());
    const ninfer::RuntimeStats& stats = snapshot.runtime;
    const std::uint32_t max_lanes     = service_->engine_options().max_concurrency;
    nlohmann::json lanes              = nlohmann::json::array();
    for (std::uint32_t lane = 0; lane < max_lanes; ++lane) {
        lanes.push_back(nlohmann::json{
            {"id", lane},
            {"busy", lane < stats.running_requests},
        });
    }
    nlohmann::json kv_pools = nlohmann::json::array();
    for (const auto& pool : stats.kv_pools) {
        kv_pools.push_back(nlohmann::json{
            {"id", pool.id},
            {"label", pool.label},
            {"occupied_page_groups", pool.occupied_page_groups},
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
        {"paused", stats.paused_requests},
        {"replaying", stats.replaying_requests},
        {"kv_pools", std::move(kv_pools)},
        {"slots", std::move(lanes)},
    };
    res.set_header("Cache-Control", "no-store");
    res.set_content(out.dump(), "application/json");
}

// Human-readable fleet summary: tokens, cache behaviour, 5s-window throughput, lanes, and
// NVML energy accounting with per-day cost attribution. Copies state under short lock
// scopes and builds the response afterwards (the fork's handle_usage once deadlocked here).
void HttpServer::handle_usage(httplib::Response& res) const {
    using Clock = std::chrono::steady_clock;
    static const Clock::time_point process_start = Clock::now();
    if (service_ == nullptr) {
        res.status = 503;
        res.set_content(R"({"error":"service not attached"})", "application/json");
        return;
    }
    const auto snapshot = observations_.normalize(service_->snapshot());
    const ninfer::RuntimeStats& stats = snapshot.runtime;
    const double uptime_s =
        std::chrono::duration<double>(Clock::now() - process_start).count();

    std::uint64_t pt_total = 0, pt_cache = 0, pt_out = 0;
    std::map<std::string, double> daily_ws_copy;
    double month_ws = 0.0;
    {
        std::lock_guard lock(energy_mutex_);
        pt_total = persisted_prompt_tokens + persisted_cached_tokens +
                   stats.computed_prefill_tokens + stats.reused_prompt_tokens;
        pt_cache = persisted_cached_tokens + stats.reused_prompt_tokens;
        pt_out   = persisted_output_tokens + stats.committed_decode_tokens;
        daily_ws_copy = energy.daily_ws;
        month_ws      = energy.month_ws;
    }
    const double hit_rate = pt_total ? 100.0 * pt_cache / pt_total : 0.0;

    // Most recent reporter window. Kept fresh by the stats reporter; decayed to zero when
    // the reporter stalls so consumers never read stale activity.
    double window_prefill_tps = 0.0, window_decode_tps = 0.0, window_seconds = 0.0;
    std::uint64_t window_prefill_tokens = 0, window_decode_tokens = 0;
    bool window_active = false;
    {
        std::lock_guard lock(throughput_window_mutex_);
        window_seconds        = throughput_window_.interval_seconds;        window_prefill_tokens = throughput_window_.computed_prefill_tokens;
        window_decode_tokens  = throughput_window_.committed_decode_tokens;
        window_active         = window_seconds > 0.0;
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
    const double usd_per_kwh = options_.electricity_rate_usd_per_kwh;
    nlohmann::json days = nlohmann::json::object();
    for (const auto& [key, ws] : daily_ws_copy) {
        const std::uint64_t day_tokens =
            daily_tokens_copy.count(key) != 0 ? daily_tokens_copy.at(key) : 0;
        const double day_cost = ws / 3.6e6 * usd_per_kwh;
        days[key] = {{"wh", ws / 3600.0},
                     {"kwh", ws / 3.6e6},
                     {"cost_usd", day_cost},
                     {"tokens", day_tokens},
                     {"cost_per_m_tokens_usd",
                      day_tokens != 0 ? day_cost / static_cast<double>(day_tokens) * 1e6 : 0.0}};
    }
    const std::time_t tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::tm utc{};
    gmtime_r(&tt, &utc);
    char today_key[16];
    std::snprintf(today_key, sizeof(today_key), "%04d-%02d-%02d", utc.tm_year + 1900,
                  utc.tm_mon + 1, utc.tm_mday);
    const double today_ws =
        daily_ws_copy.count(today_key) != 0 ? daily_ws_copy.at(today_key) : 0.0;
    const double today_cost = today_ws / 3.6e6 * usd_per_kwh;
    const double month_cost = month_ws / 3.6e6 * usd_per_kwh;

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
         {{"total", stats.requests.completed + stats.requests.failed},
          {"completed", stats.requests.completed},
          {"failed", stats.requests.failed}}},
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
          {"preemptions", stats.preemptions},
          {"snapshot_restores", stats.snapshot_restores},
          {"replay_restores", stats.replay_restores}}},
        {"energy",
         {{"rate_usd_per_kwh", usd_per_kwh},
          {"nvml_available", true},
          {"today",
           {{"wh", today_ws / 3600.0},
            {"kwh", today_ws / 3.6e6},
            {"cost_usd", today_cost},
            {"tokens", tokens_today},
            {"cost_per_m_tokens_usd",
             tokens_today != 0 ? today_cost / static_cast<double>(tokens_today) * 1e6 : 0.0}}},
          {"rolling_30d",
           {{"wh", month_ws / 3600.0},
            {"kwh", month_ws / 3.6e6},
            {"cost_usd", month_cost},
            {"tokens", static_cast<std::uint64_t>(tokens_30d)},
            {"cost_per_m_tokens_usd",
             tokens_30d != 0 ? month_cost / tokens_30d * 1e6 : 0.0}}},
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
    const ninfer::LoadSummary load = service.startup_report().load;
    if (!options_.model_id_override && load.model_name.empty()) {
        throw std::logic_error("loaded artifact model name must not be empty");
    }
    public_model_id_ = options_.model_id_override.value_or(load.model_name);
    service_         = &service;
    observations_.attach(service.snapshot());
    metrics_.configure(public_model_id_, service.engine_options(), service.startup_report(),
                       observations_.start_seconds());
    request_jsonl_.write_server_start(
        options_, service.engine_options(), service.sampling_defaults(), public_model_id_,
        service.description(), service.startup_report(), service.snapshot());
}

bool HttpServer::listen() {
    if (service_ == nullptr) { throw std::logic_error("HTTP generation service is not attached"); }
    if (public_model_id_.empty()) {
        throw std::logic_error("HTTP public model id is not resolved");
    }
    if (options_.launch.logging.stats_interval_ms != 0) {
        stats_stopping_ = false;
        stats_thread_   = std::thread([this] { run_stats_reporter(); });
    }
    try {
        const bool result = server_.listen_after_bind();
        observations_.stop();
        stop_stats_reporter();
        return result;
    } catch (...) {
        observations_.stop();
        stop_stats_reporter();
        throw;
    }
}

void HttpServer::stop() {
    observations_.stop();
    server_.stop();
}

} // namespace ninfer::serve
