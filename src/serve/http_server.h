#pragma once

#include "serve/generation_service.h"
#include "serve/operational_log.h"
#include "serve/openai_responses_store.h"
#include "serve/request_log.h"
#include "serve/serve_options.h"

#include <httplib.h>

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <memory>
#include <map>
#include <string>
#include <string_view>
#include <thread>

namespace ninfer::serve {

void write_openai_error(httplib::Response& response, const ApiError& error);
void write_anthropic_error(httplib::Response& response, const ApiError& error,
                           const std::string& request_id);

// cpp-httplib invokes the error handler for every application response with status >= 400. Only
// an empty 413 is its own pre-routing payload-limit rejection; application-authored errors must be
// left untouched.
httplib::Server::HandlerResponse handle_unrendered_http_error(const ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response);

[[nodiscard]] bool matches_bearer_credential(std::string_view authorization,
                                             std::string_view api_key) noexcept;

class HttpServer {
public:
    HttpServer(ServeOptions options, std::shared_ptr<spdlog::logger> logger);

    // Reserves the configured address before model loading. The service is attached only after its
    // Engine is ready, then listen() enters the blocking accept loop on the already-bound socket.
    bool bind();
    void attach(GenerationService& service);
    bool listen();
    void stop();

    [[nodiscard]] const std::string& public_model_id() const noexcept { return public_model_id_; }

private:
    class RequestLifecycle {
    public:
        RequestLifecycle(HttpServer& owner, RequestLogContext context);

        void done(const GenerationOutcome& outcome);
        void failure(const RequestFailure& failure);
        void response_failure(const RequestFailure& failure);

        [[nodiscard]] std::uint64_t request_id() const noexcept { return context_.id; }

    private:
        enum class State : std::uint8_t {
            Pending,
            Done,
            Error,
        };

        [[nodiscard]] bool claim(State terminal) noexcept;

        HttpServer* owner_ = nullptr;
        RequestLogContext context_;
        std::atomic<State> state_{State::Pending};
    };

    [[nodiscard]] std::shared_ptr<RequestLifecycle> begin_request(RequestLogContext context);

    void register_routes();
    void handle_chat_completions(const httplib::Request& req, httplib::Response& res);
    void handle_messages(const httplib::Request& req, httplib::Response& res);
    void handle_count_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_responses(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_tokens(const httplib::Request& req, httplib::Response& res);
    void handle_response_get(const httplib::Request& req, httplib::Response& res);
    void handle_response_delete(const httplib::Request& req, httplib::Response& res);
    void handle_response_input_items(const httplib::Request& req, httplib::Response& res);
    void handle_response_cancel(const httplib::Request& req, httplib::Response& res);
    void handle_response_compact(const httplib::Request& req, httplib::Response& res);
    void handle_models(const httplib::Request& req, httplib::Response& res) const;
    void handle_model(const httplib::Request& req, httplib::Response& res) const;
    void handle_model_card(const std::string& model_id, httplib::Response& res) const;
    void handle_model_details(const std::string& model_id, httplib::Response& res) const;
    void handle_metrics(httplib::Response& res) const;
    void handle_slots(httplib::Response& res) const;
    void handle_usage(httplib::Response& res) const;
    void handle_energy(httplib::Response& res) const;

    void record_request_start(const RequestLogContext& context);
    void record_request_rejected(const RequestRejectionLogContext& context);
    void record_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
    void record_request_failure(const RequestLogContext& context, const RequestFailure& failure);
    void record_response_failure(std::uint64_t request_id, const RequestFailure& failure);
    void record_throughput(const ThroughputReport& report);
    void record_energy(double tokens, double interval_s, double avg_watts);
    void run_stats_reporter();
    void stop_stats_reporter();

    struct EnergyBuckets {
        // Watt-seconds integrated per calendar bucket (UTC day keys).
        std::map<std::string, double> daily_ws;
        // Tokens served per calendar bucket (same keys as daily_ws).
        std::map<std::string, std::uint64_t> daily_tokens;
        // Tokens served (record_energy is called once per reporter window with the
        // window's token count; daily attribution is approximate).
        std::uint64_t tokens_total = 0;
        std::uint64_t tokens_today = 0;
        std::string tokens_today_key;
        // Rolling 30-day total (recomputed from daily buckets).
        double month_ws = 0.0;
    };

    void persist_metrics_state();
    void restore_metrics_state();

    GenerationService* service_ = nullptr;
    ServeOptions options_;
    std::string public_model_id_;
    OpenAIResponsesStore openai_responses_store_;
    OperationalLog operational_log_;
    JsonlRequestLog request_jsonl_;
    httplib::Server server_;
    std::atomic<std::uint64_t> request_seq_{0};
    std::mutex stats_mutex_;
    std::condition_variable stats_cv_;
    std::thread stats_thread_;
    bool stats_stopping_ = false;
    // Most recent 5s-window throughput (pre-/post- counter delta). Kept fresh by the
    // stats reporter (options_.log_stats_interval_ms); decayed to zero if the reporter
    // stalls so consumers never read stale "activity".
    mutable std::mutex last_throughput_mutex_;
    ThroughputReport last_throughput_;
    std::chrono::steady_clock::time_point last_throughput_at_ = std::chrono::steady_clock::now();

    mutable std::mutex energy_mutex_;
    EnergyBuckets energy;
    // Lifetime token counters (this process + persisted baseline from previous runs).
    std::uint64_t persisted_prompt_tokens = 0;
    std::uint64_t persisted_cached_tokens = 0;
    std::uint64_t persisted_output_tokens = 0;
};

} // namespace ninfer::serve
