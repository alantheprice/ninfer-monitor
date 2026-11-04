#pragma once

#include "serve/generation_service.h"
#include "serve/metrics.h"
#include "serve/operational_log.h"
#include "serve/openai_responses_store.h"
#include "serve/request_log.h"
#include "product/config/options.h"

#include <httplib.h>

#include <atomic>
#include <condition_variable>
#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <memory>
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
httplib::Server::HandlerResponse handle_unrendered_http_error(const product::ServeOptions& options,
                                                              const httplib::Request& request,
                                                              httplib::Response& response);

[[nodiscard]] bool matches_bearer_credential(std::string_view authorization,
                                             std::string_view api_key) noexcept;

class HttpServer {
public:
    HttpServer(product::ServeOptions options, std::shared_ptr<spdlog::logger> logger);

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
    void register_monitor_routes();
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
    void handle_slots(httplib::Response& res) const;
    void handle_usage(httplib::Response& res) const;

    void record_request_start(const RequestLogContext& context);
    [[nodiscard]] ninfer::GenerationSchedulingObserver
    scheduling_observer(std::uint64_t request_id, std::string http_request_id);
    [[nodiscard]] ninfer::GenerationFirstTokenObserver
    first_token_observer(std::uint64_t request_id);
    void record_request_rejected(const RequestRejectionLogContext& context);
    void record_request_done(const RequestLogContext& context, const GenerationOutcome& outcome);
    void record_request_failure(const RequestLogContext& context, const RequestFailure& failure);
    void record_response_failure(std::uint64_t request_id, const RequestFailure& failure);
    void record_throughput(const ThroughputReport& report);
    void record_energy(double tokens, double interval_s, double avg_watts);
    void persist_metrics_state();
    void restore_metrics_state();
    void run_stats_reporter();
    void stop_stats_reporter();

    // Watt-seconds integrated per UTC calendar bucket, with tokens attributed to the same
    // bucket. Rolling 30-day figures are recomputed from the retained daily buckets.
    struct EnergyBuckets {
        std::map<std::string, double> daily_ws;
        std::map<std::string, std::uint64_t> daily_tokens;
        std::uint64_t tokens_total = 0;
        std::uint64_t tokens_today = 0;
        std::string tokens_today_key;
        double month_ws = 0.0;
    };

    GenerationService* service_ = nullptr;
    product::ServeOptions options_;
    std::string public_model_id_;
    OpenAIResponsesStore openai_responses_store_;
    OperationalLog operational_log_;
    ServiceObservations observations_;
    JsonlRequestLog request_jsonl_;
    Metrics metrics_;
    httplib::Server server_;
    std::atomic<std::uint64_t> request_seq_{0};
    std::mutex stats_mutex_;
    std::condition_variable stats_cv_;
    std::thread stats_thread_;
    bool stats_stopping_ = false;
    // Energy accounting. Lifetime token counters include the persisted baseline from
    // previous runs (metrics-state JSON next to the request log when configured).
    mutable std::mutex energy_mutex_;
    EnergyBuckets energy;
    std::uint64_t persisted_prompt_tokens = 0;
    std::uint64_t persisted_cached_tokens = 0;
    std::uint64_t persisted_output_tokens = 0;
    // Most recent reporter window for /usage (decayed to zero on reporter stall).
    mutable std::mutex throughput_window_mutex_;
    ThroughputReport throughput_window_{};
};

} // namespace ninfer::serve
