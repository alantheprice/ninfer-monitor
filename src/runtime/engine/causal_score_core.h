#pragma once

#include "core/device.h"
#include "ninfer/types.h"
#include "runtime/engine/kv_capacity.h"

#include <condition_variable>
#include <cstdint>
#include <exception>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace ninfer::runtime {

// Single-owner execution core for offline causal scoring. It deliberately has no request queue,
// Scheduler, ResourceManager, continuation catalog, or batching policy.
template <class Instance>
class CausalScoreCore {
public:
    using ModelContract  = typename Instance::ModelContract;
    using PreparedPrompt = typename ModelContract::PreparedPrompt;

    CausalScoreCore(Instance& instance, DeviceContext& device)
        : instance_(instance), device_(device) {
        std::promise<void> startup;
        std::future<void> started = startup.get_future();
        worker_                   = std::thread([this, startup = std::move(startup)]() mutable {
            try {
                device_.bind_to_current_thread();
                startup.set_value();
            } catch (...) {
                startup.set_exception(std::current_exception());
                return;
            }
            worker_loop();
        });
        try {
            started.get();
        } catch (...) {
            if (worker_.joinable()) { worker_.join(); }
            throw;
        }
    }

    ~CausalScoreCore() noexcept {
        {
            std::lock_guard lock(queue_mutex_);
            stopping_ = true;
        }
        queue_cv_.notify_all();
        if (worker_.joinable()) { worker_.join(); }
    }

    CausalScoreCore(const CausalScoreCore&)            = delete;
    CausalScoreCore& operator=(const CausalScoreCore&) = delete;

    [[nodiscard]] std::vector<float> score(PreparedPrompt prompt, std::uint32_t first_target) {
        // One synchronous public call owns the sole job slot until its result is delivered.
        std::scoped_lock call_lock(call_mutex_);
        auto job                               = std::make_unique<Job>();
        job->prompt                            = std::move(prompt);
        job->first_target                      = first_target;
        std::future<std::vector<float>> result = job->promise.get_future();
        {
            std::lock_guard queue_lock(queue_mutex_);
            if (stopping_) { throw std::runtime_error("causal scoring engine is stopping"); }
            if (job_ != nullptr) {
                throw std::logic_error("causal scoring core already has an in-flight job");
            }
            job_ = std::move(job);
        }
        queue_cv_.notify_one();
        return result.get();
    }

    // Offline layer-residual capture. Shares the single job slot with score().
    struct DumpResult {
        std::vector<std::byte> bytes;
        std::uint32_t tokens = 0;
        std::uint32_t width  = 0;
    };
    struct MixerDumpResult {
        std::vector<std::byte> bytes;
        std::uint32_t tokens = 0;
        std::uint32_t hidden = 0;
        std::vector<std::int32_t> widths;
    };
    [[nodiscard]] DumpResult dump_residuals(PreparedPrompt prompt,
                                            std::vector<std::uint32_t> layers) {
        std::scoped_lock call_lock(call_mutex_);
        auto job                                = std::make_unique<Job>();
        job->prompt                             = std::move(prompt);
        job->layers                             = std::move(layers);
        job->is_dump                            = true;
        std::future<DumpResult> result          = job->dump_promise.get_future();
        {
            std::lock_guard queue_lock(queue_mutex_);
            if (stopping_) { throw std::runtime_error("causal scoring engine is stopping"); }
            if (job_ != nullptr) {
                throw std::logic_error("causal scoring core already has an in-flight job");
            }
            job_ = std::move(job);
        }
        queue_cv_.notify_one();
        return result.get();
    }

    [[nodiscard]] MixerDumpResult dump_mixers(PreparedPrompt prompt,
                                              std::vector<std::uint32_t> layers,
                                              std::vector<std::int32_t> widths) {
        std::scoped_lock call_lock(call_mutex_);
        auto job                                = std::make_unique<Job>();
        job->prompt                             = std::move(prompt);
        job->layers                             = std::move(layers);
        job->widths                             = std::move(widths);
        job->is_mixer_dump                      = true;
        std::future<MixerDumpResult> result     = job->mixer_promise.get_future();
        {
            std::lock_guard queue_lock(queue_mutex_);
            if (stopping_) { throw std::runtime_error("causal scoring engine is stopping"); }
            if (job_ != nullptr) {
                throw std::logic_error("causal scoring core already has an in-flight job");
            }
            job_ = std::move(job);
        }
        queue_cv_.notify_one();
        return result.get();
    }

    [[nodiscard]] MemorySummary memory_summary() const {
        std::scoped_lock lock(execution_mutex_);
        MemorySummary out                      = instance_.program->memory_summary();
        const KvCapacityResolution& resolution = instance_.kv_capacity_resolution;
        out.kv_capacity_mode                   = resolution.mode;
        out.kv_capacity_page_groups            = resolution.main_page_groups;
        out.kv_capacity_max_page_groups        = resolution.maximum_main_page_groups;
        out.minimum_runtime_reservation_bytes  = resolution.minimum_runtime_reservation_bytes;
        out.kv_capacity_increment_bytes        = resolution.bytes_per_additional_main_page_group;
        out.runtime_reservation_bytes          = resolution.runtime_reservation_bytes;
        out.available_after_weights_bytes      = resolution.available_after_weights_bytes;
        out.available_after_startup_bytes      = resolution.available_after_startup_bytes;
        out.kv_capacity_headroom_bytes         = resolution.automatic_headroom_bytes;
        out.planned_slack_bytes                = resolution.planned_slack_bytes;
        return out;
    }

    [[nodiscard]] RuntimeStats runtime_stats() const noexcept { return {}; }

    [[nodiscard]] bool is_available() const {
        std::lock_guard lock(queue_mutex_);
        return !stopping_;
    }

    void reset_memory_peaks() noexcept {
        try {
            std::scoped_lock lock(execution_mutex_);
            instance_.program->reset_memory_peaks();
        } catch (...) {}
    }

private:
    struct Job {
        PreparedPrompt prompt;
        std::uint32_t first_target = 0;
        std::vector<std::uint32_t> layers;
        std::vector<std::int32_t> widths;
        bool is_dump       = false;
        bool is_mixer_dump = false;
        std::promise<std::vector<float>> promise;
        std::promise<DumpResult> dump_promise;
        std::promise<MixerDumpResult> mixer_promise;
    };

    void worker_loop() noexcept {
        for (;;) {
            std::unique_ptr<Job> job;
            {
                std::unique_lock lock(queue_mutex_);
                queue_cv_.wait(lock, [&] { return stopping_ || job_ != nullptr; });
                if (job_ == nullptr) {
                    if (stopping_) { return; }
                    continue;
                }
                job = std::move(job_);
            }
            try {
                if (job->is_mixer_dump) {
                    MixerDumpResult result;
                    {
                        std::scoped_lock lock(execution_mutex_);
                        instance_.program->dump_mixer_inputs(
                            std::move(job->prompt), job->layers, job->widths, result.bytes,
                            result.tokens, result.hidden, result.widths);
                    }
                    job->mixer_promise.set_value(std::move(result));
                    continue;
                }
                if (job->is_dump) {
                    DumpResult result;
                    {
                        std::scoped_lock lock(execution_mutex_);
                        instance_.program->dump_layer_residuals(
                            std::move(job->prompt), job->layers, result.bytes, result.tokens,
                            result.width);
                    }
                    job->dump_promise.set_value(std::move(result));
                    continue;
                }
                std::vector<float> result;
                {
                    std::scoped_lock lock(execution_mutex_);
                    result =
                        instance_.program->causal_score(std::move(job->prompt), job->first_target);
                }
                job->promise.set_value(std::move(result));
            } catch (...) {
                try {
                    if (job->is_mixer_dump) {
                        job->mixer_promise.set_exception(std::current_exception());
                    } else if (job->is_dump) {
                        job->dump_promise.set_exception(std::current_exception());
                    } else {
                        job->promise.set_exception(std::current_exception());
                    }
                } catch (...) {}
            }
        }
    }

    Instance& instance_;
    DeviceContext& device_;
    mutable std::mutex execution_mutex_;
    std::mutex call_mutex_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::unique_ptr<Job> job_;
    bool stopping_ = false;
    std::thread worker_;
};

} // namespace ninfer::runtime
