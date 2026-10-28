// Stage 1 of the KVA port: dump layer residuals from a real artifact.
//
// Drives TextContext::prefill_chunk with a DFlashFeatureSink pointed at a caller-chosen
// layer set, then writes the captured BF16 residuals as raw bytes. This is the training
// data source for a KVA projector: the layer-`split-1` residual goes in, the late layers'
// token-mixer inputs come out.
//
// Usage:
//   NINFER_TEST_ARTIFACT=/path/to.ninfer ./ninfer_qwen3_5_kva_extract_test \
//       <layer,layer,...> <out.bin> [prompt_file]
//
// Without a prompt file, uses a built-in text. Prints the captured shape on stdout.

#include "ninfer/engine.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<std::uint32_t> parse_layers(const char* spec) {
    std::vector<std::uint32_t> out;
    std::stringstream ss(spec);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) { continue; }
        out.push_back(static_cast<std::uint32_t>(std::stoul(item)));
    }
    if (out.empty()) { throw std::runtime_error("no layers given"); }
    return out;
}

std::string read_text(const char* path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error(std::string("cannot open ") + path); }
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

} // namespace

int main(int argc, char** argv) {
    const char* artifact = std::getenv("NINFER_TEST_ARTIFACT");
    if (!artifact || !*artifact) {
        std::cout << "skip: NINFER_TEST_ARTIFACT is not set\n";
        return 77;
    }
    if (argc < 3) {
        std::cerr << "usage: " << argv[0] << " <layer,layer,...> <out.bin> [prompt_file]\n";
        return 2;
    }
    try {
        const auto layers = parse_layers(argv[1]);
        const std::string out_path = argv[2];
        const std::string text =
            argc > 3 ? read_text(argv[3])
                     : std::string("The specification document describes a recursive descent "
                                   "parser that emits intermediate representations for each "
                                   "production rule while tracking source positions.");

        ninfer::EngineOptions options;
        options.artifact_path   = artifact;
        options.max_context     = 4096;
        options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(4096);
        options.prefill_chunk   = 4096;
        options.max_concurrency = 1;
        options.use_cuda_graph  = false;
        options.enable_vision   = false;
        options.purpose         = ninfer::EnginePurpose::CausalScoring;

        ninfer::Engine engine(options);
        const auto ids = engine.tokenize_text(text);
        std::cout << "tokens: " << ids.size() << "\n";

        // The public Engine API has no hidden-state access, so the capture goes through the
        // same feature-sink path the DFlash2 draft uses. See PORT-PLAN.md stage 1.
        const auto dumped = engine.dump_layer_residuals(ids, layers);
        std::cout << "captured layers: " << layers.size()
                  << "  tokens: " << dumped.tokens
                  << "  width: " << dumped.width
                  << "  bytes: " << dumped.bytes.size() << "\n";

        std::ofstream out(out_path, std::ios::binary);
        if (!out) { throw std::runtime_error("cannot write " + out_path); }
        out.write(reinterpret_cast<const char*>(dumped.bytes.data()),
                  static_cast<std::streamsize>(dumped.bytes.size()));
        out.close();
        std::cout << "wrote " << out_path << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
