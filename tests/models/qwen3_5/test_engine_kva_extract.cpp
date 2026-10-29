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
        std::cerr << "usage: " << argv[0] << " <layer,layer,...> <out.bin> [prompt_file]\n"
                  << "       " << argv[0] << " --mixers <layer:width,...> <out.bin> [prompt_file]\n";
        return 2;
    }
    try {
        std::string mode = "residual";
        std::vector<std::string> specs;
        int argi = 1;
        if (std::string(argv[1]) == "--mixers") {
            mode = "mixers";
            argi = 2;
        }
        if (argc <= argi + 1) {
            std::cerr << "missing output path\n";
            return 2;
        }
        {
            std::stringstream ss(argv[argi]);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) { specs.push_back(item); }
            }
        }
        std::vector<std::uint32_t> layers;
        if (mode == "residual") {
            for (const auto& s : specs) {
                layers.push_back(static_cast<std::uint32_t>(std::stoul(s)));
            }
            if (layers.empty()) { throw std::runtime_error("no layers given"); }
        }
        const std::string out_path = argv[argi + 1];
        const std::string text =
            argc > argi + 2 ? read_text(argv[argi + 2])
                            : std::string("The specification document describes a recursive descent "
                                          "parser that emits intermediate representations for each "
                                          "production rule while tracking source positions.");

        ninfer::EngineOptions options;
        options.artifact_path   = artifact;
        // Extraction chunks are bounded by max_context; a larger value amortizes the per-chunk
        // overhead and lets a corpus be walked in fewer passes.
        options.max_context     = 16384;
        options.kv_capacity     = ninfer::KvCapacityPolicy::explicit_capacity(16384);
        options.prefill_chunk   = 16384;
        options.max_concurrency = 1;
        options.use_cuda_graph  = false;
        options.enable_vision   = false;
        options.purpose         = ninfer::EnginePurpose::CausalScoring;
        // Device selection is explicit: EngineOptions::device defaults to 0 and cudaSetDevice()
        // overrides CUDA_VISIBLE_DEVICES, so relying on the env var alone silently lands on GPU 0.
        if (const char* dev = std::getenv("NINFER_TEST_DEVICE")) {
            options.device = static_cast<int>(std::stol(dev));
        }

        ninfer::Engine engine(options);
        std::cout << "device: " << engine.options().device << std::endl;
        const auto ids = engine.tokenize_text(text);
        std::cout << "tokens: " << ids.size() << std::endl;

        // Mixer-input mode: `--mixers layer:width,layer:width,...`
        if (mode == "mixers") {
            std::vector<std::uint32_t> layer_ids;
            std::vector<std::int32_t> widths;
            for (const auto& spec : specs) {
                const auto colon = spec.find(':');
                if (colon == std::string::npos) {
                    throw std::runtime_error("mixer spec must be layer:width");
                }
                layer_ids.push_back(static_cast<std::uint32_t>(std::stoul(spec.substr(0, colon))));
                widths.push_back(static_cast<std::int32_t>(std::stol(spec.substr(colon + 1))));
            }
            const auto dumped = engine.dump_mixer_inputs(ids, layer_ids, widths);
            std::cout << "mixer capture: layers " << layer_ids.size()
                      << "  tokens " << dumped.tokens
                      << "  hidden " << dumped.hidden
                      << "  bytes " << dumped.bytes.size() << std::endl;
            std::ofstream out(out_path, std::ios::binary);
            if (!out) { throw std::runtime_error("cannot write " + out_path); }
            out.write(reinterpret_cast<const char*>(dumped.bytes.data()),
                      static_cast<std::streamsize>(dumped.bytes.size()));
            out.close();
            // Sidecar: the widths, so the reader can split [hidden | proj] per layer.
            std::ofstream meta(out_path + ".json");
            meta << "{\"tokens\":" << dumped.tokens << ",\"hidden\":" << dumped.hidden
                 << ",\"layers\":[";
            for (std::size_t i = 0; i < layer_ids.size(); ++i) {
                meta << (i ? "," : "") << "{\"layer\":" << layer_ids[i]
                     << ",\"width\":" << dumped.widths[i] << "}";
            }
            meta << "]}\n";
            std::cout << "wrote " << out_path << "\n";
            return 0;
        }

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
