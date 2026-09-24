#include "llama_memory_shim.h"

#include "llama-ext.h"

extern "C" bool mlz_llama_memory_breakdown(
        const struct llama_context * ctx,
        ggml_backend_buffer_type_t exclude_model_buft,
        struct mlz_llama_memory * out) {
    if (ctx == nullptr || out == nullptr) {
        return false;
    }
    *out = {};
    try {
        for (const auto & [buft, data] : llama_get_memory_breakdown(ctx)) {
            if (buft != exclude_model_buft) {
                out->model += data.model;
            }
            out->context += data.context;
            out->compute += data.compute;
        }
    } catch (...) {
        return false;
    }
    return true;
}
