const std = @import("std");
const llama = @import("llama_cpp.zig");

pub const Error = llama.LlamaError || error{
    BackendAlreadyInUse,
    InvalidToken,
    VocabularyMismatch,
};

pub const Run = struct {
    load_ms: f64,
    decode_ms: f64,
    current_rss: ?u64,
};

pub const GgmlBackendRun = struct {
    run: Run,
    stats: llama.c.struct_mlz_ggml_residency_stats,
};

/// Runs a deterministic CPU-only llama.cpp prefill and copies the complete
/// final-token vocabulary logit row into caller-owned memory. The model/context
/// lifetime is fully contained here so its mmap does not survive validation.
pub fn sequenceLogits(
    path_z: [:0]const u8,
    tokens: []const usize,
    output: []f32,
    current_rss: *const fn () ?u64,
) !Run {
    return sequenceLogitsWithMapping(path_z, tokens, output, current_rss, false);
}

/// Reference variant for very large models where materializing every weight in
/// process RAM is unsafe. The OS-backed model mapping is outside the bounded
/// residency manager and is reported separately by validators.
pub fn sequenceLogitsMmap(
    path_z: [:0]const u8,
    tokens: []const usize,
    output: []f32,
    current_rss: *const fn () ?u64,
) !Run {
    return sequenceLogitsWithMapping(path_z, tokens, output, current_rss, true);
}

/// Runs the same native llama.cpp/GGML CPU graph as the reference path, but
/// places every model tensor in MLz's custom host buffer type through the
/// official llama_model_params.tensor_buft_overrides API. This first backend
/// milestone intentionally keeps tensors allocated for the model lifetime;
/// node-lifetime pin/release is a separate integration step.
pub fn sequenceLogitsGgmlBackend(
    path_z: [:0]const u8,
    tokens: []const usize,
    output: []f32,
    current_rss: *const fn () ?u64,
) !GgmlBackendRun {
    if (tokens.len == 0 or tokens.len > std.math.maxInt(u32)) return Error.InvalidToken;
    for (tokens) |token| {
        if (token > @as(usize, @intCast(std.math.maxInt(llama.Token)))) return Error.InvalidToken;
    }

    const backend = llama.Backend.init();
    defer backend.deinit();

    const stats_before = llama.c.mlz_ggml_residency_get_stats();
    if (stats_before.current_allocated_bytes != 0) return error.BackendAlreadyInUse;
    llama.c.mlz_ggml_residency_reset_stats();
    const pattern: [*:0]const u8 = ".*";
    var overrides = [_]llama.c.llama_model_tensor_buft_override{
        .{ .pattern = pattern, .buft = llama.c.mlz_ggml_residency_buffer_type() },
        .{ .pattern = null, .buft = null },
    };

    var load_timer = try std.time.Timer.start();
    var model_params = llama.c.llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.use_mmap = true;
    model_params.use_mlock = false;
    model_params.check_tensors = false;
    model_params.tensor_buft_overrides = &overrides;
    const model = try llama.Model.load(path_z, model_params);
    defer model.deinit();
    const load_ns = load_timer.read();

    const vocab = model.vocab() orelse return Error.VocabUnavailable;
    const vocab_size_i32 = llama.c.llama_vocab_n_tokens(vocab);
    if (vocab_size_i32 <= 0) return Error.VocabUnavailable;
    const vocab_size: usize = @intCast(vocab_size_i32);
    if (vocab_size != output.len) {
        return Error.VocabularyMismatch;
    }
    for (tokens) |token| {
        if (token >= vocab_size) return Error.InvalidToken;
    }

    var context_params = llama.c.llama_context_default_params();
    context_params.n_ctx = @intCast(@max(tokens.len, 32));
    context_params.n_batch = @intCast(tokens.len);
    context_params.n_ubatch = @intCast(tokens.len);
    context_params.n_seq_max = 1;
    context_params.n_threads = 1;
    context_params.n_threads_batch = 1;
    context_params.offload_kqv = false;
    context_params.op_offload = false;
    context_params.flash_attn_type = llama.c.LLAMA_FLASH_ATTN_TYPE_DISABLED;
    const context = try llama.Context.init(model, context_params);
    defer context.deinit();

    var batch = llama.Batch.init(@intCast(tokens.len), 0, 1);
    defer batch.deinit();
    const sequence = [_]i32{0};
    for (tokens, 0..) |token, position| {
        try batch.add(@intCast(token), @intCast(position), &sequence, position + 1 == tokens.len);
    }

    var decode_timer = try std.time.Timer.start();
    try context.decode(batch.handle);
    const logits = context.logitsIth(@intCast(tokens.len - 1)) orelse return Error.LogitsUnavailable;
    @memcpy(output, logits[0..output.len]);
    const decode_ns = decode_timer.read();

    return .{
        .run = .{
            .load_ms = @as(f64, @floatFromInt(load_ns)) / std.time.ns_per_ms,
            .decode_ms = @as(f64, @floatFromInt(decode_ns)) / std.time.ns_per_ms,
            .current_rss = current_rss(),
        },
        .stats = llama.c.mlz_ggml_residency_get_stats(),
    };
}

fn sequenceLogitsWithMapping(
    path_z: [:0]const u8,
    tokens: []const usize,
    output: []f32,
    current_rss: *const fn () ?u64,
    use_mmap: bool,
) !Run {
    if (tokens.len == 0 or tokens.len > std.math.maxInt(u32)) return Error.InvalidToken;
    for (tokens) |token| {
        if (token > @as(usize, @intCast(std.math.maxInt(llama.Token)))) return Error.InvalidToken;
    }

    const backend = llama.Backend.init();
    defer backend.deinit();

    var load_timer = try std.time.Timer.start();
    var model_params = llama.c.llama_model_default_params();
    model_params.n_gpu_layers = 0;
    model_params.use_mmap = use_mmap;
    model_params.use_mlock = false;
    const model = try llama.Model.load(path_z, model_params);
    defer model.deinit();
    const load_ns = load_timer.read();

    const vocab = model.vocab() orelse return Error.VocabUnavailable;
    const vocab_size_i32 = llama.c.llama_vocab_n_tokens(vocab);
    if (vocab_size_i32 <= 0) return Error.VocabUnavailable;
    const vocab_size: usize = @intCast(vocab_size_i32);
    if (vocab_size != output.len) {
        return Error.VocabularyMismatch;
    }
    for (tokens) |token| {
        if (token >= vocab_size) return Error.InvalidToken;
    }

    var context_params = llama.c.llama_context_default_params();
    context_params.n_ctx = @intCast(@max(tokens.len, 32));
    context_params.n_batch = @intCast(tokens.len);
    context_params.n_ubatch = @intCast(tokens.len);
    context_params.n_seq_max = 1;
    context_params.n_threads = 1;
    context_params.n_threads_batch = 1;
    context_params.offload_kqv = false;
    context_params.op_offload = false;
    context_params.flash_attn_type = llama.c.LLAMA_FLASH_ATTN_TYPE_DISABLED;
    const context = try llama.Context.init(model, context_params);
    defer context.deinit();

    var batch = llama.Batch.init(@intCast(tokens.len), 0, 1);
    defer batch.deinit();
    const sequence = [_]i32{0};
    for (tokens, 0..) |token, position| {
        try batch.add(@intCast(token), @intCast(position), &sequence, position + 1 == tokens.len);
    }

    var decode_timer = try std.time.Timer.start();
    try context.decode(batch.handle);
    const logits = context.logitsIth(@intCast(tokens.len - 1)) orelse return Error.LogitsUnavailable;
    @memcpy(output, logits[0..output.len]);
    const decode_ns = decode_timer.read();

    return .{
        .load_ms = @as(f64, @floatFromInt(load_ns)) / std.time.ns_per_ms,
        .decode_ms = @as(f64, @floatFromInt(decode_ns)) / std.time.ns_per_ms,
        .current_rss = current_rss(),
    };
}

/// Single-token compatibility wrapper.
pub fn singleTokenLogits(
    path_z: [:0]const u8,
    token: usize,
    output: []f32,
    current_rss: *const fn () ?u64,
) !Run {
    const tokens = [_]usize{token};
    return sequenceLogits(path_z, &tokens, output, current_rss);
}
