const std = @import("std");
const mlz = @import("mlz");
const llama = mlz.llama_cpp;

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
    residency_metrics: ?mlz.residency.Metrics = null,
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

/// Runs the native llama.cpp/GGML CPU graph with every model tensor placed in
/// MLz's custom host buffer AND the residency bridge active: no weight bytes
/// are copied into the process at load time; each node's kernel runs against
/// a transient file mapping acquired inside the synchronized node hooks and
/// released afterwards, bounded by the given residency budget.
pub const BackedMode = enum { single, concurrent, cancel, fail };

/// Injected failure points for `.fail`: the first acquisitions hit the token
/// embedding GET_ROWS path; later ones land on whole-node and tiled weights.
const injected_failure_points = [_]u64{ 1, 2, 3, 17, 50, 200 };

pub fn sequenceLogitsGgmlBackendBacked(
    allocator: std.mem.Allocator,
    path_z: [:0]const u8,
    tokens: []const usize,
    output: []f32,
    current_rss: *const fn () ?u64,
    budget_bytes: usize,
    mode: BackedMode,
) !GgmlBackendRun {
    if (tokens.len == 0 or tokens.len > std.math.maxInt(u32)) return Error.InvalidToken;
    for (tokens) |token| {
        if (token > @as(usize, @intCast(std.math.maxInt(llama.Token)))) return Error.InvalidToken;
    }

    const bridge = mlz.residency_ggml_bridge;
    bridge.init(allocator, path_z, budget_bytes) catch |err| switch (err) {
        error.BridgeAlreadyInitialized => return error.BackendAlreadyInUse,
        else => return err,
    };
    defer {
        llama.c.mlz_ggml_residency_set_node_hooks_enabled(false);
        llama.c.mlz_ggml_residency_set_backed_mode(false);
        llama.c.mlz_ggml_residency_set_bridge(null, null, null, null, null, null);
        bridge.deinit(allocator);
        llama.c.mlz_ggml_residency_registry_reset();
    }

    const backend = llama.Backend.init();
    defer backend.deinit();

    const stats_before = llama.c.mlz_ggml_residency_get_stats();
    if (stats_before.current_allocated_bytes != 0) return error.BackendAlreadyInUse;
    llama.c.mlz_ggml_residency_registry_reset();
    llama.c.mlz_ggml_residency_set_bridge(
        bridge.acquireCallback,
        bridge.releaseCallback,
        bridge.spanCallback,
        bridge.acquireRangeCallback,
        bridge.rangeCapacityCallback,
        bridge.acquireManyCallback,
    );
    llama.c.mlz_ggml_residency_set_backed_mode(true);

    // Registry reset above; stats reset is done by the caller as in the
    // non-backed path. Enable node hooks: they perform the per-node
    // rebase/restore around stock kernels.
    llama.c.mlz_ggml_residency_set_node_hooks_enabled(true);

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
    try bridge.syncRegistry();
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
    if (mode == .concurrent or mode == .cancel) {
        const other = try llama.Context.init(model, context_params);
        defer other.deinit();
        var abort_calls = std.atomic.Value(u32).init(0);
        const Abort = struct {
            fn callback(ctx: ?*anyopaque) callconv(.c) bool {
                const calls: *std.atomic.Value(u32) = @ptrCast(@alignCast(ctx.?));
                return calls.fetchAdd(1, .monotonic) >= 50;
            }
        };
        if (mode == .cancel) llama.c.llama_set_abort_callback(other.handle, Abort.callback, &abort_calls);
        var other_batch = llama.Batch.init(@intCast(tokens.len), 0, 1);
        defer other_batch.deinit();
        for (tokens, 0..) |token, position| {
            try other_batch.add(@intCast(token), @intCast(position), &sequence, position + 1 == tokens.len);
        }
        var start = std.atomic.Value(bool).init(false);
        const Worker = struct {
            ctx: llama.Context,
            batch: llama.Batch,
            start: *std.atomic.Value(bool),
            failure: ?anyerror = null,

            fn run(self: *@This()) void {
                while (!self.start.load(.acquire)) std.Thread.sleep(1000);
                self.ctx.decode(self.batch.handle) catch |err| {
                    self.failure = err;
                };
            }
        };
        var worker = Worker{ .ctx = other, .batch = other_batch, .start = &start };
        const thread = try std.Thread.spawn(.{}, Worker.run, .{&worker});
        start.store(true, .release);
        const main_result = context.decode(batch.handle);
        thread.join();
        if (mode == .cancel) {
            const failure = worker.failure orelse return error.CancellationNotObserved;
            if (failure != error.DecodeFailed) return failure;
            llama.c.llama_set_abort_callback(other.handle, null, null);
            if (!other.kvCacheSeqRm(0, -1, -1)) return error.CancelledStateNotCleared;
            try other.decode(other_batch.handle);
        } else if (worker.failure) |err| return err;
        try main_result;
        const other_logits = other.logitsIth(@intCast(tokens.len - 1)) orelse return Error.LogitsUnavailable;
        const main_logits = context.logitsIth(@intCast(tokens.len - 1)) orelse return Error.LogitsUnavailable;
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(other_logits[0..output.len]), std.mem.sliceAsBytes(main_logits[0..output.len]))) {
            return error.ConcurrentLogitMismatch;
        }
    } else if (mode == .fail) {
        // Each injected acquire failure must stop the graph with an error,
        // leave no pin behind, and let the same context decode exactly again.
        for (injected_failure_points, 1..) |after, expected_failures| {
            bridge.injectAcquireFailure(after);
            if (context.decode(batch.handle)) |_| {
                bridge.injectAcquireFailure(0);
                return error.InjectedFailureNotObserved;
            } else |err| if (err != error.DecodeFailed) return err;
            const diagnostics = bridge.diagnostics() orelse return error.ResidencyBridgeMissing;
            if (diagnostics.open_pins != 0) return error.FailedGraphLeakedPins;
            if (diagnostics.failures.injected != expected_failures) return error.InjectedFailureNotRecorded;
            const stats = llama.c.mlz_ggml_residency_get_stats();
            if (stats.graph_failures != expected_failures) return error.GraphFailureNotReported;
            if (stats.current_active_nodes != 0 or stats.node_pre_calls != stats.node_post_calls) {
                return error.NodeHookImbalance;
            }
            if (!context.kvCacheSeqRm(0, -1, -1)) return error.FailedStateNotCleared;
        }
        try context.decode(batch.handle);
    } else {
        try context.decode(batch.handle);
    }
    const logits = context.logitsIth(@intCast(tokens.len - 1)) orelse return Error.LogitsUnavailable;
    @memcpy(output, logits[0..output.len]);
    const decode_ns = decode_timer.read();

    const stats = llama.c.mlz_ggml_residency_get_stats();
    const residency_metrics = bridge.metrics();
    const rss = current_rss();

    return .{
        .run = .{
            .load_ms = @as(f64, @floatFromInt(load_ns)) / std.time.ns_per_ms,
            .decode_ms = @as(f64, @floatFromInt(decode_ns)) / std.time.ns_per_ms,
            .current_rss = rss,
        },
        .stats = stats,
        .residency_metrics = residency_metrics,
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
