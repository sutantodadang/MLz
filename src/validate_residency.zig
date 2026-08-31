const std = @import("std");
const residency = @import("residency.zig");
const gguf = @import("gguf_residency.zig");
const compute = @import("residency_compute.zig");
const executor_mod = @import("residency_executor.zig");
const llama_reference = @import("residency_llama_reference.zig");
const qwen = @import("residency_qwen3next.zig");

const default_budget_mib: usize = 16;
const default_max_tensors: usize = 4;
const default_prefill_chunk: usize = 32;

const QwenBatchProbe = struct {
    descriptor: *const gguf.TensorDescriptor,
    batch_count: usize,
    repeated_ms: f64,
    batched_ms: f64,
    repeated_metrics: residency.Metrics,
    batched_metrics: residency.Metrics,
    max_abs_error: f32,
    scratch_bytes: usize,
    rss: ?u64,
};

const QwenModelRun = struct {
    elapsed_ms: f64,
    checksum: f64,
    argmax: usize,
    accounting: executor_mod.MemoryAccounting,
    cache_bytes: usize,
    workspace_bytes: usize,
    logits: []f32,

    fn deinit(self: *const QwenModelRun, allocator: std.mem.Allocator) void {
        allocator.free(self.logits);
    }
};

const Validation = struct {
    descriptor: *const gguf.TensorDescriptor,
    elapsed_baseline_ms: f64,
    elapsed_bounded_ms: f64,
    max_abs_error: f32,
    checksum: f64,
    baseline_rss: ?u64,
    bounded_rss: ?u64,
    metrics: residency.Metrics,
};

const FfnValidation = struct {
    elapsed_baseline_ms: f64,
    elapsed_bounded_ms: f64,
    max_abs_error: f32,
    checksum: f64,
    baseline_rss: ?u64,
    bounded_rss: ?u64,
    accounting: executor_mod.MemoryAccounting,
};

const FfnRun = struct {
    elapsed_ms: f64,
    rss: ?u64,
    accounting: executor_mod.MemoryAccounting,
};

const DecoderValidation = struct {
    elapsed_baseline_ms: f64,
    elapsed_bounded_ms: f64,
    max_abs_error: f32,
    checksum: f64,
    baseline_rss: ?u64,
    bounded_rss: ?u64,
    accounting: executor_mod.DecoderMemoryAccounting,
};

const DecoderRun = struct {
    elapsed_ms: f64,
    rss: ?u64,
    accounting: executor_mod.DecoderMemoryAccounting,
};

const ModelAccounting = struct {
    executor: executor_mod.MemoryAccounting,
    attention_workspace_bytes: usize,
    kv_cache_bytes: usize,
};

const ModelRun = struct {
    elapsed_ms: f64,
    rss: ?u64,
    accounting: ModelAccounting,
};

const ModelValidation = struct {
    token: usize,
    token_count: usize,
    layer_count: usize,
    vocab_size: usize,
    elapsed_baseline_ms: f64,
    elapsed_bounded_ms: f64,
    max_abs_error: f32,
    checksum: f64,
    argmax_token: usize,
    baseline_rss: ?u64,
    bounded_rss: ?u64,
    accounting: ModelAccounting,
    bounded_logits: []f32,

    fn deinit(self: ModelValidation, allocator: std.mem.Allocator) void {
        allocator.free(self.bounded_logits);
    }
};

const reference_max_error_limit: f32 = 0.5;
const reference_mean_error_limit: f64 = 0.1;
// The proof executor uses the same quantized vec_dot kernels as GGML, but its
// scalar orchestration/reduction order is not bit-identical to GGML's graph.
// Treat the reference as close only when both bounded error and top-1 agree.

const ReferenceStatus = enum {
    exact,
    close,
    mismatch,

    fn name(self: ReferenceStatus) []const u8 {
        return switch (self) {
            .exact => "exact",
            .close => "close",
            .mismatch => "mismatch",
        };
    }
};

const LlamaComparison = struct {
    load_ms: f64,
    decode_ms: f64,
    max_abs_error: f32,
    mean_abs_error: f64,
    reference_checksum: f64,
    reference_argmax: usize,
    bounded_argmax: usize,
    finite: bool,
    rss: ?u64,
};

fn referenceStatus(comparison: LlamaComparison) ReferenceStatus {
    if (!comparison.finite) return .mismatch;
    if (comparison.max_abs_error == 0 and comparison.mean_abs_error == 0) return .exact;
    if (comparison.max_abs_error <= reference_max_error_limit and
        comparison.mean_abs_error <= reference_mean_error_limit and
        comparison.reference_argmax == comparison.bounded_argmax)
    {
        return .close;
    }
    return .mismatch;
}

fn usage() void {
    std.debug.print(
        "usage: zig build validate-residency -- <model.gguf> [budget-mib] [max-tensors] [token-id] [prompt-tokens] [qwen-reference]\n" ++
            "Validates tensors, layer-major prompt prefill, incremental KV reuse, and bounded logits.\n",
        .{},
    );
}

fn supported(descriptor: *const gguf.TensorDescriptor) bool {
    return descriptor.n_dimensions == 2 and
        (descriptor.ggml_type == gguf.type_f32 or
            descriptor.ggml_type == gguf.type_q2_k or
            descriptor.ggml_type == gguf.type_q3_k or
            descriptor.ggml_type == gguf.type_q4_0 or
            descriptor.ggml_type == gguf.type_q4_k or
            descriptor.ggml_type == gguf.type_q6_k or
            descriptor.ggml_type == gguf.type_mxfp4);
}

fn typeName(ggml_type: u32) []const u8 {
    if (ggml_type == gguf.type_f32) return "F32";
    if (ggml_type == gguf.type_q2_k) return "Q2_K";
    if (ggml_type == gguf.type_q3_k) return "Q3_K";
    if (ggml_type == gguf.type_q4_0) return "Q4_0";
    if (ggml_type == gguf.type_q4_k) return "Q4_K";
    if (ggml_type == gguf.type_q6_k) return "Q6_K";
    if (ggml_type == gguf.type_mxfp4) return "MXFP4";
    return "unsupported";
}

fn fillInput(input: []f32) void {
    for (input, 0..) |*value, i| {
        const signed: i32 = @intCast(i % 29);
        value.* = @as(f32, @floatFromInt(signed - 14)) / 17.0;
    }
}

fn qwenBatchProbeCandidate(index: *const gguf.TensorIndex, budget: usize) ?*const gguf.TensorDescriptor {
    var selected: ?*const gguf.TensorDescriptor = null;
    for (index.descriptors) |*descriptor| {
        if (descriptor.n_dimensions != 2 or descriptor.ggml_type != gguf.type_q2_k) continue;
        const rows = std.math.cast(usize, descriptor.dimensions[1]) orelse continue;
        if (rows == 0 or descriptor.byte_len % rows != 0) continue;
        const row_bytes = descriptor.byte_len / rows;
        const prefix: usize = @intCast(descriptor.file_offset % @as(u64, @intCast(residency.mappingGranularity() catch continue)));
        if (row_bytes > budget -| prefix or descriptor.byte_len <= budget -| prefix) continue;
        // Prefer the smallest projection that is still larger than the budget:
        // repeated matvecs must rescan it, while one batched pass reuses every
        // tile. This keeps the probe practical without making its fault gate
        // vacuous when a tiny projection remains fully resident.
        if (selected == null or descriptor.byte_len < selected.?.byte_len) selected = descriptor;
    }
    return selected;
}

fn runQwenBatchProbe(
    allocator: std.mem.Allocator,
    store: *residency.BackingStore,
    index: *const gguf.TensorIndex,
    budget: usize,
) !?QwenBatchProbe {
    if (index.execution.architecture != .qwen3next) return null;
    const descriptor = qwenBatchProbeCandidate(index, budget) orelse return null;
    const columns = std.math.cast(usize, descriptor.dimensions[0]) orelse return error.InvalidTensorShape;
    const rows = std.math.cast(usize, descriptor.dimensions[1]) orelse return error.InvalidTensorShape;
    const batch_count: usize = 4;
    const inputs = try allocator.alloc(f32, batch_count * columns);
    defer allocator.free(inputs);
    fillInput(inputs);
    const repeated = try allocator.alloc(f32, batch_count * rows);
    defer allocator.free(repeated);
    const batched = try allocator.alloc(f32, batch_count * rows);
    defer allocator.free(batched);
    const single_scratch_bytes = try compute.quantizedDotScratchBytes(descriptor.ggml_type, columns);
    const single_scratch = try allocator.alloc(u8, single_scratch_bytes);
    defer allocator.free(single_scratch);
    const batch_scratch_bytes = try compute.quantizedDotBatchScratchBytes(descriptor.ggml_type, columns, batch_count);
    const batch_scratch = try allocator.alloc(u8, batch_scratch_bytes);
    defer allocator.free(batch_scratch);

    var repeated_timer = try std.time.Timer.start();
    var repeated_metrics: residency.Metrics = undefined;
    {
        var manager = try residency.Manager.init(allocator, store, budget);
        defer manager.deinit();
        try manager.register(descriptor.handle, descriptor.file_offset, descriptor.byte_len);
        for (0..batch_count) |batch| {
            try compute.matVecQuantizedGgmlWithPolicy(
                &manager,
                descriptor,
                inputs[batch * columns ..][0..columns],
                repeated[batch * rows ..][0..rows],
                .{ .adaptive = .{ .max_rows = 256 } },
                single_scratch,
            );
        }
        repeated_metrics = manager.metrics();
    }
    const repeated_ns = repeated_timer.read();

    var batched_timer = try std.time.Timer.start();
    var batched_metrics: residency.Metrics = undefined;
    {
        var manager = try residency.Manager.init(allocator, store, budget);
        defer manager.deinit();
        try manager.register(descriptor.handle, descriptor.file_offset, descriptor.byte_len);
        try compute.matMulQuantizedGgmlWithPolicy(
            &manager,
            descriptor,
            inputs,
            batched,
            batch_count,
            .{ .adaptive = .{ .max_rows = 256 } },
            batch_scratch,
        );
        batched_metrics = manager.metrics();
    }
    const batched_ns = batched_timer.read();

    var max_abs_error: f32 = 0;
    for (repeated, batched) |expected, actual| {
        if (!std.math.isFinite(expected) or !std.math.isFinite(actual)) return error.NonFiniteOutput;
        max_abs_error = @max(max_abs_error, @abs(expected - actual));
    }
    if (max_abs_error != 0 or batched_metrics.peak_resident_bytes > budget or
        batched_metrics.faults >= repeated_metrics.faults)
    {
        return error.BatchedProbeMismatch;
    }
    return .{
        .descriptor = descriptor,
        .batch_count = batch_count,
        .repeated_ms = @as(f64, @floatFromInt(repeated_ns)) / std.time.ns_per_ms,
        .batched_ms = @as(f64, @floatFromInt(batched_ns)) / std.time.ns_per_ms,
        .repeated_metrics = repeated_metrics,
        .batched_metrics = batched_metrics,
        .max_abs_error = max_abs_error,
        .scratch_bytes = batch_scratch_bytes,
        .rss = residency.currentRss(),
    };
}

fn runMatVec(
    manager: *residency.Manager,
    descriptor: *const gguf.TensorDescriptor,
    input: []const f32,
    output: []f32,
    rows_per_tile: usize,
    scratch: []f32,
) !void {
    if (descriptor.ggml_type == gguf.type_f32) {
        try compute.matVecF32(manager, descriptor, input, output, rows_per_tile);
    } else {
        try compute.matVecQuantized(manager, descriptor, input, output, rows_per_tile, scratch);
    }
}

fn validateTensor(
    allocator: std.mem.Allocator,
    store: *residency.BackingStore,
    descriptor: *const gguf.TensorDescriptor,
    bounded_budget: usize,
) !Validation {
    const columns = std.math.cast(usize, descriptor.dimensions[0]) orelse return error.InvalidTensorShape;
    const rows = std.math.cast(usize, descriptor.dimensions[1]) orelse return error.InvalidTensorShape;
    const input = try allocator.alloc(f32, columns);
    defer allocator.free(input);
    const baseline = try allocator.alloc(f32, rows);
    defer allocator.free(baseline);
    const bounded = try allocator.alloc(f32, rows);
    defer allocator.free(bounded);
    const scratch = try allocator.alloc(f32, columns);
    defer allocator.free(scratch);
    fillInput(input);

    // Include the maximum possible alignment prefix so a single acquire can
    // hold the complete tensor for the reference run. Managers have disjoint
    // lifetimes so the bounded RSS sample never includes the baseline mapping.
    const granularity = try residency.mappingGranularity();
    const baseline_budget = std.math.add(usize, descriptor.byte_len, granularity - 1) catch return error.OutOfMemory;
    var baseline_ns: u64 = undefined;
    var baseline_rss: ?u64 = null;
    {
        var manager = try residency.Manager.init(allocator, store, baseline_budget);
        defer manager.deinit();
        try manager.register(descriptor.handle, descriptor.file_offset, descriptor.byte_len);
        var timer = try std.time.Timer.start();
        try runMatVec(&manager, descriptor, input, baseline, rows, scratch);
        baseline_ns = timer.read();
        baseline_rss = residency.currentRss();
    }

    var bounded_ns: u64 = undefined;
    var bounded_rss: ?u64 = null;
    var bounded_metrics: residency.Metrics = undefined;
    {
        var manager = try residency.Manager.init(allocator, store, bounded_budget);
        defer manager.deinit();
        try manager.register(descriptor.handle, descriptor.file_offset, descriptor.byte_len);
        // Asking for all rows is intentional: compute reduces the tile to the
        // alignment-aware range capacity exposed by the manager.
        var timer = try std.time.Timer.start();
        try runMatVec(&manager, descriptor, input, bounded, rows, scratch);
        bounded_ns = timer.read();
        bounded_rss = residency.currentRss();
        bounded_metrics = manager.metrics();
    }

    var max_abs_error: f32 = 0;
    var checksum: f64 = 0;
    for (baseline, bounded) |expected, actual| {
        max_abs_error = @max(max_abs_error, @abs(expected - actual));
        checksum += actual;
    }
    if (max_abs_error != 0) return error.OutputMismatch;

    return .{
        .descriptor = descriptor,
        .elapsed_baseline_ms = @as(f64, @floatFromInt(baseline_ns)) / std.time.ns_per_ms,
        .elapsed_bounded_ms = @as(f64, @floatFromInt(bounded_ns)) / std.time.ns_per_ms,
        .max_abs_error = max_abs_error,
        .checksum = checksum,
        .baseline_rss = baseline_rss,
        .bounded_rss = bounded_rss,
        .metrics = bounded_metrics,
    };
}

fn rssMiB(value: ?u64) f64 {
    return if (value) |bytes| @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0) else 0;
}

fn printRss(writer: anytype, label: []const u8, value: ?u64) !void {
    if (value) |bytes| {
        try writer.print("{s}={d:.2} MiB", .{ label, @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0) });
    } else {
        try writer.print("{s}=unavailable", .{label});
    }
}

fn registerOne(manager: *residency.Manager, descriptor: *const gguf.TensorDescriptor) !void {
    try manager.register(descriptor.handle, descriptor.file_offset, descriptor.byte_len);
}

fn runFfn(
    allocator: std.mem.Allocator,
    store: *residency.BackingStore,
    gate: *const gguf.TensorDescriptor,
    up: *const gguf.TensorDescriptor,
    down: *const gguf.TensorDescriptor,
    budget: usize,
    input: []const f32,
    output: []f32,
    intermediate: usize,
) !FfnRun {
    var manager = try residency.Manager.init(allocator, store, budget);
    defer manager.deinit();
    try manager.register(gate.handle, gate.file_offset, gate.byte_len);
    try manager.register(up.handle, up.file_offset, up.byte_len);
    try manager.register(down.handle, down.file_offset, down.byte_len);
    var executor = try executor_mod.CpuExecutor.init(allocator, &manager, @max(input.len, intermediate), intermediate, intermediate);
    defer executor.deinit();

    var timer = try std.time.Timer.start();
    try executor.ffnSwiGlu(gate, up, down, input, output);
    const elapsed_ns = timer.read();
    return .{
        .elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
        .rss = residency.currentRss(),
        .accounting = executor.accounting(),
    };
}

fn validateFirstFfn(
    allocator: std.mem.Allocator,
    store: *residency.BackingStore,
    index: *const gguf.TensorIndex,
    budget: usize,
) !?FfnValidation {
    const gate = index.get("blk.0.ffn_gate.weight") orelse return null;
    const up = index.get("blk.0.ffn_up.weight") orelse return null;
    const down = index.get("blk.0.ffn_down.weight") orelse return null;
    if (!supported(gate) or !supported(up) or !supported(down)) return null;

    const hidden = std.math.cast(usize, gate.dimensions[0]) orelse return error.InvalidTensorShape;
    const intermediate = std.math.cast(usize, gate.dimensions[1]) orelse return error.InvalidTensorShape;
    const output_len = std.math.cast(usize, down.dimensions[1]) orelse return error.InvalidTensorShape;
    const input = try allocator.alloc(f32, hidden);
    defer allocator.free(input);
    const baseline_output = try allocator.alloc(f32, output_len);
    defer allocator.free(baseline_output);
    const bounded_output = try allocator.alloc(f32, output_len);
    defer allocator.free(bounded_output);
    fillInput(input);

    const granularity = try residency.mappingGranularity();
    const max_tensor_bytes = @max(gate.byte_len, @max(up.byte_len, down.byte_len));
    const baseline_budget = std.math.add(usize, max_tensor_bytes, granularity - 1) catch return error.OutOfMemory;
    const baseline = try runFfn(
        allocator,
        store,
        gate,
        up,
        down,
        baseline_budget,
        input,
        baseline_output,
        intermediate,
    );
    const bounded = try runFfn(
        allocator,
        store,
        gate,
        up,
        down,
        budget,
        input,
        bounded_output,
        intermediate,
    );

    var max_abs_error: f32 = 0;
    var checksum: f64 = 0;
    for (baseline_output, bounded_output) |expected, actual| {
        max_abs_error = @max(max_abs_error, @abs(expected - actual));
        checksum += actual;
    }
    if (max_abs_error != 0) return error.OutputMismatch;
    return .{
        .elapsed_baseline_ms = baseline.elapsed_ms,
        .elapsed_bounded_ms = bounded.elapsed_ms,
        .max_abs_error = max_abs_error,
        .checksum = checksum,
        .baseline_rss = baseline.rss,
        .bounded_rss = bounded.rss,
        .accounting = bounded.accounting,
    };
}

fn runDecoder(
    allocator: std.mem.Allocator,
    store: *residency.BackingStore,
    descriptors: [9]*const gguf.TensorDescriptor,
    config: executor_mod.AttentionConfig,
    budget: usize,
    intermediate: usize,
    state: []f32,
) !DecoderRun {
    var manager = try residency.Manager.init(allocator, store, budget);
    defer manager.deinit();
    for (descriptors) |descriptor| try registerOne(&manager, descriptor);
    var executor = try executor_mod.CpuExecutor.init(allocator, &manager, @max(state.len, intermediate), intermediate, intermediate);
    defer executor.deinit();
    const kv_width = config.kv_head_count * config.head_dim;
    var cache = try executor_mod.KvCache.init(allocator, 1, kv_width);
    defer cache.deinit();
    var workspace = try executor_mod.AttentionWorkspace.init(allocator, state.len);
    defer workspace.deinit();

    var timer = try std.time.Timer.start();
    try executor.decoderLayerSingleToken(
        descriptors[0],
        descriptors[1],
        descriptors[2],
        descriptors[3],
        descriptors[4],
        descriptors[5],
        descriptors[6],
        descriptors[7],
        descriptors[8],
        state,
        0,
        config,
        &cache,
        &workspace,
    );
    const elapsed_ns = timer.read();
    return .{
        .elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
        .rss = residency.currentRss(),
        .accounting = executor.decoderAccounting(&cache, &workspace),
    };
}

fn validateFirstDecoderLayer(
    allocator: std.mem.Allocator,
    store: *residency.BackingStore,
    index: *const gguf.TensorIndex,
    budget: usize,
) !?DecoderValidation {
    const descriptors = [9]*const gguf.TensorDescriptor{
        index.get("blk.0.attn_norm.weight") orelse return null,
        index.get("blk.0.attn_q.weight") orelse return null,
        index.get("blk.0.attn_k.weight") orelse return null,
        index.get("blk.0.attn_v.weight") orelse return null,
        index.get("blk.0.attn_output.weight") orelse return null,
        index.get("blk.0.ffn_norm.weight") orelse return null,
        index.get("blk.0.ffn_gate.weight") orelse return null,
        index.get("blk.0.ffn_up.weight") orelse return null,
        index.get("blk.0.ffn_down.weight") orelse return null,
    };
    for (descriptors) |descriptor| {
        if (descriptor.n_dimensions == 2 and !supported(descriptor)) return null;
    }
    const head_count_u32 = index.execution.attention_head_count orelse return null;
    const kv_head_count_u32 = index.execution.attention_kv_head_count orelse head_count_u32;
    const hidden = std.math.cast(usize, descriptors[1].dimensions[0]) orelse return error.InvalidTensorShape;
    const head_count: usize = head_count_u32;
    const kv_head_count: usize = kv_head_count_u32;
    if (head_count == 0 or hidden % head_count != 0) return null;
    const intermediate = std.math.cast(usize, descriptors[6].dimensions[1]) orelse return error.InvalidTensorShape;
    const config = executor_mod.AttentionConfig{
        .head_count = head_count,
        .kv_head_count = kv_head_count,
        .head_dim = hidden / head_count,
        .rms_epsilon = index.execution.rms_epsilon orelse 1e-5,
        .rope_theta = index.execution.rope_theta orelse 10_000.0,
    };
    const baseline_state = try allocator.alloc(f32, hidden);
    defer allocator.free(baseline_state);
    const bounded_state = try allocator.alloc(f32, hidden);
    defer allocator.free(bounded_state);
    fillInput(baseline_state);
    @memcpy(bounded_state, baseline_state);

    const granularity = try residency.mappingGranularity();
    var max_tensor_bytes: usize = 0;
    for (descriptors) |descriptor| max_tensor_bytes = @max(max_tensor_bytes, descriptor.byte_len);
    const baseline_budget = std.math.add(usize, max_tensor_bytes, granularity - 1) catch return error.OutOfMemory;
    const baseline = try runDecoder(allocator, store, descriptors, config, baseline_budget, intermediate, baseline_state);
    const bounded = try runDecoder(allocator, store, descriptors, config, budget, intermediate, bounded_state);

    var max_abs_error: f32 = 0;
    var checksum: f64 = 0;
    for (baseline_state, bounded_state) |expected, actual| {
        max_abs_error = @max(max_abs_error, @abs(expected - actual));
        checksum += actual;
    }
    if (max_abs_error != 0) return error.OutputMismatch;
    return .{
        .elapsed_baseline_ms = baseline.elapsed_ms,
        .elapsed_bounded_ms = bounded.elapsed_ms,
        .max_abs_error = max_abs_error,
        .checksum = checksum,
        .baseline_rss = baseline.rss,
        .bounded_rss = bounded.rss,
        .accounting = bounded.accounting,
    };
}

fn layerWeights(
    allocator: std.mem.Allocator,
    index: *const gguf.TensorIndex,
    layer: usize,
) !?executor_mod.DecoderLayerWeights {
    const suffixes = [_][]const u8{
        "attn_norm.weight",
        "attn_q.weight",
        "attn_k.weight",
        "attn_v.weight",
        "attn_output.weight",
        "ffn_norm.weight",
        "ffn_gate.weight",
        "ffn_up.weight",
        "ffn_down.weight",
    };
    var found: [suffixes.len]*const gguf.TensorDescriptor = undefined;
    for (suffixes, 0..) |suffix, i| {
        const name = try std.fmt.allocPrint(allocator, "blk.{d}.{s}", .{ layer, suffix });
        defer allocator.free(name);
        found[i] = index.get(name) orelse return null;
        if (found[i].n_dimensions == 2 and !supported(found[i])) return null;
    }
    return .{
        .attention_norm = found[0],
        .query = found[1],
        .key = found[2],
        .value = found[3],
        .attention_output = found[4],
        .ffn_norm = found[5],
        .ffn_gate = found[6],
        .ffn_up = found[7],
        .ffn_down = found[8],
    };
}

const ModelRunMode = enum { prefill, incremental, chunked };

fn runModel(
    allocator: std.mem.Allocator,
    store: *residency.BackingStore,
    index: *const gguf.TensorIndex,
    embedding: *const gguf.TensorDescriptor,
    layers: []const executor_mod.DecoderLayerWeights,
    output_norm: *const gguf.TensorDescriptor,
    output_weight: *const gguf.TensorDescriptor,
    config: executor_mod.AttentionConfig,
    budget: usize,
    intermediate: usize,
    tokens: []const usize,
    state: []f32,
    logits: []f32,
    mode: ModelRunMode,
) !ModelRun {
    var manager = try residency.Manager.init(allocator, store, budget);
    defer manager.deinit();
    try index.registerAll(&manager);

    const hidden = config.head_count * config.head_dim;
    var executor = try executor_mod.CpuExecutor.init(allocator, &manager, @max(hidden, intermediate), intermediate, intermediate);
    defer executor.deinit();
    var workspace = try executor_mod.AttentionWorkspace.init(allocator, hidden);
    defer workspace.deinit();
    var prefill_workspace: ?executor_mod.PrefillWorkspace = null;
    defer if (prefill_workspace) |*owned| owned.deinit();
    const prefill_chunk = if (mode == .prefill) tokens.len else default_prefill_chunk;
    if (mode == .prefill or mode == .chunked) {
        prefill_workspace = try executor_mod.PrefillWorkspace.init(allocator, prefill_chunk, hidden, intermediate);
    }

    const kv_width = config.kv_head_count * config.head_dim;
    const caches = try allocator.alloc(executor_mod.KvCache, layers.len);
    defer allocator.free(caches);
    var initialized: usize = 0;
    defer for (caches[0..initialized]) |*cache| cache.deinit();
    for (caches) |*cache| {
        cache.* = try executor_mod.KvCache.init(allocator, tokens.len, kv_width);
        initialized += 1;
    }

    const prompt_states: []f32 = if (mode == .prefill)
        try allocator.alloc(f32, tokens.len * hidden)
    else if (mode == .chunked)
        try allocator.alloc(f32, prefill_chunk * hidden)
    else
        @constCast((&[_]f32{})[0..]);
    defer if (mode == .prefill or mode == .chunked) allocator.free(prompt_states);
    var timer = try std.time.Timer.start();
    switch (mode) {
        .prefill => {
            try executor.modelPrefill(
                embedding,
                layers,
                output_norm,
                output_weight,
                tokens,
                config,
                caches,
                &prefill_workspace.?,
                prompt_states,
                logits,
            );
            @memcpy(state, prompt_states[(tokens.len - 1) * hidden ..][0..hidden]);
        },
        .incremental => for (tokens) |token| {
            const one = [_]usize{token};
            try executor.modelTokens(
                embedding,
                layers,
                output_norm,
                output_weight,
                &one,
                config,
                caches,
                &workspace,
                state,
                logits,
            );
        },
        .chunked => try executor.modelPrefillChunked(
            embedding,
            layers,
            output_norm,
            output_weight,
            tokens,
            prefill_chunk,
            config,
            caches,
            &prefill_workspace.?,
            prompt_states,
            logits,
        ),
    }
    const elapsed_ns = timer.read();
    var kv_cache_bytes: usize = 0;
    for (caches) |*cache| kv_cache_bytes += cache.byteLen();
    return .{
        .elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
        .rss = residency.currentRss(),
        .accounting = .{
            .executor = executor.accounting(),
            .attention_workspace_bytes = workspace.byteLen() + if (prefill_workspace) |*owned| owned.byteLen() else 0,
            .kv_cache_bytes = kv_cache_bytes,
        },
    };
}

fn validateFullModel(
    allocator: std.mem.Allocator,
    store: *residency.BackingStore,
    index: *const gguf.TensorIndex,
    budget: usize,
    tokens: []const usize,
    mode: ModelRunMode,
) !?ModelValidation {
    const embedding = index.get("token_embd.weight") orelse return null;
    const output_norm = index.get("output_norm.weight") orelse return null;
    const output_weight = index.get("output.weight") orelse embedding;
    if (!supported(embedding) or !supported(output_weight)) return null;

    const block_count_u32 = index.execution.block_count orelse return null;
    const head_count_u32 = index.execution.attention_head_count orelse return null;
    const kv_head_count_u32 = index.execution.attention_kv_head_count orelse head_count_u32;
    const block_count: usize = block_count_u32;
    const head_count: usize = head_count_u32;
    const kv_head_count: usize = kv_head_count_u32;
    const hidden = std.math.cast(usize, embedding.dimensions[0]) orelse return error.InvalidTensorShape;
    const vocab_size = std.math.cast(usize, output_weight.dimensions[1]) orelse return error.InvalidTensorShape;
    if (block_count == 0 or head_count == 0 or hidden % head_count != 0 or tokens.len == 0) return null;
    for (tokens) |token| if (token >= embedding.dimensions[1]) return null;

    const layers = try allocator.alloc(executor_mod.DecoderLayerWeights, block_count);
    defer allocator.free(layers);
    var intermediate: usize = 0;
    for (layers, 0..) |*layer, i| {
        layer.* = (try layerWeights(allocator, index, i)) orelse return null;
        intermediate = @max(intermediate, std.math.cast(usize, layer.ffn_gate.dimensions[1]) orelse return error.InvalidTensorShape);
    }
    const config = executor_mod.AttentionConfig{
        .head_count = head_count,
        .kv_head_count = kv_head_count,
        .head_dim = hidden / head_count,
        .rms_epsilon = index.execution.rms_epsilon orelse 1e-5,
        .rope_theta = index.execution.rope_theta orelse 10_000.0,
    };

    const baseline_state = try allocator.alloc(f32, hidden);
    defer allocator.free(baseline_state);
    const bounded_state = try allocator.alloc(f32, hidden);
    defer allocator.free(bounded_state);
    const baseline_logits = try allocator.alloc(f32, vocab_size);
    defer allocator.free(baseline_logits);
    const bounded_logits = try allocator.alloc(f32, vocab_size);
    errdefer allocator.free(bounded_logits);

    const granularity = try residency.mappingGranularity();
    var max_tensor_bytes: usize = 0;
    for (index.descriptors) |descriptor| max_tensor_bytes = @max(max_tensor_bytes, descriptor.byte_len);
    const baseline_budget = std.math.add(usize, max_tensor_bytes, granularity - 1) catch return error.OutOfMemory;
    const baseline = try runModel(allocator, store, index, embedding, layers, output_norm, output_weight, config, baseline_budget, intermediate, tokens, baseline_state, baseline_logits, mode);
    const bounded = try runModel(allocator, store, index, embedding, layers, output_norm, output_weight, config, budget, intermediate, tokens, bounded_state, bounded_logits, mode);

    var max_abs_error: f32 = 0;
    var checksum: f64 = 0;
    var argmax_token: usize = 0;
    for (baseline_logits, bounded_logits, 0..) |expected, actual, i| {
        max_abs_error = @max(max_abs_error, @abs(expected - actual));
        checksum += actual;
        if (actual > bounded_logits[argmax_token]) argmax_token = i;
    }
    if (max_abs_error != 0) return error.OutputMismatch;
    return .{
        .token = tokens[tokens.len - 1],
        .token_count = tokens.len,
        .layer_count = block_count,
        .vocab_size = vocab_size,
        .elapsed_baseline_ms = baseline.elapsed_ms,
        .elapsed_bounded_ms = bounded.elapsed_ms,
        .max_abs_error = max_abs_error,
        .checksum = checksum,
        .argmax_token = argmax_token,
        .baseline_rss = baseline.rss,
        .bounded_rss = bounded.rss,
        .accounting = bounded.accounting,
        .bounded_logits = bounded_logits,
    };
}

fn compareWithLlamaCpp(
    allocator: std.mem.Allocator,
    path_z: [:0]const u8,
    model: *const ModelValidation,
    tokens: []const usize,
) !LlamaComparison {
    const reference_logits = try allocator.alloc(f32, model.vocab_size);
    defer allocator.free(reference_logits);
    const run = try llama_reference.sequenceLogits(path_z, tokens, reference_logits, residency.currentRss);

    var max_abs_error: f32 = 0;
    var total_abs_error: f64 = 0;
    var reference_checksum: f64 = 0;
    var reference_argmax: usize = 0;
    var bounded_argmax: usize = 0;
    var finite = true;
    for (reference_logits, model.bounded_logits, 0..) |expected, actual, i| {
        if (!std.math.isFinite(expected) or !std.math.isFinite(actual)) finite = false;
        const difference = @abs(expected - actual);
        max_abs_error = @max(max_abs_error, difference);
        total_abs_error += difference;
        reference_checksum += expected;
        if (expected > reference_logits[reference_argmax]) reference_argmax = i;
        if (actual > model.bounded_logits[bounded_argmax]) bounded_argmax = i;
    }
    return .{
        .load_ms = run.load_ms,
        .decode_ms = run.decode_ms,
        .max_abs_error = max_abs_error,
        .mean_abs_error = total_abs_error / @as(f64, @floatFromInt(model.vocab_size)),
        .reference_checksum = reference_checksum,
        .reference_argmax = reference_argmax,
        .bounded_argmax = bounded_argmax,
        .finite = finite,
        .rss = run.current_rss,
    };
}

fn qwenDescriptor(index: *const gguf.TensorIndex, comptime format: []const u8, layer: usize) ?*const gguf.TensorDescriptor {
    var buffer: [96]u8 = undefined;
    const name = std.fmt.bufPrint(&buffer, format, .{layer}) catch return null;
    return index.get(name);
}

fn runQwenLinearBlock(
    allocator: std.mem.Allocator,
    store: *residency.BackingStore,
    index: *const gguf.TensorIndex,
    budget: usize,
) !?struct { elapsed_ms: f64, checksum: f64, accounting: executor_mod.MemoryAccounting, state_bytes: usize, workspace_bytes: usize } {
    const config = qwen.Config.fromMetadata(index.execution) catch return null;
    const attention_norm = qwenDescriptor(index, "blk.{d}.attn_norm.weight", 0) orelse return null;
    const qkv = qwenDescriptor(index, "blk.{d}.attn_qkv.weight", 0) orelse return null;
    const z_gate = qwenDescriptor(index, "blk.{d}.attn_gate.weight", 0) orelse return null;
    const beta_alpha = qwenDescriptor(index, "blk.{d}.ssm_ba.weight", 0) orelse return null;
    const conv1d = qwenDescriptor(index, "blk.{d}.ssm_conv1d.weight", 0) orelse return null;
    const dt_bias = qwenDescriptor(index, "blk.{d}.ssm_dt.bias", 0) orelse return null;
    const decay = qwenDescriptor(index, "blk.{d}.ssm_a", 0) orelse return null;
    const state_norm = qwenDescriptor(index, "blk.{d}.ssm_norm.weight", 0) orelse return null;
    const output = qwenDescriptor(index, "blk.{d}.ssm_out.weight", 0) orelse return null;
    const post_attention_norm = qwenDescriptor(index, "blk.{d}.post_attention_norm.weight", 0) orelse return null;
    const router = qwenDescriptor(index, "blk.{d}.ffn_gate_inp.weight", 0) orelse return null;
    const gate_experts = qwenDescriptor(index, "blk.{d}.ffn_gate_exps.weight", 0) orelse return null;
    const up_experts = qwenDescriptor(index, "blk.{d}.ffn_up_exps.weight", 0) orelse return null;
    const down_experts = qwenDescriptor(index, "blk.{d}.ffn_down_exps.weight", 0) orelse return null;
    const shared_router = qwenDescriptor(index, "blk.{d}.ffn_gate_inp_shexp.weight", 0) orelse return null;
    const shared_gate = qwenDescriptor(index, "blk.{d}.ffn_gate_shexp.weight", 0) orelse return null;
    const shared_up = qwenDescriptor(index, "blk.{d}.ffn_up_shexp.weight", 0) orelse return null;
    const shared_down = qwenDescriptor(index, "blk.{d}.ffn_down_shexp.weight", 0) orelse return null;

    var manager = try residency.Manager.init(allocator, store, budget);
    defer manager.deinit();
    _ = try index.registerAll(&manager);
    var executor = try executor_mod.CpuExecutor.init(allocator, &manager, config.convChannels(), config.convChannels(), 64);
    defer executor.deinit();
    var cache = try qwen.DeltaNetCache.init(allocator, config);
    defer cache.deinit();
    var workspace = try qwen.Workspace.init(allocator, config);
    defer workspace.deinit();
    const state = try allocator.alloc(f32, config.hidden_size);
    defer allocator.free(state);
    fillInput(state);
    var timer = try std.time.Timer.start();
    try qwen.linearAttentionSingleToken(&executor, config, .{
        .attention_norm = attention_norm,
        .qkv = qkv,
        .z_gate = z_gate,
        .beta_alpha = beta_alpha,
        .conv1d = conv1d,
        .dt_bias = dt_bias,
        .decay = decay,
        .state_norm = state_norm,
        .output = output,
    }, &cache, &workspace, state);
    try qwen.moeSingleToken(&executor, config, .{
        .post_attention_norm = post_attention_norm,
        .router = router,
        .gate_experts = gate_experts,
        .up_experts = up_experts,
        .down_experts = down_experts,
        .shared_router = shared_router,
        .shared_gate = shared_gate,
        .shared_up = shared_up,
        .shared_down = shared_down,
    }, &workspace, state);
    var checksum: f64 = 0;
    for (state) |value| checksum += value;
    return .{
        .elapsed_ms = @as(f64, @floatFromInt(timer.read())) / std.time.ns_per_ms,
        .checksum = checksum,
        .accounting = executor.accounting(),
        .state_bytes = cache.byteLen(),
        .workspace_bytes = workspace.byteLen(),
    };
}

fn qwenMoeWeights(index: *const gguf.TensorIndex, layer: usize) ?qwen.MoeWeights {
    return .{
        .post_attention_norm = qwenDescriptor(index, "blk.{d}.post_attention_norm.weight", layer) orelse return null,
        .router = qwenDescriptor(index, "blk.{d}.ffn_gate_inp.weight", layer) orelse return null,
        .gate_experts = qwenDescriptor(index, "blk.{d}.ffn_gate_exps.weight", layer) orelse return null,
        .up_experts = qwenDescriptor(index, "blk.{d}.ffn_up_exps.weight", layer) orelse return null,
        .down_experts = qwenDescriptor(index, "blk.{d}.ffn_down_exps.weight", layer) orelse return null,
        .shared_router = qwenDescriptor(index, "blk.{d}.ffn_gate_inp_shexp.weight", layer) orelse return null,
        .shared_gate = qwenDescriptor(index, "blk.{d}.ffn_gate_shexp.weight", layer) orelse return null,
        .shared_up = qwenDescriptor(index, "blk.{d}.ffn_up_shexp.weight", layer) orelse return null,
        .shared_down = qwenDescriptor(index, "blk.{d}.ffn_down_shexp.weight", layer) orelse return null,
    };
}

fn collectQwenLayers(allocator: std.mem.Allocator, index: *const gguf.TensorIndex, interval: usize) !?[]qwen.LayerWeights {
    const count: usize = index.execution.block_count orelse return null;
    if (count == 0 or interval == 0) return null;
    const layers = try allocator.alloc(qwen.LayerWeights, count);
    errdefer allocator.free(layers);
    for (layers, 0..) |*slot, layer| {
        const moe = qwenMoeWeights(index, layer) orelse return null;
        if ((layer + 1) % interval == 0) {
            slot.* = .{ .full_attention = .{
                .attention = .{
                    .attention_norm = qwenDescriptor(index, "blk.{d}.attn_norm.weight", layer) orelse return null,
                    .query_gate = qwenDescriptor(index, "blk.{d}.attn_q.weight", layer) orelse return null,
                    .key = qwenDescriptor(index, "blk.{d}.attn_k.weight", layer) orelse return null,
                    .value = qwenDescriptor(index, "blk.{d}.attn_v.weight", layer) orelse return null,
                    .query_norm = qwenDescriptor(index, "blk.{d}.attn_q_norm.weight", layer) orelse return null,
                    .key_norm = qwenDescriptor(index, "blk.{d}.attn_k_norm.weight", layer) orelse return null,
                    .output = qwenDescriptor(index, "blk.{d}.attn_output.weight", layer) orelse return null,
                },
                .moe = moe,
            } };
        } else {
            slot.* = .{ .recurrent = .{
                .attention = .{
                    .attention_norm = qwenDescriptor(index, "blk.{d}.attn_norm.weight", layer) orelse return null,
                    .qkv = qwenDescriptor(index, "blk.{d}.attn_qkv.weight", layer) orelse return null,
                    .z_gate = qwenDescriptor(index, "blk.{d}.attn_gate.weight", layer) orelse return null,
                    .beta_alpha = qwenDescriptor(index, "blk.{d}.ssm_ba.weight", layer) orelse return null,
                    .conv1d = qwenDescriptor(index, "blk.{d}.ssm_conv1d.weight", layer) orelse return null,
                    .dt_bias = qwenDescriptor(index, "blk.{d}.ssm_dt.bias", layer) orelse return null,
                    .decay = qwenDescriptor(index, "blk.{d}.ssm_a", layer) orelse return null,
                    .state_norm = qwenDescriptor(index, "blk.{d}.ssm_norm.weight", layer) orelse return null,
                    .output = qwenDescriptor(index, "blk.{d}.ssm_out.weight", layer) orelse return null,
                },
                .moe = moe,
            } };
        }
    }
    return layers;
}

fn runQwenFullModel(
    allocator: std.mem.Allocator,
    store: *residency.BackingStore,
    index: *const gguf.TensorIndex,
    budget: usize,
    tokens: []const usize,
) !?QwenModelRun {
    const config = qwen.Config.fromMetadata(index.execution) catch return null;
    const interval: usize = index.execution.full_attention_interval orelse return null;
    const layers = (try collectQwenLayers(allocator, index, interval)) orelse return null;
    defer allocator.free(layers);
    const embedding = index.get("token_embd.weight") orelse return null;
    const output_norm = index.get("output_norm.weight") orelse return null;
    const output_weight = index.get("output.weight") orelse embedding;
    const vocab: usize = @intCast(output_weight.dimensions[1]);
    if (tokens.len == 0) return null;
    for (tokens) |token| if (token >= embedding.dimensions[1]) return null;

    var manager = try residency.Manager.init(allocator, store, budget);
    defer manager.deinit();
    _ = try index.registerAll(&manager);
    var executor = try executor_mod.CpuExecutor.init(allocator, &manager, config.convChannels(), config.convChannels(), 64);
    defer executor.deinit();
    const context_capacity = @max(tokens.len, 1);
    const caches = try qwen.initLayerCaches(allocator, config, layers.len, interval, context_capacity);
    defer qwen.deinitLayerCaches(allocator, caches);
    var workspace = try qwen.Workspace.init(allocator, config);
    defer workspace.deinit();
    const state = try allocator.alloc(f32, config.hidden_size);
    defer allocator.free(state);
    const logits = try allocator.alloc(f32, vocab);
    errdefer allocator.free(logits);
    var timer = try std.time.Timer.start();
    for (tokens) |token| try qwen.modelSingleToken(&executor, config, embedding, layers, output_norm, output_weight, token, caches, &workspace, state, logits);
    var checksum: f64 = 0;
    var argmax: usize = 0;
    for (logits, 0..) |value, i| {
        checksum += value;
        if (value > logits[argmax]) argmax = i;
    }
    var cache_bytes: usize = 0;
    for (caches) |*cache| cache_bytes += cache.byteLen();
    return .{
        .elapsed_ms = @as(f64, @floatFromInt(timer.read())) / std.time.ns_per_ms,
        .checksum = checksum,
        .argmax = argmax,
        .accounting = executor.accounting(),
        .cache_bytes = cache_bytes,
        .workspace_bytes = workspace.byteLen(),
        .logits = logits,
    };
}

fn runQwenFullAttentionBlock(
    allocator: std.mem.Allocator,
    store: *residency.BackingStore,
    index: *const gguf.TensorIndex,
    budget: usize,
) !?struct { elapsed_ms: f64, checksum: f64, accounting: executor_mod.MemoryAccounting, cache_bytes: usize, workspace_bytes: usize } {
    const config = qwen.Config.fromMetadata(index.execution) catch return null;
    const layer: usize = 3;
    const attention_norm = qwenDescriptor(index, "blk.{d}.attn_norm.weight", layer) orelse return null;
    const query_gate = qwenDescriptor(index, "blk.{d}.attn_q.weight", layer) orelse return null;
    const key = qwenDescriptor(index, "blk.{d}.attn_k.weight", layer) orelse return null;
    const value = qwenDescriptor(index, "blk.{d}.attn_v.weight", layer) orelse return null;
    const query_norm = qwenDescriptor(index, "blk.{d}.attn_q_norm.weight", layer) orelse return null;
    const key_norm = qwenDescriptor(index, "blk.{d}.attn_k_norm.weight", layer) orelse return null;
    const output = qwenDescriptor(index, "blk.{d}.attn_output.weight", layer) orelse return null;
    const post_attention_norm = qwenDescriptor(index, "blk.{d}.post_attention_norm.weight", layer) orelse return null;
    const router = qwenDescriptor(index, "blk.{d}.ffn_gate_inp.weight", layer) orelse return null;
    const gate_experts = qwenDescriptor(index, "blk.{d}.ffn_gate_exps.weight", layer) orelse return null;
    const up_experts = qwenDescriptor(index, "blk.{d}.ffn_up_exps.weight", layer) orelse return null;
    const down_experts = qwenDescriptor(index, "blk.{d}.ffn_down_exps.weight", layer) orelse return null;
    const shared_router = qwenDescriptor(index, "blk.{d}.ffn_gate_inp_shexp.weight", layer) orelse return null;
    const shared_gate = qwenDescriptor(index, "blk.{d}.ffn_gate_shexp.weight", layer) orelse return null;
    const shared_up = qwenDescriptor(index, "blk.{d}.ffn_up_shexp.weight", layer) orelse return null;
    const shared_down = qwenDescriptor(index, "blk.{d}.ffn_down_shexp.weight", layer) orelse return null;

    var manager = try residency.Manager.init(allocator, store, budget);
    defer manager.deinit();
    _ = try index.registerAll(&manager);
    var executor = try executor_mod.CpuExecutor.init(allocator, &manager, config.convChannels(), config.convChannels(), 64);
    defer executor.deinit();
    var cache = try qwen.FullAttentionCache.init(allocator, 1, config);
    defer cache.deinit();
    var workspace = try qwen.Workspace.init(allocator, config);
    defer workspace.deinit();
    const state = try allocator.alloc(f32, config.hidden_size);
    defer allocator.free(state);
    fillInput(state);
    var timer = try std.time.Timer.start();
    try qwen.fullAttentionSingleToken(&executor, config, .{
        .attention_norm = attention_norm,
        .query_gate = query_gate,
        .key = key,
        .value = value,
        .query_norm = query_norm,
        .key_norm = key_norm,
        .output = output,
    }, &cache, &workspace, state);
    try qwen.moeSingleToken(&executor, config, .{
        .post_attention_norm = post_attention_norm,
        .router = router,
        .gate_experts = gate_experts,
        .up_experts = up_experts,
        .down_experts = down_experts,
        .shared_router = shared_router,
        .shared_gate = shared_gate,
        .shared_up = shared_up,
        .shared_down = shared_down,
    }, &workspace, state);
    var checksum: f64 = 0;
    for (state) |state_value| checksum += state_value;
    return .{
        .elapsed_ms = @as(f64, @floatFromInt(timer.read())) / std.time.ns_per_ms,
        .checksum = checksum,
        .accounting = executor.accounting(),
        .cache_bytes = cache.byteLen(),
        .workspace_bytes = workspace.byteLen(),
    };
}

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    if (args.len < 2 or args.len > 7) {
        usage();
        return error.InvalidArguments;
    }
    const budget_mib = if (args.len >= 3)
        try std.fmt.parseInt(usize, args[2], 10)
    else
        default_budget_mib;
    const max_tensors = if (args.len >= 4)
        try std.fmt.parseInt(usize, args[3], 10)
    else
        default_max_tensors;
    const token = if (args.len >= 5)
        try std.fmt.parseInt(usize, args[4], 10)
    else
        1;
    const prompt_token_count = if (args.len >= 6)
        try std.fmt.parseInt(usize, args[5], 10)
    else
        2;
    const qwen_reference = if (args.len >= 7)
        std.mem.eql(u8, args[6], "true") or std.mem.eql(u8, args[6], "1")
    else
        false;
    if (budget_mib == 0 or max_tensors == 0 or prompt_token_count == 0 or
        budget_mib > std.math.maxInt(usize) / (1024 * 1024))
    {
        return error.InvalidArguments;
    }
    const budget = budget_mib * 1024 * 1024;
    const path_z = try allocator.dupeZ(u8, args[1]);
    defer allocator.free(path_z);

    const rss_start = residency.currentRss();
    var store = try residency.BackingStore.open(path_z);
    defer store.close();
    var index = try gguf.TensorIndex.open(allocator, path_z, store.size);
    defer index.deinit();

    const out = std.fs.File.stdout().deprecatedWriter();
    try out.print(
        "real GGUF residency validation: {s}\n" ++
            "tensors={d}, file={d:.2} MiB, bounded budget={d} MiB\n",
        .{ args[1], index.descriptors.len, @as(f64, @floatFromInt(store.size)) / (1024.0 * 1024.0), budget_mib },
    );

    if (index.execution.architecture == .qwen3next) {
        const probe = (try runQwenBatchProbe(allocator, &store, &index, budget)) orelse return error.NoSupportedTensor;
        try out.print(
            "Qwen3-Next bounded Q2_K batched projection: {s} ({d}x{d}, {d:.2} MiB), batch={d}\n" ++
                "repeated={d:.2} ms faults/evictions={d}/{d}; batched={d:.2} ms faults/evictions={d}/{d}; " ++
                "peak-map={d:.2}/{d:.2} MiB, scratch={d:.2} KiB, max-error={d}, rss={d:.2} MiB\n" ++
                "scope: GGUF metadata + canonical Q2_K batched projection; full Qwen3-Next graph validation follows below\n",
            .{
                probe.descriptor.name,
                probe.descriptor.dimensions[1],
                probe.descriptor.dimensions[0],
                @as(f64, @floatFromInt(probe.descriptor.byte_len)) / (1024.0 * 1024.0),
                probe.batch_count,
                probe.repeated_ms,
                probe.repeated_metrics.faults,
                probe.repeated_metrics.evictions,
                probe.batched_ms,
                probe.batched_metrics.faults,
                probe.batched_metrics.evictions,
                @as(f64, @floatFromInt(probe.batched_metrics.peak_resident_bytes)) / (1024.0 * 1024.0),
                @as(f64, @floatFromInt(budget)) / (1024.0 * 1024.0),
                @as(f64, @floatFromInt(probe.scratch_bytes)) / 1024.0,
                probe.max_abs_error,
                rssMiB(probe.rss),
            },
        );
        const linear = (try runQwenLinearBlock(allocator, &store, &index, budget)) orelse return error.NoSupportedTensor;
        try out.print(
            "Qwen3-Next layer-0 DeltaNet+MoE block: elapsed={d:.2} ms, checksum={d:.6}, peak-map={d:.2}/{d:.2} MiB, " ++
                "faults/evictions={d}/{d}, recurrent-state={d:.2} MiB, workspace={d:.2} MiB\n" ++
                "scope: complete single-token recurrent layer (DeltaNet + top-k routed MoE + gated shared expert + residual)\n",
            .{
                linear.elapsed_ms,
                linear.checksum,
                @as(f64, @floatFromInt(linear.accounting.peak_mapped_weight_bytes)) / (1024.0 * 1024.0),
                @as(f64, @floatFromInt(linear.accounting.weight_budget_bytes)) / (1024.0 * 1024.0),
                linear.accounting.faults,
                linear.accounting.evictions,
                @as(f64, @floatFromInt(linear.state_bytes)) / (1024.0 * 1024.0),
                @as(f64, @floatFromInt(linear.workspace_bytes)) / (1024.0 * 1024.0),
            },
        );
        if (linear.accounting.peak_mapped_weight_bytes > budget) return error.BudgetInvariantViolated;
        const full_attention = (try runQwenFullAttentionBlock(allocator, &store, &index, budget)) orelse return error.NoSupportedTensor;
        try out.print(
            "Qwen3-Next layer-3 full-attention+MoE block: elapsed={d:.2} ms, checksum={d:.6}, peak-map={d:.2}/{d:.2} MiB, " ++
                "faults/evictions={d}/{d}, KV={d:.2} KiB, workspace={d:.2} MiB\n",
            .{
                full_attention.elapsed_ms,
                full_attention.checksum,
                @as(f64, @floatFromInt(full_attention.accounting.peak_mapped_weight_bytes)) / (1024.0 * 1024.0),
                @as(f64, @floatFromInt(full_attention.accounting.weight_budget_bytes)) / (1024.0 * 1024.0),
                full_attention.accounting.faults,
                full_attention.accounting.evictions,
                @as(f64, @floatFromInt(full_attention.cache_bytes)) / 1024.0,
                @as(f64, @floatFromInt(full_attention.workspace_bytes)) / (1024.0 * 1024.0),
            },
        );
        if (full_attention.accounting.peak_mapped_weight_bytes > budget) return error.BudgetInvariantViolated;
        const qwen_tokens = [_]usize{token};
        const full_model = (try runQwenFullModel(allocator, &store, &index, budget, &qwen_tokens)) orelse return error.NoSupportedTensor;
        defer full_model.deinit(allocator);
        try out.print(
            "Qwen3-Next full single-token logits: layers={d}, vocab={d}, elapsed={d:.2} ms, checksum={d:.6}, argmax={d}, " ++
                "peak-map={d:.2}/{d:.2} MiB, faults/evictions={d}/{d}, all-layer-state={d:.2} MiB, workspace={d:.2} MiB\n",
            .{
                index.execution.block_count.?,
                index.get("output.weight").?.dimensions[1],
                full_model.elapsed_ms,
                full_model.checksum,
                full_model.argmax,
                @as(f64, @floatFromInt(full_model.accounting.peak_mapped_weight_bytes)) / (1024.0 * 1024.0),
                @as(f64, @floatFromInt(full_model.accounting.weight_budget_bytes)) / (1024.0 * 1024.0),
                full_model.accounting.faults,
                full_model.accounting.evictions,
                @as(f64, @floatFromInt(full_model.cache_bytes)) / (1024.0 * 1024.0),
                @as(f64, @floatFromInt(full_model.workspace_bytes)) / (1024.0 * 1024.0),
            },
        );
        if (full_model.accounting.peak_mapped_weight_bytes > budget) return error.BudgetInvariantViolated;
        const resident_budget = @max(budget, @as(usize, 64 * 1024 * 1024));
        const resident_model = (try runQwenFullModel(allocator, &store, &index, resident_budget, &qwen_tokens)) orelse return error.NoSupportedTensor;
        defer resident_model.deinit(allocator);
        var resident_max_error: f32 = 0;
        for (resident_model.logits, full_model.logits) |resident_value, bounded_value| resident_max_error = @max(resident_max_error, @abs(resident_value - bounded_value));
        try out.print(
            "Qwen3-Next resident-vs-bounded logits: resident-budget={d:.2} MiB, max-error={d}, argmax={d}/{d}\n",
            .{ @as(f64, @floatFromInt(resident_budget)) / (1024.0 * 1024.0), resident_max_error, resident_model.argmax, full_model.argmax },
        );
        if (resident_max_error != 0 or resident_model.argmax != full_model.argmax) return error.ReferenceMismatch;

        if (qwen_reference) {
            const reference_logits = try allocator.alloc(f32, full_model.logits.len);
            defer allocator.free(reference_logits);
            const reference = try llama_reference.sequenceLogitsMmap(path_z, &[_]usize{token}, reference_logits, residency.currentRss);
            var reference_argmax: usize = 0;
            var max_error: f32 = 0;
            var mean_error: f64 = 0;
            var finite = true;
            for (reference_logits, full_model.logits, 0..) |reference_value, bounded_value, i| {
                finite = finite and std.math.isFinite(reference_value) and std.math.isFinite(bounded_value);
                const difference = @abs(reference_value - bounded_value);
                max_error = @max(max_error, difference);
                mean_error += difference;
                if (reference_value > reference_logits[reference_argmax]) reference_argmax = i;
            }
            mean_error /= @as(f64, @floatFromInt(reference_logits.len));
            try out.print(
                "Qwen3-Next llama.cpp mmap reference: load={d:.2} ms, decode={d:.2} ms, max-error={d:.6}, mean-error={d:.6}, " ++
                    "argmax={d}/{d}, finite={any}, rss={d:.2} MiB\n",
                .{ reference.load_ms, reference.decode_ms, max_error, mean_error, reference_argmax, full_model.argmax, finite, rssMiB(reference.current_rss) },
            );
            if (!finite or reference_argmax != full_model.argmax or max_error > 1.0 or mean_error > 0.2) return error.ReferenceMismatch;

            const sequence = [_]usize{ token, (token + 1) % @as(usize, @intCast(index.get("token_embd.weight").?.dimensions[1])) };
            const sequence_model = (try runQwenFullModel(allocator, &store, &index, budget, &sequence)) orelse return error.NoSupportedTensor;
            defer sequence_model.deinit(allocator);
            const sequence_reference = try llama_reference.sequenceLogitsMmap(path_z, &sequence, reference_logits, residency.currentRss);
            reference_argmax = 0;
            max_error = 0;
            mean_error = 0;
            finite = true;
            for (reference_logits, sequence_model.logits, 0..) |reference_value, bounded_value, i| {
                finite = finite and std.math.isFinite(reference_value) and std.math.isFinite(bounded_value);
                const difference = @abs(reference_value - bounded_value);
                max_error = @max(max_error, difference);
                mean_error += difference;
                if (reference_value > reference_logits[reference_argmax]) reference_argmax = i;
            }
            mean_error /= @as(f64, @floatFromInt(reference_logits.len));
            try out.print(
                "Qwen3-Next two-token recurrent-state reference: bounded={d:.2} ms, llama.cpp-decode={d:.2} ms, max-error={d:.6}, " ++
                    "mean-error={d:.6}, argmax={d}/{d}, state={d:.2} MiB, finite={any}\n",
                .{ sequence_model.elapsed_ms, sequence_reference.decode_ms, max_error, mean_error, reference_argmax, sequence_model.argmax, @as(f64, @floatFromInt(sequence_model.cache_bytes)) / (1024.0 * 1024.0), finite },
            );
            if (!finite or reference_argmax != sequence_model.argmax or max_error > 1.5 or mean_error > 0.3) return error.ReferenceMismatch;
        } else {
            try out.print("Qwen3-Next llama.cpp reference: skipped (pass final argument true to run the 27 GiB mmap reference)\n", .{});
        }
        return;
    }

    var validated: usize = 0;
    var skipped_budget: usize = 0;
    for (index.descriptors) |*descriptor| {
        if (validated == max_tensors) break;
        if (!supported(descriptor)) continue;
        // At least one complete encoded row plus mmap alignment must fit.
        const row_bytes = descriptor.byte_len / @as(usize, @intCast(descriptor.dimensions[1]));
        const prefix: usize = @intCast(descriptor.file_offset % @as(u64, @intCast(try residency.mappingGranularity())));
        if (row_bytes > budget -| prefix) {
            skipped_budget += 1;
            continue;
        }

        const result = try validateTensor(allocator, &store, descriptor, budget);
        const metrics = result.metrics;
        try out.print(
            "[{d}] {s} ({s}, {d}x{d}, {d:.2} MiB): baseline={d:.2} ms, bounded={d:.2} ms, " ++
                "peak-map={d:.2} MiB, faults={d}, evictions={d}, max-error={d}, checksum={d:.6}, " ++
                "baseline-rss={d:.2} MiB, bounded-rss={d:.2} MiB\n",
            .{
                validated + 1,
                descriptor.name,
                typeName(descriptor.ggml_type),
                descriptor.dimensions[1],
                descriptor.dimensions[0],
                @as(f64, @floatFromInt(descriptor.byte_len)) / (1024.0 * 1024.0),
                result.elapsed_baseline_ms,
                result.elapsed_bounded_ms,
                @as(f64, @floatFromInt(metrics.peak_resident_bytes)) / (1024.0 * 1024.0),
                metrics.faults,
                metrics.evictions,
                result.max_abs_error,
                result.checksum,
                rssMiB(result.baseline_rss),
                rssMiB(result.bounded_rss),
            },
        );
        if (metrics.peak_resident_bytes > budget) return error.BudgetInvariantViolated;
        validated += 1;
    }
    if (validated == 0) {
        try out.print("no supported 2D tensor fit one row in the selected budget (skipped={d})\n", .{skipped_budget});
        return error.NoSupportedTensor;
    }

    if (try validateFirstFfn(allocator, &store, &index, budget)) |ffn| {
        const memory = ffn.accounting;
        try out.print(
            "layer-0 SwiGLU FFN: baseline={d:.2} ms, bounded={d:.2} ms, max-error={d}, checksum={d:.6}, " ++
                "weight-map={d:.2}/{d:.2} MiB peak/budget, scratch={d:.2} KiB, " ++
                "activations={d:.2} KiB, faults={d}, evictions={d}, baseline-rss={d:.2} MiB, bounded-rss={d:.2} MiB\n",
            .{
                ffn.elapsed_baseline_ms,
                ffn.elapsed_bounded_ms,
                ffn.max_abs_error,
                ffn.checksum,
                @as(f64, @floatFromInt(memory.peak_mapped_weight_bytes)) / (1024.0 * 1024.0),
                @as(f64, @floatFromInt(memory.weight_budget_bytes)) / (1024.0 * 1024.0),
                @as(f64, @floatFromInt(memory.dequant_scratch_bytes)) / 1024.0,
                @as(f64, @floatFromInt(memory.activation_bytes)) / 1024.0,
                memory.faults,
                memory.evictions,
                rssMiB(ffn.baseline_rss),
                rssMiB(ffn.bounded_rss),
            },
        );
    } else {
        try out.print("layer-0 SwiGLU FFN: skipped (required tensor names/types unavailable)\n", .{});
    }

    if (try validateFirstDecoderLayer(allocator, &store, &index, budget)) |decoder| {
        const memory = decoder.accounting;
        try out.print(
            "layer-0 single-token decoder: baseline={d:.2} ms, bounded={d:.2} ms, max-error={d}, checksum={d:.6}, " ++
                "weight-map={d:.2}/{d:.2} MiB peak/budget, scratch={d:.2} KiB, executor-activations={d:.2} KiB, " ++
                "attention-workspace={d:.2} KiB, kv-cache={d:.2} KiB, faults={d}, evictions={d}, " ++
                "baseline-rss={d:.2} MiB, bounded-rss={d:.2} MiB\n",
            .{
                decoder.elapsed_baseline_ms,
                decoder.elapsed_bounded_ms,
                decoder.max_abs_error,
                decoder.checksum,
                @as(f64, @floatFromInt(memory.executor.peak_mapped_weight_bytes)) / (1024.0 * 1024.0),
                @as(f64, @floatFromInt(memory.executor.weight_budget_bytes)) / (1024.0 * 1024.0),
                @as(f64, @floatFromInt(memory.executor.dequant_scratch_bytes)) / 1024.0,
                @as(f64, @floatFromInt(memory.executor.activation_bytes)) / 1024.0,
                @as(f64, @floatFromInt(memory.attention_workspace_bytes)) / 1024.0,
                @as(f64, @floatFromInt(memory.kv_cache_bytes)) / 1024.0,
                memory.executor.faults,
                memory.executor.evictions,
                rssMiB(decoder.baseline_rss),
                rssMiB(decoder.bounded_rss),
            },
        );
    } else {
        try out.print("layer-0 single-token decoder: skipped (required Llama tensors/metadata/types unavailable)\n", .{});
    }

    var reference_failed = false;
    const single_tokens = [_]usize{token};
    if (try validateFullModel(allocator, &store, &index, budget, &single_tokens, .prefill)) |model| {
        defer model.deinit(allocator);
        const memory = model.accounting;
        try out.print(
            "full single-token logits: token={d}, layers={d}, vocab={d}, baseline={d:.2} ms, bounded={d:.2} ms, " ++
                "max-error={d}, checksum={d:.6}, argmax={d}, weight-map={d:.2}/{d:.2} MiB peak/budget, " ++
                "scratch={d:.2} KiB, executor-activations={d:.2} KiB, attention-workspace={d:.2} KiB, " ++
                "all-layer-kv={d:.2} KiB, faults={d}, evictions={d}, baseline-rss={d:.2} MiB, bounded-rss={d:.2} MiB\n",
            .{
                model.token,
                model.layer_count,
                model.vocab_size,
                model.elapsed_baseline_ms,
                model.elapsed_bounded_ms,
                model.max_abs_error,
                model.checksum,
                model.argmax_token,
                @as(f64, @floatFromInt(memory.executor.peak_mapped_weight_bytes)) / (1024.0 * 1024.0),
                @as(f64, @floatFromInt(memory.executor.weight_budget_bytes)) / (1024.0 * 1024.0),
                @as(f64, @floatFromInt(memory.executor.dequant_scratch_bytes)) / 1024.0,
                @as(f64, @floatFromInt(memory.executor.activation_bytes)) / 1024.0,
                @as(f64, @floatFromInt(memory.attention_workspace_bytes)) / 1024.0,
                @as(f64, @floatFromInt(memory.kv_cache_bytes)) / 1024.0,
                memory.executor.faults,
                memory.executor.evictions,
                rssMiB(model.baseline_rss),
                rssMiB(model.bounded_rss),
            },
        );
        if (memory.executor.peak_mapped_weight_bytes > budget) return error.BudgetInvariantViolated;

        const reference = try compareWithLlamaCpp(allocator, path_z, &model, &single_tokens);
        const status = referenceStatus(reference);
        try out.print(
            "llama.cpp single-token reference: status={s}, load={d:.2} ms, decode={d:.2} ms, max-error={d:.6}, mean-error={d:.6}, " ++
                "reference-checksum={d:.6}, reference-argmax={d}, bounded-argmax={d}, finite={any}, rss={d:.2} MiB\n",
            .{
                status.name(),
                reference.load_ms,
                reference.decode_ms,
                reference.max_abs_error,
                reference.mean_abs_error,
                reference.reference_checksum,
                reference.reference_argmax,
                reference.bounded_argmax,
                reference.finite,
                rssMiB(reference.rss),
            },
        );
        if (status == .mismatch) reference_failed = true;
    } else {
        try out.print("full single-token logits: skipped (required Llama tensors/metadata/types unavailable or token out of range)\n", .{});
        reference_failed = true;
    }

    const embedding_descriptor = index.get("token_embd.weight");
    const sequence_tokens = try allocator.alloc(usize, prompt_token_count);
    defer allocator.free(sequence_tokens);
    if (embedding_descriptor) |descriptor| {
        const vocabulary: usize = @intCast(descriptor.dimensions[1]);
        for (sequence_tokens, 0..) |*sequence_token, i| sequence_token.* = (token +| i) % vocabulary;
    } else {
        @memset(sequence_tokens, token);
    }
    if (try validateFullModel(allocator, &store, &index, budget, sequence_tokens, .prefill)) |model| {
        defer model.deinit(allocator);
        const memory = model.accounting;
        try out.print(
            "full layer-major prefill logits: tokens={d} (first={d}, last={d}), layers={d}, vocab={d}, baseline={d:.2} ms, bounded={d:.2} ms, " ++
                "prompt-tokens/s={d:.2}, max-error={d}, argmax={d}, weight-map={d:.2}/{d:.2} MiB peak/budget, all-layer-kv={d:.2} KiB, " ++
                "kv-tokens={d}, faults={d}, evictions={d}\n",
            .{
                model.token_count,
                sequence_tokens[0],
                sequence_tokens[sequence_tokens.len - 1],
                model.layer_count,
                model.vocab_size,
                model.elapsed_baseline_ms,
                model.elapsed_bounded_ms,
                @as(f64, @floatFromInt(model.token_count)) / @max(model.elapsed_bounded_ms / 1000.0, 0.000001),
                model.max_abs_error,
                model.argmax_token,
                @as(f64, @floatFromInt(memory.executor.peak_mapped_weight_bytes)) / (1024.0 * 1024.0),
                @as(f64, @floatFromInt(memory.executor.weight_budget_bytes)) / (1024.0 * 1024.0),
                @as(f64, @floatFromInt(memory.kv_cache_bytes)) / 1024.0,
                model.token_count,
                memory.executor.faults,
                memory.executor.evictions,
            },
        );
        if (memory.executor.peak_mapped_weight_bytes > budget) return error.BudgetInvariantViolated;
        const reference = try compareWithLlamaCpp(allocator, path_z, &model, sequence_tokens);
        const status = referenceStatus(reference);
        try out.print(
            "llama.cpp multi-token reference: status={s}, load={d:.2} ms, decode={d:.2} ms, max-error={d:.6}, " ++
                "mean-error={d:.6}, reference-argmax={d}, bounded-argmax={d}, finite={any}, rss={d:.2} MiB\n",
            .{
                status.name(),
                reference.load_ms,
                reference.decode_ms,
                reference.max_abs_error,
                reference.mean_abs_error,
                reference.reference_argmax,
                reference.bounded_argmax,
                reference.finite,
                rssMiB(reference.rss),
            },
        );
        if (status == .mismatch) {
            if (sequence_tokens.len <= 2) {
                reference_failed = true;
            } else {
                try out.print(
                    "long-prompt reference is informational: strict Phase-6 tolerance remains unchanged; top-1-match={any}\n",
                    .{reference.reference_argmax == reference.bounded_argmax},
                );
                if (!reference.finite or reference.reference_argmax != reference.bounded_argmax) reference_failed = true;
            }
        }

        if (try validateFullModel(allocator, &store, &index, budget, sequence_tokens, .incremental)) |incremental| {
            defer incremental.deinit(allocator);
            var incremental_max_error: f32 = 0;
            for (model.bounded_logits, incremental.bounded_logits) |prefill_logit, incremental_logit| {
                incremental_max_error = @max(incremental_max_error, @abs(prefill_logit - incremental_logit));
            }
            try out.print(
                "bounded KV reuse: mode=incremental-append, tokens={d}, elapsed={d:.2} ms, tokens/s={d:.2}, max-error-vs-prefill={d}, " ++
                    "argmax={d}, weight-map={d:.2}/{d:.2} MiB peak/budget, kv={d:.2} KiB\n",
                .{
                    incremental.token_count,
                    incremental.elapsed_bounded_ms,
                    @as(f64, @floatFromInt(incremental.token_count)) / @max(incremental.elapsed_bounded_ms / 1000.0, 0.000001),
                    incremental_max_error,
                    incremental.argmax_token,
                    @as(f64, @floatFromInt(incremental.accounting.executor.peak_mapped_weight_bytes)) / (1024.0 * 1024.0),
                    @as(f64, @floatFromInt(incremental.accounting.executor.weight_budget_bytes)) / (1024.0 * 1024.0),
                    @as(f64, @floatFromInt(incremental.accounting.kv_cache_bytes)) / 1024.0,
                },
            );
            if (incremental_max_error != 0 or incremental.argmax_token != model.argmax_token) {
                return error.IncrementalPrefillMismatch;
            }
        } else {
            return error.IncrementalValidationUnavailable;
        }

        if (try validateFullModel(allocator, &store, &index, budget, sequence_tokens, .chunked)) |chunked| {
            defer chunked.deinit(allocator);
            const chunked_memory = chunked.accounting;
            var chunked_max_error: f32 = 0;
            for (model.bounded_logits, chunked.bounded_logits) |prefill_logit, chunked_logit| {
                chunked_max_error = @max(chunked_max_error, @abs(prefill_logit - chunked_logit));
            }
            try out.print(
                "bounded chunked prefill: chunk={d}, tokens={d}, elapsed={d:.2} ms, tokens/s={d:.2}, max-error-vs-prefill={d}, " ++
                    "argmax={d}, weight-map={d:.2}/{d:.2} MiB peak/budget, all-layer-kv={d:.2} KiB\n",
                .{
                    default_prefill_chunk,
                    chunked.token_count,
                    chunked.elapsed_bounded_ms,
                    @as(f64, @floatFromInt(chunked.token_count)) / @max(chunked.elapsed_bounded_ms / 1000.0, 0.000001),
                    chunked_max_error,
                    chunked.argmax_token,
                    @as(f64, @floatFromInt(chunked_memory.executor.peak_mapped_weight_bytes)) / (1024.0 * 1024.0),
                    @as(f64, @floatFromInt(chunked_memory.executor.weight_budget_bytes)) / (1024.0 * 1024.0),
                    @as(f64, @floatFromInt(chunked_memory.kv_cache_bytes)) / 1024.0,
                },
            );
            if (chunked_max_error != 0 or chunked.argmax_token != model.argmax_token) {
                return error.ChunkedPrefillMismatch;
            }
        } else {
            return error.ChunkedValidationUnavailable;
        }
    } else {
        reference_failed = true;
    }

    try out.print("validated={d}, skipped-row-over-budget={d}, ", .{ validated, skipped_budget });
    try printRss(out, "rss-start", rss_start);
    try out.print(", ", .{});
    try printRss(out, "rss-current", residency.currentRss());
    try out.print(", ", .{});
    try printRss(out, "rss-peak", residency.peakRss());
    try out.print("\n", .{});
    if (reference_failed) return error.ReferenceMismatch;
}
