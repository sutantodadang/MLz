const std = @import("std");
const llama = @import("llama_cpp.zig");
const reference = @import("residency_llama_reference.zig");
const residency = @import("residency.zig");

fn silentLog(_: llama.c.ggml_log_level, _: [*c]const u8, _: ?*anyopaque) callconv(.c) void {}

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    if (args.len < 2 or args.len > 3) {
        std.debug.print("usage: validate-ggml-backend <model.gguf> [token-id]\n", .{});
        return error.InvalidArguments;
    }

    const path_z = try llama.dupeZ(allocator, args[1]);
    defer allocator.free(path_z);
    llama.c.llama_log_set(silentLog, null);
    const token = if (args.len == 3)
        try std.fmt.parseInt(usize, args[2], 10)
    else
        1;

    const vocab_count = try vocabularySize(path_z);
    // llama_token is signed in the llama.cpp ABI: reject values which cannot
    // survive the cast as well as values outside this model's vocabulary.
    if (token >= vocab_count or token > @as(usize, @intCast(std.math.maxInt(llama.Token)))) {
        return error.InvalidToken;
    }

    const ordinary_logits = try allocator.alloc(f32, vocab_count);
    defer allocator.free(ordinary_logits);
    const backend_logits = try allocator.alloc(f32, vocab_count);
    defer allocator.free(backend_logits);
    const tokens = [_]usize{token};

    const ordinary = try reference.sequenceLogitsMmap(
        path_z,
        &tokens,
        ordinary_logits,
        residency.currentRss,
    );
    const custom = try reference.sequenceLogitsGgmlBackend(
        path_z,
        &tokens,
        backend_logits,
        residency.currentRss,
    );

    var max_error: f32 = 0;
    var sum_error: f64 = 0;
    var exact = true;
    for (ordinary_logits, backend_logits) |expected, actual| {
        if (!std.math.isFinite(expected) or !std.math.isFinite(actual)) {
            return error.NonFiniteLogits;
        }
        const difference = @abs(expected - actual);
        max_error = @max(max_error, difference);
        sum_error += difference;
        exact = exact and @as(u32, @bitCast(expected)) == @as(u32, @bitCast(actual));
    }

    const expected_argmax = argmax(ordinary_logits);
    const actual_argmax = argmax(backend_logits);
    const stats = custom.stats;
    const mean_error = sum_error / @as(f64, @floatFromInt(vocab_count));
    const top1_matches = expected_argmax == actual_argmax;
    const numerically_close = max_error <= 0.1 and mean_error <= 0.02 and top1_matches;
    std.debug.print(
        \\official GGML residency backend validation
        \\  model: {s}
        \\  token: {d}, vocab: {d}
        \\  ordinary llama.cpp: load={d:.2} ms decode={d:.2} ms
        \\  MLz buffer backend: load={d:.2} ms decode={d:.2} ms
        \\  logits: exact={any}, max-error={d:.9}, mean-error={d:.9}, argmax={d}/{d}
        \\  backend buffers: allocated={d}, tensors={d}, uploads={d}, uploaded={d:.2} MiB
        \\  backend allocation: current={d:.2} MiB, peak={d:.2} MiB
        \\
    , .{
        args[1],
        token,
        vocab_count,
        ordinary.load_ms,
        ordinary.decode_ms,
        custom.run.load_ms,
        custom.run.decode_ms,
        exact,
        max_error,
        mean_error,
        expected_argmax,
        actual_argmax,
        stats.buffers_allocated,
        stats.tensors_initialized,
        stats.tensor_uploads,
        mib(stats.uploaded_bytes),
        mib(stats.current_allocated_bytes),
        mib(stats.peak_allocated_bytes),
    });

    // CPU_REPACK uses a different packed layout/reduction kernel than the
    // ordinary host buffer selected by this override. With cpu-repack=false
    // this gate is bit-exact; with the default repack build require the same
    // conservative numerical/top-1 gate used by the native reference path.
    if (!exact and !numerically_close) {
        return error.LogitMismatch;
    }
    if (stats.buffers_allocated == 0 or stats.tensors_initialized == 0 or
        stats.tensor_uploads == 0 or stats.uploaded_bytes == 0 or
        stats.current_allocated_bytes == 0 or stats.peak_allocated_bytes == 0)
    {
        return error.CustomBackendNotUsed;
    }
}

fn vocabularySize(path_z: [:0]const u8) !usize {
    const backend = llama.Backend.init();
    defer backend.deinit();

    var model_params = llama.c.llama_model_default_params();
    // Parse only GGUF metadata/vocabulary. The two validation runs perform the
    // actual ordinary/custom loads; avoid materializing weights a third time.
    model_params.vocab_only = true;
    model_params.use_mmap = true;
    model_params.use_mlock = false;
    const model = try llama.Model.load(path_z, model_params);
    defer model.deinit();

    const vocab = model.vocab() orelse return error.VocabUnavailable;
    const count = llama.c.llama_vocab_n_tokens(vocab);
    if (count <= 0) return error.VocabUnavailable;
    return @intCast(count);
}

fn argmax(values: []const f32) usize {
    var best: usize = 0;
    for (values[1..], 1..) |value, index| {
        if (value > values[best]) best = index;
    }
    return best;
}

fn mib(bytes: anytype) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0);
}
