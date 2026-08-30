const std = @import("std");
const residency = @import("residency.zig");

const tensor_size = 1024 * 1024;
const tensor_count = 8;
const passes = 16;
const long_context_tokens = 128;
const prefetch_worker_count = 1;
const prefetch_queue_capacity = 2;

extern "c" fn quantize_row_q4_K_ref(x: [*]const f32, y: ?*anyopaque, k: i64) void;

fn usage() void {
    std.debug.print("usage: zig build bench-residency -- [backing-file]\n", .{});
}

pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    if (args.len > 2) {
        usage();
        return error.InvalidArguments;
    }

    var tmp: ?std.testing.TmpDir = null;
    defer if (tmp) |*dir| dir.cleanup();
    const path_z: [:0]u8 = if (args.len == 2)
        try allocator.dupeZ(u8, args[1])
    else blk: {
        tmp = std.testing.tmpDir(.{});
        var file = try tmp.?.dir.createFile("residency-bench.bin", .{});
        defer file.close();
        var block: [64 * 1024]u8 = undefined;
        for (&block, 0..) |*byte, i| byte.* = @truncate(i *% 131 +% 17);
        var remaining: usize = tensor_size * tensor_count;
        while (remaining != 0) {
            const n = @min(remaining, block.len);
            try file.writeAll(block[0..n]);
            remaining -= n;
        }
        const path = try tmp.?.dir.realpathAlloc(allocator, "residency-bench.bin");
        defer allocator.free(path);
        break :blk try allocator.dupeZ(u8, path);
    };
    defer allocator.free(path_z);

    var store = try residency.BackingStore.open(path_z);
    defer store.close();
    if (store.size < tensor_size * tensor_count) return error.BackingFileTooSmall;

    // Baseline: budget for every tensor, so only the first pass faults.
    const baseline = try run(allocator, &store, tensor_size * tensor_count);
    // Bounded: two tensors resident. Sequential scans intentionally exercise
    // the replacement/fault path on each pass after warm-up.
    const bounded = try run(allocator, &store, tensor_size * 2);
    // One logical tensor is larger than the budget and is traversed through
    // range views. This is the path required by future tiled compute kernels.
    const chunked = try runChunked(allocator, &store, tensor_size);
    // Same traversal, but each pass synchronously prefaults its first window.
    // The following acquire must be a residency hit; timings expose whether
    // this workload benefits from moving page faults before consumption.
    const prefetched = try runChunkedPrefetch(allocator, &store, tensor_size);
    // Fixed-worker look-ahead prefetch overlaps page touch for the next tensor
    // with consumption of the current tensor without spawning per request.
    const scheduled = try runScheduledPrefetch(allocator, &store, tensor_size * 2);
    const token_loop = try runTokenLoop(allocator, &store, tensor_size * 2, false);
    const token_loop_prefetch = try runTokenLoop(allocator, &store, tensor_size * 2, true);

    // Parallel matmul benchmark: requires a larger backing file (16 MiB
    // weights). A dedicated temp file guarantees the size. Rows are real
    // Q4_K encodings of one deterministic float row so outputs are finite
    // and the sequential/parallel equality check is meaningful.
    const matmul_row_bytes: usize = matmul_columns / 256 * 144;
    var matmul_tmp = std.testing.tmpDir(.{});
    defer matmul_tmp.cleanup();
    {
        const columns_f: usize = matmul_columns;
        const weights_row = try allocator.alloc(f32, columns_f);
        defer allocator.free(weights_row);
        for (weights_row, 0..) |*value, i| value.* = @as(f32, @floatFromInt(i % 31)) / 13.0 - 1.0;
        const quant_row = try allocator.alloc(u8, matmul_row_bytes);
        defer allocator.free(quant_row);
        quantize_row_q4_K_ref(weights_row.ptr, quant_row.ptr, @intCast(columns_f));
        var file = try matmul_tmp.dir.createFile("matmul-bench.bin", .{});
        defer file.close();
        var remaining: usize = matmul_rows;
        while (remaining != 0) : (remaining -= 1) try file.writeAll(quant_row);
    }
    const matmul_path = try matmul_tmp.dir.realpathAlloc(allocator, "matmul-bench.bin");
    defer allocator.free(matmul_path);
    const matmul_path_z = try allocator.dupeZ(u8, matmul_path);
    defer allocator.free(matmul_path_z);
    var matmul_store = try residency.BackingStore.open(matmul_path_z);
    defer matmul_store.close();

    const matmul_granularity = try residency.mappingGranularity();
    const matmul_rows_per_tile = @max(@as(usize, 1), matmul_granularity / matmul_row_bytes);
    const matmul_budget = matmul_granularity * 32;
    const matmul_sequential = try runMatMul(allocator, &matmul_store, matmul_budget, matmul_rows_per_tile, 1);
    defer allocator.free(matmul_sequential.output);
    const matmul_parallel = try runMatMul(allocator, &matmul_store, matmul_budget, matmul_rows_per_tile, 4);
    defer allocator.free(matmul_parallel.output);

    const out = std.fs.File.stdout().deprecatedWriter();
    try out.print(
        "residency benchmark ({d} tensors x {d} MiB, {d} passes)\n" ++
            "unbounded-ish: {d:.2} ms, peak={d} MiB, faults={d}, evictions={d}\n" ++
            "bounded 2MiB: {d:.2} ms, peak={d} MiB, faults={d}, evictions={d}\n" ++
            "chunked 8MiB tensor / 1MiB budget: {d:.2} ms, peak={d} MiB, faults={d}, evictions={d}\n" ++
            "prefaulted chunked: {d:.2} ms, peak={d} MiB, faults={d}, hits={d}, prefetches={d}\n" ++
            "scheduled look-ahead: {d:.2} ms, peak={d} MiB, faults={d}, hits={d}, prefetches={d}\n",
        .{
            tensor_count,
            tensor_size / (1024 * 1024),
            passes,
            baseline.elapsed_ms,
            baseline.metrics.peak_resident_bytes / (1024 * 1024),
            baseline.metrics.faults,
            baseline.metrics.evictions,
            bounded.elapsed_ms,
            bounded.metrics.peak_resident_bytes / (1024 * 1024),
            bounded.metrics.faults,
            bounded.metrics.evictions,
            chunked.elapsed_ms,
            chunked.metrics.peak_resident_bytes / (1024 * 1024),
            chunked.metrics.faults,
            chunked.metrics.evictions,
            prefetched.elapsed_ms,
            prefetched.metrics.peak_resident_bytes / (1024 * 1024),
            prefetched.metrics.faults,
            prefetched.metrics.hits,
            prefetched.metrics.prefetches,
            scheduled.elapsed_ms,
            scheduled.metrics.peak_resident_bytes / (1024 * 1024),
            scheduled.metrics.faults,
            scheduled.metrics.hits,
            scheduled.metrics.prefetches,
        },
    );
    try out.print(
        "token loop ({d} tokens): bounded={d:.2} ms ({d:.2} ms/token), scheduled={d:.2} ms ({d:.2} ms/token)\n" ++
            "bounded overhead: {d:.2}x, chunked overhead: {d:.2}x, prefault/chunked: {d:.2}x, scheduled/bounded: {d:.2}x, sink={d}\n",
        .{
            long_context_tokens,
            token_loop.elapsed_ms,
            token_loop.elapsed_ms / long_context_tokens,
            token_loop_prefetch.elapsed_ms,
            token_loop_prefetch.elapsed_ms / long_context_tokens,
            bounded.elapsed_ms / @max(baseline.elapsed_ms, 0.000001),
            chunked.elapsed_ms / @max(baseline.elapsed_ms, 0.000001),
            prefetched.elapsed_ms / @max(chunked.elapsed_ms, 0.000001),
            scheduled.elapsed_ms / @max(bounded.elapsed_ms, 0.000001),
            baseline.sink ^ bounded.sink ^ chunked.sink ^ prefetched.sink ^ scheduled.sink ^ token_loop.sink ^ token_loop_prefetch.sink,
        },
    );
    try out.print(
        "matmul Q4_K {d}x{d} batch={d} budget={d} MiB: sequential={d:.2} ms (faults={d}), parallel 4T={d:.2} ms (faults={d}), speedup={d:.2}x, identical={}, sink={d}\n",
        .{
            matmul_rows,
            matmul_columns,
            matmul_batch,
            matmul_budget / (1024 * 1024),
            matmul_sequential.elapsed_ms,
            matmul_sequential.metrics.faults,
            matmul_parallel.elapsed_ms,
            matmul_parallel.metrics.faults,
            matmul_sequential.elapsed_ms / @max(matmul_parallel.elapsed_ms, 0.000001),
            std.mem.eql(f32, matmul_sequential.output, matmul_parallel.output),
            @as(u32, @bitCast(matmul_sequential.sink)) ^ @as(u32, @bitCast(matmul_parallel.sink)),
        },
    );
}

const Result = struct {
    elapsed_ms: f64,
    metrics: residency.Metrics,
    sink: u64,
};

fn runChunked(allocator: std.mem.Allocator, store: *residency.BackingStore, budget: usize) !Result {
    var manager = try residency.Manager.init(allocator, store, budget);
    defer manager.deinit();
    const tensor = residency.TensorHandle{ .id = 0 };
    try manager.register(tensor, 0, tensor_size * tensor_count);

    var timer = try std.time.Timer.start();
    var sink: u64 = 0;
    for (0..passes) |_| {
        for (0..tensor_count) |chunk| {
            var view = try manager.acquireRange(tensor, chunk * tensor_size, tensor_size);
            const bytes = view.bytes();
            var page: usize = 0;
            while (page < bytes.len) : (page += 4096) sink +%= bytes[page];
            view.release();
        }
    }
    const elapsed_ns = timer.read();
    return .{
        .elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
        .metrics = manager.metrics(),
        .sink = sink,
    };
}

fn runChunkedPrefetch(allocator: std.mem.Allocator, store: *residency.BackingStore, budget: usize) !Result {
    var manager = try residency.Manager.init(allocator, store, budget);
    defer manager.deinit();
    const tensor = residency.TensorHandle{ .id = 0 };
    try manager.register(tensor, 0, tensor_size * tensor_count);

    var timer = try std.time.Timer.start();
    var sink: u64 = 0;
    for (0..passes) |_| {
        for (0..tensor_count) |chunk| {
            const offset = chunk * tensor_size;
            try manager.prefetchRange(tensor, offset, tensor_size);
            var view = try manager.acquireRange(tensor, offset, tensor_size);
            const bytes = view.bytes();
            var page: usize = 0;
            while (page < bytes.len) : (page += 4096) sink +%= bytes[page];
            view.release();
        }
    }
    const elapsed_ns = timer.read();
    return .{
        .elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
        .metrics = manager.metrics(),
        .sink = sink,
    };
}

fn touch(bytes: []const u8, sink: *u64) void {
    var page: usize = 0;
    while (page < bytes.len) : (page += 4096) sink.* +%= bytes[page];
}

fn runScheduledPrefetch(allocator: std.mem.Allocator, store: *residency.BackingStore, budget: usize) !Result {
    var manager = try residency.Manager.init(allocator, store, budget);
    defer manager.deinit();
    for (0..tensor_count) |i| try manager.register(.{ .id = i }, i * tensor_size, tensor_size);

    const scheduler = try residency.PrefetchScheduler.init(
        allocator,
        &manager,
        prefetch_worker_count,
        prefetch_queue_capacity,
    );
    defer scheduler.deinit();

    var timer = try std.time.Timer.start();
    var sink: u64 = 0;
    for (0..passes) |_| {
        var next = try scheduler.submit(.{ .id = 0 }, 0, tensor_size);
        for (0..tensor_count) |i| {
            try next.wait();
            const following: ?residency.ScheduledPrefetchTask = if (i + 1 < tensor_count)
                try scheduler.submit(.{ .id = i + 1 }, 0, tensor_size)
            else
                null;
            var view = try manager.acquire(.{ .id = i });
            touch(view.bytes(), &sink);
            view.release();
            if (following) |task| next = task;
        }
    }
    scheduler.waitIdle();
    const elapsed_ns = timer.read();
    return .{
        .elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
        .metrics = manager.metrics(),
        .sink = sink,
    };
}

fn runTokenLoop(
    allocator: std.mem.Allocator,
    store: *residency.BackingStore,
    budget: usize,
    use_scheduler: bool,
) !Result {
    var manager = try residency.Manager.init(allocator, store, budget);
    defer manager.deinit();
    for (0..tensor_count) |i| try manager.register(.{ .id = i }, i * tensor_size, tensor_size);

    const scheduler: ?*residency.PrefetchScheduler = if (use_scheduler)
        try residency.PrefetchScheduler.init(allocator, &manager, 1, 2)
    else
        null;
    defer if (scheduler) |active| active.deinit();

    var timer = try std.time.Timer.start();
    var sink: u64 = 0;
    for (0..long_context_tokens) |_| {
        var next: ?residency.ScheduledPrefetchTask = if (scheduler) |active|
            try active.submit(.{ .id = 0 }, 0, tensor_size)
        else
            null;
        for (0..tensor_count) |i| {
            if (next) |*task| try task.wait();
            next = if (scheduler) |active| if (i + 1 < tensor_count)
                try active.submit(.{ .id = i + 1 }, 0, tensor_size)
            else
                null else null;
            var view = try manager.acquire(.{ .id = i });
            touch(view.bytes(), &sink);
            view.release();
        }
    }
    if (scheduler) |active| active.waitIdle();
    const elapsed_ns = timer.read();
    return .{
        .elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
        .metrics = manager.metrics(),
        .sink = sink,
    };
}

const matmul_rows = 4096;
const matmul_columns = 4096;
const matmul_batch = 8;

const MatMulResult = struct {
    elapsed_ms: f64,
    metrics: residency.Metrics,
    sink: f32,
    output: []f32,
};

/// Bounded Q4_K matmul over a 16 MiB weight matrix with a 4 MiB budget:
/// sequential tiled kernel versus the parallel tiled driver. Both must
/// produce identical output; timings and residency metrics are compared.
fn runMatMul(
    allocator: std.mem.Allocator,
    store: *residency.BackingStore,
    budget: usize,
    rows_per_tile: usize,
    threads: usize,
) !MatMulResult {
    const compute = @import("residency_compute.zig");
    const gguf = @import("gguf_residency.zig");

    var manager = try residency.Manager.init(allocator, store, budget);
    defer manager.deinit();

    const row_bytes: usize = matmul_columns / 256 * 144;
    const byte_len = row_bytes * matmul_rows;
    if (store.size < byte_len) return error.BackingFileTooSmall;
    try manager.register(.{ .id = 0 }, 0, byte_len);

    const descriptor = gguf.TensorDescriptor{
        .handle = .{ .id = 0 },
        .name = "matmul_bench.weight",
        .file_offset = 0,
        .byte_len = byte_len,
        .ggml_type = gguf.type_q4_k,
        .n_dimensions = 2,
        .dimensions = .{ matmul_columns, matmul_rows, 1, 1 },
    };

    const inputs = try allocator.alloc(f32, matmul_batch * matmul_columns);
    defer allocator.free(inputs);
    for (inputs, 0..) |*value, i| value.* = @as(f32, @floatFromInt(i % 31)) / 13.0 - 1.0;
    const outputs = try allocator.alloc(f32, matmul_batch * matmul_rows);
    defer allocator.free(outputs);

    const scratch_len = try compute.quantizedDotBatchScratchBytes(gguf.type_q4_k, matmul_columns, matmul_batch);
    const scratch = try allocator.alignedAlloc(u8, compute.quantScratchAlignment, scratch_len);
    defer allocator.free(scratch);

    var timer = try std.time.Timer.start();
    if (threads <= 1) {
        try compute.matMulQuantizedGgml(
            &manager,
            &descriptor,
            inputs,
            outputs,
            matmul_batch,
            rows_per_tile,
            scratch,
        );
    } else {
        try compute.parallelMatMul(
            &manager,
            &descriptor,
            inputs,
            outputs,
            matmul_batch,
            .{ .threads = threads, .rows_per_tile = rows_per_tile },
        );
    }
    const elapsed_ms = @as(f64, @floatFromInt(timer.read())) / std.time.ns_per_ms;

    var sink: f32 = 0;
    for (outputs) |value| sink += value;

    return .{
        .elapsed_ms = elapsed_ms,
        .metrics = manager.metrics(),
        .sink = sink,
        .output = try allocator.dupe(f32, outputs),
    };
}

fn run(allocator: std.mem.Allocator, store: *residency.BackingStore, budget: usize) !Result {
    var manager = try residency.Manager.init(allocator, store, budget);
    defer manager.deinit();
    for (0..tensor_count) |i| {
        try manager.register(.{ .id = i }, i * tensor_size, tensor_size);
    }

    var timer = try std.time.Timer.start();
    var sink: u64 = 0;
    for (0..passes) |_| {
        for (0..tensor_count) |i| {
            var view = try manager.acquire(.{ .id = i });
            const bytes = view.bytes();
            // Touch one byte per page so page faults are included, without
            // making memcpy throughput dominate the residency benchmark.
            var page: usize = 0;
            while (page < bytes.len) : (page += 4096) sink +%= bytes[page];
            view.release();
        }
    }
    const elapsed_ns = timer.read();
    return .{
        .elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
        .metrics = manager.metrics(),
        .sink = sink,
    };
}
