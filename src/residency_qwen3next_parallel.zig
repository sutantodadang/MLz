//! Opt-in, bit-exact parallel execution of the Qwen3-Next DeltaNet recurrence.
//!
//! A pool owns a fixed number of reused background threads. Work is split only
//! across independent value heads; convolution, Q/K normalization, and every
//! operation within one head retain the scalar implementation's order.

const std = @import("std");
const qwen = @import("residency_qwen3next.zig");

pub const max_worker_count: usize = 64;

/// Pool currently installed into `qwen.parallel_step_hook`, if any.
var hooked_pool: ?*Pool = null;

pub const InitError = std.Thread.SpawnError || error{
    InvalidWorkerCount,
};

const Job = struct {
    heads: qwen.DeltaNetHeadContext,
    worker_count: usize,
};

/// Fixed, reusable worker pool for DeltaNet value-head recurrence.
///
/// `worker_count` includes the submitting thread, so a value of one is an
/// exact scalar execution and allocates no background threads. Calls on one
/// pool are serialized; separate pools may execute independently.
pub const Pool = struct {
    allocator: std.mem.Allocator,
    threads: []std.Thread,
    worker_ids: []usize,
    worker_ready: []bool,
    mutex: std.Thread.Mutex = .{},
    submit_mutex: std.Thread.Mutex = .{},
    work_available: std.Thread.Condition = .{},
    work_done: std.Thread.Condition = .{},
    generation: usize = 0,
    completed: usize = 0,
    stopping: bool = false,
    job: ?*const Job = null,

    pub fn init(allocator: std.mem.Allocator, worker_count: usize) InitError!*Pool {
        if (worker_count == 0 or worker_count > max_worker_count) return error.InvalidWorkerCount;

        const self = try allocator.create(Pool);
        errdefer allocator.destroy(self);
        self.* = .{
            .allocator = allocator,
            .threads = &.{},
            .worker_ids = &.{},
            .worker_ready = &.{},
        };

        const background_count = worker_count - 1;
        self.threads = try allocator.alloc(std.Thread, background_count);
        errdefer allocator.free(self.threads);
        self.worker_ids = try allocator.alloc(usize, background_count);
        errdefer allocator.free(self.worker_ids);
        self.worker_ready = try allocator.alloc(bool, background_count);
        errdefer allocator.free(self.worker_ready);
        @memset(self.worker_ready, false);

        var started: usize = 0;
        errdefer {
            self.mutex.lock();
            self.stopping = true;
            self.work_available.broadcast();
            self.mutex.unlock();
            for (self.threads[0..started]) |thread| thread.join();
        }
        while (started < background_count) : (started += 1) {
            self.worker_ids[started] = started;
            self.threads[started] = try std.Thread.spawn(.{}, workerMain, .{ self, &self.worker_ids[started] });
        }
        self.mutex.lock();
        while (!allWorkersReady(self.worker_ready)) self.work_done.wait(&self.mutex);
        self.mutex.unlock();

        // Install the opt-in hook so qwen.deltaNetStep call sites route through
        // this pool. Restored to null on deinit.
        qwen.parallel_step_hook = deltaNetStepHook;
        hooked_pool = self;
        return self;
    }

    pub fn deinit(self: *Pool) void {
        qwen.parallel_step_hook = null;
        hooked_pool = null;
        self.submit_mutex.lock();
        self.mutex.lock();
        self.stopping = true;
        self.work_available.broadcast();
        self.mutex.unlock();
        for (self.threads) |thread| thread.join();
        self.submit_mutex.unlock();

        const allocator = self.allocator;
        allocator.free(self.worker_ready);
        allocator.free(self.worker_ids);
        allocator.free(self.threads);
        allocator.destroy(self);
    }

    pub fn workerCount(self: *const Pool) usize {
        return self.threads.len + 1;
    }

    /// Execute one complete DeltaNet step. The pool is the explicit opt-in;
    /// the existing `qwen.deltaNetStep` API remains scalar.
    pub fn deltaNetStep(
        self: *Pool,
        config: qwen.Config,
        cache: *qwen.DeltaNetCache,
        qkv: []f32,
        z: []const f32,
        beta_alpha: []const f32,
        conv_weights: []align(1) const f32,
        dt_bias: []align(1) const f32,
        decay: []align(1) const f32,
        norm_weights: []align(1) const f32,
        output: []f32,
    ) qwen.Error!void {
        self.submit_mutex.lock();
        defer self.submit_mutex.unlock();

        const heads = try qwen.deltaNetPrepare(config, cache, qkv, z, beta_alpha, conv_weights, dt_bias, decay, norm_weights, output);
        const worker_count = self.workerCount();
        if (worker_count == 1) {
            qwen.deltaNetHeadRange(heads, 0, config.value_head_count);
            qwen.deltaNetFinish(cache);
            return;
        }

        const job = Job{ .heads = heads, .worker_count = worker_count };
        self.mutex.lock();
        self.completed = 0;
        self.job = &job;
        self.generation +%= 1;
        self.work_available.broadcast();
        self.mutex.unlock();

        runPartition(&job, worker_count - 1);

        self.mutex.lock();
        while (self.completed != self.threads.len) self.work_done.wait(&self.mutex);
        self.job = null;
        self.mutex.unlock();

        qwen.deltaNetFinish(cache);
    }

    /// Static trampoline for `qwen.parallel_step_hook`. The hook signature has
    /// no pool parameter; exactly one hooked pool is expected at a time.
    fn deltaNetStepHook(
        config: qwen.Config,
        cache: *qwen.DeltaNetCache,
        qkv: []f32,
        z: []const f32,
        beta_alpha: []const f32,
        conv_weights: []align(1) const f32,
        dt_bias: []align(1) const f32,
        decay: []align(1) const f32,
        norm_weights: []align(1) const f32,
        output: []f32,
    ) qwen.Error!void {
        // Reach the owning pool via the singleton registry installed at init.
        try hooked_pool.?.deltaNetStep(config, cache, qkv, z, beta_alpha, conv_weights, dt_bias, decay, norm_weights, output);
    }

    fn workerMain(self: *Pool, worker_id: *const usize) void {
        var seen_generation: usize = 0;
        self.mutex.lock();
        self.worker_ready[worker_id.*] = true;
        self.work_done.signal();
        self.mutex.unlock();
        while (true) {
            self.mutex.lock();
            while (!self.stopping and self.generation == seen_generation) self.work_available.wait(&self.mutex);
            if (self.stopping) {
                self.mutex.unlock();
                return;
            }
            seen_generation = self.generation;
            const job = self.job.?;
            self.mutex.unlock();

            runPartition(job, worker_id.*);

            self.mutex.lock();
            self.completed += 1;
            if (self.completed == self.threads.len) self.work_done.signal();
            self.mutex.unlock();
        }
    }
};

fn allWorkersReady(ready: []const bool) bool {
    for (ready) |value| if (!value) return false;
    return true;
}

fn runPartition(job: *const Job, partition: usize) void {
    const head_count = job.heads.config.value_head_count;
    const base = head_count / job.worker_count;
    const extra = head_count % job.worker_count;
    const start = partition * base + @min(partition, extra);
    const len = base + @intFromBool(partition < extra);
    qwen.deltaNetHeadRange(job.heads, start, start + len);
}

test "parallel DeltaNet is bit-exact with scalar over multiple steps" {
    const allocator = std.testing.allocator;
    const config = qwen.Config{
        .hidden_size = 8,
        .rms_epsilon = 1e-6,
        .state_size = 4,
        .key_head_count = 2,
        .value_head_count = 6,
        .inner_size = 24,
        .conv_kernel = 3,
        .attention_head_count = 2,
        .attention_kv_head_count = 1,
        .attention_head_dim = 4,
        .rope_dimension_count = 4,
        .rope_theta = 10_000,
        .expert_count = 2,
        .expert_used_count = 1,
    };

    var scalar_cache = try qwen.DeltaNetCache.init(allocator, config);
    defer scalar_cache.deinit();
    var parallel_cache = try qwen.DeltaNetCache.init(allocator, config);
    defer parallel_cache.deinit();
    const pool = try Pool.init(allocator, 4);
    defer pool.deinit();
    try std.testing.expectEqual(@as(usize, 4), pool.workerCount());

    const channels = config.convChannels();
    const conv_weights = try allocator.alloc(f32, channels * config.conv_kernel);
    defer allocator.free(conv_weights);
    const dt_bias = try allocator.alloc(f32, config.value_head_count);
    defer allocator.free(dt_bias);
    const decay = try allocator.alloc(f32, config.value_head_count);
    defer allocator.free(decay);
    const norm = try allocator.alloc(f32, config.state_size);
    defer allocator.free(norm);
    const z = try allocator.alloc(f32, config.inner_size);
    defer allocator.free(z);
    const beta_alpha = try allocator.alloc(f32, 2 * config.value_head_count);
    defer allocator.free(beta_alpha);
    for (conv_weights, 0..) |*value, i| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) * 0.03125;
    for (dt_bias, 0..) |*value, i| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 2)) * 0.07;
    for (decay, 0..) |*value, i| value.* = -0.15 - @as(f32, @floatFromInt(i)) * 0.025;
    for (norm, 0..) |*value, i| value.* = 0.75 + @as(f32, @floatFromInt(i)) * 0.125;
    for (z, 0..) |*value, i| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 9)) - 4)) * 0.11;
    for (beta_alpha, 0..) |*value, i| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0.09;

    var step: usize = 0;
    while (step < 7) : (step += 1) {
        const scalar_qkv = try allocator.alloc(f32, channels);
        defer allocator.free(scalar_qkv);
        const parallel_qkv = try allocator.alloc(f32, channels);
        defer allocator.free(parallel_qkv);
        for (scalar_qkv, 0..) |*value, i| {
            const raw: i32 = @intCast((i * 13 + step * 17) % 29);
            value.* = @as(f32, @floatFromInt(raw - 14)) * 0.0375;
        }
        @memcpy(parallel_qkv, scalar_qkv);
        const scalar_output = try allocator.alloc(f32, config.inner_size);
        defer allocator.free(scalar_output);
        const parallel_output = try allocator.alloc(f32, config.inner_size);
        defer allocator.free(parallel_output);

        try qwen.deltaNetStep(config, &scalar_cache, scalar_qkv, z, beta_alpha, conv_weights, dt_bias, decay, norm, scalar_output);
        try pool.deltaNetStep(config, &parallel_cache, parallel_qkv, z, beta_alpha, conv_weights, dt_bias, decay, norm, parallel_output);

        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(scalar_qkv), std.mem.sliceAsBytes(parallel_qkv));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(scalar_output), std.mem.sliceAsBytes(parallel_output));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(scalar_cache.conv_history), std.mem.sliceAsBytes(parallel_cache.conv_history));
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(scalar_cache.recurrent), std.mem.sliceAsBytes(parallel_cache.recurrent));
        try std.testing.expectEqual(scalar_cache.position, parallel_cache.position);
    }
}

test "DeltaNet pool rejects unbounded worker counts" {
    try std.testing.expectError(error.InvalidWorkerCount, Pool.init(std.testing.allocator, 0));
    try std.testing.expectError(error.InvalidWorkerCount, Pool.init(std.testing.allocator, max_worker_count + 1));
}
