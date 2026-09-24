//! C ABI bridge between the GGML residency backend (C) and the bounded
//! residency manager (Zig). In backed mode the C backend registers each model
//! weight's GGUF span at load time (resolved by tensor name through
//! `spanCallback`) and rebase/restores `tensor->data` around each node:
//!
//!   pre-hook:  acquire(span) -> mapped host address for the whole tensor
//!   kernel:    stock GGML kernels read tensor->data directly
//!   post-hook: release(address) -> mapping becomes evictable again
//!
//! The bridge owns exactly one Manager instance per model and is
//! single-model: validation tools open one model before enabling hooks.

const std = @import("std");
const residency = @import("residency.zig");
const gguf_residency = @import("gguf_residency.zig");
const llama_cpp = @import("llama_cpp.zig");

const Manager = residency.Manager;
const BackingStore = residency.BackingStore;
const TensorHandle = residency.TensorHandle;

pub const Error = residency.Error || gguf_residency.Error ||
    error{ BridgeAlreadyInitialized, BridgeNotInitialized, BudgetTooSmall, UnsupportedTensorLayout };

const SourceSpan = struct {
    handle: TensorHandle,
    file_offset: u64,
    byte_len: usize,
};

const Instance = struct {
    allocator: std.mem.Allocator,
    store: BackingStore,
    manager: Manager,
    index: gguf_residency.TensorIndex,
    // Weight spans keyed by C-registry source id (1-based, registration
    // order). The C side assigns ids in set_tensor order; we mirror them here
    // as the callbacks arrive.
    sources: std.ArrayList(SourceSpan),
    // Each graph/tile acquisition owns an independent manager pin, even when
    // another graph is reading the same source at the same time.
    open_views: std.AutoHashMap(u64, residency.TensorView),
    next_pin_token: u64 = 1,
    views_mutex: std.Thread.Mutex = .{},
    pin_released: std.Thread.Condition = .{},
    // Guarded by views_mutex.
    failures: FailureCounts = .{},
    last_failure: FailureKind = .none,
    admission_waits: u64 = 0,
    acquisitions: u64 = 0,
    acquire_ns_total: u64 = 0,
    acquire_ns_max: u64 = 0,
    /// Test-only fault injection: fail the acquire attempt once this reaches 1.
    fail_countdown: u64 = 0,
};

/// Why a weight acquisition failed. Budget failures are split into requests
/// that can never fit (configuration) and requests that waited too long for
/// other graphs to release their pins (contention).
pub const FailureKind = enum { none, budget_impossible, admission_timeout, io, injected, invalid };

pub const FailureCounts = struct {
    budget_impossible: u64 = 0,
    admission_timeout: u64 = 0,
    io: u64 = 0,
    injected: u64 = 0,
    invalid: u64 = 0,
};

pub const Diagnostics = struct {
    failures: FailureCounts,
    last_failure: FailureKind,
    open_pins: usize,
    admission_waits: u64,
    /// Successful acquire calls (a node's whole-span set counts once).
    acquisitions: u64,
    /// Time inside successful acquire calls, including admission waits and
    /// mapping/faulting the range.
    acquire_ns_total: u64,
    acquire_ns_max: u64,
};

const admission_timeout_ns = 30 * std.time.ns_per_s;
/// Upper bound on distinct weight sources pinned together by one graph node.
pub const max_node_sources = 16;

var g_instance: ?*Instance = null;

/// Opens the GGUF file as backing storage and prepares the bounded manager.
/// Weight spans are learned lazily from the C backend during model load.
/// `budget_bytes` bounds the sum of live mapped windows at any point during
/// graph execution.
pub fn init(
    allocator: std.mem.Allocator,
    path_z: [:0]const u8,
    budget_bytes: usize,
) Error!void {
    if (g_instance != null) return Error.BridgeAlreadyInitialized;

    const instance = try allocator.create(Instance);
    errdefer allocator.destroy(instance);

    instance.allocator = allocator;
    instance.store = try BackingStore.open(path_z);
    errdefer instance.store.close();

    instance.index = try gguf_residency.TensorIndex.open(
        allocator,
        path_z,
        instance.store.size,
    );
    errdefer instance.index.deinit();
    // Official GGML execution is architecture-agnostic: llama.cpp builds the
    // graph and the hooks only need each tensor's GGUF span.
    try preflightWeightRows(&instance.index, budget_bytes);

    instance.manager = try Manager.init(allocator, &instance.store, budget_bytes);
    errdefer instance.manager.deinit();
    instance.sources = .empty;
    instance.open_views = std.AutoHashMap(u64, residency.TensorView).init(allocator);
    instance.next_pin_token = 1;
    instance.views_mutex = .{};
    instance.pin_released = .{};
    instance.failures = .{};
    instance.last_failure = .none;
    instance.admission_waits = 0;
    instance.acquisitions = 0;
    instance.acquire_ns_total = 0;
    instance.acquire_ns_max = 0;
    instance.fail_countdown = 0;

    g_instance = instance;
}

fn preflightWeightRows(index: *const gguf_residency.TensorIndex, budget_bytes: usize) Error!void {
    const granularity = try residency.mappingGranularity();
    for (index.descriptors) |descriptor| {
        const first_prefix: usize = @intCast(descriptor.file_offset % granularity);
        if (descriptor.byte_len <= budget_bytes and first_prefix <= budget_bytes - descriptor.byte_len) continue;
        if (descriptor.n_dimensions < 2) {
            std.log.err("residency tensor {s} needs {d} mapped bytes; budget is {d}", .{ descriptor.name, descriptor.byte_len + first_prefix, budget_bytes });
            return Error.BudgetTooSmall;
        }
        const row_bytes = llama_cpp.c.ggml_row_size(@intCast(descriptor.ggml_type), @intCast(descriptor.dimensions[0]));
        if (row_bytes == 0 or row_bytes > descriptor.byte_len or descriptor.byte_len % row_bytes != 0) {
            std.log.err("residency unsupported row layout for {s}", .{descriptor.name});
            return Error.UnsupportedTensorLayout;
        }
        const row_count = descriptor.byte_len / row_bytes;
        const period = granularity / std.math.gcd(granularity, row_bytes);
        for (0..@min(row_count, period)) |row| {
            const prefix: usize = @intCast((descriptor.file_offset + row * row_bytes) % granularity);
            if (row_bytes > budget_bytes or prefix > budget_bytes - row_bytes) {
                std.log.err("residency tensor {s} needs {d} mapped bytes per row; budget is {d}", .{ descriptor.name, row_bytes + prefix, budget_bytes });
                return Error.BudgetTooSmall;
            }
        }
    }
}

pub fn deinit(allocator: std.mem.Allocator) void {
    const instance = g_instance orelse return;
    std.debug.assert(instance.open_views.count() == 0); // unbalanced pre/post hooks
    instance.open_views.deinit();
    instance.sources.deinit(allocator);
    instance.manager.deinit();
    instance.index.deinit();
    instance.store.close();
    allocator.destroy(instance);
    g_instance = null;
}

pub fn metrics() ?residency.Metrics {
    const instance = g_instance orelse return null;
    return instance.manager.metrics();
}

pub fn diagnostics() ?Diagnostics {
    const instance = g_instance orelse return null;
    instance.views_mutex.lock();
    defer instance.views_mutex.unlock();
    return .{
        .failures = instance.failures,
        .last_failure = instance.last_failure,
        .open_pins = instance.open_views.count(),
        .admission_waits = instance.admission_waits,
        .acquisitions = instance.acquisitions,
        .acquire_ns_total = instance.acquire_ns_total,
        .acquire_ns_max = instance.acquire_ns_max,
    };
}

/// Test hook: make the `after`-th subsequent acquire attempt fail (1 = next).
/// Zero disables injection.
pub fn injectAcquireFailure(after: u64) void {
    const instance = g_instance orelse return;
    instance.views_mutex.lock();
    defer instance.views_mutex.unlock();
    instance.fail_countdown = after;
}

fn recordFailure(instance: *Instance, kind: FailureKind) void {
    switch (kind) {
        .none => return,
        .budget_impossible => instance.failures.budget_impossible += 1,
        .admission_timeout => instance.failures.admission_timeout += 1,
        .io => instance.failures.io += 1,
        .injected => instance.failures.injected += 1,
        .invalid => instance.failures.invalid += 1,
    }
    instance.last_failure = kind;
    std.log.warn("residency weight acquire failed: {s}", .{@tagName(kind)});
}

fn consumeInjection(instance: *Instance) bool {
    if (instance.fail_countdown == 0) return false;
    instance.fail_countdown -= 1;
    return instance.fail_countdown == 0;
}

/// Resolves a tensor name to its absolute GGUF file span. Used by the C
/// backend at load time to register each weight's source range.
pub fn spanCallback(name: [*c]const u8, file_offset: [*c]u64, byte_len: [*c]usize) callconv(.c) bool {
    const instance = g_instance orelse return false;
    if (name == null or file_offset == null or byte_len == null) return false;
    const descriptor = instance.index.get(std.mem.span(@as([*:0]const u8, @ptrCast(name)))) orelse return false;
    file_offset.* = descriptor.file_offset;
    byte_len.* = descriptor.byte_len;
    return true;
}

/// Called by the C backend during model load for every registered weight, in
/// registration order. Mirrors the C registry so source ids line up.
pub fn syncRegistry() Error!void {
    const instance = g_instance orelse return Error.BridgeNotInitialized;
    const count = llama_cpp.c.mlz_ggml_residency_registry_count();
    var index: usize = 1;
    while (index <= count) : (index += 1) {
        var file_offset: u64 = 0;
        var byte_len: usize = 0;
        if (!llama_cpp.c.mlz_ggml_residency_registry_span(index, &file_offset, &byte_len)) {
            return Error.UnknownTensor;
        }
        if (instance.sources.items.len >= index) continue; // already mirrored
        const descriptor = blk: {
            for (instance.index.descriptors) |*d| {
                if (d.file_offset == file_offset and d.byte_len == byte_len) break :blk d;
            }
            std.debug.print(
                "mlz bridge: registry span {d} ({d},{d}) not in GGUF index\n",
                .{ index, file_offset, byte_len },
            );
            return Error.UnknownTensor;
        };
        try instance.sources.append(instance.allocator, .{
            .handle = descriptor.handle,
            .file_offset = file_offset,
            .byte_len = byte_len,
        });
        errdefer _ = instance.sources.pop();
        try instance.manager.register(descriptor.handle, file_offset, byte_len);
    }
}

/// Maps a registered tensor's full span and records the open view under the
/// registry's 1-based source id. Called by the backend pre-hook on graph
/// thread 0 between barriers; the returned pointer stays valid until the
/// matching release call.
fn rememberView(instance: *Instance, view: residency.TensorView, pin_token: [*c]u64) ?*anyopaque {
    const token = instance.next_pin_token;
    if (token == 0) {
        var owned = view;
        owned.release();
        return null;
    }
    instance.open_views.put(token, view) catch {
        var owned = view;
        owned.release();
        return null;
    };
    instance.next_pin_token +%= 1;
    pin_token.* = token;
    return @ptrCast(@constCast(view.data.ptr));
}

const Range = struct { handle: TensorHandle, offset: usize, len: usize };

/// Pins every range or none of them. A graph can transiently hold the full
/// weight budget while another graph is ready to run; waiting happens only
/// while this caller holds no pin from the current attempt (no hold-and-wait),
/// and releases views_mutex so owners can release their pins.
fn acquireAllAdmitted(instance: *Instance, ranges: []const Range, views: []residency.TensorView) bool {
    std.debug.assert(ranges.len == views.len);
    for (ranges) |range| {
        const capacity = instance.manager.rangeCapacity(range.handle, range.offset) catch {
            recordFailure(instance, .invalid);
            return false;
        };
        if (range.len > capacity) {
            recordFailure(instance, .budget_impossible);
            return false;
        }
    }
    var timer = std.time.Timer.start() catch {
        recordFailure(instance, .invalid);
        return false;
    };
    attempt: while (true) {
        if (consumeInjection(instance)) {
            recordFailure(instance, .injected);
            return false;
        }
        for (ranges, 0..) |range, index| {
            views[index] = instance.manager.acquireRange(range.handle, range.offset, range.len) catch |err| {
                for (views[0..index]) |*view| view.release();
                if (err != error.BudgetExceeded) {
                    recordFailure(instance, .io);
                    return false;
                }
                // No other graph holds a pin: waiting can never make room.
                if (instance.open_views.count() == 0) {
                    recordFailure(instance, .budget_impossible);
                    return false;
                }
                const elapsed = timer.read();
                if (elapsed >= admission_timeout_ns) {
                    recordFailure(instance, .admission_timeout);
                    return false;
                }
                instance.admission_waits += 1;
                instance.pin_released.timedWait(&instance.views_mutex, admission_timeout_ns - elapsed) catch {};
                continue :attempt;
            };
        }
        const elapsed = timer.read();
        instance.acquisitions += 1;
        instance.acquire_ns_total +%= elapsed;
        instance.acquire_ns_max = @max(instance.acquire_ns_max, elapsed);
        return true;
    }
}

fn acquireAdmitted(instance: *Instance, handle: TensorHandle, offset: usize, len: usize) ?residency.TensorView {
    var views: [1]residency.TensorView = undefined;
    const ranges = [1]Range{.{ .handle = handle, .offset = offset, .len = len }};
    if (!acquireAllAdmitted(instance, &ranges, &views)) return null;
    return views[0];
}

/// Pins the complete spans of `count` distinct registered sources atomically:
/// either every `mapped[i]`/`pin_tokens[i]` is filled or nothing stays pinned.
/// Graph nodes use this for all of their weight sources so two graphs can
/// never each hold part of what the other needs.
pub fn acquireManyCallback(count: usize, source_ids: [*c]const u32, mapped: [*c]?*anyopaque, pin_tokens: [*c]u64) callconv(.c) bool {
    const instance = g_instance orelse return false;
    instance.views_mutex.lock();
    defer instance.views_mutex.unlock();
    if (count == 0 or count > max_node_sources or source_ids == null or mapped == null or pin_tokens == null) {
        recordFailure(instance, .invalid);
        return false;
    }
    var ranges: [max_node_sources]Range = undefined;
    for (0..count) |index| {
        const id = source_ids[index];
        if (id == 0 or id > instance.sources.items.len) {
            recordFailure(instance, .invalid);
            return false;
        }
        for (source_ids[0..index]) |previous| {
            if (previous == id) {
                recordFailure(instance, .invalid);
                return false;
            }
        }
        const source = instance.sources.items[id - 1];
        ranges[index] = .{ .handle = source.handle, .offset = 0, .len = source.byte_len };
    }
    var views: [max_node_sources]residency.TensorView = undefined;
    if (!acquireAllAdmitted(instance, ranges[0..count], views[0..count])) return false;
    for (0..count) |index| {
        mapped[index] = rememberView(instance, views[index], &pin_tokens[index]) orelse {
            for (views[index + 1 .. count]) |*view| view.release();
            for (pin_tokens[0..index]) |token| {
                var removed = (instance.open_views.fetchRemove(token) orelse unreachable).value;
                removed.release();
            }
            instance.pin_released.broadcast();
            recordFailure(instance, .invalid);
            return false;
        };
    }
    return true;
}

pub fn acquireCallback(source_id: u32, file_offset: u64, byte_len: usize, pin_token: [*c]u64) callconv(.c) ?*anyopaque {
    const instance = g_instance orelse {
        std.debug.print("mlz bridge: acquire with no instance\n", .{});
        return null;
    };
    instance.views_mutex.lock();
    defer instance.views_mutex.unlock();
    if (pin_token == null or source_id == 0 or source_id > instance.sources.items.len) {
        std.debug.print("mlz bridge: acquire bad source_id={d}\n", .{source_id});
        recordFailure(instance, .invalid);
        return null;
    }
    const source = instance.sources.items[source_id - 1];
    if (source.file_offset != file_offset or source.byte_len != byte_len) {
        std.debug.print(
            "mlz bridge: acquire span mismatch id={d}: C({d},{d}) vs Zig({d},{d})\n",
            .{ source_id, file_offset, byte_len, source.file_offset, source.byte_len },
        );
        recordFailure(instance, .invalid);
        return null; // span mismatch: C registry disagrees with Zig mirror
    }

    const view = acquireAdmitted(instance, source.handle, 0, source.byte_len) orelse return null;
    return rememberView(instance, view, pin_token);
}

/// Maps a validated logical subrange of a registered tensor. Tiled kernels
/// reuse the same per-source slot as whole-node mapping and release it before
/// acquiring the next tile.
pub fn acquireRangeCallback(source_id: u32, tensor_offset: usize, byte_len: usize, pin_token: [*c]u64) callconv(.c) ?*anyopaque {
    const instance = g_instance orelse return null;
    instance.views_mutex.lock();
    defer instance.views_mutex.unlock();
    if (pin_token == null or source_id == 0 or source_id > instance.sources.items.len or byte_len == 0) {
        std.debug.print("mlz bridge: invalid range args id={d} count={d} bytes={d} token_null={}\n", .{ source_id, instance.sources.items.len, byte_len, pin_token == null });
        recordFailure(instance, .invalid);
        return null;
    }
    const source = instance.sources.items[source_id - 1];
    if (tensor_offset > source.byte_len or byte_len > source.byte_len - tensor_offset) {
        std.debug.print("mlz bridge: range outside source id={d} offset={d} bytes={d} source={d}\n", .{ source_id, tensor_offset, byte_len, source.byte_len });
        recordFailure(instance, .invalid);
        return null;
    }

    const view = acquireAdmitted(instance, source.handle, tensor_offset, byte_len) orelse return null;
    return rememberView(instance, view, pin_token);
}

/// Reports the largest logical range starting at `tensor_offset` that can be
/// mapped under the manager budget, or zero for an invalid source/range.
pub fn rangeCapacityCallback(source_id: u32, tensor_offset: usize) callconv(.c) usize {
    const instance = g_instance orelse return 0;
    if (source_id == 0 or source_id > instance.sources.items.len) return 0;
    const source = instance.sources.items[source_id - 1];
    if (tensor_offset >= source.byte_len) return 0;
    instance.views_mutex.lock();
    defer instance.views_mutex.unlock();
    return instance.manager.rangeCapacity(source.handle, tensor_offset) catch 0;
}

/// Releases exactly the acquired pin, independently of other graph readers.
pub fn releaseCallback(pin_token: u64) callconv(.c) bool {
    const instance = g_instance orelse return false;
    instance.views_mutex.lock();
    defer instance.views_mutex.unlock();
    const removed = instance.open_views.fetchRemove(pin_token) orelse return false;
    var view = removed.value;
    view.release();
    instance.pin_released.broadcast();
    return true;
}

const TestBridge = struct {
    tmp: std.testing.TmpDir,
    instance: Instance,

    // Two 64-byte tensors on separate mapping-granularity boundaries under a
    // 64-byte budget: either fits alone, both never fit together.
    fn init(self: *TestBridge) !void {
        self.tmp = std.testing.tmpDir(.{});
        errdefer self.tmp.cleanup();
        const granularity = try residency.mappingGranularity();
        var file = try self.tmp.dir.createFile("weights.bin", .{});
        try file.setEndPos(granularity + 64);
        file.close();
        const path = try self.tmp.dir.realpathAlloc(std.testing.allocator, "weights.bin");
        defer std.testing.allocator.free(path);
        const path_z = try std.testing.allocator.dupeZ(u8, path);
        defer std.testing.allocator.free(path_z);

        self.instance = .{
            .allocator = std.testing.allocator,
            .store = try BackingStore.open(path_z),
            .manager = undefined,
            .index = undefined,
            .sources = .empty,
            .open_views = std.AutoHashMap(u64, residency.TensorView).init(std.testing.allocator),
        };
        self.instance.manager = try Manager.init(std.testing.allocator, &self.instance.store, 64);
        try self.instance.manager.register(.{ .id = 1 }, 0, 64);
        try self.instance.manager.register(.{ .id = 2 }, granularity, 64);
    }

    fn deinit(self: *TestBridge) void {
        var it = self.instance.open_views.valueIterator();
        while (it.next()) |view| view.release();
        self.instance.open_views.deinit();
        self.instance.manager.deinit();
        self.instance.store.close();
        self.tmp.cleanup();
    }

    fn range(id: u64) Range {
        return .{ .handle = .{ .id = id }, .offset = 0, .len = 64 };
    }
};

test "node sources are pinned all-or-nothing and impossible sets fail fast" {
    var t: TestBridge = undefined;
    try t.init();
    defer t.deinit();
    const instance = &t.instance;
    instance.views_mutex.lock();
    defer instance.views_mutex.unlock();

    var views: [2]residency.TensorView = undefined;
    const both = [_]Range{ TestBridge.range(1), TestBridge.range(2) };
    try std.testing.expect(!acquireAllAdmitted(instance, &both, &views));
    try std.testing.expectEqual(@as(u64, 1), instance.failures.budget_impossible);
    try std.testing.expectEqual(FailureKind.budget_impossible, instance.last_failure);

    // Tensor 1 must not have stayed pinned: tensor 2 alone now fits.
    const second = [_]Range{TestBridge.range(2)};
    try std.testing.expect(acquireAllAdmitted(instance, &second, views[0..1]));
    views[0].release();
    try std.testing.expectEqual(@as(u64, 1), instance.acquisitions);
}

test "admission waits for another graph's pin instead of failing" {
    var t: TestBridge = undefined;
    try t.init();
    defer t.deinit();
    const instance = &t.instance;

    // Another graph holds the whole budget through tensor 1.
    instance.views_mutex.lock();
    var held: [1]residency.TensorView = undefined;
    const first = [_]Range{TestBridge.range(1)};
    try std.testing.expect(acquireAllAdmitted(instance, &first, &held));
    var token: u64 = 0;
    try std.testing.expect(rememberView(instance, held[0], &token) != null);
    instance.views_mutex.unlock();

    const Releaser = struct {
        fn run(inst: *Instance, pin: u64) void {
            std.Thread.sleep(20 * std.time.ns_per_ms);
            inst.views_mutex.lock();
            defer inst.views_mutex.unlock();
            var view = (inst.open_views.fetchRemove(pin) orelse unreachable).value;
            view.release();
            inst.pin_released.broadcast();
        }
    };
    const thread = try std.Thread.spawn(.{}, Releaser.run, .{ instance, token });
    defer thread.join();

    instance.views_mutex.lock();
    defer instance.views_mutex.unlock();
    var views: [1]residency.TensorView = undefined;
    const second = [_]Range{TestBridge.range(2)};
    try std.testing.expect(acquireAllAdmitted(instance, &second, &views));
    views[0].release();
    try std.testing.expect(instance.admission_waits >= 1);
    try std.testing.expectEqual(@as(u64, 0), instance.failures.admission_timeout);
}

test "injected acquisition failure fires once on the chosen attempt" {
    var t: TestBridge = undefined;
    try t.init();
    defer t.deinit();
    const instance = &t.instance;
    instance.views_mutex.lock();
    defer instance.views_mutex.unlock();

    instance.fail_countdown = 2;
    var views: [1]residency.TensorView = undefined;
    const first = [_]Range{TestBridge.range(1)};
    try std.testing.expect(acquireAllAdmitted(instance, &first, &views));
    views[0].release();
    try std.testing.expect(!acquireAllAdmitted(instance, &first, &views));
    try std.testing.expectEqual(@as(u64, 1), instance.failures.injected);
    try std.testing.expect(acquireAllAdmitted(instance, &first, &views));
    views[0].release();
}
