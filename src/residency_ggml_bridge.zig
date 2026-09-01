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
    error{ BridgeAlreadyInitialized, BridgeNotInitialized };

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
    // Views acquired by the pre-hook, keyed by source id. Hooks run on graph
    // thread 0 only, between barriers, so plain indexing is sufficient and
    // allocation-free. Indexed by (source_id - 1).
    open_views: std.ArrayList(?residency.TensorView),
};

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

    instance.manager = try Manager.init(allocator, &instance.store, budget_bytes);
    errdefer instance.manager.deinit();
    instance.sources = .empty;
    instance.open_views = .empty;

    g_instance = instance;
}

pub fn deinit(allocator: std.mem.Allocator) void {
    const instance = g_instance orelse return;
    for (instance.open_views.items) |maybe_view| {
        std.debug.assert(maybe_view == null); // unbalanced pre/post hooks
    }
    instance.open_views.deinit(allocator);
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
        try instance.open_views.append(instance.allocator, null);
        errdefer _ = instance.open_views.pop();
        try instance.manager.register(descriptor.handle, file_offset, byte_len);
        const capacity = try instance.manager.rangeCapacity(descriptor.handle, 0);
        if (capacity < byte_len) return Error.BudgetExceeded;
    }
}

/// Maps a registered tensor's full span and records the open view under the
/// registry's 1-based source id. Called by the backend pre-hook on graph
/// thread 0 between barriers; the returned pointer stays valid until the
/// matching release call.
pub fn acquireCallback(source_id: u32, file_offset: u64, byte_len: usize) callconv(.c) ?*anyopaque {
    const instance = g_instance orelse {
        std.debug.print("mlz bridge: acquire with no instance\n", .{});
        return null;
    };
    if (source_id == 0 or source_id > instance.sources.items.len) {
        std.debug.print("mlz bridge: acquire bad source_id={d}\n", .{source_id});
        return null;
    }
    const slot = &instance.open_views.items[source_id - 1];
    if (slot.* != null) {
        std.debug.print("mlz bridge: acquire source_id={d} already open\n", .{source_id});
        return null; // unbalanced pre-hook
    }

    const source = instance.sources.items[source_id - 1];
    if (source.file_offset != file_offset or source.byte_len != byte_len) {
        std.debug.print(
            "mlz bridge: acquire span mismatch id={d}: C({d},{d}) vs Zig({d},{d})\n",
            .{ source_id, file_offset, byte_len, source.file_offset, source.byte_len },
        );
        return null; // span mismatch: C registry disagrees with Zig mirror
    }

    slot.* = instance.manager.acquire(source.handle) catch |err| {
        std.debug.print("mlz bridge: acquire manager error {s} (budget={d})\n", .{ @errorName(err), instance.manager.budget_bytes });
        return null;
    };
    return @ptrCast(@constCast(slot.*.?.data.ptr));
}

/// Releases the view pinned for a 1-based source id. Called by the post-hook.
pub fn releaseCallback(source_id: u32) callconv(.c) bool {
    const instance = g_instance orelse return false;
    if (source_id == 0 or source_id > instance.open_views.items.len) return false;
    const slot = &instance.open_views.items[source_id - 1];
    var view = slot.* orelse return false;
    view.release();
    slot.* = null;
    return true;
}
