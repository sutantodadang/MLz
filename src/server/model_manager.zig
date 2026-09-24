//! Generic refcount-pinned LRU manager. Used to keep a bounded set of loaded
//! models resident and evict the least-recently-used one when capacity is hit.
//!
//! `acquire(name)` returns a pinned handle (refcount > 0); the held entry can't
//! be evicted until `release()`. That makes it safe to load a model, run a whole
//! request against it, and release afterward even while other threads evict.
//!
//! Entries live in a fixed-capacity slice (never reallocated), so the pointer
//! returned by `Handle.value()` is stable for the entry's lifetime.

const std = @import("std");

pub fn LruManager(comptime V: type) type {
    return struct {
        const Self = @This();

        pub const LoadFn = *const fn (ctx: *anyopaque, name: []const u8) anyerror!V;
        pub const UnloadFn = *const fn (ctx: *anyopaque, value: *V) void;

        const Entry = struct {
            name: []u8 = &.{},
            value: V = undefined,
            refs: u32 = 0,
            tick: u64 = 0,
            used: bool = false,
        };

        allocator: std.mem.Allocator,
        entries: []Entry,
        tick: u64 = 0,
        mutex: std.Thread.Mutex = .{},
        load_ctx: *anyopaque,
        load_fn: LoadFn,
        unload_fn: UnloadFn,

        pub const Handle = struct {
            mgr: *Self,
            idx: usize,
            pub fn value(self: Handle) *V {
                return &self.mgr.entries[self.idx].value;
            }
            pub fn release(self: Handle) void {
                self.mgr.releaseIdx(self.idx);
            }
        };

        pub fn init(
            allocator: std.mem.Allocator,
            capacity: usize,
            load_ctx: *anyopaque,
            load_fn: LoadFn,
            unload_fn: UnloadFn,
        ) !Self {
            std.debug.assert(capacity >= 1);
            const entries = try allocator.alloc(Entry, capacity);
            for (entries) |*e| e.* = .{};
            return .{
                .allocator = allocator,
                .entries = entries,
                .load_ctx = load_ctx,
                .load_fn = load_fn,
                .unload_fn = unload_fn,
            };
        }

        pub fn deinit(self: *Self) void {
            for (self.entries) |*e| {
                if (e.used) {
                    self.unload_fn(self.load_ctx, &e.value);
                    self.allocator.free(e.name);
                }
            }
            self.allocator.free(self.entries);
        }

        /// Acquire a pinned handle to the model named `name`, loading it (and
        /// evicting the LRU unpinned entry) if not resident.
        pub fn acquire(self: *Self, name: []const u8) !Handle {
            self.mutex.lock();
            defer self.mutex.unlock();

            // Already resident?
            for (self.entries, 0..) |*e, i| {
                if (e.used and std.mem.eql(u8, e.name, name)) {
                    e.refs += 1;
                    self.tick += 1;
                    e.tick = self.tick;
                    return .{ .mgr = self, .idx = i };
                }
            }

            // Find a slot: prefer an empty one, else evict the LRU unpinned entry.
            var slot: ?usize = null;
            for (self.entries, 0..) |*e, i| {
                if (!e.used) {
                    slot = i;
                    break;
                }
            }
            if (slot == null) {
                var best: ?usize = null;
                var best_tick: u64 = std.math.maxInt(u64);
                for (self.entries, 0..) |*e, i| {
                    if (e.refs == 0 and e.tick < best_tick) {
                        best_tick = e.tick;
                        best = i;
                    }
                }
                const ev = best orelse return error.AllModelsPinned;
                self.unload_fn(self.load_ctx, &self.entries[ev].value);
                self.allocator.free(self.entries[ev].name);
                self.entries[ev] = .{};
                slot = ev;
            }

            const i = slot.?;
            const value = try self.load_fn(self.load_ctx, name);
            self.tick += 1;
            self.entries[i] = .{
                .name = try self.allocator.dupe(u8, name),
                .value = value,
                .refs = 1,
                .tick = self.tick,
                .used = true,
            };
            return .{ .mgr = self, .idx = i };
        }

        fn releaseIdx(self: *Self, idx: usize) void {
            self.mutex.lock();
            defer self.mutex.unlock();
            if (self.entries[idx].refs > 0) self.entries[idx].refs -= 1;
        }

        pub fn loadedCount(self: *Self) usize {
            self.mutex.lock();
            defer self.mutex.unlock();
            var n: usize = 0;
            for (self.entries) |e| {
                if (e.used) n += 1;
            }
            return n;
        }
    };
}

// -----------------------------------------------------------------------------
const TestLoader = struct {
    loads: usize = 0,
    unloads: usize = 0,
    next_id: usize = 0,

    fn load(ctx: *anyopaque, name: []const u8) anyerror!usize {
        _ = name;
        const self: *TestLoader = @ptrCast(@alignCast(ctx));
        self.loads += 1;
        self.next_id += 1;
        return self.next_id;
    }
    fn unload(ctx: *anyopaque, value: *usize) void {
        _ = value;
        const self: *TestLoader = @ptrCast(@alignCast(ctx));
        self.unloads += 1;
    }
};

test "LRU: resident hit does not reload" {
    const a = std.testing.allocator;
    var ldr = TestLoader{};
    var mgr = try LruManager(usize).init(a, 2, &ldr, TestLoader.load, TestLoader.unload);
    defer mgr.deinit();

    const h1 = try mgr.acquire("a");
    h1.release();
    const h2 = try mgr.acquire("a"); // resident, no reload
    h2.release();
    try std.testing.expectEqual(@as(usize, 1), ldr.loads);
    try std.testing.expectEqual(@as(usize, 1), mgr.loadedCount());
}

test "LRU: evicts least-recently-used when over capacity" {
    const a = std.testing.allocator;
    var ldr = TestLoader{};
    var mgr = try LruManager(usize).init(a, 2, &ldr, TestLoader.load, TestLoader.unload);
    defer mgr.deinit();

    (try mgr.acquire("a")).release();
    (try mgr.acquire("b")).release();
    // touch "a" so "b" is now LRU
    (try mgr.acquire("a")).release();
    (try mgr.acquire("c")).release(); // must evict "b"

    try std.testing.expectEqual(@as(usize, 2), mgr.loadedCount());
    try std.testing.expectEqual(@as(usize, 1), ldr.unloads); // b unloaded
    // "a" still resident (no reload), "b" gone (would reload)
    try std.testing.expectEqual(@as(usize, 3), ldr.loads); // a,b,c
    (try mgr.acquire("a")).release();
    try std.testing.expectEqual(@as(usize, 3), ldr.loads); // a was kept
}

test "LRU: pinned entry is not evicted" {
    const a = std.testing.allocator;
    var ldr = TestLoader{};
    var mgr = try LruManager(usize).init(a, 2, &ldr, TestLoader.load, TestLoader.unload);
    defer mgr.deinit();

    const pinned = try mgr.acquire("a"); // held
    (try mgr.acquire("b")).release();
    // both slots full, "a" pinned -> evict "b" for "c"
    (try mgr.acquire("c")).release();
    try std.testing.expectEqual(@as(usize, 1), ldr.unloads); // only b
    pinned.release();
}

test "LRU: all pinned returns error" {
    const a = std.testing.allocator;
    var ldr = TestLoader{};
    var mgr = try LruManager(usize).init(a, 2, &ldr, TestLoader.load, TestLoader.unload);
    defer mgr.deinit();

    const h1 = try mgr.acquire("a");
    const h2 = try mgr.acquire("b");
    try std.testing.expectError(error.AllModelsPinned, mgr.acquire("c"));
    h1.release();
    h2.release();
}
