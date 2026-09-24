const std = @import("std");

const c = @cImport({
    @cInclude("residency_mmap.h");
});

pub const Error = error{
    InvalidBudget,
    InvalidRange,
    DuplicateTensor,
    UnknownTensor,
    TensorBusy,
    BudgetExceeded,
    BackingOpenFailed,
    MapFailed,
    PrefaultFailed,
    ThreadSpawnFailed,
    InvalidSchedulerConfig,
    PrefetchQueueFull,
    SchedulerShuttingDown,
    OutOfMemory,
};

/// Stable identifier used by callers instead of retaining an address that can
/// become invalid after an eviction.
pub const TensorHandle = struct {
    id: u64,

    pub fn eql(a: TensorHandle, b: TensorHandle) bool {
        return a.id == b.id;
    }
};

pub const Residency = enum {
    non_resident,
    resident,
};

/// A read-only, mmap-capable source for tensor bytes. Mappings are established
/// per tensor or range fault rather than mapping the complete model eagerly.
pub const BackingStore = struct {
    handle: *c.mlz_backing_handle,
    size: u64,

    pub fn open(path_z: [:0]const u8) Error!BackingStore {
        var size: u64 = 0;
        const handle = c.mlz_backing_open(path_z.ptr, &size) orelse return Error.BackingOpenFailed;
        return .{ .handle = handle, .size = size };
    }

    pub fn close(self: *BackingStore) void {
        c.mlz_backing_close(self.handle);
        self.* = undefined;
    }

    fn map(self: *BackingStore, offset: u64, len: usize) Error!MappedRegion {
        if (len == 0 or offset > self.size or len > self.size - offset) {
            return Error.InvalidRange;
        }
        var region: c.mlz_mapped_region = undefined;
        if (c.mlz_backing_map(self.handle, offset, len, &region) != 0) {
            return Error.MapFailed;
        }
        return .{ .native = region };
    }
};

const MappedRegion = struct {
    native: c.mlz_mapped_region,

    fn bytes(self: *const MappedRegion, len: usize) []const u8 {
        const ptr: [*]const u8 = @ptrCast(self.native.data);
        return ptr[0..len];
    }

    fn unmap(self: *MappedRegion) void {
        c.mlz_backing_unmap(&self.native);
    }
};

const Entry = struct {
    /// Complete logical tensor range in the backing store.
    offset: u64,
    len: usize,
};

/// One active mapped window. A tensor may have several windows at once, as
/// long as the union of their mapped bytes stays within the budget.
const Window = struct {
    tensor: TensorHandle,
    /// Mapped logical subrange within the tensor.
    mapped_offset: u64,
    mapped_len: usize,
    mapping_bytes: usize,
    mapping: MappedRegion,
    last_used: u64 = 0,
    pin_count: usize = 0,

    fn covers(self: *const Window, absolute_offset: u64, requested_end: u64) bool {
        const window_end = self.mapped_offset + @as(u64, @intCast(self.mapped_len));
        return absolute_offset >= self.mapped_offset and requested_end <= window_end;
    }
};

pub const ReplacementPolicy = enum {
    /// Evict the least recently accessed unpinned mapping.
    lru,
    /// Prefer the largest unpinned mapping, using age as a tie-breaker. This
    /// can satisfy large incoming windows with fewer unmap operations.
    largest_first,
};

pub const Metrics = struct {
    budget_bytes: usize,
    resident_bytes: usize,
    peak_resident_bytes: usize,
    registered_tensors: usize,
    resident_tensors: usize,
    faults: u64,
    hits: u64,
    evictions: u64,
    bytes_mapped: u64,
    bytes_evicted: u64,
    prefetches: u64,
    prefetched_bytes: u64,

    pub fn faultRate(self: Metrics) f64 {
        const accesses = self.faults + self.hits;
        if (accesses == 0) return 0;
        return @as(f64, @floatFromInt(self.faults)) / @as(f64, @floatFromInt(accesses));
    }
};

/// A budgeted residency manager for immutable tensor data.
///
/// `acquire` transparently faults a tensor range in from the backing store.
/// Callers must keep the returned `TensorView` alive while reading its bytes;
/// the pin prevents LRU eviction until `release` is called.
pub const Manager = struct {
    allocator: std.mem.Allocator,
    store: *BackingStore,
    budget_bytes: usize,
    resident_bytes: usize = 0,
    peak_resident_bytes: usize = 0,
    faults: u64 = 0,
    hits: u64 = 0,
    evictions: u64 = 0,
    bytes_mapped: u64 = 0,
    bytes_evicted: u64 = 0,
    prefetches: u64 = 0,
    prefetched_bytes: u64 = 0,
    clock: u64 = 0,
    replacement_policy: ReplacementPolicy = .lru,
    mutex: std.Thread.Mutex = .{},
    entries: std.AutoHashMap(TensorHandle, Entry),
    /// Active mapped windows across all tensors, keyed by global slot index.
    windows: std.AutoHashMap(u32, Window),
    next_window_slot: u32 = 0,

    pub fn init(allocator: std.mem.Allocator, store: *BackingStore, budget_bytes: usize) Error!Manager {
        if (budget_bytes == 0) return Error.InvalidBudget;
        if (@sizeOf(usize) < @sizeOf(u64) and store.size > std.math.maxInt(usize)) {
            return Error.InvalidRange;
        }
        return .{
            .allocator = allocator,
            .store = store,
            .budget_bytes = budget_bytes,
            .entries = std.AutoHashMap(TensorHandle, Entry).init(allocator),
            .windows = std.AutoHashMap(u32, Window).init(allocator),
        };
    }

    pub fn deinit(self: *Manager) void {
        self.mutex.lock();
        var it = self.windows.valueIterator();
        while (it.next()) |window| {
            std.debug.assert(window.pin_count == 0);
            window.mapping.unmap();
        }
        self.windows.deinit();
        self.entries.deinit();
        self.mutex.unlock();
        self.* = undefined;
    }

    pub fn setReplacementPolicy(self: *Manager, policy: ReplacementPolicy) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.replacement_policy = policy;
    }

    pub fn getReplacementPolicy(self: *Manager) ReplacementPolicy {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.replacement_policy;
    }

    pub fn register(self: *Manager, handle: TensorHandle, offset: u64, len: usize) Error!void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (len == 0 or offset > self.store.size or len > self.store.size - offset) {
            return Error.InvalidRange;
        }
        if (self.entries.contains(handle)) return Error.DuplicateTensor;
        self.entries.put(handle, .{
            .offset = offset,
            .len = len,
        }) catch return Error.OutOfMemory;
    }
    pub fn unregister(self: *Manager, handle: TensorHandle) Error!void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (!self.entries.contains(handle)) return Error.UnknownTensor;
        // Reject while any window of this tensor is pinned.
        var pinned = false;
        var wit = self.windows.iterator();
        while (wit.next()) |w| {
            if (!w.value_ptr.tensor.eql(handle)) continue;
            if (w.value_ptr.pin_count != 0) {
                pinned = true;
                break;
            }
        }
        if (pinned) return Error.TensorBusy;
        self.evictTensorLocked(handle);
        _ = self.entries.remove(handle);
    }

    pub fn state(self: *Manager, handle: TensorHandle) Error!Residency {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (!self.entries.contains(handle)) return Error.UnknownTensor;
        var wit = self.windows.iterator();
        while (wit.next()) |w| {
            if (w.value_ptr.tensor.eql(handle)) return .resident;
        }
        return .non_resident;
    }

    pub fn acquire(self: *Manager, handle: TensorHandle) Error!TensorView {
        self.mutex.lock();
        const entry = self.entries.get(handle) orelse {
            self.mutex.unlock();
            return Error.UnknownTensor;
        };
        const len = entry.len;
        self.mutex.unlock();
        return self.acquireRange(handle, 0, len);
    }

    /// Maximum logical bytes that can be mapped at `tensor_offset` without
    /// exceeding the budget after OS mapping-alignment overhead is included.
    pub fn rangeCapacity(self: *Manager, handle: TensorHandle, tensor_offset: usize) Error!usize {
        self.mutex.lock();
        defer self.mutex.unlock();
        const entry = self.entries.get(handle) orelse return Error.UnknownTensor;
        if (tensor_offset >= entry.len) return Error.InvalidRange;
        const absolute_offset = entry.offset + @as(u64, @intCast(tensor_offset));
        const granularity = try mappingGranularity();
        const alignment_prefix: usize = @intCast(absolute_offset % @as(u64, @intCast(granularity)));
        if (alignment_prefix >= self.budget_bytes) return Error.BudgetExceeded;
        return @min(entry.len - tensor_offset, self.budget_bytes - alignment_prefix);
    }

    /// Acquires a logical subrange of a tensor. This is the preferred access
    /// path for tensors larger than the residency budget: callers process the
    /// tensor in bounded windows and release each view before requesting the
    /// next one.
    pub fn acquireRange(self: *Manager, handle: TensorHandle, tensor_offset: usize, len: usize) Error!TensorView {
        self.mutex.lock();
        defer self.mutex.unlock();
        return self.acquireRangeLocked(handle, tensor_offset, len);
    }

    fn acquireRangeLocked(self: *Manager, handle: TensorHandle, tensor_offset: usize, len: usize) Error!TensorView {
        const entry = self.entries.getPtr(handle) orelse return Error.UnknownTensor;
        if (len == 0 or tensor_offset > entry.len or len > entry.len - tensor_offset) {
            return Error.InvalidRange;
        }
        const absolute_offset = entry.offset + @as(u64, @intCast(tensor_offset));
        const requested_end = absolute_offset + @as(u64, @intCast(len));

        // Multi-window hit: any existing window of this tensor that covers the
        // requested range satisfies the acquire.
        var hit_slot: ?u32 = null;
        var wit = self.windows.iterator();
        while (wit.next()) |w| {
            if (!w.value_ptr.tensor.eql(handle)) continue;
            if (w.value_ptr.covers(absolute_offset, requested_end)) {
                hit_slot = w.key_ptr.*;
                break;
            }
        }

        var slot: u32 = undefined;
        if (hit_slot) |found| {
            self.hits += 1;
            slot = found;
        } else {
            const mapping_bytes = try mappedSize(absolute_offset, len);
            if (mapping_bytes > self.budget_bytes) return Error.BudgetExceeded;
            try self.evictUntilFits(mapping_bytes);

            // Map only after enough budget is available. If mmap fails, all
            // counters and the registry remain internally consistent.
            const window = Window{
                .tensor = handle,
                .mapped_offset = absolute_offset,
                .mapped_len = len,
                .mapping_bytes = mapping_bytes,
                .mapping = try self.store.map(absolute_offset, len),
            };
            // Insertion can only fail before the mapping is registered in the
            // accounting; release it explicitly on OOM.
            slot = self.next_window_slot;
            self.windows.put(slot, window) catch {
                var failed = window;
                failed.mapping.unmap();
                return Error.OutOfMemory;
            };
            self.next_window_slot += 1;
            self.resident_bytes += mapping_bytes;
            self.peak_resident_bytes = @max(self.peak_resident_bytes, self.resident_bytes);
            self.faults += 1;
            self.bytes_mapped += mapping_bytes;
        }

        self.clock +%= 1;
        const wptr = self.windows.getPtr(slot) orelse unreachable;
        wptr.last_used = self.clock;
        wptr.pin_count += 1;

        return .{
            .manager = self,
            .handle = handle,
            .window_slot = slot,
            .data = self.windowBytesLocked(slot, tensor_offset, len),
        };
    }

    /// Maps (or reuses) a range and synchronously touches every intersecting OS
    /// page. This prepares the range for a later acquire without retaining a
    /// pin after the call returns.
    pub fn prefetchRange(self: *Manager, handle: TensorHandle, tensor_offset: usize, len: usize) Error!void {
        var view = try self.acquireRange(handle, tensor_offset, len);
        defer view.release();

        // The view pin keeps the mapping stable while the potentially slow OS
        // page touches run. Do not retain the manager mutex during this I/O so
        // unrelated resident hits/releases can proceed concurrently.
        self.mutex.lock();
        const window = self.windows.getPtr(view.window_slot) orelse unreachable;
        var native = window.mapping.native;
        native.data = view.data.ptr;
        self.mutex.unlock();

        if (c.mlz_mapped_region_prefault(&native, len) != 0) {
            return Error.PrefaultFailed;
        }

        self.mutex.lock();
        self.prefetches += 1;
        self.prefetched_bytes += len;
        self.mutex.unlock();
    }

    pub fn prefetch(self: *Manager, handle: TensorHandle) Error!void {
        self.mutex.lock();
        const entry = self.entries.get(handle) orelse {
            self.mutex.unlock();
            return Error.UnknownTensor;
        };
        const len = entry.len;
        self.mutex.unlock();
        return self.prefetchRange(handle, 0, len);
    }

    pub fn prefetchRangeAsync(
        self: *Manager,
        allocator: std.mem.Allocator,
        handle: TensorHandle,
        tensor_offset: usize,
        len: usize,
    ) Error!PrefetchTask {
        const task_state = allocator.create(PrefetchTask.State) catch return Error.OutOfMemory;
        task_state.* = .{
            .manager = self,
            .handle = handle,
            .tensor_offset = tensor_offset,
            .len = len,
        };
        const thread = std.Thread.spawn(.{}, PrefetchTask.run, .{task_state}) catch {
            allocator.destroy(task_state);
            return Error.ThreadSpawnFailed;
        };
        return .{ .allocator = allocator, .state = task_state, .thread = thread };
    }

    pub fn metrics(self: *Manager) Metrics {
        self.mutex.lock();
        defer self.mutex.unlock();
        // Count distinct tensors that have at least one active window.
        var resident_tensors: usize = 0;
        var eit = self.entries.keyIterator();
        while (eit.next()) |key| {
            var wit = self.windows.valueIterator();
            while (wit.next()) |w| {
                if (w.tensor.eql(key.*)) {
                    resident_tensors += 1;
                    break;
                }
            }
        }
        return .{
            .budget_bytes = self.budget_bytes,
            .resident_bytes = self.resident_bytes,
            .peak_resident_bytes = self.peak_resident_bytes,
            .registered_tensors = self.entries.count(),
            .resident_tensors = resident_tensors,
            .faults = self.faults,
            .hits = self.hits,
            .evictions = self.evictions,
            .bytes_mapped = self.bytes_mapped,
            .bytes_evicted = self.bytes_evicted,
            .prefetches = self.prefetches,
            .prefetched_bytes = self.prefetched_bytes,
        };
    }

    fn shouldReplaceVictim(self: *Manager, candidate: *const Window, current: *const Window) bool {
        return switch (self.replacement_policy) {
            .lru => candidate.last_used < current.last_used,
            .largest_first => candidate.mapping_bytes > current.mapping_bytes or
                (candidate.mapping_bytes == current.mapping_bytes and candidate.last_used < current.last_used),
        };
    }

    fn evictUntilFits(self: *Manager, incoming: usize) Error!void {
        while (incoming > self.budget_bytes - self.resident_bytes) {
            var victim: ?u32 = null;
            var it = self.windows.iterator();
            while (it.next()) |item| {
                const candidate = item.value_ptr;
                if (candidate.pin_count != 0) continue;
                if (victim) |current_slot| {
                    const current = self.windows.getPtr(current_slot) orelse unreachable;
                    if (self.shouldReplaceVictim(candidate, current)) victim = item.key_ptr.*;
                } else {
                    victim = item.key_ptr.*;
                }
            }
            const slot = victim orelse return Error.BudgetExceeded;
            self.evictWindowLocked(slot);
        }
    }

    fn evictWindowLocked(self: *Manager, slot: u32) void {
        const window = self.windows.getPtr(slot) orelse unreachable;
        std.debug.assert(window.pin_count == 0);
        window.mapping.unmap();
        self.resident_bytes -= window.mapping_bytes;
        self.evictions += 1;
        self.bytes_evicted += window.mapping_bytes;
        _ = self.windows.remove(slot);
    }

    fn evictTensorLocked(self: *Manager, handle: TensorHandle) void {
        var slots: [64]u32 = undefined;
        var count: usize = 0;
        var it = self.windows.iterator();
        while (it.next()) |item| {
            if (!item.value_ptr.tensor.eql(handle)) continue;
            if (count == slots.len) break; // defensive: never expected
            slots[count] = item.key_ptr.*;
            count += 1;
        }
        for (slots[0..count]) |slot| self.evictWindowLocked(slot);
    }

    fn windowBytesLocked(self: *Manager, slot: u32, tensor_offset: usize, len: usize) []const u8 {
        const window = self.windows.getPtr(slot) orelse unreachable;
        const entry = self.entries.getPtr(window.tensor) orelse unreachable;
        const absolute_offset = entry.offset + @as(u64, @intCast(tensor_offset));
        std.debug.assert(absolute_offset >= window.mapped_offset);
        const mapping_offset: usize = @intCast(absolute_offset - window.mapped_offset);
        return window.mapping.bytes(window.mapped_len)[mapping_offset..][0..len];
    }

    fn release(self: *Manager, slot: u32) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        const window = self.windows.getPtr(slot) orelse unreachable;
        std.debug.assert(window.pin_count > 0);
        window.pin_count -= 1;
    }
};

pub const TensorView = struct {
    manager: *Manager,
    handle: TensorHandle,
    window_slot: u32,
    data: []const u8,
    released: bool = false,

    pub fn bytes(self: *TensorView) []const u8 {
        std.debug.assert(!self.released);
        return self.data;
    }

    pub fn release(self: *TensorView) void {
        if (!self.released) {
            self.manager.release(self.window_slot);
            self.released = true;
        }
    }
};

pub const PrefetchTask = struct {
    const State = struct {
        manager: *Manager,
        handle: TensorHandle,
        tensor_offset: usize,
        len: usize,
        result: Error!void = {},
    };

    allocator: std.mem.Allocator,
    state: *State,
    thread: std.Thread,
    joined: bool = false,

    fn run(state: *State) void {
        state.result = state.manager.prefetchRange(state.handle, state.tensor_offset, state.len);
    }

    /// Waits for the background prefault and consumes the task. Must be called
    /// exactly once so the thread is joined and task state is released.
    pub fn wait(self: *PrefetchTask) Error!void {
        std.debug.assert(!self.joined);
        self.thread.join();
        self.joined = true;
        defer self.allocator.destroy(self.state);
        return self.state.result;
    }
};

pub const PrefetchSchedulerMetrics = struct {
    worker_count: usize,
    queue_capacity: usize,
    queued: usize,
    active_workers: usize,
    submitted: u64,
    completed: u64,
    succeeded: u64,
    failed: u64,
};

/// Fixed-size worker pool for bounded asynchronous prefetch. The scheduler is
/// heap allocated so worker threads always receive a stable address.
///
/// Submission is deliberately non-blocking: callers receive
/// `error.PrefetchQueueFull` when the bounded queue is saturated and may fall
/// back to synchronous access instead of growing memory without limit.
pub const PrefetchScheduler = struct {
    const TaskState = struct {
        manager: *Manager,
        handle: TensorHandle,
        tensor_offset: usize,
        len: usize,
        mutex: std.Thread.Mutex = .{},
        completed: std.Thread.Condition = .{},
        done: bool = false,
        result: Error!void = {},
    };

    allocator: std.mem.Allocator,
    manager: *Manager,
    workers: []std.Thread,
    queue: []*TaskState,
    queue_head: usize = 0,
    queue_tail: usize = 0,
    queue_count: usize = 0,
    active_workers: usize = 0,
    submitted: u64 = 0,
    completed_count: u64 = 0,
    succeeded: u64 = 0,
    failed: u64 = 0,
    shutting_down: bool = false,
    mutex: std.Thread.Mutex = .{},
    work_available: std.Thread.Condition = .{},
    idle: std.Thread.Condition = .{},

    pub fn init(
        allocator: std.mem.Allocator,
        manager: *Manager,
        worker_count: usize,
        queue_capacity: usize,
    ) Error!*PrefetchScheduler {
        if (worker_count == 0 or queue_capacity == 0) return Error.InvalidSchedulerConfig;

        const self = allocator.create(PrefetchScheduler) catch return Error.OutOfMemory;
        errdefer allocator.destroy(self);
        const workers = allocator.alloc(std.Thread, worker_count) catch return Error.OutOfMemory;
        errdefer allocator.free(workers);
        const queue = allocator.alloc(*TaskState, queue_capacity) catch return Error.OutOfMemory;
        errdefer allocator.free(queue);
        self.* = .{
            .allocator = allocator,
            .manager = manager,
            .workers = workers,
            .queue = queue,
        };

        var spawned: usize = 0;
        errdefer {
            self.mutex.lock();
            self.shutting_down = true;
            self.work_available.broadcast();
            self.mutex.unlock();
            for (self.workers[0..spawned]) |thread| thread.join();
        }
        while (spawned < worker_count) : (spawned += 1) {
            self.workers[spawned] = std.Thread.spawn(.{}, workerMain, .{self}) catch {
                return Error.ThreadSpawnFailed;
            };
        }
        return self;
    }

    pub fn submit(
        self: *PrefetchScheduler,
        handle: TensorHandle,
        tensor_offset: usize,
        len: usize,
    ) Error!ScheduledPrefetchTask {
        const state = self.allocator.create(TaskState) catch return Error.OutOfMemory;
        state.* = .{
            .manager = self.manager,
            .handle = handle,
            .tensor_offset = tensor_offset,
            .len = len,
        };

        self.mutex.lock();
        if (self.shutting_down) {
            self.mutex.unlock();
            self.allocator.destroy(state);
            return Error.SchedulerShuttingDown;
        }
        if (self.queue_count == self.queue.len) {
            self.mutex.unlock();
            self.allocator.destroy(state);
            return Error.PrefetchQueueFull;
        }
        self.queue[self.queue_tail] = state;
        self.queue_tail = (self.queue_tail + 1) % self.queue.len;
        self.queue_count += 1;
        self.submitted += 1;
        self.work_available.signal();
        self.mutex.unlock();
        return .{ .allocator = self.allocator, .state = state };
    }

    /// Waits until every currently accepted task has completed. Another thread
    /// may submit immediately afterward unless `shutdown` was called first.
    pub fn waitIdle(self: *PrefetchScheduler) void {
        self.mutex.lock();
        while (self.queue_count != 0 or self.active_workers != 0) {
            self.idle.wait(&self.mutex);
        }
        self.mutex.unlock();
    }

    pub fn shutdown(self: *PrefetchScheduler) void {
        self.mutex.lock();
        if (!self.shutting_down) {
            self.shutting_down = true;
            self.work_available.broadcast();
        }
        self.mutex.unlock();
    }

    /// Drains accepted work and joins workers. Every returned task must still
    /// be consumed with `wait`; task state is owned by its task handle.
    pub fn deinit(self: *PrefetchScheduler) void {
        self.shutdown();
        for (self.workers) |thread| thread.join();
        std.debug.assert(self.queue_count == 0);
        std.debug.assert(self.active_workers == 0);
        const allocator = self.allocator;
        allocator.free(self.queue);
        allocator.free(self.workers);
        allocator.destroy(self);
    }

    pub fn metrics(self: *PrefetchScheduler) PrefetchSchedulerMetrics {
        self.mutex.lock();
        defer self.mutex.unlock();
        return .{
            .worker_count = self.workers.len,
            .queue_capacity = self.queue.len,
            .queued = self.queue_count,
            .active_workers = self.active_workers,
            .submitted = self.submitted,
            .completed = self.completed_count,
            .succeeded = self.succeeded,
            .failed = self.failed,
        };
    }

    fn workerMain(self: *PrefetchScheduler) void {
        while (true) {
            self.mutex.lock();
            while (self.queue_count == 0 and !self.shutting_down) {
                self.work_available.wait(&self.mutex);
            }
            if (self.queue_count == 0 and self.shutting_down) {
                self.mutex.unlock();
                return;
            }
            const state = self.queue[self.queue_head];
            self.queue_head = (self.queue_head + 1) % self.queue.len;
            self.queue_count -= 1;
            self.active_workers += 1;
            self.mutex.unlock();

            const result = state.manager.prefetchRange(state.handle, state.tensor_offset, state.len);
            state.mutex.lock();
            state.result = result;
            state.done = true;
            state.completed.broadcast();
            state.mutex.unlock();

            self.mutex.lock();
            self.active_workers -= 1;
            self.completed_count += 1;
            if (result) |_| {
                self.succeeded += 1;
            } else |_| {
                self.failed += 1;
            }
            if (self.queue_count == 0 and self.active_workers == 0) self.idle.broadcast();
            self.mutex.unlock();
        }
    }
};

pub const ScheduledPrefetchTask = struct {
    allocator: std.mem.Allocator,
    state: *PrefetchScheduler.TaskState,
    consumed: bool = false,

    /// Waits for completion and consumes the task. Must be called exactly once.
    pub fn wait(self: *ScheduledPrefetchTask) Error!void {
        std.debug.assert(!self.consumed);
        const state = self.state;
        state.mutex.lock();
        while (!state.done) state.completed.wait(&state.mutex);
        const result = state.result;
        state.mutex.unlock();
        self.consumed = true;
        self.allocator.destroy(state);
        return result;
    }
};

pub fn mappingGranularity() Error!usize {
    const granularity = c.mlz_backing_granularity();
    if (granularity == 0) return Error.MapFailed;
    return granularity;
}

/// Current process resident set size in bytes, or null when unsupported.
pub fn currentRss() ?u64 {
    const bytes = c.mlz_process_current_rss();
    return if (bytes == 0) null else bytes;
}

/// Peak process resident set size in bytes, or null when unsupported.
pub fn peakRss() ?u64 {
    const bytes = c.mlz_process_peak_rss();
    return if (bytes == 0) null else bytes;
}

fn mappedSize(offset: u64, len: usize) Error!usize {
    const granularity = try mappingGranularity();
    const delta_u64 = offset % @as(u64, @intCast(granularity));
    if (delta_u64 > std.math.maxInt(usize)) return Error.InvalidRange;
    return std.math.add(usize, @as(usize, @intCast(delta_u64)), len) catch Error.InvalidRange;
}

fn createTestBacking(tmp: *std.testing.TmpDir, len: usize) ![:0]u8 {
    var file = try tmp.dir.createFile("tensor.bin", .{});
    defer file.close();
    var buffer: [256]u8 = undefined;
    var written: usize = 0;
    while (written < len) {
        const n = @min(buffer.len, len - written);
        for (buffer[0..n], 0..) |*byte, i| byte.* = @truncate(written + i);
        try file.writeAll(buffer[0..n]);
        written += n;
    }
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "tensor.bin");
    defer std.testing.allocator.free(path);
    return try std.testing.allocator.dupeZ(u8, path);
}

test "faults tensors transparently and enforces resident budget with LRU" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const granularity = c.mlz_backing_granularity();
    try std.testing.expect(granularity > 0);
    const path_z = try createTestBacking(&tmp, granularity * 2 + 64);
    defer std.testing.allocator.free(path_z);

    var store = try BackingStore.open(path_z);
    defer store.close();
    var manager = try Manager.init(std.testing.allocator, &store, 128);
    defer manager.deinit();

    const a = TensorHandle{ .id = 1 };
    const b = TensorHandle{ .id = 2 };
    const c_handle = TensorHandle{ .id = 3 };
    try manager.register(a, 0, 64);
    try manager.register(b, granularity, 64);
    try manager.register(c_handle, granularity * 2, 64);

    var va = try manager.acquire(a);
    try std.testing.expectEqual(@as(u8, 0), va.bytes()[0]);
    va.release();
    var vb = try manager.acquire(b);
    try std.testing.expectEqual(@as(u8, @truncate(granularity)), vb.bytes()[0]);
    vb.release();

    // Refresh A; B is now the least-recently-used unpinned tensor.
    va = try manager.acquire(a);
    va.release();
    var vc = try manager.acquire(c_handle);
    try std.testing.expectEqual(@as(u8, @truncate(granularity * 2)), vc.bytes()[0]);
    vc.release();

    try std.testing.expectEqual(Residency.resident, try manager.state(a));
    try std.testing.expectEqual(Residency.non_resident, try manager.state(b));
    try std.testing.expectEqual(Residency.resident, try manager.state(c_handle));

    // Re-accessing the evicted tensor transparently maps its original backing
    // range again and returns the same bytes.
    vb = try manager.acquire(b);
    const b0: u8 = @truncate(granularity);
    try std.testing.expectEqualSlices(u8, &.{ b0, b0 +% 1, b0 +% 2, b0 +% 3 }, vb.bytes()[0..4]);
    vb.release();

    const metrics = manager.metrics();
    try std.testing.expectEqual(@as(usize, 128), metrics.resident_bytes);
    try std.testing.expectEqual(@as(usize, 128), metrics.peak_resident_bytes);
    try std.testing.expectEqual(@as(u64, 4), metrics.faults);
    try std.testing.expectEqual(@as(u64, 1), metrics.hits);
    try std.testing.expectEqual(@as(u64, 2), metrics.evictions);
}

test "largest-first replacement can free a large window with one eviction" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const granularity = try mappingGranularity();
    const large = granularity * 2;
    const small = granularity;
    const path_z = try createTestBacking(&tmp, large * 2 + small);
    defer std.testing.allocator.free(path_z);

    var store = try BackingStore.open(path_z);
    defer store.close();
    var manager = try Manager.init(std.testing.allocator, &store, large + small);
    defer manager.deinit();
    manager.setReplacementPolicy(.largest_first);
    try std.testing.expectEqual(ReplacementPolicy.largest_first, manager.getReplacementPolicy());

    const large_old = TensorHandle{ .id = 1 };
    const small_new = TensorHandle{ .id = 2 };
    const incoming = TensorHandle{ .id = 3 };
    try manager.register(large_old, 0, large);
    try manager.register(small_new, large, small);
    try manager.register(incoming, large + small, large);

    var first = try manager.acquire(large_old);
    first.release();
    var second = try manager.acquire(small_new);
    second.release();
    var third = try manager.acquire(incoming);
    third.release();

    try std.testing.expectEqual(Residency.non_resident, try manager.state(large_old));
    try std.testing.expectEqual(Residency.resident, try manager.state(small_new));
    try std.testing.expectEqual(Residency.resident, try manager.state(incoming));
    try std.testing.expectEqual(@as(u64, 1), manager.metrics().evictions);
}

test "pinned tensor cannot be evicted" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const granularity = c.mlz_backing_granularity();
    try std.testing.expect(granularity > 0);
    const path_z = try createTestBacking(&tmp, granularity + 64);
    defer std.testing.allocator.free(path_z);

    var store = try BackingStore.open(path_z);
    defer store.close();
    var manager = try Manager.init(std.testing.allocator, &store, 64);
    defer manager.deinit();
    const a = TensorHandle{ .id = 1 };
    const b = TensorHandle{ .id = 2 };
    try manager.register(a, 0, 64);
    try manager.register(b, granularity, 64);

    var view = try manager.acquire(a);
    try std.testing.expectError(Error.BudgetExceeded, manager.acquire(b));
    try std.testing.expectEqualSlices(u8, &.{ 0, 1, 2, 3 }, view.bytes()[0..4]);
    view.release();

    var second = try manager.acquire(b);
    second.release();
    try std.testing.expectEqual(Residency.non_resident, try manager.state(a));
}

test "budget accounts for mmap alignment overhead" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const granularity = c.mlz_backing_granularity();
    try std.testing.expect(granularity > 1);
    const tensor_len: usize = 32;
    const path_z = try createTestBacking(&tmp, granularity + tensor_len);
    defer std.testing.allocator.free(path_z);

    var store = try BackingStore.open(path_z);
    defer store.close();

    // An unaligned range maps its logical bytes plus the prefix back to the OS
    // allocation boundary. A budget for logical bytes alone must not admit it.
    var too_small = try Manager.init(std.testing.allocator, &store, tensor_len);
    defer too_small.deinit();
    try too_small.register(.{ .id = 1 }, 1, tensor_len);
    try std.testing.expectError(Error.BudgetExceeded, too_small.acquire(.{ .id = 1 }));

    var exact = try Manager.init(std.testing.allocator, &store, tensor_len + 1);
    defer exact.deinit();
    try exact.register(.{ .id = 1 }, 1, tensor_len);
    var view = try exact.acquire(.{ .id = 1 });
    defer view.release();
    try std.testing.expectEqual(@as(u8, 1), view.bytes()[0]);
    try std.testing.expectEqual(tensor_len + 1, exact.metrics().resident_bytes);
}

test "range access streams a tensor larger than the budget" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const granularity = c.mlz_backing_granularity();
    try std.testing.expect(granularity > 0);
    const path_z = try createTestBacking(&tmp, granularity * 3);
    defer std.testing.allocator.free(path_z);

    var store = try BackingStore.open(path_z);
    defer store.close();
    var manager = try Manager.init(std.testing.allocator, &store, granularity);
    defer manager.deinit();
    const tensor = TensorHandle{ .id = 1 };
    try manager.register(tensor, 0, granularity * 3);

    // Whole-tensor access cannot fit, but page-sized windows can traverse all
    // bytes while the active mapping remains bounded to one page.
    try std.testing.expectError(Error.BudgetExceeded, manager.acquire(tensor));
    for (0..3) |page| {
        var view = try manager.acquireRange(tensor, page * granularity, granularity);
        try std.testing.expectEqual(@as(u8, @truncate(page * granularity)), view.bytes()[0]);
        try std.testing.expectEqual(@as(u8, @truncate((page + 1) * granularity - 1)), view.bytes()[granularity - 1]);
        view.release();
        try std.testing.expect(manager.metrics().resident_bytes <= granularity);
    }

    const metrics = manager.metrics();
    try std.testing.expectEqual(granularity, metrics.peak_resident_bytes);
    try std.testing.expectEqual(@as(u64, 3), metrics.faults);
    try std.testing.expectEqual(@as(u64, 2), metrics.evictions);
}

test "range hits reuse a containing mapping and pinned views prevent remap" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const granularity = c.mlz_backing_granularity();
    try std.testing.expect(granularity >= 128);
    const path_z = try createTestBacking(&tmp, granularity * 2);
    defer std.testing.allocator.free(path_z);

    var store = try BackingStore.open(path_z);
    defer store.close();
    var manager = try Manager.init(std.testing.allocator, &store, granularity);
    defer manager.deinit();
    const tensor = TensorHandle{ .id = 1 };
    try manager.register(tensor, 0, granularity * 2);

    var outer = try manager.acquireRange(tensor, 0, granularity);
    var inner = try manager.acquireRange(tensor, 32, 64);
    try std.testing.expectEqualSlices(u8, outer.bytes()[32..96], inner.bytes());
    // A disjoint second window would exceed the budget, and the only resident
    // window is pinned, so eviction is impossible.
    try std.testing.expectError(Error.BudgetExceeded, manager.acquireRange(tensor, granularity, granularity));
    inner.release();
    outer.release();

    var next = try manager.acquireRange(tensor, granularity, granularity);
    next.release();
    const metrics = manager.metrics();
    try std.testing.expectEqual(@as(u64, 2), metrics.faults);
    try std.testing.expectEqual(@as(u64, 1), metrics.hits);
}

test "multiple concurrent windows of one tensor coexist within budget" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const granularity = try mappingGranularity();
    const path_z = try createTestBacking(&tmp, granularity * 3);
    defer std.testing.allocator.free(path_z);

    var store = try BackingStore.open(path_z);
    defer store.close();
    // Budget fits two page-sized windows simultaneously.
    var manager = try Manager.init(std.testing.allocator, &store, granularity * 2);
    defer manager.deinit();
    const tensor = TensorHandle{ .id = 7 };
    try manager.register(tensor, 0, granularity * 3);

    // Two disjoint windows of the SAME tensor stay resident at once.
    var first = try manager.acquireRange(tensor, 0, granularity);
    var second = try manager.acquireRange(tensor, granularity, granularity);
    defer first.release();
    defer second.release();

    try std.testing.expectEqual(@as(usize, granularity * 2), manager.metrics().resident_bytes);
    try std.testing.expectEqual(@as(usize, 1), manager.metrics().resident_tensors);
    try std.testing.expectEqual(Residency.resident, try manager.state(tensor));

    // Each window reads its own logical bytes.
    try std.testing.expectEqual(@as(u8, 0), first.bytes()[0]);
    try std.testing.expectEqual(@as(u8, @truncate(granularity)), second.bytes()[0]);

    // A third disjoint window does not fit: both resident windows are pinned.
    try std.testing.expectError(Error.BudgetExceeded, manager.acquireRange(tensor, granularity * 2, granularity));

    // After releasing the first window its budget is free again, so a window
    // over the third page faults in without evicting the pinned second one.
    first.release();
    var third = try manager.acquireRange(tensor, granularity * 2, granularity);
    defer third.release();
    try std.testing.expectEqual(@as(u8, @truncate(granularity * 2)), third.bytes()[0]);
    try std.testing.expectEqual(@as(usize, granularity * 2), manager.metrics().resident_bytes);
}

test "unregister rejects while any window is pinned and succeeds after release" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const granularity = try mappingGranularity();
    const path_z = try createTestBacking(&tmp, granularity * 2);
    defer std.testing.allocator.free(path_z);

    var store = try BackingStore.open(path_z);
    defer store.close();
    var manager = try Manager.init(std.testing.allocator, &store, granularity * 2);
    defer manager.deinit();
    const tensor = TensorHandle{ .id = 9 };
    try manager.register(tensor, 0, granularity * 2);

    var view = try manager.acquireRange(tensor, granularity, granularity);
    try std.testing.expectError(Error.TensorBusy, manager.unregister(tensor));
    view.release();
    try manager.unregister(tensor);
    try std.testing.expectError(Error.UnknownTensor, manager.state(tensor));
}

test "synchronous and asynchronous prefetch become later acquire hits" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const granularity = try mappingGranularity();
    const path_z = try createTestBacking(&tmp, granularity * 2);
    defer std.testing.allocator.free(path_z);

    var store = try BackingStore.open(path_z);
    defer store.close();
    var manager = try Manager.init(std.testing.allocator, &store, granularity * 2);
    defer manager.deinit();
    const a = TensorHandle{ .id = 1 };
    const b = TensorHandle{ .id = 2 };
    try manager.register(a, 0, granularity);
    try manager.register(b, granularity, granularity);

    try manager.prefetch(a);
    var first = try manager.acquire(a);
    try std.testing.expectEqual(@as(u8, 0), first.bytes()[0]);
    first.release();

    var task = try manager.prefetchRangeAsync(std.testing.allocator, b, 0, granularity);
    try task.wait();
    var second = try manager.acquire(b);
    try std.testing.expectEqual(@as(u8, @truncate(granularity)), second.bytes()[0]);
    second.release();

    const metrics = manager.metrics();
    try std.testing.expectEqual(@as(u64, 2), metrics.prefetches);
    try std.testing.expectEqual(@as(u64, granularity * 2), metrics.prefetched_bytes);
    try std.testing.expectEqual(@as(u64, 2), metrics.faults);
    try std.testing.expectEqual(@as(u64, 2), metrics.hits);
    try std.testing.expect(metrics.peak_resident_bytes <= metrics.budget_bytes);
}

test "bounded prefetch scheduler applies backpressure and drains accepted work" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const granularity = try mappingGranularity();
    const path_z = try createTestBacking(&tmp, granularity * 2);
    defer std.testing.allocator.free(path_z);

    var store = try BackingStore.open(path_z);
    defer store.close();
    var manager = try Manager.init(std.testing.allocator, &store, granularity * 2);
    defer manager.deinit();
    const a = TensorHandle{ .id = 1 };
    const b = TensorHandle{ .id = 2 };
    try manager.register(a, 0, granularity);
    try manager.register(b, granularity, granularity);

    const scheduler = try PrefetchScheduler.init(std.testing.allocator, &manager, 1, 1);

    // Hold the manager lock until the only worker has taken its first task.
    // This makes saturation deterministic: one running task plus one queued.
    manager.mutex.lock();
    var first = try scheduler.submit(a, 0, granularity);
    while (scheduler.metrics().active_workers == 0) std.Thread.yield() catch {};
    var second = try scheduler.submit(b, 0, granularity);
    try std.testing.expectError(Error.PrefetchQueueFull, scheduler.submit(a, 0, granularity));
    manager.mutex.unlock();

    scheduler.shutdown();
    try std.testing.expectError(Error.SchedulerShuttingDown, scheduler.submit(a, 0, granularity));
    scheduler.deinit();
    try first.wait();
    try second.wait();

    var first_hit = try manager.acquire(a);
    first_hit.release();
    var second_hit = try manager.acquire(b);
    second_hit.release();
    const metrics = manager.metrics();
    try std.testing.expectEqual(@as(u64, 2), metrics.prefetches);
    try std.testing.expectEqual(@as(u64, 2), metrics.hits);
    try std.testing.expectEqual(@as(u64, granularity * 2), metrics.prefetched_bytes);
    try std.testing.expect(metrics.peak_resident_bytes <= metrics.budget_bytes);
}

test "prefetch scheduler reports task failures without abandoning shutdown" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const granularity = try mappingGranularity();
    const path_z = try createTestBacking(&tmp, granularity);
    defer std.testing.allocator.free(path_z);

    var store = try BackingStore.open(path_z);
    defer store.close();
    var manager = try Manager.init(std.testing.allocator, &store, granularity);
    defer manager.deinit();
    const scheduler = try PrefetchScheduler.init(std.testing.allocator, &manager, 1, 2);

    var failed = try scheduler.submit(.{ .id = 99 }, 0, granularity);
    scheduler.waitIdle();
    const before_deinit = scheduler.metrics();
    try std.testing.expectEqual(@as(u64, 1), before_deinit.submitted);
    try std.testing.expectEqual(@as(u64, 1), before_deinit.completed);
    try std.testing.expectEqual(@as(u64, 0), before_deinit.succeeded);
    try std.testing.expectEqual(@as(u64, 1), before_deinit.failed);
    scheduler.deinit();
    try std.testing.expectError(Error.UnknownTensor, failed.wait());
}

const ConcurrentAcquireContext = struct {
    manager: *Manager,
    handle: TensorHandle,
    expected: u8,
    failed: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    fn run(self: *ConcurrentAcquireContext) void {
        for (0..200) |_| {
            var view = self.manager.acquire(self.handle) catch {
                self.failed.store(true, .release);
                return;
            };
            if (view.bytes()[0] != self.expected) self.failed.store(true, .release);
            view.release();
        }
    }
};

/// Concurrent workers hold disjoint windows of the SAME tensor at the same
/// time. This is only possible with multi-window residency; the single-window
/// design rejected overlapping access with error.TensorBusy.
const ConcurrentWindowContext = struct {
    manager: *Manager,
    handle: TensorHandle,
    tensor_offset: usize,
    expected: u8,
    failed: std.atomic.Value(bool) = std.atomic.Value(bool).init(false),

    fn run(self: *ConcurrentWindowContext) void {
        for (0..200) |_| {
            var view = self.manager.acquireRange(self.handle, self.tensor_offset, 32) catch {
                self.failed.store(true, .release);
                return;
            };
            if (view.bytes()[0] != self.expected) self.failed.store(true, .release);
            view.release();
        }
    }
};

test "concurrent executors hold disjoint windows of the same tensor" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const granularity = try mappingGranularity();
    const path_z = try createTestBacking(&tmp, granularity * 2);
    defer std.testing.allocator.free(path_z);

    var store = try BackingStore.open(path_z);
    defer store.close();
    var manager = try Manager.init(std.testing.allocator, &store, granularity * 2);
    defer manager.deinit();
    const tensor = TensorHandle{ .id = 5 };
    try manager.register(tensor, 0, granularity * 2);

    var contexts = [_]ConcurrentWindowContext{
        .{ .manager = &manager, .handle = tensor, .tensor_offset = 0, .expected = 0 },
        .{ .manager = &manager, .handle = tensor, .tensor_offset = 32, .expected = 32 },
        .{ .manager = &manager, .handle = tensor, .tensor_offset = 64, .expected = 64 },
        .{ .manager = &manager, .handle = tensor, .tensor_offset = 96, .expected = 96 },
    };
    var threads: [contexts.len]std.Thread = undefined;
    for (&threads, &contexts) |*thread, *context| {
        thread.* = try std.Thread.spawn(.{}, ConcurrentWindowContext.run, .{context});
    }
    for (&threads) |*thread| thread.join();
    for (&contexts) |*context| try std.testing.expect(!context.failed.load(.acquire));

    const metrics = manager.metrics();
    try std.testing.expectEqual(@as(u64, 0), metrics.evictions);
    try std.testing.expect(metrics.peak_resident_bytes <= metrics.budget_bytes);
}

test "manager serializes concurrent faults hits and releases" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const granularity = try mappingGranularity();
    const path_z = try createTestBacking(&tmp, granularity * 2);
    defer std.testing.allocator.free(path_z);

    var store = try BackingStore.open(path_z);
    defer store.close();
    var manager = try Manager.init(std.testing.allocator, &store, granularity * 2);
    defer manager.deinit();
    const a = TensorHandle{ .id = 1 };
    const b = TensorHandle{ .id = 2 };
    try manager.register(a, 0, granularity);
    try manager.register(b, granularity, granularity);

    var contexts = [_]ConcurrentAcquireContext{
        .{ .manager = &manager, .handle = a, .expected = 0 },
        .{ .manager = &manager, .handle = a, .expected = 0 },
        .{ .manager = &manager, .handle = b, .expected = @truncate(granularity) },
        .{ .manager = &manager, .handle = b, .expected = @truncate(granularity) },
    };
    var threads: [contexts.len]std.Thread = undefined;
    for (&threads, &contexts) |*thread, *context| {
        thread.* = try std.Thread.spawn(.{}, ConcurrentAcquireContext.run, .{context});
    }
    for (&threads) |*thread| thread.join();
    for (&contexts) |*context| try std.testing.expect(!context.failed.load(.acquire));

    const metrics = manager.metrics();
    try std.testing.expectEqual(@as(u64, 2), metrics.faults);
    try std.testing.expectEqual(@as(u64, 798), metrics.hits);
    try std.testing.expect(metrics.peak_resident_bytes <= metrics.budget_bytes);
}

test "rejects invalid ranges and tensors larger than the budget" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path_z = try createTestBacking(&tmp, 64);
    defer std.testing.allocator.free(path_z);

    var store = try BackingStore.open(path_z);
    defer store.close();
    var manager = try Manager.init(std.testing.allocator, &store, 32);
    defer manager.deinit();
    try std.testing.expectError(Error.InvalidRange, manager.register(.{ .id = 1 }, 60, 8));
    try manager.register(.{ .id = 2 }, 0, 64);
    try std.testing.expectError(Error.BudgetExceeded, manager.acquire(.{ .id = 2 }));
    try std.testing.expectEqual(@as(usize, 0), manager.metrics().resident_bytes);
}
