//! Continuous-batching scheduler (Phase 1).
//!
//! A single owner thread drives the llama context. Each decode step packs the
//! continuing-decode token of every active slot plus prefill chunks of newly
//! admitted requests into ONE `llama_decode` call, then samples each slot from
//! its own logits index. This is what lets concurrent requests share decode
//! steps instead of serializing one-at-a-time.
//!
//! Slot i owns sequence id i in the KV cache (independent per-seq KV). A request
//! carries its own sampler, so per-slot sampling state stays independent even
//! when sequences are interleaved in a batch.
//!
//! Gated: the engine only builds a Scheduler when `max_concurrent > 1`. The
//! single-stream path (prefix cache, speculative decoding) is unchanged at N=1.
//!
//! ponytail: the owner thread also calls each request's sink (socket write).
//! A slow client therefore stalls the decode loop (head-of-line blocking).
//! Upgrade path if it matters: per-slot output queue drained by writer threads.

const std = @import("std");
const llama = @import("llama_cpp.zig");
const inference = @import("inference.zig");
const prefix = @import("prefix_cache.zig");

/// One unit of work. Allocated by the submitter (handler thread) and kept alive
/// until `wait()` returns — the submitter blocks for the whole generation, so a
/// stack-allocated Request is safe. The scheduler thread fills the output fields
/// and signals `done`.
pub const Request = struct {
    // --- input (owned by submitter, read-only to scheduler) ---
    prompt_tokens: []const llama.Token,
    sampler: llama.Sampler,
    max_tokens: usize,
    sink: ?inference.TokenSink,
    allocator: std.mem.Allocator,

    // --- output (guarded by mutex) ---
    mutex: std.Thread.Mutex = .{},
    cond: std.Thread.Condition = .{},
    done: bool = false,
    text: std.ArrayList(u8) = .empty,
    completion_tokens: usize = 0,
    finish: inference.GenerationResult.FinishReason = .stop,

    /// Block until the scheduler finishes this request.
    pub fn wait(self: *Request) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        while (!self.done) self.cond.wait(&self.mutex);
    }

    fn complete(self: *Request, reason: inference.GenerationResult.FinishReason, n: usize) void {
        self.mutex.lock();
        defer self.mutex.unlock();
        self.completion_tokens = n;
        self.finish = reason;
        self.done = true;
        self.cond.signal();
    }
};

const SlotState = enum { idle, prefill, decode };

const Slot = struct {
    seq_id: i32,
    req: ?*Request = null,
    state: SlotState = .idle,
    /// Next prompt index to feed during prefill.
    prefill_pos: usize = 0,
    /// Tokens committed to this seq's KV cache (== next decode position).
    n_past: usize = 0,
    /// Generated (emitted) tokens so far.
    n_decoded: usize = 0,
    /// Last sampled token, fed at the next decode step.
    last_token: llama.Token = 0,
};

pub const Scheduler = struct {
    allocator: std.mem.Allocator,
    ctx: llama.Context,
    vocab: ?*const llama.c.llama_vocab,
    batch: llama.Batch,
    n_batch: usize,
    n_ctx: usize,

    slots: []Slot,
    /// Per-slot batch index that carries logits this step, or -1. Reused each step.
    sample_idx: []i32,

    queue: std.ArrayList(*Request),
    mutex: std.Thread.Mutex = .{},
    cond: std.Thread.Condition = .{},
    thread: ?std.Thread = null,
    running: std.atomic.Value(bool),
    max_queue: usize,

    // --- Phase 2 prefix sharing ---
    // Cross-slot prefix cache (RadixAttention-lite): a pool of dedicated cache
    // sequences holds prompt prefixes; a request on any slot reuses the longest
    // cached prefix via full-sequence seq_cp. Opt-in via prefix_cache. null pool
    // => no reuse (each request decodes from a clean KV).
    prefix_cache: bool,
    prefix_pool: ?prefix.PrefixCache,
    n_requests: u64 = 0,
    reused_tokens: u64 = 0,
    prefilled_tokens: u64 = 0,

    /// `n_slots` serving slots (seq ids 0..n_slots). `n_cache` extra cache
    /// sequences (ids n_slots..n_slots+n_cache) back the cross-slot prefix cache;
    /// pass 0 to disable. The llama context must have n_seq_max >= n_slots+n_cache.
    pub fn init(
        allocator: std.mem.Allocator,
        ctx: llama.Context,
        vocab: ?*const llama.c.llama_vocab,
        n_slots: usize,
        n_batch: usize,
        prefix_cache: bool,
        n_cache: usize,
    ) !*Scheduler {
        const self = try allocator.create(Scheduler);
        errdefer allocator.destroy(self);

        const slots = try allocator.alloc(Slot, n_slots);
        errdefer allocator.free(slots);
        for (slots, 0..) |*s, i| s.* = .{ .seq_id = @intCast(i) };

        const sample_idx = try allocator.alloc(i32, n_slots);
        errdefer allocator.free(sample_idx);

        var pool: ?prefix.PrefixCache = null;
        if (prefix_cache and n_cache > 0) {
            pool = try prefix.PrefixCache.init(allocator, ctx, n_cache, @intCast(n_slots));
        }
        errdefer if (pool) |*p| p.deinit();

        self.* = .{
            .allocator = allocator,
            .ctx = ctx,
            .vocab = vocab,
            .batch = llama.Batch.init(@intCast(n_batch), 0, @intCast(n_slots + n_cache)),
            .n_batch = n_batch,
            .n_ctx = @intCast(ctx.nCtx()),
            .slots = slots,
            .sample_idx = sample_idx,
            .queue = .empty,
            .running = std.atomic.Value(bool).init(true),
            .max_queue = n_slots * 4,
            .prefix_cache = prefix_cache,
            .prefix_pool = pool,
        };
        self.thread = try std.Thread.spawn(.{}, runLoop, .{self});
        return self;
    }

    pub fn deinit(self: *Scheduler) void {
        self.running.store(false, .release);
        self.mutex.lock();
        self.cond.broadcast();
        self.mutex.unlock();
        if (self.thread) |t| t.join();

        if (self.n_requests > 0) {
            const total = self.reused_tokens + self.prefilled_tokens;
            const pct: u64 = if (total > 0) self.reused_tokens * 100 / total else 0;
            const hits: u64 = if (self.prefix_pool) |p| p.hits else 0;
            std.log.info(
                "scheduler: {d} requests, {d} prefix-cache hits, reused {d}/{d} prompt tokens ({d}%)",
                .{ self.n_requests, hits, self.reused_tokens, total, pct },
            );
        }

        if (self.prefix_pool) |*p| p.deinit();
        self.batch.deinit();
        self.queue.deinit(self.allocator);
        self.allocator.free(self.sample_idx);
        self.allocator.free(self.slots);
        self.allocator.destroy(self);
    }

    /// Enqueue a request. Returns error.QueueFull when backpressured.
    pub fn submit(self: *Scheduler, req: *Request) !void {
        self.mutex.lock();
        defer self.mutex.unlock();
        if (self.queue.items.len >= self.max_queue) return error.QueueFull;
        try self.queue.append(self.allocator, req);
        self.cond.signal();
    }

    fn runLoop(self: *Scheduler) void {
        while (self.running.load(.acquire)) {
            // --- admit queued requests into idle slots, count active ---
            self.mutex.lock();
            while (self.queue.items.len > 0) {
                const slot_idx = self.pickSlot() orelse break;
                const req = self.queue.orderedRemove(0);
                self.admit(&self.slots[slot_idx], req);
            }
            var active: usize = 0;
            for (self.slots) |slot| {
                if (slot.state != .idle) active += 1;
            }
            if (active == 0) {
                while (self.running.load(.acquire) and self.queue.items.len == 0) {
                    self.cond.wait(&self.mutex);
                }
                self.mutex.unlock();
                continue;
            }
            self.mutex.unlock();

            // From here slots are only touched by this thread; submit() only
            // appends to the queue, which we re-lock to drain next iteration.
            self.step() catch |err| {
                // A decode failure is unrecoverable for the in-flight slots:
                // finish them all as aborted so submitters wake up.
                for (self.slots) |*slot| {
                    if (slot.req) |req| {
                        req.complete(.aborted, slot.n_decoded);
                        _ = self.ctx.kvCacheSeqRm(slot.seq_id, -1, -1);
                        slot.req = null;
                        slot.state = .idle;
                    }
                }
                std.log.err("scheduler decode step failed: {s}", .{@errorName(err)});
            };
        }
    }

    /// First idle slot, or null when all are busy. (Prefix reuse is cross-slot
    /// via the shared cache pool, so slot choice doesn't affect reuse.)
    fn pickSlot(self: *Scheduler) ?usize {
        for (self.slots, 0..) |*slot, i| {
            if (slot.state == .idle) return i;
        }
        return null;
    }

    /// Assign a request to a slot. With the prefix cache, reuse the longest
    /// cached prefix (full-copied into the slot); otherwise clear the slot's KV.
    /// Either way the slot then prefills tokens [start, prompt.len).
    fn admit(self: *Scheduler, slot: *Slot, req: *Request) void {
        self.n_requests += 1;
        var start: usize = 0;
        if (self.prefix_pool) |*pool| {
            start = pool.acquire(slot.seq_id, req.prompt_tokens);
        }
        if (start == 0) {
            // No reuse: clear the slot's sequence for a clean decode.
            _ = self.ctx.kvCacheSeqRm(slot.seq_id, -1, -1);
        }
        self.reused_tokens += start;
        self.prefilled_tokens += req.prompt_tokens.len - start;

        slot.req = req;
        slot.state = .prefill;
        slot.prefill_pos = start;
        slot.n_past = start;
        slot.n_decoded = 0;
    }

    fn step(self: *Scheduler) !void {
        self.batch.clear();
        for (self.sample_idx) |*s| s.* = -1;

        var budget: usize = self.n_batch;

        // Pass 1: one continuing token per decoding slot.
        for (self.slots, 0..) |*slot, i| {
            if (slot.state != .decode) continue;
            if (budget == 0) break;
            try self.batch.add(slot.last_token, @intCast(slot.n_past), &[_]i32{slot.seq_id}, true);
            self.sample_idx[i] = self.batch.handle.n_tokens - 1;
            budget -= 1;
        }

        // Pass 2: fill remaining budget with prefill chunks.
        for (self.slots, 0..) |*slot, i| {
            if (slot.state != .prefill) continue;
            if (budget == 0) break;
            const remaining = slot.req.?.prompt_tokens.len - slot.prefill_pos;
            const chunk = @min(remaining, budget);
            const toks = slot.req.?.prompt_tokens;
            for (0..chunk) |j| {
                const pos = slot.prefill_pos + j;
                const is_last = (pos == toks.len - 1);
                try self.batch.add(toks[pos], @intCast(pos), &[_]i32{slot.seq_id}, is_last);
            }
            slot.prefill_pos += chunk;
            slot.n_past = slot.prefill_pos;
            if (slot.prefill_pos == toks.len) {
                // Last prompt token carried logits → sample first gen token.
                self.sample_idx[i] = self.batch.handle.n_tokens - 1;
            }
            budget -= chunk;
        }

        try self.ctx.decode(self.batch.handle);

        // Sample + emit for every slot that produced logits this step.
        for (self.slots, 0..) |*slot, i| {
            if (self.sample_idx[i] < 0) continue;
            const req = slot.req.?;
            const tok = req.sampler.sampleAt(self.ctx, self.sample_idx[i]);

            if (slot.state == .decode) {
                // The token fed this step is now committed to KV.
                slot.n_past += 1;
            } else {
                // Prefill just finished; the slot's KV is exactly the prompt
                // ([0, prompt.len), no generated token committed yet). Cache it
                // for cross-slot reuse before generation appends to the slot.
                slot.state = .decode;
                if (self.prefix_pool) |*pool| pool.store(slot.seq_id, req.prompt_tokens);
            }

            if (llama.c.llama_vocab_is_eog(self.vocab, tok)) {
                self.finishSlot(slot, .stop);
                continue;
            }
            if (slot.n_decoded >= req.max_tokens) {
                self.finishSlot(slot, .length);
                continue;
            }
            if (slot.n_past >= self.n_ctx) {
                self.finishSlot(slot, .context_limit);
                continue;
            }

            self.emitPiece(req, tok) catch {
                // Sink write failed (client gone) — abort this slot.
                self.finishSlot(slot, .aborted);
                continue;
            };
            slot.n_decoded += 1;
            slot.last_token = tok;
        }
    }

    fn finishSlot(self: *Scheduler, slot: *Slot, reason: inference.GenerationResult.FinishReason) void {
        slot.req.?.complete(reason, slot.n_decoded);
        // Free the slot's KV. The reusable prefix lives in the cache pool (a
        // separate set of sequences populated at the prefill->decode transition),
        // so nothing is lost by clearing the slot here.
        _ = self.ctx.kvCacheSeqRm(slot.seq_id, -1, -1);
        slot.req = null;
        slot.state = .idle;
    }

    fn emitPiece(self: *Scheduler, req: *Request, tok: llama.Token) !void {
        var buf: [256]u8 = undefined;
        const n = llama.c.llama_token_to_piece(self.vocab, tok, &buf, buf.len, 0, false);
        if (n > 0) {
            const piece = buf[0..@intCast(n)];
            if (req.sink) |s| try s.write(piece);
            try req.text.appendSlice(req.allocator, piece);
        } else if (n < 0) {
            const need: usize = @intCast(-n);
            const large = try req.allocator.alloc(u8, need);
            defer req.allocator.free(large);
            const n2 = llama.c.llama_token_to_piece(self.vocab, tok, large.ptr, @intCast(large.len), 0, false);
            if (n2 > 0) {
                const piece = large[0..@intCast(n2)];
                if (req.sink) |s| try s.write(piece);
                try req.text.appendSlice(req.allocator, piece);
            }
        }
    }
};
