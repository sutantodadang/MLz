//! Cross-slot prefix cache (RadixAttention-lite).
//!
//! A small pool of dedicated cache sequences, each holding one cached prompt
//! prefix. A new request on ANY serving slot can reuse a prefix that a request
//! on a DIFFERENT slot prefilled — the cross-slot win per-slot reuse can't give.
//!
//! Built on the only KV primitive this llama build allows for cross-sequence
//! sharing: FULL-sequence `seq_cp(src, dst, 0, -1)` (sub-range copy aborts with
//! GGML_ASSERT — verified empirically). To reuse a prefix of length L:
//!   1. truncate the cache seq to exactly [0, L) with end-aligned seq_rm,
//!   2. full-copy that seq into the slot,
//!   3. the caller prefills the suffix.
//! To store a prefix: full-copy the slot's prompt KV into a free/LRU cache seq.
//! seq_cp shares KV cells rather than duplicating them, so this is cheap.
//!
//! Owner-thread-only: the scheduler calls acquire()/store() from its decode
//! thread, so no locking.

const std = @import("std");
const llama = @import("llama_cpp.zig");
const Token = llama.Token;

pub const PrefixCache = struct {
    const Entry = struct {
        seq_id: i32,
        /// Tokens this cache sequence holds in KV at positions [0, len).
        tokens: std.ArrayList(Token) = .empty,
        used: bool = false,
        lru: u64 = 0,
    };

    allocator: std.mem.Allocator,
    ctx: llama.Context,
    entries: []Entry,
    clock: u64 = 0,

    // metrics
    hits: u64 = 0,
    reused_tokens: u64 = 0,

    /// `n_cache` cache sequences with ids [base_seq, base_seq + n_cache).
    pub fn init(allocator: std.mem.Allocator, ctx: llama.Context, n_cache: usize, base_seq: i32) !PrefixCache {
        const entries = try allocator.alloc(Entry, n_cache);
        for (entries, 0..) |*e, i| e.* = .{ .seq_id = base_seq + @as(i32, @intCast(i)) };
        return .{ .allocator = allocator, .ctx = ctx, .entries = entries };
    }

    pub fn deinit(self: *PrefixCache) void {
        for (self.entries) |*e| e.tokens.deinit(self.allocator);
        self.allocator.free(self.entries);
    }

    fn bump(self: *PrefixCache) u64 {
        self.clock += 1;
        return self.clock;
    }

    /// Reuse the longest cached prefix of `prompt` into `slot_seq`. On a hit the
    /// slot is cleared and the matched prefix KV is copied in; returns the number
    /// of reused tokens. Returns 0 (slot left untouched) when nothing matches —
    /// the caller must then clear the slot itself.
    pub fn acquire(self: *PrefixCache, slot_seq: i32, prompt: []const Token) usize {
        var best: ?usize = null;
        var best_m: usize = 0;
        for (self.entries, 0..) |*e, i| {
            if (!e.used) continue;
            const m = lcp(prompt, e.tokens.items);
            if (m > best_m) {
                best_m = m;
                best = i;
            }
        }
        const bi = best orelse return 0;
        // Leave at least one token to prefill so the step produces logits.
        const target = if (best_m >= prompt.len) prompt.len - 1 else best_m;
        if (target == 0) return 0;

        const e = &self.entries[bi];
        if (e.tokens.items.len != target) {
            // Truncate the cache seq to exactly [0, target) (end-aligned remove).
            if (!self.ctx.kvCacheSeqRm(e.seq_id, @intCast(target), -1)) return 0;
            e.tokens.shrinkRetainingCapacity(target);
        }
        _ = self.ctx.kvCacheSeqRm(slot_seq, -1, -1);
        self.ctx.kvCacheSeqCp(e.seq_id, slot_seq, 0, -1);
        e.lru = self.bump();
        self.hits += 1;
        self.reused_tokens += target;
        return target;
    }

    /// Cache `prompt`, whose prefilled KV currently occupies `slot_seq` at
    /// [0, prompt.len). Evicts the LRU entry when the pool is full.
    pub fn store(self: *PrefixCache, slot_seq: i32, prompt: []const Token) void {
        if (prompt.len == 0) return;
        // Already cached identically? just touch LRU.
        for (self.entries) |*e| {
            if (e.used and e.tokens.items.len == prompt.len and lcp(prompt, e.tokens.items) == prompt.len) {
                e.lru = self.bump();
                return;
            }
        }
        const idx = self.freeOrLru();
        const e = &self.entries[idx];
        _ = self.ctx.kvCacheSeqRm(e.seq_id, -1, -1);
        self.ctx.kvCacheSeqCp(slot_seq, e.seq_id, 0, -1);
        e.tokens.clearRetainingCapacity();
        e.tokens.appendSlice(self.allocator, prompt) catch {
            _ = self.ctx.kvCacheSeqRm(e.seq_id, -1, -1);
            e.used = false;
            return;
        };
        e.used = true;
        e.lru = self.bump();
    }

    fn freeOrLru(self: *PrefixCache) usize {
        var lru_idx: usize = 0;
        var lru_val: u64 = std.math.maxInt(u64);
        for (self.entries, 0..) |e, i| {
            if (!e.used) return i;
            if (e.lru < lru_val) {
                lru_val = e.lru;
                lru_idx = i;
            }
        }
        return lru_idx;
    }
};

/// Length of the longest common prefix of two token slices.
fn lcp(a: []const Token, b: []const Token) usize {
    const n = @min(a.len, b.len);
    var i: usize = 0;
    while (i < n and a[i] == b[i]) : (i += 1) {}
    return i;
}

test "lcp" {
    const T = Token;
    try std.testing.expectEqual(@as(usize, 3), lcp(&[_]T{ 1, 2, 3, 4 }, &[_]T{ 1, 2, 3, 9 }));
    try std.testing.expectEqual(@as(usize, 0), lcp(&[_]T{ 5, 2 }, &[_]T{ 1, 2 }));
    try std.testing.expectEqual(@as(usize, 2), lcp(&[_]T{ 1, 2 }, &[_]T{ 1, 2, 3 }));
}
