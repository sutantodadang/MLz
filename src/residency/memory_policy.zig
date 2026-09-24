//! Non-weight memory policy for the official GGML residency path.
//!
//! Mapped immutable weights are bounded exactly by the residency manager's
//! weight budget. Everything else a context allocates is planned here from
//! llama.cpp's own allocation sizes (`llama_model_params.no_alloc` simulates
//! the context without allocating), so a configuration is rejected before any
//! KV, recurrent, or workspace buffer exists.
//!
//! Not covered by any hard limit: filesystem page cache behind the weight
//! mappings, allocator slack, thread stacks, and the sampler/tokenizer heap.

const std = @import("std");

pub const Estimate = struct {
    /// KV cache plus recurrent/DeltaNet state for all sequences.
    state_bytes: usize,
    /// Graph workspace reserved by the llama.cpp scheduler.
    compute_bytes: usize,
    /// Host logits buffer for the largest decode the engine issues.
    output_bytes: usize,

    pub fn total(self: Estimate) error{Overflow}!usize {
        const partial = try std.math.add(usize, self.state_bytes, self.compute_bytes);
        return std.math.add(usize, partial, self.output_bytes);
    }
};

pub const Verdict = enum {
    within,
    /// Fits, but above the soft limit (90% of the hard limit).
    warning,
    rejected,
};

pub fn evaluate(estimate: Estimate, hard_limit: usize) Verdict {
    const needed = estimate.total() catch return .rejected;
    if (needed > hard_limit) return .rejected;
    if (needed > hard_limit - hard_limit / 10) return .warning;
    return .within;
}

/// Bytes of the f32 logits rows llama.cpp keeps for `n_outputs` outputs.
pub fn outputBytes(n_vocab: usize, n_outputs: usize) error{Overflow}!usize {
    return std.math.mul(usize, n_vocab, try std.math.mul(usize, n_outputs, @sizeOf(f32)));
}

test "exact limit is admitted and one byte under is rejected" {
    const estimate = Estimate{ .state_bytes = 1000, .compute_bytes = 200, .output_bytes = 24 };
    try std.testing.expectEqual(@as(usize, 1224), try estimate.total());
    try std.testing.expect(evaluate(estimate, 1224) != .rejected);
    try std.testing.expectEqual(Verdict.rejected, evaluate(estimate, 1223));
    try std.testing.expectEqual(Verdict.rejected, evaluate(estimate, 0));
}

test "soft limit warns above ninety percent" {
    const estimate = Estimate{ .state_bytes = 900, .compute_bytes = 0, .output_bytes = 0 };
    try std.testing.expectEqual(Verdict.within, evaluate(estimate, 1000));
    try std.testing.expectEqual(Verdict.warning, evaluate(.{ .state_bytes = 901, .compute_bytes = 0, .output_bytes = 0 }, 1000));
}

test "overflowing estimates are rejected" {
    const max = std.math.maxInt(usize);
    try std.testing.expectEqual(Verdict.rejected, evaluate(.{ .state_bytes = max, .compute_bytes = 1, .output_bytes = 0 }, max));
    try std.testing.expectError(error.Overflow, outputBytes(max, 2));
    try std.testing.expectEqual(@as(usize, 151936 * 4 * 4), try outputBytes(151936, 4));
}
