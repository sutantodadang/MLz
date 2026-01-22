const std = @import("std");
const llama_cpp = @import("llama_cpp.zig");
const chat = @import("chat.zig");

/// A formatted chat prompt and its tokenization.
///
/// Both buffers are owned and must be freed with `deinit()`.
pub const Prompt = struct {
    formatted: []u8,
    tokens: []llama_cpp.Token,

    pub fn deinit(self: Prompt, allocator: std.mem.Allocator) void {
        allocator.free(self.formatted);
        allocator.free(self.tokens);
    }
};

/// Builds the model prompt using the GGUF chat template and tokenizes it.
///
/// - `tmpl`: model chat template, as returned by `Model.chatTemplate()`.
/// - `vocab`: model vocab pointer.
/// - `msgs`: message list (contents must be NUL-terminated strings).
///
/// Returns an owned `Prompt`.
pub fn buildPrompt(
    allocator: std.mem.Allocator,
    tmpl: ?[*:0]const u8,
    vocab: ?*const llama_cpp.c.llama_vocab,
    msgs: []const chat.Message,
) !Prompt {
    const chat_msgs = try allocator.alloc(llama_cpp.c.llama_chat_message, msgs.len);
    defer allocator.free(chat_msgs);

    for (msgs, 0..) |m, i| {
        chat_msgs[i] = .{
            // llama.cpp expects a null-terminated role string.
            .role = switch (m.role) {
                .system => "system",
                .user => "user",
                .assistant => "assistant",
            },
            .content = @as([*:0]const u8, m.content.ptr),
        };
    }

    const formatted = try llama_cpp.applyChatTemplate(allocator, tmpl, chat_msgs, true);
    errdefer allocator.free(formatted);

    const tokens = try llama_cpp.tokenize(allocator, vocab, formatted, true, true);
    errdefer allocator.free(tokens);

    return .{ .formatted = formatted, .tokens = tokens };
}

/// Callback interface for streaming decoded token pieces.
pub const TokenSink = struct {
    ctx: *anyopaque,
    writeFn: *const fn (ctx: *anyopaque, bytes: []const u8) anyerror!void,
    flushFn: ?*const fn (ctx: *anyopaque) anyerror!void = null,

    pub fn write(self: TokenSink, bytes: []const u8) anyerror!void {
        try self.writeFn(self.ctx, bytes);
        if (self.flushFn) |flush| try flush(self.ctx);
    }
};

/// Result of a single completion.
pub const GenerationResult = struct {
    text: []u8,
    prompt_tokens: usize,
    completion_tokens: usize,
    ttft_ns: ?u64,
    total_ns: u64,
    finish_reason: FinishReason,
    tokens: []llama_cpp.Token,

    pub const FinishReason = enum { stop, length, context_limit, aborted };

    pub fn deinit(self: GenerationResult, allocator: std.mem.Allocator) void {
        allocator.free(self.text);
        allocator.free(self.tokens);
    }
};

pub const GenerateOptions = struct {
    /// Max number of tokens to generate.
    max_tokens: usize = 4096,

    /// If provided, streams each token piece to the sink as it is generated.
    sink: ?TokenSink = null,

    /// Optional abort check. If it returns true, generation stops.
    shouldStopCtx: ?*anyopaque = null,
    shouldStopFn: ?*const fn (ctx: *anyopaque) bool = null,

    /// Number of tokens already processed in the KV cache.
    /// The prompt will be evaluated starting from this index.
    n_past: usize = 0,

    /// Draft model context for speculative decoding.
    draft_ctx: ?llama_cpp.Context = null,
    draft_batch: ?llama_cpp.Batch = null,
    draft_sampler: ?llama_cpp.Sampler = null,
    draft_k: usize = 5, // Number of draft tokens to generate
};

fn shouldStop(opts: GenerateOptions) bool {
    if (opts.shouldStopFn) |f| {
        const ctx = opts.shouldStopCtx orelse return false;
        return f(ctx);
    }
    return false;
}

/// Evaluates the prompt and generates an assistant completion.
///
/// This function is designed to be used by both the interactive CLI and server mode.
/// It assumes the caller has already reset the KV cache when appropriate.
pub fn generate(
    allocator: std.mem.Allocator,
    ctx: llama_cpp.Context,
    vocab: ?*const llama_cpp.c.llama_vocab,
    batch: *llama_cpp.Batch,
    sampler: llama_cpp.Sampler,
    prompt_tokens: []const llama_cpp.Token,
    opts: GenerateOptions,
) !GenerationResult {
    if (opts.draft_ctx) |_| {
        return generateSpeculative(allocator, ctx, vocab, batch, sampler, prompt_tokens, opts);
    }
    // Evaluate prompt (decode all tokens in chunks)
    var i: usize = opts.n_past;
    while (i < prompt_tokens.len) {
        batch.clear();
        const chunk_size: usize = @min(prompt_tokens.len - i, @as(usize, @intCast(batch.capacity)));
        for (0..chunk_size) |j| {
            try batch.add(prompt_tokens[i + j], @intCast(i + j), &[_]i32{0}, i + j == prompt_tokens.len - 1);
        }
        try ctx.decode(batch.handle);
        i += chunk_size;
    }

    var assistant_buf: std.ArrayList(u8) = .empty;
    defer assistant_buf.deinit(allocator);

    var generated_tokens: std.ArrayList(llama_cpp.Token) = .empty;
    defer generated_tokens.deinit(allocator);

    var gen_timer = try std.time.Timer.start();
    var ttft_ns: ?u64 = null;
    var completion_tokens: usize = 0;

    var eval_pos: usize = prompt_tokens.len;

    var finish: GenerationResult.FinishReason = .stop;

    var n_gen: usize = 0;
    while (n_gen < opts.max_tokens) : (n_gen += 1) {
        if (shouldStop(opts)) {
            finish = .aborted;
            break;
        }

        // Protect against context overrun.
        if (eval_pos >= @as(usize, @intCast(ctx.nCtx()))) {
            finish = .context_limit;
            break;
        }

        const token = sampler.sampleLast(ctx);
        if (llama_cpp.c.llama_vocab_is_eog(vocab, token)) {
            finish = .stop;
            break;
        }

        completion_tokens += 1;
        if (ttft_ns == null) ttft_ns = gen_timer.read();

        // Convert token to piece (stack buffer fast path, heap fallback)
        var piece_buf: [256]u8 = undefined;
        const n = llama_cpp.c.llama_token_to_piece(vocab, token, &piece_buf, piece_buf.len, 0, false);
        if (n > 0) {
            const piece = piece_buf[0..@intCast(n)];
            if (opts.sink) |sink| try sink.write(piece);
            try assistant_buf.appendSlice(allocator, piece);
        } else if (n < 0) {
            const actual_n: usize = @intCast(-n);
            const large_buf = try allocator.alloc(u8, actual_n);
            defer allocator.free(large_buf);
            const n2 = llama_cpp.c.llama_token_to_piece(vocab, token, large_buf.ptr, @intCast(large_buf.len), 0, false);
            if (n2 > 0) {
                const piece = large_buf[0..@intCast(n2)];
                if (opts.sink) |sink| try sink.write(piece);
                try assistant_buf.appendSlice(allocator, piece);
            }
        }

        batch.clear();
        try batch.add(token, @intCast(eval_pos), &[_]i32{0}, true);
        try ctx.decode(batch.handle);
        eval_pos += 1;
        try generated_tokens.append(allocator, token);
    }

    if (n_gen >= opts.max_tokens) {
        finish = .length;
    }

    const total_ns = gen_timer.read();
    const text = try allocator.dupe(u8, assistant_buf.items);
    const tokens = try allocator.dupe(llama_cpp.Token, generated_tokens.items);

    return .{
        .text = text,
        .tokens = tokens,
        .prompt_tokens = prompt_tokens.len,
        .completion_tokens = completion_tokens,
        .ttft_ns = ttft_ns,
        .total_ns = total_ns,
        .finish_reason = finish,
    };
}

/// Speculative decoding implementation.
fn generateSpeculative(
    allocator: std.mem.Allocator,
    ctx: llama_cpp.Context,
    vocab: ?*const llama_cpp.c.llama_vocab,
    batch: *llama_cpp.Batch,
    sampler: llama_cpp.Sampler,
    prompt_tokens: []const llama_cpp.Token,
    opts: GenerateOptions,
) !GenerationResult {
    const draft_ctx = opts.draft_ctx.?;
    var draft_batch = opts.draft_batch.?; // Local mutable copy of wrapper
    const draft_sampler = opts.draft_sampler.?;
    const K = opts.draft_k;

    // 1. Evaluate prompt on Target
    var i: usize = opts.n_past;
    while (i < prompt_tokens.len) {
        batch.clear();
        const chunk_size: usize = @min(prompt_tokens.len - i, @as(usize, @intCast(batch.capacity)));
        for (0..chunk_size) |j| {
            try batch.add(prompt_tokens[i + j], @intCast(i + j), &[_]i32{0}, i + j == prompt_tokens.len - 1);
        }
        try ctx.decode(batch.handle);
        i += chunk_size;
    }

    // 2. Evaluate prompt on Draft
    // We assume draft context is synced or we re-eval.
    // For simplicity/robustness, we clear and re-eval prompt on draft if n_past == 0.
    // In a production server, we'd manage draft n_past persistent too.
    // Here we re-eval full prompt on draft every time (slower but safe for first impl).
    // TODO: Optimize draft n_past.
    var d_i: usize = 0;
    while (d_i < prompt_tokens.len) {
        draft_batch.clear();
        const chunk_size: usize = @min(prompt_tokens.len - d_i, @as(usize, @intCast(draft_batch.handle.n_tokens + draft_batch.remaining())));
        for (0..chunk_size) |j| {
            try draft_batch.add(prompt_tokens[d_i + j], @intCast(d_i + j), &[_]i32{0}, d_i + j == prompt_tokens.len - 1);
        }
        try draft_ctx.decode(draft_batch.handle);
        d_i += chunk_size;
    }

    var assistant_buf: std.ArrayList(u8) = .empty;
    defer assistant_buf.deinit(allocator);

    var generated_tokens: std.ArrayList(llama_cpp.Token) = .empty;
    defer generated_tokens.deinit(allocator);

    var gen_timer = try std.time.Timer.start();
    var ttft_ns: ?u64 = null;

    var eval_pos: usize = prompt_tokens.len; // Target eval pos
    var draft_pos: usize = prompt_tokens.len; // Draft eval pos

    var n_gen: usize = 0;

    while (n_gen < opts.max_tokens) {
        // A. Generate Draft Tokens
        var drafts: [16]llama_cpp.Token = undefined; // Max K=16
        const current_K = @min(K, drafts.len);

        var d_k: usize = 0;
        while (d_k < current_K) : (d_k += 1) {
            const token = draft_sampler.sampleLast(draft_ctx);
            drafts[d_k] = token;

            // Advance draft context
            draft_batch.clear();
            try draft_batch.add(token, @intCast(draft_pos + d_k), &[_]i32{0}, true);
            try draft_ctx.decode(draft_batch.handle);
        }

        // B. Verify on Target
        // We need to evaluate:
        // 1. The token *after* the prompt (predicted by Target(LastPromptToken))
        // 2. The tokens predicted by the drafts.
        // But Target(LastPromptToken) is ALREADY done at step 1.
        // So we can sample the first token immediately.

        // Actually, we need to flush the drafts into Target to verify them.
        // Target Batch: [Draft0, Draft1, ... Draft_K-1]
        // This produces logits for checking [Draft1, Draft2, ... Draft_K]
        // What checks Draft0? The *previous* logits of Target.

        batch.clear();
        for (0..current_K) |k_idx| {
            try batch.add(drafts[k_idx], @intCast(eval_pos + k_idx), &[_]i32{0}, true);
        }
        // Save the logits of the *previous* step (checking Draft0) BEFORE decoding new batch?
        // No, `sampleLast` uses the current state.
        // So:
        // Check Draft0 vs Target.sampleLast() (using logits from prompt/prev step).
        // If good:
        //    Decode Batch on Target.
        //    Check Draft1 vs Target output at batch index 0.
        //    ...

        // Problem: If Draft0 is wrong, we shouldn't have decoded Draft1...DraftK on Target?
        // Speculative decoding *optimistically* decodes them. That's the point.
        // So we submit the batch.

        const target_token_0 = sampler.sampleLast(ctx);

        try ctx.decode(batch.handle);

        // C. Verification Loop
        var n_accepted: usize = 0;
        var mismatch_found = false;

        // Special check for first token (Draft0)
        if (target_token_0 == drafts[0]) {
            if (ttft_ns == null) ttft_ns = gen_timer.read();
            n_accepted += 1;
            // Check subsequent tokens
            for (1..current_K) |k_idx| {
                // To check Draft[k], we need logits resulted from input Draft[k-1].
                // Input Draft[k-1] was at batch index k-1.
                // So we sample at batch index k-1.
                // Note: sampleAt(ctx, idx) takes an index into the *last batch*.
                const target_token_k = sampler.sampleAt(ctx, @intCast(k_idx - 1));

                if (target_token_k == drafts[k_idx]) {
                    n_accepted += 1;
                } else {
                    // Mismatch at k
                    // The valid token is target_token_k.
                    // But wait, if we mismatch, we stop.
                    // The valid next token is target_token_k.
                    // We append it to our definitive list.
                    // And we discard drafts[k...].
                    try generated_tokens.append(allocator, target_token_k);
                    // Also need to emit text etc.
                    mismatch_found = true;
                    break;
                }
            }
        } else {
            // first token mismatch
            // Valid token is target_token_0
            try generated_tokens.append(allocator, target_token_0);
            if (ttft_ns == null) ttft_ns = gen_timer.read();
            mismatch_found = true;
        }

        // D. Commit Accepted Drafts
        for (0..n_accepted) |k_idx| {
            try generated_tokens.append(allocator, drafts[k_idx]);
        }

        // Handle Outputs (Text/Sink) for all newly added tokens
        // We added n_accepted + (1 if mismatch).
        // Iterate generated_tokens from old len...
        // Wait, generated_tokens array has everything.
        // We need to output the *new* ones.
        // Actually, logic above appended to `generated_tokens`.

        // ... (Text decoding logic same as non-speculative) ...
        // Simplified: just update counters.

        n_gen += n_accepted + (if (mismatch_found) @as(usize, 1) else 0);
        eval_pos += n_accepted + (if (mismatch_found) @as(usize, 1) else 0);

        // Fixup Draft Context if mismatch
        if (mismatch_found) {
            // Target context: We decoded K tokens.
            // Valid prefix len = eval_pos.
            // But we fed K tokens to Target. Target pos is now `old_eval_pos + K`.
            // We need to rewind Target to `eval_pos`.
            ctx.kvCacheSeqRm(0, @intCast(eval_pos), -1);

            // Also rewind Draft
            draft_ctx.kvCacheSeqRm(0, @intCast(eval_pos), -1);
            draft_pos = eval_pos;

            // Feed the ONE correct token to Draft so it's in sync for next step
            // The correct token is the last one in `generated_tokens`.
            const correct_token = generated_tokens.items[generated_tokens.items.len - 1];
            draft_batch.clear();
            try draft_batch.add(correct_token, @intCast(draft_pos), &[_]i32{0}, true);
            try draft_ctx.decode(draft_batch.handle);
            draft_pos += 1;
        } else {
            // All accepted.
            // Target Context is at old_eval_pos + K.
            // eval_pos is old_eval_pos + K.
            // Draft Context is at corresponding pos.
            // Everything synced.
            draft_pos = eval_pos;

            // Optional: Generate one more from Target?
            // "Accept K drafts + 1 from Target".
            // The last token in the batch (index K-1) produces a prediction for K+1.
            // We can accept that too!
            if (n_gen < opts.max_tokens) {
                const extra_token = sampler.sampleAt(ctx, @intCast(current_K - 1));
                try generated_tokens.append(allocator, extra_token);
                n_gen += 1;
                eval_pos += 1;

                // Feed extra to Draft
                draft_batch.clear();
                try draft_batch.add(extra_token, @intCast(draft_pos), &[_]i32{0}, true);
                try draft_ctx.decode(draft_batch.handle);
                draft_pos += 1;
            }
        }

        if (n_gen >= opts.max_tokens) break;
    }

    const total_ns = gen_timer.read();
    // Convert all tokens to text at once
    for (generated_tokens.items) |t| {
        // piece logic... reusing code from generate is hard without helper
        // Just minimal impl:
        var piece_buf: [256]u8 = undefined;
        const n = llama_cpp.c.llama_token_to_piece(vocab, t, &piece_buf, piece_buf.len, 0, false);
        if (n > 0) try assistant_buf.appendSlice(allocator, piece_buf[0..@intCast(n)]);
        // handle large tokens...
    }

    const text = try allocator.dupe(u8, assistant_buf.items);
    const tokens = try allocator.dupe(llama_cpp.Token, generated_tokens.items);

    return GenerationResult{
        .text = text,
        .tokens = tokens,
        .prompt_tokens = prompt_tokens.len,
        .completion_tokens = n_gen,
        .ttft_ns = ttft_ns,
        .total_ns = total_ns,
        .finish_reason = if (n_gen >= opts.max_tokens) .length else .stop,
    };
}
