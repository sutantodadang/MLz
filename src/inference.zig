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

    pub const FinishReason = enum { stop, length, context_limit, aborted };

    pub fn deinit(self: GenerationResult, allocator: std.mem.Allocator) void {
        allocator.free(self.text);
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
    }

    if (n_gen >= opts.max_tokens) {
        finish = .length;
    }

    const total_ns = gen_timer.read();
    const text = try allocator.dupe(u8, assistant_buf.items);

    return .{
        .text = text,
        .prompt_tokens = prompt_tokens.len,
        .completion_tokens = completion_tokens,
        .ttft_ns = ttft_ns,
        .total_ns = total_ns,
        .finish_reason = finish,
    };
}
