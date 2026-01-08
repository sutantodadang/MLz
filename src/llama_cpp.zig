const std = @import("std");

pub const c = @cImport({
    @cInclude("llama.h");
});

pub const Token = c.llama_token;

pub const LlamaError = error{
    OutOfMemory,
    BackendInitFailed,
    ModelLoadFailed,
    ContextInitFailed,
    VocabUnavailable,
    TemplateFailed,
    TokenizeFailed,
    DecodeFailed,
    LogitsUnavailable,
    CapacityExceeded,
    SeqIdOverflow,
    SamplerInitFailed,
};

/// Duplicate a slice as a null-terminated string.
/// Caller owns the returned memory.
pub fn dupeZ(allocator: std.mem.Allocator, s: []const u8) ![:0]u8 {
    const out = try allocator.alloc(u8, s.len + 1);
    @memcpy(out[0..s.len], s);
    out[s.len] = 0;
    return out[0..s.len :0];
}

/// Returns system info string. Caller owns the returned memory.
/// The underlying C function returns a static buffer; we dupe it for safety.
pub fn systemInfo(allocator: std.mem.Allocator) ![]const u8 {
    const ptr = c.llama_print_system_info();
    return try allocator.dupe(u8, std.mem.span(ptr));
}

/// Global backend lifecycle.
pub const Backend = struct {
    pub fn init() Backend {
        c.llama_backend_init();
        c.ggml_backend_load_all();
        return .{};
    }

    pub fn deinit(_: Backend) void {
        c.llama_backend_free();
    }
};

pub const Model = struct {
    handle: *c.llama_model,

    pub fn load(path_z: [:0]const u8, params: c.llama_model_params) LlamaError!Model {
        const m = c.llama_model_load_from_file(path_z.ptr, params) orelse return LlamaError.ModelLoadFailed;
        return .{ .handle = m };
    }

    /// Frees the model. Takes self by value so both const and var work.
    pub fn deinit(self: Model) void {
        c.llama_model_free(self.handle);
    }

    pub fn vocab(self: Model) ?*const c.llama_vocab {
        return c.llama_model_get_vocab(self.handle);
    }

    pub fn chatTemplate(self: Model) ?[*:0]const u8 {
        return c.llama_model_chat_template(self.handle, null);
    }

    pub fn nCtxTrain(self: Model) i32 {
        return c.llama_model_n_ctx_train(self.handle);
    }
};

pub const Context = struct {
    handle: *c.llama_context,

    pub fn init(model: Model, params: c.llama_context_params) LlamaError!Context {
        const ctx = c.llama_init_from_model(model.handle, params) orelse return LlamaError.ContextInitFailed;
        return .{ .handle = ctx };
    }

    /// Frees the context. Takes self by value.
    pub fn deinit(self: Context) void {
        c.llama_free(self.handle);
    }

    pub fn decode(self: Context, batch: c.llama_batch) LlamaError!void {
        if (c.llama_decode(self.handle, batch) != 0) return LlamaError.DecodeFailed;
    }

    pub fn logitsIth(self: Context, i: i32) ?[*]const f32 {
        return c.llama_get_logits_ith(self.handle, i);
    }

    pub fn kvCacheSeqRm(self: Context, seq_id: i32, p0: i32, p1: i32) void {
        _ = c.llama_memory_seq_rm(c.llama_get_memory(self.handle), seq_id, p0, p1);
    }

    pub fn nCtx(self: Context) u32 {
        return c.llama_n_ctx(self.handle);
    }
};

/// Configuration for sampler chain. All fields have sane defaults.
pub const SamplerConfig = struct {
    temp: f32 = 0.8,
    top_k: i32 = 40,
    top_p: f32 = 0.95,
    min_p: f32 = 0.05,
    seed: u32 = 1234,

    /// Returns true if greedy sampling (temp <= 0).
    pub fn isGreedy(self: SamplerConfig) bool {
        return self.temp <= 0;
    }
};

pub const Sampler = struct {
    handle: *c.llama_sampler,

    /// Initialize with default parameters.
    pub fn initDefault() LlamaError!Sampler {
        return initWithConfig(.{});
    }

    /// Initialize with custom configuration.
    pub fn initWithConfig(config: SamplerConfig) LlamaError!Sampler {
        const params = c.llama_sampler_chain_default_params();
        const chain = c.llama_sampler_chain_init(params) orelse return LlamaError.SamplerInitFailed;
        errdefer c.llama_sampler_free(chain);

        if (config.isGreedy()) {
            // Greedy: just use temp=0
            const temp_sampler = c.llama_sampler_init_temp(0.0) orelse return LlamaError.SamplerInitFailed;
            c.llama_sampler_chain_add(chain, temp_sampler);
        } else {
            // Stochastic: top_k -> top_p -> min_p -> temp
            const top_k_sampler = c.llama_sampler_init_top_k(config.top_k) orelse return LlamaError.SamplerInitFailed;
            c.llama_sampler_chain_add(chain, top_k_sampler);

            const top_p_sampler = c.llama_sampler_init_top_p(config.top_p, 1) orelse return LlamaError.SamplerInitFailed;
            c.llama_sampler_chain_add(chain, top_p_sampler);

            const min_p_sampler = c.llama_sampler_init_min_p(config.min_p, 1) orelse return LlamaError.SamplerInitFailed;
            c.llama_sampler_chain_add(chain, min_p_sampler);

            const temp_sampler = c.llama_sampler_init_temp(config.temp) orelse return LlamaError.SamplerInitFailed;
            c.llama_sampler_chain_add(chain, temp_sampler);
        }

        // Distribution sampler (always needed for actual sampling)
        const dist_sampler = c.llama_sampler_init_dist(config.seed) orelse return LlamaError.SamplerInitFailed;
        c.llama_sampler_chain_add(chain, dist_sampler);

        return .{ .handle = chain };
    }

    /// Legacy API for backward compatibility.
    pub fn initAdvanced(temp: f32, top_k: i32, top_p: f32, seed: u32) LlamaError!Sampler {
        return initWithConfig(.{
            .temp = temp,
            .top_k = top_k,
            .top_p = top_p,
            .seed = seed,
        });
    }

    /// Frees the sampler. Takes self by value.
    pub fn deinit(self: Sampler) void {
        c.llama_sampler_free(self.handle);
    }

    /// Sample from the last token's logits (idx = -1).
    pub fn sampleLast(self: Sampler, ctx: Context) Token {
        return c.llama_sampler_sample(self.handle, ctx.handle, -1);
    }

    /// Sample from a specific token index's logits.
    pub fn sampleAt(self: Sampler, ctx: Context, idx: i32) Token {
        return c.llama_sampler_sample(self.handle, ctx.handle, idx);
    }
};

pub const Batch = struct {
    handle: c.llama_batch,
    capacity: i32,
    n_seq_max: i32,

    pub fn init(n_tokens: i32, embd: i32, n_seq_max: i32) Batch {
        return .{
            .handle = c.llama_batch_init(n_tokens, embd, n_seq_max),
            .capacity = n_tokens,
            .n_seq_max = n_seq_max,
        };
    }

    /// Frees the batch. Takes self by value.
    pub fn deinit(self: Batch) void {
        c.llama_batch_free(self.handle);
    }

    pub fn clear(self: *Batch) void {
        self.handle.n_tokens = 0;
    }

    /// Add a token to the batch. Returns error if capacity exceeded.
    pub fn add(self: *Batch, token: Token, pos: i32, seq_ids: []const i32, logits: bool) LlamaError!void {
        if (self.handle.n_tokens >= self.capacity) {
            return LlamaError.CapacityExceeded;
        }
        if (seq_ids.len > @as(usize, @intCast(self.n_seq_max))) {
            return LlamaError.SeqIdOverflow;
        }

        const i: usize = @intCast(self.handle.n_tokens);
        self.handle.token[i] = token;
        self.handle.pos[i] = pos;
        self.handle.n_seq_id[i] = @intCast(seq_ids.len);
        for (seq_ids, 0..) |id, j| {
            self.handle.seq_id[i][j] = id;
        }
        self.handle.logits[i] = if (logits) 1 else 0;
        self.handle.n_tokens += 1;
    }

    /// Returns remaining capacity.
    pub fn remaining(self: Batch) i32 {
        return self.capacity - self.handle.n_tokens;
    }

    /// Returns true if at capacity.
    pub fn isFull(self: Batch) bool {
        return self.handle.n_tokens >= self.capacity;
    }
};

/// Apply the model's chat template to a list of messages.
/// Returns an owned buffer containing UTF-8 text (NOT null-terminated).
pub fn applyChatTemplate(
    allocator: std.mem.Allocator,
    template: ?[*:0]const u8,
    chat: []const c.llama_chat_message,
    add_assistant: bool,
) LlamaError![]u8 {
    const need_len = c.llama_chat_apply_template(template, chat.ptr, chat.len, add_assistant, null, 0);
    if (need_len <= 0) return LlamaError.TemplateFailed;

    var buf = allocator.alloc(u8, @intCast(need_len)) catch return LlamaError.OutOfMemory;
    errdefer allocator.free(buf);

    const out_len = c.llama_chat_apply_template(template, chat.ptr, chat.len, add_assistant, buf.ptr, need_len);
    if (out_len <= 0) {
        return LlamaError.TemplateFailed;
    }

    var used: usize = @intCast(out_len);
    if (used > 0 and buf[used - 1] == 0) {
        used -= 1;
    } else if (std.mem.indexOfScalar(u8, buf[0..used], 0)) |z| {
        used = z;
    }

    return allocator.realloc(buf, used) catch |err| {
        allocator.free(buf);
        return err;
    };
}

/// Tokenize UTF-8 text into llama tokens.
/// Returns an owned slice whose length matches the allocation size.
pub fn tokenize(
    allocator: std.mem.Allocator,
    vocab: ?*const c.llama_vocab,
    text: []const u8,
    add_bos: bool,
    special: bool,
) LlamaError![]c.llama_token {
    // Initial estimate: 1 token per 4 chars + some slack
    const initial_capacity = @max(text.len / 4 + 32, 64);
    var tmp_tokens = allocator.alloc(c.llama_token, initial_capacity) catch return LlamaError.OutOfMemory;
    errdefer allocator.free(tmp_tokens);

    var n_tok = c.llama_tokenize(vocab, text.ptr, @intCast(text.len), tmp_tokens.ptr, @intCast(tmp_tokens.len), add_bos, special);

    if (n_tok < 0) {
        // Buffer too small, reallocate with exact size
        const need: usize = @intCast(-n_tok);
        tmp_tokens = allocator.realloc(tmp_tokens, need) catch return LlamaError.OutOfMemory;
        n_tok = c.llama_tokenize(vocab, text.ptr, @intCast(text.len), tmp_tokens.ptr, @intCast(tmp_tokens.len), add_bos, special);
    }

    if (n_tok <= 0) {
        return LlamaError.TokenizeFailed;
    }

    const n_tok_usize: usize = @intCast(n_tok);
    return allocator.realloc(tmp_tokens, n_tok_usize) catch |err| {
        allocator.free(tmp_tokens);
        return err;
    };
}

/// Detokenize a single token to a string piece.
/// Returns null if the token cannot be converted.
pub fn tokenToPiece(allocator: std.mem.Allocator, vocab: ?*const c.llama_vocab, token: Token) !?[]const u8 {
    var stack_buf: [256]u8 = undefined;
    const n = c.llama_token_to_piece(vocab, token, &stack_buf, stack_buf.len, 0, false);

    if (n == 0) {
        return null;
    } else if (n > 0) {
        return try allocator.dupe(u8, stack_buf[0..@intCast(n)]);
    } else {
        // Buffer too small, allocate exact size
        const actual_n: usize = @intCast(-n);
        const large_buf = try allocator.alloc(u8, actual_n);
        const n2 = c.llama_token_to_piece(vocab, token, large_buf.ptr, @intCast(large_buf.len), 0, false);
        if (n2 > 0) {
            const used: usize = @intCast(n2);
            if (used < large_buf.len) {
                return allocator.realloc(large_buf, used) catch large_buf[0..used];
            }
            return large_buf;
        }
        allocator.free(large_buf);
        return null;
    }
}
