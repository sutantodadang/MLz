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
};

pub fn dupeZ(allocator: std.mem.Allocator, s: []const u8) ![:0]u8 {
    const out = try allocator.alloc(u8, s.len + 1);
    @memcpy(out[0..s.len], s);
    out[s.len] = 0;
    return out[0..s.len :0];
}

pub fn systemInfo() []const u8 {
    return std.mem.span(c.llama_print_system_info());
}

/// Global backend lifecycle.
pub const Backend = struct {
    pub fn init() Backend {
        c.llama_backend_init();
        _ = c.ggml_backend_load_all();
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

    pub fn deinit(self: *Model) void {
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

    pub fn deinit(self: *Context) void {
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

pub const Sampler = struct {
    handle: *c.llama_sampler,

    pub fn initDefault() Sampler {
        // llama_sampler_chain_init(params)
        // For simplicity, we can use llama_sampler_chain_default() if available,
        // or build a standard chain. Standard chain: temp -> top_k -> top_p -> min_p -> softmax.
        const chain = c.llama_sampler_chain_init(c.llama_sampler_chain_default_params());
        c.llama_sampler_chain_add(chain, c.llama_sampler_init_top_k(40));
        c.llama_sampler_chain_add(chain, c.llama_sampler_init_top_p(0.95, 1));
        c.llama_sampler_chain_add(chain, c.llama_sampler_init_temp(0.8));
        c.llama_sampler_chain_add(chain, c.llama_sampler_init_dist(1234)); // Seed or handle externally
        return .{ .handle = chain };
    }

    pub fn initAdvanced(temp: f32, top_k: i32, top_p: f32, seed: u32) Sampler {
        const params = c.llama_sampler_chain_default_params();
        const chain = c.llama_sampler_chain_init(params);
        if (temp > 0) {
            c.llama_sampler_chain_add(chain, c.llama_sampler_init_top_k(top_k));
            c.llama_sampler_chain_add(chain, c.llama_sampler_init_top_p(top_p, 1));
            c.llama_sampler_chain_add(chain, c.llama_sampler_init_temp(temp));
        } else {
            c.llama_sampler_chain_add(chain, c.llama_sampler_init_temp(0.0));
        }
        c.llama_sampler_chain_add(chain, c.llama_sampler_init_dist(seed));
        return .{ .handle = chain };
    }

    pub fn deinit(self: *Sampler) void {
        c.llama_sampler_free(self.handle);
    }

    pub fn sample(self: Sampler, ctx: Context, idx: i32) Token {
        return c.llama_sampler_sample(self.handle, ctx.handle, idx);
    }
};

pub const Batch = struct {
    handle: c.llama_batch,

    pub fn init(n_tokens: i32, embd: i32, n_seq_max: i32) Batch {
        return .{ .handle = c.llama_batch_init(n_tokens, embd, n_seq_max) };
    }

    pub fn deinit(self: *Batch) void {
        c.llama_batch_free(self.handle);
    }

    pub fn clear(self: *Batch) void {
        self.handle.n_tokens = 0;
    }

    pub fn add(self: *Batch, token: Token, pos: i32, seq_ids: []const i32, logits: bool) void {
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

    var buf = try allocator.alloc(u8, @intCast(need_len));
    const out_len = c.llama_chat_apply_template(template, chat.ptr, chat.len, add_assistant, @ptrCast(buf.ptr), need_len);
    if (out_len <= 0) {
        allocator.free(buf);
        return LlamaError.TemplateFailed;
    }

    var used: usize = @intCast(out_len);
    if (used > 0 and buf[used - 1] == 0) {
        used -= 1;
    } else if (std.mem.indexOfScalar(u8, buf[0..used], 0)) |z| {
        used = z;
    }

    // Make the returned slice match the allocation size so `allocator.free()` is correct.
    buf = try allocator.realloc(buf, used);
    return buf;
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
    var tmp_tokens = try allocator.alloc(c.llama_token, text.len + 32);
    var n_tok = c.llama_tokenize(vocab, @ptrCast(text.ptr), @intCast(text.len), tmp_tokens.ptr, @intCast(tmp_tokens.len), add_bos, special);
    if (n_tok < 0) {
        const need: usize = @intCast(-n_tok);
        tmp_tokens = try allocator.realloc(tmp_tokens, need);
        n_tok = c.llama_tokenize(vocab, @ptrCast(text.ptr), @intCast(text.len), tmp_tokens.ptr, @intCast(tmp_tokens.len), add_bos, special);
    }
    if (n_tok <= 0) {
        allocator.free(tmp_tokens);
        return LlamaError.TokenizeFailed;
    }

    const n_tok_usize: usize = @intCast(n_tok);
    tmp_tokens = try allocator.realloc(tmp_tokens, n_tok_usize);
    return tmp_tokens;
}
