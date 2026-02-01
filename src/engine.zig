const std = @import("std");
const llama = @import("llama_cpp.zig");
const chat_lib = @import("chat.zig");
const signal = @import("signal.zig");
const inference = @import("inference.zig");
const openai = @import("openai.zig");

/// Configuration for the inference engine.
pub const EngineConfig = struct {
    // Model/context
    /// Context window size (tokens).
    n_ctx: u32 = 4096,
    /// Number of layers to offload to GPU.
    n_gpu_layers: i32 = 999,
    /// Number of threads to use for generation (null = auto-detect).
    threads: ?i32 = null,

    // Sampler defaults
    /// Sampling temperature (higher = more random).
    temp: f32 = 0.8,
    /// Top-K sampling.
    top_k: i32 = 40,
    /// Top-P (nucleus) sampling.
    top_p: f32 = 0.95,
    /// Min-P sampling.
    min_p: f32 = 0.05,
    /// Random seed.
    seed: u32 = 42,

    // Optional grammar
    /// Path to a GBNF grammar file.
    grammar_path: ?[]const u8 = null,
    /// Root rule name for the grammar.
    grammar_root: []const u8 = "root",

    /// Path to a draft model for speculative decoding.
    draft_model_path: ?[]const u8 = null,
};

pub const ChatOptions = struct {
    temp: ?f32 = null,
    top_k: ?i32 = null,
    top_p: ?f32 = null,
    min_p: ?f32 = null,
    seed: ?u32 = null,
    max_tokens: ?usize = null,
    sink: ?inference.TokenSink = null,
    shouldStopCtx: ?*anyopaque = null,
    shouldStopFn: ?*const fn (ctx: *anyopaque) bool = null,
};

/// High-level engine managing model, context, and KV cache.
pub const Engine = struct {
    model_path: []const u8,

    model: llama.Model,
    ctx: llama.Context,
    vocab: *const llama.c.llama_vocab,
    tmpl: ?[*:0]const u8,

    batch: llama.Batch,

    draft_model: ?llama.Model = null,
    draft_ctx: ?llama.Context = null,
    draft_batch: ?llama.Batch = null,
    draft_sampler: ?llama.Sampler = null,

    cfg: EngineConfig,
    grammar_z: ?[:0]u8,
    grammar_root_z: ?[:0]u8,

    id_counter: u64,

    mutex: std.Thread.Mutex = .{},
    cached_tokens: std.ArrayList(llama.Token),

    pub fn init(allocator: std.mem.Allocator, model_path: []const u8, cfg: EngineConfig) !Engine {
        const path_z = try llama.dupeZ(allocator, model_path);
        defer allocator.free(path_z);

        var mparams = llama.c.llama_model_default_params();
        mparams.n_gpu_layers = cfg.n_gpu_layers;
        mparams.use_mmap = true;
        mparams.use_mlock = false;

        const model = try llama.Model.load(path_z, mparams);
        errdefer model.deinit();

        var cparams = llama.c.llama_context_default_params();
        cparams.n_ctx = cfg.n_ctx;
        cparams.n_batch = 1024;
        cparams.n_ubatch = 512;
        cparams.n_seq_max = 1;
        cparams.offload_kqv = true;

        const cpu_count: i32 = @intCast(std.Thread.getCpuCount() catch 4);
        const final_threads = cfg.threads orelse cpu_count;
        cparams.n_threads = final_threads;
        cparams.n_threads_batch = final_threads;

        const ctx = try llama.Context.init(model, cparams);
        errdefer ctx.deinit();

        const vocab = model.vocab() orelse return error.ModelLoadFailed;
        const tmpl = model.chatTemplate();

        var grammar_z: ?[:0]u8 = null;
        var grammar_root_z: ?[:0]u8 = null;
        if (cfg.grammar_path) |gp| {
            const bytes = try std.fs.cwd().readFileAlloc(allocator, gp, 4 * 1024 * 1024);
            defer allocator.free(bytes);
            grammar_z = try chat_lib.dupeZ(allocator, bytes);
            grammar_root_z = try chat_lib.dupeZ(allocator, cfg.grammar_root);
        }

        var draft_model: ?llama.Model = null;
        var draft_ctx: ?llama.Context = null;
        var draft_batch: ?llama.Batch = null;
        var draft_sampler: ?llama.Sampler = null;

        if (cfg.draft_model_path) |draft_path| {
            const draft_path_z = try llama.dupeZ(allocator, draft_path);
            defer allocator.free(draft_path_z);

            var draft_params = mparams;
            draft_params.n_gpu_layers = -1; // Try to retain some for main model, or simple defaults

            draft_model = try llama.Model.load(draft_path_z, draft_params);

            var draft_cparams = cparams;
            draft_cparams.n_batch = 512; // Smaller batch for draft

            draft_ctx = try llama.Context.init(draft_model.?, draft_cparams);
            draft_batch = llama.Batch.init(512, 0, 1);

            // Greedier sampling for draft usually works better for speculation efficiency
            draft_sampler = try llama.Sampler.initAdvanced(0.0, 1, 1.0, 42);
        }
        errdefer if (draft_model) |m| m.deinit();
        errdefer if (draft_ctx) |c| c.deinit();
        errdefer if (draft_batch) |b| b.deinit();
        errdefer if (draft_sampler) |s| s.deinit();

        return .{
            .model_path = model_path,
            .model = model,
            .ctx = ctx,
            .vocab = vocab,
            .tmpl = tmpl,
            .batch = llama.Batch.init(1024, 0, 1),
            .draft_model = draft_model,
            .draft_ctx = draft_ctx,
            .draft_batch = draft_batch,
            .draft_sampler = draft_sampler,
            .cfg = cfg,
            .grammar_z = grammar_z,
            .grammar_root_z = grammar_root_z,
            .id_counter = 1,
            .cached_tokens = .{},
            .mutex = .{},
        };
    }

    pub fn deinit(self: *Engine, allocator: std.mem.Allocator) void {
        self.cached_tokens.deinit(allocator);
        self.batch.deinit();
        if (self.draft_sampler) |s| s.deinit();
        if (self.draft_batch) |b| b.deinit();
        if (self.draft_ctx) |c| c.deinit();
        if (self.draft_model) |m| m.deinit();
        self.ctx.deinit();
        self.model.deinit();
        if (self.grammar_z) |g| allocator.free(g);
        if (self.grammar_root_z) |r| allocator.free(r);
    }

    pub fn reset(self: *Engine) void {
        self.cached_tokens.clearRetainingCapacity();
        self.ctx.kvCacheSeqRm(0, -1, -1);
        self.batch.clear();
        if (self.draft_batch) |*b| b.clear();
        if (self.draft_ctx) |c| c.kvCacheSeqRm(0, -1, -1);
    }

    pub fn modelId(self: *Engine) []const u8 {
        return std.fs.path.basename(self.model_path);
    }

    pub fn nextIdAlloc(self: *Engine, allocator: std.mem.Allocator) ![]u8 {
        // Produces an ID like chatcmpl-000000000001 (hex).
        var buf: [32]u8 = undefined;
        const n = try std.fmt.bufPrint(&buf, "chatcmpl-{x:0>12}", .{self.id_counter});
        self.id_counter += 1;
        return try allocator.dupe(u8, n);
    }

    pub const Completion = struct {
        value: openai.ChatCompletionResponse,
        finish_reason: []const u8,

        pub fn deinit(self: *Completion, allocator: std.mem.Allocator) void {
            allocator.free(self.value.id);
            allocator.free(self.value.model);
            if (self.value.choices.len > 0) {
                allocator.free(self.value.choices[0].message.content);
            }
            allocator.free(self.value.choices);
        }

        pub fn finishReasonString(self: *Completion) []const u8 {
            return self.finish_reason;
        }
    };

    pub fn chat(
        self: *Engine,
        allocator: std.mem.Allocator,
        messages: []const chat_lib.Message,
        opts: ChatOptions,
    ) !inference.GenerationResult {
        // Create a local mutable copy of messages for trimming
        var msgs = try std.ArrayList(chat_lib.Message).initCapacity(allocator, messages.len);
        defer msgs.deinit(allocator);
        // Since we need to own the strings if we drop them?
        // chat.Message content is []const u8.
        // If we are just slicing/referencing, we don't own content.
        // dropOldestNonSystem moves items.
        // We can just copy the structs.
        for (messages) |m| {
            try msgs.append(allocator, m);
        }

        // Build prompt and trim to fit context.
        const ctx_reserve: usize = 256;
        const ctx_limit: usize = @as(usize, @intCast(self.ctx.nCtx())) - ctx_reserve;

        var prompt = try inference.buildPrompt(allocator, self.tmpl, self.vocab, msgs.items);
        while (prompt.tokens.len > ctx_limit) {
            prompt.deinit(allocator);
            if (!chat_lib.dropOldestNonSystem(&msgs, allocator)) break;
            prompt = try inference.buildPrompt(allocator, self.tmpl, self.vocab, msgs.items);
        }
        defer prompt.deinit(allocator);

        if (prompt.tokens.len > ctx_limit) {
            return error.ContextTooSmall;
        }

        self.mutex.lock();
        defer self.mutex.unlock();

        // Calculate common prefix with cached tokens.
        var n_past: usize = 0;
        const n_common = @min(self.cached_tokens.items.len, prompt.tokens.len);
        for (0..n_common) |i| {
            if (self.cached_tokens.items[i] != prompt.tokens[i]) break;
            n_past += 1;
        }

        // Reset KV cache after the common prefix.
        if (n_past < self.cached_tokens.items.len) {
            self.ctx.kvCacheSeqRm(0, @intCast(n_past), -1);
            self.cached_tokens.shrinkRetainingCapacity(n_past);
        }

        // Sampler config with request overrides.
        const s_cfg = llama.SamplerConfig{
            .temp = opts.temp orelse self.cfg.temp,
            .top_k = opts.top_k orelse self.cfg.top_k,
            .top_p = opts.top_p orelse self.cfg.top_p,
            .min_p = opts.min_p orelse self.cfg.min_p,
            .seed = opts.seed orelse self.cfg.seed,
        };

        var sampler: llama.Sampler = undefined;
        if (self.grammar_z) |g| {
            const default_root: [:0]const u8 = "root";
            const root = self.grammar_root_z orelse default_root;
            sampler = try llama.Sampler.initWithConfigAndGrammar(s_cfg, self.vocab, g, root);
        } else {
            sampler = try llama.Sampler.initWithConfig(s_cfg);
        }
        defer sampler.deinit();
        sampler.reset();

        const max_tokens: usize = opts.max_tokens orelse 4096;

        const gen = try inference.generate(allocator, self.ctx, self.vocab, &self.batch, sampler, prompt.tokens, .{
            .max_tokens = max_tokens,
            .sink = opts.sink,
            .shouldStopCtx = opts.shouldStopCtx,
            .shouldStopFn = opts.shouldStopFn,
            .n_past = n_past,
            .draft_ctx = self.draft_ctx,
            .draft_batch = self.draft_batch,
            .draft_sampler = self.draft_sampler,
        });
        // Caller owns gen (GenerationResult)

        // Update cache with new tokens.
        if (n_past < prompt.tokens.len) {
            try self.cached_tokens.appendSlice(allocator, prompt.tokens[n_past..]);
        }
        try self.cached_tokens.appendSlice(allocator, gen.tokens);

        return gen;
    }

    pub fn complete(
        self: *Engine,
        allocator: std.mem.Allocator,
        req: openai.ChatCompletionRequest,
        sink: ?inference.TokenSink,
        forced_id: ?[]const u8,
    ) !Completion {
        // Convert request messages to internal representation.
        var msgs: std.ArrayList(chat_lib.Message) = .empty;
        defer chat_lib.deinitMessages(allocator, &msgs);

        for (req.messages) |m| {
            const role = if (std.ascii.eqlIgnoreCase(m.role, "system")) chat_lib.Role.system else if (std.ascii.eqlIgnoreCase(m.role, "user")) chat_lib.Role.user else if (std.ascii.eqlIgnoreCase(m.role, "assistant")) chat_lib.Role.assistant else return error.InvalidRole;
            const content_z = try chat_lib.dupeZ(allocator, m.content);
            try msgs.append(allocator, .{ .role = role, .content = content_z });
        }

        const max_tokens = if (req.max_tokens) |m| @as(usize, @intCast(m)) else null;

        const gen = try self.chat(allocator, msgs.items, .{
            .temp = req.temperature,
            .top_p = req.top_p,
            .seed = req.seed,
            .max_tokens = max_tokens,
            .sink = sink,
        });
        defer gen.deinit(allocator);

        const finish_reason = switch (gen.finish_reason) {
            .stop => "stop",
            .length => "length",
            .context_limit => "length",
            .aborted => "stop",
        };

        const id = if (forced_id) |fid| try allocator.dupe(u8, fid) else try self.nextIdAlloc(allocator);
        const model_name = try allocator.dupe(u8, req.model orelse self.modelId());
        const msg_content = try allocator.dupe(u8, gen.text);

        const choice = openai.ChatCompletionChoice{
            .index = 0,
            .message = .{ .role = "assistant", .content = msg_content },
            .finish_reason = finish_reason,
        };

        const choices = try allocator.alloc(openai.ChatCompletionChoice, 1);
        choices[0] = choice;

        const usage = openai.Usage{
            .prompt_tokens = gen.prompt_tokens,
            .completion_tokens = gen.completion_tokens,
            .total_tokens = gen.prompt_tokens + gen.completion_tokens,
        };

        const resp = openai.ChatCompletionResponse{
            .id = id,
            .object = "chat.completion",
            .created = std.time.timestamp(),
            .model = model_name,
            .choices = choices,
            .usage = usage,
        };

        return .{ .value = resp, .finish_reason = finish_reason };
    }
};
