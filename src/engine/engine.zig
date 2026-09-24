const std = @import("std");
const llama = @import("../llama/llama_cpp.zig");
const chat_lib = @import("chat.zig");
const signal = @import("../app/signal.zig");
const inference = @import("inference.zig");
const openai = @import("../server/openai.zig");
const sched = @import("scheduler.zig");
const residency = @import("../residency/manager.zig");
const residency_bridge = @import("../residency/ggml_bridge.zig");
const memory_policy = @import("../residency/memory_policy.zig");

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

    /// Override chat template name (e.g. "gemma") passed to
    /// llama_chat_apply_template instead of the model's Jinja template.
    chat_template: ?[]const u8 = null,

    /// Path to a draft model for speculative decoding.
    draft_model_path: ?[]const u8 = null,

    /// Max concurrent sequences. >1 enables the continuous-batching scheduler;
    /// 1 keeps the single-stream path (prefix cache + speculative decoding).
    max_concurrent: u32 = 1,

    /// Enable the cross-slot prefix cache in the batched scheduler: a pool of
    /// cache sequences holds prompt prefixes; a request on any slot reuses the
    /// longest cached prefix (full-copied in) and prefills only the suffix.
    /// Default on (validated correct + ~8-11x lower prefill latency on shared
    /// prefixes / multi-turn on transformer and hybrid models); disable with
    /// --no-prefix-cache.
    prefix_cache: bool = true,

    /// Run the normal Engine on the official file-backed GGML buffer backend.
    residency_enabled: bool = false,
    /// Maximum active mapped immutable weight bytes.
    residency_weight_budget_bytes: usize = 0,
    /// Optional hard limit on non-weight context memory (KV + recurrent state,
    /// graph workspace, logits buffer), checked before allocation using
    /// llama.cpp's simulated allocation sizes.
    residency_state_budget_bytes: ?usize = null,
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
    chat_template_z: ?[:0]u8,

    /// Monotonic counter for completion IDs. Accessed lock-free across
    /// concurrent request handlers (server.zig:279 calls nextIdAlloc outside
    /// the engine mutex), so reads/writes must go through atomic operations.
    id_counter: std.atomic.Value(u64),

    mutex: std.Thread.Mutex = .{},
    cached_tokens: std.ArrayList(llama.Token),

    /// Continuous-batching scheduler, present only when max_concurrent > 1.
    scheduler: ?*sched.Scheduler = null,

    /// True when this engine owns the process-global official residency bridge.
    residency_active: bool = false,
    /// Planned and allocated non-weight memory (residency mode only).
    residency_memory: ?MemoryReport = null,

    pub fn residencyMetrics(self: *const Engine) ?residency.Metrics {
        if (!self.residency_active) return null;
        return residency_bridge.metrics();
    }

    pub fn residencyStats(self: *const Engine) ?llama.c.struct_mlz_ggml_residency_stats {
        if (!self.residency_active) return null;
        return llama.c.mlz_ggml_residency_get_stats();
    }

    fn graphFailures(self: *const Engine) u64 {
        const stats = self.residencyStats() orelse return 0;
        return stats.graph_failures;
    }

    /// A decode that failed because a bounded weight could not be mapped is
    /// reported distinctly from other decode errors.
    fn classifyDecodeError(self: *const Engine, err: anyerror, failures_before: u64) anyerror {
        if (err == error.DecodeFailed and self.graphFailures() > failures_before) {
            return error.ResidencyExecutionFailed;
        }
        return err;
    }

    pub fn init(allocator: std.mem.Allocator, model_path: []const u8, cfg: EngineConfig) !Engine {
        const path_z = try llama.dupeZ(allocator, model_path);
        defer allocator.free(path_z);

        var residency_active = false;
        errdefer if (residency_active) teardownResidency(allocator);
        if (cfg.residency_enabled) {
            if (cfg.residency_weight_budget_bytes == 0) {
                std.log.err("residency: weight_budget_mib must be greater than zero", .{});
                return llama.LlamaError.ResidencyInvalidBudget;
            }
            if (cfg.draft_model_path != null) {
                std.log.err("residency: speculative draft models are not supported; remove draft_model", .{});
                return error.ResidencyIncompatibleDraftModel;
            }
            if (!llama.c.mlz_ggml_residency_node_hooks_available()) {
                std.log.err("residency: this binary was built without node hooks; rebuild with -Dggml-residency-hooks=true", .{});
                return llama.LlamaError.ResidencyHooksUnavailable;
            }
            // Logs the offending tensor when a single row cannot fit the budget.
            residency_bridge.init(allocator, path_z, cfg.residency_weight_budget_bytes) catch |err| {
                std.log.err("residency: cannot prepare bounded weights for {s}: {s}", .{ model_path, @errorName(err) });
                return err;
            };
            residency_active = true;

            llama.c.mlz_ggml_residency_registry_reset();
            llama.c.mlz_ggml_residency_reset_stats();
            llama.c.mlz_ggml_residency_set_bridge(
                residency_bridge.acquireCallback,
                residency_bridge.releaseCallback,
                residency_bridge.spanCallback,
                residency_bridge.acquireRangeCallback,
                residency_bridge.rangeCapacityCallback,
                residency_bridge.acquireManyCallback,
            );
            llama.c.mlz_ggml_residency_set_backed_mode(true);
            llama.c.mlz_ggml_residency_set_node_hooks_enabled(true);
            // Test-only: fail the Nth weight acquisition once, so end-to-end
            // tests can prove a failed request unwinds cleanly.
            if (std.process.getEnvVarOwned(allocator, "MLZ_RESIDENCY_INJECT_FAILURE") catch null) |v| {
                defer allocator.free(v);
                const after = std.fmt.parseInt(u64, v, 10) catch return error.InvalidResidencyInjection;
                std.log.warn("residency failure injection armed: acquisition {d} will fail", .{after});
                residency_bridge.injectAcquireFailure(after);
            }
        }

        var mparams = llama.c.llama_model_default_params();
        var residency_overrides = [_]llama.c.llama_model_tensor_buft_override{
            .{ .pattern = ".*", .buft = llama.c.mlz_ggml_residency_buffer_type() },
            .{ .pattern = null, .buft = null },
        };
        mparams.n_gpu_layers = if (cfg.residency_enabled) 0 else cfg.n_gpu_layers;
        mparams.use_mmap = true;
        mparams.use_mlock = false;
        if (cfg.residency_enabled) {
            mparams.check_tensors = false;
            mparams.tensor_buft_overrides = &residency_overrides;
        }

        // Plan every non-weight allocation before the real model/context
        // exist, so an oversized configuration fails without partial state.
        const n_cache: u32 = if (cfg.max_concurrent > 1 and cfg.prefix_cache) cfg.max_concurrent else 0;
        var memory_estimate: ?memory_policy.Estimate = null;
        if (cfg.residency_enabled) {
            const estimate = try estimateContextMemory(path_z, cfg, n_cache);
            memory_estimate = estimate;
            if (cfg.residency_state_budget_bytes) |limit| try enforceStateBudget(estimate, limit);
        }

        const model = try llama.Model.load(path_z, mparams);
        errdefer model.deinit();
        if (cfg.residency_enabled) try residency_bridge.syncRegistry();

        const cparams = try contextParams(cfg, model, n_cache);
        const ctx = try llama.Context.init(model, cparams);
        errdefer ctx.deinit();

        var residency_memory: ?MemoryReport = null;
        if (memory_estimate) |estimate| {
            var actual: llama.c.struct_mlz_llama_memory = undefined;
            if (!llama.c.mlz_llama_memory_breakdown(ctx.handle, llama.c.mlz_ggml_residency_buffer_type(), &actual)) {
                return error.ResidencyMemoryEstimateFailed;
            }
            const report = MemoryReport{
                .estimate = estimate,
                .state_bytes = actual.context,
                .compute_bytes = actual.compute,
                .host_weight_bytes = actual.model,
                .state_budget_bytes = cfg.residency_state_budget_bytes,
            };
            // The simulated plan must bound the real allocation; otherwise the
            // budget decision above was made on wrong numbers.
            if (actual.context > estimate.state_bytes or actual.compute > estimate.compute_bytes) {
                std.log.err(
                    "residency memory estimate below allocation: state {d} > {d} or compute {d} > {d}",
                    .{ actual.context, estimate.state_bytes, actual.compute, estimate.compute_bytes },
                );
                if (cfg.residency_state_budget_bytes) |limit| try enforceStateBudget(.{
                    .state_bytes = actual.context,
                    .compute_bytes = actual.compute,
                    .output_bytes = estimate.output_bytes,
                }, limit);
            }
            std.log.info(
                "residency memory: planned state={d} compute={d} output={d}; allocated state={d} compute={d} host-weights={d}",
                .{ estimate.state_bytes, estimate.compute_bytes, estimate.output_bytes, actual.context, actual.compute, actual.model },
            );
            residency_memory = report;
        }

        const vocab = model.vocab() orelse return error.ModelLoadFailed;
        const tmpl = model.chatTemplate();

        var chat_template_z: ?[:0]u8 = null;
        if (cfg.chat_template) |ct| {
            chat_template_z = try chat_lib.dupeZ(allocator, ct);
        }
        errdefer if (chat_template_z) |t| allocator.free(t);

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

        // Continuous-batching scheduler (opt-in via max_concurrent > 1).
        var scheduler: ?*sched.Scheduler = null;
        if (cfg.max_concurrent > 1) {
            scheduler = try sched.Scheduler.init(allocator, ctx, vocab, cfg.max_concurrent, 1024, cfg.prefix_cache, n_cache);
        }
        errdefer if (scheduler) |s| s.deinit();

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
            .chat_template_z = chat_template_z,
            .id_counter = std.atomic.Value(u64).init(1),
            .cached_tokens = .{},
            .mutex = .{},
            .scheduler = scheduler,
            .residency_active = residency_active,
            .residency_memory = residency_memory,
        };
    }

    pub fn deinit(self: *Engine, allocator: std.mem.Allocator) void {
        // Stop the scheduler thread first — it owns the llama context while running.
        if (self.scheduler) |s| s.deinit();
        self.cached_tokens.deinit(allocator);
        self.batch.deinit();
        if (self.draft_sampler) |s| s.deinit();
        if (self.draft_batch) |b| b.deinit();
        if (self.draft_ctx) |c| c.deinit();
        if (self.draft_model) |m| m.deinit();
        self.ctx.deinit();
        self.model.deinit();
        if (self.residency_active) teardownResidency(allocator);
        if (self.grammar_z) |g| allocator.free(g);
        if (self.grammar_root_z) |r| allocator.free(r);
        if (self.chat_template_z) |t| allocator.free(t);
    }

    pub fn reset(self: *Engine) void {
        self.cached_tokens.clearRetainingCapacity();
        _ = self.ctx.kvCacheSeqRm(0, -1, -1);
        self.batch.clear();
        if (self.draft_batch) |*b| b.clear();
        if (self.draft_ctx) |c| _ = c.kvCacheSeqRm(0, -1, -1);
    }

    pub fn modelId(self: *Engine) []const u8 {
        return std.fs.path.basename(self.model_path);
    }

    pub fn nextIdAlloc(self: *Engine, allocator: std.mem.Allocator) ![]u8 {
        // Produces an ID like chatcmpl-000000000001 (hex). Lock-free atomic
        // increment so concurrent request handlers never observe duplicate IDs.
        const id = self.id_counter.fetchAdd(1, .monotonic);
        return formatChatCompletionId(allocator, id);
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

        const tmpl: ?[*:0]const u8 = if (self.chat_template_z) |t| t.ptr else self.tmpl;
        var prompt = try inference.buildPrompt(allocator, tmpl, self.vocab, msgs.items);
        while (prompt.tokens.len > ctx_limit) {
            // `msgs` is a shallow, borrowed copy of caller-owned messages.
            // Remove entries without freeing their content, and only destroy
            // the old prompt once we know a replacement will be built.
            const start_index: usize = if (msgs.items.len > 0 and msgs.items[0].role == .system) 1 else 0;
            if (msgs.items.len <= start_index) break;
            _ = msgs.orderedRemove(start_index);
            prompt.deinit(allocator);
            prompt = try inference.buildPrompt(allocator, tmpl, self.vocab, msgs.items);
        }
        defer prompt.deinit(allocator);

        if (prompt.tokens.len > ctx_limit) {
            return error.ContextTooSmall;
        }

        // Sampler config with request overrides. Built before the engine mutex
        // because both paths need it and sampler creation is independent state.
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
        sampler.reset();

        const max_tokens: usize = opts.max_tokens orelse 4096;

        // Continuous-batching path: hand off to the scheduler thread. No engine
        // mutex (each request is an independent KV sequence); no prefix cache or
        // speculative decoding in this path yet.
        if (self.scheduler) |sch| {
            defer sampler.deinit();
            var req = sched.Request{
                .prompt_tokens = prompt.tokens,
                .sampler = sampler,
                .max_tokens = max_tokens,
                .sink = opts.sink,
                .allocator = allocator,
            };
            const failures_before = self.graphFailures();
            try sch.submit(&req);
            req.wait();
            defer req.text.deinit(allocator);
            // The scheduler already cleared every in-flight slot's KV.
            if (req.failure) |err| return self.classifyDecodeError(err, failures_before);

            const text = try allocator.dupe(u8, req.text.items);
            const tokens = try allocator.alloc(llama.Token, 0);
            return .{
                .text = text,
                .tokens = tokens,
                .prompt_tokens = prompt.tokens.len,
                .completion_tokens = req.completion_tokens,
                .ttft_ns = null,
                .total_ns = 0,
                .finish_reason = req.finish,
            };
        }

        // Single-stream path.
        defer sampler.deinit();
        self.mutex.lock();
        defer self.mutex.unlock();

        // Calculate common prefix with cached tokens.
        var n_past: usize = 0;
        const n_common = @min(self.cached_tokens.items.len, prompt.tokens.len);
        for (0..n_common) |i| {
            if (self.cached_tokens.items[i] != prompt.tokens[i]) break;
            n_past += 1;
        }
        // llama_get_logits_ith refers to the most recent decode, which may
        // belong to the previous completion. Re-evaluate the last prompt token
        // so a repeated request samples from its own prompt.
        if (prompt.tokens.len > 0) n_past = @min(n_past, prompt.tokens.len - 1);

        // Reset KV cache after the common prefix.
        if (n_past < self.cached_tokens.items.len) {
            if (!self.ctx.kvCacheSeqRm(0, @intCast(n_past), -1)) {
                // M-RoPE: partial sequence removal failed. Fall back to full clear.
                _ = self.ctx.kvCacheSeqRm(0, -1, -1); // whole sequence always succeeds
                self.cached_tokens.clearRetainingCapacity();
                n_past = 0;
            } else {
                self.cached_tokens.shrinkRetainingCapacity(n_past);
            }
        }

        const failures_before = self.graphFailures();
        const gen = inference.generate(allocator, self.ctx, self.vocab, &self.batch, sampler, prompt.tokens, .{
            .max_tokens = max_tokens,
            .sink = opts.sink,
            .shouldStopCtx = opts.shouldStopCtx,
            .shouldStopFn = opts.shouldStopFn,
            .n_past = n_past,
            .draft_ctx = self.draft_ctx,
            .draft_batch = self.draft_batch,
            .draft_sampler = self.draft_sampler,
        }) catch |err| {
            // llama.cpp rolls back only the failing ubatch; earlier ubatches of
            // this request may already sit in KV beyond `cached_tokens`. Drop
            // the whole sequence so no later request reuses a partial prefix.
            _ = self.ctx.kvCacheSeqRm(0, -1, -1);
            self.cached_tokens.clearRetainingCapacity();
            return self.classifyDecodeError(err, failures_before);
        };
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

fn teardownResidency(allocator: std.mem.Allocator) void {
    llama.c.mlz_ggml_residency_set_node_hooks_enabled(false);
    llama.c.mlz_ggml_residency_set_backed_mode(false);
    llama.c.mlz_ggml_residency_set_bridge(null, null, null, null, null, null);
    residency_bridge.deinit(allocator);
    llama.c.mlz_ggml_residency_registry_reset();
}

/// Non-weight memory planned before allocation and measured afterwards.
pub const MemoryReport = struct {
    estimate: memory_policy.Estimate,
    /// Allocated KV cache plus recurrent state.
    state_bytes: usize,
    /// Allocated scheduler graph workspace.
    compute_bytes: usize,
    /// Weight bytes in ordinary host buffers (zero when every tensor is backed).
    host_weight_bytes: usize,
    state_budget_bytes: ?usize,
};

fn contextParams(cfg: EngineConfig, model: llama.Model, n_cache: u32) !llama.c.llama_context_params {
    var cparams = llama.c.llama_context_default_params();
    cparams.n_ctx = if (cfg.n_ctx == 0) blk: {
        const train = model.nCtxTrain();
        break :blk if (train > 0) @as(u32, @intCast(train)) else 4096;
    } else cfg.n_ctx;
    cparams.n_batch = 1024;
    cparams.n_ubatch = 512;
    // Cross-slot prefix cache reserves `n_cache` extra sequences past the
    // serving slots to hold cached prefixes.
    cparams.n_seq_max = std.math.add(u32, @max(@as(u32, 1), cfg.max_concurrent), n_cache) catch
        return error.InvalidConcurrentConfig;
    cparams.offload_kqv = !cfg.residency_enabled;
    if (cfg.residency_enabled) {
        // Match the validated CPU graph path: these optimizations can read
        // reserved weight pointers outside the synchronized node hooks.
        cparams.op_offload = false;
        cparams.flash_attn_type = llama.c.LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    const cpu_count: i32 = @intCast(std.Thread.getCpuCount() catch 4);
    const final_threads = cfg.threads orelse cpu_count;
    cparams.n_threads = final_threads;
    cparams.n_threads_batch = final_threads;
    return cparams;
}

/// Asks llama.cpp for the exact buffer sizes of this context configuration by
/// simulating the model and context with `no_alloc`: nothing is allocated,
/// and weights are neither read nor mapped.
fn estimateContextMemory(path_z: [:0]const u8, cfg: EngineConfig, n_cache: u32) !memory_policy.Estimate {
    var mparams = llama.c.llama_model_default_params();
    mparams.n_gpu_layers = 0;
    mparams.use_mmap = false; // llama.cpp requires this for no_alloc
    mparams.use_mlock = false;
    mparams.check_tensors = false;
    mparams.no_alloc = true;
    const model = try llama.Model.load(path_z, mparams);
    defer model.deinit();
    const cparams = try contextParams(cfg, model, n_cache);
    const ctx = try llama.Context.init(model, cparams);
    defer ctx.deinit();

    var planned: llama.c.struct_mlz_llama_memory = undefined;
    if (!llama.c.mlz_llama_memory_breakdown(ctx.handle, null, &planned)) {
        return error.ResidencyMemoryEstimateFailed;
    }
    const vocab = model.vocab() orelse return error.ModelLoadFailed;
    const n_vocab: usize = @intCast(@max(0, llama.c.llama_vocab_n_tokens(vocab)));
    // Every engine decode requests at most one logits row per sequence.
    return .{
        .state_bytes = planned.context,
        .compute_bytes = planned.compute,
        .output_bytes = memory_policy.outputBytes(n_vocab, cparams.n_seq_max) catch
            return llama.LlamaError.ResidencyStateBudgetExceeded,
    };
}

fn enforceStateBudget(estimate: memory_policy.Estimate, limit: usize) llama.LlamaError!void {
    const verdict = memory_policy.evaluate(estimate, limit);
    const needed = estimate.total() catch std.math.maxInt(usize);
    switch (verdict) {
        .within => {},
        .warning => std.log.warn(
            "residency state plan uses {d} of {d} bytes (state={d} compute={d} output={d})",
            .{ needed, limit, estimate.state_bytes, estimate.compute_bytes, estimate.output_bytes },
        ),
        .rejected => {
            std.log.err(
                "residency state budget exceeded: need {d} bytes (state={d} compute={d} output={d}), budget {d}; lower n_ctx/max_concurrent or raise state_budget_mib",
                .{ needed, estimate.state_bytes, estimate.compute_bytes, estimate.output_bytes, limit },
            );
            return llama.LlamaError.ResidencyStateBudgetExceeded;
        },
    }
}

/// Format a chat completion ID from a numeric counter value. Pulled out of
/// `Engine.nextIdAlloc` so it can be tested without spinning up a model.
pub fn formatChatCompletionId(allocator: std.mem.Allocator, id: u64) ![]u8 {
    var buf: [32]u8 = undefined;
    const n = try std.fmt.bufPrint(&buf, "chatcmpl-{x:0>12}", .{id});
    return try allocator.dupe(u8, n);
}

test "formatChatCompletionId: deterministic format" {
    const t = std.testing;
    const a = try formatChatCompletionId(t.allocator, 1);
    defer t.allocator.free(a);
    try t.expectEqualStrings("chatcmpl-000000000001", a);

    const b = try formatChatCompletionId(t.allocator, 0xdeadbeef);
    defer t.allocator.free(b);
    try t.expectEqualStrings("chatcmpl-0000deadbeef", b);
}

test "id_counter: concurrent fetchAdd never collides" {
    const t = std.testing;
    var counter = std.atomic.Value(u64).init(0);

    const N = 8;
    const PER = 200;

    const Worker = struct {
        fn run(c: *std.atomic.Value(u64), out: []u64) void {
            for (out) |*slot| {
                slot.* = c.fetchAdd(1, .monotonic);
            }
        }
    };

    var slots: [N][PER]u64 = undefined;
    var threads: [N]std.Thread = undefined;
    for (0..N) |i| {
        threads[i] = try std.Thread.spawn(.{}, Worker.run, .{ &counter, slots[i][0..] });
    }
    for (threads) |th| th.join();

    var all = try t.allocator.alloc(u64, N * PER);
    defer t.allocator.free(all);
    var idx: usize = 0;
    for (0..N) |i| {
        for (slots[i]) |v| {
            all[idx] = v;
            idx += 1;
        }
    }
    std.mem.sort(u64, all, {}, std.sort.asc(u64));
    for (0..all.len) |i| try t.expectEqual(@as(u64, i), all[i]);
}
