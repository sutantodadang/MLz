const std = @import("std");
const residency = @import("residency.zig");
const gguf = @import("gguf_residency.zig");
const executor_mod = @import("residency_executor.zig");
const llama = @import("llama_cpp.zig");

pub const Error = executor_mod.Error || llama.LlamaError || gguf.Error || error{
    VocabUnavailable,
    PromptEmpty,
    MissingMetadata,
    TimerUnsupported,
};

/// Serving boundary for bounded-residency completion on a single GGUF model.
///
/// The llama.cpp model handle provides the vocabulary/detokenizer only (mmap);
/// it is never used for compute. All weight compute runs through the
/// bounded-residency CPU executor with an explicit weight-mapping budget.
pub const ResidencyService = struct {
    allocator: std.mem.Allocator,
    store: residency.BackingStore,
    index: gguf.TensorIndex,
    model: llama.Model,
    vocab: *const llama.c.llama_vocab,
    vocab_size: usize,
    eos_token: usize,
    bos_token: usize,

    pub const Config = struct {
        budget_bytes: usize,
        prefill_chunk: usize = 32,
        context_capacity: usize = 1024,
    };

    pub fn open(allocator: std.mem.Allocator, path: []const u8, config: Config) Error!ResidencyService {
        if (config.budget_bytes == 0 or config.prefill_chunk == 0 or config.context_capacity == 0) {
            return Error.InvalidBudget;
        }

        const path_z = try allocator.dupeZ(u8, path);
        defer allocator.free(path_z);

        var store = try residency.BackingStore.open(path_z);
        errdefer store.close();

        var index = try gguf.TensorIndex.open(allocator, path_z, store.size);
        errdefer index.deinit();

        // Vocab-only llama.cpp load; compute weights are never decoded through
        // this handle.
        const model = try llama.Model.load(path_z, llama.c.llama_model_default_params());
        errdefer model.deinit();
        const vocab = model.vocab() orelse return Error.VocabUnavailable;
        const vocab_size_i64 = llama.c.llama_vocab_n_tokens(vocab);
        if (vocab_size_i64 <= 0) return Error.VocabUnavailable;
        const eos = llama.c.llama_vocab_eos(vocab);
        const bos = llama.c.llama_vocab_bos(vocab);
        if (eos < 0 or bos < 0) return Error.VocabUnavailable;

        return .{
            .allocator = allocator,
            .store = store,
            .index = index,
            .model = model,
            .vocab = vocab,
            .vocab_size = @intCast(vocab_size_i64),
            .eos_token = @intCast(eos),
            .bos_token = @intCast(bos),
        };
    }

    pub fn close(self: *ResidencyService) void {
        self.model.deinit();
        self.index.deinit();
        self.store.close();
    }

    fn computeSupported(descriptor: *const gguf.TensorDescriptor) bool {
        return switch (descriptor.ggml_type) {
            gguf.type_f32, gguf.type_q4_0, gguf.type_q4_k, gguf.type_q6_k, gguf.type_q2_k => true,
            else => false,
        };
    }

    fn layerWeights(self: *ResidencyService, layer: usize) ?executor_mod.DecoderLayerWeights {
        const suffixes = [_][]const u8{
            "attn_norm.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_norm.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        };
        var found: [suffixes.len]*const gguf.TensorDescriptor = undefined;
        for (suffixes, 0..) |suffix, i| {
            const name = std.fmt.allocPrint(self.allocator, "blk.{d}.{s}", .{ layer, suffix }) catch return null;
            defer self.allocator.free(name);
            found[i] = self.index.get(name) orelse return null;
            if (found[i].n_dimensions == 2 and !computeSupported(found[i])) return null;
        }
        return .{
            .attention_norm = found[0],
            .query = found[1],
            .key = found[2],
            .value = found[3],
            .attention_output = found[4],
            .ffn_norm = found[5],
            .ffn_gate = found[6],
            .ffn_up = found[7],
            .ffn_down = found[8],
        };
    }

    pub const TokenizeResult = struct {
        tokens: []usize,
    };

    pub fn tokenize(self: *ResidencyService, text: []const u8, add_bos: bool) Error!TokenizeResult {
        if (text.len == 0) return Error.PromptEmpty;
        const tokens_c = try llama.tokenize(self.allocator, self.vocab, text, add_bos, false);
        defer self.allocator.free(tokens_c);
        const tokens = try self.allocator.alloc(usize, tokens_c.len);
        errdefer self.allocator.free(tokens);
        for (tokens_c, 0..) |token, i| {
            const unsigned: usize = @intCast(@as(i64, token));
            if (unsigned >= self.vocab_size) return Error.InvalidToken;
            tokens[i] = unsigned;
        }
        return .{ .tokens = tokens };
    }

    pub fn tokenPiece(self: *ResidencyService, token: usize, buffer: []u8) ?[]const u8 {
        const n = llama.c.llama_token_to_piece(
            self.vocab,
            @intCast(token),
            buffer.ptr,
            @intCast(buffer.len),
            0,
            false,
        );
        if (n <= 0 or @as(usize, @intCast(n)) > buffer.len) return null;
        return buffer[0..@intCast(n)];
    }

    pub const CompletionResult = struct {
        text: []u8,
        tokens: []usize,
        prompt_tokens: usize,
        elapsed_ms: f64,
        weight_budget_bytes: usize,
        peak_mapped_weight_bytes: usize,
        dequant_scratch_bytes: usize,
        activation_bytes: usize,
        attention_workspace_bytes: usize,
        kv_cache_bytes: usize,
        faults: u64,
        hits: u64,
        evictions: u64,
        rss_bytes: ?u64,
    };

    pub const CompletionOptions = struct {
        max_tokens: usize = 32,
        add_bos: bool = false,
        /// Pre-tokenized prompt; skips `tokenize()` when provided.
        prompt_tokens: ?[]const usize = null,
    };

    /// Runs a greedy completion through the bounded-residency executor. The
    /// residency manager (and every weight window) is unmapped when it returns.
    pub fn complete(
        self: *ResidencyService,
        config: Config,
        options: CompletionOptions,
    ) Error!CompletionResult {
        const embedding = self.index.get("token_embd.weight") orelse return Error.MissingMetadata;
        const output_norm = self.index.get("output_norm.weight") orelse return Error.MissingMetadata;
        const output_weight = self.index.get("output.weight") orelse embedding;
        if (!computeSupported(embedding) or !computeSupported(output_weight)) {
            return Error.UnsupportedGguf;
        }

        const block_count_u32 = self.index.execution.block_count orelse return Error.MissingMetadata;
        const head_count_u32 = self.index.execution.attention_head_count orelse return Error.MissingMetadata;
        const kv_head_count_u32 = self.index.execution.attention_kv_head_count orelse head_count_u32;
        const block_count: usize = block_count_u32;
        const head_count: usize = head_count_u32;
        if (block_count == 0 or head_count == 0) return Error.MissingMetadata;
        const hidden = std.math.cast(usize, embedding.dimensions[0]) orelse return Error.InvalidExecutionShape;
        if (hidden % head_count != 0) return Error.UnsupportedGguf;
        const kv_head_count: usize = kv_head_count_u32;
        const kv_width = std.math.mul(usize, kv_head_count, hidden / head_count) catch return Error.InvalidExecutionShape;

        var intermediate: usize = 0;
        for (self.index.descriptors) |descriptor| {
            if (std.mem.startsWith(u8, descriptor.name, "blk.") and
                std.mem.endsWith(u8, descriptor.name, ".ffn_gate.weight"))
            {
                intermediate = @max(intermediate, std.math.cast(usize, descriptor.dimensions[1]) orelse return Error.InvalidExecutionShape);
            }
        }
        if (intermediate == 0) return Error.MissingMetadata;

        const layers = try self.allocator.alloc(executor_mod.DecoderLayerWeights, block_count);
        defer self.allocator.free(layers);
        for (layers, 0..) |*layer, i| {
            layer.* = self.layerWeights(i) orelse return Error.UnsupportedGguf;
        }

        const attention_config = executor_mod.AttentionConfig{
            .head_count = head_count,
            .kv_head_count = kv_head_count,
            .head_dim = hidden / head_count,
            .rms_epsilon = self.index.execution.rms_epsilon orelse 1e-5,
            .rope_theta = self.index.execution.rope_theta orelse 10_000.0,
        };

        const prompt_tokens: []const usize = options.prompt_tokens orelse blk: {
            // complete() requires pre-tokenized prompts when text is absent.
            break :blk &[_]usize{};
        };
        if (prompt_tokens.len == 0) return Error.PromptEmpty;
        for (prompt_tokens) |token| {
            if (token >= self.vocab_size) return Error.InvalidToken;
        }
        const generation_capacity = std.math.add(usize, prompt_tokens.len, options.max_tokens) catch return Error.KvCacheFull;
        if (generation_capacity > config.context_capacity) return Error.KvCacheFull;

        var manager = try residency.Manager.init(self.allocator, &self.store, config.budget_bytes);
        defer manager.deinit();
        try self.index.registerAll(&manager);

        var executor = try executor_mod.CpuExecutor.init(
            self.allocator,
            &manager,
            @max(hidden, intermediate),
            intermediate,
            intermediate,
        );
        defer executor.deinit();

        var workspace = try executor_mod.AttentionWorkspace.init(self.allocator, hidden);
        defer workspace.deinit();

        var prefill_workspace = try executor_mod.PrefillWorkspace.init(
            self.allocator,
            config.prefill_chunk,
            hidden,
            intermediate,
        );
        defer prefill_workspace.deinit();

        const caches = try self.allocator.alloc(executor_mod.KvCache, block_count);
        defer self.allocator.free(caches);
        var initialized: usize = 0;
        defer for (caches[0..initialized]) |*cache| cache.deinit();
        for (caches) |*cache| {
            cache.* = try executor_mod.KvCache.init(self.allocator, config.context_capacity, kv_width);
            initialized += 1;
        }

        const chunk_states = try self.allocator.alloc(f32, config.prefill_chunk * hidden);
        defer self.allocator.free(chunk_states);
        const state = try self.allocator.alloc(f32, hidden);
        defer self.allocator.free(state);
        const logits = try self.allocator.alloc(f32, self.vocab_size);
        defer self.allocator.free(logits);

        var timer = try std.time.Timer.start();
        try executor.modelPrefillChunked(
            embedding,
            layers,
            output_norm,
            output_weight,
            prompt_tokens,
            config.prefill_chunk,
            attention_config,
            caches,
            &prefill_workspace,
            chunk_states,
            logits,
        );

        var generated: std.ArrayList(usize) = .empty;
        errdefer generated.deinit(self.allocator);
        var text: std.ArrayList(u8) = .empty;
        errdefer text.deinit(self.allocator);

        var piece_buffer: [256]u8 = undefined;
        var total_tokens: usize = 0;
        while (total_tokens < options.max_tokens) {
            const argmax = blk: {
                var best: usize = 0;
                for (logits, 0..) |value, i| {
                    if (value > logits[best]) best = i;
                }
                break :blk best;
            };
            if (argmax == self.eos_token) break;
            try generated.append(self.allocator, argmax);

            if (self.tokenPiece(argmax, &piece_buffer)) |piece| {
                try text.appendSlice(self.allocator, piece);
            }

            const next_position = caches[0].len;
            if (next_position >= config.context_capacity) break;
            const single = [_]usize{argmax};
            try executor.modelTokens(
                embedding,
                layers,
                output_norm,
                output_weight,
                &single,
                attention_config,
                caches,
                &workspace,
                state,
                logits,
            );
            total_tokens += 1;
        }
        const elapsed_ms = @as(f64, @floatFromInt(timer.read())) / 1.0e6;

        const accounting = executor.decoderAccounting(&caches[0], &workspace);
        return .{
            .text = try text.toOwnedSlice(self.allocator),
            .tokens = try generated.toOwnedSlice(self.allocator),
            .prompt_tokens = prompt_tokens.len,
            .elapsed_ms = elapsed_ms,
            .weight_budget_bytes = accounting.executor.weight_budget_bytes,
            .peak_mapped_weight_bytes = accounting.executor.peak_mapped_weight_bytes,
            .dequant_scratch_bytes = accounting.executor.dequant_scratch_bytes,
            .activation_bytes = accounting.executor.activation_bytes,
            .attention_workspace_bytes = accounting.attention_workspace_bytes,
            .kv_cache_bytes = accounting.kv_cache_bytes,
            .faults = accounting.executor.faults,
            .hits = accounting.executor.hits,
            .evictions = accounting.executor.evictions,
            .rss_bytes = residency.currentRss(),
        };
    }
};
