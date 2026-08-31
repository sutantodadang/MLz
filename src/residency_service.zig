const std = @import("std");
const residency = @import("residency.zig");
const gguf = @import("gguf_residency.zig");
const executor_mod = @import("residency_executor.zig");
const llama = @import("llama_cpp.zig");
const qwen = @import("residency_qwen3next.zig");

pub const Error = executor_mod.Error || llama.LlamaError || qwen.Error || gguf.Error || error{
    VocabUnavailable,
    PromptEmpty,
    MissingMetadata,
    InvalidSamplingOptions,
    TimerUnsupported,
};

pub const SamplingStrategy = enum {
    /// Select the highest logit. This preserves the original completion behavior.
    greedy,
    /// Sample from the temperature-scaled distribution, optionally restricted
    /// to the highest `top_k` logits.
    temperature,
};

const Candidate = struct {
    token: usize,
    logit: f32,
};

/// Small, explicitly defined PRNG so a fixed seed gives a reproducible sample
/// stream. SplitMix64 is sufficient here because sampling does not require a
/// cryptographically secure stream.
const SamplingPrng = struct {
    state: u64,

    fn init(seed: u64) SamplingPrng {
        return .{ .state = seed };
    }

    fn next(self: *SamplingPrng) u64 {
        self.state +%= 0x9e3779b97f4a7c15;
        var value = self.state;
        value = (value ^ (value >> 30)) *% 0xbf58476d1ce4e5b9;
        value = (value ^ (value >> 27)) *% 0x94d049bb133111eb;
        return value ^ (value >> 31);
    }

    fn unitFloat(self: *SamplingPrng) f64 {
        // Use the high 53 bits to produce a value in [0, 1).
        return @as(f64, @floatFromInt(self.next() >> 11)) * (1.0 / 9_007_199_254_740_992.0);
    }
};

fn greedyToken(logits: []const f32) usize {
    var best: usize = 0;
    for (logits[1..], 1..) |value, i| {
        if (value > logits[best]) best = i;
    }
    return best;
}

fn candidateBetter(left: Candidate, right: Candidate) bool {
    const left_nan = left.logit != left.logit;
    const right_nan = right.logit != right.logit;
    if (left_nan != right_nan) return !left_nan;
    if (left.logit != right.logit) return left.logit > right.logit;
    return left.token < right.token;
}

fn candidateWorse(left: Candidate, right: Candidate) bool {
    return candidateBetter(right, left);
}

fn siftWorstDown(candidates: []Candidate, start: usize) void {
    var root = start;
    while (root * 2 + 1 < candidates.len) {
        const left = root * 2 + 1;
        const right = left + 1;
        var worst = root;
        if (candidateWorse(candidates[left], candidates[worst])) worst = left;
        if (right < candidates.len and candidateWorse(candidates[right], candidates[worst])) worst = right;
        if (worst == root) return;
        std.mem.swap(Candidate, &candidates[root], &candidates[worst]);
        root = worst;
    }
}

fn fillTopCandidates(logits: []const f32, candidates: []Candidate) void {
    std.debug.assert(candidates.len > 0 and candidates.len < logits.len);
    for (candidates, 0..) |*candidate, token| {
        candidate.* = .{ .token = token, .logit = logits[token] };
    }

    // Keep the worst retained candidate at the root of a min-heap, making
    // top-k selection O(vocabulary * log(k)) without a vocabulary-sized copy.
    var heap_index = candidates.len / 2;
    while (heap_index > 0) {
        heap_index -= 1;
        siftWorstDown(candidates, heap_index);
    }

    for (logits[candidates.len..], candidates.len..) |logit, token| {
        const candidate = Candidate{ .token = token, .logit = logit };
        if (candidateBetter(candidate, candidates[0])) {
            candidates[0] = candidate;
            siftWorstDown(candidates, 0);
        }
    }
}

fn sampleCandidates(
    logits: []const f32,
    candidates: ?[]const Candidate,
    temperature: f32,
    prng: *SamplingPrng,
) usize {
    var maximum = -std.math.inf(f32);
    var positive_infinities: usize = 0;
    var negative_infinities: usize = 0;
    var valid_count: usize = 0;
    var last_valid: ?usize = null;

    const count = if (candidates) |items| items.len else logits.len;
    for (0..count) |i| {
        const candidate = if (candidates) |items| items[i] else Candidate{ .token = i, .logit = logits[i] };
        if (candidate.logit != candidate.logit) continue;
        valid_count += 1;
        last_valid = candidate.token;
        if (candidate.logit == std.math.inf(f32)) positive_infinities += 1;
        if (candidate.logit == -std.math.inf(f32)) negative_infinities += 1;
        maximum = @max(maximum, candidate.logit);
    }

    // Infinities cannot participate in a finite softmax. Treat equal positive
    // infinities as an equiprobable set.
    if (positive_infinities > 0) {
        const selected: usize = @intFromFloat(prng.unitFloat() * @as(f64, @floatFromInt(positive_infinities)));
        var seen: usize = 0;
        for (0..count) |i| {
            const candidate = if (candidates) |items| items[i] else Candidate{ .token = i, .logit = logits[i] };
            if (candidate.logit == std.math.inf(f32)) {
                if (seen == selected) return candidate.token;
                seen += 1;
            }
        }
        unreachable;
    }

    // If every valid logit is negative infinity, they are also equiprobable.
    if (negative_infinities == valid_count and valid_count > 0) {
        const selected: usize = @intFromFloat(prng.unitFloat() * @as(f64, @floatFromInt(valid_count)));
        var seen: usize = 0;
        for (0..count) |i| {
            const candidate = if (candidates) |items| items[i] else Candidate{ .token = i, .logit = logits[i] };
            if (candidate.logit != candidate.logit) continue;
            if (seen == selected) return candidate.token;
            seen += 1;
        }
        unreachable;
    }

    // All-NaN logits are invalid model output, but retaining the former greedy
    // fallback keeps completion deterministic rather than introducing a new
    // runtime failure mode.
    const fallback = last_valid orelse return greedyToken(logits);
    var total: f64 = 0;
    for (0..count) |i| {
        const candidate = if (candidates) |items| items[i] else Candidate{ .token = i, .logit = logits[i] };
        if (candidate.logit != candidate.logit) continue;
        total += @exp((@as(f64, candidate.logit) - @as(f64, maximum)) / @as(f64, temperature));
    }

    const selected = prng.unitFloat() * total;
    var cumulative: f64 = 0;
    for (0..count) |i| {
        const candidate = if (candidates) |items| items[i] else Candidate{ .token = i, .logit = logits[i] };
        if (candidate.logit != candidate.logit) continue;
        cumulative += @exp((@as(f64, candidate.logit) - @as(f64, maximum)) / @as(f64, temperature));
        if (selected < cumulative) return candidate.token;
    }
    return fallback;
}

fn sampleToken(
    logits: []const f32,
    strategy: SamplingStrategy,
    temperature: f32,
    top_k: usize,
    prng: *SamplingPrng,
    candidate_scratch: []Candidate,
) usize {
    std.debug.assert(logits.len > 0);
    if (strategy == .greedy) return greedyToken(logits);
    if (top_k == 0 or top_k >= logits.len) {
        return sampleCandidates(logits, null, temperature, prng);
    }
    std.debug.assert(candidate_scratch.len == top_k);
    fillTopCandidates(logits, candidate_scratch);
    return sampleCandidates(logits, candidate_scratch, temperature, prng);
}

/// Serving boundary for bounded-residency completion on a single GGUF model.
///
/// The llama.cpp model handle provides the vocabulary/detokenizer only (mmap);
/// it is never used for compute. All weight compute runs through the
/// bounded-residency CPU executor with an explicit weight-mapping budget.
fn chunkStatesLen(chunk: usize, hidden: usize) usize {
    return chunk * hidden * @sizeOf(f32);
}

fn stateLen(hidden: usize) usize {
    return hidden * @sizeOf(f32);
}

fn logitsLen(vocab_size: usize) usize {
    return vocab_size * @sizeOf(f32);
}

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

    fn computeSupported(desc: *const gguf.TensorDescriptor) bool {
        return switch (desc.ggml_type) {
            gguf.type_f32, gguf.type_q4_0, gguf.type_q3_k, gguf.type_q4_k, gguf.type_q6_k, gguf.type_q2_k, gguf.type_mxfp4 => true,
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

    fn descriptor(self: *ResidencyService, layer: usize, suffix: []const u8) Error!*const gguf.TensorDescriptor {
        const name = try std.fmt.allocPrint(self.allocator, "blk.{d}.{s}", .{ layer, suffix });
        defer self.allocator.free(name);
        return self.index.get(name) orelse return Error.MissingMetadata;
    }

    fn qwenMoeWeights(self: *ResidencyService, layer: usize) Error!qwen.MoeWeights {
        return .{
            .post_attention_norm = try self.descriptor(layer, "post_attention_norm.weight"),
            .router = try self.descriptor(layer, "ffn_gate_inp.weight"),
            .gate_experts = try self.descriptor(layer, "ffn_gate_exps.weight"),
            .up_experts = try self.descriptor(layer, "ffn_up_exps.weight"),
            .down_experts = try self.descriptor(layer, "ffn_down_exps.weight"),
            .shared_router = try self.descriptor(layer, "ffn_gate_inp_shexp.weight"),
            .shared_gate = try self.descriptor(layer, "ffn_gate_shexp.weight"),
            .shared_up = try self.descriptor(layer, "ffn_up_shexp.weight"),
            .shared_down = try self.descriptor(layer, "ffn_down_shexp.weight"),
        };
    }

    fn qwenLayerWeights(self: *ResidencyService, layer: usize, full_attention_interval: usize) Error!qwen.LayerWeights {
        const moe = try self.qwenMoeWeights(layer);
        if ((layer + 1) % full_attention_interval == 0) {
            return .{ .full_attention = .{
                .attention = .{
                    .attention_norm = try self.descriptor(layer, "attn_norm.weight"),
                    .query_gate = try self.descriptor(layer, "attn_qkv.weight"),
                    .key = try self.descriptor(layer, "attn_k.weight"),
                    .value = try self.descriptor(layer, "attn_v.weight"),
                    .query_norm = try self.descriptor(layer, "attn_q_norm.weight"),
                    .key_norm = try self.descriptor(layer, "attn_k_norm.weight"),
                    .output = try self.descriptor(layer, "attn_output.weight"),
                },
                .moe = moe,
            } };
        }
        return .{ .recurrent = .{
            .attention = .{
                .attention_norm = try self.descriptor(layer, "attn_norm.weight"),
                .qkv = try self.descriptor(layer, "attn_qkv.weight"),
                .z_gate = try self.descriptor(layer, "attn_gate.weight"),
                .beta_alpha = try self.descriptor(layer, "ssm_ba.weight"),
                .conv1d = try self.descriptor(layer, "ssm_conv1d.weight"),
                .dt_bias = try self.descriptor(layer, "ssm_dt.bias"),
                .decay = try self.descriptor(layer, "ssm_a"),
                .state_norm = try self.descriptor(layer, "ssm_norm.weight"),
                .output = try self.descriptor(layer, "ssm_out.weight"),
            },
            .moe = moe,
        } };
    }

    fn validateQwenWeightTypes(layers: []const qwen.LayerWeights) Error!void {
        for (layers) |layer| switch (layer) {
            .recurrent => |weights| {
                const matrices = [_]*const gguf.TensorDescriptor{
                    weights.attention.qkv,    weights.attention.z_gate, weights.attention.beta_alpha,
                    weights.attention.output, weights.moe.router,       weights.moe.gate_experts,
                    weights.moe.up_experts,   weights.moe.down_experts, weights.moe.shared_gate,
                    weights.moe.shared_up,    weights.moe.shared_down,
                };
                for (matrices) |matrix| if (!computeSupported(matrix)) return Error.UnsupportedGguf;
            },
            .full_attention => |weights| {
                const matrices = [_]*const gguf.TensorDescriptor{
                    weights.attention.query_gate, weights.attention.key,    weights.attention.value,
                    weights.attention.output,     weights.moe.router,       weights.moe.gate_experts,
                    weights.moe.up_experts,       weights.moe.down_experts, weights.moe.shared_gate,
                    weights.moe.shared_up,        weights.moe.shared_down,
                };
                for (matrices) |matrix| if (!computeSupported(matrix)) return Error.UnsupportedGguf;
            },
        };
    }

    pub const TokenizeResult = struct {
        tokens: []usize,
    };

    pub const ChatMessage = struct {
        role: []const u8,
        content: []const u8,
    };

    /// Renders `messages` through the model's embedded jinja chat template and
    /// tokenizes the result with special tokens enabled (the template output
    /// may contain control tokens such as `<|eot_id|>`).
    pub fn applyChatTemplate(
        self: *ResidencyService,
        messages: []const ChatMessage,
    ) Error!TokenizeResult {
        if (messages.len == 0) return Error.PromptEmpty;
        const template = self.model.chatTemplate() orelse return Error.VocabUnavailable;

        const chat_msgs = try self.allocator.alloc(llama.c.llama_chat_message, messages.len);
        defer self.allocator.free(chat_msgs);
        for (messages, 0..) |m, i| {
            chat_msgs[i] = .{
                .role = m.role.ptr,
                .content = m.content.ptr,
            };
        }

        const formatted = try llama.applyChatTemplateJinja(self.allocator, template, chat_msgs, true);
        defer self.allocator.free(formatted);

        const tokens_c = try llama.tokenize(self.allocator, self.vocab, formatted, false, true);
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

    pub const TokenSink = struct {
        context: ?*anyopaque = null,
        /// Called synchronously for every generated token that detokenizes to a
        /// piece, before that piece is appended to the buffered result. `piece`
        /// borrows an internal buffer and is valid only for the callback.
        callback: *const fn (context: ?*anyopaque, token: usize, piece: []const u8) void,

        fn emit(self: TokenSink, token: usize, piece: []const u8) void {
            self.callback(self.context, token, piece);
        }
    };

    pub const CompletionOptions = struct {
        max_tokens: usize = 32,
        add_bos: bool = false,
        /// Greedy by default, preserving the original completion behavior.
        sampling: SamplingStrategy = .greedy,
        /// Softmax temperature used by `.temperature` sampling. Must be finite
        /// and greater than zero when temperature sampling is selected.
        temperature: f32 = 1.0,
        /// Restrict temperature sampling to the highest K logits. Zero means
        /// all logits; values at least as large as the vocabulary are also
        /// treated as unrestricted.
        top_k: usize = 0,
        /// Seed for deterministic temperature sampling.
        seed: u64 = 0,
        /// Optional synchronous stream of generated token pieces.
        token_sink: ?TokenSink = null,
        /// Pre-tokenized prompt; skips `tokenize()` when provided.
        prompt_tokens: ?[]const usize = null,
        /// Explicit non-weight memory policy. `null` keeps legacy unlimited
        /// allocation for KV caches and execution workspaces.
        state_budget: ?executor_mod.StateBudget = null,
    };

    /// Runs a completion through the architecture-specific bounded-residency
    /// executor selected from GGUF metadata.
    pub fn complete(
        self: *ResidencyService,
        config: Config,
        options: CompletionOptions,
    ) Error!CompletionResult {
        return switch (self.index.execution.architecture) {
            .llama => self.completeLlama(config, options),
            .qwen3next => self.completeQwen(config, options),
            .unknown => Error.UnsupportedGguf,
        };
    }

    fn completeLlama(
        self: *ResidencyService,
        config: Config,
        options: CompletionOptions,
    ) Error!CompletionResult {
        if (options.sampling == .temperature and
            (!(options.temperature > 0.0) or !std.math.isFinite(options.temperature)))
        {
            return Error.InvalidSamplingOptions;
        }

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
        for (self.index.descriptors) |desc| {
            if (std.mem.startsWith(u8, desc.name, "blk.") and
                std.mem.endsWith(u8, desc.name, ".ffn_gate.weight"))
            {
                intermediate = @max(intermediate, std.math.cast(usize, desc.dimensions[1]) orelse return Error.InvalidExecutionShape);
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
        const candidate_count = if (options.sampling == .temperature and
            options.top_k > 0 and options.top_k < self.vocab_size)
            options.top_k
        else
            0;
        const candidate_bytes = std.math.mul(usize, candidate_count, @sizeOf(Candidate)) catch
            return Error.InvalidExecutionShape;

        // Non-weight state budget policy: validated BEFORE any workspace,
        // cache, or manager allocation so rejection is transactional.
        const state_budget = options.state_budget orelse executor_mod.StateBudget{};
        {
            const attention_bytes = try executor_mod.attentionWorkspaceBytes(hidden);
            const prefill_bytes = try executor_mod.prefillWorkspaceBytes(config.prefill_chunk, hidden, intermediate);
            const workspace_bytes = attention_bytes + prefill_bytes +
                chunkStatesLen(config.prefill_chunk, hidden) + stateLen(hidden) +
                logitsLen(self.vocab_size) + candidate_bytes;
            try state_budget.checkWorkspace(workspace_bytes);
            const cache_total = (try executor_mod.kvCacheBytes(config.context_capacity, kv_width)) * block_count;
            try state_budget.checkCache(cache_total);
        }

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

        // Non-weight state budget policy: pre-validated above; the cache
        // allocation path re-checks the policy before touching memory.
        const caches = try executor_mod.initKvCachesBudgeted(
            self.allocator,
            block_count,
            config.context_capacity,
            kv_width,
            state_budget,
        );
        defer {
            for (caches) |*cache| cache.deinit();
            self.allocator.free(caches);
        }

        const chunk_states = try self.allocator.alloc(f32, config.prefill_chunk * hidden);
        defer self.allocator.free(chunk_states);
        const state = try self.allocator.alloc(f32, hidden);
        defer self.allocator.free(state);
        const logits = try self.allocator.alloc(f32, self.vocab_size);
        defer self.allocator.free(logits);
        const candidates = try self.allocator.alloc(Candidate, candidate_count);
        defer self.allocator.free(candidates);
        var sampling_prng = SamplingPrng.init(options.seed);

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
            const next_token = sampleToken(
                logits,
                options.sampling,
                options.temperature,
                options.top_k,
                &sampling_prng,
                candidates,
            );
            if (next_token == self.eos_token) break;
            try generated.append(self.allocator, next_token);

            if (self.tokenPiece(next_token, &piece_buffer)) |piece| {
                if (options.token_sink) |sink| sink.emit(next_token, piece);
                try text.appendSlice(self.allocator, piece);
            }

            const next_position = caches[0].len;
            if (next_position >= config.context_capacity) break;
            const single = [_]usize{next_token};
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
        const owned_text = try text.toOwnedSlice(self.allocator);
        errdefer self.allocator.free(owned_text);
        const owned_tokens = try generated.toOwnedSlice(self.allocator);
        return .{
            .text = owned_text,
            .tokens = owned_tokens,
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

    fn completeQwen(
        self: *ResidencyService,
        service_config: Config,
        options: CompletionOptions,
    ) Error!CompletionResult {
        if (options.sampling == .temperature and
            (!(options.temperature > 0.0) or !std.math.isFinite(options.temperature)))
        {
            return Error.InvalidSamplingOptions;
        }

        const model_config = try qwen.Config.fromMetadata(self.index.execution);
        const block_count: usize = self.index.execution.block_count orelse return Error.MissingMetadata;
        const full_attention_interval: usize = self.index.execution.full_attention_interval orelse return Error.MissingMetadata;
        if (block_count == 0 or full_attention_interval == 0) return Error.MissingMetadata;

        const embedding = self.index.get("token_embd.weight") orelse return Error.MissingMetadata;
        const output_norm = self.index.get("output_norm.weight") orelse return Error.MissingMetadata;
        const output_weight = self.index.get("output.weight") orelse embedding;
        if (!computeSupported(embedding) or !computeSupported(output_weight)) return Error.UnsupportedGguf;

        const prompt_tokens: []const usize = options.prompt_tokens orelse &[_]usize{};
        if (prompt_tokens.len == 0) return Error.PromptEmpty;
        for (prompt_tokens) |token| if (token >= self.vocab_size) return Error.InvalidToken;
        const generation_capacity = std.math.add(usize, prompt_tokens.len, options.max_tokens) catch return Error.KvCacheFull;
        if (generation_capacity > service_config.context_capacity) return Error.KvCacheFull;

        const layers = try self.allocator.alloc(qwen.LayerWeights, block_count);
        defer self.allocator.free(layers);
        for (layers, 0..) |*layer, index| layer.* = try self.qwenLayerWeights(index, full_attention_interval);
        try validateQwenWeightTypes(layers);

        const candidate_count = if (options.sampling == .temperature and options.top_k > 0 and options.top_k < self.vocab_size)
            options.top_k
        else
            0;
        const candidate_bytes = std.math.mul(usize, candidate_count, @sizeOf(Candidate)) catch return Error.InvalidExecutionShape;
        const state_bytes = std.math.mul(usize, model_config.hidden_size, @sizeOf(f32)) catch return Error.InvalidExecutionShape;
        const logits_bytes = std.math.mul(usize, self.vocab_size, @sizeOf(f32)) catch return Error.InvalidExecutionShape;
        const auxiliary_bytes = std.math.add(usize, state_bytes, std.math.add(usize, logits_bytes, candidate_bytes) catch return Error.InvalidExecutionShape) catch return Error.InvalidExecutionShape;

        // CompletionOptions retains the public executor budget type. Convert it
        // to Qwen's policy, reserving request-owned state/logit/sampling arrays
        // from the workspace allowance before any state allocation occurs.
        const requested_budget = options.state_budget orelse executor_mod.StateBudget{};
        var qwen_budget = qwen.StateBudget{
            .cache_bytes = requested_budget.cache_bytes,
            .workspace_bytes = null,
        };
        if (requested_budget.workspace_bytes) |limit| {
            if (auxiliary_bytes > limit) return Error.StateBudgetExceeded;
            qwen_budget.workspace_bytes = limit - auxiliary_bytes;
        }
        try qwen_budget.validatePrefill(
            model_config,
            block_count,
            full_attention_interval,
            service_config.context_capacity,
            service_config.prefill_chunk,
        );

        const expert_ff: usize = self.index.execution.expert_feed_forward_length orelse return Error.MissingMetadata;
        const shared_ff: usize = self.index.execution.shared_expert_feed_forward_length orelse return Error.MissingMetadata;
        if (expert_ff == 0 or shared_ff == 0) return Error.MissingMetadata;
        const max_projection = @max(model_config.hidden_size, @max(model_config.inner_size, @max(expert_ff, shared_ff)));
        const activation_capacity = @max(model_config.expert_count, @max(expert_ff, shared_ff));

        var manager = try residency.Manager.init(self.allocator, &self.store, service_config.budget_bytes);
        defer manager.deinit();
        try self.index.registerAll(&manager);

        var executor = try executor_mod.CpuExecutor.init(
            self.allocator,
            &manager,
            max_projection,
            activation_capacity,
            activation_capacity,
        );
        defer executor.deinit();

        var prefill_workspace = try qwen.PrefillWorkspace.initBudgeted(
            self.allocator,
            model_config,
            service_config.prefill_chunk,
            qwen_budget,
        );
        defer prefill_workspace.deinit();
        const caches = try qwen.initLayerCachesBudgeted(
            self.allocator,
            model_config,
            block_count,
            full_attention_interval,
            service_config.context_capacity,
            qwen_budget,
        );
        defer qwen.deinitLayerCaches(self.allocator, caches);

        const state = try self.allocator.alloc(f32, model_config.hidden_size);
        defer self.allocator.free(state);
        const logits = try self.allocator.alloc(f32, self.vocab_size);
        defer self.allocator.free(logits);
        const candidates = try self.allocator.alloc(Candidate, candidate_count);
        defer self.allocator.free(candidates);
        var sampling_prng = SamplingPrng.init(options.seed);

        var timer = try std.time.Timer.start();
        try qwen.modelPrefillChunked(
            &executor,
            model_config,
            embedding,
            layers,
            output_norm,
            output_weight,
            prompt_tokens,
            caches,
            &prefill_workspace,
            logits,
        );

        var generated: std.ArrayList(usize) = .empty;
        errdefer generated.deinit(self.allocator);
        var text: std.ArrayList(u8) = .empty;
        errdefer text.deinit(self.allocator);
        var piece_buffer: [256]u8 = undefined;

        var total_tokens: usize = 0;
        while (total_tokens < options.max_tokens) {
            const next_token = sampleToken(
                logits,
                options.sampling,
                options.temperature,
                options.top_k,
                &sampling_prng,
                candidates,
            );
            if (next_token == self.eos_token) break;
            try generated.append(self.allocator, next_token);
            total_tokens += 1;

            if (self.tokenPiece(next_token, &piece_buffer)) |piece| {
                if (options.token_sink) |sink| sink.emit(next_token, piece);
                try text.appendSlice(self.allocator, piece);
            }

            // The final emitted token needs no forward pass. Otherwise advance
            // both recurrent and full-attention caches for the next sample.
            if (total_tokens == options.max_tokens) break;
            try qwen.modelSingleToken(
                &executor,
                model_config,
                embedding,
                layers,
                output_norm,
                output_weight,
                next_token,
                caches,
                &prefill_workspace.token,
                state,
                logits,
            );
        }
        const elapsed_ms = @as(f64, @floatFromInt(timer.read())) / 1.0e6;

        var cache_bytes: usize = 0;
        for (caches) |*cache| cache_bytes = std.math.add(usize, cache_bytes, cache.byteLen()) catch return Error.InvalidExecutionShape;
        const accounting = executor.accounting();
        const owned_text = try text.toOwnedSlice(self.allocator);
        errdefer self.allocator.free(owned_text);
        const owned_tokens = try generated.toOwnedSlice(self.allocator);
        return .{
            .text = owned_text,
            .tokens = owned_tokens,
            .prompt_tokens = prompt_tokens.len,
            .elapsed_ms = elapsed_ms,
            .weight_budget_bytes = accounting.weight_budget_bytes,
            .peak_mapped_weight_bytes = accounting.peak_mapped_weight_bytes,
            .dequant_scratch_bytes = accounting.dequant_scratch_bytes,
            .activation_bytes = accounting.activation_bytes,
            .attention_workspace_bytes = prefill_workspace.byteLen(),
            .kv_cache_bytes = cache_bytes,
            .faults = accounting.faults,
            .hits = accounting.hits,
            .evictions = accounting.evictions,
            .rss_bytes = residency.currentRss(),
        };
    }
};

test "greedy sampling preserves argmax behavior" {
    const logits = [_]f32{ -2.0, 4.0, 4.0, 1.0 };
    var prng = SamplingPrng.init(123);
    var scratch: [0]Candidate = .{};

    // The original argmax selected the first token when logits tied.
    try std.testing.expectEqual(@as(usize, 1), sampleToken(
        &logits,
        .greedy,
        1.0,
        0,
        &prng,
        &scratch,
    ));
}

test "temperature sampling is deterministic for a seed" {
    const logits = [_]f32{ 0.1, 0.2, 0.3, 0.4 };
    var first = SamplingPrng.init(0x1234_5678);
    var second = SamplingPrng.init(0x1234_5678);
    var first_scratch: [3]Candidate = undefined;
    var second_scratch: [3]Candidate = undefined;

    for (0..32) |_| {
        const first_token = sampleToken(&logits, .temperature, 0.8, 3, &first, &first_scratch);
        const second_token = sampleToken(&logits, .temperature, 0.8, 3, &second, &second_scratch);
        try std.testing.expectEqual(first_token, second_token);
        try std.testing.expect(first_token != 0);
    }
}

test "top-k one always selects highest logit" {
    const logits = [_]f32{ 8.0, -1.0, 9.0, 3.0 };
    var prng = SamplingPrng.init(99);
    var scratch: [1]Candidate = undefined;

    for (0..16) |_| {
        try std.testing.expectEqual(@as(usize, 2), sampleToken(
            &logits,
            .temperature,
            100.0,
            1,
            &prng,
            &scratch,
        ));
    }
}

test "token sink receives token and borrowed piece" {
    const Capture = struct {
        calls: usize = 0,
        token: usize = 0,
        piece_len: usize = 0,
        first_byte: u8 = 0,

        fn callback(context: ?*anyopaque, token: usize, piece: []const u8) void {
            const self: *@This() = @ptrCast(@alignCast(context.?));
            self.calls += 1;
            self.token = token;
            self.piece_len = piece.len;
            self.first_byte = piece[0];
        }
    };

    var capture: Capture = .{};
    const sink = ResidencyService.TokenSink{
        .context = &capture,
        .callback = Capture.callback,
    };
    sink.emit(42, "piece");

    try std.testing.expectEqual(@as(usize, 1), capture.calls);
    try std.testing.expectEqual(@as(usize, 42), capture.token);
    try std.testing.expectEqual(@as(usize, 5), capture.piece_len);
    try std.testing.expectEqual(@as(u8, 'p'), capture.first_byte);
}
