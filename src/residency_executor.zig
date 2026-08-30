const std = @import("std");
const residency = @import("residency.zig");
const gguf = @import("gguf_residency.zig");
const compute = @import("residency_compute.zig");

pub const Error = compute.Error || error{
    InvalidExecutionShape,
    ScratchCapacityExceeded,
    ActivationCapacityExceeded,
    KvCacheFull,
    InvalidPosition,
    InvalidToken,
    TooManyExperts,
    OutOfMemory,
};

pub const AttentionConfig = struct {
    head_count: usize,
    kv_head_count: usize,
    head_dim: usize,
    rms_epsilon: f32 = 1e-5,
    rope_theta: f32 = 10_000.0,
};

/// Caller-visible KV storage. Unlike mapped immutable weights, this memory is
/// writable and has an independent, explicit lifetime and byte count.
pub const KvCache = struct {
    allocator: std.mem.Allocator,
    keys: []f32,
    values: []f32,
    scores: []f32,
    capacity: usize,
    kv_width: usize,
    len: usize = 0,

    pub fn init(allocator: std.mem.Allocator, capacity: usize, kv_width: usize) Error!KvCache {
        if (capacity == 0 or kv_width == 0) return Error.InvalidExecutionShape;
        const elements = std.math.mul(usize, capacity, kv_width) catch return Error.OutOfMemory;
        const keys = allocator.alloc(f32, elements) catch return Error.OutOfMemory;
        errdefer allocator.free(keys);
        const values = allocator.alloc(f32, elements) catch return Error.OutOfMemory;
        errdefer allocator.free(values);
        const scores = allocator.alloc(f32, capacity) catch return Error.OutOfMemory;
        return .{ .allocator = allocator, .keys = keys, .values = values, .scores = scores, .capacity = capacity, .kv_width = kv_width };
    }

    pub fn deinit(self: *KvCache) void {
        self.allocator.free(self.scores);
        self.allocator.free(self.values);
        self.allocator.free(self.keys);
        self.* = undefined;
    }

    pub fn byteLen(self: *const KvCache) usize {
        return (self.keys.len + self.values.len + self.scores.len) * @sizeOf(f32);
    }
};

/// Memory owned or controlled by one CPU execution adapter. Weight mapping
/// numbers come from the residency manager; scratch and activation numbers are
/// reported separately so the mmap budget is not mistaken for a total-memory
/// budget.
pub const MemoryAccounting = struct {
    weight_budget_bytes: usize,
    current_mapped_weight_bytes: usize,
    peak_mapped_weight_bytes: usize,
    dequant_scratch_bytes: usize,
    batch_scratch_bytes: usize,
    activation_bytes: usize,
    faults: u64,
    hits: u64,
    evictions: u64,
};

pub const AttentionWorkspace = struct {
    allocator: std.mem.Allocator,
    query: []f32,
    context: []f32,

    pub fn init(allocator: std.mem.Allocator, hidden_size: usize) Error!AttentionWorkspace {
        if (hidden_size == 0) return Error.InvalidExecutionShape;
        const query = allocator.alloc(f32, hidden_size) catch return Error.OutOfMemory;
        errdefer allocator.free(query);
        const context = allocator.alloc(f32, hidden_size) catch return Error.OutOfMemory;
        return .{ .allocator = allocator, .query = query, .context = context };
    }

    pub fn deinit(self: *AttentionWorkspace) void {
        self.allocator.free(self.context);
        self.allocator.free(self.query);
        self.* = undefined;
    }

    pub fn byteLen(self: *const AttentionWorkspace) usize {
        return (self.query.len + self.context.len) * @sizeOf(f32);
    }
};

pub const PrefillWorkspace = struct {
    allocator: std.mem.Allocator,
    capacity: usize,
    hidden: usize,
    intermediate: usize,
    normalized: []f32,
    query: []f32,
    context: []f32,
    gate: []f32,
    up: []f32,

    pub fn init(allocator: std.mem.Allocator, capacity: usize, hidden: usize, intermediate: usize) Error!PrefillWorkspace {
        if (capacity == 0 or hidden == 0 or intermediate == 0) return Error.InvalidExecutionShape;
        const hidden_elements = std.math.mul(usize, capacity, hidden) catch return Error.InvalidExecutionShape;
        const intermediate_elements = std.math.mul(usize, capacity, intermediate) catch return Error.InvalidExecutionShape;
        const normalized = allocator.alloc(f32, hidden_elements) catch return Error.OutOfMemory;
        errdefer allocator.free(normalized);
        const query = allocator.alloc(f32, hidden_elements) catch return Error.OutOfMemory;
        errdefer allocator.free(query);
        const context = allocator.alloc(f32, hidden_elements) catch return Error.OutOfMemory;
        errdefer allocator.free(context);
        const gate = allocator.alloc(f32, intermediate_elements) catch return Error.OutOfMemory;
        errdefer allocator.free(gate);
        const up = allocator.alloc(f32, intermediate_elements) catch return Error.OutOfMemory;
        return .{
            .allocator = allocator,
            .capacity = capacity,
            .hidden = hidden,
            .intermediate = intermediate,
            .normalized = normalized,
            .query = query,
            .context = context,
            .gate = gate,
            .up = up,
        };
    }

    pub fn deinit(self: *PrefillWorkspace) void {
        self.allocator.free(self.up);
        self.allocator.free(self.gate);
        self.allocator.free(self.context);
        self.allocator.free(self.query);
        self.allocator.free(self.normalized);
        self.* = undefined;
    }

    pub fn byteLen(self: *const PrefillWorkspace) usize {
        return (self.normalized.len + self.query.len + self.context.len + self.gate.len + self.up.len) * @sizeOf(f32);
    }
};

pub const DecoderLayerWeights = struct {
    attention_norm: *const gguf.TensorDescriptor,
    query: *const gguf.TensorDescriptor,
    key: *const gguf.TensorDescriptor,
    value: *const gguf.TensorDescriptor,
    attention_output: *const gguf.TensorDescriptor,
    ffn_norm: *const gguf.TensorDescriptor,
    ffn_gate: *const gguf.TensorDescriptor,
    ffn_up: *const gguf.TensorDescriptor,
    ffn_down: *const gguf.TensorDescriptor,
};

pub const DecoderMemoryAccounting = struct {
    executor: MemoryAccounting,
    attention_workspace_bytes: usize,
    kv_cache_bytes: usize,
};

/// CPU operation boundary for bounded GGUF weights.
///
/// Every matrix operation delegates to a tiled compute function that acquires
/// and pins a TensorView only while its kernel is reading that tile. No mapped
/// weight pointer is stored in this object or returned to its caller.
const batch_scratch_alignment = std.mem.Alignment.fromByteUnits(64);

pub const CpuExecutor = struct {
    allocator: std.mem.Allocator,
    manager: *residency.Manager,
    tile_policy: compute.TilePolicy,
    dequant_scratch: []f32,
    batch_scratch: []align(64) u8,
    activation_a: []f32,
    activation_b: []f32,

    pub fn init(
        allocator: std.mem.Allocator,
        manager: *residency.Manager,
        max_input_elements: usize,
        max_intermediate_elements: usize,
        rows_per_tile: usize,
    ) Error!CpuExecutor {
        if (max_input_elements == 0 or rows_per_tile == 0) return Error.InvalidExecutionShape;

        const dequant_scratch = allocator.alloc(f32, max_input_elements) catch return Error.OutOfMemory;
        errdefer allocator.free(dequant_scratch);
        const activation_a = allocator.alloc(f32, max_intermediate_elements) catch return Error.OutOfMemory;
        errdefer allocator.free(activation_a);
        const activation_b = allocator.alloc(f32, max_intermediate_elements) catch return Error.OutOfMemory;
        errdefer allocator.free(activation_b);
        const batch_scratch = allocator.alignedAlloc(u8, batch_scratch_alignment, 0) catch return Error.OutOfMemory;

        return .{
            .allocator = allocator,
            .manager = manager,
            .tile_policy = .{ .fixed_rows = rows_per_tile },
            .dequant_scratch = dequant_scratch,
            .batch_scratch = batch_scratch,
            .activation_a = activation_a,
            .activation_b = activation_b,
        };
    }

    pub fn setTilePolicy(self: *CpuExecutor, policy: compute.TilePolicy) Error!void {
        switch (policy) {
            .fixed_rows => |rows| if (rows == 0) return Error.InvalidExecutionShape,
            .adaptive => |options| if (options.max_rows == 0) return Error.InvalidExecutionShape,
        }
        self.tile_policy = policy;
    }

    pub fn deinit(self: *CpuExecutor) void {
        self.allocator.free(self.batch_scratch);
        self.allocator.free(self.activation_b);
        self.allocator.free(self.activation_a);
        self.allocator.free(self.dequant_scratch);
        self.* = undefined;
    }

    /// Executes one GGUF matrix-vector operation without exposing a weight
    /// pointer outside the pinned tile lifetime.
    pub fn matVec(
        self: *CpuExecutor,
        descriptor: *const gguf.TensorDescriptor,
        input: []const f32,
        output: []f32,
    ) Error!void {
        if (descriptor.n_dimensions != 2) return Error.InvalidExecutionShape;
        const columns = std.math.cast(usize, descriptor.dimensions[0]) orelse return Error.InvalidExecutionShape;
        if (columns > self.dequant_scratch.len) return Error.ScratchCapacityExceeded;

        if (descriptor.ggml_type == gguf.type_f32) {
            try compute.matVecF32WithPolicy(self.manager, descriptor, input, output, self.tile_policy);
        } else {
            // Match ggml_mul_mat arithmetic: quantize the activation to the
            // weight format's vec_dot type, then invoke GGML's canonical dot
            // kernel while each weight tile is pinned. Dequantize-then-dot is
            // useful as a simple reference, but does not reproduce llama.cpp
            // logits because GGML quantizes the activation for quantized
            // matrix multiplication.
            try compute.matVecQuantizedGgmlWithPolicy(
                self.manager,
                descriptor,
                input,
                output,
                self.tile_policy,
                std.mem.sliceAsBytes(self.dequant_scratch),
            );
        }
    }

    fn ensureBatchScratch(self: *CpuExecutor, descriptor: *const gguf.TensorDescriptor, batch_count: usize) Error!void {
        if (descriptor.ggml_type == gguf.type_f32) return;
        const columns = std.math.cast(usize, descriptor.dimensions[0]) orelse return Error.InvalidExecutionShape;
        const required = try compute.quantizedDotBatchScratchBytes(descriptor.ggml_type, columns, batch_count);
        if (required > self.batch_scratch.len) {
            const grown = self.allocator.alignedAlloc(u8, batch_scratch_alignment, required) catch return Error.OutOfMemory;
            self.allocator.free(self.batch_scratch);
            self.batch_scratch = grown;
        }
    }

    fn validateMatrix(descriptor: *const gguf.TensorDescriptor, columns: usize, rows: usize) Error!void {
        const shape = try matrixShape(descriptor);
        if (shape.columns != columns or shape.rows != rows) return Error.InvalidExecutionShape;
        if (descriptor.ggml_type != gguf.type_f32) _ = try compute.quantizedDotScratchBytes(descriptor.ggml_type, columns);
    }

    fn validateNorm(descriptor: *const gguf.TensorDescriptor, elements: usize) Error!void {
        const bytes = std.math.mul(usize, elements, @sizeOf(f32)) catch return Error.InvalidExecutionShape;
        if (descriptor.ggml_type != gguf.type_f32 or descriptor.n_dimensions != 1 or
            descriptor.dimensions[0] != elements or descriptor.byte_len != bytes)
        {
            return Error.InvalidExecutionShape;
        }
    }

    /// Executes a batch of activation rows while each mapped weight tile is
    /// pinned once. Inputs are [batch, columns], outputs are [batch, rows].
    /// Quantized activation scratch is reusable and grows only to the largest
    /// batch submitted to this executor.
    pub fn matMul(
        self: *CpuExecutor,
        descriptor: *const gguf.TensorDescriptor,
        inputs: []const f32,
        outputs: []f32,
        batch_count: usize,
    ) Error!void {
        if (descriptor.n_dimensions != 2 or batch_count == 0) return Error.InvalidExecutionShape;
        if (descriptor.ggml_type == gguf.type_f32) {
            return compute.matMulF32WithPolicy(
                self.manager,
                descriptor,
                inputs,
                outputs,
                batch_count,
                self.tile_policy,
            );
        }

        try self.ensureBatchScratch(descriptor, batch_count);
        return compute.matMulQuantizedGgmlWithPolicy(
            self.manager,
            descriptor,
            inputs,
            outputs,
            batch_count,
            self.tile_policy,
            self.batch_scratch,
        );
    }

    /// Executes one expert matrix selected from a 3D GGML tensor. Dimension 2
    /// is the expert index and only that matrix slice enters residency.
    pub fn matVecExpert(
        self: *CpuExecutor,
        descriptor: *const gguf.TensorDescriptor,
        expert: usize,
        input: []const f32,
        output: []f32,
    ) Error!void {
        if (descriptor.n_dimensions != 3) return Error.InvalidExecutionShape;
        const columns = std.math.cast(usize, descriptor.dimensions[0]) orelse return Error.InvalidExecutionShape;
        if (columns > self.dequant_scratch.len) return Error.ScratchCapacityExceeded;
        try compute.matVecQuantizedSliceWithPolicy(
            self.manager,
            descriptor,
            expert,
            input,
            output,
            self.tile_policy,
            self.dequant_scratch,
        );
    }

    /// Qwen3-Next routed MoE branch for one token. Router logits are softmaxed
    /// over all experts; only top-k expert matrix slices are faulted. The gate
    /// and up projections share executor activations, then the down projection
    /// is accumulated into `output` using normalized top-k probabilities.
    pub fn moeSwiGlu(
        self: *CpuExecutor,
        router: *const gguf.TensorDescriptor,
        gate_experts: *const gguf.TensorDescriptor,
        up_experts: *const gguf.TensorDescriptor,
        down_experts: *const gguf.TensorDescriptor,
        expert_used_count: usize,
        input: []const f32,
        output: []f32,
    ) Error!void {
        const expert_count = std.math.cast(usize, gate_experts.dimensions[2]) orelse return Error.InvalidExecutionShape;
        const intermediate = std.math.cast(usize, gate_experts.dimensions[1]) orelse return Error.InvalidExecutionShape;
        if (expert_count == 0 or expert_used_count == 0 or expert_used_count > expert_count or
            router.n_dimensions != 2 or router.dimensions[0] != input.len or router.dimensions[1] != expert_count or
            up_experts.n_dimensions != 3 or down_experts.n_dimensions != 3 or
            up_experts.dimensions[0] != input.len or up_experts.dimensions[1] != intermediate or up_experts.dimensions[2] != expert_count or
            down_experts.dimensions[0] != intermediate or down_experts.dimensions[1] != output.len or down_experts.dimensions[2] != expert_count or
            expert_count > self.activation_a.len or intermediate > self.activation_a.len or intermediate > self.activation_b.len or
            output.len > self.dequant_scratch.len)
        {
            return Error.InvalidExecutionShape;
        }

        const router_logits = self.activation_a[0..expert_count];
        try self.matVec(router, input, router_logits);
        const selected = self.allocator.alloc(usize, expert_used_count) catch return Error.OutOfMemory;
        defer self.allocator.free(selected);
        const probabilities = self.allocator.alloc(f32, expert_used_count) catch return Error.OutOfMemory;
        defer self.allocator.free(probabilities);
        const expert_output = self.allocator.alloc(f32, output.len) catch return Error.OutOfMemory;
        defer self.allocator.free(expert_output);

        // Selection is deterministic for equal values: the lower expert index wins.
        for (selected, 0..) |*slot, rank| {
            var best: ?usize = null;
            for (router_logits, 0..) |logit, candidate| {
                var already_selected = false;
                for (selected[0..rank]) |prior| if (prior == candidate) {
                    already_selected = true;
                    break;
                };
                if (already_selected) continue;
                if (best == null or logit > router_logits[best.?]) best = candidate;
            }
            slot.* = best orelse return Error.TooManyExperts;
        }
        var maximum: f32 = -std.math.inf(f32);
        for (selected) |expert| maximum = @max(maximum, router_logits[expert]);
        var denominator: f32 = 0;
        for (selected, probabilities) |expert, *probability| {
            probability.* = @exp(router_logits[expert] - maximum);
            denominator += probability.*;
        }
        for (probabilities) |*probability| probability.* /= denominator;

        @memset(output, 0);
        for (selected, probabilities) |expert, probability| {
            const gate_values = self.activation_a[0..intermediate];
            const up_values = self.activation_b[0..intermediate];
            try self.matVecExpert(gate_experts, expert, input, gate_values);
            try self.matVecExpert(up_experts, expert, input, up_values);
            for (gate_values, up_values) |*gate_value, up_value| {
                gate_value.* = gate_value.* / (1.0 + @exp(-gate_value.*)) * up_value;
            }
            // Router scores are no longer needed after top-k selection.
            try self.matVecExpert(down_experts, expert, gate_values, expert_output);
            for (output, expert_output) |*value, expert_value| value.* += probability * expert_value;
        }
    }

    /// Executes the weight-bearing part of a Llama-style SwiGLU FFN:
    ///
    ///     output = down(silu(gate(input)) * up(input))
    ///
    /// Gate/up intermediates are executor-owned and explicitly accounted. Each
    /// matrix operation releases all pinned weight views before the next one.
    pub fn ffnSwiGlu(
        self: *CpuExecutor,
        gate: *const gguf.TensorDescriptor,
        up: *const gguf.TensorDescriptor,
        down: *const gguf.TensorDescriptor,
        input: []const f32,
        output: []f32,
    ) Error!void {
        const gate_shape = try matrixShape(gate);
        const up_shape = try matrixShape(up);
        const down_shape = try matrixShape(down);
        if (gate_shape.columns != input.len or
            up_shape.columns != input.len or
            gate_shape.rows != up_shape.rows or
            down_shape.columns != gate_shape.rows or
            down_shape.rows != output.len)
        {
            return Error.InvalidExecutionShape;
        }
        if (gate_shape.rows > self.activation_a.len or gate_shape.rows > self.activation_b.len) {
            return Error.ActivationCapacityExceeded;
        }

        const gate_values = self.activation_a[0..gate_shape.rows];
        const up_values = self.activation_b[0..gate_shape.rows];
        try self.matVec(gate, input, gate_values);
        try self.matVec(up, input, up_values);

        for (gate_values, up_values) |*gate_value, up_value| {
            gate_value.* = silu(gate_value.*) * up_value;
        }
        try self.matVec(down, gate_values, output);
    }

    /// Reads exactly one token embedding row. Quantized embeddings are
    /// dequantized directly into caller-owned state while only that row is
    /// pinned; no vocabulary-sized allocation is needed.
    pub fn tokenEmbedding(
        self: *CpuExecutor,
        embedding: *const gguf.TensorDescriptor,
        token: usize,
        state: []f32,
    ) Error!void {
        if (embedding.n_dimensions != 2 or embedding.dimensions[0] != state.len or
            token >= embedding.dimensions[1])
        {
            return Error.InvalidToken;
        }
        try compute.readMatrixRow(self.manager, embedding, token, state);
    }

    /// Applies an F32 RMSNorm weight vector directly from a pinned residency
    /// view. The weight pointer cannot escape this call.
    pub fn rmsNorm(
        self: *CpuExecutor,
        weight: *const gguf.TensorDescriptor,
        input: []const f32,
        output: []f32,
        epsilon: f32,
    ) Error!void {
        if (weight.ggml_type != gguf.type_f32 or weight.n_dimensions != 1 or
            weight.dimensions[0] != input.len or output.len != input.len or
            weight.byte_len != input.len * @sizeOf(f32) or epsilon <= 0)
        {
            return Error.InvalidExecutionShape;
        }
        var view = try self.manager.acquire(weight.handle);
        defer view.release();
        const values: []align(1) const f32 = std.mem.bytesAsSlice(f32, view.bytes());

        // Match GGML exactly: each square is rounded as F32 first, then the
        // products are accumulated in `ggml_float` (f64). Squaring in f64
        // changes the normalization scale slightly and compounds per layer.
        var sum_squares: f64 = 0;
        for (input) |value| sum_squares += @as(f64, value * value);
        const mean: f32 = @floatCast(sum_squares / @as(f64, @floatFromInt(input.len)));
        const scale = 1.0 / @sqrt(mean + epsilon);
        for (output, input, values) |*result, value, norm_weight| result.* = value * scale * norm_weight;
    }

    pub fn rmsNormBatch(
        self: *CpuExecutor,
        weight: *const gguf.TensorDescriptor,
        inputs: []const f32,
        outputs: []f32,
        batch_count: usize,
        epsilon: f32,
    ) Error!void {
        if (batch_count == 0 or inputs.len != outputs.len or inputs.len % batch_count != 0) return Error.InvalidExecutionShape;
        const hidden = inputs.len / batch_count;
        if (weight.ggml_type != gguf.type_f32 or weight.n_dimensions != 1 or
            weight.dimensions[0] != hidden or weight.byte_len != hidden * @sizeOf(f32) or epsilon <= 0)
        {
            return Error.InvalidExecutionShape;
        }
        var view = try self.manager.acquire(weight.handle);
        defer view.release();
        const values: []align(1) const f32 = std.mem.bytesAsSlice(f32, view.bytes());
        for (0..batch_count) |batch| {
            const input = inputs[batch * hidden ..][0..hidden];
            const output = outputs[batch * hidden ..][0..hidden];
            var sum_squares: f64 = 0;
            for (input) |value| sum_squares += @as(f64, value * value);
            const mean: f32 = @floatCast(sum_squares / @as(f64, @floatFromInt(hidden)));
            const scale = 1.0 / @sqrt(mean + epsilon);
            for (output, input, values) |*result, value, norm_weight| result.* = value * scale * norm_weight;
        }
    }

    /// Runs causal, single-token grouped-query attention. Q/K/V/O weights use
    /// the same bounded matvec boundary; writable K/V state lives in `cache`
    /// and is accounted independently from immutable mapped weights.
    pub fn attentionSingleToken(
        self: *CpuExecutor,
        query_weight: *const gguf.TensorDescriptor,
        key_weight: *const gguf.TensorDescriptor,
        value_weight: *const gguf.TensorDescriptor,
        output_weight: *const gguf.TensorDescriptor,
        normalized_input: []const f32,
        position: usize,
        config: AttentionConfig,
        cache: *KvCache,
        workspace: *AttentionWorkspace,
        output: []f32,
    ) Error!void {
        const hidden = std.math.mul(usize, config.head_count, config.head_dim) catch return Error.InvalidExecutionShape;
        const kv_width = std.math.mul(usize, config.kv_head_count, config.head_dim) catch return Error.InvalidExecutionShape;
        if (config.head_count == 0 or config.kv_head_count == 0 or config.head_dim == 0 or
            config.head_count % config.kv_head_count != 0 or config.head_dim % 2 != 0 or
            config.rope_theta <= 0 or normalized_input.len != hidden or output.len != hidden or
            workspace.query.len < hidden or workspace.context.len < hidden or
            cache.kv_width != kv_width or position != cache.len or position >= cache.capacity)
        {
            return if (position >= cache.capacity) Error.KvCacheFull else if (position != cache.len) Error.InvalidPosition else Error.InvalidExecutionShape;
        }
        const q_shape = try matrixShape(query_weight);
        const k_shape = try matrixShape(key_weight);
        const v_shape = try matrixShape(value_weight);
        const o_shape = try matrixShape(output_weight);
        if (q_shape.columns != hidden or q_shape.rows != hidden or
            k_shape.columns != hidden or k_shape.rows != kv_width or
            v_shape.columns != hidden or v_shape.rows != kv_width or
            o_shape.columns != hidden or o_shape.rows != hidden)
        {
            return Error.InvalidExecutionShape;
        }

        const query = workspace.query[0..hidden];
        const context = workspace.context[0..hidden];
        const key = cache.keys[position * kv_width ..][0..kv_width];
        const value = cache.values[position * kv_width ..][0..kv_width];
        try self.matVec(query_weight, normalized_input, query);
        try self.matVec(key_weight, normalized_input, key);
        try self.matVec(value_weight, normalized_input, value);
        applyRope(query, config.head_count, config.head_dim, position, config.rope_theta);
        applyRope(key, config.kv_head_count, config.head_dim, position, config.rope_theta);
        const token_count = position + 1;

        const group_size = config.head_count / config.kv_head_count;
        const attention_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(config.head_dim)));
        for (0..config.head_count) |head| {
            const query_head = query[head * config.head_dim ..][0..config.head_dim];
            const kv_head = head / group_size;
            const scores = cache.scores[0..token_count];
            var maximum: f32 = -std.math.inf(f32);
            for (scores, 0..) |*score, token| {
                const cached_key = cache.keys[token * kv_width + kv_head * config.head_dim ..][0..config.head_dim];
                var dot: f32 = 0;
                for (query_head, cached_key) |q, k| dot += q * k;
                score.* = dot * attention_scale;
                maximum = @max(maximum, score.*);
            }
            var denominator: f32 = 0;
            for (scores) |*score| {
                score.* = @exp(score.* - maximum);
                denominator += score.*;
            }
            const context_head = context[head * config.head_dim ..][0..config.head_dim];
            @memset(context_head, 0);
            for (scores, 0..) |score, token| {
                const cached_value = cache.values[token * kv_width + kv_head * config.head_dim ..][0..config.head_dim];
                const probability = score / denominator;
                for (context_head, cached_value) |*result, cached| result.* += probability * cached;
            }
        }
        try self.matVec(output_weight, context, output);
        cache.len = token_count;
    }

    /// One Llama-style pre-norm decoder layer for a single causal token.
    /// `state` is updated in place with attention and FFN residuals.
    pub fn decoderLayerSingleToken(
        self: *CpuExecutor,
        attention_norm: *const gguf.TensorDescriptor,
        query_weight: *const gguf.TensorDescriptor,
        key_weight: *const gguf.TensorDescriptor,
        value_weight: *const gguf.TensorDescriptor,
        output_weight: *const gguf.TensorDescriptor,
        ffn_norm: *const gguf.TensorDescriptor,
        gate: *const gguf.TensorDescriptor,
        up: *const gguf.TensorDescriptor,
        down: *const gguf.TensorDescriptor,
        state: []f32,
        position: usize,
        config: AttentionConfig,
        cache: *KvCache,
        workspace: *AttentionWorkspace,
    ) Error!void {
        if (self.activation_a.len < state.len or workspace.context.len < state.len or workspace.query.len < state.len) {
            return Error.ActivationCapacityExceeded;
        }
        try self.rmsNorm(attention_norm, state, workspace.context[0..state.len], config.rms_epsilon);
        try self.attentionSingleToken(query_weight, key_weight, value_weight, output_weight, workspace.context[0..state.len], position, config, cache, workspace, self.activation_a[0..state.len]);
        for (state, self.activation_a[0..state.len]) |*value, residual| value.* += residual;

        try self.rmsNorm(ffn_norm, state, workspace.context[0..state.len], config.rms_epsilon);
        try self.ffnSwiGlu(gate, up, down, workspace.context[0..state.len], workspace.query[0..state.len]);
        for (state, workspace.query[0..state.len]) |*value, residual| value.* += residual;
    }

    /// Layer-major causal prefill. Each projection scans a weight tensor once
    /// for the complete token batch, then attention consumes the resulting Q/K/V
    /// rows in causal order. Existing cache entries may precede this batch.
    pub fn decoderLayerPrefill(
        self: *CpuExecutor,
        weights: DecoderLayerWeights,
        states: []f32,
        batch_count: usize,
        config: AttentionConfig,
        cache: *KvCache,
        workspace: *PrefillWorkspace,
    ) Error!void {
        const hidden = std.math.mul(usize, config.head_count, config.head_dim) catch return Error.InvalidExecutionShape;
        const kv_width = std.math.mul(usize, config.kv_head_count, config.head_dim) catch return Error.InvalidExecutionShape;
        const gate_shape = try matrixShape(weights.ffn_gate);
        const up_shape = try matrixShape(weights.ffn_up);
        const down_shape = try matrixShape(weights.ffn_down);
        const hidden_elements = std.math.mul(usize, batch_count, hidden) catch return Error.InvalidExecutionShape;
        const intermediate_elements = std.math.mul(usize, batch_count, gate_shape.rows) catch return Error.InvalidExecutionShape;
        const end_position = std.math.add(usize, cache.len, batch_count) catch return Error.KvCacheFull;
        if (batch_count == 0 or states.len != hidden_elements or workspace.capacity < batch_count or
            workspace.hidden != hidden or cache.kv_width != kv_width or
            config.head_count == 0 or config.kv_head_count == 0 or config.head_count % config.kv_head_count != 0 or
            config.head_dim == 0 or config.head_dim % 2 != 0 or !std.math.isFinite(config.rope_theta) or
            !std.math.isFinite(config.rms_epsilon) or config.rope_theta <= 0 or config.rms_epsilon <= 0 or
            end_position > cache.capacity or gate_shape.columns != hidden or up_shape.columns != hidden or
            gate_shape.rows != up_shape.rows or down_shape.columns != gate_shape.rows or
            down_shape.rows != hidden or workspace.intermediate < gate_shape.rows)
        {
            return Error.InvalidExecutionShape;
        }
        try validateMatrix(weights.query, hidden, hidden);
        try validateMatrix(weights.key, hidden, kv_width);
        try validateMatrix(weights.value, hidden, kv_width);
        try validateMatrix(weights.attention_output, hidden, hidden);
        try validateMatrix(weights.ffn_gate, hidden, gate_shape.rows);
        try validateMatrix(weights.ffn_up, hidden, gate_shape.rows);
        try validateMatrix(weights.ffn_down, gate_shape.rows, hidden);
        try self.ensureBatchScratch(weights.query, batch_count);
        try self.ensureBatchScratch(weights.key, batch_count);
        try self.ensureBatchScratch(weights.value, batch_count);
        try self.ensureBatchScratch(weights.attention_output, batch_count);
        try self.ensureBatchScratch(weights.ffn_gate, batch_count);
        try self.ensureBatchScratch(weights.ffn_up, batch_count);
        try self.ensureBatchScratch(weights.ffn_down, batch_count);
        const normalized = workspace.normalized[0..hidden_elements];
        const queries = workspace.query[0..hidden_elements];
        const contexts = workspace.context[0..hidden_elements];
        const gates = workspace.gate[0..intermediate_elements];
        const ups = workspace.up[0..intermediate_elements];
        const start_position = cache.len;
        const key_start = std.math.mul(usize, start_position, kv_width) catch return Error.InvalidExecutionShape;
        const batch_kv_elements = std.math.mul(usize, batch_count, kv_width) catch return Error.InvalidExecutionShape;
        const new_keys = cache.keys[key_start..][0..batch_kv_elements];
        const new_values = cache.values[key_start..][0..batch_kv_elements];

        try self.rmsNormBatch(weights.attention_norm, states, normalized, batch_count, config.rms_epsilon);
        try self.matMul(weights.query, normalized, queries, batch_count);
        try self.matMul(weights.key, normalized, new_keys, batch_count);
        try self.matMul(weights.value, normalized, new_values, batch_count);
        for (0..batch_count) |batch| {
            const position = start_position + batch;
            applyRope(queries[batch * hidden ..][0..hidden], config.head_count, config.head_dim, position, config.rope_theta);
            applyRope(new_keys[batch * kv_width ..][0..kv_width], config.kv_head_count, config.head_dim, position, config.rope_theta);
        }

        const group_size = config.head_count / config.kv_head_count;
        const attention_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(config.head_dim)));
        for (0..batch_count) |batch| {
            const token_count = start_position + batch + 1;
            for (0..config.head_count) |head| {
                const query_head = queries[batch * hidden + head * config.head_dim ..][0..config.head_dim];
                const kv_head = head / group_size;
                const scores = cache.scores[0..token_count];
                var maximum: f32 = -std.math.inf(f32);
                for (scores, 0..) |*score, token| {
                    const cached_key = cache.keys[token * kv_width + kv_head * config.head_dim ..][0..config.head_dim];
                    var dot: f32 = 0;
                    for (query_head, cached_key) |q, k| dot += q * k;
                    score.* = dot * attention_scale;
                    maximum = @max(maximum, score.*);
                }
                var denominator: f32 = 0;
                for (scores) |*score| {
                    score.* = @exp(score.* - maximum);
                    denominator += score.*;
                }
                const context_head = contexts[batch * hidden + head * config.head_dim ..][0..config.head_dim];
                @memset(context_head, 0);
                for (scores, 0..) |score, token| {
                    const cached_value = cache.values[token * kv_width + kv_head * config.head_dim ..][0..config.head_dim];
                    const probability = score / denominator;
                    for (context_head, cached_value) |*result, cached| result.* += probability * cached;
                }
            }
        }
        // Reuse normalized as the output-projection buffer after Q/K/V no
        // longer need the normalized input.
        try self.matMul(weights.attention_output, contexts, normalized, batch_count);
        for (states, normalized) |*state, residual| state.* += residual;
        cache.len = start_position + batch_count;

        try self.rmsNormBatch(weights.ffn_norm, states, normalized, batch_count, config.rms_epsilon);
        try self.matMul(weights.ffn_gate, normalized, gates, batch_count);
        try self.matMul(weights.ffn_up, normalized, ups, batch_count);
        for (gates, ups) |*gate, up| gate.* = silu(gate.*) * up;
        try self.matMul(weights.ffn_down, gates, contexts, batch_count);
        for (states, contexts) |*state, residual| state.* += residual;
    }

    /// Executes one complete Llama-style token path using bounded immutable
    /// weights: embedding lookup, every decoder block, final RMSNorm, and LM
    /// head. Each layer owns an independent writable KV cache; weight views do
    /// not survive their individual operation boundary.
    /// Executes a caller-owned token sequence in order, appending positions to
    /// the existing per-layer caches and returning logits for the final token.
    /// Cache storage is reused; this function neither clears nor reallocates it.
    /// Layer-major prompt prefill. Token states remain caller-owned for the
    /// duration of the request so every layer can reuse each weight tile across
    /// all prompt positions. Logits are produced for the final token only.
    pub fn modelPrefill(
        self: *CpuExecutor,
        embedding: *const gguf.TensorDescriptor,
        layers: []const DecoderLayerWeights,
        output_norm: *const gguf.TensorDescriptor,
        output_weight: *const gguf.TensorDescriptor,
        tokens: []const usize,
        config: AttentionConfig,
        caches: []KvCache,
        workspace: *PrefillWorkspace,
        states: []f32,
        logits: []f32,
    ) Error!void {
        return self.modelPrefillInner(embedding, layers, output_norm, output_weight, tokens, config, caches, workspace, states, logits, true);
    }

    fn modelPrefillInner(
        self: *CpuExecutor,
        embedding: *const gguf.TensorDescriptor,
        layers: []const DecoderLayerWeights,
        output_norm: *const gguf.TensorDescriptor,
        output_weight: *const gguf.TensorDescriptor,
        tokens: []const usize,
        config: AttentionConfig,
        caches: []KvCache,
        workspace: *PrefillWorkspace,
        states: []f32,
        logits: []f32,
        want_logits: bool,
    ) Error!void {
        if (tokens.len == 0 or layers.len == 0 or caches.len != layers.len or workspace.capacity < tokens.len) {
            return Error.InvalidExecutionShape;
        }
        const hidden = std.math.mul(usize, config.head_count, config.head_dim) catch return Error.InvalidExecutionShape;
        const state_elements = std.math.mul(usize, tokens.len, hidden) catch return Error.InvalidExecutionShape;
        if (states.len != state_elements or embedding.n_dimensions != 2 or embedding.dimensions[0] != hidden) {
            return Error.InvalidExecutionShape;
        }
        const start_position = caches[0].len;
        const end_position = std.math.add(usize, start_position, tokens.len) catch return Error.KvCacheFull;
        try validateMatrix(output_weight, hidden, logits.len);
        try validateNorm(output_norm, hidden);
        for (tokens) |token| if (token >= embedding.dimensions[1]) return Error.InvalidToken;
        const kv_width = std.math.mul(usize, config.kv_head_count, config.head_dim) catch return Error.InvalidExecutionShape;
        for (layers) |layer| {
            const gate_shape = try matrixShape(layer.ffn_gate);
            try validateNorm(layer.attention_norm, hidden);
            try validateNorm(layer.ffn_norm, hidden);
            try validateMatrix(layer.query, hidden, hidden);
            try validateMatrix(layer.key, hidden, kv_width);
            try validateMatrix(layer.value, hidden, kv_width);
            try validateMatrix(layer.attention_output, hidden, hidden);
            try validateMatrix(layer.ffn_gate, hidden, gate_shape.rows);
            try validateMatrix(layer.ffn_up, hidden, gate_shape.rows);
            try validateMatrix(layer.ffn_down, gate_shape.rows, hidden);
        }
        for (caches) |cache| {
            if (cache.len != start_position) return Error.InvalidPosition;
            if (end_position > cache.capacity) return Error.KvCacheFull;
        }
        for (tokens, 0..) |token, batch| {
            try self.tokenEmbedding(embedding, token, states[batch * hidden ..][0..hidden]);
        }
        for (layers, caches) |layer, *cache| {
            try self.decoderLayerPrefill(layer, states, tokens.len, config, cache, workspace);
        }
        if (!want_logits) return;
        const final_state = states[(tokens.len - 1) * hidden ..][0..hidden];
        const normalized = workspace.normalized[0..hidden];
        try self.rmsNorm(output_norm, final_state, normalized, config.rms_epsilon);
        try self.matVec(output_weight, normalized, logits);
    }

    /// Chunked layer-major prompt prefill. The prompt is executed in chunks of
    /// at most `chunk_size` tokens so the caller-owned prompt state and the
    /// prefill workspace stay bounded by the chunk size rather than the full
    /// prompt length. Per-layer KV caches persist across chunks; each chunk
    /// attends to all previous positions. Logits are produced for the final
    /// prompt token only. The complete request is validated before any cache
    /// mutation.
    pub fn modelPrefillChunked(
        self: *CpuExecutor,
        embedding: *const gguf.TensorDescriptor,
        layers: []const DecoderLayerWeights,
        output_norm: *const gguf.TensorDescriptor,
        output_weight: *const gguf.TensorDescriptor,
        tokens: []const usize,
        chunk_size: usize,
        config: AttentionConfig,
        caches: []KvCache,
        workspace: *PrefillWorkspace,
        chunk_states: []f32,
        logits: []f32,
    ) Error!void {
        if (chunk_size == 0 or workspace.capacity < chunk_size) return Error.InvalidExecutionShape;
        const hidden = std.math.mul(usize, config.head_count, config.head_dim) catch return Error.InvalidExecutionShape;
        const chunk_state_elements = std.math.mul(usize, chunk_size, hidden) catch return Error.InvalidExecutionShape;
        if (chunk_states.len != chunk_state_elements) return Error.InvalidExecutionShape;

        // Validate the complete request before mutating writable KV state.
        if (tokens.len == 0 or layers.len == 0 or caches.len != layers.len) {
            return Error.InvalidExecutionShape;
        }
        const start_position = caches[0].len;
        const end_position = std.math.add(usize, start_position, tokens.len) catch return Error.KvCacheFull;
        try validateMatrix(output_weight, hidden, logits.len);
        try validateNorm(output_norm, hidden);
        for (tokens) |token| if (token >= embedding.dimensions[1]) return Error.InvalidToken;
        const kv_width = std.math.mul(usize, config.kv_head_count, config.head_dim) catch return Error.InvalidExecutionShape;
        for (layers) |layer| {
            const gate_shape = try matrixShape(layer.ffn_gate);
            try validateNorm(layer.attention_norm, hidden);
            try validateNorm(layer.ffn_norm, hidden);
            try validateMatrix(layer.query, hidden, hidden);
            try validateMatrix(layer.key, hidden, kv_width);
            try validateMatrix(layer.value, hidden, kv_width);
            try validateMatrix(layer.attention_output, hidden, hidden);
            try validateMatrix(layer.ffn_gate, hidden, gate_shape.rows);
            try validateMatrix(layer.ffn_up, hidden, gate_shape.rows);
            try validateMatrix(layer.ffn_down, gate_shape.rows, hidden);
        }
        for (caches) |cache| {
            if (cache.len != start_position) return Error.InvalidPosition;
            if (end_position > cache.capacity) return Error.KvCacheFull;
        }

        var processed: usize = 0;
        while (processed < tokens.len) {
            const count = @min(chunk_size, tokens.len - processed);
            const chunk = tokens[processed..][0..count];
            const last_chunk = processed + count == tokens.len;
            try self.modelPrefillInner(
                embedding,
                layers,
                output_norm,
                output_weight,
                chunk,
                config,
                caches,
                workspace,
                chunk_states[0 .. count * hidden],
                logits,
                last_chunk,
            );
            processed += count;
        }
    }

    /// Compatibility token-major path used for incremental append. New prompt
    /// prefill callers should use `modelPrefill` to reuse each projection tile
    /// across all prompt positions.
    pub fn modelTokens(
        self: *CpuExecutor,
        embedding: *const gguf.TensorDescriptor,
        layers: []const DecoderLayerWeights,
        output_norm: *const gguf.TensorDescriptor,
        output_weight: *const gguf.TensorDescriptor,
        tokens: []const usize,
        config: AttentionConfig,
        caches: []KvCache,
        workspace: *AttentionWorkspace,
        state: []f32,
        logits: []f32,
    ) Error!void {
        if (tokens.len == 0 or layers.len == 0 or caches.len != layers.len or
            self.activation_a.len < state.len or workspace.context.len < state.len)
        {
            return Error.InvalidExecutionShape;
        }
        const output_shape = try matrixShape(output_weight);
        if (output_shape.columns != state.len or output_shape.rows != logits.len or
            embedding.n_dimensions != 2 or embedding.dimensions[0] != state.len)
        {
            return Error.InvalidExecutionShape;
        }

        // Validate the complete request before mutating writable KV state.
        const start_position = caches[0].len;
        const end_position = std.math.add(usize, start_position, tokens.len) catch return Error.KvCacheFull;
        for (tokens) |token| {
            if (token >= embedding.dimensions[1]) return Error.InvalidToken;
        }
        for (caches) |cache| {
            if (cache.len != start_position) return Error.InvalidPosition;
            if (end_position > cache.capacity) return Error.KvCacheFull;
        }

        for (tokens, start_position..) |token, position| {
            try self.modelTokenAtPosition(
                embedding,
                layers,
                output_norm,
                output_weight,
                token,
                position,
                config,
                caches,
                workspace,
                state,
                logits,
            );
        }
    }

    /// Compatibility entry point for one token at an explicit cache position.
    pub fn modelSingleToken(
        self: *CpuExecutor,
        embedding: *const gguf.TensorDescriptor,
        layers: []const DecoderLayerWeights,
        output_norm: *const gguf.TensorDescriptor,
        output_weight: *const gguf.TensorDescriptor,
        token: usize,
        position: usize,
        config: AttentionConfig,
        caches: []KvCache,
        workspace: *AttentionWorkspace,
        state: []f32,
        logits: []f32,
    ) Error!void {
        if (caches.len == 0 or caches[0].len != position) return Error.InvalidPosition;
        const tokens = [_]usize{token};
        try self.modelTokens(
            embedding,
            layers,
            output_norm,
            output_weight,
            &tokens,
            config,
            caches,
            workspace,
            state,
            logits,
        );
    }

    fn modelTokenAtPosition(
        self: *CpuExecutor,
        embedding: *const gguf.TensorDescriptor,
        layers: []const DecoderLayerWeights,
        output_norm: *const gguf.TensorDescriptor,
        output_weight: *const gguf.TensorDescriptor,
        token: usize,
        position: usize,
        config: AttentionConfig,
        caches: []KvCache,
        workspace: *AttentionWorkspace,
        state: []f32,
        logits: []f32,
    ) Error!void {
        try self.tokenEmbedding(embedding, token, state);
        for (layers, caches) |layer, *cache| {
            try self.decoderLayerSingleToken(
                layer.attention_norm,
                layer.query,
                layer.key,
                layer.value,
                layer.attention_output,
                layer.ffn_norm,
                layer.ffn_gate,
                layer.ffn_up,
                layer.ffn_down,
                state,
                position,
                config,
                cache,
                workspace,
            );
        }
        try self.rmsNorm(output_norm, state, self.activation_a[0..state.len], config.rms_epsilon);
        try self.matVec(output_weight, self.activation_a[0..state.len], logits);
    }

    pub fn decoderAccounting(self: *const CpuExecutor, cache: *const KvCache, workspace: *const AttentionWorkspace) DecoderMemoryAccounting {
        return .{ .executor = self.accounting(), .attention_workspace_bytes = workspace.byteLen(), .kv_cache_bytes = cache.byteLen() };
    }

    pub fn accounting(self: *const CpuExecutor) MemoryAccounting {
        const metrics = self.manager.metrics();
        return .{
            .weight_budget_bytes = metrics.budget_bytes,
            .current_mapped_weight_bytes = metrics.resident_bytes,
            .peak_mapped_weight_bytes = metrics.peak_resident_bytes,
            .dequant_scratch_bytes = self.dequant_scratch.len * @sizeOf(f32),
            .batch_scratch_bytes = self.batch_scratch.len,
            .activation_bytes = (self.activation_a.len + self.activation_b.len) * @sizeOf(f32),
            .faults = metrics.faults,
            .hits = metrics.hits,
            .evictions = metrics.evictions,
        };
    }
};

const MatrixShape = struct {
    columns: usize,
    rows: usize,
};

fn matrixShape(descriptor: *const gguf.TensorDescriptor) Error!MatrixShape {
    if (descriptor.n_dimensions != 2) return Error.InvalidExecutionShape;
    const columns = std.math.cast(usize, descriptor.dimensions[0]) orelse return Error.InvalidExecutionShape;
    const rows = std.math.cast(usize, descriptor.dimensions[1]) orelse return Error.InvalidExecutionShape;
    if (columns == 0 or rows == 0) return Error.InvalidExecutionShape;
    return .{ .columns = columns, .rows = rows };
}

fn silu(value: f32) f32 {
    return value / (1.0 + @exp(-value));
}

fn applyRope(values: []f32, head_count: usize, head_dim: usize, position: usize, theta: f32) void {
    const position_f: f32 = @floatFromInt(position);
    for (0..head_count) |head| {
        const head_values = values[head * head_dim ..][0..head_dim];
        var pair: usize = 0;
        while (pair < head_dim) : (pair += 2) {
            const exponent = @as(f32, @floatFromInt(pair)) / @as(f32, @floatFromInt(head_dim));
            const angle = position_f / std.math.pow(f32, theta, exponent);
            const cosine = @cos(angle);
            const sine = @sin(angle);
            // Llama uses LLAMA_ROPE_TYPE_NORM: rotate consecutive values.
            // Half-head pairing is GGML's NEOX layout and is incorrect here.
            const first = head_values[pair];
            const second = head_values[pair + 1];
            head_values[pair] = first * cosine - second * sine;
            head_values[pair + 1] = first * sine + second * cosine;
        }
    }
}

fn referenceMatVec(weights: []const f32, columns: usize, input: []const f32, output: []f32) void {
    for (output, 0..) |*result, row| {
        var sum: f32 = 0;
        for (weights[row * columns ..][0..columns], input) |weight, input_value| {
            sum += weight * input_value;
        }
        result.* = sum;
    }
}

fn testDescriptor(handle_id: u64, name: []const u8, offset: u64, columns: usize, rows: usize) gguf.TensorDescriptor {
    var dimensions = [_]u64{0} ** gguf.max_dimensions;
    dimensions[0] = columns;
    dimensions[1] = rows;
    return .{
        .handle = .{ .id = handle_id },
        .name = name,
        .file_offset = offset,
        .byte_len = columns * rows * @sizeOf(f32),
        .ggml_type = gguf.type_f32,
        .n_dimensions = 2,
        .dimensions = dimensions,
    };
}

const MultiSequenceContext = struct {
    manager: *residency.Manager,
    descriptor: *const gguf.TensorDescriptor,
    input: [4]f32,
    output: [5]f32 = undefined,
    failed: bool = false,

    fn run(self: *MultiSequenceContext) void {
        var executor = CpuExecutor.init(std.heap.page_allocator, self.manager, 4, 5, 5) catch {
            self.failed = true;
            return;
        };
        defer executor.deinit();
        executor.matVec(self.descriptor, &self.input, &self.output) catch {
            self.failed = true;
        };
    }
};

test "multiple executors share one residency manager safely" {
    const allocator = std.testing.allocator;
    const columns: usize = 4;
    const rows: usize = 5;
    const weights = [_]f32{
        1, 2, 3, 4, 2,  1, 0,    -1,   0.5, -0.5, 1.5, -1.5,
        3, 0, 2, 1, -2, 1, 0.25, 0.75,
    };
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var file = try tmp.dir.createFile("shared-manager.bin", .{});
    try file.writeAll(std.mem.sliceAsBytes(&weights));
    file.close();
    const path = try tmp.dir.realpathAlloc(allocator, "shared-manager.bin");
    defer allocator.free(path);
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);
    var store = try residency.BackingStore.open(path_z);
    defer store.close();
    const descriptor = testDescriptor(51, "shared.weight", 0, columns, rows);
    const budget = try residency.mappingGranularity();
    var manager = try residency.Manager.init(allocator, &store, budget);
    defer manager.deinit();
    try manager.register(descriptor.handle, 0, descriptor.byte_len);

    var contexts = [_]MultiSequenceContext{
        .{ .manager = &manager, .descriptor = &descriptor, .input = .{ 1, 0, -1, 2 } },
        .{ .manager = &manager, .descriptor = &descriptor, .input = .{ 0.5, 1.5, -0.5, 0 } },
        .{ .manager = &manager, .descriptor = &descriptor, .input = .{ -1, 2, 0.25, 1 } },
        .{ .manager = &manager, .descriptor = &descriptor, .input = .{ 2, -1, 0, 0.5 } },
    };
    var threads: [contexts.len]std.Thread = undefined;
    for (&threads, &contexts) |*thread, *context| thread.* = try std.Thread.spawn(.{}, MultiSequenceContext.run, .{context});
    for (&threads) |*thread| thread.join();

    for (&contexts) |*context| {
        try std.testing.expect(!context.failed);
        var expected: [rows]f32 = undefined;
        referenceMatVec(&weights, columns, &context.input, &expected);
        try std.testing.expectEqualSlices(f32, &expected, &context.output);
    }
    const metrics = manager.metrics();
    try std.testing.expect(metrics.peak_resident_bytes <= budget);
    try std.testing.expect(metrics.hits >= contexts.len - 1);
}

test "CPU executor batched F32 multiply reuses weight tiles" {
    const allocator = std.testing.allocator;
    const columns: usize = 4;
    const rows: usize = 5;
    const batch_count: usize = 3;
    const weights = [_]f32{
        1, 2, 3, 4, 2,  1, 0,    -1,   0.5, -0.5, 1.5, -1.5,
        3, 0, 2, 1, -2, 1, 0.25, 0.75,
    };
    const inputs = [_]f32{
        1,   0,   -1,   2,
        0.5, 1.5, -0.5, 0,
        -1,  2,   0.25, 1,
    };
    var expected: [batch_count * rows]f32 = undefined;
    for (0..batch_count) |batch| {
        referenceMatVec(&weights, columns, inputs[batch * columns ..][0..columns], expected[batch * rows ..][0..rows]);
    }

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var file = try tmp.dir.createFile("executor-batch-f32.bin", .{});
    try file.writeAll(std.mem.sliceAsBytes(&weights));
    file.close();
    const path = try tmp.dir.realpathAlloc(allocator, "executor-batch-f32.bin");
    defer allocator.free(path);
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);
    var store = try residency.BackingStore.open(path_z);
    defer store.close();
    const descriptor = testDescriptor(50, "batch.weight", 0, columns, rows);
    var manager = try residency.Manager.init(allocator, &store, try residency.mappingGranularity());
    defer manager.deinit();
    try manager.register(descriptor.handle, 0, descriptor.byte_len);
    var executor = try CpuExecutor.init(allocator, &manager, columns, rows, 2);
    defer executor.deinit();
    var actual: [batch_count * rows]f32 = undefined;
    try executor.matMul(&descriptor, &inputs, &actual, batch_count);
    try std.testing.expectEqualSlices(f32, &expected, &actual);
    try std.testing.expectEqual(@as(usize, 0), executor.accounting().batch_scratch_bytes);
}

test "CPU execution boundary runs bounded SwiGLU FFN and accounts non-weight memory" {
    const allocator = std.testing.allocator;
    const granularity = try residency.mappingGranularity();
    const hidden: usize = 4;
    const intermediate: usize = 3;

    const gate_weights = [_]f32{
        0.2,  -0.4, 0.6,  0.8,
        -0.3, 0.7,  0.1,  -0.5,
        0.9,  0.2,  -0.8, 0.4,
    };
    const up_weights = [_]f32{
        0.5,  0.1,  -0.2, 0.3,
        -0.6, 0.4,  0.8,  0.2,
        0.7,  -0.9, 0.3,  0.6,
    };
    const down_weights = [_]f32{
        0.4,  -0.2, 0.8,
        -0.7, 0.5,  0.1,
        0.3,  0.9,  -0.4,
        0.6,  -0.1, 0.2,
    };
    const input = [_]f32{ 0.25, -0.5, 0.75, 1.0 };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var file = try tmp.dir.createFile("ffn.bin", .{});
    try file.seekTo(0);
    try file.writeAll(std.mem.sliceAsBytes(&gate_weights));
    try file.seekTo(granularity);
    try file.writeAll(std.mem.sliceAsBytes(&up_weights));
    try file.seekTo(granularity * 2);
    try file.writeAll(std.mem.sliceAsBytes(&down_weights));
    file.close();

    const path = try tmp.dir.realpathAlloc(allocator, "ffn.bin");
    defer allocator.free(path);
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);

    const gate = testDescriptor(1, "blk.0.ffn_gate.weight", 0, hidden, intermediate);
    const up = testDescriptor(2, "blk.0.ffn_up.weight", granularity, hidden, intermediate);
    const down = testDescriptor(3, "blk.0.ffn_down.weight", granularity * 2, intermediate, hidden);

    var store = try residency.BackingStore.open(path_z);
    defer store.close();
    var manager = try residency.Manager.init(allocator, &store, granularity);
    defer manager.deinit();
    try manager.register(gate.handle, gate.file_offset, gate.byte_len);
    try manager.register(up.handle, up.file_offset, up.byte_len);
    try manager.register(down.handle, down.file_offset, down.byte_len);

    var executor = try CpuExecutor.init(allocator, &manager, hidden, intermediate, 2);
    defer executor.deinit();
    var actual: [hidden]f32 = undefined;
    try executor.ffnSwiGlu(&gate, &up, &down, &input, &actual);

    var expected_gate: [intermediate]f32 = undefined;
    var expected_up: [intermediate]f32 = undefined;
    var expected: [hidden]f32 = undefined;
    referenceMatVec(&gate_weights, hidden, &input, &expected_gate);
    referenceMatVec(&up_weights, hidden, &input, &expected_up);
    for (&expected_gate, expected_up) |*gate_value, up_value| gate_value.* = silu(gate_value.*) * up_value;
    referenceMatVec(&down_weights, intermediate, &expected_gate, &expected);

    for (expected, actual) |wanted, got| try std.testing.expectApproxEqAbs(wanted, got, 1e-6);
    const accounting = executor.accounting();
    try std.testing.expect(accounting.peak_mapped_weight_bytes <= granularity);
    try std.testing.expectEqual(hidden * @sizeOf(f32), accounting.dequant_scratch_bytes);
    try std.testing.expectEqual(intermediate * 2 * @sizeOf(f32), accounting.activation_bytes);
    try std.testing.expect(accounting.faults >= 3);
    // Multi-window residency: small weight windows may coexist within the
    // budget, so evictions are no longer guaranteed. The invariant that must
    // hold is the budget itself, across the manager and the executor view.
    try std.testing.expect(manager.metrics().resident_bytes <= granularity);
}

fn testVectorDescriptor(handle_id: u64, name: []const u8, offset: u64, elements: usize) gguf.TensorDescriptor {
    var dimensions = [_]u64{1} ** gguf.max_dimensions;
    dimensions[0] = elements;
    return .{
        .handle = .{ .id = handle_id },
        .name = name,
        .file_offset = offset,
        .byte_len = elements * @sizeOf(f32),
        .ggml_type = gguf.type_f32,
        .n_dimensions = 1,
        .dimensions = dimensions,
    };
}

fn registerDescriptors(manager: *residency.Manager, descriptors: []const gguf.TensorDescriptor) !void {
    for (descriptors) |descriptor| try manager.register(descriptor.handle, descriptor.file_offset, descriptor.byte_len);
}

fn runTestDecoder(
    executor: *CpuExecutor,
    descriptors: []const gguf.TensorDescriptor,
    cache: *KvCache,
    workspace: *AttentionWorkspace,
    first: []f32,
    second: []f32,
) !void {
    const config = AttentionConfig{ .head_count = 2, .kv_head_count = 1, .head_dim = 2 };
    try executor.decoderLayerSingleToken(
        &descriptors[0],
        &descriptors[1],
        &descriptors[2],
        &descriptors[3],
        &descriptors[4],
        &descriptors[5],
        &descriptors[6],
        &descriptors[7],
        &descriptors[8],
        first,
        0,
        config,
        cache,
        workspace,
    );
    try executor.decoderLayerSingleToken(
        &descriptors[0],
        &descriptors[1],
        &descriptors[2],
        &descriptors[3],
        &descriptors[4],
        &descriptors[5],
        &descriptors[6],
        &descriptors[7],
        &descriptors[8],
        second,
        1,
        config,
        cache,
        workspace,
    );
}

test "bounded decoder layer matches resident baseline and accounts KV memory" {
    const allocator = std.testing.allocator;
    const granularity = try residency.mappingGranularity();
    const hidden: usize = 4;
    const intermediate: usize = 6;

    const norm = [_]f32{ 1.0, 0.8, 1.2, 0.9 };
    const query = [_]f32{ 0.5, 0.1, 0.0, -0.2, 0.0, 0.6, 0.2, 0.0, -0.1, 0.0, 0.7, 0.1, 0.2, -0.1, 0.0, 0.8 };
    const key = [_]f32{ 0.4, -0.2, 0.1, 0.3, 0.0, 0.5, -0.4, 0.2 };
    const value = [_]f32{ 0.7, 0.1, -0.3, 0.2, 0.2, -0.5, 0.4, 0.6 };
    const output = [_]f32{ 0.6, 0.0, 0.1, -0.2, 0.1, 0.7, -0.1, 0.0, 0.0, 0.2, 0.5, 0.1, -0.1, 0.0, 0.2, 0.6 };
    const ffn_norm = [_]f32{ 0.9, 1.1, 0.8, 1.0 };
    const gate = [_]f32{ 0.2, -0.1, 0.3, 0.4, -0.3, 0.5, 0.1, -0.2, 0.6, 0.2, -0.4, 0.1, 0.1, 0.3, 0.5, -0.2, -0.5, 0.2, 0.4, 0.3, 0.4, -0.3, 0.2, 0.6 };
    const up = [_]f32{ 0.5, 0.2, -0.1, 0.3, -0.2, 0.4, 0.6, 0.1, 0.3, -0.5, 0.2, 0.4, 0.1, 0.7, -0.3, 0.2, -0.4, 0.1, 0.5, 0.6, 0.2, -0.2, 0.4, 0.3 };
    const down = [_]f32{ 0.3, -0.2, 0.1, 0.4, 0.2, -0.1, -0.1, 0.5, 0.2, -0.3, 0.4, 0.1, 0.4, 0.1, -0.2, 0.2, -0.1, 0.5, 0.2, -0.3, 0.6, 0.1, 0.3, -0.2 };
    const tensors = [_][]const f32{ &norm, &query, &key, &value, &output, &ffn_norm, &gate, &up, &down };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const backing = try allocator.alloc(u8, granularity * tensors.len);
    defer allocator.free(backing);
    @memset(backing, 0);
    for (tensors, 0..) |tensor, i| @memcpy(backing[i * granularity ..][0 .. tensor.len * @sizeOf(f32)], std.mem.sliceAsBytes(tensor));
    var file = try tmp.dir.createFile("decoder.bin", .{});
    try file.writeAll(backing);
    file.close();
    const path = try tmp.dir.realpathAlloc(allocator, "decoder.bin");
    defer allocator.free(path);
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);

    const descriptors = [_]gguf.TensorDescriptor{
        testVectorDescriptor(1, "attn_norm", 0, hidden),
        testDescriptor(2, "query", granularity, hidden, hidden),
        testDescriptor(3, "key", granularity * 2, hidden, 2),
        testDescriptor(4, "value", granularity * 3, hidden, 2),
        testDescriptor(5, "output", granularity * 4, hidden, hidden),
        testVectorDescriptor(6, "ffn_norm", granularity * 5, hidden),
        testDescriptor(7, "gate", granularity * 6, hidden, intermediate),
        testDescriptor(8, "up", granularity * 7, hidden, intermediate),
        testDescriptor(9, "down", granularity * 8, intermediate, hidden),
    };

    var store = try residency.BackingStore.open(path_z);
    defer store.close();
    var baseline_manager = try residency.Manager.init(allocator, &store, granularity * tensors.len);
    defer baseline_manager.deinit();
    try registerDescriptors(&baseline_manager, &descriptors);
    const bounded_budget: usize = 128;
    var bounded_manager = try residency.Manager.init(allocator, &store, bounded_budget);
    defer bounded_manager.deinit();
    try registerDescriptors(&bounded_manager, &descriptors);

    var baseline_executor = try CpuExecutor.init(allocator, &baseline_manager, intermediate, intermediate, intermediate);
    defer baseline_executor.deinit();
    var bounded_executor = try CpuExecutor.init(allocator, &bounded_manager, intermediate, intermediate, intermediate);
    defer bounded_executor.deinit();
    var baseline_cache = try KvCache.init(allocator, 2, 2);
    defer baseline_cache.deinit();
    var bounded_cache = try KvCache.init(allocator, 2, 2);
    defer bounded_cache.deinit();
    var baseline_workspace = try AttentionWorkspace.init(allocator, hidden);
    defer baseline_workspace.deinit();
    var bounded_workspace = try AttentionWorkspace.init(allocator, hidden);
    defer bounded_workspace.deinit();

    var baseline_first = [_]f32{ 0.2, -0.4, 0.7, 0.1 };
    var baseline_second = [_]f32{ -0.3, 0.5, 0.2, 0.8 };
    var bounded_first = baseline_first;
    var bounded_second = baseline_second;
    try runTestDecoder(&baseline_executor, &descriptors, &baseline_cache, &baseline_workspace, &baseline_first, &baseline_second);
    try runTestDecoder(&bounded_executor, &descriptors, &bounded_cache, &bounded_workspace, &bounded_first, &bounded_second);

    for (baseline_first, bounded_first) |expected, actual| try std.testing.expectApproxEqAbs(expected, actual, 1e-6);
    for (baseline_second, bounded_second) |expected, actual| try std.testing.expectApproxEqAbs(expected, actual, 1e-6);
    const memory = bounded_executor.decoderAccounting(&bounded_cache, &bounded_workspace);
    try std.testing.expect(memory.executor.peak_mapped_weight_bytes <= bounded_budget);
    try std.testing.expectEqual(@as(usize, 40), memory.kv_cache_bytes);
    try std.testing.expectEqual(hidden * 2 * @sizeOf(f32), memory.attention_workspace_bytes);
    try std.testing.expect(memory.executor.evictions > 0);
    try std.testing.expectEqual(@as(usize, 2), bounded_cache.len);
}

test "position greater than zero uses Llama normal adjacent-pair RoPE" {
    var values = [_]f32{ 1, 2, 3, 4 };
    applyRope(&values, 1, 4, 1, 10_000);
    const c0 = @cos(@as(f32, 1));
    const s0 = @sin(@as(f32, 1));
    const c1 = @cos(@as(f32, 0.01));
    const s1 = @sin(@as(f32, 0.01));
    try std.testing.expectApproxEqAbs(1 * c0 - 2 * s0, values[0], 1e-6);
    try std.testing.expectApproxEqAbs(1 * s0 + 2 * c0, values[1], 1e-6);
    try std.testing.expectApproxEqAbs(3 * c1 - 4 * s1, values[2], 1e-6);
    try std.testing.expectApproxEqAbs(3 * s1 + 4 * c1, values[3], 1e-6);
}

test "chunked prefill matches full prefill and incremental append exactly" {
    const allocator = std.testing.allocator;
    const granularity = try residency.mappingGranularity();
    const hidden: usize = 4;
    const vocab: usize = 3;
    const tokens = [_]usize{ 0, 2, 1, 2 };
    const chunk_size: usize = 3;

    var prng = std.Random.DefaultPrng.init(0x5eed);
    const random = prng.random();
    const norm = [_]f32{ 1.0, 0.8, 1.2, 0.9 };
    const ffn_norm = [_]f32{ 0.9, 1.1, 0.8, 1.0 };
    const intermediate: usize = 6;
    const embedding_values = try allocator.alloc(f32, hidden * vocab);
    defer allocator.free(embedding_values);
    for (embedding_values) |*value| value.* = random.float(f32) * 2.0 - 1.0;
    const head_values = try allocator.alloc(f32, hidden * vocab);
    defer allocator.free(head_values);
    for (head_values) |*value| value.* = random.float(f32) * 2.0 - 1.0;
    const query_values = try allocator.alloc(f32, hidden * hidden);
    defer allocator.free(query_values);
    for (query_values) |*value| value.* = random.float(f32) * 0.5 - 0.25;
    const key_values = try allocator.alloc(f32, 2 * hidden);
    defer allocator.free(key_values);
    for (key_values) |*value| value.* = random.float(f32) * 0.5 - 0.25;
    const value_values = try allocator.alloc(f32, 2 * hidden);
    defer allocator.free(value_values);
    for (value_values) |*value| value.* = random.float(f32) * 0.5 - 0.25;
    const output_values = try allocator.alloc(f32, hidden * hidden);
    defer allocator.free(output_values);
    for (output_values) |*value| value.* = random.float(f32) * 0.5 - 0.25;
    const gate_values = try allocator.alloc(f32, hidden * intermediate);
    defer allocator.free(gate_values);
    for (gate_values) |*value| value.* = random.float(f32) * 0.5 - 0.25;
    const up_values = try allocator.alloc(f32, hidden * intermediate);
    defer allocator.free(up_values);
    for (up_values) |*value| value.* = random.float(f32) * 0.5 - 0.25;
    const down_values = try allocator.alloc(f32, intermediate * hidden);
    defer allocator.free(down_values);
    for (down_values) |*value| value.* = random.float(f32) * 0.5 - 0.25;

    const tensors = [_][]const f32{
        embedding_values, head_values,   &norm,     query_values, key_values,
        value_values,     output_values, &ffn_norm, gate_values,  up_values,
        down_values,
    };
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const backing = try allocator.alloc(u8, granularity * tensors.len);
    defer allocator.free(backing);
    @memset(backing, 0);
    for (tensors, 0..) |tensor, i| @memcpy(backing[i * granularity ..][0 .. tensor.len * @sizeOf(f32)], std.mem.sliceAsBytes(tensor));
    var file = try tmp.dir.createFile("chunked.bin", .{});
    try file.writeAll(backing);
    file.close();
    const path = try tmp.dir.realpathAlloc(allocator, "chunked.bin");
    defer allocator.free(path);
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);

    var embedding_dims = [_]u64{0} ** gguf.max_dimensions;
    embedding_dims[0] = hidden;
    embedding_dims[1] = vocab;
    const embedding = gguf.TensorDescriptor{
        .handle = .{ .id = 1 },
        .name = "token_embd",
        .file_offset = 0,
        .byte_len = embedding_values.len * @sizeOf(f32),
        .ggml_type = gguf.type_f32,
        .n_dimensions = 2,
        .dimensions = embedding_dims,
    };
    var head_dims = [_]u64{0} ** gguf.max_dimensions;
    head_dims[0] = hidden;
    head_dims[1] = vocab;
    const head = gguf.TensorDescriptor{
        .handle = .{ .id = 2 },
        .name = "output.weight",
        .file_offset = granularity,
        .byte_len = head_values.len * @sizeOf(f32),
        .ggml_type = gguf.type_f32,
        .n_dimensions = 2,
        .dimensions = head_dims,
    };
    const attn_norm = testVectorDescriptor(3, "attn_norm", granularity * 2, hidden);
    const query = testDescriptor(4, "query", granularity * 3, hidden, hidden);
    const key = testDescriptor(5, "key", granularity * 4, hidden, 2);
    const value = testDescriptor(6, "value", granularity * 5, hidden, 2);
    const attention_output = testDescriptor(7, "output", granularity * 6, hidden, hidden);
    const ffn_norm_descriptor = testVectorDescriptor(8, "ffn_norm", granularity * 7, hidden);
    const gate = testDescriptor(9, "gate", granularity * 8, hidden, intermediate);
    const up = testDescriptor(10, "up", granularity * 9, hidden, intermediate);
    const down = testDescriptor(11, "down", granularity * 10, intermediate, hidden);
    const layer = DecoderLayerWeights{
        .attention_norm = &attn_norm,
        .query = &query,
        .key = &key,
        .value = &value,
        .attention_output = &attention_output,
        .ffn_norm = &ffn_norm_descriptor,
        .ffn_gate = &gate,
        .ffn_up = &up,
        .ffn_down = &down,
    };
    const layers = [_]DecoderLayerWeights{layer};
    const config = AttentionConfig{ .head_count = 2, .kv_head_count = 1, .head_dim = 2 };

    const descriptors = [_]gguf.TensorDescriptor{
        embedding, head,             attn_norm,           query, key,
        value,     attention_output, ffn_norm_descriptor, gate,  up,
        down,
    };
    const logits_len: usize = vocab;

    var store = try residency.BackingStore.open(path_z);
    defer store.close();
    var full_logits: [logits_len]f32 = undefined;
    var chunked_logits: [logits_len]f32 = undefined;
    var incremental_logits: [logits_len]f32 = undefined;

    {
        var manager = try residency.Manager.init(allocator, &store, granularity * 16);
        defer manager.deinit();
        try registerDescriptors(&manager, &descriptors);
        var executor = try CpuExecutor.init(allocator, &manager, intermediate, intermediate, intermediate);
        defer executor.deinit();
        var caches = [_]KvCache{try KvCache.init(allocator, 8, 2)};
        defer caches[0].deinit();
        var workspace = try PrefillWorkspace.init(allocator, tokens.len, hidden, intermediate);
        defer workspace.deinit();
        const states = try allocator.alloc(f32, tokens.len * hidden);
        defer allocator.free(states);
        try executor.modelPrefill(&embedding, &layers, &attn_norm, &head, &tokens, config, &caches, &workspace, states, &full_logits);
        try std.testing.expectEqual(@as(usize, tokens.len), caches[0].len);
    }
    {
        var manager = try residency.Manager.init(allocator, &store, granularity * 16);
        defer manager.deinit();
        try registerDescriptors(&manager, &descriptors);
        var executor = try CpuExecutor.init(allocator, &manager, intermediate, intermediate, intermediate);
        defer executor.deinit();
        var caches = [_]KvCache{try KvCache.init(allocator, 8, 2)};
        defer caches[0].deinit();
        var workspace = try PrefillWorkspace.init(allocator, chunk_size, hidden, intermediate);
        defer workspace.deinit();
        const chunk_states = try allocator.alloc(f32, chunk_size * hidden);
        defer allocator.free(chunk_states);
        try executor.modelPrefillChunked(&embedding, &layers, &attn_norm, &head, &tokens, chunk_size, config, &caches, &workspace, chunk_states, &chunked_logits);
        try std.testing.expectEqual(@as(usize, tokens.len), caches[0].len);
    }
    {
        var manager = try residency.Manager.init(allocator, &store, granularity * 16);
        defer manager.deinit();
        try registerDescriptors(&manager, &descriptors);
        var executor = try CpuExecutor.init(allocator, &manager, intermediate, intermediate, intermediate);
        defer executor.deinit();
        var caches = [_]KvCache{try KvCache.init(allocator, 8, 2)};
        defer caches[0].deinit();
        var workspace = try AttentionWorkspace.init(allocator, hidden);
        defer workspace.deinit();
        var state: [hidden]f32 = undefined;
        for (tokens) |token| {
            const single = [1]usize{token};
            try executor.modelTokens(&embedding, &layers, &attn_norm, &head, &single, config, &caches, &workspace, &state, &incremental_logits);
        }
        try std.testing.expectEqual(@as(usize, tokens.len), caches[0].len);
    }

    for (full_logits, chunked_logits) |expected, actual| try std.testing.expectEqual(expected, actual);
    for (full_logits, incremental_logits) |expected, actual| try std.testing.expectEqual(expected, actual);
}

const GenerationContext = struct {
    manager: *residency.Manager,
    embedding: *const gguf.TensorDescriptor,
    layers: []const DecoderLayerWeights,
    output_norm: *const gguf.TensorDescriptor,
    head: *const gguf.TensorDescriptor,
    config: AttentionConfig,
    tokens: []const usize,
    logits: [][3]f32,
    failed: bool = false,

    fn run(self: *GenerationContext) void {
        const allocator = std.heap.page_allocator;
        var executor = CpuExecutor.init(allocator, self.manager, 6, 6, 6) catch {
            self.failed = true;
            return;
        };
        defer executor.deinit();
        var cache = KvCache.init(allocator, 8, 2) catch {
            self.failed = true;
            return;
        };
        defer cache.deinit();
        var workspace = AttentionWorkspace.init(allocator, 4) catch {
            self.failed = true;
            return;
        };
        defer workspace.deinit();
        var state: [4]f32 = undefined;
        for (self.tokens, 0..) |token, step| {
            const single = [1]usize{token};
            self.runStep(&executor, &single, &state, &self.logits[step], &cache, &workspace) catch {
                self.failed = true;
                return;
            };
        }
    }

    fn runStep(
        self: *GenerationContext,
        executor: *CpuExecutor,
        single: *const [1]usize,
        state: *[4]f32,
        out: *[3]f32,
        cache: *KvCache,
        workspace: *AttentionWorkspace,
    ) !void {
        try executor.modelTokens(self.embedding, self.layers, self.output_norm, self.head, single, self.config, @as(*[1]KvCache, @ptrCast(cache)), workspace, state, out);
    }
};

test "concurrent multi-sequence generation matches sequential baseline through one manager" {
    const allocator = std.testing.allocator;
    const granularity = try residency.mappingGranularity();
    const hidden: usize = 4;
    const vocab: usize = 3;
    const intermediate: usize = 6;
    const tokens = [_]usize{ 0, 2, 1 };
    const config = AttentionConfig{ .head_count = 2, .kv_head_count = 1, .head_dim = 2 };

    var prng = std.Random.DefaultPrng.init(0xfa11);
    const random = prng.random();
    const norm = [_]f32{ 1.0, 0.8, 1.2, 0.9 };
    const ffn_norm = [_]f32{ 0.9, 1.1, 0.8, 1.0 };
    const embedding_values = try allocator.alloc(f32, hidden * vocab);
    defer allocator.free(embedding_values);
    for (embedding_values) |*value| value.* = random.float(f32) * 2.0 - 1.0;
    const head_values = try allocator.alloc(f32, hidden * vocab);
    defer allocator.free(head_values);
    for (head_values) |*value| value.* = random.float(f32) * 2.0 - 1.0;
    const query_values = try allocator.alloc(f32, hidden * hidden);
    defer allocator.free(query_values);
    for (query_values) |*value| value.* = random.float(f32) * 0.5 - 0.25;
    const key_values = try allocator.alloc(f32, 2 * hidden);
    defer allocator.free(key_values);
    for (key_values) |*value| value.* = random.float(f32) * 0.5 - 0.25;
    const value_values = try allocator.alloc(f32, 2 * hidden);
    defer allocator.free(value_values);
    for (value_values) |*value| value.* = random.float(f32) * 0.5 - 0.25;
    const output_values = try allocator.alloc(f32, hidden * hidden);
    defer allocator.free(output_values);
    for (output_values) |*value| value.* = random.float(f32) * 0.5 - 0.25;
    const gate_values = try allocator.alloc(f32, hidden * intermediate);
    defer allocator.free(gate_values);
    for (gate_values) |*value| value.* = random.float(f32) * 0.5 - 0.25;
    const up_values = try allocator.alloc(f32, hidden * intermediate);
    defer allocator.free(up_values);
    for (up_values) |*value| value.* = random.float(f32) * 0.5 - 0.25;
    const down_values = try allocator.alloc(f32, intermediate * hidden);
    defer allocator.free(down_values);
    for (down_values) |*value| value.* = random.float(f32) * 0.5 - 0.25;

    const tensors = [_][]const f32{
        embedding_values, head_values,   &norm,     query_values, key_values,
        value_values,     output_values, &ffn_norm, gate_values,  up_values,
        down_values,
    };
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const backing = try allocator.alloc(u8, granularity * tensors.len);
    defer allocator.free(backing);
    @memset(backing, 0);
    for (tensors, 0..) |tensor, i| @memcpy(backing[i * granularity ..][0 .. tensor.len * @sizeOf(f32)], std.mem.sliceAsBytes(tensor));
    var file = try tmp.dir.createFile("multi-seq.bin", .{});
    try file.writeAll(backing);
    file.close();
    const path = try tmp.dir.realpathAlloc(allocator, "multi-seq.bin");
    defer allocator.free(path);
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);

    var embedding_dims = [_]u64{0} ** gguf.max_dimensions;
    embedding_dims[0] = hidden;
    embedding_dims[1] = vocab;
    const embedding = gguf.TensorDescriptor{
        .handle = .{ .id = 1 },
        .name = "token_embd",
        .file_offset = 0,
        .byte_len = embedding_values.len * @sizeOf(f32),
        .ggml_type = gguf.type_f32,
        .n_dimensions = 2,
        .dimensions = embedding_dims,
    };
    var head_dims = [_]u64{0} ** gguf.max_dimensions;
    head_dims[0] = hidden;
    head_dims[1] = vocab;
    const head = gguf.TensorDescriptor{
        .handle = .{ .id = 2 },
        .name = "output.weight",
        .file_offset = granularity,
        .byte_len = head_values.len * @sizeOf(f32),
        .ggml_type = gguf.type_f32,
        .n_dimensions = 2,
        .dimensions = head_dims,
    };
    const attn_norm = testVectorDescriptor(3, "attn_norm", granularity * 2, hidden);
    const query = testDescriptor(4, "query", granularity * 3, hidden, hidden);
    const key = testDescriptor(5, "key", granularity * 4, hidden, 2);
    const value = testDescriptor(6, "value", granularity * 5, hidden, 2);
    const attention_output = testDescriptor(7, "output", granularity * 6, hidden, hidden);
    const ffn_norm_descriptor = testVectorDescriptor(8, "ffn_norm", granularity * 7, hidden);
    const gate = testDescriptor(9, "gate", granularity * 8, hidden, intermediate);
    const up = testDescriptor(10, "up", granularity * 9, hidden, intermediate);
    const down = testDescriptor(11, "down", granularity * 10, intermediate, hidden);
    const layer = DecoderLayerWeights{
        .attention_norm = &attn_norm,
        .query = &query,
        .key = &key,
        .value = &value,
        .attention_output = &attention_output,
        .ffn_norm = &ffn_norm_descriptor,
        .ffn_gate = &gate,
        .ffn_up = &up,
        .ffn_down = &down,
    };
    const layers = [_]DecoderLayerWeights{layer};
    const descriptors = [_]gguf.TensorDescriptor{
        embedding, head,             attn_norm,           query, key,
        value,     attention_output, ffn_norm_descriptor, gate,  up,
        down,
    };

    var store = try residency.BackingStore.open(path_z);
    defer store.close();
    var manager = try residency.Manager.init(allocator, &store, granularity * 16);
    defer manager.deinit();
    try registerDescriptors(&manager, &descriptors);

    var baseline: [tokens.len][vocab]f32 = undefined;
    {
        var executor = try CpuExecutor.init(allocator, &manager, intermediate, intermediate, intermediate);
        defer executor.deinit();
        var cache = try KvCache.init(allocator, 8, 2);
        defer cache.deinit();
        var workspace = try AttentionWorkspace.init(allocator, hidden);
        defer workspace.deinit();
        var state: [hidden]f32 = undefined;
        for (tokens, 0..) |token, step| {
            const single = [1]usize{token};
            try executor.modelTokens(&embedding, &layers, &attn_norm, &head, &single, config, @as(*[1]KvCache, @ptrCast(&cache)), &workspace, &state, &baseline[step]);
        }
        try std.testing.expectEqual(@as(usize, tokens.len), cache.len);
    }

    var contexts: [2]GenerationContext = undefined;
    for (&contexts) |*context| {
        context.* = .{
            .manager = &manager,
            .embedding = &embedding,
            .layers = &layers,
            .output_norm = &attn_norm,
            .head = &head,
            .config = config,
            .tokens = &tokens,
            .logits = try allocator.alloc([vocab]f32, tokens.len),
        };
    }
    defer {
        for (&contexts) |*context| allocator.free(context.logits);
    }
    var threads: [contexts.len]std.Thread = undefined;
    for (&threads, &contexts) |*thread, *context| thread.* = try std.Thread.spawn(.{}, GenerationContext.run, .{context});
    for (&threads) |*thread| thread.join();

    for (&contexts) |*context| {
        try std.testing.expect(!context.failed);
        for (context.logits, baseline) |actual, expected| {
            for (actual, expected) |a, e| try std.testing.expectEqual(e, a);
        }
    }
    const metrics = manager.metrics();
    try std.testing.expect(metrics.peak_resident_bytes <= granularity * 16);
}

test "CPU execution boundary rejects mismatched FFN shapes" {
    var dimensions = [_]u64{0} ** gguf.max_dimensions;
    dimensions[0] = 4;
    dimensions[1] = 3;
    const gate = gguf.TensorDescriptor{
        .handle = .{ .id = 1 },
        .name = "gate",
        .file_offset = 0,
        .byte_len = 48,
        .ggml_type = gguf.type_f32,
        .n_dimensions = 2,
        .dimensions = dimensions,
    };
    var wrong_down = gate;
    wrong_down.handle = .{ .id = 2 };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var file = try tmp.dir.createFile("shape.bin", .{});
    try file.writeAll(&([_]u8{0} ** 96));
    file.close();
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "shape.bin");
    defer std.testing.allocator.free(path);
    const path_z = try std.testing.allocator.dupeZ(u8, path);
    defer std.testing.allocator.free(path_z);
    var store = try residency.BackingStore.open(path_z);
    defer store.close();
    var manager = try residency.Manager.init(std.testing.allocator, &store, try residency.mappingGranularity());
    defer manager.deinit();
    var executor = try CpuExecutor.init(std.testing.allocator, &manager, 4, 3, 1);
    defer executor.deinit();

    var output: [4]f32 = undefined;
    try std.testing.expectError(
        Error.InvalidExecutionShape,
        executor.ffnSwiGlu(&gate, &gate, &wrong_down, &([_]f32{ 1, 2, 3, 4 }), &output),
    );
}
