const std = @import("std");
const residency = @import("residency.zig");
const gguf = @import("gguf_residency.zig");
const executor_mod = @import("residency_executor.zig");

pub const Error = executor_mod.Error || error{
    InvalidQwenConfig,
};

pub const Config = struct {
    hidden_size: usize,
    rms_epsilon: f32,
    state_size: usize,
    key_head_count: usize,
    value_head_count: usize,
    inner_size: usize,
    conv_kernel: usize,
    attention_head_count: usize,
    attention_kv_head_count: usize,
    attention_head_dim: usize,
    rope_dimension_count: usize,
    rope_theta: f32,
    expert_count: usize,
    expert_used_count: usize,

    pub fn fromMetadata(metadata: gguf.ExecutionMetadata) Error!Config {
        if (metadata.architecture != .qwen3next) return Error.InvalidQwenConfig;
        const hidden = metadata.embedding_length orelse return Error.InvalidQwenConfig;
        const state = metadata.ssm_state_size orelse return Error.InvalidQwenConfig;
        const key_heads = metadata.ssm_group_count orelse return Error.InvalidQwenConfig;
        const value_heads = metadata.ssm_time_step_rank orelse return Error.InvalidQwenConfig;
        const inner = metadata.ssm_inner_size orelse return Error.InvalidQwenConfig;
        const conv = metadata.ssm_conv_kernel orelse return Error.InvalidQwenConfig;
        const attention_heads = metadata.attention_head_count orelse return Error.InvalidQwenConfig;
        const attention_kv_heads = metadata.attention_kv_head_count orelse return Error.InvalidQwenConfig;
        const key_length = metadata.attention_key_length orelse return Error.InvalidQwenConfig;
        const value_length = metadata.attention_value_length orelse return Error.InvalidQwenConfig;
        const rope_dims = metadata.rope_dimension_count orelse return Error.InvalidQwenConfig;
        const experts = metadata.expert_count orelse return Error.InvalidQwenConfig;
        const experts_used = metadata.expert_used_count orelse return Error.InvalidQwenConfig;
        const epsilon = metadata.rms_epsilon orelse return Error.InvalidQwenConfig;
        if (hidden == 0 or state == 0 or key_heads == 0 or value_heads == 0 or
            value_heads % key_heads != 0 or inner != state * value_heads or conv < 2 or
            attention_heads == 0 or attention_kv_heads == 0 or attention_heads % attention_kv_heads != 0 or
            key_length == 0 or key_length != value_length or rope_dims == 0 or rope_dims > key_length or
            rope_dims % 2 != 0 or experts == 0 or experts_used == 0 or experts_used > experts or epsilon <= 0)
        {
            return Error.InvalidQwenConfig;
        }
        return .{
            .hidden_size = hidden,
            .rms_epsilon = epsilon,
            .state_size = state,
            .key_head_count = key_heads,
            .value_head_count = value_heads,
            .inner_size = inner,
            .conv_kernel = conv,
            .attention_head_count = attention_heads,
            .attention_kv_head_count = attention_kv_heads,
            .attention_head_dim = key_length,
            .rope_dimension_count = rope_dims,
            .rope_theta = metadata.rope_theta orelse 1_000_000.0,
            .expert_count = experts,
            .expert_used_count = experts_used,
        };
    }

    pub fn convChannels(self: Config) usize {
        return 2 * self.state_size * self.key_head_count + self.inner_size;
    }

    pub fn convHistory(self: Config) usize {
        return self.conv_kernel - 1;
    }
};

/// Writable recurrent state for one Qwen3-Next linear-attention layer and one
/// sequence. It is independent from the immutable mapped-weight budget.
pub const DeltaNetCache = struct {
    allocator: std.mem.Allocator,
    conv_history: []f32,
    recurrent: []f32,
    position: usize = 0,

    pub fn init(allocator: std.mem.Allocator, config: Config) Error!DeltaNetCache {
        const conv_elements = std.math.mul(usize, config.convHistory(), config.convChannels()) catch return Error.OutOfMemory;
        const state_matrix = std.math.mul(usize, config.state_size, config.state_size) catch return Error.OutOfMemory;
        const recurrent_elements = std.math.mul(usize, state_matrix, config.value_head_count) catch return Error.OutOfMemory;
        const conv_history = allocator.alloc(f32, conv_elements) catch return Error.OutOfMemory;
        errdefer allocator.free(conv_history);
        const recurrent = allocator.alloc(f32, recurrent_elements) catch return Error.OutOfMemory;
        @memset(conv_history, 0);
        @memset(recurrent, 0);
        return .{ .allocator = allocator, .conv_history = conv_history, .recurrent = recurrent };
    }

    pub fn deinit(self: *DeltaNetCache) void {
        self.allocator.free(self.conv_history);
        self.allocator.free(self.recurrent);
        self.* = undefined;
    }

    pub fn byteLen(self: *const DeltaNetCache) usize {
        return (self.conv_history.len + self.recurrent.len) * @sizeOf(f32);
    }
};

pub const LinearWeights = struct {
    attention_norm: *const gguf.TensorDescriptor,
    qkv: *const gguf.TensorDescriptor,
    z_gate: *const gguf.TensorDescriptor,
    beta_alpha: *const gguf.TensorDescriptor,
    conv1d: *const gguf.TensorDescriptor,
    dt_bias: *const gguf.TensorDescriptor,
    decay: *const gguf.TensorDescriptor,
    state_norm: *const gguf.TensorDescriptor,
    output: *const gguf.TensorDescriptor,
};

pub const FullAttentionWeights = struct {
    attention_norm: *const gguf.TensorDescriptor,
    query_gate: *const gguf.TensorDescriptor,
    key: *const gguf.TensorDescriptor,
    value: *const gguf.TensorDescriptor,
    query_norm: *const gguf.TensorDescriptor,
    key_norm: *const gguf.TensorDescriptor,
    output: *const gguf.TensorDescriptor,
};

pub const FullAttentionCache = struct {
    allocator: std.mem.Allocator,
    keys: []f32,
    values: []f32,
    scores: []f32,
    capacity: usize,
    kv_width: usize,
    len: usize = 0,

    pub fn init(allocator: std.mem.Allocator, capacity: usize, config: Config) Error!FullAttentionCache {
        const kv_width = std.math.mul(usize, config.attention_kv_head_count, config.attention_head_dim) catch return Error.OutOfMemory;
        const elements = std.math.mul(usize, capacity, kv_width) catch return Error.OutOfMemory;
        const keys = allocator.alloc(f32, elements) catch return Error.OutOfMemory;
        errdefer allocator.free(keys);
        const values = allocator.alloc(f32, elements) catch return Error.OutOfMemory;
        errdefer allocator.free(values);
        const scores = allocator.alloc(f32, capacity) catch return Error.OutOfMemory;
        return .{ .allocator = allocator, .keys = keys, .values = values, .scores = scores, .capacity = capacity, .kv_width = kv_width };
    }

    pub fn deinit(self: *FullAttentionCache) void {
        self.allocator.free(self.keys);
        self.allocator.free(self.values);
        self.allocator.free(self.scores);
        self.* = undefined;
    }

    pub fn byteLen(self: *const FullAttentionCache) usize {
        return (self.keys.len + self.values.len + self.scores.len) * @sizeOf(f32);
    }
};

pub const MoeWeights = struct {
    post_attention_norm: *const gguf.TensorDescriptor,
    router: *const gguf.TensorDescriptor,
    gate_experts: *const gguf.TensorDescriptor,
    up_experts: *const gguf.TensorDescriptor,
    down_experts: *const gguf.TensorDescriptor,
    shared_router: *const gguf.TensorDescriptor,
    shared_gate: *const gguf.TensorDescriptor,
    shared_up: *const gguf.TensorDescriptor,
    shared_down: *const gguf.TensorDescriptor,
};
pub const LayerWeights = union(enum) {
    recurrent: struct { attention: LinearWeights, moe: MoeWeights },
    full_attention: struct { attention: FullAttentionWeights, moe: MoeWeights },
};

pub const LayerCache = union(enum) {
    recurrent: DeltaNetCache,
    full_attention: FullAttentionCache,

    pub fn deinit(self: *LayerCache) void {
        switch (self.*) {
            .recurrent => |*cache| cache.deinit(),
            .full_attention => |*cache| cache.deinit(),
        }
    }

    pub fn byteLen(self: *const LayerCache) usize {
        return switch (self.*) {
            .recurrent => |*cache| cache.byteLen(),
            .full_attention => |*cache| cache.byteLen(),
        };
    }
};

pub const Workspace = struct {
    allocator: std.mem.Allocator,
    normalized: []f32,
    qkv: []f32,
    z: []f32,
    beta_alpha: []f32,
    delta_output: []f32,
    projected: []f32,
    routed: []f32,
    shared: []f32,
    attention_query_gate: []f32,
    attention_context: []f32,

    pub fn init(allocator: std.mem.Allocator, config: Config) Error!Workspace {
        const normalized = allocator.alloc(f32, config.hidden_size) catch return Error.OutOfMemory;
        errdefer allocator.free(normalized);
        const qkv = allocator.alloc(f32, config.convChannels()) catch return Error.OutOfMemory;
        errdefer allocator.free(qkv);
        const z = allocator.alloc(f32, config.inner_size) catch return Error.OutOfMemory;
        errdefer allocator.free(z);
        const beta_alpha = allocator.alloc(f32, 2 * config.value_head_count) catch return Error.OutOfMemory;
        errdefer allocator.free(beta_alpha);
        const delta_output = allocator.alloc(f32, config.inner_size) catch return Error.OutOfMemory;
        errdefer allocator.free(delta_output);
        const projected = allocator.alloc(f32, config.hidden_size) catch return Error.OutOfMemory;
        errdefer allocator.free(projected);
        const routed = allocator.alloc(f32, config.hidden_size) catch return Error.OutOfMemory;
        errdefer allocator.free(routed);
        const shared = allocator.alloc(f32, config.hidden_size) catch return Error.OutOfMemory;
        errdefer allocator.free(shared);
        const query_gate_len = std.math.mul(usize, 2, config.attention_head_count * config.attention_head_dim) catch return Error.OutOfMemory;
        const attention_query_gate = allocator.alloc(f32, query_gate_len) catch return Error.OutOfMemory;
        errdefer allocator.free(attention_query_gate);
        const attention_context = allocator.alloc(f32, config.attention_head_count * config.attention_head_dim) catch return Error.OutOfMemory;
        return .{
            .allocator = allocator,
            .normalized = normalized,
            .qkv = qkv,
            .z = z,
            .beta_alpha = beta_alpha,
            .delta_output = delta_output,
            .projected = projected,
            .routed = routed,
            .shared = shared,
            .attention_query_gate = attention_query_gate,
            .attention_context = attention_context,
        };
    }

    pub fn deinit(self: *Workspace) void {
        self.allocator.free(self.normalized);
        self.allocator.free(self.qkv);
        self.allocator.free(self.z);
        self.allocator.free(self.beta_alpha);
        self.allocator.free(self.delta_output);
        self.allocator.free(self.projected);
        self.allocator.free(self.routed);
        self.allocator.free(self.shared);
        self.allocator.free(self.attention_query_gate);
        self.allocator.free(self.attention_context);
        self.* = undefined;
    }

    pub fn byteLen(self: *const Workspace) usize {
        return (self.normalized.len + self.qkv.len + self.z.len + self.beta_alpha.len +
            self.delta_output.len + self.projected.len + self.routed.len + self.shared.len +
            self.attention_query_gate.len + self.attention_context.len) * @sizeOf(f32);
    }
};

fn sigmoid(value: f32) f32 {
    return 1.0 / (1.0 + @exp(-value));
}

fn softplus(value: f32) f32 {
    if (value > 20.0) return value;
    if (value < -20.0) return @exp(value);
    return @log(1.0 + @exp(value));
}

fn silu(value: f32) f32 {
    return value * sigmoid(value);
}

fn l2Normalize(values: []f32, epsilon: f32) void {
    var sum: f64 = 0;
    for (values) |value| sum += @as(f64, value * value);
    // Match ggml_compute_forward_l2_norm_f32: epsilon is a lower bound on
    // the vector norm, not a term added under the square root.
    const norm: f32 = @floatCast(@sqrt(sum));
    const scale = 1.0 / @max(norm, epsilon);
    for (values) |*value| value.* *= scale;
}

fn dotF32Vector(manager: *residency.Manager, descriptor: *const gguf.TensorDescriptor, input: []const f32) Error!f32 {
    if (descriptor.ggml_type != gguf.type_f32 or descriptor.n_dimensions != 1 or
        descriptor.dimensions[0] != input.len or descriptor.byte_len != input.len * @sizeOf(f32))
    {
        return Error.InvalidExecutionShape;
    }
    var view = try manager.acquire(descriptor.handle);
    defer view.release();
    const values: []align(1) const f32 = std.mem.bytesAsSlice(f32, view.bytes());
    var result: f32 = 0;
    for (values, input) |weight, value| result += weight * value;
    return result;
}

pub fn deltaNetStep(
    config: Config,
    cache: *DeltaNetCache,
    qkv: []f32,
    z: []const f32,
    beta_alpha: []const f32,
    conv_weights: []align(1) const f32,
    dt_bias: []align(1) const f32,
    decay: []align(1) const f32,
    norm_weights: []align(1) const f32,
    output: []f32,
) Error!void {
    const channels = config.convChannels();
    const dim = config.state_size;
    const ratio = config.value_head_count / config.key_head_count;
    if (qkv.len != channels or z.len != config.inner_size or beta_alpha.len != 2 * config.value_head_count or
        conv_weights.len != channels * config.conv_kernel or dt_bias.len != config.value_head_count or
        decay.len != config.value_head_count or norm_weights.len != dim or output.len != config.inner_size or
        cache.conv_history.len != config.convHistory() * channels or
        cache.recurrent.len != dim * dim * config.value_head_count)
    {
        return Error.InvalidExecutionShape;
    }

    // GGML stores ssm_conv1d as [kernel, channels], so each channel owns one
    // contiguous kernel row. The cache keeps raw pre-convolution Q/K/V values.
    const history = config.convHistory();
    for (0..channels) |channel| {
        var convolved: f32 = 0;
        for (0..history) |tap| convolved += cache.conv_history[channel * history + tap] * conv_weights[channel * config.conv_kernel + tap];
        convolved += qkv[channel] * conv_weights[channel * config.conv_kernel + history];
        if (history > 1) {
            std.mem.copyForwards(f32, cache.conv_history[channel * history ..][0 .. history - 1], cache.conv_history[channel * history + 1 ..][0 .. history - 1]);
        }
        cache.conv_history[channel * history + history - 1] = qkv[channel];
        qkv[channel] = silu(convolved);
    }

    const key_width = dim * config.key_head_count;
    const q_all = qkv[0..key_width];
    const k_all = qkv[key_width .. 2 * key_width];
    const v_all = qkv[2 * key_width ..];
    for (0..config.key_head_count) |head| {
        l2Normalize(q_all[head * dim ..][0..dim], config.rms_epsilon);
        l2Normalize(k_all[head * dim ..][0..dim], config.rms_epsilon);
    }

    const query_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(dim)));
    for (0..config.value_head_count) |value_head| {
        const key_head = value_head / ratio;
        const q = q_all[key_head * dim ..][0..dim];
        const k = k_all[key_head * dim ..][0..dim];
        const v = v_all[value_head * dim ..][0..dim];
        const group = key_head * ratio;
        const within_group = value_head - group;
        const ba_base = key_head * 2 * ratio;
        const beta = sigmoid(beta_alpha[ba_base + within_group]);
        const gate = decay[value_head] * softplus(beta_alpha[ba_base + ratio + within_group] + dt_bias[value_head]);
        const gate_exp = @exp(gate);
        const state = cache.recurrent[value_head * dim * dim ..][0 .. dim * dim];

        // S is laid out [value, key] with key contiguous, matching GGML's
        // physical [D_key, D_value, H_value] state tensor.
        for (0..dim) |value_index| {
            const row = state[value_index * dim ..][0..dim];
            var prediction: f32 = 0;
            for (row, k) |state_value, key_value| prediction += state_value * gate_exp * key_value;
            const delta = beta * (v[value_index] - prediction);
            for (row, k) |*state_value, key_value| state_value.* = state_value.* * gate_exp + delta * key_value;
        }

        const head_output = output[value_head * dim ..][0..dim];
        for (head_output, 0..) |*result, value_index| {
            const row = state[value_index * dim ..][0..dim];
            var sum: f32 = 0;
            for (row, q) |state_value, query_value| sum += state_value * (query_value * query_scale);
            result.* = sum;
        }

        var sum_squares: f64 = 0;
        for (head_output) |value| sum_squares += @as(f64, value * value);
        const mean: f32 = @floatCast(sum_squares / @as(f64, @floatFromInt(dim)));
        const norm_scale = 1.0 / @sqrt(mean + config.rms_epsilon);
        const z_head = z[value_head * dim ..][0..dim];
        for (head_output, norm_weights, z_head) |*value, norm_weight, gate_value| value.* = value.* * norm_scale * norm_weight * silu(gate_value);
    }
    cache.position += 1;
}

pub fn linearAttentionSingleToken(
    executor: *executor_mod.CpuExecutor,
    config: Config,
    weights: LinearWeights,
    cache: *DeltaNetCache,
    workspace: *Workspace,
    state: []f32,
) Error!void {
    if (state.len != config.hidden_size) return Error.InvalidExecutionShape;
    try executor.rmsNorm(weights.attention_norm, state, workspace.normalized, config.rms_epsilon);
    try executor.matVec(weights.qkv, workspace.normalized, workspace.qkv);
    try executor.matVec(weights.z_gate, workspace.normalized, workspace.z);
    try executor.matVec(weights.beta_alpha, workspace.normalized, workspace.beta_alpha);

    const channels = config.convChannels();
    const conv_elements = channels * config.conv_kernel;
    if (weights.conv1d.ggml_type != gguf.type_f32 or weights.conv1d.n_dimensions != 2 or
        weights.conv1d.dimensions[0] != config.conv_kernel or weights.conv1d.dimensions[1] != channels or
        weights.conv1d.byte_len != conv_elements * @sizeOf(f32) or
        weights.dt_bias.ggml_type != gguf.type_f32 or weights.dt_bias.n_dimensions != 1 or weights.dt_bias.dimensions[0] != config.value_head_count or
        weights.decay.ggml_type != gguf.type_f32 or weights.decay.n_dimensions != 1 or weights.decay.dimensions[0] != config.value_head_count or
        weights.state_norm.ggml_type != gguf.type_f32 or weights.state_norm.n_dimensions != 1 or weights.state_norm.dimensions[0] != config.state_size)
    {
        return Error.InvalidExecutionShape;
    }
    var conv_view = try executor.manager.acquire(weights.conv1d.handle);
    defer conv_view.release();
    var dt_view = try executor.manager.acquire(weights.dt_bias.handle);
    defer dt_view.release();
    var decay_view = try executor.manager.acquire(weights.decay.handle);
    defer decay_view.release();
    var norm_view = try executor.manager.acquire(weights.state_norm.handle);
    defer norm_view.release();
    const conv_weights: []align(1) const f32 = std.mem.bytesAsSlice(f32, conv_view.bytes());
    const dt_bias: []align(1) const f32 = std.mem.bytesAsSlice(f32, dt_view.bytes());
    const decay: []align(1) const f32 = std.mem.bytesAsSlice(f32, decay_view.bytes());
    const norm_weights: []align(1) const f32 = std.mem.bytesAsSlice(f32, norm_view.bytes());

    try deltaNetStep(config, cache, workspace.qkv, workspace.z, workspace.beta_alpha, conv_weights, dt_bias, decay, norm_weights, workspace.delta_output);
    try executor.matVec(weights.output, workspace.delta_output, workspace.projected);
    for (state, workspace.projected) |*value, projected| value.* += projected;
}

fn rmsNormHeads(values: []f32, weights: []align(1) const f32, head_count: usize, head_dim: usize, epsilon: f32) void {
    for (0..head_count) |head| {
        const head_values = values[head * head_dim ..][0..head_dim];
        var sum_squares: f64 = 0;
        for (head_values) |value| sum_squares += @as(f64, value * value);
        const mean: f32 = @floatCast(sum_squares / @as(f64, @floatFromInt(head_dim)));
        const scale = 1.0 / @sqrt(mean + epsilon);
        for (head_values, weights) |*value, weight| value.* *= scale * weight;
    }
}

fn applyPartialRope(values: []f32, head_count: usize, head_dim: usize, rope_dims: usize, position: usize, theta: f32) void {
    const position_f: f32 = @floatFromInt(position);
    for (0..head_count) |head| {
        const head_values = values[head * head_dim ..][0..head_dim];
        var pair: usize = 0;
        while (pair < rope_dims) : (pair += 2) {
            const exponent = @as(f32, @floatFromInt(pair)) / @as(f32, @floatFromInt(rope_dims));
            const angle = position_f / std.math.pow(f32, theta, exponent);
            const cosine = @cos(angle);
            const sine = @sin(angle);
            const first = head_values[pair];
            const second = head_values[pair + 1];
            head_values[pair] = first * cosine - second * sine;
            head_values[pair + 1] = first * sine + second * cosine;
        }
    }
}

pub fn fullAttentionSingleToken(
    executor: *executor_mod.CpuExecutor,
    config: Config,
    weights: FullAttentionWeights,
    cache: *FullAttentionCache,
    workspace: *Workspace,
    state: []f32,
) Error!void {
    const query_width = config.attention_head_count * config.attention_head_dim;
    const kv_width = config.attention_kv_head_count * config.attention_head_dim;
    if (state.len != config.hidden_size or query_width != config.inner_size or
        cache.kv_width != kv_width or cache.len >= cache.capacity)
    {
        return if (cache.len >= cache.capacity) Error.KvCacheFull else Error.InvalidExecutionShape;
    }
    try executor.rmsNorm(weights.attention_norm, state, workspace.normalized, config.rms_epsilon);
    try executor.matVec(weights.query_gate, workspace.normalized, workspace.qkv);
    const query = workspace.attention_query_gate[0..query_width];
    const gate = workspace.attention_query_gate[query_width .. 2 * query_width];
    for (0..config.attention_head_count) |head| {
        const raw_head = workspace.qkv[head * 2 * config.attention_head_dim ..][0 .. 2 * config.attention_head_dim];
        @memcpy(query[head * config.attention_head_dim ..][0..config.attention_head_dim], raw_head[0..config.attention_head_dim]);
        @memcpy(gate[head * config.attention_head_dim ..][0..config.attention_head_dim], raw_head[config.attention_head_dim .. 2 * config.attention_head_dim]);
    }
    const key = cache.keys[cache.len * kv_width ..][0..kv_width];
    const value = cache.values[cache.len * kv_width ..][0..kv_width];
    try executor.matVec(weights.key, workspace.normalized, key);
    try executor.matVec(weights.value, workspace.normalized, value);

    if (weights.query_norm.ggml_type != gguf.type_f32 or weights.query_norm.n_dimensions != 1 or weights.query_norm.dimensions[0] != config.attention_head_dim or
        weights.key_norm.ggml_type != gguf.type_f32 or weights.key_norm.n_dimensions != 1 or weights.key_norm.dimensions[0] != config.attention_head_dim)
    {
        return Error.InvalidExecutionShape;
    }
    var qnorm_view = try executor.manager.acquire(weights.query_norm.handle);
    defer qnorm_view.release();
    var knorm_view = try executor.manager.acquire(weights.key_norm.handle);
    defer knorm_view.release();
    const qnorm: []align(1) const f32 = std.mem.bytesAsSlice(f32, qnorm_view.bytes());
    const knorm: []align(1) const f32 = std.mem.bytesAsSlice(f32, knorm_view.bytes());
    rmsNormHeads(query, qnorm, config.attention_head_count, config.attention_head_dim, config.rms_epsilon);
    rmsNormHeads(key, knorm, config.attention_kv_head_count, config.attention_head_dim, config.rms_epsilon);
    applyPartialRope(query, config.attention_head_count, config.attention_head_dim, config.rope_dimension_count, cache.len, config.rope_theta);
    applyPartialRope(key, config.attention_kv_head_count, config.attention_head_dim, config.rope_dimension_count, cache.len, config.rope_theta);

    const group_size = config.attention_head_count / config.attention_kv_head_count;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(config.attention_head_dim)));
    const token_count = cache.len + 1;
    const context = workspace.attention_context[0..query_width];
    for (0..config.attention_head_count) |head| {
        const query_head = query[head * config.attention_head_dim ..][0..config.attention_head_dim];
        const kv_head = head / group_size;
        const scores = cache.scores[0..token_count];
        var maximum: f32 = -std.math.inf(f32);
        for (scores, 0..) |*score, token| {
            const cached_key = cache.keys[token * kv_width + kv_head * config.attention_head_dim ..][0..config.attention_head_dim];
            var dot: f32 = 0;
            for (query_head, cached_key) |q, k| dot += q * k;
            score.* = dot * scale;
            maximum = @max(maximum, score.*);
        }
        var denominator: f32 = 0;
        for (scores) |*score| {
            score.* = @exp(score.* - maximum);
            denominator += score.*;
        }
        const context_head = context[head * config.attention_head_dim ..][0..config.attention_head_dim];
        @memset(context_head, 0);
        for (scores, 0..) |score, token| {
            const cached_value = cache.values[token * kv_width + kv_head * config.attention_head_dim ..][0..config.attention_head_dim];
            const probability = score / denominator;
            for (context_head, cached_value) |*result, cached| result.* += probability * cached;
        }
        const gate_head = gate[head * config.attention_head_dim ..][0..config.attention_head_dim];
        for (context_head, gate_head) |*result, gate_value| result.* *= sigmoid(gate_value);
    }
    try executor.matVec(weights.output, context, workspace.projected);
    for (state, workspace.projected) |*state_value, projected| state_value.* += projected;
    cache.len = token_count;
}

pub fn moeSingleToken(
    executor: *executor_mod.CpuExecutor,
    config: Config,
    weights: MoeWeights,
    workspace: *Workspace,
    state: []f32,
) Error!void {
    if (state.len != config.hidden_size) return Error.InvalidExecutionShape;
    try executor.rmsNorm(weights.post_attention_norm, state, workspace.normalized, config.rms_epsilon);
    try executor.moeSwiGlu(weights.router, weights.gate_experts, weights.up_experts, weights.down_experts, config.expert_used_count, workspace.normalized, workspace.routed);
    try executor.ffnSwiGlu(weights.shared_gate, weights.shared_up, weights.shared_down, workspace.normalized, workspace.shared);
    const shared_scale = sigmoid(try dotF32Vector(executor.manager, weights.shared_router, workspace.normalized));
    for (state, workspace.routed, workspace.shared) |*value, routed, shared| value.* += routed + shared_scale * shared;
}

pub fn initLayerCaches(
    allocator: std.mem.Allocator,
    config: Config,
    layer_count: usize,
    full_attention_interval: usize,
    context_capacity: usize,
) Error![]LayerCache {
    if (layer_count == 0 or full_attention_interval == 0 or context_capacity == 0) return Error.InvalidQwenConfig;
    const caches = allocator.alloc(LayerCache, layer_count) catch return Error.OutOfMemory;
    var initialized: usize = 0;
    errdefer {
        for (caches[0..initialized]) |*cache| cache.deinit();
        allocator.free(caches);
    }
    for (caches, 0..) |*cache, layer| {
        if ((layer + 1) % full_attention_interval == 0) {
            cache.* = .{ .full_attention = try FullAttentionCache.init(allocator, context_capacity, config) };
        } else {
            cache.* = .{ .recurrent = try DeltaNetCache.init(allocator, config) };
        }
        initialized += 1;
    }
    return caches;
}

pub fn deinitLayerCaches(allocator: std.mem.Allocator, caches: []LayerCache) void {
    for (caches) |*cache| cache.deinit();
    allocator.free(caches);
}

pub fn modelSingleToken(
    executor: *executor_mod.CpuExecutor,
    config: Config,
    embedding: *const gguf.TensorDescriptor,
    layers: []const LayerWeights,
    output_norm: *const gguf.TensorDescriptor,
    output_weight: *const gguf.TensorDescriptor,
    token: usize,
    caches: []LayerCache,
    workspace: *Workspace,
    state: []f32,
    logits: []f32,
) Error!void {
    if (layers.len == 0 or layers.len != caches.len or state.len != config.hidden_size) return Error.InvalidExecutionShape;
    try executor.tokenEmbedding(embedding, token, state);
    for (layers, caches) |layer, *cache| switch (layer) {
        .recurrent => |weights| switch (cache.*) {
            .recurrent => |*delta_cache| {
                try linearAttentionSingleToken(executor, config, weights.attention, delta_cache, workspace, state);
                try moeSingleToken(executor, config, weights.moe, workspace, state);
            },
            else => return Error.InvalidExecutionShape,
        },
        .full_attention => |weights| switch (cache.*) {
            .full_attention => |*attention_cache| {
                try fullAttentionSingleToken(executor, config, weights.attention, attention_cache, workspace, state);
                try moeSingleToken(executor, config, weights.moe, workspace, state);
            },
            else => return Error.InvalidExecutionShape,
        },
    };
    try executor.rmsNorm(output_norm, state, workspace.normalized, config.rms_epsilon);
    try executor.matVec(output_weight, workspace.normalized, logits);
}

test "DeltaNet recurrence preserves convolution history and matches scalar reference" {
    const allocator = std.testing.allocator;
    const config = Config{
        .hidden_size = 4,
        .rms_epsilon = 1e-6,
        .state_size = 2,
        .key_head_count = 1,
        .value_head_count = 2,
        .inner_size = 4,
        .conv_kernel = 2,
        .attention_head_count = 2,
        .attention_kv_head_count = 1,
        .attention_head_dim = 2,
        .rope_dimension_count = 2,
        .rope_theta = 10_000,
        .expert_count = 2,
        .expert_used_count = 1,
    };
    var cache = try DeltaNetCache.init(allocator, config);
    defer cache.deinit();
    const channels = config.convChannels();
    var conv_weights = try allocator.alloc(f32, channels * config.conv_kernel);
    defer allocator.free(conv_weights);
    for (0..channels) |channel| {
        conv_weights[channel * 2] = 0.25;
        conv_weights[channel * 2 + 1] = 0.75;
    }
    const dt_bias = [_]f32{ 0.1, -0.2 };
    const decay = [_]f32{ -0.5, -0.25 };
    const norm = [_]f32{ 1.0, 0.75 };
    const beta_alpha = [_]f32{ 0.2, -0.1, 0.3, 0.4 };
    const z = [_]f32{ 0.5, -0.25, 0.75, 0.125 };
    var first = [_]f32{ 0.4, -0.2, 0.1, 0.3, -0.5, 0.7, 0.2, -0.1 };
    var first_output: [4]f32 = undefined;
    try deltaNetStep(config, &cache, &first, &z, &beta_alpha, conv_weights, &dt_bias, &decay, &norm, &first_output);
    try std.testing.expectEqual(@as(usize, 1), cache.position);
    for (first_output) |value| try std.testing.expect(std.math.isFinite(value));
    const history_after_first = try allocator.dupe(f32, cache.conv_history);
    defer allocator.free(history_after_first);

    var second = [_]f32{ -0.1, 0.6, 0.2, -0.4, 0.8, -0.3, 0.5, 0.25 };
    var second_output: [4]f32 = undefined;
    try deltaNetStep(config, &cache, &second, &z, &beta_alpha, conv_weights, &dt_bias, &decay, &norm, &second_output);
    try std.testing.expectEqual(@as(usize, 2), cache.position);
    for (second_output) |value| try std.testing.expect(std.math.isFinite(value));
    try std.testing.expect(!std.mem.eql(u8, std.mem.sliceAsBytes(&first_output), std.mem.sliceAsBytes(&second_output)));
    try std.testing.expect(!std.mem.eql(u8, std.mem.sliceAsBytes(history_after_first), std.mem.sliceAsBytes(cache.conv_history)));
}

test "Qwen3-Next metadata config validates hybrid dimensions" {
    const config = try Config.fromMetadata(.{
        .architecture = .qwen3next,
        .attention_head_count = 16,
        .attention_kv_head_count = 2,
        .attention_key_length = 256,
        .attention_value_length = 256,
        .embedding_length = 2048,
        .rms_epsilon = 1e-6,
        .rope_theta = 1_000_000,
        .rope_dimension_count = 64,
        .expert_count = 512,
        .expert_used_count = 10,
        .ssm_conv_kernel = 4,
        .ssm_state_size = 128,
        .ssm_group_count = 16,
        .ssm_time_step_rank = 32,
        .ssm_inner_size = 4096,
    });
    try std.testing.expectEqual(@as(usize, 8192), config.convChannels());
    try std.testing.expectEqual(@as(usize, 3), config.convHistory());
    try std.testing.expectEqual(@as(usize, 4096), config.attention_head_count * config.attention_head_dim);
}
