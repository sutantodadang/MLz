const std = @import("std");
const residency = @import("residency.zig");

const c = @cImport({
    @cInclude("gguf.h");
});

pub const Error = residency.Error || error{
    InvalidGguf,
    UnsupportedGguf,
    DuplicateTensorName,
};

/// Immutable metadata needed to connect one GGUF tensor to the bounded
/// residency manager. `file_offset` is absolute, not relative to GGUF's data
/// section, so it can be passed directly to `Manager.register`.
pub const max_dimensions: usize = c.GGML_MAX_DIMS;
pub const type_f32: u32 = @intCast(c.GGML_TYPE_F32);
pub const type_q4_0: u32 = @intCast(c.GGML_TYPE_Q4_0);
pub const type_q2_k: u32 = @intCast(c.GGML_TYPE_Q2_K);
pub const type_q3_k: u32 = @intCast(c.GGML_TYPE_Q3_K);
pub const type_q4_k: u32 = @intCast(c.GGML_TYPE_Q4_K);
pub const type_q6_k: u32 = @intCast(c.GGML_TYPE_Q6_K);
pub const type_mxfp4: u32 = @intCast(c.GGML_TYPE_MXFP4);

pub const TensorDescriptor = struct {
    handle: residency.TensorHandle,
    name: []const u8,
    file_offset: u64,
    byte_len: usize,
    ggml_type: u32,
    n_dimensions: u8,
    dimensions: [max_dimensions]u64,
};

pub const Architecture = enum {
    unknown,
    llama,
    qwen3next,
};

/// Optional execution metadata used by the proof execution adapters. Models
/// remain indexable even when architecture-specific keys are absent.
pub const ExecutionMetadata = struct {
    architecture: Architecture = .unknown,
    attention_head_count: ?u32 = null,
    attention_kv_head_count: ?u32 = null,
    attention_key_length: ?u32 = null,
    attention_value_length: ?u32 = null,
    block_count: ?u32 = null,
    context_length: ?u32 = null,
    embedding_length: ?u32 = null,
    rms_epsilon: ?f32 = null,
    rope_theta: ?f32 = null,
    rope_dimension_count: ?u32 = null,
    expert_count: ?u32 = null,
    expert_used_count: ?u32 = null,
    expert_feed_forward_length: ?u32 = null,
    shared_expert_feed_forward_length: ?u32 = null,
    ssm_conv_kernel: ?u32 = null,
    ssm_state_size: ?u32 = null,
    ssm_group_count: ?u32 = null,
    ssm_time_step_rank: ?u32 = null,
    ssm_inner_size: ?u32 = null,
    full_attention_interval: ?u32 = null,
};

fn optionalU32(ctx: *c.gguf_context, key: [*:0]const u8) ?u32 {
    const id = c.gguf_find_key(ctx, key);
    if (id < 0 or c.gguf_get_kv_type(ctx, id) != c.GGUF_TYPE_UINT32) return null;
    return c.gguf_get_val_u32(ctx, id);
}

fn optionalF32(ctx: *c.gguf_context, key: [*:0]const u8) ?f32 {
    const id = c.gguf_find_key(ctx, key);
    if (id < 0 or c.gguf_get_kv_type(ctx, id) != c.GGUF_TYPE_FLOAT32) return null;
    return c.gguf_get_val_f32(ctx, id);
}

fn architecture(ctx: *c.gguf_context) Architecture {
    const id = c.gguf_find_key(ctx, "general.architecture");
    if (id < 0 or c.gguf_get_kv_type(ctx, id) != c.GGUF_TYPE_STRING) return .unknown;
    const value = c.gguf_get_val_str(ctx, id) orelse return .unknown;
    const name = std.mem.span(value);
    if (std.mem.eql(u8, name, "llama")) return .llama;
    if (std.mem.eql(u8, name, "qwen3next")) return .qwen3next;
    return .unknown;
}

/// Owns a validated tensor index for a GGUF file. Parsing uses gguf's official
/// metadata reader with data allocation disabled; tensor bytes remain solely in
/// the backing file and are faulted by `residency.Manager`.
pub const TensorIndex = struct {
    allocator: std.mem.Allocator,
    descriptors: []TensorDescriptor,
    by_name: std.StringHashMap(usize),
    execution: ExecutionMetadata,

    pub fn open(allocator: std.mem.Allocator, path_z: [:0]const u8, backing_size: u64) Error!TensorIndex {
        var tensor_ctx: ?*c.ggml_context = null;
        const params = c.gguf_init_params{
            .no_alloc = true,
            .ctx = &tensor_ctx,
        };
        const ctx = c.gguf_init_from_file(path_z.ptr, params) orelse return Error.InvalidGguf;
        defer c.gguf_free(ctx);
        defer if (tensor_ctx) |owned| c.ggml_free(owned);

        if (tensor_ctx == null) return Error.InvalidGguf;
        if (c.gguf_get_version(ctx) > c.GGUF_VERSION) return Error.UnsupportedGguf;
        const count_i64 = c.gguf_get_n_tensors(ctx);
        if (count_i64 < 0 or @as(u64, @intCast(count_i64)) > std.math.maxInt(usize)) {
            return Error.InvalidGguf;
        }
        const count: usize = @intCast(count_i64);
        const data_offset: u64 = @intCast(c.gguf_get_data_offset(ctx));

        const descriptors = allocator.alloc(TensorDescriptor, count) catch return Error.OutOfMemory;
        errdefer allocator.free(descriptors);
        var initialized: usize = 0;
        errdefer for (descriptors[0..initialized]) |descriptor| allocator.free(descriptor.name);

        var by_name = std.StringHashMap(usize).init(allocator);
        errdefer by_name.deinit();

        for (descriptors, 0..) |*descriptor, i| {
            const id: i64 = @intCast(i);
            const name_ptr = c.gguf_get_tensor_name(ctx, id) orelse return Error.InvalidGguf;
            const name = std.mem.span(name_ptr);
            if (name.len == 0 or by_name.contains(name)) return Error.DuplicateTensorName;

            const relative_offset: u64 = @intCast(c.gguf_get_tensor_offset(ctx, id));
            const byte_len: usize = c.gguf_get_tensor_size(ctx, id);
            const file_offset = std.math.add(u64, data_offset, relative_offset) catch return Error.InvalidGguf;
            if (byte_len == 0 or file_offset > backing_size or byte_len > backing_size - file_offset) {
                return Error.InvalidGguf;
            }

            const tensor = c.ggml_get_tensor(tensor_ctx.?, name_ptr) orelse return Error.InvalidGguf;
            const n_dimensions_i32 = c.ggml_n_dims(tensor);
            if (n_dimensions_i32 <= 0 or n_dimensions_i32 > max_dimensions) return Error.InvalidGguf;
            var dimensions: [max_dimensions]u64 = [_]u64{1} ** max_dimensions;
            for (0..@intCast(n_dimensions_i32)) |dimension| {
                if (tensor.*.ne[dimension] <= 0) return Error.InvalidGguf;
                dimensions[dimension] = @intCast(tensor.*.ne[dimension]);
            }

            const owned_name = allocator.dupe(u8, name) catch return Error.OutOfMemory;
            descriptor.* = .{
                .handle = .{ .id = @as(u64, @intCast(i)) + 1 },
                .name = owned_name,
                .file_offset = file_offset,
                .byte_len = byte_len,
                .ggml_type = @intCast(c.gguf_get_tensor_type(ctx, id)),
                .n_dimensions = @intCast(n_dimensions_i32),
                .dimensions = dimensions,
            };
            initialized += 1;
            by_name.put(owned_name, i) catch return Error.OutOfMemory;
        }

        const arch = architecture(ctx);
        const execution = switch (arch) {
            .llama => ExecutionMetadata{
                .architecture = arch,
                .attention_head_count = optionalU32(ctx, "llama.attention.head_count"),
                .attention_kv_head_count = optionalU32(ctx, "llama.attention.head_count_kv"),
                .block_count = optionalU32(ctx, "llama.block_count"),
                .context_length = optionalU32(ctx, "llama.context_length"),
                .embedding_length = optionalU32(ctx, "llama.embedding_length"),
                .rms_epsilon = optionalF32(ctx, "llama.attention.layer_norm_rms_epsilon"),
                .rope_theta = optionalF32(ctx, "llama.rope.freq_base"),
            },
            .qwen3next => ExecutionMetadata{
                .architecture = arch,
                .attention_head_count = optionalU32(ctx, "qwen3next.attention.head_count"),
                .attention_kv_head_count = optionalU32(ctx, "qwen3next.attention.head_count_kv"),
                .attention_key_length = optionalU32(ctx, "qwen3next.attention.key_length"),
                .attention_value_length = optionalU32(ctx, "qwen3next.attention.value_length"),
                .block_count = optionalU32(ctx, "qwen3next.block_count"),
                .context_length = optionalU32(ctx, "qwen3next.context_length"),
                .embedding_length = optionalU32(ctx, "qwen3next.embedding_length"),
                .rms_epsilon = optionalF32(ctx, "qwen3next.attention.layer_norm_rms_epsilon"),
                .rope_theta = optionalF32(ctx, "qwen3next.rope.freq_base"),
                .rope_dimension_count = optionalU32(ctx, "qwen3next.rope.dimension_count"),
                .expert_count = optionalU32(ctx, "qwen3next.expert_count"),
                .expert_used_count = optionalU32(ctx, "qwen3next.expert_used_count"),
                .expert_feed_forward_length = optionalU32(ctx, "qwen3next.expert_feed_forward_length"),
                .shared_expert_feed_forward_length = optionalU32(ctx, "qwen3next.expert_shared_feed_forward_length"),
                .ssm_conv_kernel = optionalU32(ctx, "qwen3next.ssm.conv_kernel"),
                .ssm_state_size = optionalU32(ctx, "qwen3next.ssm.state_size"),
                .ssm_group_count = optionalU32(ctx, "qwen3next.ssm.group_count"),
                .ssm_time_step_rank = optionalU32(ctx, "qwen3next.ssm.time_step_rank"),
                .ssm_inner_size = optionalU32(ctx, "qwen3next.ssm.inner_size"),
                .full_attention_interval = optionalU32(ctx, "qwen3next.full_attention_interval"),
            },
            .unknown => ExecutionMetadata{},
        };

        return .{
            .allocator = allocator,
            .descriptors = descriptors,
            .by_name = by_name,
            .execution = execution,
        };
    }

    pub fn deinit(self: *TensorIndex) void {
        self.by_name.deinit();
        for (self.descriptors) |descriptor| self.allocator.free(descriptor.name);
        self.allocator.free(self.descriptors);
        self.* = undefined;
    }

    pub fn get(self: *const TensorIndex, name: []const u8) ?*const TensorDescriptor {
        const index = self.by_name.get(name) orelse return null;
        return &self.descriptors[index];
    }

    /// Registers every indexed tensor with the manager. This is transactional:
    /// if one registration fails, tensors added by this call are removed again.
    pub fn registerAll(self: *const TensorIndex, manager: *residency.Manager) Error!void {
        var registered: usize = 0;
        errdefer {
            for (self.descriptors[0..registered]) |descriptor| {
                manager.unregister(descriptor.handle) catch {};
            }
        }
        for (self.descriptors) |descriptor| {
            try manager.register(descriptor.handle, descriptor.file_offset, descriptor.byte_len);
            registered += 1;
        }
    }
};

fn writeInt(comptime T: type, bytes: []u8, cursor: *usize, value: T) void {
    const size = @sizeOf(T);
    std.mem.writeInt(T, bytes[cursor.*..][0..size], value, .little);
    cursor.* += size;
}

fn writeString(bytes: []u8, cursor: *usize, value: []const u8) void {
    writeInt(u64, bytes, cursor, value.len);
    @memcpy(bytes[cursor.*..][0..value.len], value);
    cursor.* += value.len;
}

fn createTestGguf(tmp: *std.testing.TmpDir) ![:0]u8 {
    // Minimal valid GGUF v3 containing two F32 tensors. Tensor offsets are
    // relative to the aligned data section and each tensor starts on GGUF's
    // default 32-byte alignment.
    const data_offset: usize = 128;
    const file_len: usize = data_offset + 32 + 16;
    var bytes = [_]u8{0} ** file_len;
    var cursor: usize = 0;

    @memcpy(bytes[cursor..][0..4], "GGUF");
    cursor += 4;
    writeInt(u32, &bytes, &cursor, 3);
    writeInt(i64, &bytes, &cursor, 2);
    writeInt(i64, &bytes, &cursor, 0);

    writeString(&bytes, &cursor, "weight.a");
    writeInt(u32, &bytes, &cursor, 1);
    writeInt(i64, &bytes, &cursor, 4);
    writeInt(u32, &bytes, &cursor, @intCast(c.GGML_TYPE_F32));
    writeInt(u64, &bytes, &cursor, 0);

    writeString(&bytes, &cursor, "weight.b");
    writeInt(u32, &bytes, &cursor, 2);
    writeInt(i64, &bytes, &cursor, 2);
    writeInt(i64, &bytes, &cursor, 2);
    writeInt(u32, &bytes, &cursor, @intCast(c.GGML_TYPE_F32));
    writeInt(u64, &bytes, &cursor, 32);
    if (cursor > data_offset) return error.InvalidGguf;

    for (0..16) |i| bytes[data_offset + i] = @intCast(i);
    for (0..16) |i| bytes[data_offset + 32 + i] = @intCast(0x80 + i);

    var file = try tmp.dir.createFile("fixture.gguf", .{});
    defer file.close();
    try file.writeAll(&bytes);

    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "fixture.gguf");
    defer std.testing.allocator.free(path);
    return std.testing.allocator.dupeZ(u8, path);
}

test "indexes GGUF metadata and faults registered tensor bytes" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path_z = try createTestGguf(&tmp);
    defer std.testing.allocator.free(path_z);

    var store = try residency.BackingStore.open(path_z);
    defer store.close();
    var index = try TensorIndex.open(std.testing.allocator, path_z, store.size);
    defer index.deinit();

    try std.testing.expectEqual(@as(usize, 2), index.descriptors.len);
    const a = index.get("weight.a") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u64, 128), a.file_offset);
    try std.testing.expectEqual(@as(usize, 16), a.byte_len);
    try std.testing.expectEqual(@as(u32, @intCast(c.GGML_TYPE_F32)), a.ggml_type);
    try std.testing.expectEqual(@as(u8, 1), a.n_dimensions);
    try std.testing.expectEqual(@as(u64, 4), a.dimensions[0]);

    const b = index.get("weight.b") orelse return error.TestUnexpectedResult;
    try std.testing.expectEqual(@as(u64, 160), b.file_offset);
    try std.testing.expectEqual(@as(u8, 2), b.n_dimensions);
    try std.testing.expectEqualSlices(u64, &.{ 2, 2 }, b.dimensions[0..2]);
    try std.testing.expect(index.get("missing") == null);

    var manager = try residency.Manager.init(std.testing.allocator, &store, 1024);
    defer manager.deinit();
    try index.registerAll(&manager);

    var a_view = try manager.acquire(a.handle);
    try std.testing.expectEqualSlices(u8, &.{ 0, 1, 2, 3 }, a_view.bytes()[0..4]);
    a_view.release();
    var b_view = try manager.acquire(b.handle);
    try std.testing.expectEqualSlices(u8, &.{ 0x80, 0x81, 0x82, 0x83 }, b_view.bytes()[0..4]);
    b_view.release();
}

test "rejects GGUF descriptor whose tensor data exceeds backing file" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path_z = try createTestGguf(&tmp);
    defer std.testing.allocator.free(path_z);

    try std.testing.expectError(
        Error.InvalidGguf,
        TensorIndex.open(std.testing.allocator, path_z, 160 + 15),
    );
}

test "registerAll rolls back descriptors added before a conflict" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path_z = try createTestGguf(&tmp);
    defer std.testing.allocator.free(path_z);

    var store = try residency.BackingStore.open(path_z);
    defer store.close();
    var index = try TensorIndex.open(std.testing.allocator, path_z, store.size);
    defer index.deinit();
    var manager = try residency.Manager.init(std.testing.allocator, &store, 64);
    defer manager.deinit();

    try manager.register(.{ .id = 2 }, 0, 1);
    try std.testing.expectError(Error.DuplicateTensor, index.registerAll(&manager));
    try std.testing.expectError(Error.UnknownTensor, manager.state(.{ .id = 1 }));
    try std.testing.expectEqual(residency.Residency.non_resident, try manager.state(.{ .id = 2 }));
}
