const std = @import("std");
const residency = @import("residency.zig");
const gguf = @import("gguf_residency.zig");

const c = @cImport({
    @cInclude("ggml.h");
    @cInclude("ggml-cpu.h");
});

extern "c" fn dequantize_row_q2_K(x: ?*const anyopaque, y: [*]f32, k: i64) void;
extern "c" fn quantize_row_q2_K_ref(x: [*]const f32, y: ?*anyopaque, k: i64) void;
extern "c" fn dequantize_row_q3_K(x: ?*const anyopaque, y: [*]f32, k: i64) void;
extern "c" fn quantize_row_q3_K_ref(x: [*]const f32, y: ?*anyopaque, k: i64) void;
extern "c" fn dequantize_row_q4_0(x: ?*const anyopaque, y: [*]f32, k: i64) void;
extern "c" fn quantize_row_q4_0_ref(x: [*]const f32, y: ?*anyopaque, k: i64) void;
extern "c" fn dequantize_row_q4_K(x: ?*const anyopaque, y: [*]f32, k: i64) void;
extern "c" fn quantize_row_q4_K_ref(x: [*]const f32, y: ?*anyopaque, k: i64) void;
extern "c" fn dequantize_row_q6_K(x: ?*const anyopaque, y: [*]f32, k: i64) void;
extern "c" fn quantize_row_q6_K_ref(x: [*]const f32, y: ?*anyopaque, k: i64) void;
extern "c" fn dequantize_row_mxfp4(x: ?*const anyopaque, y: [*]f32, k: i64) void;
extern "c" fn quantize_row_mxfp4_ref(x: [*]const f32, y: ?*anyopaque, k: i64) void;

var ggml_cpu_once = std.once(initializeGgmlCpu);

fn initializeGgmlCpu() void {
    c.ggml_cpu_init();
}

fn ensureGgmlCpuInitialized() void {
    ggml_cpu_once.call();
}

pub const Error = residency.Error || error{
    InvalidTensorType,
    InvalidTensorShape,
    InvalidInput,
    ChunkTooSmall,
    DequantBufferTooSmall,
    QuantizedScratchTooSmall,
};

pub const TilePolicy = union(enum) {
    fixed_rows: usize,
    adaptive: struct {
        /// Zero uses the largest window that fits the manager budget.
        target_bytes: usize = 0,
        max_rows: usize = std.math.maxInt(usize),
        prefault: bool = false,
    },
};

fn validateTilePolicy(policy: TilePolicy) Error!void {
    switch (policy) {
        .fixed_rows => |rows| if (rows == 0) return Error.ChunkTooSmall,
        .adaptive => |options| if (options.max_rows == 0) return Error.ChunkTooSmall,
    }
}

fn planTileRows(policy: TilePolicy, capacity: usize, row_bytes: usize, remaining_rows: usize) Error!usize {
    const capacity_rows = capacity / row_bytes;
    if (capacity_rows == 0) return Error.ChunkTooSmall;
    const requested_rows = switch (policy) {
        .fixed_rows => |rows| if (rows == 0) return Error.ChunkTooSmall else rows,
        .adaptive => |options| blk: {
            if (options.max_rows == 0) return Error.ChunkTooSmall;
            if (options.target_bytes == 0) break :blk options.max_rows;
            break :blk @max(@as(usize, 1), options.target_bytes / row_bytes);
        },
    };
    return @min(@min(requested_rows, capacity_rows), remaining_rows);
}

fn policyPrefault(policy: TilePolicy) bool {
    return switch (policy) {
        .fixed_rows => false,
        .adaptive => |options| options.prefault,
    };
}

fn acquireTile(
    manager: *residency.Manager,
    descriptor: *const gguf.TensorDescriptor,
    byte_offset: usize,
    byte_len: usize,
    policy: TilePolicy,
) Error!residency.TensorView {
    if (policyPrefault(policy)) try manager.prefetchRange(descriptor.handle, byte_offset, byte_len);
    return manager.acquireRange(descriptor.handle, byte_offset, byte_len);
}

/// Proof-of-integration compute boundary for GGUF F32 matrix weights.
///
/// The descriptor shape follows GGML ordering: dimensions[0] is the row width
/// and dimensions[1] is the number of rows. Weight bytes are never retained
/// beyond a pinned TensorView and can therefore be evicted between row tiles.
pub fn matVecF32(
    manager: *residency.Manager,
    descriptor: *const gguf.TensorDescriptor,
    input: []const f32,
    output: []f32,
    rows_per_tile: usize,
) Error!void {
    return matVecF32WithPolicy(manager, descriptor, input, output, .{ .fixed_rows = rows_per_tile });
}

pub fn matVecF32WithPolicy(
    manager: *residency.Manager,
    descriptor: *const gguf.TensorDescriptor,
    input: []const f32,
    output: []f32,
    policy: TilePolicy,
) Error!void {
    return matMulF32WithPolicy(manager, descriptor, input, output, 1, policy);
}

/// Batched F32 matrix multiply. Inputs are row-major [batch_count, columns]
/// and outputs are row-major [batch_count, rows]. Each weight tile remains
/// pinned while every activation row is processed.
pub fn matMulF32(
    manager: *residency.Manager,
    descriptor: *const gguf.TensorDescriptor,
    inputs: []const f32,
    outputs: []f32,
    batch_count: usize,
    rows_per_tile: usize,
) Error!void {
    return matMulF32WithPolicy(
        manager,
        descriptor,
        inputs,
        outputs,
        batch_count,
        .{ .fixed_rows = rows_per_tile },
    );
}

pub fn matMulF32WithPolicy(
    manager: *residency.Manager,
    descriptor: *const gguf.TensorDescriptor,
    inputs: []const f32,
    outputs: []f32,
    batch_count: usize,
    policy: TilePolicy,
) Error!void {
    if (descriptor.ggml_type != gguf.type_f32) return Error.InvalidTensorType;
    if (descriptor.n_dimensions != 2) return Error.InvalidTensorShape;
    try validateTilePolicy(policy);

    const columns = std.math.cast(usize, descriptor.dimensions[0]) orelse return Error.InvalidTensorShape;
    const rows = std.math.cast(usize, descriptor.dimensions[1]) orelse return Error.InvalidTensorShape;
    if (columns == 0 or rows == 0) return Error.InvalidTensorShape;
    if (batch_count == 0) return Error.InvalidInput;
    const input_elements = std.math.mul(usize, batch_count, columns) catch return Error.InvalidInput;
    const output_elements = std.math.mul(usize, batch_count, rows) catch return Error.InvalidInput;
    if (inputs.len != input_elements or outputs.len != output_elements) return Error.InvalidInput;

    const elements = std.math.mul(usize, columns, rows) catch return Error.InvalidTensorShape;
    const expected_bytes = std.math.mul(usize, elements, @sizeOf(f32)) catch return Error.InvalidTensorShape;
    if (descriptor.byte_len != expected_bytes) return Error.InvalidTensorShape;

    const row_bytes = std.math.mul(usize, columns, @sizeOf(f32)) catch return Error.InvalidTensorShape;
    var row_start: usize = 0;
    while (row_start < rows) {
        const byte_offset = std.math.mul(usize, row_start, row_bytes) catch return Error.InvalidTensorShape;
        const capacity = try manager.rangeCapacity(descriptor.handle, byte_offset);
        const tile_rows = try planTileRows(policy, capacity, row_bytes, rows - row_start);
        const byte_len = std.math.mul(usize, tile_rows, row_bytes) catch return Error.InvalidTensorShape;
        var view = try acquireTile(manager, descriptor, byte_offset, byte_len, policy);
        defer view.release();

        const values: []align(1) const f32 = std.mem.bytesAsSlice(f32, view.bytes());
        for (0..tile_rows) |tile_row| {
            const weights = values[tile_row * columns ..][0..columns];
            for (0..batch_count) |batch| {
                const input = inputs[batch * columns ..][0..columns];
                var sum: f32 = 0;
                for (weights, input) |weight, value| sum += weight * value;
                outputs[batch * rows + row_start + tile_row] = sum;
            }
        }
        row_start += tile_rows;
    }
}

const DequantizeFn = *const fn (?*const anyopaque, [*]f32, i64) callconv(.c) void;
const QuantizeFn = *const fn ([*]const f32, ?*anyopaque, i64) callconv(.c) void;

const QuantizedFormat = struct {
    block_elements: usize,
    block_bytes: usize,
    dequantize: DequantizeFn,
};

fn quantizedFormat(ggml_type: u32) ?QuantizedFormat {
    return if (ggml_type == gguf.type_q2_k)
        .{ .block_elements = 256, .block_bytes = 84, .dequantize = &dequantize_row_q2_K }
    else if (ggml_type == gguf.type_q3_k)
        .{ .block_elements = 256, .block_bytes = 110, .dequantize = &dequantize_row_q3_K }
    else if (ggml_type == gguf.type_q4_0)
        .{ .block_elements = 32, .block_bytes = 18, .dequantize = &dequantize_row_q4_0 }
    else if (ggml_type == gguf.type_q4_k)
        .{ .block_elements = 256, .block_bytes = 144, .dequantize = &dequantize_row_q4_K }
    else if (ggml_type == gguf.type_q6_k)
        .{ .block_elements = 256, .block_bytes = 210, .dequantize = &dequantize_row_q6_K }
    else if (ggml_type == gguf.type_mxfp4)
        .{ .block_elements = 32, .block_bytes = 17, .dequantize = &dequantize_row_mxfp4 }
    else
        null;
}

pub fn quantizedDotScratchBytes(ggml_type: u32, columns: usize) Error!usize {
    const format = quantizedFormat(ggml_type) orelse return Error.InvalidTensorType;
    if (columns == 0 or columns % format.block_elements != 0) return Error.InvalidTensorShape;
    ensureGgmlCpuInitialized();
    const weight_traits = c.ggml_get_type_traits_cpu(ggml_type);
    if (weight_traits == null or weight_traits.*.vec_dot == null) return Error.InvalidTensorType;
    const activation_traits = c.ggml_get_type_traits_cpu(weight_traits.*.vec_dot_type);
    if (activation_traits == null or activation_traits.*.from_float == null) return Error.InvalidTensorType;
    return c.ggml_row_size(weight_traits.*.vec_dot_type, @intCast(columns));
}

pub fn quantizedDotBatchScratchBytes(ggml_type: u32, columns: usize, batch_count: usize) Error!usize {
    if (batch_count == 0) return Error.InvalidInput;
    const row_bytes = try quantizedDotScratchBytes(ggml_type, columns);
    return std.math.mul(usize, row_bytes, batch_count) catch return Error.InvalidInput;
}

/// Bounded matrix-vector multiply using the canonical GGML CPU quantized-dot
/// semantics. The activation is converted once to the vec_dot type selected by
/// GGML (Q8_0/Q8_K for the formats used here), then each mapped weight row is
/// passed to the same vec_dot kernel used by ggml_mul_mat.
pub fn matVecQuantizedGgml(
    manager: *residency.Manager,
    descriptor: *const gguf.TensorDescriptor,
    input: []const f32,
    output: []f32,
    rows_per_tile: usize,
    quantized_input_scratch: []u8,
) Error!void {
    return matVecQuantizedGgmlWithPolicy(
        manager,
        descriptor,
        input,
        output,
        .{ .fixed_rows = rows_per_tile },
        quantized_input_scratch,
    );
}

pub fn matVecQuantizedGgmlWithPolicy(
    manager: *residency.Manager,
    descriptor: *const gguf.TensorDescriptor,
    input: []const f32,
    output: []f32,
    policy: TilePolicy,
    quantized_input_scratch: []u8,
) Error!void {
    return matMulQuantizedGgmlWithPolicy(
        manager,
        descriptor,
        input,
        output,
        1,
        policy,
        quantized_input_scratch,
    );
}

/// Batched canonical GGML quantized matrix multiply. Inputs are row-major
/// [batch_count, columns], outputs are row-major [batch_count, rows], and the
/// scratch contains one converted activation row per batch item.
pub fn matMulQuantizedGgml(
    manager: *residency.Manager,
    descriptor: *const gguf.TensorDescriptor,
    inputs: []const f32,
    outputs: []f32,
    batch_count: usize,
    rows_per_tile: usize,
    quantized_input_scratch: []u8,
) Error!void {
    return matMulQuantizedGgmlWithPolicy(
        manager,
        descriptor,
        inputs,
        outputs,
        batch_count,
        .{ .fixed_rows = rows_per_tile },
        quantized_input_scratch,
    );
}

pub fn matMulQuantizedGgmlWithPolicy(
    manager: *residency.Manager,
    descriptor: *const gguf.TensorDescriptor,
    inputs: []const f32,
    outputs: []f32,
    batch_count: usize,
    policy: TilePolicy,
    quantized_input_scratch: []u8,
) Error!void {
    if (descriptor.n_dimensions != 2) return Error.InvalidTensorShape;
    try validateTilePolicy(policy);
    const format = quantizedFormat(descriptor.ggml_type) orelse return Error.InvalidTensorType;
    const columns = std.math.cast(usize, descriptor.dimensions[0]) orelse return Error.InvalidTensorShape;
    const rows = std.math.cast(usize, descriptor.dimensions[1]) orelse return Error.InvalidTensorShape;
    if (columns == 0 or rows == 0 or columns % format.block_elements != 0) return Error.InvalidTensorShape;
    if (batch_count == 0) return Error.InvalidInput;
    const input_elements = std.math.mul(usize, batch_count, columns) catch return Error.InvalidInput;
    const output_elements = std.math.mul(usize, batch_count, rows) catch return Error.InvalidInput;
    if (inputs.len != input_elements or outputs.len != output_elements) return Error.InvalidInput;

    const row_bytes = std.math.mul(usize, columns / format.block_elements, format.block_bytes) catch return Error.InvalidTensorShape;
    const expected_bytes = std.math.mul(usize, row_bytes, rows) catch return Error.InvalidTensorShape;
    if (descriptor.byte_len != expected_bytes) return Error.InvalidTensorShape;

    ensureGgmlCpuInitialized();
    const weight_traits = c.ggml_get_type_traits_cpu(descriptor.ggml_type);
    if (weight_traits == null or weight_traits.*.vec_dot == null) return Error.InvalidTensorType;
    const activation_traits = c.ggml_get_type_traits_cpu(weight_traits.*.vec_dot_type);
    if (activation_traits == null or activation_traits.*.from_float == null) return Error.InvalidTensorType;
    const input_bytes = c.ggml_row_size(weight_traits.*.vec_dot_type, @intCast(columns));
    const scratch_bytes = std.math.mul(usize, input_bytes, batch_count) catch return Error.InvalidInput;
    if (quantized_input_scratch.len < scratch_bytes) return Error.QuantizedScratchTooSmall;
    for (0..batch_count) |batch| {
        const input = inputs[batch * columns ..][0..columns];
        activation_traits.*.from_float.?(
            input.ptr,
            quantized_input_scratch[batch * input_bytes ..].ptr,
            @intCast(columns),
        );
    }

    var row_start: usize = 0;
    while (row_start < rows) {
        const byte_offset = std.math.mul(usize, row_start, row_bytes) catch return Error.InvalidTensorShape;
        const capacity = try manager.rangeCapacity(descriptor.handle, byte_offset);
        const tile_rows = try planTileRows(policy, capacity, row_bytes, rows - row_start);
        const byte_len = std.math.mul(usize, tile_rows, row_bytes) catch return Error.InvalidTensorShape;
        var view = try acquireTile(manager, descriptor, byte_offset, byte_len, policy);
        defer view.release();
        for (0..tile_rows) |tile_row| {
            const quantized_row = view.bytes()[tile_row * row_bytes ..][0..row_bytes];
            for (0..batch_count) |batch| {
                var sum: f32 = 0;
                weight_traits.*.vec_dot.?(
                    @intCast(columns),
                    &sum,
                    0,
                    quantized_row.ptr,
                    0,
                    quantized_input_scratch[batch * input_bytes ..].ptr,
                    0,
                    1,
                );
                outputs[batch * rows + row_start + tile_row] = sum;
            }
        }
        row_start += tile_rows;
    }
}

/// Bounded matrix-vector multiply for the quantized GGUF formats supported by
/// this proof path (Q4_0, Q4_K, and Q6_K). A row is dequantized into explicit
/// caller-owned scratch while its backing range is pinned.
pub fn matVecQuantized(
    manager: *residency.Manager,
    descriptor: *const gguf.TensorDescriptor,
    input: []const f32,
    output: []f32,
    rows_per_tile: usize,
    dequant_scratch: []f32,
) Error!void {
    return matVecQuantizedWithPolicy(
        manager,
        descriptor,
        input,
        output,
        .{ .fixed_rows = rows_per_tile },
        dequant_scratch,
    );
}

pub fn matVecQuantizedWithPolicy(
    manager: *residency.Manager,
    descriptor: *const gguf.TensorDescriptor,
    input: []const f32,
    output: []f32,
    policy: TilePolicy,
    dequant_scratch: []f32,
) Error!void {
    if (descriptor.n_dimensions != 2) return Error.InvalidTensorShape;
    return matVecQuantizedAt(manager, descriptor, 0, input, output, policy, dequant_scratch);
}

/// Multiplies one matrix selected from dimension 2 of a GGML [columns, rows,
/// slices] tensor. Expert weights use this layout, so only the selected expert
/// is faulted and pinned; the complete expert tensor is never mapped.
pub fn matVecQuantizedSlice(
    manager: *residency.Manager,
    descriptor: *const gguf.TensorDescriptor,
    slice: usize,
    input: []const f32,
    output: []f32,
    rows_per_tile: usize,
    dequant_scratch: []f32,
) Error!void {
    return matVecQuantizedSliceWithPolicy(
        manager,
        descriptor,
        slice,
        input,
        output,
        .{ .fixed_rows = rows_per_tile },
        dequant_scratch,
    );
}

pub fn matVecQuantizedSliceWithPolicy(
    manager: *residency.Manager,
    descriptor: *const gguf.TensorDescriptor,
    slice: usize,
    input: []const f32,
    output: []f32,
    policy: TilePolicy,
    dequant_scratch: []f32,
) Error!void {
    if (descriptor.n_dimensions != 3) return Error.InvalidTensorShape;
    const slices = std.math.cast(usize, descriptor.dimensions[2]) orelse return Error.InvalidTensorShape;
    if (slice >= slices) return Error.InvalidInput;
    const format = quantizedFormat(descriptor.ggml_type) orelse return Error.InvalidTensorType;
    const columns = std.math.cast(usize, descriptor.dimensions[0]) orelse return Error.InvalidTensorShape;
    const rows = std.math.cast(usize, descriptor.dimensions[1]) orelse return Error.InvalidTensorShape;
    if (columns == 0 or rows == 0 or columns % format.block_elements != 0) return Error.InvalidTensorShape;
    const row_bytes = std.math.mul(usize, columns / format.block_elements, format.block_bytes) catch return Error.InvalidTensorShape;
    const matrix_bytes = std.math.mul(usize, row_bytes, rows) catch return Error.InvalidTensorShape;
    const expected_bytes = std.math.mul(usize, matrix_bytes, slices) catch return Error.InvalidTensorShape;
    if (descriptor.byte_len != expected_bytes) return Error.InvalidTensorShape;
    const base_byte_offset = std.math.mul(usize, slice, matrix_bytes) catch return Error.InvalidTensorShape;
    return matVecQuantizedAt(manager, descriptor, base_byte_offset, input, output, policy, dequant_scratch);
}

fn matVecQuantizedAt(
    manager: *residency.Manager,
    descriptor: *const gguf.TensorDescriptor,
    base_byte_offset: usize,
    input: []const f32,
    output: []f32,
    policy: TilePolicy,
    dequant_scratch: []f32,
) Error!void {
    const format = quantizedFormat(descriptor.ggml_type) orelse return Error.InvalidTensorType;
    if (descriptor.n_dimensions != 2 and descriptor.n_dimensions != 3) return Error.InvalidTensorShape;
    try validateTilePolicy(policy);

    const columns = std.math.cast(usize, descriptor.dimensions[0]) orelse return Error.InvalidTensorShape;
    const rows = std.math.cast(usize, descriptor.dimensions[1]) orelse return Error.InvalidTensorShape;
    if (columns == 0 or rows == 0 or columns % format.block_elements != 0) return Error.InvalidTensorShape;
    if (input.len != columns or output.len != rows) return Error.InvalidInput;
    if (dequant_scratch.len < columns) return Error.DequantBufferTooSmall;

    const row_bytes = std.math.mul(usize, columns / format.block_elements, format.block_bytes) catch return Error.InvalidTensorShape;
    const matrix_bytes = std.math.mul(usize, row_bytes, rows) catch return Error.InvalidTensorShape;
    if (descriptor.n_dimensions == 2 and descriptor.byte_len != matrix_bytes) return Error.InvalidTensorShape;
    if (base_byte_offset > descriptor.byte_len or matrix_bytes > descriptor.byte_len - base_byte_offset) return Error.InvalidTensorShape;
    const columns_i64 = std.math.cast(i64, columns) orelse return Error.InvalidTensorShape;

    var row_start: usize = 0;
    while (row_start < rows) {
        const relative_row_offset = std.math.mul(usize, row_start, row_bytes) catch return Error.InvalidTensorShape;
        const byte_offset = std.math.add(usize, base_byte_offset, relative_row_offset) catch return Error.InvalidTensorShape;
        const capacity = try manager.rangeCapacity(descriptor.handle, byte_offset);
        const tile_rows = try planTileRows(policy, capacity, row_bytes, rows - row_start);
        const byte_len = std.math.mul(usize, tile_rows, row_bytes) catch return Error.InvalidTensorShape;
        var view = try acquireTile(manager, descriptor, byte_offset, byte_len, policy);
        defer view.release();

        const bytes = view.bytes();
        for (0..tile_rows) |tile_row| {
            const quantized_row = bytes[tile_row * row_bytes ..][0..row_bytes];
            format.dequantize(quantized_row.ptr, dequant_scratch.ptr, columns_i64);
            var sum: f32 = 0;
            for (dequant_scratch[0..columns], input) |weight, value| sum += weight * value;
            output[row_start + tile_row] = sum;
        }
        row_start += tile_rows;
    }
}

/// Decodes one logical matrix row through a bounded residency view. This is
/// used for token embedding lookup without mapping or dequantizing the full
/// vocabulary matrix.
pub fn readMatrixRow(
    manager: *residency.Manager,
    descriptor: *const gguf.TensorDescriptor,
    row: usize,
    output: []f32,
) Error!void {
    if (descriptor.n_dimensions != 2) return Error.InvalidTensorShape;
    const columns = std.math.cast(usize, descriptor.dimensions[0]) orelse return Error.InvalidTensorShape;
    const rows = std.math.cast(usize, descriptor.dimensions[1]) orelse return Error.InvalidTensorShape;
    if (columns == 0 or rows == 0 or row >= rows or output.len != columns) return Error.InvalidInput;

    if (descriptor.ggml_type == gguf.type_f32) {
        const row_bytes = std.math.mul(usize, columns, @sizeOf(f32)) catch return Error.InvalidTensorShape;
        const expected_bytes = std.math.mul(usize, row_bytes, rows) catch return Error.InvalidTensorShape;
        if (descriptor.byte_len != expected_bytes) return Error.InvalidTensorShape;
        const byte_offset = std.math.mul(usize, row, row_bytes) catch return Error.InvalidTensorShape;
        var view = try manager.acquireRange(descriptor.handle, byte_offset, row_bytes);
        defer view.release();
        const values: []align(1) const f32 = std.mem.bytesAsSlice(f32, view.bytes());
        @memcpy(output, values);
        return;
    }

    const format = quantizedFormat(descriptor.ggml_type) orelse return Error.InvalidTensorType;
    if (columns % format.block_elements != 0) return Error.InvalidTensorShape;
    const row_bytes = std.math.mul(usize, columns / format.block_elements, format.block_bytes) catch return Error.InvalidTensorShape;
    const expected_bytes = std.math.mul(usize, row_bytes, rows) catch return Error.InvalidTensorShape;
    if (descriptor.byte_len != expected_bytes) return Error.InvalidTensorShape;
    const byte_offset = std.math.mul(usize, row, row_bytes) catch return Error.InvalidTensorShape;
    var view = try manager.acquireRange(descriptor.handle, byte_offset, row_bytes);
    defer view.release();
    const columns_i64 = std.math.cast(i64, columns) orelse return Error.InvalidTensorShape;
    format.dequantize(view.bytes().ptr, output.ptr, columns_i64);
}

/// Compatibility wrapper for callers that require a Q4_0 descriptor.
pub fn matVecQ4_0(
    manager: *residency.Manager,
    descriptor: *const gguf.TensorDescriptor,
    input: []const f32,
    output: []f32,
    rows_per_tile: usize,
    dequant_scratch: []f32,
) Error!void {
    if (descriptor.ggml_type != gguf.type_q4_0) return Error.InvalidTensorType;
    return matVecQuantized(manager, descriptor, input, output, rows_per_tile, dequant_scratch);
}

fn referenceMatVec(weights: []const f32, columns: usize, input: []const f32, output: []f32) void {
    for (output, 0..) |*result, row| {
        var sum: f32 = 0;
        for (weights[row * columns ..][0..columns], input) |weight, value| sum += weight * value;
        result.* = sum;
    }
}

fn createMatrixBacking(tmp: *std.testing.TmpDir, weights: []const f32) ![:0]u8 {
    var file = try tmp.dir.createFile("matrix.bin", .{});
    defer file.close();
    try file.writeAll(std.mem.sliceAsBytes(weights));
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "matrix.bin");
    defer std.testing.allocator.free(path);
    return std.testing.allocator.dupeZ(u8, path);
}

test "bounded GGUF descriptor matvec matches resident baseline" {
    const granularity = try residency.mappingGranularity();
    const columns: usize = granularity / @sizeOf(f32);
    const rows: usize = 3;
    const weights = try std.testing.allocator.alloc(f32, columns * rows);
    defer std.testing.allocator.free(weights);
    const input = try std.testing.allocator.alloc(f32, columns);
    defer std.testing.allocator.free(input);
    for (input, 0..) |*value, i| value.* = @as(f32, @floatFromInt(i % 7)) - 3;
    for (weights, 0..) |*weight, i| weight.* = @as(f32, @floatFromInt(i % 11)) / 8 - 0.5;

    var expected: [rows]f32 = undefined;
    referenceMatVec(weights, columns, input, &expected);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path_z = try createMatrixBacking(&tmp, weights);
    defer std.testing.allocator.free(path_z);
    var store = try residency.BackingStore.open(path_z);
    defer store.close();

    const descriptor = gguf.TensorDescriptor{
        .handle = .{ .id = 1 },
        .name = "matrix.weight",
        .file_offset = 0,
        .byte_len = weights.len * @sizeOf(f32),
        .ggml_type = gguf.type_f32,
        .n_dimensions = 2,
        .dimensions = .{ columns, rows, 1, 1 },
    };

    var baseline_manager = try residency.Manager.init(std.testing.allocator, &store, descriptor.byte_len);
    defer baseline_manager.deinit();
    try baseline_manager.register(descriptor.handle, descriptor.file_offset, descriptor.byte_len);
    var baseline: [rows]f32 = undefined;
    try matVecF32(&baseline_manager, &descriptor, input, &baseline, rows);
    try std.testing.expectEqualSlices(f32, &expected, &baseline);

    var bounded_manager = try residency.Manager.init(std.testing.allocator, &store, granularity);
    defer bounded_manager.deinit();
    try bounded_manager.register(descriptor.handle, descriptor.file_offset, descriptor.byte_len);
    var bounded: [rows]f32 = undefined;
    try matVecF32(&bounded_manager, &descriptor, input, &bounded, 1);
    try std.testing.expectEqualSlices(f32, &baseline, &bounded);

    const metrics = bounded_manager.metrics();
    try std.testing.expectEqual(granularity, metrics.peak_resident_bytes);
    try std.testing.expectEqual(@as(u64, rows), metrics.faults);
    try std.testing.expectEqual(@as(u64, rows - 1), metrics.evictions);
}

test "adaptive F32 tiling preserves output and reduces mapping faults" {
    const granularity = try residency.mappingGranularity();
    const columns: usize = granularity / (4 * @sizeOf(f32));
    const rows: usize = 12;
    const weights = try std.testing.allocator.alloc(f32, columns * rows);
    defer std.testing.allocator.free(weights);
    const input = try std.testing.allocator.alloc(f32, columns);
    defer std.testing.allocator.free(input);
    for (weights, 0..) |*weight, i| weight.* = @as(f32, @floatFromInt(i % 19)) / 7 - 1;
    for (input, 0..) |*value, i| value.* = @as(f32, @floatFromInt(i % 5)) - 2;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path_z = try createMatrixBacking(&tmp, weights);
    defer std.testing.allocator.free(path_z);
    var store = try residency.BackingStore.open(path_z);
    defer store.close();
    const descriptor = gguf.TensorDescriptor{
        .handle = .{ .id = 404 },
        .name = "adaptive.weight",
        .file_offset = 0,
        .byte_len = weights.len * @sizeOf(f32),
        .ggml_type = gguf.type_f32,
        .n_dimensions = 2,
        .dimensions = .{ columns, rows, 1, 1 },
    };

    var fixed_manager = try residency.Manager.init(std.testing.allocator, &store, granularity);
    defer fixed_manager.deinit();
    try fixed_manager.register(descriptor.handle, 0, descriptor.byte_len);
    var fixed: [rows]f32 = undefined;
    try matVecF32(&fixed_manager, &descriptor, input, &fixed, 1);

    var adaptive_manager = try residency.Manager.init(std.testing.allocator, &store, granularity);
    defer adaptive_manager.deinit();
    try adaptive_manager.register(descriptor.handle, 0, descriptor.byte_len);
    var adaptive: [rows]f32 = undefined;
    try matVecF32WithPolicy(
        &adaptive_manager,
        &descriptor,
        input,
        &adaptive,
        .{ .adaptive = .{ .prefault = true } },
    );

    try std.testing.expectEqualSlices(f32, &fixed, &adaptive);
    const fixed_metrics = fixed_manager.metrics();
    const adaptive_metrics = adaptive_manager.metrics();
    try std.testing.expect(adaptive_metrics.faults < fixed_metrics.faults);
    try std.testing.expectEqual(adaptive_metrics.faults, adaptive_metrics.prefetches);
    try std.testing.expectEqual(adaptive_metrics.faults, adaptive_metrics.hits);
    try std.testing.expect(adaptive_metrics.peak_resident_bytes <= granularity);
}

test "matrix row lookup faults only requested F32 row" {
    const columns: usize = 4;
    const rows: usize = 3;
    const weights = [_]f32{
        1, 2,  3,  4,
        5, 6,  7,  8,
        9, 10, 11, 12,
    };
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const path_z = try createMatrixBacking(&tmp, &weights);
    defer std.testing.allocator.free(path_z);
    var store = try residency.BackingStore.open(path_z);
    defer store.close();
    const descriptor = gguf.TensorDescriptor{
        .handle = .{ .id = 99 },
        .name = "embedding.weight",
        .file_offset = 0,
        .byte_len = @sizeOf(@TypeOf(weights)),
        .ggml_type = gguf.type_f32,
        .n_dimensions = 2,
        .dimensions = .{ columns, rows, 1, 1 },
    };
    var manager = try residency.Manager.init(std.testing.allocator, &store, try residency.mappingGranularity());
    defer manager.deinit();
    try manager.register(descriptor.handle, 0, descriptor.byte_len);
    var output: [columns]f32 = undefined;
    try readMatrixRow(&manager, &descriptor, 1, &output);
    try std.testing.expectEqualSlices(f32, weights[columns .. columns * 2], &output);
    try std.testing.expectEqual(@as(u64, 1), manager.metrics().faults);
    try std.testing.expectError(Error.InvalidInput, readMatrixRow(&manager, &descriptor, rows, &output));
}

test "matrix row lookup dequantizes one Q4_0 row" {
    const columns: usize = 32;
    const rows: usize = 2;
    const row_bytes: usize = 18;
    var weights: [columns * rows]f32 = undefined;
    for (&weights, 0..) |*weight, i| weight.* = @as(f32, @floatFromInt(i % 17)) / 5 - 1.0;
    var encoded: [row_bytes * rows]u8 = undefined;
    for (0..rows) |row| quantize_row_q4_0_ref(weights[row * columns ..][0..columns].ptr, encoded[row * row_bytes ..][0..row_bytes].ptr, columns);
    var expected: [columns]f32 = undefined;
    dequantize_row_q4_0(encoded[row_bytes..].ptr, &expected, columns);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var file = try tmp.dir.createFile("embedding-q4.bin", .{});
    try file.writeAll(&encoded);
    file.close();
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "embedding-q4.bin");
    defer std.testing.allocator.free(path);
    const path_z = try std.testing.allocator.dupeZ(u8, path);
    defer std.testing.allocator.free(path_z);
    var store = try residency.BackingStore.open(path_z);
    defer store.close();
    const descriptor = gguf.TensorDescriptor{
        .handle = .{ .id = 100 },
        .name = "embedding-q4.weight",
        .file_offset = 0,
        .byte_len = encoded.len,
        .ggml_type = gguf.type_q4_0,
        .n_dimensions = 2,
        .dimensions = .{ columns, rows, 1, 1 },
    };
    var manager = try residency.Manager.init(std.testing.allocator, &store, try residency.mappingGranularity());
    defer manager.deinit();
    try manager.register(descriptor.handle, 0, descriptor.byte_len);
    var output: [columns]f32 = undefined;
    try readMatrixRow(&manager, &descriptor, 1, &output);
    try std.testing.expectEqualSlices(f32, &expected, &output);
}

test "bounded Q4_0 descriptor matvec matches canonical dequantized reference" {
    const granularity = try residency.mappingGranularity();
    const columns: usize = 32;
    const row_bytes: usize = 18;
    const rows_per_tile: usize = granularity / row_bytes;
    const rows: usize = rows_per_tile * 3;

    const weights = try std.testing.allocator.alloc(f32, columns * rows);
    defer std.testing.allocator.free(weights);
    const input = try std.testing.allocator.alloc(f32, columns);
    defer std.testing.allocator.free(input);
    for (weights, 0..) |*weight, i| weight.* = @as(f32, @floatFromInt(i % 29)) / 9 - 1.5;
    for (input, 0..) |*value, i| value.* = @as(f32, @floatFromInt(i % 7)) - 3;

    const quantized = try std.testing.allocator.alloc(u8, row_bytes * rows);
    defer std.testing.allocator.free(quantized);
    for (0..rows) |row| {
        quantize_row_q4_0_ref(
            weights[row * columns ..][0..columns].ptr,
            quantized[row * row_bytes ..][0..row_bytes].ptr,
            columns,
        );
    }

    const dequantized = try std.testing.allocator.alloc(f32, weights.len);
    defer std.testing.allocator.free(dequantized);
    dequantize_row_q4_0(quantized.ptr, dequantized.ptr, @intCast(weights.len));
    const expected = try std.testing.allocator.alloc(f32, rows);
    defer std.testing.allocator.free(expected);
    referenceMatVec(dequantized, columns, input, expected);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var file = try tmp.dir.createFile("matrix-q4_0.bin", .{});
    try file.writeAll(quantized);
    file.close();
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "matrix-q4_0.bin");
    defer std.testing.allocator.free(path);
    const path_z = try std.testing.allocator.dupeZ(u8, path);
    defer std.testing.allocator.free(path_z);
    var store = try residency.BackingStore.open(path_z);
    defer store.close();

    const descriptor = gguf.TensorDescriptor{
        .handle = .{ .id = 2 },
        .name = "matrix-q4_0.weight",
        .file_offset = 0,
        .byte_len = quantized.len,
        .ggml_type = gguf.type_q4_0,
        .n_dimensions = 2,
        .dimensions = .{ columns, rows, 1, 1 },
    };

    var baseline_manager = try residency.Manager.init(std.testing.allocator, &store, quantized.len);
    defer baseline_manager.deinit();
    try baseline_manager.register(descriptor.handle, 0, quantized.len);
    const baseline = try std.testing.allocator.alloc(f32, rows);
    defer std.testing.allocator.free(baseline);
    var baseline_scratch: [columns]f32 = undefined;
    try matVecQ4_0(&baseline_manager, &descriptor, input, baseline, rows, &baseline_scratch);
    try std.testing.expectEqualSlices(f32, expected, baseline);

    const bounded_budget = granularity + row_bytes - 1;
    var bounded_manager = try residency.Manager.init(std.testing.allocator, &store, bounded_budget);
    defer bounded_manager.deinit();
    try bounded_manager.register(descriptor.handle, 0, quantized.len);
    const bounded = try std.testing.allocator.alloc(f32, rows);
    defer std.testing.allocator.free(bounded);
    var bounded_scratch: [columns]f32 = undefined;
    try matVecQ4_0(&bounded_manager, &descriptor, input, bounded, rows_per_tile, &bounded_scratch);
    try std.testing.expectEqualSlices(f32, baseline, bounded);

    const metrics = bounded_manager.metrics();
    try std.testing.expect(metrics.peak_resident_bytes <= bounded_budget);
    try std.testing.expect(metrics.faults >= 3);
    try std.testing.expectEqual(metrics.faults - 1, metrics.evictions);
}

test "GGML quantized dot matvec produces canonical vec-dot results" {
    const columns: usize = 256;
    const rows: usize = 3;
    const row_bytes: usize = 144;
    var weights: [columns * rows]f32 = undefined;
    var input: [columns]f32 = undefined;
    for (&weights, 0..) |*weight, i| weight.* = @as(f32, @floatFromInt(i % 31 + 1)) / 31.0;
    for (&input, 0..) |*value, i| value.* = @as(f32, @floatFromInt(i % 13 + 1)) / 13.0;

    var quantized: [row_bytes * rows]u8 = undefined;
    for (0..rows) |row| {
        quantize_row_q4_K_ref(
            weights[row * columns ..][0..columns].ptr,
            quantized[row * row_bytes ..][0..row_bytes].ptr,
            columns,
        );
    }

    ensureGgmlCpuInitialized();
    const weight_traits = c.ggml_get_type_traits_cpu(gguf.type_q4_k).?;
    const activation_traits = c.ggml_get_type_traits_cpu(weight_traits.*.vec_dot_type).?;
    const input_bytes = c.ggml_row_size(weight_traits.*.vec_dot_type, columns);
    const quantized_input = try std.testing.allocator.alloc(u8, input_bytes);
    defer std.testing.allocator.free(quantized_input);
    activation_traits.*.from_float.?(&input, quantized_input.ptr, columns);
    var expected: [rows]f32 = undefined;
    for (0..rows) |row| {
        weight_traits.*.vec_dot.?(
            columns,
            &expected[row],
            0,
            quantized[row * row_bytes ..][0..row_bytes].ptr,
            0,
            quantized_input.ptr,
            0,
            1,
        );
    }

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var file = try tmp.dir.createFile("matrix-ggml-dot.bin", .{});
    try file.writeAll(&quantized);
    file.close();
    const path = try tmp.dir.realpathAlloc(std.testing.allocator, "matrix-ggml-dot.bin");
    defer std.testing.allocator.free(path);
    const path_z = try std.testing.allocator.dupeZ(u8, path);
    defer std.testing.allocator.free(path_z);
    var store = try residency.BackingStore.open(path_z);
    defer store.close();

    const descriptor = gguf.TensorDescriptor{
        .handle = .{ .id = 22 },
        .name = "matrix-ggml-dot.weight",
        .file_offset = 0,
        .byte_len = quantized.len,
        .ggml_type = gguf.type_q4_k,
        .n_dimensions = 2,
        .dimensions = .{ columns, rows, 1, 1 },
    };
    var manager = try residency.Manager.init(std.testing.allocator, &store, try residency.mappingGranularity());
    defer manager.deinit();
    try manager.register(descriptor.handle, 0, descriptor.byte_len);
    var actual: [rows]f32 = undefined;
    const scratch = try std.testing.allocator.alloc(u8, input_bytes);
    defer std.testing.allocator.free(scratch);
    try matVecQuantizedGgml(&manager, &descriptor, &input, &actual, rows, scratch);
    try std.testing.expectEqualSlices(f32, &expected, &actual);
    var has_nonzero = false;
    for (actual) |value| has_nonzero = has_nonzero or value != 0;
    try std.testing.expect(has_nonzero);
}

test "bounded Q2_K batch reuses each weight tile across activation rows" {
    const allocator = std.testing.allocator;
    const columns: usize = 256;
    const row_bytes: usize = 84;
    const batch_count: usize = 4;
    const granularity = try residency.mappingGranularity();
    const rows_per_tile = @max(@as(usize, 1), granularity / row_bytes);
    const rows = rows_per_tile * 6;

    const weights = try allocator.alloc(f32, columns * rows);
    defer allocator.free(weights);
    for (weights, 0..) |*value, i| value.* = @as(f32, @floatFromInt(i % 41)) / 13.0 - 1.2;
    const quantized = try allocator.alloc(u8, row_bytes * rows);
    defer allocator.free(quantized);
    for (0..rows) |row| {
        quantize_row_q2_K_ref(weights[row * columns ..][0..columns].ptr, quantized[row * row_bytes ..].ptr, columns);
    }
    const inputs = try allocator.alloc(f32, batch_count * columns);
    defer allocator.free(inputs);
    for (inputs, 0..) |*value, i| value.* = @as(f32, @floatFromInt(i % 23)) / 9.0 - 0.8;
    const repeated = try allocator.alloc(f32, batch_count * rows);
    defer allocator.free(repeated);
    const batched = try allocator.alloc(f32, batch_count * rows);
    defer allocator.free(batched);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var file = try tmp.dir.createFile("q2-k-batch.bin", .{});
    try file.writeAll(quantized);
    file.close();
    const path = try tmp.dir.realpathAlloc(allocator, "q2-k-batch.bin");
    defer allocator.free(path);
    const path_z = try allocator.dupeZ(u8, path);
    defer allocator.free(path_z);
    var store = try residency.BackingStore.open(path_z);
    defer store.close();
    const descriptor = gguf.TensorDescriptor{
        .handle = .{ .id = 45 },
        .name = "q2-k-batch.weight",
        .file_offset = 0,
        .byte_len = quantized.len,
        .ggml_type = gguf.type_q2_k,
        .n_dimensions = 2,
        .dimensions = .{ columns, rows, 1, 1 },
    };
    const budget = granularity * 2;
    const single_scratch_len = try quantizedDotScratchBytes(gguf.type_q2_k, columns);
    const single_scratch = try allocator.alloc(u8, single_scratch_len);
    defer allocator.free(single_scratch);
    var repeated_manager = try residency.Manager.init(allocator, &store, budget);
    defer repeated_manager.deinit();
    try repeated_manager.register(descriptor.handle, 0, descriptor.byte_len);
    for (0..batch_count) |batch| {
        try matVecQuantizedGgml(
            &repeated_manager,
            &descriptor,
            inputs[batch * columns ..][0..columns],
            repeated[batch * rows ..][0..rows],
            rows_per_tile,
            single_scratch,
        );
    }
    const repeated_metrics = repeated_manager.metrics();

    const batch_scratch_len = try quantizedDotBatchScratchBytes(gguf.type_q2_k, columns, batch_count);
    const batch_scratch = try allocator.alloc(u8, batch_scratch_len);
    defer allocator.free(batch_scratch);
    var batched_manager = try residency.Manager.init(allocator, &store, budget);
    defer batched_manager.deinit();
    try batched_manager.register(descriptor.handle, 0, descriptor.byte_len);
    try matMulQuantizedGgml(
        &batched_manager,
        &descriptor,
        inputs,
        batched,
        batch_count,
        rows_per_tile,
        batch_scratch,
    );
    const batched_metrics = batched_manager.metrics();

    try std.testing.expectEqualSlices(f32, repeated, batched);
    try std.testing.expect(batched_metrics.peak_resident_bytes <= budget);
    try std.testing.expect(batched_metrics.faults < repeated_metrics.faults);
    try std.testing.expectEqual(batched_metrics.faults * batch_count, repeated_metrics.faults);
}

test "bounded quantized matvec matches canonical dequantized reference" {
    const Case = struct {
        ggml_type: u32,
        block_elements: usize,
        block_bytes: usize,
        quantize: QuantizeFn,
        dequantize: DequantizeFn,
    };
    const cases = [_]Case{
        .{ .ggml_type = gguf.type_q2_k, .block_elements = 256, .block_bytes = 84, .quantize = &quantize_row_q2_K_ref, .dequantize = &dequantize_row_q2_K },
        .{ .ggml_type = gguf.type_q3_k, .block_elements = 256, .block_bytes = 110, .quantize = &quantize_row_q3_K_ref, .dequantize = &dequantize_row_q3_K },
        .{ .ggml_type = gguf.type_q4_k, .block_elements = 256, .block_bytes = 144, .quantize = &quantize_row_q4_K_ref, .dequantize = &dequantize_row_q4_K },
        .{ .ggml_type = gguf.type_q6_k, .block_elements = 256, .block_bytes = 210, .quantize = &quantize_row_q6_K_ref, .dequantize = &dequantize_row_q6_K },
        .{ .ggml_type = gguf.type_mxfp4, .block_elements = 32, .block_bytes = 17, .quantize = &quantize_row_mxfp4_ref, .dequantize = &dequantize_row_mxfp4 },
    };

    const granularity = try residency.mappingGranularity();
    const columns: usize = 256;
    const rows: usize = 3 * (granularity / 210);
    const weights = try std.testing.allocator.alloc(f32, columns * rows);
    defer std.testing.allocator.free(weights);
    const input = try std.testing.allocator.alloc(f32, columns);
    defer std.testing.allocator.free(input);
    for (weights, 0..) |*weight, i| weight.* = @as(f32, @floatFromInt(i % 37)) / 11 - 1.6;
    for (input, 0..) |*value, i| value.* = @as(f32, @floatFromInt(i % 13)) / 4 - 1.5;

    inline for (cases, 0..) |case, case_index| {
        const row_bytes = (columns / case.block_elements) * case.block_bytes;
        const quantized = try std.testing.allocator.alloc(u8, row_bytes * rows);
        defer std.testing.allocator.free(quantized);
        for (0..rows) |row| {
            case.quantize(weights[row * columns ..][0..columns].ptr, quantized[row * row_bytes ..].ptr, columns);
        }

        const dequantized = try std.testing.allocator.alloc(f32, weights.len);
        defer std.testing.allocator.free(dequantized);
        case.dequantize(quantized.ptr, dequantized.ptr, @intCast(weights.len));
        const expected = try std.testing.allocator.alloc(f32, rows);
        defer std.testing.allocator.free(expected);
        referenceMatVec(dequantized, columns, input, expected);

        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        const file_name = try std.fmt.allocPrint(std.testing.allocator, "matrix-k-{d}.bin", .{case_index});
        defer std.testing.allocator.free(file_name);
        var file = try tmp.dir.createFile(file_name, .{});
        try file.writeAll(quantized);
        file.close();
        const path = try tmp.dir.realpathAlloc(std.testing.allocator, file_name);
        defer std.testing.allocator.free(path);
        const path_z = try std.testing.allocator.dupeZ(u8, path);
        defer std.testing.allocator.free(path_z);
        var store = try residency.BackingStore.open(path_z);
        defer store.close();

        const descriptor = gguf.TensorDescriptor{
            .handle = .{ .id = case_index + 10 },
            .name = "matrix-k.weight",
            .file_offset = 0,
            .byte_len = quantized.len,
            .ggml_type = case.ggml_type,
            .n_dimensions = 2,
            .dimensions = .{ columns, rows, 1, 1 },
        };
        var baseline_manager = try residency.Manager.init(std.testing.allocator, &store, descriptor.byte_len);
        defer baseline_manager.deinit();
        try baseline_manager.register(descriptor.handle, 0, descriptor.byte_len);
        const baseline = try std.testing.allocator.alloc(f32, rows);
        defer std.testing.allocator.free(baseline);
        var baseline_scratch: [columns]f32 = undefined;
        try matVecQuantized(&baseline_manager, &descriptor, input, baseline, rows, &baseline_scratch);
        try std.testing.expectEqualSlices(f32, expected, baseline);

        const budget = granularity + row_bytes - 1;
        var manager = try residency.Manager.init(std.testing.allocator, &store, budget);
        defer manager.deinit();
        try manager.register(descriptor.handle, 0, descriptor.byte_len);
        const actual = try std.testing.allocator.alloc(f32, rows);
        defer std.testing.allocator.free(actual);
        var scratch: [columns]f32 = undefined;
        try matVecQuantized(&manager, &descriptor, input, actual, rows, &scratch);
        try std.testing.expectEqualSlices(f32, baseline, actual);

        const metrics = manager.metrics();
        try std.testing.expect(metrics.peak_resident_bytes <= budget);
        try std.testing.expect(metrics.faults >= 2);
        try std.testing.expectEqual(metrics.faults - 1, metrics.evictions);
    }
}

test "Q4_0 matvec validates row shape and scratch" {
    var descriptor = gguf.TensorDescriptor{
        .handle = .{ .id = 1 },
        .name = "invalid-q4_0",
        .file_offset = 0,
        .byte_len = 18,
        .ggml_type = gguf.type_q4_0,
        .n_dimensions = 2,
        .dimensions = .{ 32, 1, 1, 1 },
    };
    var manager: residency.Manager = undefined;
    var output: [1]f32 = undefined;
    var short_scratch: [31]f32 = undefined;
    try std.testing.expectError(Error.DequantBufferTooSmall, matVecQ4_0(&manager, &descriptor, &([_]f32{0} ** 32), &output, 1, &short_scratch));
    descriptor.dimensions[0] = 31;
    try std.testing.expectError(Error.InvalidTensorShape, matVecQ4_0(&manager, &descriptor, &([_]f32{0} ** 31), &output, 1, &short_scratch));
}

test "matvec validates GGUF type shape and tile size" {
    var descriptor = gguf.TensorDescriptor{
        .handle = .{ .id = 1 },
        .name = "invalid",
        .file_offset = 0,
        .byte_len = 4,
        .ggml_type = 1,
        .n_dimensions = 2,
        .dimensions = .{ 1, 1, 1, 1 },
    };

    // These errors are rejected before manager access, so an undefined manager
    // is safe and keeps this validation test independent of a backing fixture.
    var manager: residency.Manager = undefined;
    var output: [1]f32 = undefined;
    try std.testing.expectError(Error.InvalidTensorType, matVecF32(&manager, &descriptor, &.{1}, &output, 1));
    descriptor.ggml_type = 0;
    descriptor.n_dimensions = 1;
    try std.testing.expectError(Error.InvalidTensorShape, matVecF32(&manager, &descriptor, &.{1}, &output, 1));
    descriptor.n_dimensions = 2;
    try std.testing.expectError(Error.ChunkTooSmall, matVecF32(&manager, &descriptor, &.{1}, &output, 0));
}
