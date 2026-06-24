const std = @import("std");

/// The hook declaration and call to insert into ggml_compute_forward_mul_mat
const hook_code_mul_mat =
    \\#ifdef GGML_USE_SIMD_BACKEND
    \\    // Custom SIMD backend hook - try optimized kernels first
    \\    extern int ggml_simd_try_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);
    \\    if (ggml_simd_try_mul_mat(params, dst)) {
    \\        return;
    \\    }
    \\#endif
    \\
;

/// The hook declaration and call to insert into ggml_compute_forward_flash_attn_ext
const hook_code_flash_attn =
    \\#ifdef GGML_USE_SIMD_BACKEND
    \\    // Custom SIMD backend hook for flash attention
    \\    extern int ggml_simd_try_flash_attn(const struct ggml_compute_params * params, struct ggml_tensor * tensor);
    \\    if (ggml_simd_try_flash_attn(params, tensor)) {
    \\        break;
    \\    }
    \\#endif
    \\
;

/// The hook declaration and call to insert into ggml_compute_forward_silu
const hook_code_silu =
    \\#ifdef GGML_USE_SIMD_BACKEND
    \\    // Custom SIMD backend hook for SiLU activation
    \\    extern int ggml_simd_hook_silu(const float * src, float * dst, int n);
    \\    {
    \\        const int64_t ne = ggml_nelements(src0);
    \\        if (ggml_simd_hook_silu((const float *)src0->data, (float *)dst->data, (int)ne)) {
    \\            return;
    \\        }
    \\    }
    \\#endif
    \\
;

/// The hook declaration and call to insert into ggml_compute_forward_norm
const hook_code_norm =
    \\#ifdef GGML_USE_SIMD_BACKEND
    \\    // Custom SIMD backend hook for layer normalization
    \\    extern int ggml_simd_hook_norm(const float * src, float * dst, int n, float eps);
    \\    {
    \\        const int64_t ne = ggml_nelements(src0);
    \\        float eps;
    \\        memcpy(&eps, dst->op_params, sizeof(float));
    \\        if (ggml_simd_hook_norm((const float *)src0->data, (float *)dst->data, (int)ne, eps)) {
    \\            return;
    \\        }
    \\    }
    \\#endif
    \\
;

pub fn patchGgmlCpu(allocator: std.mem.Allocator, input_path: []const u8, output_path: []const u8) !void {
    // Read the original file
    const input_file = try std.fs.openFileAbsolute(input_path, .{});
    defer input_file.close();

    const file_size = try input_file.getEndPos();
    var content = try allocator.alloc(u8, file_size);
    defer allocator.free(content);
    _ = try input_file.readAll(content);

    // --- Patch 1: mul_mat ---
    const target_function_mul_mat = "void ggml_compute_forward_mul_mat(";
    const insertion_marker_mul_mat = "GGML_TENSOR_BINARY_OP_LOCALS";

    const func_start_mul_mat = std.mem.indexOf(u8, content, target_function_mul_mat) orelse {
        std.debug.print("ERROR: Could not find function: {s}\n", .{target_function_mul_mat});
        return error.FunctionNotFound;
    };

    const marker_pos_mul_mat = std.mem.indexOf(u8, content[func_start_mul_mat..], insertion_marker_mul_mat) orelse {
        std.debug.print("ERROR: Could not find insertion marker: {s}\n", .{insertion_marker_mul_mat});
        return error.MarkerNotFound;
    };
    const absolute_marker_pos_mul_mat = func_start_mul_mat + marker_pos_mul_mat;

    const line_end_mul_mat = std.mem.indexOf(u8, content[absolute_marker_pos_mul_mat..], "\n") orelse {
        return error.LineEndNotFound;
    };
    const insertion_point_mul_mat = absolute_marker_pos_mul_mat + line_end_mul_mat + 1;

    var modified_content = std.ArrayList(u8).empty;
    defer modified_content.deinit(allocator);

    try modified_content.appendSlice(allocator, content[0..insertion_point_mul_mat]);
    try modified_content.appendSlice(allocator, "\n");
    try modified_content.appendSlice(allocator, hook_code_mul_mat);

    var skip_pos_mul_mat = insertion_point_mul_mat;
    while (skip_pos_mul_mat < content.len and content[skip_pos_mul_mat] == '\n') {
        skip_pos_mul_mat += 1;
    }

    // Continue scanning for flash attn
    const remaining_content = content[skip_pos_mul_mat..];

    // --- Patch 2: flash_attn_ext ---
    const target_marker_flash_attn = "case GGML_OP_FLASH_ATTN_EXT:";
    const marker_pos_flash_attn = std.mem.indexOf(u8, remaining_content, target_marker_flash_attn) orelse {
        std.debug.print("WARNING: Could not find {s}. Not patching flash attention.\n", .{target_marker_flash_attn});
        try modified_content.appendSlice(allocator, remaining_content);
        return writeOutput(output_path, modified_content.items);
    };

    const absolute_marker_pos_flash_attn = marker_pos_flash_attn;
    const line_end_flash_attn = std.mem.indexOf(u8, remaining_content[absolute_marker_pos_flash_attn..], "\n") orelse {
        return error.LineEndNotFound;
    };
    const insertion_point_flash_attn = absolute_marker_pos_flash_attn + line_end_flash_attn + 1;

    try modified_content.appendSlice(allocator, remaining_content[0..insertion_point_flash_attn]);
    try modified_content.appendSlice(allocator, "\n");
    try modified_content.appendSlice(allocator, hook_code_flash_attn);

    var skip_pos_flash_attn = insertion_point_flash_attn;
    while (skip_pos_flash_attn < remaining_content.len and remaining_content[skip_pos_flash_attn] == '\n') {
        skip_pos_flash_attn += 1;
    }

    // Continue scanning for SILU
    const remaining_after_flash = remaining_content[skip_pos_flash_attn..];

    // --- Patch 3: SILU ---
    const target_marker_silu = "case GGML_OP_SILU:";
    const marker_pos_silu = std.mem.indexOf(u8, remaining_after_flash, target_marker_silu) orelse {
        std.debug.print("WARNING: Could not find {s}. Not patching SILU.\n", .{target_marker_silu});
        try modified_content.appendSlice(allocator, remaining_after_flash);
        return writeOutput(output_path, modified_content.items);
    };

    const absolute_marker_pos_silu = marker_pos_silu;
    const line_end_silu = std.mem.indexOf(u8, remaining_after_flash[absolute_marker_pos_silu..], "\n") orelse {
        return error.LineEndNotFound;
    };
    const insertion_point_silu = absolute_marker_pos_silu + line_end_silu + 1;

    try modified_content.appendSlice(allocator, remaining_after_flash[0..insertion_point_silu]);
    try modified_content.appendSlice(allocator, "\n");
    try modified_content.appendSlice(allocator, hook_code_silu);

    var skip_pos_silu = insertion_point_silu;
    while (skip_pos_silu < remaining_after_flash.len and remaining_after_flash[skip_pos_silu] == '\n') {
        skip_pos_silu += 1;
    }

    // Continue scanning for NORM
    const remaining_after_silu = remaining_after_flash[skip_pos_silu..];

    // --- Patch 4: NORM (LayerNorm) ---
    const target_marker_norm = "case GGML_OP_NORM:";
    const marker_pos_norm = std.mem.indexOf(u8, remaining_after_silu, target_marker_norm) orelse {
        std.debug.print("WARNING: Could not find {s}. Not patching NORM.\n", .{target_marker_norm});
        try modified_content.appendSlice(allocator, remaining_after_silu);
        return writeOutput(output_path, modified_content.items);
    };

    const absolute_marker_pos_norm = marker_pos_norm;
    const line_end_norm = std.mem.indexOf(u8, remaining_after_silu[absolute_marker_pos_norm..], "\n") orelse {
        return error.LineEndNotFound;
    };
    const insertion_point_norm = absolute_marker_pos_norm + line_end_norm + 1;

    try modified_content.appendSlice(allocator, remaining_after_silu[0..insertion_point_norm]);
    try modified_content.appendSlice(allocator, "\n");
    try modified_content.appendSlice(allocator, hook_code_norm);

    var skip_pos_norm = insertion_point_norm;
    while (skip_pos_norm < remaining_after_silu.len and remaining_after_silu[skip_pos_norm] == '\n') {
        skip_pos_norm += 1;
    }
    try modified_content.appendSlice(allocator, remaining_after_silu[skip_pos_norm..]);

    try writeOutput(output_path, modified_content.items);
    std.debug.print("Successfully patched ggml-cpu.c with SIMD, Flash Attention, SILU, and NORM backend hooks\n", .{});
}

fn writeOutput(output_path: []const u8, content: []const u8) !void {
    const output_file = try std.fs.createFileAbsolute(output_path, .{});
    defer output_file.close();
    try output_file.writeAll(content);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    if (args.len != 3) {
        std.debug.print("Usage: patch_ggml <input_path> <output_path>\n", .{});
        return error.InvalidArguments;
    }

    try patchGgmlCpu(allocator, args[1], args[2]);
}
