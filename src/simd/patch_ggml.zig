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

    var modified_content = std.ArrayList(u8).init(allocator);
    defer modified_content.deinit();

    try modified_content.appendSlice(content[0..insertion_point_mul_mat]);
    try modified_content.appendSlice("\n");
    try modified_content.appendSlice(hook_code_mul_mat);

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
        try modified_content.appendSlice(remaining_content);
        return writeOutput(output_path, modified_content.items);
    };
    
    const absolute_marker_pos_flash_attn = marker_pos_flash_attn;
    const line_end_flash_attn = std.mem.indexOf(u8, remaining_content[absolute_marker_pos_flash_attn..], "\n") orelse {
        return error.LineEndNotFound;
    };
    const insertion_point_flash_attn = absolute_marker_pos_flash_attn + line_end_flash_attn + 1;

    try modified_content.appendSlice(remaining_content[0..insertion_point_flash_attn]);
    try modified_content.appendSlice("\n");
    try modified_content.appendSlice(hook_code_flash_attn);

    var skip_pos_flash_attn = insertion_point_flash_attn;
    while (skip_pos_flash_attn < remaining_content.len and remaining_content[skip_pos_flash_attn] == '\n') {
        skip_pos_flash_attn += 1;
    }
    try modified_content.appendSlice(remaining_content[skip_pos_flash_attn..]);

    try writeOutput(output_path, modified_content.items);
    std.debug.print("Successfully patched ggml-cpu.c with SIMD and Flash Attention backend hooks\n", .{});
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
