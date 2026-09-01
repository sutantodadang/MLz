const std = @import("std");

const declaration_marker = "#include \"common.h\"\n";
const declaration_patch = declaration_marker ++
    "\n#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS\n" ++
    "extern bool mlz_ggml_residency_node_hooks_enabled(void);\n" ++
    "extern void mlz_ggml_residency_node_pre(struct ggml_tensor * node);\n" ++
    "extern void mlz_ggml_residency_node_post(struct ggml_tensor * node);\n" ++
    "#endif /* GGML_USE_MLZ_RESIDENCY_HOOKS */\n";

const loop_marker =
    "        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time\n" ++
    "        // Try fused ops, fall back to normal compute\n" ++
    "        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);\n" ++
    "        if (n_fused > 0) {\n" ++
    "            node_n += n_fused;\n" ++
    "        } else {\n" ++
    "            ggml_compute_forward(&params, node);\n" ++
    "        }\n";

const loop_patch =
    "#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS\n" ++
    "        if (mlz_hooks_enabled) {\n" ++
    "            if (state->ith == 0) {\n" ++
    "                mlz_ggml_residency_node_pre(node);\n" ++
    "            }\n" ++
    "            ggml_barrier(state->threadpool);\n" ++
    "            ggml_compute_forward(&params, node);\n" ++
    "        } else {\n" ++
    "#endif\n" ++
    loop_marker ++
    "#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS\n" ++
    "        }\n" ++
    "#endif\n";

const before_loop_marker =
    "    for (int node_n = 0; node_n < cgraph->n_nodes && atomic_load_explicit(&tp->abort, memory_order_relaxed) != node_n; node_n++) {\n";
const before_loop_patch =
    "#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS\n" ++
    "    // The caller changes this switch only between graph executions. Sample\n" ++
    "    // once so every node in this graph uses one barrier protocol.\n" ++
    "    const bool mlz_hooks_enabled = mlz_ggml_residency_node_hooks_enabled();\n" ++
    "#endif\n\n" ++ before_loop_marker;

const barrier_marker =
    "        if (node_n + 1 < cgraph->n_nodes) {\n" ++
    "            ggml_barrier(state->threadpool);\n" ++
    "        }\n";
const barrier_patch =
    "#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS\n" ++
    "        if (mlz_hooks_enabled) {\n" ++
    "            ggml_barrier(state->threadpool);\n" ++
    "            if (state->ith == 0) {\n" ++
    "                mlz_ggml_residency_node_post(node);\n" ++
    "            }\n" ++
    "            ggml_barrier(state->threadpool);\n" ++
    "        } else {\n" ++
    "#endif\n" ++
    barrier_marker ++
    "#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS\n" ++
    "        }\n" ++
    "#endif\n";

pub fn main() !void {
    var gpa: std.heap.DebugAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    if (args.len != 3) {
        std.debug.print("usage: patch-ggml-residency <input-ggml-cpu.c> <output-ggml-cpu.c>\n", .{});
        return error.InvalidArguments;
    }

    const input = try std.fs.cwd().readFileAlloc(allocator, args[1], 128 * 1024 * 1024);
    defer allocator.free(input);
    if (std.mem.indexOf(u8, input, "GGML_USE_MLZ_RESIDENCY_HOOKS") != null) {
        std.debug.print("refusing to patch ggml-cpu.c: residency hook patch is already present\n", .{});
        return error.AlreadyPatched;
    }

    const patched_declaration = try replaceExactlyOnce(allocator, input, declaration_marker, declaration_patch, "include declaration");
    defer allocator.free(patched_declaration);
    const patched_loop_start = try replaceExactlyOnce(allocator, patched_declaration, before_loop_marker, before_loop_patch, "graph loop start");
    defer allocator.free(patched_loop_start);
    const patched_compute = try replaceExactlyOnce(allocator, patched_loop_start, loop_marker, loop_patch, "node compute block");
    defer allocator.free(patched_compute);
    const patched = try replaceExactlyOnce(allocator, patched_compute, barrier_marker, barrier_patch, "node barrier block");
    defer allocator.free(patched);

    const output = try std.fs.cwd().createFile(args[2], .{});
    defer output.close();
    try output.writeAll(patched);
}

fn replaceExactlyOnce(
    allocator: std.mem.Allocator,
    input: []const u8,
    marker: []const u8,
    replacement: []const u8,
    name: []const u8,
) ![]u8 {
    const first = std.mem.indexOf(u8, input, marker) orelse {
        std.debug.print("cannot patch ggml-cpu.c: missing exact {s} marker\n", .{name});
        return error.MarkerMissing;
    };
    if (std.mem.indexOfPos(u8, input, first + marker.len, marker) != null) {
        std.debug.print("cannot patch ggml-cpu.c: {s} marker is not unique\n", .{name});
        return error.MarkerNotUnique;
    }

    const result = try allocator.alloc(u8, input.len - marker.len + replacement.len);
    @memcpy(result[0..first], input[0..first]);
    @memcpy(result[first .. first + replacement.len], replacement);
    @memcpy(result[first + replacement.len ..], input[first + marker.len ..]);
    return result;
}
