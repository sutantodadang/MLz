const std = @import("std");

const declaration_marker = "#include \"common.h\"\n";
const declaration_patch = declaration_marker ++
    "\n#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS\n" ++
    "extern bool mlz_ggml_residency_node_hooks_enabled(void);\n" ++
    "extern void mlz_ggml_residency_node_pre(struct ggml_tensor * node);\n" ++
    "extern void mlz_ggml_residency_node_post(struct ggml_tensor * node);\n" ++
    "extern bool mlz_ggml_residency_should_tile_mul_mat(struct ggml_tensor * node);\n" ++
    "extern bool mlz_ggml_residency_should_tile_mul_mat_id(struct ggml_tensor * node);\n" ++
    "extern size_t mlz_ggml_residency_tile_capacity(struct ggml_tensor * tensor, size_t tensor_offset);\n" ++
    "extern bool mlz_ggml_residency_tile_acquire(struct ggml_tensor * tensor, size_t tensor_offset, size_t byte_len);\n" ++
    "extern bool mlz_ggml_residency_tile_release(struct ggml_tensor * tensor);\n" ++
    "#endif /* GGML_USE_MLZ_RESIDENCY_HOOKS */\n";

const mul_mat_marker =
    "void ggml_compute_forward_mul_mat(\n" ++
    "        const struct ggml_compute_params * params,\n" ++
    "              struct ggml_tensor * dst) {\n";

const mul_mat_patch = mul_mat_marker ++
    "#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS\n" ++
    "    if (mlz_ggml_residency_should_tile_mul_mat(dst)) {\n" ++
    "        // Ensure every worker has selected the tiled branch before thread\n" ++
    "        // 0 is allowed to replace src0's reserved identity pointer.\n" ++
    "        ggml_barrier(params->threadpool);\n" ++
    "        struct ggml_tensor * mlz_src0 = dst->src[0];\n" ++
    "        const struct ggml_tensor * mlz_src1 = dst->src[1];\n" ++
    "        const int mlz_ith = params->ith;\n" ++
    "        const int mlz_nth = params->nth;\n" ++
    "        const enum ggml_type mlz_vec_dot_type = type_traits_cpu[mlz_src0->type].vec_dot_type;\n" ++
    "        const ggml_from_float_t mlz_from_float = type_traits_cpu[mlz_vec_dot_type].from_float;\n" ++
    "        const int64_t mlz_ne10 = mlz_src1->ne[0];\n" ++
    "\n" ++
    "        // Prepare src1 exactly as the stock kernel does. Decode normally\n" ++
    "        // arrives as F32 and must be converted to src0's vec-dot type.\n" ++
    "        if (mlz_src1->type != mlz_vec_dot_type) {\n" ++
    "            char * mlz_wdata = params->wdata;\n" ++
    "            const size_t mlz_nbw0 = ggml_type_size(mlz_vec_dot_type);\n" ++
    "            const size_t mlz_nbw1 = ggml_row_size(mlz_vec_dot_type, mlz_ne10);\n" ++
    "            const size_t mlz_nbw2 = mlz_nbw1 * mlz_src1->ne[1];\n" ++
    "            const size_t mlz_nbw3 = mlz_nbw2 * mlz_src1->ne[2];\n" ++
    "            GGML_ASSERT(params->wsize >= mlz_src1->ne[3] * mlz_nbw3);\n" ++
    "            GGML_ASSERT(mlz_src1->type == GGML_TYPE_F32);\n" ++
    "            for (int64_t i13 = 0; i13 < mlz_src1->ne[3]; ++i13) {\n" ++
    "                for (int64_t i12 = 0; i12 < mlz_src1->ne[2]; ++i12) {\n" ++
    "                    for (int64_t i11 = 0; i11 < mlz_src1->ne[1]; ++i11) {\n" ++
    "                        const size_t bs = ggml_blck_size(mlz_vec_dot_type);\n" ++
    "                        const int64_t block_start = (mlz_ith * mlz_ne10 / bs) / mlz_nth;\n" ++
    "                        const int64_t block_end = ((mlz_ith + 1) * mlz_ne10 / bs) / mlz_nth;\n" ++
    "                        mlz_from_float(\n" ++
    "                            (float *) ((char *) mlz_src1->data + i13 * mlz_src1->nb[3] + i12 * mlz_src1->nb[2] + i11 * mlz_src1->nb[1] + block_start * bs * mlz_src1->nb[0]),\n" ++
    "                            (void *) (mlz_wdata + i13 * mlz_nbw3 + i12 * mlz_nbw2 + i11 * mlz_nbw1 + block_start * mlz_nbw0),\n" ++
    "                            (block_end - block_start) * bs);\n" ++
    "                    }\n" ++
    "                }\n" ++
    "            }\n" ++
    "        }\n" ++
    "        ggml_barrier(params->threadpool);\n" ++
    "\n" ++
    "        const int64_t mlz_nr0 = mlz_src0->ne[1];\n" ++
    "        const int64_t mlz_nr1 = mlz_src1->ne[1] * mlz_src1->ne[2] * mlz_src1->ne[3];\n" ++
    "        const size_t mlz_row_bytes = mlz_src0->nb[1];\n" ++
    "        for (int64_t row_start = 0; row_start < mlz_nr0; ) {\n" ++
    "            if (mlz_ith == 0) {\n" ++
    "                const size_t tensor_offset = (size_t) row_start * mlz_row_bytes;\n" ++
    "                const size_t capacity = mlz_ggml_residency_tile_capacity(mlz_src0, tensor_offset);\n" ++
    "                int64_t tile_rows = (int64_t) (capacity / mlz_row_bytes);\n" ++
    "                tile_rows = MIN(tile_rows, mlz_nr0 - row_start);\n" ++
    "                if (tile_rows < 1 || tile_rows > INT_MAX ||\n" ++
    "                    !mlz_ggml_residency_tile_acquire(\n" ++
    "                        mlz_src0, tensor_offset, (size_t) tile_rows * mlz_row_bytes)) {\n" ++
    "                    atomic_store_explicit(&params->threadpool->current_chunk, -1, memory_order_release);\n" ++
    "                } else {\n" ++
    "                    atomic_store_explicit(&params->threadpool->current_chunk, (int) tile_rows, memory_order_release);\n" ++
    "                }\n" ++
    "            }\n" ++
    "            ggml_barrier(params->threadpool);\n" ++
    "            const int64_t tile_rows = atomic_load_explicit(\n" ++
    "                &params->threadpool->current_chunk, memory_order_acquire);\n" ++
    "            if (tile_rows < 1) {\n" ++
    "                GGML_ABORT(\"MLz MUL_MAT tile acquisition failed\");\n" ++
    "            }\n" ++
    "            const int64_t ir0_start = row_start + tile_rows * mlz_ith / mlz_nth;\n" ++
    "            const int64_t ir0_end = row_start + tile_rows * (mlz_ith + 1) / mlz_nth;\n" ++
    "            ggml_compute_forward_mul_mat_one_chunk(\n" ++
    "                params, dst, mlz_src0->type, 1, ir0_start, ir0_end, 0, mlz_nr1);\n" ++
    "            ggml_barrier(params->threadpool);\n" ++
    "            if (mlz_ith == 0 && !mlz_ggml_residency_tile_release(mlz_src0)) {\n" ++
    "                GGML_ABORT(\"MLz MUL_MAT tile release failed\");\n" ++
    "            }\n" ++
    "            ggml_barrier(params->threadpool);\n" ++
    "            row_start += tile_rows;\n" ++
    "        }\n" ++
    "        return;\n" ++
    "    }\n" ++
    "#endif /* GGML_USE_MLZ_RESIDENCY_HOOKS */\n";

const mul_mat_id_marker =
    "        const char * src0_cur = (const char *) src0->data + cur_a * nb02;\n" ++
    "        const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;\n";

const mul_mat_id_patch =
    "#ifdef GGML_USE_MLZ_RESIDENCY_HOOKS\n" ++
    "        if (mlz_ggml_residency_should_tile_mul_mat_id(dst)) {\n" ++
    "            const void * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;\n" ++
    "            const size_t row_size = ggml_row_size(vec_dot_type, ne10);\n" ++
    "            const int64_t nr0 = ne01;\n" ++
    "            for (int64_t row_start = 0; row_start < nr0; ) {\n" ++
    "                if (ith == 0) {\n" ++
    "                    const size_t tensor_offset = (size_t) cur_a * nb02 + (size_t) row_start * nb01;\n" ++
    "                    const size_t capacity = mlz_ggml_residency_tile_capacity((struct ggml_tensor *) src0, tensor_offset);\n" ++
    "                    int64_t tile_rows = (int64_t) (capacity / nb01);\n" ++
    "                    tile_rows = MIN(tile_rows, nr0 - row_start);\n" ++
    "                    if (tile_rows < 1 || tile_rows > INT_MAX ||\n" ++
    "                        !mlz_ggml_residency_tile_acquire((struct ggml_tensor *) src0, tensor_offset, (size_t) tile_rows * nb01)) {\n" ++
    "                        atomic_store_explicit((atomic_int *) (atomic_current_chunk + cur_a), -1, memory_order_release);\n" ++
    "                    } else {\n" ++
    "                        atomic_store_explicit((atomic_int *) (atomic_current_chunk + cur_a), (int) tile_rows, memory_order_release);\n" ++
    "                    }\n" ++
    "                }\n" ++
    "                ggml_barrier(params->threadpool);\n" ++
    "                const int64_t tile_rows = atomic_load_explicit((atomic_int *) (atomic_current_chunk + cur_a), memory_order_acquire);\n" ++
    "                if (tile_rows < 1) { GGML_ABORT(\"MLz MUL_MAT_ID tile acquisition failed\"); }\n" ++
    "                const int64_t ir0_start = row_start + tile_rows * ith / nth;\n" ++
    "                const int64_t ir0_end = row_start + tile_rows * (ith + 1) / nth;\n" ++
    "                const char * src0_cur = (const char *) src0->data + cur_a * nb02;\n" ++
    "                ggml_compute_forward_mul_mat_id_one_chunk(dst, src0, src1, ids, cur_a,\n" ++
    "                    ir0_start, ir0_end, 0, cne1, src0_cur, matrix_rows, row_size, src1_cont, wdata);\n" ++
    "                ggml_barrier(params->threadpool);\n" ++
    "                if (ith == 0 && !mlz_ggml_residency_tile_release((struct ggml_tensor *) src0)) {\n" ++
    "                    GGML_ABORT(\"MLz MUL_MAT_ID tile release failed\");\n" ++
    "                }\n" ++
    "                ggml_barrier(params->threadpool);\n" ++
    "                row_start += tile_rows;\n" ++
    "            }\n" ++
    "            continue;\n" ++
    "        }\n" ++
    "#endif /* GGML_USE_MLZ_RESIDENCY_HOOKS */\n" ++
    mul_mat_id_marker;

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
    const already_patched = std.mem.indexOf(u8, input, "mlz_ggml_residency_node_pre(node);") != null;
    if (already_patched) {
        std.debug.print("refusing to patch ggml-cpu.c: residency hook patch is already present\n", .{});
        return error.AlreadyPatched;
    }

    const patched_declaration = try replaceExactlyOnce(allocator, input, declaration_marker, declaration_patch, "include declaration");
    defer allocator.free(patched_declaration);
    const patched_mul_mat = try replaceExactlyOnce(allocator, patched_declaration, mul_mat_marker, mul_mat_patch, "MUL_MAT function");
    defer allocator.free(patched_mul_mat);
    const patched_mul_mat_id = try replaceExactlyOnce(allocator, patched_mul_mat, mul_mat_id_marker, mul_mat_id_patch, "MUL_MAT_ID expert loop");
    defer allocator.free(patched_mul_mat_id);
    const patched_loop_start = try replaceExactlyOnce(allocator, patched_mul_mat_id, before_loop_marker, before_loop_patch, "graph loop start");
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
