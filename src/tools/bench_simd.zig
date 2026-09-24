const std = @import("std");

/// When true (--json), each kernel emits one NDJSON line instead of a table row,
/// and the human-readable headers are suppressed. Consumed by tools/bench_gate.py.
var json_mode: bool = false;

/// Emit one kernel result: NDJSON in --json mode, otherwise a table row.
fn report(name: []const u8, sec: f64, metric: f64) void {
    if (json_mode) {
        std.debug.print("{{\"kernel\":\"{s}\",\"metric\":{d:.4}}}\n", .{ name, metric });
    } else {
        std.debug.print("{s:<28} | {d:<10.4} | {d:<10.2}\n", .{ name, sec, metric });
    }
}

// vec_dot kernels (x86_64 AVX2 + AVX-512)
extern "c" fn simd_vec_dot_q4_0_q8_0_avx2(n: c_int, result: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q4_0_q8_0_avx512(n: c_int, result: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q8_0_q8_0_avx2(n: c_int, result: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q8_0_q8_0_avx512(n: c_int, result: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q2_k_q8_k_avx2(n: c_int, result: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q2_k_q8_k_avx512(n: c_int, result: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q3_k_q8_k_avx2(n: c_int, result: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q3_k_q8_k_avx512(n: c_int, result: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q4_k_q8_k_avx2(n: c_int, result: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q4_k_q8_k_avx512(n: c_int, result: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q5_k_q8_k_avx2(n: c_int, result: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q5_k_q8_k_avx512(n: c_int, result: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q6_k_q8_k_avx2(n: c_int, result: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q6_k_q8_k_avx512(n: c_int, result: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q8_k_q8_k_avx2(n: c_int, result: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q8_k_q8_k_avx512(n: c_int, result: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;

// unary kernels (x86_64 AVX2 + AVX-512, opt-in env-gated)
extern "c" fn simd_rms_norm_f32_avx2(n: c_int, eps: f32, x: ?*const f32, y: ?*f32) void;
extern "c" fn simd_rms_norm_f32_avx512(n: c_int, eps: f32, x: ?*const f32, y: ?*f32) void;
extern "c" fn simd_rope_neox_f32_avx2(n_pairs: i64, cache: ?*const f32, src: ?*const f32, dst: ?*f32) void;
extern "c" fn simd_rope_neox_f32_avx512(n_pairs: i64, cache: ?*const f32, src: ?*const f32, dst: ?*f32) void;

// new unary kernels — quantization, SiLU, layer_norm, rope_standard, vec_dot_f32
extern "c" fn simd_quantize_q8_0_f32_avx2(n: c_int, x: ?*const f32, y: ?*anyopaque) void;
extern "c" fn simd_quantize_q8_0_f32_avx512(n: c_int, x: ?*const f32, y: ?*anyopaque) void;
extern "c" fn simd_quantize_q8_k_f32_avx2(n: c_int, x: ?*const f32, y: ?*anyopaque) void;
extern "c" fn simd_quantize_q8_k_f32_avx512(n: c_int, x: ?*const f32, y: ?*anyopaque) void;
extern "c" fn simd_silu_f32_avx2(n: c_int, x: ?*const f32, y: ?*f32) void;
extern "c" fn simd_silu_f32_avx512(n: c_int, x: ?*const f32, y: ?*f32) void;
extern "c" fn simd_layer_norm_f32_avx2(n: c_int, eps: f32, x: ?*const f32, y: ?*f32) void;
extern "c" fn simd_layer_norm_f32_avx512(n: c_int, eps: f32, x: ?*const f32, y: ?*f32) void;
extern "c" fn simd_rope_standard_f32_avx2(n_pairs: i64, cache: ?*const f32, src: ?*const f32, dst: ?*f32) void;
extern "c" fn simd_rope_standard_f32_avx512(n_pairs: i64, cache: ?*const f32, src: ?*const f32, dst: ?*f32) void;
extern "c" fn simd_vec_dot_f32_f32_avx2(n: c_int, result: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_f32_f32_avx512(n: c_int, result: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_check_avx512_vnni() bool;
extern "c" fn simd_gemm_s8s8s32_avx2(M: c_int, N: c_int, K: c_int, A: [*]const i8, B: [*]const i8, C: [*]i32) void;
extern "c" fn simd_gemm_s8s8s32_avx512vnni(M: c_int, N: c_int, K: c_int, A: [*]const i8, B: [*]const i8, C: [*]i32) void;
extern "c" fn simd_gemm_s8s8s32_avx2_t(M: c_int, N: c_int, K: c_int, A: [*]const i8, B: [*]const i8, C: [*]i32) void;
extern "c" fn simd_gemm_s8s8s32_avx512vnni_t(M: c_int, N: c_int, K: c_int, A: [*]const i8, B: [*]const i8, C: [*]i32) void;

// NEON kernels (aarch64)
extern "c" fn simd_quantize_q8_0_f32_neon(n: c_int, x: ?*const f32, y: ?*anyopaque) void;
extern "c" fn simd_quantize_q8_k_f32_neon(n: c_int, x: ?*const f32, y: ?*anyopaque) void;
extern "c" fn simd_silu_f32_neon(n: c_int, x: ?*const f32, y: ?*f32) void;
extern "c" fn simd_layer_norm_f32_neon(n: c_int, eps: f32, x: ?*const f32, y: ?*f32) void;
extern "c" fn simd_rope_standard_f32_neon(n_pairs: i64, cache: ?*const f32, src: ?*const f32, dst: ?*f32) void;
extern "c" fn simd_vec_dot_f32_f32_neon(n: c_int, result: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;

pub fn main() !void {
    const N = 4096;
    const num_blocks_legacy = N / 32;
    const num_blocks_k = N / 256;

    const q4_row_size = num_blocks_legacy * 18;
    const q8_row_size = num_blocks_legacy * 34;

    // Q2_K: block 256 → 84 bytes/block
    const q2_k_row_size = num_blocks_k * 84;
    // Q3_K: block 256 → 126 bytes/block
    const q3_k_row_size = num_blocks_k * 126;
    // Q4_K: block 256 → 144 bytes/block
    const q4_k_row_size = num_blocks_k * 144;
    // Q5_K: block 256 → 176 bytes/block (d+dmin=4, scales=12, qh=32, qs=128)
    const q5_k_row_size = num_blocks_k * 176;
    // Q6_K: block 256 → 210 bytes/block
    const q6_k_row_size = num_blocks_k * 210;
    // Q8_K: block 256 → 292 bytes/block
    const q8_k_row_size = num_blocks_k * 292;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    {
        const args = try std.process.argsAlloc(allocator);
        defer std.process.argsFree(allocator, args);
        for (args) |a| {
            if (std.mem.eql(u8, a, "--json")) json_mode = true;
        }
    }

    // Align to 64 bytes for AVX-512
    const alignment = comptime std.mem.Alignment.fromByteUnits(64);

    // vec_dot buffers
    const vx_q4 = try allocator.alignedAlloc(u8, alignment, q4_row_size);
    const vy_q8 = try allocator.alignedAlloc(u8, alignment, q8_row_size);
    const vx_q8 = try allocator.alignedAlloc(u8, alignment, q8_row_size);
    const vx_q2_k = try allocator.alignedAlloc(u8, alignment, q2_k_row_size);
    const vx_q3_k = try allocator.alignedAlloc(u8, alignment, q3_k_row_size);
    const vx_q4_k = try allocator.alignedAlloc(u8, alignment, q4_k_row_size);
    const vx_q5_k = try allocator.alignedAlloc(u8, alignment, q5_k_row_size);
    const vx_q6_k = try allocator.alignedAlloc(u8, alignment, q6_k_row_size);
    const vx_q8_k = try allocator.alignedAlloc(u8, alignment, q8_k_row_size);
    const vy_q8_k = try allocator.alignedAlloc(u8, alignment, q8_k_row_size);

    // unary buffers (f32, aligned)
    const x_rms = try allocator.alignedAlloc(f32, alignment, N);
    const y_rms = try allocator.alignedAlloc(f32, alignment, N);
    const rope_cache = try allocator.alignedAlloc(f32, alignment, N); // N/2 pairs * 2 = N floats
    const rope_src = try allocator.alignedAlloc(f32, alignment, N);
    const rope_dst = try allocator.alignedAlloc(f32, alignment, N);

    // quantize output buffers
    const x_quant = try allocator.alignedAlloc(f32, alignment, N);
    const y_quant_q8_0 = try allocator.alignedAlloc(u8, alignment, q8_row_size);
    const y_quant_q8_k = try allocator.alignedAlloc(u8, alignment, q8_k_row_size);

    // Random init
    std.crypto.random.bytes(vx_q4);
    std.crypto.random.bytes(vy_q8);
    std.crypto.random.bytes(vx_q8);
    std.crypto.random.bytes(vx_q2_k);
    std.crypto.random.bytes(vx_q3_k);
    std.crypto.random.bytes(vx_q4_k);
    std.crypto.random.bytes(vx_q5_k);
    std.crypto.random.bytes(vx_q6_k);
    std.crypto.random.bytes(vx_q8_k);
    std.crypto.random.bytes(vy_q8_k);

    // Fill unary buffers with valid f32
    for (x_rms) |*v| v.* = @as(f32, @floatFromInt(std.crypto.random.int(u32) % 1000)) - 500.0;
    // rope cache: cos/sin interleaved for N/2 pairs → N floats
    for (rope_cache) |*v| v.* = @as(f32, @floatFromInt(std.crypto.random.int(u32) % 2000)) / 1000.0;
    for (rope_src) |*v| v.* = @as(f32, @floatFromInt(std.crypto.random.int(u32) % 2000)) / 1000.0;

    // Fill quantize input with valid f32
    for (x_quant) |*v| v.* = @as(f32, @floatFromInt(std.crypto.random.int(u32) % 1000)) - 500.0;

    const iterations = 100_000;
    var res: f32 = 0;

    if (!json_mode) {
        std.debug.print("=== MLz SIMD Baseline Bench (PLAN-ASSEMBLY-REWRITE Step 2) ===\n", .{});
        std.debug.print("N={d}  iterations={d}\n\n", .{ N, iterations });
        std.debug.print("{s:<28} | {s:<10} | {s:<14}\n", .{ "Kernel", "Time (s)", "GFLOPS" });
        std.debug.print("{s:-<56}\n", .{""});
    }

    // ------- vec_dot benchmarks -------
    // Warmup
    simd_vec_dot_q4_0_q8_0_avx2(N, &res, vx_q4.ptr, vy_q8.ptr);

    // Q4_0
    run_vec_dot("Q4_0 x Q8_0 (AVX2)", simd_vec_dot_q4_0_q8_0_avx2, N, vx_q4.ptr, vy_q8.ptr, iterations);
    run_vec_dot("Q4_0 x Q8_0 (AVX-512)", simd_vec_dot_q4_0_q8_0_avx512, N, vx_q4.ptr, vy_q8.ptr, iterations);

    // Q8_0
    run_vec_dot("Q8_0 x Q8_0 (AVX2)", simd_vec_dot_q8_0_q8_0_avx2, N, vx_q8.ptr, vy_q8.ptr, iterations);
    run_vec_dot("Q8_0 x Q8_0 (AVX-512)", simd_vec_dot_q8_0_q8_0_avx512, N, vx_q8.ptr, vy_q8.ptr, iterations);

    // Q2_K
    run_vec_dot("Q2_K x Q8_K (AVX2)", simd_vec_dot_q2_k_q8_k_avx2, N, vx_q2_k.ptr, vy_q8_k.ptr, iterations);
    run_vec_dot("Q2_K x Q8_K (AVX-512)", simd_vec_dot_q2_k_q8_k_avx512, N, vx_q2_k.ptr, vy_q8_k.ptr, iterations);

    // Q3_K
    run_vec_dot("Q3_K x Q8_K (AVX2)", simd_vec_dot_q3_k_q8_k_avx2, N, vx_q3_k.ptr, vy_q8_k.ptr, iterations);
    run_vec_dot("Q3_K x Q8_K (AVX-512)", simd_vec_dot_q3_k_q8_k_avx512, N, vx_q3_k.ptr, vy_q8_k.ptr, iterations);

    // Q4_K
    run_vec_dot("Q4_K x Q8_K (AVX2)", simd_vec_dot_q4_k_q8_k_avx2, N, vx_q4_k.ptr, vy_q8_k.ptr, iterations);
    run_vec_dot("Q4_K x Q8_K (AVX-512)", simd_vec_dot_q4_k_q8_k_avx512, N, vx_q4_k.ptr, vy_q8_k.ptr, iterations);

    // Q5_K
    run_vec_dot("Q5_K x Q8_K (AVX2)", simd_vec_dot_q5_k_q8_k_avx2, N, vx_q5_k.ptr, vy_q8_k.ptr, iterations);
    run_vec_dot("Q5_K x Q8_K (AVX-512)", simd_vec_dot_q5_k_q8_k_avx512, N, vx_q5_k.ptr, vy_q8_k.ptr, iterations);

    // Q6_K
    run_vec_dot("Q6_K x Q8_K (AVX2)", simd_vec_dot_q6_k_q8_k_avx2, N, vx_q6_k.ptr, vy_q8_k.ptr, iterations);
    run_vec_dot("Q6_K x Q8_K (AVX-512)", simd_vec_dot_q6_k_q8_k_avx512, N, vx_q6_k.ptr, vy_q8_k.ptr, iterations);

    // Q8_K
    run_vec_dot("Q8_K x Q8_K (AVX2)", simd_vec_dot_q8_k_q8_k_avx2, N, vx_q8_k.ptr, vy_q8_k.ptr, iterations);
    run_vec_dot("Q8_K x Q8_K (AVX-512)", simd_vec_dot_q8_k_q8_k_avx512, N, vx_q8_k.ptr, vy_q8_k.ptr, iterations);

    // ------- unary benchmarks -------
    if (!json_mode) {
        std.debug.print("\n--- Unary (Opt-in, env-gated) ---\n", .{});
        std.debug.print("{s:<28} | {s:<10} | {s:<14}\n", .{ "Kernel", "Time (s)", "GigaOps/s" });
        std.debug.print("{s:-<56}\n", .{""});
    }

    const eps: f32 = 1e-5;
    const rope_iterations = 50_000;

    // rms_norm
    run_rms_norm("rms_norm_f32 (AVX2)", simd_rms_norm_f32_avx2, N, eps, @ptrCast(x_rms.ptr), @ptrCast(y_rms.ptr), iterations);
    run_rms_norm("rms_norm_f32 (AVX-512)", simd_rms_norm_f32_avx512, N, eps, @ptrCast(x_rms.ptr), @ptrCast(y_rms.ptr), iterations);

    // rope_neox — n_pairs = N/2
    const n_pairs: i64 = N / 2;
    run_rope("rope_neox_f32 (AVX2)", simd_rope_neox_f32_avx2, n_pairs, @ptrCast(rope_cache.ptr), @ptrCast(rope_src.ptr), @ptrCast(rope_dst.ptr), rope_iterations);
    run_rope("rope_neox_f32 (AVX-512)", simd_rope_neox_f32_avx512, n_pairs, @ptrCast(rope_cache.ptr), @ptrCast(rope_src.ptr), @ptrCast(rope_dst.ptr), rope_iterations);

    // ------- new kernel benchmarks -------
    const enable_new_kernels = true;
    if (enable_new_kernels) {
        if (!json_mode) {
            std.debug.print("\n--- New Kernels ---\n", .{});
        std.debug.print("{s:<28} | {s:<10} | {s:<14}\n", .{ "Kernel", "Time (s)", "GigaOps/s" });
        std.debug.print("{s:-<56}\n", .{""});
    }

    // quantize_q8_0_f32
    run_quantize("quantize_q8_0_f32 (AVX2)", simd_quantize_q8_0_f32_avx2, N, @ptrCast(x_quant.ptr), @ptrCast(y_quant_q8_0.ptr), iterations);
    run_quantize("quantize_q8_0_f32 (AVX-512)", simd_quantize_q8_0_f32_avx512, N, @ptrCast(x_quant.ptr), @ptrCast(y_quant_q8_0.ptr), iterations);

    // quantize_q8_k_f32
    run_quantize("quantize_q8_k_f32 (AVX2)", simd_quantize_q8_k_f32_avx2, N, @ptrCast(x_quant.ptr), @ptrCast(y_quant_q8_k.ptr), iterations);
    run_quantize("quantize_q8_k_f32 (AVX-512)", simd_quantize_q8_k_f32_avx512, N, @ptrCast(x_quant.ptr), @ptrCast(y_quant_q8_k.ptr), iterations);

    // silu_f32
    run_unary("silu_f32 (AVX2)", simd_silu_f32_avx2, N, @ptrCast(x_rms.ptr), @ptrCast(y_rms.ptr), iterations);
    run_unary("silu_f32 (AVX-512)", simd_silu_f32_avx512, N, @ptrCast(x_rms.ptr), @ptrCast(y_rms.ptr), iterations);

    // layer_norm_f32
    run_rms_norm("layer_norm_f32 (AVX2)", simd_layer_norm_f32_avx2, N, eps, @ptrCast(x_rms.ptr), @ptrCast(y_rms.ptr), iterations);
    run_rms_norm("layer_norm_f32 (AVX-512)", simd_layer_norm_f32_avx512, N, eps, @ptrCast(x_rms.ptr), @ptrCast(y_rms.ptr), iterations);

    // rope_standard_f32
    run_rope("rope_standard_f32 (AVX2)", simd_rope_standard_f32_avx2, n_pairs, @ptrCast(rope_cache.ptr), @ptrCast(rope_src.ptr), @ptrCast(rope_dst.ptr), rope_iterations);
    run_rope("rope_standard_f32 (AVX-512)", simd_rope_standard_f32_avx512, n_pairs, @ptrCast(rope_cache.ptr), @ptrCast(rope_src.ptr), @ptrCast(rope_dst.ptr), rope_iterations);

    // vec_dot_f32_f32
    run_vec_dot("vec_dot_f32_f32 (AVX2)", simd_vec_dot_f32_f32_avx2, N, @ptrCast(x_rms.ptr), @ptrCast(y_rms.ptr), iterations);
    run_vec_dot("vec_dot_f32_f32 (AVX-512)", simd_vec_dot_f32_f32_avx512, N, @ptrCast(x_rms.ptr), @ptrCast(y_rms.ptr), iterations);

    // gemm_s8s8s32 — INT8 GEMM microkernel (M x K . N x K^T -> M x N)
    {
        const GM: usize = 64;
        const GN: usize = 64;
        const GK: usize = 512;
        const gemm_iters: usize = 400;
        const ga = try allocator.alloc(i8, GM * GK);
        defer allocator.free(ga);
        const gb = try allocator.alloc(i8, GN * GK);
        defer allocator.free(gb);
        const gc = try allocator.alloc(i32, GM * GN);
        defer allocator.free(gc);
        for (ga, 0..) |*v, i| v.* = @intCast(@as(i32, @intCast(i % 255)) - 127);
        for (gb, 0..) |*v, i| v.* = @intCast(@as(i32, @intCast((i * 7) % 255)) - 127);
        run_gemm("gemm_s8s8s32 (AVX2)", simd_gemm_s8s8s32_avx2, GM, GN, GK, ga.ptr, gb.ptr, gc.ptr, gemm_iters);
        run_gemm("gemm_s8s8s32 (AVX2-tiled)", simd_gemm_s8s8s32_avx2_t, GM, GN, GK, ga.ptr, gb.ptr, gc.ptr, gemm_iters);
        if (simd_check_avx512_vnni()) {
            run_gemm("gemm_s8s8s32 (VNNI)", simd_gemm_s8s8s32_avx512vnni, GM, GN, GK, ga.ptr, gb.ptr, gc.ptr, gemm_iters);
            run_gemm("gemm_s8s8s32 (VNNI-tiled)", simd_gemm_s8s8s32_avx512vnni_t, GM, GN, GK, ga.ptr, gb.ptr, gc.ptr, gemm_iters);
        }
    }
    } // end enable_new_kernels

    // Prevent optimizer from eliminating unused result
    std.process.cleanExit();
}

fn run_vec_dot(name: []const u8, kernel: anytype, N: usize, vx: ?*const anyopaque, vy: ?*const anyopaque, iter: usize) void {
    var res: f32 = 0;
    var timer = std.time.Timer.start() catch unreachable;
    for (0..iter) |_| {
        kernel(@intCast(N), &res, vx, vy);
    }
    const ns = timer.read();
    const sec = @as(f64, @floatFromInt(ns)) / 1e9;
    // vec_dot: 2*N - 1 FLOPS per call (N muls + N-1 adds)
    const flops = @as(f64, @floatFromInt(2 * N - 1)) * @as(f64, @floatFromInt(iter));
    const gflops = flops / sec / 1e9;
    report(name, sec, gflops);
}

fn run_gemm(name: []const u8, kernel: anytype, M: usize, N: usize, K: usize, A: [*]const i8, B: [*]const i8, C: [*]i32, iter: usize) void {
    kernel(@intCast(M), @intCast(N), @intCast(K), A, B, C); // warm
    var timer = std.time.Timer.start() catch unreachable;
    for (0..iter) |_| {
        kernel(@intCast(M), @intCast(N), @intCast(K), A, B, C);
    }
    const ns = timer.read();
    const sec = @as(f64, @floatFromInt(ns)) / 1e9;
    // GEMM: 2*M*N*K ops (one mul + one add per inner term)
    const ops = @as(f64, @floatFromInt(2 * M * N * K)) * @as(f64, @floatFromInt(iter));
    const gops = ops / sec / 1e9;
    report(name, sec, gops);
}

fn run_rms_norm(name: []const u8, kernel: anytype, n: usize, eps: f32, x: ?*const f32, y: ?*f32, iter: usize) void {
    // Compute y once (kernel writes to y, we don't care about results)
    kernel(@intCast(n), eps, x, y);
    var timer = std.time.Timer.start() catch unreachable;
    for (0..iter) |_| {
        kernel(@intCast(n), eps, x, y);
    }
    const ns = timer.read();
    const sec = @as(f64, @floatFromInt(ns)) / 1e9;
    // rms_norm: ~2*N ops (N squares, reduce add, N mults by 1/rms)
    const ops = @as(f64, @floatFromInt(2 * n)) * @as(f64, @floatFromInt(iter));
    const gops = ops / sec / 1e9;
    report(name, sec, gops);
}

fn run_rope(name: []const u8, kernel: anytype, n_pairs: i64, cache: ?*const f32, src: ?*const f32, dst: ?*f32, iter: usize) void {
    // Warmup
    kernel(n_pairs, cache, src, dst);
    var timer = std.time.Timer.start() catch unreachable;
    for (0..iter) |_| {
        kernel(n_pairs, cache, src, dst);
    }
    const ns = timer.read();
    const sec = @as(f64, @floatFromInt(ns)) / 1e9;
    // rope_neox: 6 FLOPS per pair (4 muls + 2 adds per complex rotation)
    const ops = @as(f64, @floatFromInt(6 * @as(usize, @intCast(n_pairs)))) * @as(f64, @floatFromInt(iter));
    const gops = ops / sec / 1e9;
    report(name, sec, gops);
}

fn run_quantize(name: []const u8, kernel: anytype, n: usize, x: ?*const f32, y: ?*anyopaque, iter: usize) void {
    // Warmup
    kernel(@intCast(n), x, y);
    var timer = std.time.Timer.start() catch unreachable;
    for (0..iter) |_| {
        kernel(@intCast(n), x, y);
    }
    const ns = timer.read();
    const sec = @as(f64, @floatFromInt(ns)) / 1e9;
    // quantize: ~4*N ops (abs max scan, scale compute, N quantizations)
    const ops = @as(f64, @floatFromInt(4 * n)) * @as(f64, @floatFromInt(iter));
    const gops = ops / sec / 1e9;
    report(name, sec, gops);
}

fn run_unary(name: []const u8, kernel: anytype, n: usize, x: ?*const f32, y: ?*f32, iter: usize) void {
    // Warmup
    kernel(@intCast(n), x, y);
    var timer = std.time.Timer.start() catch unreachable;
    for (0..iter) |_| {
        kernel(@intCast(n), x, y);
    }
    const ns = timer.read();
    const sec = @as(f64, @floatFromInt(ns)) / 1e9;
    // unary: ~3*N ops (exp, add, div or similar per element)
    const ops = @as(f64, @floatFromInt(3 * n)) * @as(f64, @floatFromInt(iter));
    const gops = ops / sec / 1e9;
    report(name, sec, gops);
}
