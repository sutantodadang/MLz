const std = @import("std");

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
extern "c" fn simd_vec_dot_q6_k_q8_k_avx2(n: c_int, result: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q6_k_q8_k_avx512(n: c_int, result: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q8_k_q8_k_avx2(n: c_int, result: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q8_k_q8_k_avx512(n: c_int, result: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;

pub fn main() !void {
    const N = 4096;
    const num_blocks_legacy = N / 32;
    const num_blocks_k = N / 256;

    const q4_row_size = num_blocks_legacy * 18;
    const q8_row_size = num_blocks_legacy * 34;

    // K-Quants sizes (GGML_TYPE_Q2_K = 256 elements per block)
    // Q2_K: block size 256.
    //   - scales: 16 bytes (u8)
    //   - qs: 64 bytes (2-bit weights)
    //   - d: 2 bytes (f16), dmin: 2 bytes (f16) -> 4 bytes
    //   Total: 16 + 64 + 4 = 84 bytes per block (256 elements)
    const q2_k_row_size = num_blocks_k * 84;

    // Q6_K: block size 256.
    //   - ql: 128 bytes (4-bit)
    //   - qh: 64 bytes (2-bit)
    //   - scales: 16 bytes (i8)
    //   - d: 2 bytes (f16)
    //   Total: 128 + 64 + 16 + 2 = 210 bytes per block
    const q6_k_row_size = num_blocks_k * 210;

    // Q3_K: block size 256.
    //   - d: 2 bytes (f16)
    //   - scales: 12 bytes (u8)
    //   - hmask: 16 bytes (sign mask, 1 bit per 2 weights)
    //   - qs: 96 bytes (3-bit weights packed)
    //   Total: 2 + 12 + 16 + 96 = 126 bytes per block
    const q3_k_row_size = num_blocks_k * 126;

    // Q4_K: block size 256.
    //   - d, dmin: 4 bytes (2 * f16)
    //   - scales: 12 bytes (u8)
    //   - qs: 128 bytes (4-bit)
    //   Total: 4 + 12 + 128 = 144 bytes per block
    const q4_k_row_size = num_blocks_k * 144;

    // Q8_K: block size 256.
    //   - d: 4 bytes (f32)
    //   - qs: 256 bytes (i8)
    //   - bsums: 32 bytes (16 x i16)
    //   Total: 4 + 256 + 32 = 292 bytes per block
    const q8_k_row_size = num_blocks_k * 292;

    // Allocate 8 rows for GEMM test
    const q4_size = q4_row_size;
    const q8_size = q8_row_size;

    const q2_k_size = q2_k_row_size;
    const q3_k_size = q3_k_row_size;
    const q4_k_size = q4_k_row_size;
    const q6_k_size = q6_k_row_size;
    const q8_k_size = q8_k_row_size;

    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    // Align to 64 bytes for AVX-512 (Alignment is enum, must be comptime)
    const alignment = comptime std.mem.Alignment.fromByteUnits(64);
    const vx_q4 = try allocator.alignedAlloc(u8, alignment, q4_size);
    const vy_q8 = try allocator.alignedAlloc(u8, alignment, q8_size);
    const vx_q8 = try allocator.alignedAlloc(u8, alignment, q8_size);

    const vx_q2_k = try allocator.alignedAlloc(u8, alignment, q2_k_size);
    const vx_q3_k = try allocator.alignedAlloc(u8, alignment, q3_k_size);
    const vx_q4_k = try allocator.alignedAlloc(u8, alignment, q4_k_size);
    const vx_q6_k = try allocator.alignedAlloc(u8, alignment, q6_k_size);
    const vx_q8_k = try allocator.alignedAlloc(u8, alignment, q8_k_size);
    const vy_q8_k = try allocator.alignedAlloc(u8, alignment, q8_k_size);

    // Initialize with random data
    std.crypto.random.bytes(vx_q4);
    std.crypto.random.bytes(vy_q8);
    std.crypto.random.bytes(vx_q8);
    std.crypto.random.bytes(vx_q2_k);
    std.crypto.random.bytes(vx_q3_k);
    std.crypto.random.bytes(vx_q4_k);
    std.crypto.random.bytes(vx_q6_k);
    std.crypto.random.bytes(vx_q8_k);
    std.crypto.random.bytes(vy_q8_k);

    const iterations = 100_000;
    var res: f32 = 0;

    std.debug.print("Benchmarking N={} Iterations={}\n", .{ N, iterations });
    std.debug.print("--------------------------------------------------\n", .{});
    std.debug.print("{s:<25} | {s:<10} | {s:<15}\n", .{ "Kernel", "Time (s)", "Throughput" });
    std.debug.print("--------------------------------------------------\n", .{});

    // Warmup
    simd_vec_dot_q4_0_q8_0_avx2(N, &res, vx_q4.ptr, vy_q8.ptr);

    // Benchmark Q4 AVX2
    var timer = try std.time.Timer.start();
    for (0..iterations) |_| {
        simd_vec_dot_q4_0_q8_0_avx2(N, &res, vx_q4.ptr, vy_q8.ptr);
    }
    var t_ns = timer.read();
    printStats("Q4_0 x Q8_0 (AVX2)", t_ns, N, iterations); // Single row throughput

    // Benchmark Q4 AVX512 (Scalar)
    timer.reset();
    for (0..iterations) |_| {
        simd_vec_dot_q4_0_q8_0_avx512(N, &res, vx_q4.ptr, vy_q8.ptr);
    }
    t_ns = timer.read();
    printStats("Q4_0 x Q8_0 (AVX-512)", t_ns, N, iterations);

    // Benchmark Q8 AVX2
    timer.reset();
    for (0..iterations) |_| {
        simd_vec_dot_q8_0_q8_0_avx2(N, &res, vx_q8.ptr, vy_q8.ptr);
    }
    t_ns = timer.read();
    printStats("Q8_0 x Q8_0 (AVX2)", t_ns, N, iterations);

    // Benchmark Q8 AVX512
    timer.reset();
    for (0..iterations) |_| {
        simd_vec_dot_q8_0_q8_0_avx512(N, &res, vx_q8.ptr, vy_q8.ptr);
    }
    t_ns = timer.read();
    printStats("Q8_0 x Q8_0 (AVX-512)", t_ns, N, iterations);

    // Benchmark Q2_K AVX2
    timer.reset();
    for (0..iterations) |_| {
        simd_vec_dot_q2_k_q8_k_avx2(N, &res, vx_q2_k.ptr, vy_q8_k.ptr);
    }
    t_ns = timer.read();
    printStats("Q2_K x Q8_K (AVX2)", t_ns, N, iterations);

    // Benchmark Q2_K AVX512
    timer.reset();
    for (0..iterations) |_| {
        simd_vec_dot_q2_k_q8_k_avx512(N, &res, vx_q2_k.ptr, vy_q8_k.ptr);
    }
    t_ns = timer.read();
    printStats("Q2_K x Q8_K (AVX-512)", t_ns, N, iterations);

    // Benchmark Q3_K AVX2
    timer.reset();
    for (0..iterations) |_| {
        simd_vec_dot_q3_k_q8_k_avx2(N, &res, vx_q3_k.ptr, vy_q8_k.ptr);
    }
    t_ns = timer.read();
    printStats("Q3_K x Q8_K (AVX2)", t_ns, N, iterations);

    // Benchmark Q3_K AVX512
    timer.reset();
    for (0..iterations) |_| {
        simd_vec_dot_q3_k_q8_k_avx512(N, &res, vx_q3_k.ptr, vy_q8_k.ptr);
    }
    t_ns = timer.read();
    printStats("Q3_K x Q8_K (AVX-512)", t_ns, N, iterations);

    // Benchmark Q4_K AVX2
    timer.reset();
    for (0..iterations) |_| {
        simd_vec_dot_q4_k_q8_k_avx2(N, &res, vx_q4_k.ptr, vy_q8_k.ptr);
    }
    t_ns = timer.read();
    printStats("Q4_K x Q8_K (AVX2)", t_ns, N, iterations);

    // Benchmark Q4_K AVX512
    timer.reset();
    for (0..iterations) |_| {
        simd_vec_dot_q4_k_q8_k_avx512(N, &res, vx_q4_k.ptr, vy_q8_k.ptr);
    }
    t_ns = timer.read();
    printStats("Q4_K x Q8_K (AVX-512)", t_ns, N, iterations);

    // Benchmark Q6_K AVX2
    timer.reset();
    for (0..iterations) |_| {
        simd_vec_dot_q6_k_q8_k_avx2(N, &res, vx_q6_k.ptr, vy_q8_k.ptr);
    }
    t_ns = timer.read();
    printStats("Q6_K x Q8_K (AVX2)", t_ns, N, iterations);

    // Benchmark Q6_K AVX512
    timer.reset();
    for (0..iterations) |_| {
        simd_vec_dot_q6_k_q8_k_avx512(N, &res, vx_q6_k.ptr, vy_q8_k.ptr);
    }
    t_ns = timer.read();
    printStats("Q6_K x Q8_K (AVX-512)", t_ns, N, iterations);

    // Benchmark Q8_K AVX2
    timer.reset();
    for (0..iterations) |_| {
        simd_vec_dot_q8_k_q8_k_avx2(N, &res, vx_q8_k.ptr, vy_q8_k.ptr);
    }
    t_ns = timer.read();
    printStats("Q8_K x Q8_K (AVX2)", t_ns, N, iterations);

    // Benchmark Q8_K AVX512
    timer.reset();
    for (0..iterations) |_| {
        simd_vec_dot_q8_k_q8_k_avx512(N, &res, vx_q8_k.ptr, vy_q8_k.ptr);
    }
    t_ns = timer.read();
    printStats("Q8_K x Q8_K (AVX-512)", t_ns, N, iterations);
}

fn printStats(name: []const u8, ns: u64, N: usize, iter: usize) void {
    const sec = @as(f64, @floatFromInt(ns)) / 1e9;
    const ops = @as(f64, @floatFromInt(N)) * @as(f64, @floatFromInt(iter)); // elements processed
    // Throughput in elements/sec
    const elems_per_sec = ops / sec;
    const gelem_per_sec = elems_per_sec / 1e9;

    std.debug.print("{s:<25} | {d:<10.4} | {d:<10.2} GElem/s\n", .{ name, sec, gelem_per_sec });
}
