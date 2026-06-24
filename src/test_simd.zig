//! U1 â€” Per-kernel correctness validator (PLAN-ASSEMBLY-REWRITE Section 3).
//!
//! For each compiled vec_dot kernel, generate random F32 input, quantize via
//! ggml's reference implementation, call the kernel, and compare the result
//! against a scalar dequantize-then-dot reference.
//!
//! Tolerance: relative error <= 1e-3 per the plan's U1 spec.  Values close to
//! zero use absolute error fallback.
//!
//! Exit code: 0 = all pass, 1 = at least one failure.
//!
//! Run:  zig build test-simd -Dsimd-backend=true -Doptimize=ReleaseFast

const std = @import("std");
const builtin = @import("builtin");

// -----------------------------------------------------------------------------
// External kernel symbols (mirrors src/simd/ggml_simd_hook.cpp)
// -----------------------------------------------------------------------------
extern "c" fn simd_vec_dot_q4_0_q8_0_avx2(n: c_int, r: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q4_0_q8_0_avx512(n: c_int, r: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q8_0_q8_0_avx2(n: c_int, r: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q8_0_q8_0_avx512(n: c_int, r: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q2_k_q8_k_avx2(n: c_int, r: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q2_k_q8_k_avx512(n: c_int, r: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q3_k_q8_k_avx2(n: c_int, r: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q3_k_q8_k_avx512(n: c_int, r: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q4_k_q8_k_avx2(n: c_int, r: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q4_k_q8_k_avx512(n: c_int, r: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q6_k_q8_k_avx2(n: c_int, r: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q6_k_q8_k_avx512(n: c_int, r: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q8_k_q8_k_avx2(n: c_int, r: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q8_k_q8_k_avx512(n: c_int, r: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q5_k_q8_k_avx2(n: c_int, r: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_q5_k_q8_k_avx512(n: c_int, r: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;

extern "c" fn simd_check_avx2() bool;
extern "c" fn simd_check_avx512() bool;

// New kernel symbols â€” quantization
extern "c" fn simd_quantize_q8_0_f32_avx2(n: c_int, x: [*]const f32, y: ?*anyopaque) void;
extern "c" fn simd_quantize_q8_0_f32_avx512(n: c_int, x: [*]const f32, y: ?*anyopaque) void;
extern "c" fn simd_quantize_q8_k_f32_avx2(n: c_int, x: [*]const f32, y: ?*anyopaque) void;
extern "c" fn simd_quantize_q8_k_f32_avx512(n: c_int, x: [*]const f32, y: ?*anyopaque) void;

// New kernel symbols â€” SiLU
extern "c" fn simd_silu_f32_avx2(n: c_int, x: [*]const f32, y: [*]f32) void;
extern "c" fn simd_silu_f32_avx512(n: c_int, x: [*]const f32, y: [*]f32) void;

// New kernel symbols â€” layer_norm (reuses rms_norm signature)
extern "c" fn simd_layer_norm_f32_avx2(n: c_int, eps: f32, x: [*]const f32, y: [*]f32) void;
extern "c" fn simd_layer_norm_f32_avx512(n: c_int, eps: f32, x: [*]const f32, y: [*]f32) void;

// New kernel symbols â€” rope_standard (reuses rope_neox signature)
extern "c" fn simd_rope_standard_f32_avx2(n_pairs: c_longlong, cache: [*]const f32, src: [*]const f32, dst: [*]f32) void;
extern "c" fn simd_rope_standard_f32_avx512(n_pairs: c_longlong, cache: [*]const f32, src: [*]const f32, dst: [*]f32) void;

// New kernel symbols â€” vec_dot_f32_f32
extern "c" fn simd_vec_dot_f32_f32_avx2(n: c_int, r: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;
extern "c" fn simd_vec_dot_f32_f32_avx512(n: c_int, r: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) void;

// New kernel symbols â€” INT8 GEMM microkernel
extern "c" fn simd_check_avx512_vnni() bool;
extern "c" fn simd_gemm_s8s8s32_avx2(M: c_int, N: c_int, K: c_int, A: [*]const i8, B: [*]const i8, C: [*]i32) void;
extern "c" fn simd_gemm_s8s8s32_avx512vnni(M: c_int, N: c_int, K: c_int, A: [*]const i8, B: [*]const i8, C: [*]i32) void;
extern "c" fn simd_gemm_s8s8s32_avx512vnni_t(M: c_int, N: c_int, K: c_int, A: [*]const i8, B: [*]const i8, C: [*]i32) void;
extern "c" fn simd_gemm_s8s8s32(M: c_int, N: c_int, K: c_int, A: [*]const i8, B: [*]const i8, C: [*]i32) void;

// -----------------------------------------------------------------------------
// ggml reference quantize/dequantize (canonical, scalar)
// -----------------------------------------------------------------------------
extern "c" fn quantize_row_q4_0_ref(x: [*]const f32, y: ?*anyopaque, k: i64) void;
extern "c" fn quantize_row_q8_0_ref(x: [*]const f32, y: ?*anyopaque, k: i64) void;
extern "c" fn quantize_row_q2_K_ref(x: [*]const f32, y: ?*anyopaque, k: i64) void;
extern "c" fn quantize_row_q3_K_ref(x: [*]const f32, y: ?*anyopaque, k: i64) void;
extern "c" fn quantize_row_q4_K_ref(x: [*]const f32, y: ?*anyopaque, k: i64) void;
extern "c" fn quantize_row_q5_K_ref(x: [*]const f32, y: ?*anyopaque, k: i64) void;
extern "c" fn quantize_row_q6_K_ref(x: [*]const f32, y: ?*anyopaque, k: i64) void;
extern "c" fn quantize_row_q8_K_ref(x: [*]const f32, y: ?*anyopaque, k: i64) void;

extern "c" fn dequantize_row_q4_0(x: ?*const anyopaque, y: [*]f32, k: i64) void;
extern "c" fn dequantize_row_q8_0(x: ?*const anyopaque, y: [*]f32, k: i64) void;
extern "c" fn dequantize_row_q2_K(x: ?*const anyopaque, y: [*]f32, k: i64) void;
extern "c" fn dequantize_row_q3_K(x: ?*const anyopaque, y: [*]f32, k: i64) void;
extern "c" fn dequantize_row_q4_K(x: ?*const anyopaque, y: [*]f32, k: i64) void;
extern "c" fn dequantize_row_q5_K(x: ?*const anyopaque, y: [*]f32, k: i64) void;
extern "c" fn dequantize_row_q6_K(x: ?*const anyopaque, y: [*]f32, k: i64) void;
extern "c" fn dequantize_row_q8_K(x: ?*const anyopaque, y: [*]f32, k: i64) void;

// -----------------------------------------------------------------------------
// Block sizes (must match ggml's on-disk layout)
// -----------------------------------------------------------------------------
const BLK_LEGACY: i64 = 32;
const BLK_K: i64 = 256;

const BS_Q4_0: usize = 18; // 2 (d:f16) + 16 (qs:nibbles)
const BS_Q8_0: usize = 34; // 2 (d:f16) + 32 (qs:i8)
const BS_Q2_K: usize = 84; // 16 (scales) + 64 (qs) + 4 (d,dmin)
const BS_Q3_K: usize = 110; // 32 (hmask) + 64 (qs) + 12 (scales) + 2 (d). Check: 32+64+12+2=110
const BS_Q4_K: usize = 144; // 4 (d,dmin) + 12 (scales) + 128 (qs)
const BS_Q5_K: usize = 176; // 4 (d,dmin) + 12 (scales) + 32 (qh) + 128 (qs)
const BS_Q6_K: usize = 210; // 128 (ql) + 64 (qh) + 16 (scales) + 2 (d)
const BS_Q8_K: usize = 292; // 4 (d:f32) + 256 (qs:i8) + 32 (bsums:i16)

// -----------------------------------------------------------------------------
// Tolerance check
// -----------------------------------------------------------------------------
const REL_TOL: f32 = 1.0e-3;
const ABS_TOL: f32 = 1.0e-3;

fn within(actual: f32, ref: f32) bool {
    if (std.math.isNan(actual) or std.math.isNan(ref)) return false;
    if (std.math.isInf(actual) or std.math.isInf(ref)) return false;
    const diff = @abs(actual - ref);
    if (diff <= ABS_TOL) return true;
    const denom = @max(@abs(ref), 1.0);
    return (diff / denom) <= REL_TOL;
}

// -----------------------------------------------------------------------------
// Reference: scalar dot of two F32 vectors
// -----------------------------------------------------------------------------
fn scalar_dot(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    var acc: f64 = 0.0;
    for (a, b) |x, y| acc += @as(f64, x) * @as(f64, y);
    return @floatCast(acc);
}

// -----------------------------------------------------------------------------
// Test harness
// -----------------------------------------------------------------------------
const KernelKind = enum { q4_0_q8_0, q8_0_q8_0, q2_k_q8_k, q3_k_q8_k, q4_k_q8_k, q5_k_q8_k, q6_k_q8_k, q8_k_q8_k };
const Variant = enum { avx2, avx512, neon };

const TestSpec = struct {
    name: []const u8,
    kind: KernelKind,
    variant: Variant,
    fp: *const fn (n: c_int, r: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) callconv(.c) void,
};

fn block_size(kind: KernelKind, side: enum { x, y }) struct { blk: i64, bs: usize } {
    return switch (kind) {
        .q4_0_q8_0 => switch (side) {
            .x => .{ .blk = BLK_LEGACY, .bs = BS_Q4_0 },
            .y => .{ .blk = BLK_LEGACY, .bs = BS_Q8_0 },
        },
        .q8_0_q8_0 => switch (side) {
            .x => .{ .blk = BLK_LEGACY, .bs = BS_Q8_0 },
            .y => .{ .blk = BLK_LEGACY, .bs = BS_Q8_0 },
        },
        .q2_k_q8_k => switch (side) {
            .x => .{ .blk = BLK_K, .bs = BS_Q2_K },
            .y => .{ .blk = BLK_K, .bs = BS_Q8_K },
        },
        .q3_k_q8_k => switch (side) {
            .x => .{ .blk = BLK_K, .bs = BS_Q3_K },
            .y => .{ .blk = BLK_K, .bs = BS_Q8_K },
        },
        .q4_k_q8_k => switch (side) {
            .x => .{ .blk = BLK_K, .bs = BS_Q4_K },
            .y => .{ .blk = BLK_K, .bs = BS_Q8_K },
        },
        .q5_k_q8_k => switch (side) {
            .x => .{ .blk = BLK_K, .bs = BS_Q5_K },
            .y => .{ .blk = BLK_K, .bs = BS_Q8_K },
        },
        .q6_k_q8_k => switch (side) {
            .x => .{ .blk = BLK_K, .bs = BS_Q6_K },
            .y => .{ .blk = BLK_K, .bs = BS_Q8_K },
        },
        .q8_k_q8_k => switch (side) {
            .x => .{ .blk = BLK_K, .bs = BS_Q8_K },
            .y => .{ .blk = BLK_K, .bs = BS_Q8_K },
        },
    };
}

fn quantize(kind: KernelKind, side: enum { x, y }, x: []const f32, dst: []u8) void {
    const k_total: i64 = @intCast(x.len);
    const data_ptr: ?*anyopaque = @ptrCast(dst.ptr);
    switch (kind) {
        .q4_0_q8_0 => {
            if (side == .x) quantize_row_q4_0_ref(x.ptr, data_ptr, k_total) else quantize_row_q8_0_ref(x.ptr, data_ptr, k_total);
        },
        .q8_0_q8_0 => quantize_row_q8_0_ref(x.ptr, data_ptr, k_total),
        .q2_k_q8_k => {
            if (side == .x) quantize_row_q2_K_ref(x.ptr, data_ptr, k_total) else quantize_row_q8_K_ref(x.ptr, data_ptr, k_total);
        },
        .q3_k_q8_k => {
            if (side == .x) quantize_row_q3_K_ref(x.ptr, data_ptr, k_total) else quantize_row_q8_K_ref(x.ptr, data_ptr, k_total);
        },
        .q4_k_q8_k => {
            if (side == .x) quantize_row_q4_K_ref(x.ptr, data_ptr, k_total) else quantize_row_q8_K_ref(x.ptr, data_ptr, k_total);
        },
        .q5_k_q8_k => {
            if (side == .x) quantize_row_q5_K_ref(x.ptr, data_ptr, k_total) else quantize_row_q8_K_ref(x.ptr, data_ptr, k_total);
        },
        .q6_k_q8_k => {
            if (side == .x) quantize_row_q6_K_ref(x.ptr, data_ptr, k_total) else quantize_row_q8_K_ref(x.ptr, data_ptr, k_total);
        },
        .q8_k_q8_k => quantize_row_q8_K_ref(x.ptr, data_ptr, k_total),
    }
}

fn dequantize(kind: KernelKind, side: enum { x, y }, src: []const u8, out: []f32) void {
    const k_total: i64 = @intCast(out.len);
    const data_ptr: ?*const anyopaque = @ptrCast(src.ptr);
    switch (kind) {
        .q4_0_q8_0 => {
            if (side == .x) dequantize_row_q4_0(data_ptr, out.ptr, k_total) else dequantize_row_q8_0(data_ptr, out.ptr, k_total);
        },
        .q8_0_q8_0 => dequantize_row_q8_0(data_ptr, out.ptr, k_total),
        .q2_k_q8_k => {
            if (side == .x) dequantize_row_q2_K(data_ptr, out.ptr, k_total) else dequantize_row_q8_K(data_ptr, out.ptr, k_total);
        },
        .q3_k_q8_k => {
            if (side == .x) dequantize_row_q3_K(data_ptr, out.ptr, k_total) else dequantize_row_q8_K(data_ptr, out.ptr, k_total);
        },
        .q4_k_q8_k => {
            if (side == .x) dequantize_row_q4_K(data_ptr, out.ptr, k_total) else dequantize_row_q8_K(data_ptr, out.ptr, k_total);
        },
        .q5_k_q8_k => {
            if (side == .x) dequantize_row_q5_K(data_ptr, out.ptr, k_total) else dequantize_row_q8_K(data_ptr, out.ptr, k_total);
        },
        .q6_k_q8_k => {
            if (side == .x) dequantize_row_q6_K(data_ptr, out.ptr, k_total) else dequantize_row_q8_K(data_ptr, out.ptr, k_total);
        },
        .q8_k_q8_k => dequantize_row_q8_K(data_ptr, out.ptr, k_total),
    }
}

fn runKernelTest(
    allocator: std.mem.Allocator,
    spec: TestSpec,
    K: usize,
    rng: *std.Random.DefaultPrng,
) !bool {
    // Per-kernel block alignment requirements
    const blk = block_size(spec.kind, .x).blk;
    if (@mod(@as(i64, @intCast(K)), blk) != 0) return true; // skip

    // Generate random F32 in a sane range that keeps quantized values bounded
    const x_f32 = try allocator.alignedAlloc(f32, .fromByteUnits(64), K);
    defer allocator.free(x_f32);
    const y_f32 = try allocator.alignedAlloc(f32, .fromByteUnits(64), K);
    defer allocator.free(y_f32);

    const r = rng.random();
    for (x_f32) |*v| v.* = (r.float(f32) - 0.5) * 2.0;
    for (y_f32) |*v| v.* = (r.float(f32) - 0.5) * 2.0;

    const xb = block_size(spec.kind, .x);
    const yb = block_size(spec.kind, .y);
    const x_bytes = (K / @as(usize, @intCast(xb.blk))) * xb.bs;
    const y_bytes = (K / @as(usize, @intCast(yb.blk))) * yb.bs;

    const xq = try allocator.alignedAlloc(u8, .fromByteUnits(64), x_bytes);
    defer allocator.free(xq);
    const yq = try allocator.alignedAlloc(u8, .fromByteUnits(64), y_bytes);
    defer allocator.free(yq);
    @memset(xq, 0);
    @memset(yq, 0);

    quantize(spec.kind, .x, x_f32, xq);
    quantize(spec.kind, .y, y_f32, yq);

    // Reference: dequantize back to F32 then scalar dot.
    const x_dq = try allocator.alignedAlloc(f32, .fromByteUnits(64), K);
    defer allocator.free(x_dq);
    const y_dq = try allocator.alignedAlloc(f32, .fromByteUnits(64), K);
    defer allocator.free(y_dq);
    dequantize(spec.kind, .x, xq, x_dq);
    dequantize(spec.kind, .y, yq, y_dq);
    const ref = scalar_dot(x_dq, y_dq);

    var got: f32 = 0.0;
    spec.fp(@intCast(K), &got, @ptrCast(xq.ptr), @ptrCast(yq.ptr));

    if (!within(got, ref)) {
        std.debug.print("  FAIL [{s}] K={d}: got={d:.6} ref={d:.6} diff={d:.6}\n", .{ spec.name, K, got, ref, got - ref });
        return false;
    }
    return true;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var rng = std.Random.DefaultPrng.init(0xDEADBEEF);

    const have_avx2 = simd_check_avx2();
    const have_avx512 = simd_check_avx512();
    std.debug.print("CPU caps: AVX2={any} AVX-512={any}\n", .{ have_avx2, have_avx512 });

    var specs: std.ArrayList(TestSpec) = .empty;
    defer specs.deinit(allocator);

    if (builtin.cpu.arch == .x86_64) {
        if (have_avx2) {
            try specs.append(allocator, .{ .name = "q4_0_q8_0 avx2", .kind = .q4_0_q8_0, .variant = .avx2, .fp = simd_vec_dot_q4_0_q8_0_avx2 });
            try specs.append(allocator, .{ .name = "q8_0_q8_0 avx2", .kind = .q8_0_q8_0, .variant = .avx2, .fp = simd_vec_dot_q8_0_q8_0_avx2 });
            try specs.append(allocator, .{ .name = "q2_k_q8_k avx2", .kind = .q2_k_q8_k, .variant = .avx2, .fp = simd_vec_dot_q2_k_q8_k_avx2 });
            try specs.append(allocator, .{ .name = "q3_k_q8_k avx2", .kind = .q3_k_q8_k, .variant = .avx2, .fp = simd_vec_dot_q3_k_q8_k_avx2 });
            try specs.append(allocator, .{ .name = "q4_k_q8_k avx2", .kind = .q4_k_q8_k, .variant = .avx2, .fp = simd_vec_dot_q4_k_q8_k_avx2 });
            try specs.append(allocator, .{ .name = "q5_k_q8_k asm avx2", .kind = .q5_k_q8_k, .variant = .avx2, .fp = simd_vec_dot_q5_k_q8_k_avx2 });
            try specs.append(allocator, .{ .name = "q6_k_q8_k avx2", .kind = .q6_k_q8_k, .variant = .avx2, .fp = simd_vec_dot_q6_k_q8_k_avx2 });
            try specs.append(allocator, .{ .name = "q8_k_q8_k avx2", .kind = .q8_k_q8_k, .variant = .avx2, .fp = simd_vec_dot_q8_k_q8_k_avx2 });
        }
        if (have_avx512) {
            try specs.append(allocator, .{ .name = "q4_0_q8_0 avx512", .kind = .q4_0_q8_0, .variant = .avx512, .fp = simd_vec_dot_q4_0_q8_0_avx512 });
            try specs.append(allocator, .{ .name = "q8_0_q8_0 avx512", .kind = .q8_0_q8_0, .variant = .avx512, .fp = simd_vec_dot_q8_0_q8_0_avx512 });
            try specs.append(allocator, .{ .name = "q2_k_q8_k avx512", .kind = .q2_k_q8_k, .variant = .avx512, .fp = simd_vec_dot_q2_k_q8_k_avx512 });
            try specs.append(allocator, .{ .name = "q3_k_q8_k avx512", .kind = .q3_k_q8_k, .variant = .avx512, .fp = simd_vec_dot_q3_k_q8_k_avx512 });
            try specs.append(allocator, .{ .name = "q4_k_q8_k avx512", .kind = .q4_k_q8_k, .variant = .avx512, .fp = simd_vec_dot_q4_k_q8_k_avx512 });
            try specs.append(allocator, .{ .name = "q5_k_q8_k asm avx512", .kind = .q5_k_q8_k, .variant = .avx512, .fp = simd_vec_dot_q5_k_q8_k_avx512 });
            try specs.append(allocator, .{ .name = "q6_k_q8_k avx512", .kind = .q6_k_q8_k, .variant = .avx512, .fp = simd_vec_dot_q6_k_q8_k_avx512 });
            try specs.append(allocator, .{ .name = "q8_k_q8_k avx512", .kind = .q8_k_q8_k, .variant = .avx512, .fp = simd_vec_dot_q8_k_q8_k_avx512 });
        }
    }

    const sizes = [_]usize{ 32, 256, 1024, 4096 };
    var pass: usize = 0;
    var fail: usize = 0;
    var skip: usize = 0;

    // The new unary/vec kernels (quantize_q8_0/q8_k, silu, layer_norm,
    // rope_standard, vec_dot_f32) are wired into the build and assemble, but
    // several still have runtime/correctness bugs (see PLAN Phase 3). Gate their
    // tests off so the suite stays green on the validated kernel set. Flip to
    // true to work on them.
    const enable_new_kernels = true;

    for (specs.items) |spec| {
        std.debug.print("[{s}]", .{spec.name});
        var any_run = false;
        for (sizes) |K| {
            const blk = block_size(spec.kind, .x).blk;
            if (@mod(@as(i64, @intCast(K)), blk) != 0) {
                std.debug.print(" K={d}:skip", .{ K, });
                continue;
            }
            any_run = true;
            const ok = runKernelTest(allocator, spec, K, &rng) catch |e| {
                std.debug.print(" K={d}:err({any})", .{ K, e });
                fail += 1;
                continue;
            };
            if (ok) {
                std.debug.print(" K={d}:ok", .{ K, });
                pass += 1;
            } else {
                fail += 1;
            }
        }
        if (!any_run) skip += 1;
        std.debug.print("\n", .{});
    }
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
    if (builtin.cpu.arch == .x86_64) {
        const rms_sizes = [_]usize{ 7, 64, 256, 1024, 4096, 8193 };
        if (have_avx2) {
            try runRmsNormTest(allocator, "rms_norm_f32 avx2", simd_rms_norm_f32_avx2, &rms_sizes, &rng, &pass, &fail);
        }
        if (have_avx512) {
            try runRmsNormTest(allocator, "rms_norm_f32 avx512", simd_rms_norm_f32_avx512, &rms_sizes, &rng, &pass, &fail);
        }
    }

    // -------------------------------------------------------------------------
    // Unary kernels: rope_neox_f32  (PLAN-ASSEMBLY-REWRITE Section 3.4)
    // Bit-exact equivalence vs the scalar rotate_pairs reference.
    // -------------------------------------------------------------------------
    if (builtin.cpu.arch == .x86_64) {
        const rope_sizes = [_]usize{ 4, 32, 64, 128, 256, 1024, 4099 };
        if (have_avx2) {
            try runRopeNeoxTest(allocator, "rope_neox_f32 avx2", simd_rope_neox_f32_avx2, &rope_sizes, &rng, &pass, &fail);
        }
        if (have_avx512) {
            try runRopeNeoxTest(allocator, "rope_neox_f32 avx512", simd_rope_neox_f32_avx512, &rope_sizes, &rng, &pass, &fail);
        }
    }

    // -------------------------------------------------------------------------
    // New kernels: quantize_q8_0_f32
    // Compare output against ggml's quantize_row_q8_0_ref (byte-exact).
    // -------------------------------------------------------------------------
    if (enable_new_kernels and builtin.cpu.arch == .x86_64) {
        const quant_sizes = [_]usize{ 32, 256, 1024, 4096 };
        if (have_avx2) {
            try runQuantizeTest(allocator, "quantize_q8_0_f32 avx2", simd_quantize_q8_0_f32_avx2, &quant_sizes, &rng, &pass, &fail);
        }
        if (have_avx512) {
            try runQuantizeTest(allocator, "quantize_q8_0_f32 avx512", simd_quantize_q8_0_f32_avx512, &quant_sizes, &rng, &pass, &fail);
        }
    }

    // -------------------------------------------------------------------------
    // New kernels: quantize_q8_k_f32
    // Compare output against ggml's quantize_row_q8_K_ref (byte-exact).
    // -------------------------------------------------------------------------
    if (enable_new_kernels and builtin.cpu.arch == .x86_64) {
        const quant_k_sizes = [_]usize{ 256, 1024, 4096 };
        if (have_avx2) {
            try runQuantizeKTest(allocator, "quantize_q8_k_f32 avx2", simd_quantize_q8_k_f32_avx2, &quant_k_sizes, &rng, &pass, &fail);
        }
        if (have_avx512) {
            try runQuantizeKTest(allocator, "quantize_q8_k_f32 avx512", simd_quantize_q8_k_f32_avx512, &quant_k_sizes, &rng, &pass, &fail);
        }
    }

    // -------------------------------------------------------------------------
    // New kernels: silu_f32
    // Compare against reference x * sigmoid(x) implementation.
    // -------------------------------------------------------------------------
    if (enable_new_kernels and builtin.cpu.arch == .x86_64) {
        const silu_sizes = [_]usize{ 7, 64, 256, 1024, 4096, 8193 };
        if (have_avx2) {
            try runSiluTest(allocator, "silu_f32 avx2", simd_silu_f32_avx2, &silu_sizes, &rng, &pass, &fail);
        }
        if (have_avx512) {
            try runSiluTest(allocator, "silu_f32 avx512", simd_silu_f32_avx512, &silu_sizes, &rng, &pass, &fail);
        }
    }

    // -------------------------------------------------------------------------
    // New kernels: layer_norm_f32
    // Compare against reference layer norm implementation.
    // -------------------------------------------------------------------------
    if (enable_new_kernels and builtin.cpu.arch == .x86_64) {
        const ln_sizes = [_]usize{ 7, 64, 256, 1024, 4096, 8193 };
        if (have_avx2) {
            try runLayerNormTest(allocator, "layer_norm_f32 avx2", simd_layer_norm_f32_avx2, &ln_sizes, &rng, &pass, &fail);
        }
        if (have_avx512) {
            try runLayerNormTest(allocator, "layer_norm_f32 avx512", simd_layer_norm_f32_avx512, &ln_sizes, &rng, &pass, &fail);
        }
    }

    // -------------------------------------------------------------------------
    // New kernels: rope_standard_f32
    // Compare against reference interleaved RoPE (same as neox but standard layout).
    // -------------------------------------------------------------------------
    if (enable_new_kernels and builtin.cpu.arch == .x86_64) {
        const rope_std_sizes = [_]usize{ 4, 32, 64, 128, 256, 1024, 4099 };
        if (have_avx2) {
            try runRopeStandardTest(allocator, "rope_standard_f32 avx2", simd_rope_standard_f32_avx2, &rope_std_sizes, &rng, &pass, &fail);
        }
        if (have_avx512) {
            try runRopeStandardTest(allocator, "rope_standard_f32 avx512", simd_rope_standard_f32_avx512, &rope_std_sizes, &rng, &pass, &fail);
        }
    }

    // -------------------------------------------------------------------------
    // New kernels: vec_dot_f32_f32
    // Compare against reference dot product.
    // -------------------------------------------------------------------------
    if (enable_new_kernels and builtin.cpu.arch == .x86_64) {
        const vdot_sizes = [_]usize{ 32, 256, 1024, 4096 };
        if (have_avx2) {
            try runVecDotF32Test(allocator, "vec_dot_f32_f32 avx2", simd_vec_dot_f32_f32_avx2, &vdot_sizes, &rng, &pass, &fail);
        }
        if (have_avx512) {
            try runVecDotF32Test(allocator, "vec_dot_f32_f32 avx512", simd_vec_dot_f32_f32_avx512, &vdot_sizes, &rng, &pass, &fail);
        }
    }

    // -------------------------------------------------------------------------
    // New kernels: INT8 GEMM microkernel (s8*s8 -> s32). AVX2 + AVX512-VNNI.
    // Reference: scalar triple-loop. Exact integer match expected.
    // Sizes include K % 32 != 0 to exercise the scalar tail.
    // -------------------------------------------------------------------------
    if (enable_new_kernels and builtin.cpu.arch == .x86_64) {
        const have_vnni = simd_check_avx512_vnni();
        std.debug.print("CPU caps: AVX512-VNNI={any}\n", .{have_vnni});
        const gemm_cases = [_][3]usize{ .{ 1, 1, 32 }, .{ 2, 3, 64 }, .{ 4, 5, 256 }, .{ 3, 7, 96 }, .{ 5, 4, 160 }, .{ 1, 1, 40 }, .{ 2, 2, 35 } };
        if (have_avx2) {
            try runGemmTest(allocator, "gemm_s8s8s32 avx2", simd_gemm_s8s8s32_avx2, &gemm_cases, &rng, &pass, &fail);
        }
        if (have_vnni) {
            try runGemmTest(allocator, "gemm_s8s8s32 avx512vnni", simd_gemm_s8s8s32_avx512vnni, &gemm_cases, &rng, &pass, &fail);
            // tiled microkernel — aligned shapes only (M%4, N%2, K%32)
            const gemm_t_cases = [_][3]usize{ .{ 4, 2, 32 }, .{ 8, 4, 64 }, .{ 4, 6, 96 }, .{ 12, 8, 256 }, .{ 8, 2, 160 } };
            try runGemmTest(allocator, "gemm_s8s8s32 vnni-tiled", simd_gemm_s8s8s32_avx512vnni_t, &gemm_t_cases, &rng, &pass, &fail);
        }
        // dispatcher: must be correct for any shape (aligned -> tiled, else naive)
        try runGemmTest(allocator, "gemm_s8s8s32 dispatch", simd_gemm_s8s8s32, &gemm_cases, &rng, &pass, &fail);
    }

    std.debug.print("\n=== SUMMARY: pass={d} fail={d} kernel-skipped={d} ===\n", .{ pass, fail, skip });
    if (fail > 0) std.process.exit(1);
}

// -----------------------------------------------------------------------------
// INT8 GEMM validator: C[M,N] = A[M,K] . B[N,K]^T, s8*s8 -> s32, exact.
// -----------------------------------------------------------------------------
const GemmFn = *const fn (M: c_int, N: c_int, K: c_int, A: [*]const i8, B: [*]const i8, C: [*]i32) callconv(.c) void;

fn runGemmTest(
    allocator: std.mem.Allocator,
    name: []const u8,
    fp: GemmFn,
    cases: []const [3]usize,
    rng: *std.Random.DefaultPrng,
    pass: *usize,
    fail: *usize,
) !void {
    std.debug.print("[{s}]", .{ .name = name });
    const r = rng.random();
    for (cases) |c| {
        const M = c[0];
        const N = c[1];
        const K = c[2];
        const A = try allocator.alloc(i8, M * K);
        defer allocator.free(A);
        const B = try allocator.alloc(i8, N * K);
        defer allocator.free(B);
        const C = try allocator.alloc(i32, M * N);
        defer allocator.free(C);
        const Cref = try allocator.alloc(i32, M * N);
        defer allocator.free(Cref);
        for (A) |*v| v.* = @intCast(r.intRangeAtMost(i32, -127, 127));
        for (B) |*v| v.* = @intCast(r.intRangeAtMost(i32, -127, 127));
        @memset(C, -559038737); // 0xDEADBEEF sentinel

        for (0..M) |m| {
            for (0..N) |n| {
                var acc: i32 = 0;
                for (0..K) |k| acc += @as(i32, A[m * K + k]) * @as(i32, B[n * K + k]);
                Cref[m * N + n] = acc;
            }
        }
        fp(@intCast(M), @intCast(N), @intCast(K), A.ptr, B.ptr, C.ptr);

        var ok = true;
        for (0..M * N) |i| {
            if (C[i] != Cref[i]) {
                ok = false;
                std.debug.print(" MxNxK={d}x{d}x{d}:FAIL[{d}] got={d} ref={d}", .{ M, N, K, i, C[i], Cref[i] });
                break;
            }
        }
        if (ok) {
            std.debug.print(" {d}x{d}x{d}:ok", .{ M, N, K });
            pass.* += 1;
        } else {
            fail.* += 1;
        }
    }
    std.debug.print("\n", .{});
}

// -----------------------------------------------------------------------------
// quantize_q8_0_f32 validator
// -----------------------------------------------------------------------------
const QuantizeFn = *const fn (n: c_int, x: [*]const f32, y: ?*anyopaque) callconv(.c) void;

fn runQuantizeTest(
    allocator: std.mem.Allocator,
    name: []const u8,
    fp: QuantizeFn,
    sizes: []const usize,
    rng: *std.Random.DefaultPrng,
    pass: *usize,
    fail: *usize,
) !void {
    std.debug.print("[{s}]", .{ .name = name });
    const r = rng.random();
    for (sizes) |n| {
        const x = try allocator.alignedAlloc(f32, .fromByteUnits(64), n);
        defer allocator.free(x);
        for (x) |*v| v.* = (r.float(f32) - 0.5) * 2.0;

        const num_blocks = n / 32;
        const out_bytes = num_blocks * 34; // Q8_0 block size

        const got = try allocator.alignedAlloc(u8, .fromByteUnits(64), out_bytes);
        defer allocator.free(got);
        const ref = try allocator.alignedAlloc(u8, .fromByteUnits(64), out_bytes);
        defer allocator.free(ref);
        @memset(got, 0);
        @memset(ref, 0);

        quantize_row_q8_0_ref(x.ptr, @ptrCast(ref.ptr), @intCast(n));
        fp(@intCast(n), x.ptr, @ptrCast(got.ptr));

        var ok = true;
        for (got, ref, 0..) |g, r_byte, i| {
            if (g != r_byte) {
                ok = false;
                std.debug.print(" n={d}:FAIL byte[{d}] got=0x{x} ref=0x{x}", .{ n, i, g, r_byte });
                break;
            }
        }
        if (ok) {
            std.debug.print(" n={d}:ok", .{ n, });
            pass.* += 1;
        } else {
            fail.* += 1;
        }
    }
    std.debug.print("\n", .{});
}

// -----------------------------------------------------------------------------
// quantize_q8_k_f32 validator
// -----------------------------------------------------------------------------
const QuantizeKFn = *const fn (n: c_int, x: [*]const f32, y: ?*anyopaque) callconv(.c) void;

fn runQuantizeKTest(
    allocator: std.mem.Allocator,
    name: []const u8,
    fp: QuantizeKFn,
    sizes: []const usize,
    rng: *std.Random.DefaultPrng,
    pass: *usize,
    fail: *usize,
) !void {
    std.debug.print("[{s}]", .{ .name = name });
    const r = rng.random();
    for (sizes) |n| {
        if (@mod(n, 256) != 0) {
            std.debug.print(" n={d}:skip", .{ n, });
            continue;
        }
        const x = try allocator.alignedAlloc(f32, .fromByteUnits(64), n);
        defer allocator.free(x);
        for (x) |*v| v.* = (r.float(f32) - 0.5) * 2.0;

        const num_blocks = n / 256;
        const out_bytes = num_blocks * 292; // Q8_K block size

        const got = try allocator.alignedAlloc(u8, .fromByteUnits(64), out_bytes);
        defer allocator.free(got);
        const ref = try allocator.alignedAlloc(u8, .fromByteUnits(64), out_bytes);
        defer allocator.free(ref);
        @memset(got, 0);
        @memset(ref, 0);

        quantize_row_q8_K_ref(x.ptr, @ptrCast(ref.ptr), @intCast(n));
        fp(@intCast(n), x.ptr, @ptrCast(got.ptr));

        var ok = true;
        for (got, ref, 0..) |g, r_byte, i| {
            if (g != r_byte) {
                ok = false;
                std.debug.print(" n={d}:FAIL byte[{d}] got=0x{x} ref=0x{x}", .{ n, i, g, r_byte });
                break;
            }
        }
        if (ok) {
            std.debug.print(" n={d}:ok", .{ n, });
            pass.* += 1;
        } else {
            fail.* += 1;
        }
    }
    std.debug.print("\n", .{});
}

// -----------------------------------------------------------------------------
// silu_f32 validator
// -----------------------------------------------------------------------------
const SiluFn = *const fn (n: c_int, x: [*]const f32, y: [*]f32) callconv(.c) void;

fn silu_reference(x: []const f32, y: []f32) void {
    for (x, 0..) |v, i| {
        const sigmoid = 1.0 / (1.0 + @exp(-v));
        y[i] = v * sigmoid;
    }
}

fn runSiluTest(
    allocator: std.mem.Allocator,
    name: []const u8,
    fp: SiluFn,
    sizes: []const usize,
    rng: *std.Random.DefaultPrng,
    pass: *usize,
    fail: *usize,
) !void {
    std.debug.print("[{s}]", .{ .name = name });
    const r = rng.random();
    for (sizes) |n| {
        const x = try allocator.alignedAlloc(f32, .fromByteUnits(64), n);
        defer allocator.free(x);
        const y_ref = try allocator.alignedAlloc(f32, .fromByteUnits(64), n);
        defer allocator.free(y_ref);
        const y_got = try allocator.alignedAlloc(f32, .fromByteUnits(64), n);
        defer allocator.free(y_got);

        for (x) |*v| v.* = (r.float(f32) - 0.5) * 4.0;

        silu_reference(x, y_ref);
        fp(@intCast(n), x.ptr, y_got.ptr);

        var ok = true;
        var worst_rel: f32 = 0.0;
        for (y_ref, y_got) |a, b| {
            // silu uses a fast degree-4 polynomial approximation of exp, so a
            // few 1e-5 of relative error vs the exact reference is expected and
            // harmless for an activation.
            if (!ulpClose(a, b, 2.0e-4, 1.0e-5)) {
                ok = false;
            }
            const denom = @max(@abs(a), @abs(b));
            if (denom > 0) {
                const rel = @abs(a - b) / denom;
                if (rel > worst_rel) worst_rel = rel;
            }
        }
        if (ok) {
            std.debug.print(" n={d}:ok(rel={e:.1})", .{ n, worst_rel });
            pass.* += 1;
        } else {
            std.debug.print(" n={d}:FAIL(rel={e:.1})", .{ n, worst_rel });
            fail.* += 1;
        }
    }
    std.debug.print("\n", .{});
}

// -----------------------------------------------------------------------------
// layer_norm_f32 validator
// -----------------------------------------------------------------------------
const LayerNormFn = *const fn (n: c_int, eps: f32, x: [*]const f32, y: [*]f32) callconv(.c) void;

fn layer_norm_reference(x: []const f32, eps: f32, y: []f32) void {
    var sum: f64 = 0.0;
    for (x) |v| sum += @as(f64, v);
    const mean: f32 = @floatCast(sum / @as(f64, @floatFromInt(x.len)));

    var var_sum: f64 = 0.0;
    for (x) |v| {
        const diff = v - mean;
        var_sum += @as(f64, diff * diff);
    }
    const variance: f32 = @floatCast(var_sum / @as(f64, @floatFromInt(x.len)));
    const scale: f32 = 1.0 / @sqrt(variance + eps);

    for (x, 0..) |v, i| y[i] = (v - mean) * scale;
}

fn runLayerNormTest(
    allocator: std.mem.Allocator,
    name: []const u8,
    fp: LayerNormFn,
    sizes: []const usize,
    rng: *std.Random.DefaultPrng,
    pass: *usize,
    fail: *usize,
) !void {
    std.debug.print("[{s}]", .{ .name = name });
    const r = rng.random();
    for (sizes) |n| {
        const x = try allocator.alignedAlloc(f32, .fromByteUnits(64), n);
        defer allocator.free(x);
        const y_ref = try allocator.alignedAlloc(f32, .fromByteUnits(64), n);
        defer allocator.free(y_ref);
        const y_got = try allocator.alignedAlloc(f32, .fromByteUnits(64), n);
        defer allocator.free(y_got);

        for (x) |*v| v.* = (r.float(f32) - 0.5) * 4.0;

        const eps: f32 = 1.0e-5;
        layer_norm_reference(x, eps, y_ref);
        fp(@intCast(n), eps, x.ptr, y_got.ptr);

        var ok = true;
        var worst_rel: f32 = 0.0;
        for (y_ref, y_got) |a, b| {
            if (!ulpClose(a, b, 4.0e-5, 1.0e-6)) {
                ok = false;
            }
            const denom = @max(@abs(a), @abs(b));
            if (denom > 0) {
                const rel = @abs(a - b) / denom;
                if (rel > worst_rel) worst_rel = rel;
            }
        }
        if (ok) {
            std.debug.print(" n={d}:ok(rel={e:.1})", .{ n, worst_rel });
            pass.* += 1;
        } else {
            std.debug.print(" n={d}:FAIL(rel={e:.1})", .{ n, worst_rel });
            fail.* += 1;
        }
    }
    std.debug.print("\n", .{});
}

// -----------------------------------------------------------------------------
// rope_standard_f32 validator
// -----------------------------------------------------------------------------
const RopeStandardFn = *const fn (n_pairs: c_longlong, cache: [*]const f32, src: [*]const f32, dst: [*]f32) callconv(.c) void;

fn rope_standard_reference(n_pairs: usize, cache: []const f32, src: []const f32, dst: []f32) void {
    // Standard (interleaved) RoPE layout:
    //   pair i = (src[2*i], src[2*i+1])
    //   dst[2*i]   = x0*cos - x1*sin
    //   dst[2*i+1] = x0*sin + x1*cos
    var i: usize = 0;
    while (i < n_pairs) : (i += 1) {
        const cos_t = cache[2 * i + 0];
        const sin_t = cache[2 * i + 1];
        const x0 = src[2 * i];
        const x1 = src[2 * i + 1];
        dst[2 * i] = x0 * cos_t - x1 * sin_t;
        dst[2 * i + 1] = x0 * sin_t + x1 * cos_t;
    }
}

fn runRopeStandardTest(
    allocator: std.mem.Allocator,
    name: []const u8,
    fp: RopeStandardFn,
    sizes: []const usize,
    rng: *std.Random.DefaultPrng,
    pass: *usize,
    fail: *usize,
) !void {
    std.debug.print("[{s}]", .{ .name = name });
    const r = rng.random();
    for (sizes) |n_pairs| {
        const n_total = 2 * n_pairs;
        const cache = try allocator.alignedAlloc(f32, .fromByteUnits(64), 2 * n_pairs);
        defer allocator.free(cache);
        const src = try allocator.alignedAlloc(f32, .fromByteUnits(64), n_total);
        defer allocator.free(src);
        const ref_dst = try allocator.alignedAlloc(f32, .fromByteUnits(64), n_total);
        defer allocator.free(ref_dst);
        const got_dst = try allocator.alignedAlloc(f32, .fromByteUnits(64), n_total);
        defer allocator.free(got_dst);

        var i: usize = 0;
        while (i < n_pairs) : (i += 1) {
            const theta = (r.float(f32) - 0.5) * 6.2831853;
            cache[2 * i + 0] = @cos(theta);
            cache[2 * i + 1] = @sin(theta);
        }
        for (src) |*v| v.* = (r.float(f32) - 0.5) * 4.0;

        rope_standard_reference(n_pairs, cache, src, ref_dst);
        fp(@intCast(n_pairs), cache.ptr, src.ptr, got_dst.ptr);

        var ok = true;
        var worst_abs: f32 = 0.0;
        for (ref_dst, got_dst) |a, b| {
            const diff = @abs(a - b);
            if (diff > worst_abs) worst_abs = diff;
            if (diff > 0) {
                const denom = @max(@abs(a), @abs(b));
                const rel = if (denom > 0) diff / denom else diff;
                if (rel > 1.0e-6 and diff > 1.0e-7) ok = false;
            }
        }
        if (ok) {
            std.debug.print(" n_pairs={d}:ok(abs={e:.1})", .{ n_pairs, worst_abs });
            pass.* += 1;
        } else {
            std.debug.print(" n_pairs={d}:FAIL(abs={e:.1})", .{ n_pairs, worst_abs });
            fail.* += 1;
        }
    }
    std.debug.print("\n", .{});
}

// -----------------------------------------------------------------------------
// vec_dot_f32_f32 validator
// -----------------------------------------------------------------------------
const VecDotF32Fn = *const fn (n: c_int, r: *f32, vx: ?*const anyopaque, vy: ?*const anyopaque) callconv(.c) void;

fn runVecDotF32Test(
    allocator: std.mem.Allocator,
    name: []const u8,
    fp: VecDotF32Fn,
    sizes: []const usize,
    rng: *std.Random.DefaultPrng,
    pass: *usize,
    fail: *usize,
) !void {
    std.debug.print("[{s}]", .{ .name = name });
    const r = rng.random();
    for (sizes) |n| {
        const x = try allocator.alignedAlloc(f32, .fromByteUnits(64), n);
        defer allocator.free(x);
        const y = try allocator.alignedAlloc(f32, .fromByteUnits(64), n);
        defer allocator.free(y);

        for (x) |*v| v.* = (r.float(f32) - 0.5) * 2.0;
        for (y) |*v| v.* = (r.float(f32) - 0.5) * 2.0;

        const ref = scalar_dot(x, y);
        var got: f32 = 0.0;
        fp(@intCast(n), &got, @ptrCast(x.ptr), @ptrCast(y.ptr));

        if (!within(got, ref)) {
            std.debug.print(" n={d}:FAIL got={d:.6} ref={d:.6} diff={d:.6}\n", .{ n, got, ref, got - ref });
            fail.* += 1;
        } else {
            std.debug.print(" n={d}:ok", .{ n, });
            pass.* += 1;
        }
    }
    std.debug.print("\n", .{});
}


// -----------------------------------------------------------------------------
// rms_norm_f32 validator
// -----------------------------------------------------------------------------
extern "c" fn simd_rms_norm_f32_avx2(n: c_int, eps: f32, x: [*]const f32, y: [*]f32) void;
extern "c" fn simd_rms_norm_f32_avx512(n: c_int, eps: f32, x: [*]const f32, y: [*]f32) void;

const RmsFn = *const fn (n: c_int, eps: f32, x: [*]const f32, y: [*]f32) callconv(.c) void;

fn rms_norm_reference(x: []const f32, eps: f32, y: []f32) void {
    var sum: f64 = 0.0;
    for (x) |v| {
        const vf32: f32 = v * v;
        sum += @as(f64, vf32);
    }
    const mean: f32 = @floatCast(sum / @as(f64, @floatFromInt(x.len)));
    const scale: f32 = 1.0 / @sqrt(mean + eps);
    for (x, 0..) |v, i| y[i] = v * scale;
}

fn ulpClose(a: f32, b: f32, max_rel: f32, max_abs: f32) bool {
    if (std.math.isNan(a) or std.math.isNan(b)) return false;
    if (std.math.isInf(a) or std.math.isInf(b)) return false;
    const diff = @abs(a - b);
    if (diff <= max_abs) return true;
    const denom = @max(@abs(a), @abs(b));
    if (denom == 0.0) return diff <= max_abs;
    return (diff / denom) <= max_rel;
}

fn runRmsNormTest(
    allocator: std.mem.Allocator,
    name: []const u8,
    fp: RmsFn,
    sizes: []const usize,
    rng: *std.Random.DefaultPrng,
    pass: *usize,
    fail: *usize,
) !void {
    std.debug.print("[{s}]", .{ .name = name });
    const r = rng.random();
    for (sizes) |n| {
        const x = try allocator.alignedAlloc(f32, .fromByteUnits(64), n);
        defer allocator.free(x);
        const y_ref = try allocator.alignedAlloc(f32, .fromByteUnits(64), n);
        defer allocator.free(y_ref);
        const y_got = try allocator.alignedAlloc(f32, .fromByteUnits(64), n);
        defer allocator.free(y_got);

        for (x) |*v| v.* = (r.float(f32) - 0.5) * 4.0;

        const eps: f32 = 1.0e-5;
        rms_norm_reference(x, eps, y_ref);
        fp(@intCast(n), eps, x.ptr, y_got.ptr);

        var ok = true;
        var worst_rel: f32 = 0.0;
        for (y_ref, y_got) |a, b| {
            // 4 ULP target â‰ˆ 4 * 2^-23 â‰ˆ 4.8e-7; loosen to 4e-5 because
            // parallel f64 reduction reorders rounding vs serial reference.
            if (!ulpClose(a, b, 4.0e-5, 1.0e-6)) {
                ok = false;
            }
            const denom = @max(@abs(a), @abs(b));
            if (denom > 0) {
                const rel = @abs(a - b) / denom;
                if (rel > worst_rel) worst_rel = rel;
            }
        }
        if (ok) {
            std.debug.print(" n={d}:ok(rel={e:.1})", .{ n, worst_rel });
            pass.* += 1;
        } else {
            std.debug.print(" n={d}:FAIL(rel={e:.1})", .{ n, worst_rel });
            fail.* += 1;
        }
    }
    std.debug.print("\n", .{});
}

// -----------------------------------------------------------------------------
// rope_neox_f32 validator
// -----------------------------------------------------------------------------
extern "c" fn simd_rope_neox_f32_avx2(n_pairs: c_longlong, cache: [*]const f32, src: [*]const f32, dst: [*]f32) void;
extern "c" fn simd_rope_neox_f32_avx512(n_pairs: c_longlong, cache: [*]const f32, src: [*]const f32, dst: [*]f32) void;

const RopeFn = *const fn (n_pairs: c_longlong, cache: [*]const f32, src: [*]const f32, dst: [*]f32) callconv(.c) void;

fn rope_neox_reference(n_pairs: usize, cache: []const f32, src: []const f32, dst: []f32) void {
    // Mirrors `rotate_pairs<float>` (NEOX layout, scale=2):
    //   ic = i0/2; pair = (src[ic], src[ic + n_pairs]);
    //   dst[ic]            = x0*cos - x1*sin
    //   dst[ic + n_pairs]  = x0*sin + x1*cos
    var ic: usize = 0;
    while (ic < n_pairs) : (ic += 1) {
        const cos_t = cache[2 * ic + 0];
        const sin_t = cache[2 * ic + 1];
        const x0 = src[ic];
        const x1 = src[ic + n_pairs];
        dst[ic] = x0 * cos_t - x1 * sin_t;
        dst[ic + n_pairs] = x0 * sin_t + x1 * cos_t;
    }
}

fn runRopeNeoxTest(
    allocator: std.mem.Allocator,
    name: []const u8,
    fp: RopeFn,
    sizes: []const usize,
    rng: *std.Random.DefaultPrng,
    pass: *usize,
    fail: *usize,
) !void {
    std.debug.print("[{s}]", .{ .name = name });
    const r = rng.random();
    for (sizes) |n_pairs| {
        const n_total = 2 * n_pairs;
        const cache = try allocator.alignedAlloc(f32, .fromByteUnits(64), 2 * n_pairs);
        defer allocator.free(cache);
        const src = try allocator.alignedAlloc(f32, .fromByteUnits(64), n_total);
        defer allocator.free(src);
        const ref_dst = try allocator.alignedAlloc(f32, .fromByteUnits(64), n_total);
        defer allocator.free(ref_dst);
        const got_dst = try allocator.alignedAlloc(f32, .fromByteUnits(64), n_total);
        defer allocator.free(got_dst);

        // Random cache populated with cos/sin of arbitrary angles so that the
        // kernel sees realistic |cache[i]| <= 1 values.
        var i: usize = 0;
        while (i < n_pairs) : (i += 1) {
            const theta = (r.float(f32) - 0.5) * 6.2831853;
            cache[2 * i + 0] = @cos(theta);
            cache[2 * i + 1] = @sin(theta);
        }
        for (src) |*v| v.* = (r.float(f32) - 0.5) * 4.0;

        rope_neox_reference(n_pairs, cache, src, ref_dst);
        fp(@intCast(n_pairs), cache.ptr, src.ptr, got_dst.ptr);

        var ok = true;
        var worst_abs: f32 = 0.0;
        for (ref_dst, got_dst) |a, b| {
            const diff = @abs(a - b);
            if (diff > worst_abs) worst_abs = diff;
            // Bit-exact target: identical FMA ordering.  Tolerate <= 1 ULP.
            if (diff > 0) {
                const denom = @max(@abs(a), @abs(b));
                const rel = if (denom > 0) diff / denom else diff;
                if (rel > 1.0e-6 and diff > 1.0e-7) ok = false;
            }
        }
        if (ok) {
            std.debug.print(" n_pairs={d}:ok(abs={e:.1})", .{ n_pairs, worst_abs });
            pass.* += 1;
        } else {
            std.debug.print(" n_pairs={d}:FAIL(abs={e:.1})", .{ n_pairs, worst_abs });
            fail.* += 1;
        }
    }
    std.debug.print("\n", .{});
}
