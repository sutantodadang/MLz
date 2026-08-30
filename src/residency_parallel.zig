//! Thin re-export of the parallel bounded-residency matmul driver, which
//! lives in `residency_compute.zig` so it shares one GGML cImport scope.
//! Kept as a separate module for a stable import path (`residency_parallel`).

const std = @import("std");
const compute = @import("residency_compute.zig");

pub const Error = compute.Error;
pub const ParallelOptions = compute.ParallelOptions;
pub const matMul = compute.parallelMatMul;

comptime {
    _ = std;
}
