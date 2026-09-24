const std = @import("std");

/// Every `b.option` value used across the build. Declared once in `init`
/// and threaded through every build/*.zig module via `Context.options`.
pub const Options = struct {
    use_cuda: bool,
    use_vulkan: bool,
    use_metal: bool,
    no_avx512: bool,
    use_cpu_repack: bool,
    use_simd_backend: bool,
    use_ggml_residency_hooks: bool,
};

/// Values shared across the build script's modules: the build graph, the
/// resolved targets, optimize modes, the llama.cpp dependency, the
/// accumulated C/C++ compile flags, and derived option state.
pub const Context = struct {
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    actual_target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    ggml_optimize: std.builtin.OptimizeMode,
    llama_cpp_dep: *std.Build.Dependency,
    c_flags: std.ArrayList([]const u8),
    cpp_flags: std.ArrayList([]const u8),
    cuda_so_output: ?std.Build.LazyPath,
    options: Options,
};

pub fn init(b: *std.Build) Context {
    // Standard target options allow the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const use_cuda = b.option(bool, "cuda", "Use CUDA for GPU acceleration") orelse false;
    const target = b.standardTargetOptions(.{});
    var actual_target = target;
    if (use_cuda and target.result.os.tag == .windows) {
        actual_target.query.abi = .msvc;
        // Don't enforce x86_64_v3 globally as it might cause illegal instruction errors on older CPUs
        // actual_target.query.cpu_model = .{ .explicit = &std.Target.x86.cpu.x86_64_v3 };
    }
    const optimize = b.standardOptimizeOption(.{});

    const use_vulkan = b.option(bool, "vulkan", "Use Vulkan for GPU acceleration") orelse false;

    const use_metal_default = target.result.os.tag == .macos or target.result.os.tag == .ios;
    const use_metal = b.option(bool, "metal", "Use Metal for GPU acceleration (macOS/iOS)") orelse use_metal_default;

    // Handle macOS SDK root detection
    if (actual_target.result.os.tag == .macos or actual_target.result.os.tag == .ios) {
        if (b.sysroot == null) {
            if (b.graph.env_map.get("SDKROOT")) |sdk_root| {
                b.sysroot = sdk_root;
            } else {
                // Try to detect via xcrun on macOS hosts
                const argv = &[_][]const u8{ "xcrun", "--show-sdk-path" };
                if (std.process.Child.run(.{ .allocator = b.allocator, .argv = argv })) |res| {
                    defer {
                        b.allocator.free(res.stdout);
                        b.allocator.free(res.stderr);
                    }
                    if (res.term == .Exited and res.term.Exited == 0) {
                        const trimmed = std.mem.trim(u8, res.stdout, " \n\r");
                        if (trimmed.len > 0) {
                            b.sysroot = b.allocator.dupe(u8, trimmed) catch @panic("OOM");
                        }
                    }
                } else |_| {}
            }
        }
    }

    // ggml's C sources intentionally do pointer arithmetic on null pointers
    // (e.g. for size calculations). In Debug this can trap under Zig/clang's
    // runtime checks/sanitizers, so compile ggml optimized even when the rest
    // of the project is Debug.
    const ggml_optimize: std.builtin.OptimizeMode = switch (optimize) {
        .Debug => .ReleaseFast,
        else => optimize,
    };
    // It's also possible to define more custom flags to toggle optional features
    // of this build script using `b.option()`. All defined flags (including
    // target and optimize options) will be listed when running `zig build --help`
    // in this directory.

    const llama_cpp_dep = b.dependency("llama_cpp", .{});

    var c_flags: std.ArrayList([]const u8) = .empty;
    var cpp_flags: std.ArrayList([]const u8) = .empty;

    c_flags.append(b.allocator, "-std=c11") catch @panic("OOM");
    c_flags.append(b.allocator, "-D_CRT_SECURE_NO_WARNINGS") catch @panic("OOM");
    c_flags.append(b.allocator, "-DGGML_VERSION=\"100\"") catch @panic("OOM");
    c_flags.append(b.allocator, "-DGGML_COMMIT=\"unknown\"") catch @panic("OOM");

    cpp_flags.append(b.allocator, "-std=c++17") catch @panic("OOM");
    cpp_flags.append(b.allocator, "-D_CRT_SECURE_NO_WARNINGS") catch @panic("OOM");
    cpp_flags.append(b.allocator, "-DGGML_VERSION=\"100\"") catch @panic("OOM");
    cpp_flags.append(b.allocator, "-DGGML_COMMIT=\"unknown\"") catch @panic("OOM");

    // x86_64-specific flags
    // AVX512 is enabled by default for better performance on supported CPUs
    // Use -Dno-avx512=true to disable for compatibility with older CPUs
    const no_avx512 = b.option(bool, "no-avx512", "Disable AVX512 for compatibility with older CPUs") orelse false;
    if (actual_target.result.cpu.arch == .x86_64 and no_avx512) {
        c_flags.append(b.allocator, "-mno-avx512f") catch @panic("OOM");
        cpp_flags.append(b.allocator, "-mno-avx512f") catch @panic("OOM");
    }

    // Zig uses Clang in MSVC-compat mode which defines _MSC_VER. Upstream
    // ggml.c skips #include <immintrin.h> under _MSC_VER, assuming real MSVC
    // auto-provides intrinsics. Clang doesn't, so AVX512BF16 intrinsics like
    // _mm512_cvtne2ps_pbh are undeclared. Undefine the feature macro to use
    // the scalar fallback — the SIMD backend handles the hot paths anyway.
    if (actual_target.result.cpu.arch == .x86_64 and actual_target.result.os.tag == .windows) {
        c_flags.append(b.allocator, "-U__AVX512BF16__") catch @panic("OOM");
    }

    c_flags.append(b.allocator, "-DGGML_USE_CPU") catch @panic("OOM");
    cpp_flags.append(b.allocator, "-DGGML_USE_CPU") catch @panic("OOM");

    // CPU Repack accelerator: repacks weight matrices into SIMD-friendly
    // interleaved layouts at load time (q4_K_8x8, q5_K_8x8, q6_K_8x8, ...).
    // Enables specialized GEMM/GEMV dispatch via tensor->extra. Roughly 2x
    // tokens/sec improvement on local CPU inference. On by default; disable
    // with -Dcpu-repack=false if a target lacks support.
    const use_cpu_repack = b.option(bool, "cpu-repack", "Enable llama.cpp CPU repack accelerator (~2x perf)") orelse true;
    const use_simd_backend = b.option(bool, "simd-backend", "Use custom SIMD backend for F32 matrix multiplication (x86_64 and aarch64)") orelse false;
    const use_ggml_residency_hooks = b.option(bool, "ggml-residency-hooks", "Enable synchronized MLz hooks around each GGML CPU node") orelse false;
    if (use_simd_backend and use_ggml_residency_hooks) {
        @panic("-Dsimd-backend=true and -Dggml-residency-hooks=true are mutually exclusive");
    }
    if (use_ggml_residency_hooks) {
        c_flags.append(b.allocator, "-DGGML_USE_MLZ_RESIDENCY_HOOKS") catch @panic("OOM");
    }
    if (use_cpu_repack) {
        c_flags.append(b.allocator, "-DGGML_USE_CPU_REPACK") catch @panic("OOM");
        cpp_flags.append(b.allocator, "-DGGML_USE_CPU_REPACK") catch @panic("OOM");
    }

    if (actual_target.result.os.tag == .linux) {
        c_flags.append(b.allocator, "-D_GNU_SOURCE") catch @panic("OOM");
        cpp_flags.append(b.allocator, "-D_GNU_SOURCE") catch @panic("OOM");
    }

    // On Linux CUDA builds, compileCudaSources returns a LazyPath to
    // libggml-cuda.so that must be installed alongside the executable.
    // Declared here so it's accessible across the cuda setup and install scopes.
    const cuda_so_output: ?std.Build.LazyPath = null;

    if (use_vulkan) {
        c_flags.append(b.allocator, "-DGGML_USE_VULKAN") catch @panic("OOM");
        cpp_flags.append(b.allocator, "-DGGML_USE_VULKAN") catch @panic("OOM");
    } else if (use_cuda) {
        c_flags.append(b.allocator, "-DGGML_USE_CUDA") catch @panic("OOM");
        cpp_flags.append(b.allocator, "-DGGML_USE_CUDA") catch @panic("OOM");
    }

    if (use_metal) {
        c_flags.append(b.allocator, "-DGGML_USE_METAL") catch @panic("OOM");
        cpp_flags.append(b.allocator, "-DGGML_USE_METAL") catch @panic("OOM");
    }

    return .{
        .b = b,
        .target = target,
        .actual_target = actual_target,
        .optimize = optimize,
        .ggml_optimize = ggml_optimize,
        .llama_cpp_dep = llama_cpp_dep,
        .c_flags = c_flags,
        .cpp_flags = cpp_flags,
        .cuda_so_output = cuda_so_output,
        .options = .{
            .use_cuda = use_cuda,
            .use_vulkan = use_vulkan,
            .use_metal = use_metal,
            .no_avx512 = no_avx512,
            .use_cpu_repack = use_cpu_repack,
            .use_simd_backend = use_simd_backend,
            .use_ggml_residency_hooks = use_ggml_residency_hooks,
        },
    };
}
