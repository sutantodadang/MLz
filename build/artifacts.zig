const std = @import("std");
const context = @import("context.zig");
const Context = context.Context;
const cuda = @import("cuda.zig");
const vulkan = @import("vulkan.zig");
const metal = @import("metal.zig");

/// Defines the main executable plus every auxiliary artifact/step: run,
/// bench, validate-ggml-backend, test-simd, and test.
pub fn build(ctx: *Context, ggml_lib: *std.Build.Step.Compile, llama_lib: *std.Build.Step.Compile, mod: *std.Build.Module) void {
    const b = ctx.b;
    const llama_cpp_dep = ctx.llama_cpp_dep;

    const exe = b.addExecutable(.{
        .name = "MLz",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = ctx.actual_target,
            .optimize = ctx.optimize,
            .imports = &.{
                .{ .name = "MLz", .module = mod },
            },
        }),
    });
    exe.root_module.addIncludePath(b.path("src/llama"));
    exe.root_module.addIncludePath(b.path("src/residency"));

    // Link GGML
    exe.linkLibrary(ggml_lib);
    exe.addIncludePath(llama_cpp_dep.path("ggml/include"));
    exe.root_module.addIncludePath(llama_cpp_dep.path("ggml/include"));

    // Link llama.cpp C API
    exe.linkLibrary(llama_lib);
    exe.addIncludePath(llama_cpp_dep.path("include"));
    exe.root_module.addIncludePath(llama_cpp_dep.path("include"));

    exe.linkLibC();

    if (ctx.options.use_cuda) {
        cuda.linkExecutable(ctx, exe);
    }

    // Vulkan: link the system library on the executable only.  Do NOT put
    // linkSystemLibrary on static archives (ggml_lib/llama_lib) because LLD
    // will try to include the .so as an archive member and warn/error.
    // Zig 0.15 also does not propagate addLibraryPath from linkLibrary'd
    // static archives, so we duplicate the search paths here.
    if (ctx.options.use_vulkan) {
        vulkan.linkExecutable(ctx, exe);
    }

    // Link Metal frameworks to executable
    if (ctx.options.use_metal) {
        metal.linkExecutable(ctx, exe);
    }

    b.installArtifact(exe);

    // Install CUDA shared library alongside the executable (Linux only).
    // The .so encapsulates GNU libstdc++ dependencies and is found at
    // runtime via the $ORIGIN RPATH set on the executable.
    cuda.installSharedLib(ctx);

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);

    // Ensure the Metal shader is installed before running
    run_cmd.step.dependOn(b.getInstallStep());

    // Tell llama.cpp where to find the Metal shader file
    if (ctx.options.use_metal) {
        metal.setRunEnv(ctx, run_cmd);
    }

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    run_step.dependOn(&run_cmd.step);

    // Benchmark Step
    const bench_exe = b.addExecutable(.{
        .name = "bench_simd",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tools/bench_simd.zig"),
            .target = ctx.actual_target,
            .optimize = ctx.optimize,
        }),
    });
    bench_exe.linkLibrary(ggml_lib);

    const bench_run = b.addRunArtifact(bench_exe);
    if (b.args) |args| {
        bench_run.addArgs(args);
    }

    const bench_step = b.step("bench", "Run SIMD benchmarks");
    bench_step.dependOn(&bench_run.step);

    // Official GGML buffer-backend integration validator. It runs the native
    // llama.cpp CPU graph twice (ordinary CPU buffers vs MLz's host-compatible
    // tensor_buft_override) and accepts exact logits, or the documented tight
    // numerical/top-1 gate when CPU_REPACK selects a different packed kernel.
    const validate_ggml_backend_module = b.createModule(.{
        .root_source_file = b.path("src/tools/validate_ggml_backend.zig"),
        .target = ctx.actual_target,
        .optimize = ctx.optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "mlz", .module = mod },
        },
    });
    validate_ggml_backend_module.addIncludePath(llama_cpp_dep.path("include"));
    validate_ggml_backend_module.addIncludePath(llama_cpp_dep.path("ggml/include"));
    validate_ggml_backend_module.addIncludePath(llama_cpp_dep.path("ggml/src"));
    const validate_ggml_backend_exe = b.addExecutable(.{
        .name = "validate_ggml_backend",
        .root_module = validate_ggml_backend_module,
    });
    validate_ggml_backend_exe.linkLibrary(ggml_lib);
    validate_ggml_backend_exe.linkLibrary(llama_lib);
    const validate_ggml_backend_run = b.addRunArtifact(validate_ggml_backend_exe);
    if (b.args) |args| validate_ggml_backend_run.addArgs(args);
    const validate_ggml_backend_step = b.step(
        "validate-ggml-backend",
        "Validate native GGML graph execution over the MLz buffer backend",
    );
    validate_ggml_backend_step.dependOn(&validate_ggml_backend_run.step);

    // U1 — Per-kernel correctness validator (PLAN-ASSEMBLY-REWRITE Section 3).
    // Generates random F32 vectors, quantizes via ggml's reference, calls every
    // built kernel, and asserts the result matches a scalar dequantize-then-dot
    // reference within REL_TOL=1e-3.  Exits non-zero on any failure.
    const test_simd_exe = b.addExecutable(.{
        .name = "test_simd",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/tools/test_simd.zig"),
            .target = ctx.actual_target,
            .optimize = ctx.optimize,
        }),
    });
    test_simd_exe.linkLibrary(ggml_lib);
    const test_simd_run = b.addRunArtifact(test_simd_exe);
    if (b.args) |args| {
        test_simd_run.addArgs(args);
    }
    const test_simd_step = b.step("test-simd", "Validate every built SIMD kernel against a scalar reference");
    test_simd_step.dependOn(&test_simd_run.step);

    if (ctx.options.use_cuda) {
        cuda.setRunEnv(ctx, run_cmd);
    }

    const test_step = b.step("test", "Run tests");
    const mod_test_module = b.createModule(.{
        .root_source_file = b.path("src/root.zig"),
        .target = ctx.actual_target,
        .optimize = ctx.optimize,
        .link_libc = true,
    });
    mod_test_module.addIncludePath(b.path("src/llama"));
    mod_test_module.addIncludePath(b.path("src/residency"));
    mod_test_module.addIncludePath(llama_cpp_dep.path("include"));
    mod_test_module.addIncludePath(llama_cpp_dep.path("ggml/include"));
    mod_test_module.addCSourceFile(.{
        .file = b.path("src/residency/residency_mmap.c"),
        .flags = &.{"-std=c11"},
    });
    const mod_tests = b.addTest(.{
        .root_module = mod_test_module,
    });
    mod_tests.linkLibrary(ggml_lib);
    test_step.dependOn(&b.addRunArtifact(mod_tests).step);
}
