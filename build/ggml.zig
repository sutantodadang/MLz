const std = @import("std");
const context = @import("context.zig");
const Context = context.Context;

/// Creates the `ggml` static library and populates it with the core ggml
/// sources, the residency backend, and the CPU backend (including its
/// architecture-specific sources). Backend modules (vulkan/cuda/metal/simd)
/// add their own sources to the returned library afterward.
pub fn create(ctx: *Context) *std.Build.Step.Compile {
    const b = ctx.b;

    const ggml_lib = b.addLibrary(.{
        .linkage = .static,
        .name = "ggml",
        .root_module = b.createModule(.{
            .target = ctx.actual_target,
            .optimize = ctx.ggml_optimize,
        }),
    });

    ggml_lib.addCSourceFile(.{
        .file = ctx.llama_cpp_dep.path("ggml/src/ggml.c"),
        .flags = ctx.c_flags.items,
    });
    // ggml-base (C++)
    ggml_lib.addCSourceFile(.{
        .file = ctx.llama_cpp_dep.path("ggml/src/ggml.cpp"),
        .flags = ctx.cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = ctx.llama_cpp_dep.path("ggml/src/ggml-opt.cpp"),
        .flags = ctx.cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = ctx.llama_cpp_dep.path("ggml/src/ggml-quants.c"),
        .flags = ctx.c_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = ctx.llama_cpp_dep.path("ggml/src/ggml-alloc.c"),
        .flags = ctx.c_flags.items,
    });
    // GGUF container helpers (required by llama.cpp model loader)
    ggml_lib.addCSourceFile(.{
        .file = ctx.llama_cpp_dep.path("ggml/src/gguf.cpp"),
        .flags = ctx.cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = ctx.llama_cpp_dep.path("ggml/src/ggml-backend.cpp"),
        .flags = ctx.cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = ctx.llama_cpp_dep.path("ggml/src/ggml-backend-meta.cpp"),
        .flags = ctx.cpp_flags.items,
    });
    // ggml (registry)
    ggml_lib.addCSourceFile(.{
        .file = ctx.llama_cpp_dep.path("ggml/src/ggml-backend-reg.cpp"),
        .flags = ctx.cpp_flags.items,
    });
    // ggml (dynamic loading support)
    ggml_lib.addCSourceFile(.{
        .file = ctx.llama_cpp_dep.path("ggml/src/ggml-backend-dl.cpp"),
        .flags = ctx.cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = ctx.llama_cpp_dep.path("ggml/src/ggml-threading.cpp"),
        .flags = ctx.cpp_flags.items,
    });
    // MLz host-compatible buffer type used by llama tensor_buft_overrides.
    // It is linked into GGML so every executable can opt into native graph
    // execution over the same custom buffer implementation.
    ggml_lib.addCSourceFile(.{
        .file = b.path("src/residency/ggml_residency_backend.c"),
        .flags = ctx.c_flags.items,
    });

    // CPU backend (linked statically into ggml when GGML_BACKEND_DL is off).
    // Residency hooks patch the exact vendored source at build time; no forked
    // copy is kept in the repository.
    const ggml_cpu_c_source = if (ctx.options.use_ggml_residency_hooks) blk: {
        const patcher = b.addExecutable(.{
            .name = "patch-ggml-residency",
            .root_module = b.createModule(.{
                .root_source_file = b.path("src/tools/patch_ggml_residency.zig"),
                .target = b.graph.host,
                .optimize = .ReleaseSafe,
            }),
        });
        const run_patcher = b.addRunArtifact(patcher);
        run_patcher.addFileArg(ctx.llama_cpp_dep.path("ggml/src/ggml-cpu/ggml-cpu.c"));
        break :blk run_patcher.addOutputFileArg("ggml-cpu-residency.c");
    } else if (ctx.options.use_simd_backend and (ctx.actual_target.result.cpu.arch == .x86_64 or ctx.actual_target.result.cpu.arch == .aarch64))
        b.path("src/simd/ggml-cpu-simd.c")
    else
        ctx.llama_cpp_dep.path("ggml/src/ggml-cpu/ggml-cpu.c");

    ggml_lib.addCSourceFile(.{
        .file = ggml_cpu_c_source,
        .flags = ctx.c_flags.items,
    });

    ggml_lib.addCSourceFile(.{
        .file = ctx.llama_cpp_dep.path("ggml/src/ggml-cpu/quants.c"),
        .flags = ctx.c_flags.items,
    });

    // x86-specific optimized quantization kernels (AVX2/AVX-512)
    if (ctx.actual_target.result.cpu.arch == .x86_64) {
        ggml_lib.addCSourceFile(.{
            .file = ctx.llama_cpp_dep.path("ggml/src/ggml-cpu/arch/x86/quants.c"),
            .flags = ctx.c_flags.items,
        });
    }
    ggml_lib.addCSourceFile(.{
        .file = ctx.llama_cpp_dep.path("ggml/src/ggml-cpu/ggml-cpu.cpp"),
        .flags = ctx.cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = ctx.llama_cpp_dep.path("ggml/src/ggml-cpu/repack.cpp"),
        .flags = ctx.cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = ctx.llama_cpp_dep.path("ggml/src/ggml-cpu/hbm.cpp"),
        .flags = ctx.cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = ctx.llama_cpp_dep.path("ggml/src/ggml-cpu/traits.cpp"),
        .flags = ctx.cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = ctx.llama_cpp_dep.path("ggml/src/ggml-cpu/amx/amx.cpp"),
        .flags = ctx.cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = ctx.llama_cpp_dep.path("ggml/src/ggml-cpu/amx/mmq.cpp"),
        .flags = ctx.cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = ctx.llama_cpp_dep.path("ggml/src/ggml-cpu/binary-ops.cpp"),
        .flags = ctx.cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = ctx.llama_cpp_dep.path("ggml/src/ggml-cpu/unary-ops.cpp"),
        .flags = ctx.cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = ctx.llama_cpp_dep.path("ggml/src/ggml-cpu/vec.cpp"),
        .flags = ctx.cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = ctx.llama_cpp_dep.path("ggml/src/ggml-cpu/ops.cpp"),
        .flags = ctx.cpp_flags.items,
    });
    // Architecture-specific CPU backend sources
    switch (ctx.actual_target.result.cpu.arch) {
        .x86_64 => {
            // x86 feature detection + optimized repack/quants
            ggml_lib.addCSourceFile(.{
                .file = ctx.llama_cpp_dep.path("ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp"),
                .flags = ctx.cpp_flags.items,
            });
            ggml_lib.addCSourceFile(.{
                .file = ctx.llama_cpp_dep.path("ggml/src/ggml-cpu/arch/x86/repack.cpp"),
                .flags = ctx.cpp_flags.items,
            });
            ggml_lib.addCSourceFile(.{
                .file = ctx.llama_cpp_dep.path("ggml/src/ggml-cpu/arch/x86/quants.c"),
                .flags = ctx.c_flags.items,
            });
        },
        .aarch64 => {
            // ARM NEON optimizations
            ggml_lib.addCSourceFile(.{
                .file = ctx.llama_cpp_dep.path("ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp"),
                .flags = ctx.cpp_flags.items,
            });
            ggml_lib.addCSourceFile(.{
                .file = ctx.llama_cpp_dep.path("ggml/src/ggml-cpu/arch/arm/repack.cpp"),
                .flags = ctx.cpp_flags.items,
            });
            ggml_lib.addCSourceFile(.{
                .file = ctx.llama_cpp_dep.path("ggml/src/ggml-cpu/arch/arm/quants.c"),
                .flags = ctx.c_flags.items,
            });
        },
        else => {
            // Fallback: no arch-specific optimizations
        },
    }

    return ggml_lib;
}

/// Finalize the `ggml` library after every backend has added its sources:
/// shared include paths and libc/libc++ linkage.
pub fn finalize(ctx: *Context, ggml_lib: *std.Build.Step.Compile) void {
    ggml_lib.addIncludePath(ctx.llama_cpp_dep.path("ggml/include"));
    ggml_lib.addIncludePath(ctx.llama_cpp_dep.path("ggml/src"));
    ggml_lib.addIncludePath(ctx.llama_cpp_dep.path("ggml/src/ggml-cpu"));
    ggml_lib.linkLibC();
    if (ctx.actual_target.query.abi != .msvc) {
        ggml_lib.linkLibCpp();
    }
}
