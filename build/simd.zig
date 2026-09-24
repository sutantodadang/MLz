const std = @import("std");
const context = @import("context.zig");
const Context = context.Context;

/// Adds the custom SIMD backend (hand-optimized AVX2/AVX-512 assembly on
/// x86_64, NEON assembly on aarch64) to the `ggml` static library. No-op
/// unless `-Dsimd-backend=true` and the target architecture is supported.
pub fn addToGgml(ctx: *Context, ggml_lib: *std.Build.Step.Compile) void {
    // Custom SIMD backend for high-performance matrix multiplication
    // Uses hand-optimized AVX2/AVX-512 assembly (x86_64) or NEON assembly (aarch64)
    // Note: ctx.options.use_simd_backend is defined earlier when patching ggml-cpu.c
    if (ctx.options.use_simd_backend and (ctx.actual_target.result.cpu.arch == .x86_64 or ctx.actual_target.result.cpu.arch == .aarch64)) {
        ctx.c_flags.append(ctx.b.allocator, "-DGGML_USE_SIMD_BACKEND") catch @panic("OOM");
        ctx.cpp_flags.append(ctx.b.allocator, "-DGGML_USE_SIMD_BACKEND") catch @panic("OOM");

        // SIMD backend C++ sources
        var simd_cpp_flags: std.ArrayList([]const u8) = .empty;
        simd_cpp_flags.append(ctx.b.allocator, "-std=c++17") catch @panic("OOM");
        simd_cpp_flags.append(ctx.b.allocator, "-D_CRT_SECURE_NO_WARNINGS") catch @panic("OOM");

        // x86_64-specific C++ compiler flags (AVX2/FMA/F16C/AVX-512)
        if (ctx.actual_target.result.cpu.arch == .x86_64) {
            simd_cpp_flags.append(ctx.b.allocator, "-mavx2") catch @panic("OOM");
            simd_cpp_flags.append(ctx.b.allocator, "-mfma") catch @panic("OOM");
            simd_cpp_flags.append(ctx.b.allocator, "-mf16c") catch @panic("OOM");
            if (!ctx.options.no_avx512) {
                simd_cpp_flags.append(ctx.b.allocator, "-mavx512f") catch @panic("OOM");
            }
        }

        // Add SIMD backend C++ sources
        ggml_lib.addCSourceFile(.{
            .file = ctx.b.path("src/simd/simd_matmul.cpp"),
            .flags = simd_cpp_flags.items,
        });
        ggml_lib.addCSourceFile(.{
            .file = ctx.b.path("src/simd/ggml_simd_hook.cpp"),
            .flags = simd_cpp_flags.items,
        });
        ggml_lib.addCSourceFile(.{
            .file = ctx.b.path("src/simd/flash_attention.cpp"),
            .flags = simd_cpp_flags.items,
        });
        ggml_lib.addCSourceFile(.{
            .file = ctx.b.path("src/simd/fused_rope_attn.cpp"),
            .flags = simd_cpp_flags.items,
        });

        // Handwritten C++ intrinsic kernels (PLAN-ASSEMBLY-REWRITE step 8).
        // Q5_K x Q8_K vec_dot is implemented as handwritten NASM:
        //   src/simd/kernels/x86/vec/vec_dot_q5_k_q8_k_avx2.asm
        //   src/simd/kernels/x86/vec/vec_dot_q5_k_q8_k_avx512.asm
        // and as AArch64 NEON .S:
        //   src/simd/kernels/aarch64/vec/vec_dot_q5_k_q8_k_neon.S

        // Add include path for SIMD headers
        ggml_lib.addIncludePath(ctx.b.path("src/simd"));

        // ----------------------------------------------------------------
        // Manifest codegen (PLAN-ASSEMBLY-REWRITE Section 2)
        //
        // Generates `simd_kernels_manifest.h` containing the canonical
        // `extern "C"` declarations for every SIMD kernel in the project.
        // The hook (`src/simd/ggml_simd_hook.cpp`) includes this header
        // instead of hand-maintaining the prototype list.
        //
        // The manifest below is the SINGLE source of truth and is mirrored
        // by `src/simd/kernels/manifest.txt` for documentation.  When you
        // add a new kernel, append a row here AND to manifest.txt.
        // ----------------------------------------------------------------
        const manifest_h = generateSimdManifestHeader(ctx.b.allocator, ctx.options.no_avx512) catch @panic("OOM");
        const manifest_wf = ctx.b.addWriteFiles();
        const manifest_h_path = manifest_wf.add("simd_kernels_manifest.h", manifest_h);
        ggml_lib.addIncludePath(manifest_h_path.dirname());

        // Architecture-specific assembly compilation
        if (ctx.actual_target.result.cpu.arch == .x86_64) {
            // Compile NASM assembly sources
            // Note: Zig's build system can compile .asm files using system NASM
            const nasm_format = switch (ctx.actual_target.result.os.tag) {
                .windows => "win64",
                .macos => "macho64",
                else => "elf64",
            };

            // AVX2 assembly
            const avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-o",
            });
            const avx2_obj = avx2_asm.addOutputFileArg("matrix_mult_avx2.o");
            avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/matrix_mult_avx2.asm"));
            ggml_lib.addObjectFile(avx2_obj);

            // AVX512 assembly (only if not disabled)
            if (!ctx.options.no_avx512) {
                const avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-o",
                });
                const avx512_obj = avx512_asm.addOutputFileArg("matrix_mult_avx512.o");
                avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/matrix_mult_avx512.asm"));
                ggml_lib.addObjectFile(avx512_obj);

                // AVX-512 Quantized Kernels
                const q4_q8_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const q4_q8_avx512_obj = q4_q8_avx512_asm.addOutputFileArg("vec_dot_q4_0_q8_0_avx512.o");
                q4_q8_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/vec/vec_dot_q4_0_q8_0_avx512.asm"));
                ggml_lib.addObjectFile(q4_q8_avx512_obj);

                const q8_q8_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const q8_q8_avx512_obj = q8_q8_avx512_asm.addOutputFileArg("vec_dot_q8_0_q8_0_avx512.o");
                q8_q8_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/vec/vec_dot_q8_0_q8_0_avx512.asm"));
                ggml_lib.addObjectFile(q8_q8_avx512_obj);

                // AVX-512 K-Quant Kernels
                const q2_q8_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const q2_q8_avx512_obj = q2_q8_avx512_asm.addOutputFileArg("vec_dot_q2_k_q8_k_avx512.o");
                q2_q8_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/vec/vec_dot_q2_k_q8_k_avx512.asm"));
                ggml_lib.addObjectFile(q2_q8_avx512_obj);

                const q6_q8_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const q6_q8_avx512_obj = q6_q8_avx512_asm.addOutputFileArg("vec_dot_q6_k_q8_k_avx512.o");
                q6_q8_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/vec/vec_dot_q6_k_q8_k_avx512.asm"));
                ggml_lib.addObjectFile(q6_q8_avx512_obj);

                const q4_k_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const q4_k_avx512_obj = q4_k_avx512_asm.addOutputFileArg("vec_dot_q4_k_q8_k_avx512.o");
                q4_k_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/vec/vec_dot_q4_k_q8_k_avx512.asm"));
                ggml_lib.addObjectFile(q4_k_avx512_obj);

                const q8_k_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const q8_k_avx512_obj = q8_k_avx512_asm.addOutputFileArg("vec_dot_q8_k_q8_k_avx512.o");
                q8_k_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/vec/vec_dot_q8_k_q8_k_avx512.asm"));
                ggml_lib.addObjectFile(q8_k_avx512_obj);

                const q3_k_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const q3_k_avx512_obj = q3_k_avx512_asm.addOutputFileArg("vec_dot_q3_k_q8_k_avx512.o");
                q3_k_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/vec/vec_dot_q3_k_q8_k_avx512.asm"));
                ggml_lib.addObjectFile(q3_k_avx512_obj);

                // Flash Attention F32 - AVX-512
                const fa_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const fa_avx512_obj = fa_avx512_asm.addOutputFileArg("flash_attn_f32_avx512.o");
                fa_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_f32_avx512.asm"));
                ggml_lib.addObjectFile(fa_avx512_obj);

                // Flash Attention Q4_0 - AVX-512
                const fa_q4_0_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_q4_0_avx512_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
                fa_q4_0_avx512_asm.addArg("-o");
                const fa_q4_0_avx512_obj = fa_q4_0_avx512_asm.addOutputFileArg("flash_attn_q4_0_avx512.o");
                fa_q4_0_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_q4_0_avx512.asm"));
                ggml_lib.addObjectFile(fa_q4_0_avx512_obj);

                // Flash Attention Q8_0 - AVX-512
                const fa_q8_0_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_q8_0_avx512_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
                fa_q8_0_avx512_asm.addArg("-o");
                const fa_q8_0_avx512_obj = fa_q8_0_avx512_asm.addOutputFileArg("flash_attn_q8_0_avx512.o");
                fa_q8_0_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_q8_0_avx512.asm"));
                ggml_lib.addObjectFile(fa_q8_0_avx512_obj);

                // Flash Attention F16 - AVX-512
                const fa_f16_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_f16_avx512_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
                fa_f16_avx512_asm.addArg("-o");
                const fa_f16_avx512_obj = fa_f16_avx512_asm.addOutputFileArg("flash_attn_f16_avx512.o");
                fa_f16_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_f16_avx512.asm"));
                ggml_lib.addObjectFile(fa_f16_avx512_obj);

                // Flash Attention Q4_1 - AVX-512
                const fa_q4_1_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_q4_1_avx512_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
                fa_q4_1_avx512_asm.addArg("-o");
                const fa_q4_1_avx512_obj = fa_q4_1_avx512_asm.addOutputFileArg("flash_attn_q4_1_avx512.o");
                fa_q4_1_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_q4_1_avx512.asm"));
                ggml_lib.addObjectFile(fa_q4_1_avx512_obj);

                // Flash Attention Q5_0 - AVX-512
                const fa_q5_0_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_q5_0_avx512_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
                fa_q5_0_avx512_asm.addArg("-o");
                const fa_q5_0_avx512_obj = fa_q5_0_avx512_asm.addOutputFileArg("flash_attn_q5_0_avx512.o");
                fa_q5_0_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_q5_0_avx512.asm"));
                ggml_lib.addObjectFile(fa_q5_0_avx512_obj);

                // Flash Attention Q5_1 - AVX-512
                const fa_q5_1_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_q5_1_avx512_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
                fa_q5_1_avx512_asm.addArg("-o");
                const fa_q5_1_avx512_obj = fa_q5_1_avx512_asm.addOutputFileArg("flash_attn_q5_1_avx512.o");
                fa_q5_1_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_q5_1_avx512.asm"));
                ggml_lib.addObjectFile(fa_q5_1_avx512_obj);

                // Flash Attention IQ4_NL - AVX-512
                const fa_iq4_nl_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_iq4_nl_avx512_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
                fa_iq4_nl_avx512_asm.addArg("-o");
                const fa_iq4_nl_avx512_obj = fa_iq4_nl_avx512_asm.addOutputFileArg("flash_attn_iq4_nl_avx512.o");
                fa_iq4_nl_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_iq4_nl_avx512.asm"));
                ggml_lib.addObjectFile(fa_iq4_nl_avx512_obj);

                // Flash Attention Q2_K - AVX-512
                const fa_q2_k_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_q2_k_avx512_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
                fa_q2_k_avx512_asm.addArg("-o");
                const fa_q2_k_avx512_obj = fa_q2_k_avx512_asm.addOutputFileArg("flash_attn_q2_k_avx512.o");
                fa_q2_k_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_q2_k_avx512.asm"));
                ggml_lib.addObjectFile(fa_q2_k_avx512_obj);

                // Flash Attention Q3_K - AVX-512
                const fa_q3_k_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_q3_k_avx512_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
                fa_q3_k_avx512_asm.addArg("-o");
                const fa_q3_k_avx512_obj = fa_q3_k_avx512_asm.addOutputFileArg("flash_attn_q3_k_avx512.o");
                fa_q3_k_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_q3_k_avx512.asm"));
                ggml_lib.addObjectFile(fa_q3_k_avx512_obj);

                // Flash Attention Q4_K - AVX-512
                const fa_q4_k_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_q4_k_avx512_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
                fa_q4_k_avx512_asm.addArg("-o");
                const fa_q4_k_avx512_obj = fa_q4_k_avx512_asm.addOutputFileArg("flash_attn_q4_k_avx512.o");
                fa_q4_k_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_q4_k_avx512.asm"));
                ggml_lib.addObjectFile(fa_q4_k_avx512_obj);

                // Flash Attention Q5_K - AVX-512
                const fa_q5_k_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_q5_k_avx512_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
                fa_q5_k_avx512_asm.addArg("-o");
                const fa_q5_k_avx512_obj = fa_q5_k_avx512_asm.addOutputFileArg("flash_attn_q5_k_avx512.o");
                fa_q5_k_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_q5_k_avx512.asm"));
                ggml_lib.addObjectFile(fa_q5_k_avx512_obj);

                // Flash Attention Q6_K - AVX-512
                const fa_q6_k_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_q6_k_avx512_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
                fa_q6_k_avx512_asm.addArg("-o");
                const fa_q6_k_avx512_obj = fa_q6_k_avx512_asm.addOutputFileArg("flash_attn_q6_k_avx512.o");
                fa_q6_k_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_q6_k_avx512.asm"));
                ggml_lib.addObjectFile(fa_q6_k_avx512_obj);

                // Flash Attention Q8_K - AVX-512
                const fa_q8_k_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_q8_k_avx512_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
                fa_q8_k_avx512_asm.addArg("-o");
                const fa_q8_k_avx512_obj = fa_q8_k_avx512_asm.addOutputFileArg("flash_attn_q8_k_avx512.o");
                fa_q8_k_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_q8_k_avx512.asm"));
                ggml_lib.addObjectFile(fa_q8_k_avx512_obj);
            }

            // Quantized dot product kernels (Q4_0, Q8_0) - AVX2
            const q4_q8_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS", // Define WINDOWS for calling convention
                "-o",
            });
            const q4_q8_obj = q4_q8_asm.addOutputFileArg("vec_dot_q4_0_q8_0_avx2.o");
            q4_q8_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/vec/vec_dot_q4_0_q8_0_avx2.asm"));
            ggml_lib.addObjectFile(q4_q8_obj);

            const q8_q8_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS", // Define WINDOWS for calling convention
                "-o",
            });
            const q8_q8_obj = q8_q8_asm.addOutputFileArg("vec_dot_q8_0_q8_0_avx2.o");
            q8_q8_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/vec/vec_dot_q8_0_q8_0_avx2.asm"));
            ggml_lib.addObjectFile(q8_q8_obj);

            // Quantized dot product kernels (Q2_K, Q6_K) - AVX2
            const q2_q8_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
                "-o",
            });
            const q2_q8_obj = q2_q8_asm.addOutputFileArg("vec_dot_q2_k_q8_k_avx2.o");
            q2_q8_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/vec/vec_dot_q2_k_q8_k_avx2.asm"));
            ggml_lib.addObjectFile(q2_q8_obj);

            const q6_q8_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
                "-o",
            });
            const q6_q8_obj = q6_q8_asm.addOutputFileArg("vec_dot_q6_k_q8_k_avx2.o");
            q6_q8_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/vec/vec_dot_q6_k_q8_k_avx2.asm"));
            ggml_lib.addObjectFile(q6_q8_obj);

            // Quantized dot product kernels (Q4_K, Q8_K) - AVX2
            const q4_k_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
                "-o",
            });
            const q4_k_obj = q4_k_asm.addOutputFileArg("vec_dot_q4_k_q8_k_avx2.o");
            q4_k_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/vec/vec_dot_q4_k_q8_k_avx2.asm"));
            ggml_lib.addObjectFile(q4_k_obj);

            // Q5_K x Q8_K vec_dot — handwritten NASM (AVX2 + AVX-512).
            const q5_k_avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
                "-o",
            });
            const q5_k_avx2_obj = q5_k_avx2_asm.addOutputFileArg("vec_dot_q5_k_q8_k_avx2.o");
            q5_k_avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/vec/vec_dot_q5_k_q8_k_avx2.asm"));
            ggml_lib.addObjectFile(q5_k_avx2_obj);

            if (!ctx.options.no_avx512) {
                const q5_k_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const q5_k_avx512_obj = q5_k_avx512_asm.addOutputFileArg("vec_dot_q5_k_q8_k_avx512.o");
                q5_k_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/vec/vec_dot_q5_k_q8_k_avx512.asm"));
                ggml_lib.addObjectFile(q5_k_avx512_obj);
            }

            // ------------------------------------------------------------
            // Unary ops (PLAN-ASSEMBLY-REWRITE Section 0D, opt-in via
            // MLZ_SIMD_RMS_NORM=1 at runtime).  Decl in manifest.txt.
            // ------------------------------------------------------------
            const rms_avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
                "-o",
            });
            const rms_avx2_obj = rms_avx2_asm.addOutputFileArg("rms_norm_f32_avx2.o");
            rms_avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/unary/rms_norm_f32_avx2.asm"));
            ggml_lib.addObjectFile(rms_avx2_obj);

            if (!ctx.options.no_avx512) {
                const rms_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const rms_avx512_obj = rms_avx512_asm.addOutputFileArg("rms_norm_f32_avx512.o");
                rms_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/unary/rms_norm_f32_avx512.asm"));
                ggml_lib.addObjectFile(rms_avx512_obj);
            }

            // ------------------------------------------------------------
            // RoPE NEOX f32 (PLAN-ASSEMBLY-REWRITE Section 0E, opt-in via
            // MLZ_SIMD_ROPE=1).  Replaces upstream's scalar `rotate_pairs`.
            // ------------------------------------------------------------
            const rope_avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
                "-o",
            });
            const rope_avx2_obj = rope_avx2_asm.addOutputFileArg("rope_neox_f32_avx2.o");
            rope_avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/unary/rope_neox_f32_avx2.asm"));
            ggml_lib.addObjectFile(rope_avx2_obj);

            if (!ctx.options.no_avx512) {
                const rope_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const rope_avx512_obj = rope_avx512_asm.addOutputFileArg("rope_neox_f32_avx512.o");
                rope_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/unary/rope_neox_f32_avx512.asm"));
                ggml_lib.addObjectFile(rope_avx512_obj);
            }

            // layer_norm_f32 (Phase 3 kernel expansion)
            const ln_avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
                "-o",
            });
            const ln_avx2_obj = ln_avx2_asm.addOutputFileArg("layer_norm_f32_avx2.o");
            ln_avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/unary/layer_norm_f32_avx2.asm"));
            ggml_lib.addObjectFile(ln_avx2_obj);

            if (!ctx.options.no_avx512) {
                const ln_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const ln_avx512_obj = ln_avx512_asm.addOutputFileArg("layer_norm_f32_avx512.o");
                ln_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/unary/layer_norm_f32_avx512.asm"));
                ggml_lib.addObjectFile(ln_avx512_obj);
            }

            // quantize_q8_0_f32 (Phase 3 kernel expansion)
            const qz80_avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
                "-o",
            });
            const qz80_avx2_obj = qz80_avx2_asm.addOutputFileArg("quantize_q8_0_f32_avx2.o");
            qz80_avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/unary/quantize_q8_0_f32_avx2.asm"));
            ggml_lib.addObjectFile(qz80_avx2_obj);

            if (!ctx.options.no_avx512) {
                const qz80_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const qz80_avx512_obj = qz80_avx512_asm.addOutputFileArg("quantize_q8_0_f32_avx512.o");
                qz80_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/unary/quantize_q8_0_f32_avx512.asm"));
                ggml_lib.addObjectFile(qz80_avx512_obj);
            }

            // quantize_q8_k_f32 (Phase 3 kernel expansion)
            const qz8k_avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
                "-o",
            });
            const qz8k_avx2_obj = qz8k_avx2_asm.addOutputFileArg("quantize_q8_k_f32_avx2.o");
            qz8k_avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/unary/quantize_q8_k_f32_avx2.asm"));
            ggml_lib.addObjectFile(qz8k_avx2_obj);

            if (!ctx.options.no_avx512) {
                const qz8k_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const qz8k_avx512_obj = qz8k_avx512_asm.addOutputFileArg("quantize_q8_k_f32_avx512.o");
                qz8k_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/unary/quantize_q8_k_f32_avx512.asm"));
                ggml_lib.addObjectFile(qz8k_avx512_obj);
            }

            // rope_standard_f32 (Phase 3 kernel expansion)
            const rstd_avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
                "-o",
            });
            const rstd_avx2_obj = rstd_avx2_asm.addOutputFileArg("rope_standard_f32_avx2.o");
            rstd_avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/unary/rope_standard_f32_avx2.asm"));
            ggml_lib.addObjectFile(rstd_avx2_obj);

            if (!ctx.options.no_avx512) {
                const rstd_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const rstd_avx512_obj = rstd_avx512_asm.addOutputFileArg("rope_standard_f32_avx512.o");
                rstd_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/unary/rope_standard_f32_avx512.asm"));
                ggml_lib.addObjectFile(rstd_avx512_obj);
            }

            // silu_f32 (Phase 3 kernel expansion)
            const silu_avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
                "-o",
            });
            const silu_avx2_obj = silu_avx2_asm.addOutputFileArg("silu_f32_avx2.o");
            silu_avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/unary/silu_f32_avx2.asm"));
            ggml_lib.addObjectFile(silu_avx2_obj);

            if (!ctx.options.no_avx512) {
                const silu_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const silu_avx512_obj = silu_avx512_asm.addOutputFileArg("silu_f32_avx512.o");
                silu_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/unary/silu_f32_avx512.asm"));
                ggml_lib.addObjectFile(silu_avx512_obj);
            }

            // vec_dot_f32_f32 (Phase 3 kernel expansion)
            const vdf_avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
                "-o",
            });
            const vdf_avx2_obj = vdf_avx2_asm.addOutputFileArg("vec_dot_f32_f32_avx2.o");
            vdf_avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/vec/vec_dot_f32_f32_avx2.asm"));
            ggml_lib.addObjectFile(vdf_avx2_obj);

            if (!ctx.options.no_avx512) {
                const vdf_avx512_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const vdf_avx512_obj = vdf_avx512_asm.addOutputFileArg("vec_dot_f32_f32_avx512.o");
                vdf_avx512_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/vec/vec_dot_f32_f32_avx512.asm"));
                ggml_lib.addObjectFile(vdf_avx512_obj);
            }

            // gemm_s8s8s32 — INT8 GEMM microkernel (Phase 3 kernel expansion)
            const gemm_avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
                "-o",
            });
            const gemm_avx2_obj = gemm_avx2_asm.addOutputFileArg("gemm_s8s8s32_avx2.o");
            gemm_avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/vec/gemm_s8s8s32_avx2.asm"));
            ggml_lib.addObjectFile(gemm_avx2_obj);

            const gemm_avx2_t_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
                "-o",
            });
            const gemm_avx2_t_obj = gemm_avx2_t_asm.addOutputFileArg("gemm_s8s8s32_avx2_tiled.o");
            gemm_avx2_t_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/vec/gemm_s8s8s32_avx2_tiled.asm"));
            ggml_lib.addObjectFile(gemm_avx2_t_obj);

            // rope_row vectorised-sincos kernel (asm side of the asm-vs-intrinsics
            // RoPE comparison; C++ intrinsic path in fused_rope_attn.cpp wins).
            const rope_row_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
                "-o",
            });
            const rope_row_obj = rope_row_asm.addOutputFileArg("rope_row_f32_avx2.o");
            rope_row_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/unary/rope_row_f32_avx2.asm"));
            ggml_lib.addObjectFile(rope_row_obj);

            if (!ctx.options.no_avx512) {
                const gemm_vnni_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const gemm_vnni_obj = gemm_vnni_asm.addOutputFileArg("gemm_s8s8s32_avx512vnni.o");
                gemm_vnni_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/vec/gemm_s8s8s32_avx512vnni.asm"));
                ggml_lib.addObjectFile(gemm_vnni_obj);

                const gemm_vnni_t_asm = ctx.b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const gemm_vnni_t_obj = gemm_vnni_t_asm.addOutputFileArg("gemm_s8s8s32_avx512vnni_tiled.o");
                gemm_vnni_t_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/vec/gemm_s8s8s32_avx512vnni_tiled.asm"));
                ggml_lib.addObjectFile(gemm_vnni_t_obj);
            }

            const q8_k_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
                "-o",
            });
            const q8_k_obj = q8_k_asm.addOutputFileArg("vec_dot_q8_k_q8_k_avx2.o");
            q8_k_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/vec/vec_dot_q8_k_q8_k_avx2.asm"));
            ggml_lib.addObjectFile(q8_k_obj);

            const q3_k_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
                "-o",
            });
            const q3_k_obj = q3_k_asm.addOutputFileArg("vec_dot_q3_k_q8_k_avx2.o");
            q3_k_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/vec/vec_dot_q3_k_q8_k_avx2.asm"));
            ggml_lib.addObjectFile(q3_k_obj);

            // Flash Attention F32 - AVX2
            const fa_avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
                "-o",
            });
            const fa_avx2_obj = fa_avx2_asm.addOutputFileArg("flash_attn_f32_avx2.o");
            fa_avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_f32_avx2.asm"));
            ggml_lib.addObjectFile(fa_avx2_obj);

            // Flash Attention Q4_0 - AVX2
            const fa_q4_0_avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_q4_0_avx2_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
            fa_q4_0_avx2_asm.addArg("-o");
            const fa_q4_0_avx2_obj = fa_q4_0_avx2_asm.addOutputFileArg("flash_attn_q4_0_avx2.o");
            fa_q4_0_avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_q4_0_avx2.asm"));
            ggml_lib.addObjectFile(fa_q4_0_avx2_obj);

            // Flash Attention Q8_0 - AVX2
            const fa_q8_0_avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_q8_0_avx2_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
            fa_q8_0_avx2_asm.addArg("-o");
            const fa_q8_0_avx2_obj = fa_q8_0_avx2_asm.addOutputFileArg("flash_attn_q8_0_avx2.o");
            fa_q8_0_avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_q8_0_avx2.asm"));
            ggml_lib.addObjectFile(fa_q8_0_avx2_obj);

            // Flash Attention F16 - AVX2
            const fa_f16_avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_f16_avx2_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
            fa_f16_avx2_asm.addArg("-o");
            const fa_f16_avx2_obj = fa_f16_avx2_asm.addOutputFileArg("flash_attn_f16_avx2.o");
            fa_f16_avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_f16_avx2.asm"));
            ggml_lib.addObjectFile(fa_f16_avx2_obj);

            // Flash Attention Q4_1 - AVX2
            const fa_q4_1_avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_q4_1_avx2_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
            fa_q4_1_avx2_asm.addArg("-o");
            const fa_q4_1_avx2_obj = fa_q4_1_avx2_asm.addOutputFileArg("flash_attn_q4_1_avx2.o");
            fa_q4_1_avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_q4_1_avx2.asm"));
            ggml_lib.addObjectFile(fa_q4_1_avx2_obj);

            // Flash Attention Q5_0 - AVX2
            const fa_q5_0_avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_q5_0_avx2_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
            fa_q5_0_avx2_asm.addArg("-o");
            const fa_q5_0_avx2_obj = fa_q5_0_avx2_asm.addOutputFileArg("flash_attn_q5_0_avx2.o");
            fa_q5_0_avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_q5_0_avx2.asm"));
            ggml_lib.addObjectFile(fa_q5_0_avx2_obj);

            // Flash Attention Q5_1 - AVX2
            const fa_q5_1_avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_q5_1_avx2_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
            fa_q5_1_avx2_asm.addArg("-o");
            const fa_q5_1_avx2_obj = fa_q5_1_avx2_asm.addOutputFileArg("flash_attn_q5_1_avx2.o");
            fa_q5_1_avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_q5_1_avx2.asm"));
            ggml_lib.addObjectFile(fa_q5_1_avx2_obj);

            // Flash Attention IQ4_NL - AVX2
            const fa_iq4_nl_avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_iq4_nl_avx2_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
            fa_iq4_nl_avx2_asm.addArg("-o");
            const fa_iq4_nl_avx2_obj = fa_iq4_nl_avx2_asm.addOutputFileArg("flash_attn_iq4_nl_avx2.o");
            fa_iq4_nl_avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_iq4_nl_avx2.asm"));
            ggml_lib.addObjectFile(fa_iq4_nl_avx2_obj);

            // Flash Attention Q2_K - AVX2
            const fa_q2_k_avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_q2_k_avx2_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
            fa_q2_k_avx2_asm.addArg("-o");
            const fa_q2_k_avx2_obj = fa_q2_k_avx2_asm.addOutputFileArg("flash_attn_q2_k_avx2.o");
            fa_q2_k_avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_q2_k_avx2.asm"));
            ggml_lib.addObjectFile(fa_q2_k_avx2_obj);

            // Flash Attention Q3_K - AVX2
            const fa_q3_k_avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_q3_k_avx2_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
            fa_q3_k_avx2_asm.addArg("-o");
            const fa_q3_k_avx2_obj = fa_q3_k_avx2_asm.addOutputFileArg("flash_attn_q3_k_avx2.o");
            fa_q3_k_avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_q3_k_avx2.asm"));
            ggml_lib.addObjectFile(fa_q3_k_avx2_obj);

            // Flash Attention Q4_K - AVX2
            const fa_q4_k_avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_q4_k_avx2_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
            fa_q4_k_avx2_asm.addArg("-o");
            const fa_q4_k_avx2_obj = fa_q4_k_avx2_asm.addOutputFileArg("flash_attn_q4_k_avx2.o");
            fa_q4_k_avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_q4_k_avx2.asm"));
            ggml_lib.addObjectFile(fa_q4_k_avx2_obj);

            // Flash Attention Q5_K - AVX2
            const fa_q5_k_avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_q5_k_avx2_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
            fa_q5_k_avx2_asm.addArg("-o");
            const fa_q5_k_avx2_obj = fa_q5_k_avx2_asm.addOutputFileArg("flash_attn_q5_k_avx2.o");
            fa_q5_k_avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_q5_k_avx2.asm"));
            ggml_lib.addObjectFile(fa_q5_k_avx2_obj);

            // Flash Attention Q6_K - AVX2
            const fa_q6_k_avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_q6_k_avx2_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
            fa_q6_k_avx2_asm.addArg("-o");
            const fa_q6_k_avx2_obj = fa_q6_k_avx2_asm.addOutputFileArg("flash_attn_q6_k_avx2.o");
            fa_q6_k_avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_q6_k_avx2.asm"));
            ggml_lib.addObjectFile(fa_q6_k_avx2_obj);

            // Flash Attention Q8_K - AVX2
            const fa_q8_k_avx2_asm = ctx.b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_q8_k_avx2_asm.addPrefixedDirectoryArg("-I", ctx.b.path("src/simd/kernels/x86/flash/"));
            fa_q8_k_avx2_asm.addArg("-o");
            const fa_q8_k_avx2_obj = fa_q8_k_avx2_asm.addOutputFileArg("flash_attn_q8_k_avx2.o");
            fa_q8_k_avx2_asm.addFileArg(ctx.b.path("src/simd/kernels/x86/flash/flash_attn_q8_k_avx2.asm"));
            ggml_lib.addObjectFile(fa_q8_k_avx2_obj);

            std.log.info("SIMD backend enabled for x86_64 with AVX2{s}", .{
                if (ctx.options.no_avx512) "" else "+AVX512",
            });
        } else if (ctx.actual_target.result.cpu.arch == .aarch64) {
            // -----------------------------------------------------------------
            // ARM AArch64 NEON assembly (.S files compiled via built-in clang)
            // -----------------------------------------------------------------
            // Include path for neon_common.h and skeleton includes
            ggml_lib.addIncludePath(ctx.b.path("src/simd/kernels/aarch64"));
            ggml_lib.addIncludePath(ctx.b.path("src/simd/kernels/aarch64/flash"));

            // List of all ARM NEON GAS assembly source files
            const neon_asm_sources = [_][]const u8{
                // Flash Attention kernels (14 total)
                "src/simd/kernels/aarch64/flash/flash_attn_f32_neon.S",
                "src/simd/kernels/aarch64/flash/flash_attn_f16_neon.S",
                "src/simd/kernels/aarch64/flash/flash_attn_q4_0_neon.S",
                "src/simd/kernels/aarch64/flash/flash_attn_q4_1_neon.S",
                "src/simd/kernels/aarch64/flash/flash_attn_q5_0_neon.S",
                "src/simd/kernels/aarch64/flash/flash_attn_q5_1_neon.S",
                "src/simd/kernels/aarch64/flash/flash_attn_q8_0_neon.S",
                "src/simd/kernels/aarch64/flash/flash_attn_iq4_nl_neon.S",
                "src/simd/kernels/aarch64/flash/flash_attn_q2_k_neon.S",
                "src/simd/kernels/aarch64/flash/flash_attn_q3_k_neon.S",
                "src/simd/kernels/aarch64/flash/flash_attn_q4_k_neon.S",
                "src/simd/kernels/aarch64/flash/flash_attn_q5_k_neon.S",
                "src/simd/kernels/aarch64/flash/flash_attn_q6_k_neon.S",
                "src/simd/kernels/aarch64/flash/flash_attn_q8_k_neon.S",
                // Vec dot kernels (7 total)
                "src/simd/kernels/aarch64/vec/vec_dot_q4_0_q8_0_neon.S",
                "src/simd/kernels/aarch64/vec/vec_dot_q8_0_q8_0_neon.S",
                "src/simd/kernels/aarch64/vec/vec_dot_q2_k_q8_k_neon.S",
                "src/simd/kernels/aarch64/vec/vec_dot_q3_k_q8_k_neon.S",
                "src/simd/kernels/aarch64/vec/vec_dot_q4_k_q8_k_neon.S",
                "src/simd/kernels/aarch64/vec/vec_dot_q5_k_q8_k_neon.S",
                "src/simd/kernels/aarch64/vec/vec_dot_q6_k_q8_k_neon.S",
                "src/simd/kernels/aarch64/vec/vec_dot_q8_k_q8_k_neon.S",
                // Unary ops (opt-in env-gated hooks)
                "src/simd/kernels/aarch64/unary/rms_norm_f32_neon.S",
                "src/simd/kernels/aarch64/unary/rope_neox_f32_neon.S",
                "src/simd/kernels/aarch64/unary/layer_norm_f32_neon.S",
                "src/simd/kernels/aarch64/unary/quantize_q8_0_f32_neon.S",
                "src/simd/kernels/aarch64/unary/quantize_q8_k_f32_neon.S",
                "src/simd/kernels/aarch64/unary/rope_standard_f32_neon.S",
                "src/simd/kernels/aarch64/unary/silu_f32_neon.S",
                "src/simd/kernels/aarch64/vec/vec_dot_f32_f32_neon.S",
                // Matrix multiplication kernel
                "src/simd/kernels/aarch64/matrix_mult_neon.S",
            };

            for (neon_asm_sources) |asm_src| {
                ggml_lib.addCSourceFile(.{
                    .file = ctx.b.path(asm_src),
                    .flags = &.{},
                });
            }

            std.log.info("SIMD backend enabled for aarch64 with NEON", .{});
        }
    }
}

// ----------------------------------------------------------------------------
// generateSimdManifestHeader (PLAN-ASSEMBLY-REWRITE Section 2)
//
// Produces the contents of simd_kernels_manifest.h, a generated header
// that supplies canonical extern "C" declarations for every SIMD kernel.
// Including this header in src/simd/ggml_simd_hook.cpp removes the
// previously-inline extern void simd_* boilerplate.
//
// Source of truth: the static array below (mirrored as documentation in
// src/simd/kernels/manifest.txt).  When you add a new kernel, add a row
// here too.
// ----------------------------------------------------------------------------
const SimdManifestEntry = struct {
    symbol: []const u8,
    sig: enum { vec_dot, rms_norm_f32, rope_neox_f32, quantize_row, silu_f32, softmax_f32 },
    needs_avx512: bool = false,
};

const simd_manifest = [_]SimdManifestEntry{
    // vec_dot quantized kernels — x86 AVX2 + AVX-512
    .{ .symbol = "simd_vec_dot_q4_0_q8_0_avx2", .sig = .vec_dot },
    .{ .symbol = "simd_vec_dot_q4_0_q8_0_avx512", .sig = .vec_dot, .needs_avx512 = true },
    .{ .symbol = "simd_vec_dot_q8_0_q8_0_avx2", .sig = .vec_dot },
    .{ .symbol = "simd_vec_dot_q8_0_q8_0_avx512", .sig = .vec_dot, .needs_avx512 = true },
    .{ .symbol = "simd_vec_dot_q2_k_q8_k_avx2", .sig = .vec_dot },
    .{ .symbol = "simd_vec_dot_q2_k_q8_k_avx512", .sig = .vec_dot, .needs_avx512 = true },
    .{ .symbol = "simd_vec_dot_q3_k_q8_k_avx2", .sig = .vec_dot },
    .{ .symbol = "simd_vec_dot_q3_k_q8_k_avx512", .sig = .vec_dot, .needs_avx512 = true },
    .{ .symbol = "simd_vec_dot_q4_k_q8_k_avx2", .sig = .vec_dot },
    .{ .symbol = "simd_vec_dot_q4_k_q8_k_avx512", .sig = .vec_dot, .needs_avx512 = true },
    .{ .symbol = "simd_vec_dot_q5_k_q8_k_avx2", .sig = .vec_dot },
    .{ .symbol = "simd_vec_dot_q5_k_q8_k_avx512", .sig = .vec_dot, .needs_avx512 = true },
    .{ .symbol = "simd_vec_dot_q6_k_q8_k_avx2", .sig = .vec_dot },
    .{ .symbol = "simd_vec_dot_q6_k_q8_k_avx512", .sig = .vec_dot, .needs_avx512 = true },
    .{ .symbol = "simd_vec_dot_q8_k_q8_k_avx2", .sig = .vec_dot },
    .{ .symbol = "simd_vec_dot_q8_k_q8_k_avx512", .sig = .vec_dot, .needs_avx512 = true },
    // unary ops — x86 AVX2 + AVX-512 (opt-in via MLZ_SIMD_RMS_NORM=1)
    .{ .symbol = "simd_rms_norm_f32_avx2", .sig = .rms_norm_f32 },
    .{ .symbol = "simd_rms_norm_f32_avx512", .sig = .rms_norm_f32, .needs_avx512 = true },
    .{ .symbol = "simd_rope_neox_f32_avx2", .sig = .rope_neox_f32 },
    .{ .symbol = "simd_rope_neox_f32_avx512", .sig = .rope_neox_f32, .needs_avx512 = true },
    // NEON .S kernels are arch-gated via #if at the call site; declare
    // them unconditionally so the hook compiles cleanly on aarch64 hosts.
    .{ .symbol = "simd_vec_dot_q4_0_q8_0_neon", .sig = .vec_dot },
    .{ .symbol = "simd_vec_dot_q8_0_q8_0_neon", .sig = .vec_dot },
    .{ .symbol = "simd_vec_dot_q2_k_q8_k_neon", .sig = .vec_dot },
    .{ .symbol = "simd_vec_dot_q3_k_q8_k_neon", .sig = .vec_dot },
    .{ .symbol = "simd_vec_dot_q4_k_q8_k_neon", .sig = .vec_dot },
    .{ .symbol = "simd_vec_dot_q5_k_q8_k_neon", .sig = .vec_dot },
    .{ .symbol = "simd_vec_dot_q6_k_q8_k_neon", .sig = .vec_dot },
    .{ .symbol = "simd_vec_dot_q8_k_q8_k_neon", .sig = .vec_dot },
    // unary ops — aarch64 NEON (opt-in via MLZ_SIMD_RMS_NORM=1 / MLZ_SIMD_ROPE=1)
    .{ .symbol = "simd_rms_norm_f32_neon", .sig = .rms_norm_f32 },
    .{ .symbol = "simd_rope_neox_f32_neon", .sig = .rope_neox_f32 },
    // Quantization kernels — x86
    .{ .symbol = "simd_quantize_q8_0_f32_avx2", .sig = .quantize_row },
    .{ .symbol = "simd_quantize_q8_0_f32_avx512", .sig = .quantize_row, .needs_avx512 = true },
    .{ .symbol = "simd_quantize_q8_k_f32_avx2", .sig = .quantize_row },
    .{ .symbol = "simd_quantize_q8_k_f32_avx512", .sig = .quantize_row, .needs_avx512 = true },
    // SiLU — x86
    .{ .symbol = "simd_silu_f32_avx2", .sig = .silu_f32 },
    .{ .symbol = "simd_silu_f32_avx512", .sig = .silu_f32, .needs_avx512 = true },
    // LayerNorm — x86 (reuses rms_norm_f32 sig)
    .{ .symbol = "simd_layer_norm_f32_avx2", .sig = .rms_norm_f32 },
    .{ .symbol = "simd_layer_norm_f32_avx512", .sig = .rms_norm_f32, .needs_avx512 = true },
    // Standard RoPE — x86 (reuses rope_neox_f32 sig)
    .{ .symbol = "simd_rope_standard_f32_avx2", .sig = .rope_neox_f32 },
    .{ .symbol = "simd_rope_standard_f32_avx512", .sig = .rope_neox_f32, .needs_avx512 = true },
    // F32 vec_dot — x86 (reuses vec_dot sig)
    .{ .symbol = "simd_vec_dot_f32_f32_avx2", .sig = .vec_dot },
    .{ .symbol = "simd_vec_dot_f32_f32_avx512", .sig = .vec_dot, .needs_avx512 = true },
    // NEON — aarch64
    .{ .symbol = "simd_quantize_q8_0_f32_neon", .sig = .quantize_row },
    .{ .symbol = "simd_quantize_q8_k_f32_neon", .sig = .quantize_row },
    .{ .symbol = "simd_silu_f32_neon", .sig = .silu_f32 },
    .{ .symbol = "simd_layer_norm_f32_neon", .sig = .rms_norm_f32 },
    .{ .symbol = "simd_rope_standard_f32_neon", .sig = .rope_neox_f32 },
    .{ .symbol = "simd_vec_dot_f32_f32_neon", .sig = .vec_dot },
};

fn generateSimdManifestHeader(allocator: std.mem.Allocator, no_avx512: bool) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(allocator);

    const header_prologue =
        \\// AUTO-GENERATED by build.zig (generateSimdManifestHeader).
        \\// Do not edit manually.  Source of truth is the `simd_manifest` array in build.zig.
        \\// (src/simd/kernels/manifest.txt is a mirrored documentation file only.)
        \\#pragma once
        \\
        \\#ifdef __cplusplus
        \\extern "C" {
        \\#endif
        \\
        \\
    ;
    try out.appendSlice(allocator, header_prologue);

    for (simd_manifest) |e| {
        if (e.needs_avx512 and no_avx512) continue;
        const line = switch (e.sig) {
            .vec_dot => try std.fmt.allocPrint(allocator, "void {s}(int n, float * r, const void * vx, const void * vy);\n", .{e.symbol}),
            .rms_norm_f32 => try std.fmt.allocPrint(allocator, "void {s}(int n, float eps, const float * x, float * y);\n", .{e.symbol}),
            .rope_neox_f32 => try std.fmt.allocPrint(allocator, "void {s}(long long n_pairs, const float * cache, const float * src, float * dst);\n", .{e.symbol}),
            .quantize_row => try std.fmt.allocPrint(allocator, "void {s}(int n, const float * x, void * y);\n", .{e.symbol}),
            .silu_f32 => try std.fmt.allocPrint(allocator, "void {s}(int n, const float * x, float * y);\n", .{e.symbol}),
            .softmax_f32 => try std.fmt.allocPrint(allocator, "void {s}(const float * x, float * y, int n);\n", .{e.symbol}),
        };
        defer allocator.free(line);
        try out.appendSlice(allocator, line);
    }

    const header_epilogue =
        \\
        \\#ifdef __cplusplus
        \\} // extern "C"
        \\#endif
        \\
    ;
    try out.appendSlice(allocator, header_epilogue);
    return out.toOwnedSlice(allocator);
}
