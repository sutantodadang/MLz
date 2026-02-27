const std = @import("std");

// Although this function looks imperative, it does not perform the build
// directly and instead it mutates the build graph (`b`) that will be then
// executed by an external runner. The functions in `std.Build` implement a DSL
// for defining build steps and express dependencies between them, allowing the
// build runner to parallelize the build automatically (and the cache system to
// know when a step doesn't need to be re-run).
pub fn build(b: *std.Build) void {
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

    // This creates a module, which represents a collection of source files alongside
    // some compilation options, such as optimization mode and linked system libraries.
    // Zig modules are the preferred way of making Zig code available to consumers.
    // addModule defines a module that we intend to make available for importing
    // to our consumers. We must give it a name because a Zig package can expose
    // multiple modules and consumers will need to be able to specify which
    // module they want to access.
    const mod = b.addModule("MLz", .{
        // The root source file is the "entry point" of this module. Users of
        // this module will only be able to access public declarations contained
        // in this file, which means that if you have declarations that you
        // intend to expose to consumers that were defined in other files part
        // of this module, you will have to make sure to re-export them from
        // the root file.
        .root_source_file = b.path("src/root.zig"),
        // Later on we'll use this module as the root module of a test executable
        // which requires us to specify a target.
        .target = actual_target,
    });

    // Here we define an executable. An executable needs to have a root module
    // which needs to expose a `main` function. While we could add a main function
    // to the module defined above, it's sometimes preferable to split business
    // logic and the CLI into two separate modules.
    //
    // If your goal is to create a Zig library for others to use, consider if
    // it might benefit from also exposing a CLI tool. A parser library for a
    // data serialization format could also bundle a CLI syntax checker, for example.
    //
    // If instead your goal is to create an executable, consider if users might
    // be interested in also being able to embed the core functionality of your
    // program in their own executable in order to avoid the overhead involved in
    // subprocessing your CLI tool.
    //
    // If neither case applies to you, feel free to delete the declaration you
    // don't need and to put everything under a single module.
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

    c_flags.append(b.allocator, "-DGGML_USE_CPU") catch @panic("OOM");
    cpp_flags.append(b.allocator, "-DGGML_USE_CPU") catch @panic("OOM");

    if (actual_target.result.os.tag == .linux) {
        c_flags.append(b.allocator, "-D_GNU_SOURCE") catch @panic("OOM");
        cpp_flags.append(b.allocator, "-D_GNU_SOURCE") catch @panic("OOM");
    }

    // On Linux CUDA builds, compileCudaSources returns a LazyPath to
    // libggml-cuda.so that must be installed alongside the executable.
    // Declared here so it's accessible across the cuda setup and install scopes.
    var cuda_so_output: ?std.Build.LazyPath = null;

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

    const ggml_lib = b.addLibrary(.{
        .linkage = .static,
        .name = "ggml",
        .root_module = b.createModule(.{
            .target = actual_target,
            .optimize = ggml_optimize,
        }),
    });

    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml.c"),
        .flags = c_flags.items,
    });
    // ggml-base (C++)
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-opt.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-quants.c"),
        .flags = c_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-alloc.c"),
        .flags = c_flags.items,
    });
    // GGUF container helpers (required by llama.cpp model loader)
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/gguf.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-backend.cpp"),
        .flags = cpp_flags.items,
    });
    // ggml (registry)
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-backend-reg.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-threading.cpp"),
        .flags = cpp_flags.items,
    });

    // CPU backend (linked statically into ggml when GGML_BACKEND_DL is off)
    // When simd-backend is enabled, we use a patched version of ggml-cpu.c
    // that calls our custom SIMD kernels before the default implementation
    const use_simd_backend = b.option(bool, "simd-backend", "Use custom SIMD backend for F32 matrix multiplication (x86_64 and aarch64)") orelse false;

    // The SIMD backend uses a patched version of ggml-cpu.c with hook call inserted.
    // When disabled, use the original from llama.cpp dependency.
    const ggml_cpu_c_source = if (use_simd_backend and (actual_target.result.cpu.arch == .x86_64 or actual_target.result.cpu.arch == .aarch64))
        b.path("src/simd/ggml-cpu-simd.c")
    else
        llama_cpp_dep.path("ggml/src/ggml-cpu/ggml-cpu.c");

    ggml_lib.addCSourceFile(.{
        .file = ggml_cpu_c_source,
        .flags = c_flags.items,
    });

    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/quants.c"),
        .flags = c_flags.items,
    });

    // x86-specific optimized quantization kernels (AVX2/AVX-512)
    if (actual_target.result.cpu.arch == .x86_64) {
        ggml_lib.addCSourceFile(.{
            .file = llama_cpp_dep.path("ggml/src/ggml-cpu/arch/x86/quants.c"),
            .flags = c_flags.items,
        });
    }
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/ggml-cpu.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/repack.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/hbm.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/traits.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/amx/amx.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/amx/mmq.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/binary-ops.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/unary-ops.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/vec.cpp"),
        .flags = cpp_flags.items,
    });
    ggml_lib.addCSourceFile(.{
        .file = llama_cpp_dep.path("ggml/src/ggml-cpu/ops.cpp"),
        .flags = cpp_flags.items,
    });
    // Architecture-specific CPU backend sources
    switch (actual_target.result.cpu.arch) {
        .x86_64 => {
            // x86 feature detection + optimized repack/quants
            ggml_lib.addCSourceFile(.{
                .file = llama_cpp_dep.path("ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp"),
                .flags = cpp_flags.items,
            });
            ggml_lib.addCSourceFile(.{
                .file = llama_cpp_dep.path("ggml/src/ggml-cpu/arch/x86/repack.cpp"),
                .flags = cpp_flags.items,
            });
            ggml_lib.addCSourceFile(.{
                .file = llama_cpp_dep.path("ggml/src/ggml-cpu/arch/x86/quants.c"),
                .flags = c_flags.items,
            });
        },
        .aarch64 => {
            // ARM NEON optimizations
            ggml_lib.addCSourceFile(.{
                .file = llama_cpp_dep.path("ggml/src/ggml-cpu/arch/arm/cpu-feats.cpp"),
                .flags = cpp_flags.items,
            });
            ggml_lib.addCSourceFile(.{
                .file = llama_cpp_dep.path("ggml/src/ggml-cpu/arch/arm/repack.cpp"),
                .flags = cpp_flags.items,
            });
            ggml_lib.addCSourceFile(.{
                .file = llama_cpp_dep.path("ggml/src/ggml-cpu/arch/arm/quants.c"),
                .flags = c_flags.items,
            });
        },
        else => {
            // Fallback: no arch-specific optimizations
        },
    }

    if (use_vulkan) {
        // ── Vulkan Shader Generation ──
        // The ggml-vulkan backend requires SPIR-V shaders embedded as C++ source/headers.
        // We compile the upstream vulkan-shaders-gen tool (host-native) and run it at
        // build time against every .comp shader to produce ggml-vulkan-shaders.{hpp,cpp}.

        // Step 1: Build the shader generator as a native host executable
        const shader_gen_exe = b.addExecutable(.{
            .name = "vulkan-shaders-gen",
            .root_module = b.createModule(.{
                .target = b.graph.host,
                .optimize = .ReleaseFast,
            }),
        });
        shader_gen_exe.addCSourceFile(.{
            .file = llama_cpp_dep.path("ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp"),
            .flags = &.{"-std=c++17"},
        });
        shader_gen_exe.linkLibCpp();
        shader_gen_exe.linkLibC();

        // Step 2: Enumerate all .comp shader source files from the dependency
        const shader_dir_path = llama_cpp_dep.path("ggml/src/ggml-vulkan/vulkan-shaders");
        const shader_dir_abs = shader_dir_path.getPath(b);
        var shader_src_dir = if (std.fs.path.isAbsolute(shader_dir_abs))
            std.fs.openDirAbsolute(shader_dir_abs, .{ .iterate = true }) catch |err| {
                std.debug.panic("failed to open Vulkan shader dir (absolute): {s}: {any}", .{ shader_dir_abs, err });
            }
        else
            std.fs.cwd().openDir(shader_dir_abs, .{ .iterate = true }) catch |err| {
                std.debug.panic("failed to open Vulkan shader dir (relative): {s}: {any}", .{ shader_dir_abs, err });
            };
        defer shader_src_dir.close();

        var comp_files: std.ArrayList([]const u8) = .empty;
        var dir_iter = shader_src_dir.iterate();
        while (dir_iter.next() catch @panic("iterate vulkan shader dir failed")) |entry| {
            if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".comp")) {
                comp_files.append(b.allocator, b.allocator.dupe(u8, entry.name) catch @panic("OOM")) catch @panic("OOM");
            }
        }

        // Step 3: For each .comp file, run the shader generator to compile
        //         GLSL → SPIR-V and embed data into a .cpp translation unit.
        //         The generated .cpp files #include the generated hpp header
        //         (ggml-vulkan-shaders.hpp) which itself #include <cstdint>.
        //         NOTE: Do NOT use "-include cstdint" here — Zig 0.15's
        //         cross-compile C compilation caching chokes on the
        //         -include flag with system headers, producing spurious
        //         CacheCheckFailed errors for every .comp.cpp file.
        var shader_data_cpp_flags: std.ArrayList([]const u8) = .empty;
        shader_data_cpp_flags.append(b.allocator, "-std=c++17") catch @panic("OOM");

        for (comp_files.items) |comp_file| {
            const rel_path = b.fmt("ggml/src/ggml-vulkan/vulkan-shaders/{s}", .{comp_file});
            const run_gen = b.addRunArtifact(shader_gen_exe);
            run_gen.addArg("--glslc");
            run_gen.addArg("glslc");
            run_gen.addArg("--source");
            run_gen.addFileArg(llama_cpp_dep.path(rel_path));
            run_gen.addArg("--output-dir");
            _ = run_gen.addOutputDirectoryArg(b.fmt("vk-spirv-{s}", .{comp_file}));
            // The tool uses basename(target_hpp) to emit #include "..." in
            // each generated .cpp.  Without this, the .cpp starts with
            // #include "" which is a compile error.
            run_gen.addArg("--target-hpp");
            run_gen.addArg("ggml-vulkan-shaders.hpp");
            run_gen.addArg("--target-cpp");
            const gen_cpp = run_gen.addOutputFileArg(b.fmt("{s}.cpp", .{comp_file}));
            ggml_lib.addCSourceFile(.{
                .file = gen_cpp,
                .flags = shader_data_cpp_flags.items,
            });
        }

        // Step 4: Run the shader generator once without --source to produce
        //         the header file with extern declarations for all shaders.
        const run_gen_hpp = b.addRunArtifact(shader_gen_exe);
        run_gen_hpp.addArg("--output-dir");
        _ = run_gen_hpp.addOutputDirectoryArg("vk-spirv-hpp");
        run_gen_hpp.addArg("--target-hpp");
        const generated_hpp = run_gen_hpp.addOutputFileArg("ggml-vulkan-shaders.hpp");

        // Add generated header directory so #include "ggml-vulkan-shaders.hpp" resolves
        ggml_lib.addIncludePath(generated_hpp.dirname());

        // ── Vulkan Backend Source ──
        ggml_lib.addCSourceFile(.{
            .file = llama_cpp_dep.path("ggml/src/ggml-vulkan/ggml-vulkan.cpp"),
            .flags = cpp_flags.items,
        });
        ggml_lib.addIncludePath(llama_cpp_dep.path("ggml/src/ggml-vulkan"));

        // Platform-specific Vulkan SDK handling — include paths and library
        // search paths for compilation.  Do NOT linkSystemLibrary here because
        // ggml_lib is a static archive and LLD will warn about .so members in
        // the .a file.  The final exe links vulkan directly (see below).
        switch (target.result.os.tag) {
            .windows => {
                if (b.graph.env_map.get("VULKAN_SDK")) |sdk_path| {
                    const lib_path = b.pathJoin(&.{ sdk_path, "Lib" });
                    ggml_lib.addLibraryPath(.{ .cwd_relative = lib_path });
                    // Vulkan headers are at <VULKAN_SDK>/Include (not in default search paths)
                    const inc_path = b.pathJoin(&.{ sdk_path, "Include" });
                    ggml_lib.addSystemIncludePath(.{ .cwd_relative = inc_path });
                } else {
                    std.log.warn("VULKAN_SDK environment variable not set. Vulkan build may fail.", .{});
                }
            },
            .linux => {
                // Use VULKAN_SDK if available, otherwise try system paths
                if (b.graph.env_map.get("VULKAN_SDK")) |sdk_path| {
                    const lib_path = b.pathJoin(&.{ sdk_path, "lib" });
                    ggml_lib.addLibraryPath(.{ .cwd_relative = lib_path });
                    const inc_path = b.pathJoin(&.{ sdk_path, "include" });
                    ggml_lib.addSystemIncludePath(.{ .cwd_relative = inc_path });
                } else {
                    // Fallback: add standard multiarch library path for cross-compilation
                    // Zig's cross-compile linker doesn't search /usr/lib/<triple> by default.
                    // Only add the path matching the target arch to avoid FileNotFound warnings.
                    const linux_multiarch_dir: []const u8 = switch (target.result.cpu.arch) {
                        .aarch64 => "/usr/lib/aarch64-linux-gnu",
                        else => "/usr/lib/x86_64-linux-gnu",
                    };
                    ggml_lib.addLibraryPath(.{ .cwd_relative = linux_multiarch_dir });
                    ggml_lib.addSystemIncludePath(.{ .cwd_relative = "/usr/include" });
                }
            },
            .macos => {
                // macOS uses MoltenVK via Vulkan SDK
                if (b.graph.env_map.get("VULKAN_SDK")) |sdk_path| {
                    const lib_path = b.pathJoin(&.{ sdk_path, "lib" });
                    ggml_lib.addLibraryPath(.{ .cwd_relative = lib_path });
                    const inc_path = b.pathJoin(&.{ sdk_path, "include" });
                    ggml_lib.addSystemIncludePath(.{ .cwd_relative = inc_path });
                }
            },
            else => {},
        }
    } else if (use_cuda) {
        // CUDA support - detect paths from environment variables
        const cuda_path = getCudaPath(b);
        if (cuda_path == null) {
            std.log.err("CUDA_PATH environment variable not set. Please set it to your CUDA installation directory.", .{});
            std.log.err("Example: export CUDA_PATH=/usr/local/cuda", .{});
            @panic("CUDA_PATH required for CUDA build");
        }
        const cuda_root = cuda_path.?;

        const cuda_include = b.pathJoin(&.{ cuda_root, "include" });
        const ggml_cuda_path_abs = llama_cpp_dep.path("ggml/src/ggml-cuda").getPath(b);

        ggml_lib.addIncludePath(.{ .cwd_relative = cuda_include });
        ggml_lib.addIncludePath(.{ .cwd_relative = ggml_cuda_path_abs });

        // Platform-specific CUDA library paths
        const cuda_lib_path = switch (target.result.os.tag) {
            .windows => b.pathJoin(&.{ cuda_root, "lib", "x64" }),
            else => b.pathJoin(&.{ cuda_root, "lib64" }),
        };
        ggml_lib.addLibraryPath(.{ .cwd_relative = cuda_lib_path });

        // On Linux, add CUDA stubs path for libcuda.so driver API stub.
        // The CUDA toolkit ships stub libraries for CI/build environments
        // that don't have a physical GPU or NVIDIA driver installed.
        if (target.result.os.tag != .windows) {
            const cuda_stubs_path = b.pathJoin(&.{ cuda_root, "lib64", "stubs" });
            ggml_lib.addLibraryPath(.{ .cwd_relative = cuda_stubs_path });
        }

        // NOTE: Do NOT linkSystemLibrary for CUDA on ggml_lib (static archive).
        // LLD will warn/error about .so members in .a files.  The final exe
        // links CUDA libraries directly (see below).  On Linux, libstdc++ is
        // handled by the CUDA shared library (see compileCudaSources).

        // Compile CUDA sources with nvcc.  On Linux, this returns a LazyPath
        // to libggml-cuda.so which must be installed alongside the executable.
        cuda_so_output = compileCudaSources(b, ggml_lib, llama_cpp_dep, cuda_root, ggml_cuda_path_abs, target.result.os.tag);
    }

    // Metal backend for Apple Silicon GPU acceleration
    if (use_metal) {
        // Only attempt to link frameworks if we are building FOR macOS
        if (actual_target.result.os.tag == .macos or actual_target.result.os.tag == .ios) {
            // Get SDK root from sysroot (populated above)
            if (b.sysroot) |sdk_root| {
                ggml_lib.addFrameworkPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_root, "System", "Library", "Frameworks" }) });
                // Also add system include path
                ggml_lib.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ sdk_root, "usr", "include" }) });
            }

            // Metal backend include path
            ggml_lib.addIncludePath(llama_cpp_dep.path("ggml/src/ggml-metal"));

            // Objective-C flags for .m files
            var objc_flags: std.ArrayList([]const u8) = .empty;
            objc_flags.append(b.allocator, "-D_CRT_SECURE_NO_WARNINGS") catch @panic("OOM");
            objc_flags.append(b.allocator, "-DGGML_VERSION=\"100\"") catch @panic("OOM");
            objc_flags.append(b.allocator, "-DGGML_COMMIT=\"unknown\"") catch @panic("OOM");
            objc_flags.append(b.allocator, "-DGGML_USE_CPU") catch @panic("OOM");
            objc_flags.append(b.allocator, "-DGGML_USE_METAL") catch @panic("OOM");
            objc_flags.append(b.allocator, "-fno-objc-arc") catch @panic("OOM");

            // Metal C++ flags
            var metal_cpp_flags: std.ArrayList([]const u8) = .empty;
            metal_cpp_flags.append(b.allocator, "-std=c++17") catch @panic("OOM");
            metal_cpp_flags.append(b.allocator, "-D_CRT_SECURE_NO_WARNINGS") catch @panic("OOM");
            metal_cpp_flags.append(b.allocator, "-DGGML_VERSION=\"100\"") catch @panic("OOM");
            metal_cpp_flags.append(b.allocator, "-DGGML_COMMIT=\"unknown\"") catch @panic("OOM");
            metal_cpp_flags.append(b.allocator, "-DGGML_USE_CPU") catch @panic("OOM");
            metal_cpp_flags.append(b.allocator, "-DGGML_USE_METAL") catch @panic("OOM");

            // Objective-C sources
            ggml_lib.addCSourceFile(.{
                .file = llama_cpp_dep.path("ggml/src/ggml-metal/ggml-metal-context.m"),
                .flags = objc_flags.items,
            });
            ggml_lib.addCSourceFile(.{
                .file = llama_cpp_dep.path("ggml/src/ggml-metal/ggml-metal-device.m"),
                .flags = objc_flags.items,
            });

            // C++ sources
            ggml_lib.addCSourceFile(.{
                .file = llama_cpp_dep.path("ggml/src/ggml-metal/ggml-metal.cpp"),
                .flags = metal_cpp_flags.items,
            });
            ggml_lib.addCSourceFile(.{
                .file = llama_cpp_dep.path("ggml/src/ggml-metal/ggml-metal-common.cpp"),
                .flags = metal_cpp_flags.items,
            });
            ggml_lib.addCSourceFile(.{
                .file = llama_cpp_dep.path("ggml/src/ggml-metal/ggml-metal-device.cpp"),
                .flags = metal_cpp_flags.items,
            });
            ggml_lib.addCSourceFile(.{
                .file = llama_cpp_dep.path("ggml/src/ggml-metal/ggml-metal-ops.cpp"),
                .flags = metal_cpp_flags.items,
            });

            // Link Apple frameworks required for Metal
            ggml_lib.linkFramework("Metal");
            ggml_lib.linkFramework("Foundation");
            ggml_lib.linkFramework("MetalPerformanceShaders");
            ggml_lib.linkFramework("MetalPerformanceShadersGraph");
        }
    }

    // Custom SIMD backend for high-performance matrix multiplication
    // Uses hand-optimized AVX2/AVX-512 assembly (x86_64) or NEON assembly (aarch64)
    // Note: use_simd_backend is defined earlier when patching ggml-cpu.c
    if (use_simd_backend and (actual_target.result.cpu.arch == .x86_64 or actual_target.result.cpu.arch == .aarch64)) {
        c_flags.append(b.allocator, "-DGGML_USE_SIMD_BACKEND") catch @panic("OOM");
        cpp_flags.append(b.allocator, "-DGGML_USE_SIMD_BACKEND") catch @panic("OOM");

        // SIMD backend C++ sources
        var simd_cpp_flags: std.ArrayList([]const u8) = .empty;
        simd_cpp_flags.append(b.allocator, "-std=c++17") catch @panic("OOM");
        simd_cpp_flags.append(b.allocator, "-D_CRT_SECURE_NO_WARNINGS") catch @panic("OOM");

        // x86_64-specific C++ compiler flags (AVX2/FMA/F16C/AVX-512)
        if (actual_target.result.cpu.arch == .x86_64) {
            simd_cpp_flags.append(b.allocator, "-mavx2") catch @panic("OOM");
            simd_cpp_flags.append(b.allocator, "-mfma") catch @panic("OOM");
            simd_cpp_flags.append(b.allocator, "-mf16c") catch @panic("OOM");
            if (!no_avx512) {
                simd_cpp_flags.append(b.allocator, "-mavx512f") catch @panic("OOM");
            }
        }

        // Add SIMD backend C++ sources
        ggml_lib.addCSourceFile(.{
            .file = b.path("src/simd/simd_matmul.cpp"),
            .flags = simd_cpp_flags.items,
        });
        ggml_lib.addCSourceFile(.{
            .file = b.path("src/simd/ggml_simd_hook.cpp"),
            .flags = simd_cpp_flags.items,
        });
        ggml_lib.addCSourceFile(.{
            .file = b.path("src/simd/flash_attention.cpp"),
            .flags = simd_cpp_flags.items,
        });

        // Add include path for SIMD headers
        ggml_lib.addIncludePath(b.path("src/simd"));

        // Architecture-specific assembly compilation
        if (actual_target.result.cpu.arch == .x86_64) {
            // Compile NASM assembly sources
            // Note: Zig's build system can compile .asm files using system NASM
            const nasm_format = switch (actual_target.result.os.tag) {
                .windows => "win64",
                .macos => "macho64",
                else => "elf64",
            };

            // AVX2 assembly
            const avx2_asm = b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-o",
            });
            const avx2_obj = avx2_asm.addOutputFileArg("matrix_mult_avx2.o");
            avx2_asm.addFileArg(b.path("src/simd/kernels/x86/matrix_mult_avx2.asm"));
            ggml_lib.addObjectFile(avx2_obj);

            // AVX512 assembly (only if not disabled)
            if (!no_avx512) {
                const avx512_asm = b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-o",
                });
                const avx512_obj = avx512_asm.addOutputFileArg("matrix_mult_avx512.o");
                avx512_asm.addFileArg(b.path("src/simd/kernels/x86/matrix_mult_avx512.asm"));
                ggml_lib.addObjectFile(avx512_obj);

                // AVX-512 Quantized Kernels
                const q4_q8_avx512_asm = b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const q4_q8_avx512_obj = q4_q8_avx512_asm.addOutputFileArg("vec_dot_q4_0_q8_0_avx512.o");
                q4_q8_avx512_asm.addFileArg(b.path("src/simd/kernels/x86/vec/vec_dot_q4_0_q8_0_avx512.asm"));
                ggml_lib.addObjectFile(q4_q8_avx512_obj);

                const q8_q8_avx512_asm = b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const q8_q8_avx512_obj = q8_q8_avx512_asm.addOutputFileArg("vec_dot_q8_0_q8_0_avx512.o");
                q8_q8_avx512_asm.addFileArg(b.path("src/simd/kernels/x86/vec/vec_dot_q8_0_q8_0_avx512.asm"));
                ggml_lib.addObjectFile(q8_q8_avx512_obj);

                // AVX-512 K-Quant Kernels
                const q2_q8_avx512_asm = b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const q2_q8_avx512_obj = q2_q8_avx512_asm.addOutputFileArg("vec_dot_q2_k_q8_k_avx512.o");
                q2_q8_avx512_asm.addFileArg(b.path("src/simd/kernels/x86/vec/vec_dot_q2_k_q8_k_avx512.asm"));
                ggml_lib.addObjectFile(q2_q8_avx512_obj);

                const q6_q8_avx512_asm = b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const q6_q8_avx512_obj = q6_q8_avx512_asm.addOutputFileArg("vec_dot_q6_k_q8_k_avx512.o");
                q6_q8_avx512_asm.addFileArg(b.path("src/simd/kernels/x86/vec/vec_dot_q6_k_q8_k_avx512.asm"));
                ggml_lib.addObjectFile(q6_q8_avx512_obj);

                const q4_k_avx512_asm = b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const q4_k_avx512_obj = q4_k_avx512_asm.addOutputFileArg("vec_dot_q4_k_q8_k_avx512.o");
                q4_k_avx512_asm.addFileArg(b.path("src/simd/kernels/x86/vec/vec_dot_q4_k_q8_k_avx512.asm"));
                ggml_lib.addObjectFile(q4_k_avx512_obj);

                const q8_k_avx512_asm = b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const q8_k_avx512_obj = q8_k_avx512_asm.addOutputFileArg("vec_dot_q8_k_q8_k_avx512.o");
                q8_k_avx512_asm.addFileArg(b.path("src/simd/kernels/x86/vec/vec_dot_q8_k_q8_k_avx512.asm"));
                ggml_lib.addObjectFile(q8_k_avx512_obj);

                const q3_k_avx512_asm = b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const q3_k_avx512_obj = q3_k_avx512_asm.addOutputFileArg("vec_dot_q3_k_q8_k_avx512.o");
                q3_k_avx512_asm.addFileArg(b.path("src/simd/kernels/x86/vec/vec_dot_q3_k_q8_k_avx512.asm"));
                ggml_lib.addObjectFile(q3_k_avx512_obj);

                // Flash Attention F32 - AVX-512
                const fa_avx512_asm = b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                    "-o",
                });
                const fa_avx512_obj = fa_avx512_asm.addOutputFileArg("flash_attn_f32_avx512.o");
                fa_avx512_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_f32_avx512.asm"));
                ggml_lib.addObjectFile(fa_avx512_obj);

                // Flash Attention Q4_0 - AVX-512
                const fa_q4_0_avx512_asm = b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_q4_0_avx512_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
                fa_q4_0_avx512_asm.addArg("-o");
                const fa_q4_0_avx512_obj = fa_q4_0_avx512_asm.addOutputFileArg("flash_attn_q4_0_avx512.o");
                fa_q4_0_avx512_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_q4_0_avx512.asm"));
                ggml_lib.addObjectFile(fa_q4_0_avx512_obj);

                // Flash Attention Q8_0 - AVX-512
                const fa_q8_0_avx512_asm = b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_q8_0_avx512_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
                fa_q8_0_avx512_asm.addArg("-o");
                const fa_q8_0_avx512_obj = fa_q8_0_avx512_asm.addOutputFileArg("flash_attn_q8_0_avx512.o");
                fa_q8_0_avx512_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_q8_0_avx512.asm"));
                ggml_lib.addObjectFile(fa_q8_0_avx512_obj);

                // Flash Attention F16 - AVX-512
                const fa_f16_avx512_asm = b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_f16_avx512_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
                fa_f16_avx512_asm.addArg("-o");
                const fa_f16_avx512_obj = fa_f16_avx512_asm.addOutputFileArg("flash_attn_f16_avx512.o");
                fa_f16_avx512_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_f16_avx512.asm"));
                ggml_lib.addObjectFile(fa_f16_avx512_obj);

                // Flash Attention Q4_1 - AVX-512
                const fa_q4_1_avx512_asm = b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_q4_1_avx512_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
                fa_q4_1_avx512_asm.addArg("-o");
                const fa_q4_1_avx512_obj = fa_q4_1_avx512_asm.addOutputFileArg("flash_attn_q4_1_avx512.o");
                fa_q4_1_avx512_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_q4_1_avx512.asm"));
                ggml_lib.addObjectFile(fa_q4_1_avx512_obj);

                // Flash Attention Q5_0 - AVX-512
                const fa_q5_0_avx512_asm = b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_q5_0_avx512_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
                fa_q5_0_avx512_asm.addArg("-o");
                const fa_q5_0_avx512_obj = fa_q5_0_avx512_asm.addOutputFileArg("flash_attn_q5_0_avx512.o");
                fa_q5_0_avx512_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_q5_0_avx512.asm"));
                ggml_lib.addObjectFile(fa_q5_0_avx512_obj);

                // Flash Attention Q5_1 - AVX-512
                const fa_q5_1_avx512_asm = b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_q5_1_avx512_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
                fa_q5_1_avx512_asm.addArg("-o");
                const fa_q5_1_avx512_obj = fa_q5_1_avx512_asm.addOutputFileArg("flash_attn_q5_1_avx512.o");
                fa_q5_1_avx512_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_q5_1_avx512.asm"));
                ggml_lib.addObjectFile(fa_q5_1_avx512_obj);

                // Flash Attention IQ4_NL - AVX-512
                const fa_iq4_nl_avx512_asm = b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_iq4_nl_avx512_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
                fa_iq4_nl_avx512_asm.addArg("-o");
                const fa_iq4_nl_avx512_obj = fa_iq4_nl_avx512_asm.addOutputFileArg("flash_attn_iq4_nl_avx512.o");
                fa_iq4_nl_avx512_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_iq4_nl_avx512.asm"));
                ggml_lib.addObjectFile(fa_iq4_nl_avx512_obj);

                // Flash Attention Q2_K - AVX-512
                const fa_q2_k_avx512_asm = b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_q2_k_avx512_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
                fa_q2_k_avx512_asm.addArg("-o");
                const fa_q2_k_avx512_obj = fa_q2_k_avx512_asm.addOutputFileArg("flash_attn_q2_k_avx512.o");
                fa_q2_k_avx512_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_q2_k_avx512.asm"));
                ggml_lib.addObjectFile(fa_q2_k_avx512_obj);

                // Flash Attention Q3_K - AVX-512
                const fa_q3_k_avx512_asm = b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_q3_k_avx512_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
                fa_q3_k_avx512_asm.addArg("-o");
                const fa_q3_k_avx512_obj = fa_q3_k_avx512_asm.addOutputFileArg("flash_attn_q3_k_avx512.o");
                fa_q3_k_avx512_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_q3_k_avx512.asm"));
                ggml_lib.addObjectFile(fa_q3_k_avx512_obj);

                // Flash Attention Q4_K - AVX-512
                const fa_q4_k_avx512_asm = b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_q4_k_avx512_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
                fa_q4_k_avx512_asm.addArg("-o");
                const fa_q4_k_avx512_obj = fa_q4_k_avx512_asm.addOutputFileArg("flash_attn_q4_k_avx512.o");
                fa_q4_k_avx512_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_q4_k_avx512.asm"));
                ggml_lib.addObjectFile(fa_q4_k_avx512_obj);

                // Flash Attention Q5_K - AVX-512
                const fa_q5_k_avx512_asm = b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_q5_k_avx512_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
                fa_q5_k_avx512_asm.addArg("-o");
                const fa_q5_k_avx512_obj = fa_q5_k_avx512_asm.addOutputFileArg("flash_attn_q5_k_avx512.o");
                fa_q5_k_avx512_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_q5_k_avx512.asm"));
                ggml_lib.addObjectFile(fa_q5_k_avx512_obj);

                // Flash Attention Q6_K - AVX-512
                const fa_q6_k_avx512_asm = b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_q6_k_avx512_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
                fa_q6_k_avx512_asm.addArg("-o");
                const fa_q6_k_avx512_obj = fa_q6_k_avx512_asm.addOutputFileArg("flash_attn_q6_k_avx512.o");
                fa_q6_k_avx512_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_q6_k_avx512.asm"));
                ggml_lib.addObjectFile(fa_q6_k_avx512_obj);

                // Flash Attention Q8_K - AVX-512
                const fa_q8_k_avx512_asm = b.addSystemCommand(&[_][]const u8{
                    "nasm",
                    "-f",
                    nasm_format,
                    "-DWINDOWS",
                    "-DAVX512_ENABLED",
                });
                fa_q8_k_avx512_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
                fa_q8_k_avx512_asm.addArg("-o");
                const fa_q8_k_avx512_obj = fa_q8_k_avx512_asm.addOutputFileArg("flash_attn_q8_k_avx512.o");
                fa_q8_k_avx512_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_q8_k_avx512.asm"));
                ggml_lib.addObjectFile(fa_q8_k_avx512_obj);
            }

            // Quantized dot product kernels (Q4_0, Q8_0) - AVX2
            const q4_q8_asm = b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS", // Define WINDOWS for calling convention
                "-o",
            });
            const q4_q8_obj = q4_q8_asm.addOutputFileArg("vec_dot_q4_0_q8_0_avx2.o");
            q4_q8_asm.addFileArg(b.path("src/simd/kernels/x86/vec/vec_dot_q4_0_q8_0_avx2.asm"));
            ggml_lib.addObjectFile(q4_q8_obj);

            const q8_q8_asm = b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS", // Define WINDOWS for calling convention
                "-o",
            });
            const q8_q8_obj = q8_q8_asm.addOutputFileArg("vec_dot_q8_0_q8_0_avx2.o");
            q8_q8_asm.addFileArg(b.path("src/simd/kernels/x86/vec/vec_dot_q8_0_q8_0_avx2.asm"));
            ggml_lib.addObjectFile(q8_q8_obj);

            // Quantized dot product kernels (Q2_K, Q6_K) - AVX2
            const q2_q8_asm = b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
                "-o",
            });
            const q2_q8_obj = q2_q8_asm.addOutputFileArg("vec_dot_q2_k_q8_k_avx2.o");
            q2_q8_asm.addFileArg(b.path("src/simd/kernels/x86/vec/vec_dot_q2_k_q8_k_avx2.asm"));
            ggml_lib.addObjectFile(q2_q8_obj);

            const q6_q8_asm = b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
                "-o",
            });
            const q6_q8_obj = q6_q8_asm.addOutputFileArg("vec_dot_q6_k_q8_k_avx2.o");
            q6_q8_asm.addFileArg(b.path("src/simd/kernels/x86/vec/vec_dot_q6_k_q8_k_avx2.asm"));
            ggml_lib.addObjectFile(q6_q8_obj);

            // Quantized dot product kernels (Q4_K, Q8_K) - AVX2
            const q4_k_asm = b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
                "-o",
            });
            const q4_k_obj = q4_k_asm.addOutputFileArg("vec_dot_q4_k_q8_k_avx2.o");
            q4_k_asm.addFileArg(b.path("src/simd/kernels/x86/vec/vec_dot_q4_k_q8_k_avx2.asm"));
            ggml_lib.addObjectFile(q4_k_obj);

            const q8_k_asm = b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
                "-o",
            });
            const q8_k_obj = q8_k_asm.addOutputFileArg("vec_dot_q8_k_q8_k_avx2.o");
            q8_k_asm.addFileArg(b.path("src/simd/kernels/x86/vec/vec_dot_q8_k_q8_k_avx2.asm"));
            ggml_lib.addObjectFile(q8_k_obj);

            const q3_k_asm = b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
                "-o",
            });
            const q3_k_obj = q3_k_asm.addOutputFileArg("vec_dot_q3_k_q8_k_avx2.o");
            q3_k_asm.addFileArg(b.path("src/simd/kernels/x86/vec/vec_dot_q3_k_q8_k_avx2.asm"));
            ggml_lib.addObjectFile(q3_k_obj);

            // Flash Attention F32 - AVX2
            const fa_avx2_asm = b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
                "-o",
            });
            const fa_avx2_obj = fa_avx2_asm.addOutputFileArg("flash_attn_f32_avx2.o");
            fa_avx2_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_f32_avx2.asm"));
            ggml_lib.addObjectFile(fa_avx2_obj);

            // Flash Attention Q4_0 - AVX2
            const fa_q4_0_avx2_asm = b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_q4_0_avx2_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
            fa_q4_0_avx2_asm.addArg("-o");
            const fa_q4_0_avx2_obj = fa_q4_0_avx2_asm.addOutputFileArg("flash_attn_q4_0_avx2.o");
            fa_q4_0_avx2_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_q4_0_avx2.asm"));
            ggml_lib.addObjectFile(fa_q4_0_avx2_obj);

            // Flash Attention Q8_0 - AVX2
            const fa_q8_0_avx2_asm = b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_q8_0_avx2_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
            fa_q8_0_avx2_asm.addArg("-o");
            const fa_q8_0_avx2_obj = fa_q8_0_avx2_asm.addOutputFileArg("flash_attn_q8_0_avx2.o");
            fa_q8_0_avx2_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_q8_0_avx2.asm"));
            ggml_lib.addObjectFile(fa_q8_0_avx2_obj);

            // Flash Attention F16 - AVX2
            const fa_f16_avx2_asm = b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_f16_avx2_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
            fa_f16_avx2_asm.addArg("-o");
            const fa_f16_avx2_obj = fa_f16_avx2_asm.addOutputFileArg("flash_attn_f16_avx2.o");
            fa_f16_avx2_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_f16_avx2.asm"));
            ggml_lib.addObjectFile(fa_f16_avx2_obj);

            // Flash Attention Q4_1 - AVX2
            const fa_q4_1_avx2_asm = b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_q4_1_avx2_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
            fa_q4_1_avx2_asm.addArg("-o");
            const fa_q4_1_avx2_obj = fa_q4_1_avx2_asm.addOutputFileArg("flash_attn_q4_1_avx2.o");
            fa_q4_1_avx2_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_q4_1_avx2.asm"));
            ggml_lib.addObjectFile(fa_q4_1_avx2_obj);

            // Flash Attention Q5_0 - AVX2
            const fa_q5_0_avx2_asm = b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_q5_0_avx2_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
            fa_q5_0_avx2_asm.addArg("-o");
            const fa_q5_0_avx2_obj = fa_q5_0_avx2_asm.addOutputFileArg("flash_attn_q5_0_avx2.o");
            fa_q5_0_avx2_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_q5_0_avx2.asm"));
            ggml_lib.addObjectFile(fa_q5_0_avx2_obj);

            // Flash Attention Q5_1 - AVX2
            const fa_q5_1_avx2_asm = b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_q5_1_avx2_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
            fa_q5_1_avx2_asm.addArg("-o");
            const fa_q5_1_avx2_obj = fa_q5_1_avx2_asm.addOutputFileArg("flash_attn_q5_1_avx2.o");
            fa_q5_1_avx2_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_q5_1_avx2.asm"));
            ggml_lib.addObjectFile(fa_q5_1_avx2_obj);

            // Flash Attention IQ4_NL - AVX2
            const fa_iq4_nl_avx2_asm = b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_iq4_nl_avx2_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
            fa_iq4_nl_avx2_asm.addArg("-o");
            const fa_iq4_nl_avx2_obj = fa_iq4_nl_avx2_asm.addOutputFileArg("flash_attn_iq4_nl_avx2.o");
            fa_iq4_nl_avx2_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_iq4_nl_avx2.asm"));
            ggml_lib.addObjectFile(fa_iq4_nl_avx2_obj);

            // Flash Attention Q2_K - AVX2
            const fa_q2_k_avx2_asm = b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_q2_k_avx2_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
            fa_q2_k_avx2_asm.addArg("-o");
            const fa_q2_k_avx2_obj = fa_q2_k_avx2_asm.addOutputFileArg("flash_attn_q2_k_avx2.o");
            fa_q2_k_avx2_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_q2_k_avx2.asm"));
            ggml_lib.addObjectFile(fa_q2_k_avx2_obj);

            // Flash Attention Q3_K - AVX2
            const fa_q3_k_avx2_asm = b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_q3_k_avx2_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
            fa_q3_k_avx2_asm.addArg("-o");
            const fa_q3_k_avx2_obj = fa_q3_k_avx2_asm.addOutputFileArg("flash_attn_q3_k_avx2.o");
            fa_q3_k_avx2_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_q3_k_avx2.asm"));
            ggml_lib.addObjectFile(fa_q3_k_avx2_obj);

            // Flash Attention Q4_K - AVX2
            const fa_q4_k_avx2_asm = b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_q4_k_avx2_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
            fa_q4_k_avx2_asm.addArg("-o");
            const fa_q4_k_avx2_obj = fa_q4_k_avx2_asm.addOutputFileArg("flash_attn_q4_k_avx2.o");
            fa_q4_k_avx2_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_q4_k_avx2.asm"));
            ggml_lib.addObjectFile(fa_q4_k_avx2_obj);

            // Flash Attention Q5_K - AVX2
            const fa_q5_k_avx2_asm = b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_q5_k_avx2_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
            fa_q5_k_avx2_asm.addArg("-o");
            const fa_q5_k_avx2_obj = fa_q5_k_avx2_asm.addOutputFileArg("flash_attn_q5_k_avx2.o");
            fa_q5_k_avx2_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_q5_k_avx2.asm"));
            ggml_lib.addObjectFile(fa_q5_k_avx2_obj);

            // Flash Attention Q6_K - AVX2
            const fa_q6_k_avx2_asm = b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_q6_k_avx2_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
            fa_q6_k_avx2_asm.addArg("-o");
            const fa_q6_k_avx2_obj = fa_q6_k_avx2_asm.addOutputFileArg("flash_attn_q6_k_avx2.o");
            fa_q6_k_avx2_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_q6_k_avx2.asm"));
            ggml_lib.addObjectFile(fa_q6_k_avx2_obj);

            // Flash Attention Q8_K - AVX2
            const fa_q8_k_avx2_asm = b.addSystemCommand(&[_][]const u8{
                "nasm",
                "-f",
                nasm_format,
                "-DWINDOWS",
            });
            fa_q8_k_avx2_asm.addPrefixedDirectoryArg("-I", b.path("src/simd/kernels/x86/flash/"));
            fa_q8_k_avx2_asm.addArg("-o");
            const fa_q8_k_avx2_obj = fa_q8_k_avx2_asm.addOutputFileArg("flash_attn_q8_k_avx2.o");
            fa_q8_k_avx2_asm.addFileArg(b.path("src/simd/kernels/x86/flash/flash_attn_q8_k_avx2.asm"));
            ggml_lib.addObjectFile(fa_q8_k_avx2_obj);

            std.log.info("SIMD backend enabled for x86_64 with AVX2{s}", .{
                if (no_avx512) "" else "+AVX512",
            });
        } else if (actual_target.result.cpu.arch == .aarch64) {
            // -----------------------------------------------------------------
            // ARM AArch64 NEON assembly (.S files compiled via built-in clang)
            // -----------------------------------------------------------------
            // Include path for neon_common.h and skeleton includes
            ggml_lib.addIncludePath(b.path("src/simd/kernels/aarch64"));
            ggml_lib.addIncludePath(b.path("src/simd/kernels/aarch64/flash"));

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
                "src/simd/kernels/aarch64/vec/vec_dot_q6_k_q8_k_neon.S",
                "src/simd/kernels/aarch64/vec/vec_dot_q8_k_q8_k_neon.S",
                // Matrix multiplication kernel
                "src/simd/kernels/aarch64/matrix_mult_neon.S",
            };

            for (neon_asm_sources) |asm_src| {
                ggml_lib.addCSourceFile(.{
                    .file = b.path(asm_src),
                    .flags = &.{},
                });
            }

            std.log.info("SIMD backend enabled for aarch64 with NEON", .{});
        }
    }

    ggml_lib.addIncludePath(llama_cpp_dep.path("ggml/include"));
    ggml_lib.addIncludePath(llama_cpp_dep.path("ggml/src"));
    ggml_lib.addIncludePath(llama_cpp_dep.path("ggml/src/ggml-cpu"));
    ggml_lib.addIncludePath(b.path("src")); // To find ggml_shim.h
    ggml_lib.linkLibC();
    if (actual_target.query.abi != .msvc) {
        ggml_lib.linkLibCpp();
    }

    // Build llama.cpp as a static library so we can use the official llama.h C API.
    // We compile it optimized even in Debug for the same reason as ggml (avoids UB traps).
    const llama_lib = b.addLibrary(.{
        .linkage = .static,
        .name = "llama",
        .root_module = b.createModule(.{
            .target = actual_target,
            .optimize = ggml_optimize,
        }),
    });

    llama_lib.addIncludePath(llama_cpp_dep.path("include"));
    llama_lib.addIncludePath(llama_cpp_dep.path("src"));
    llama_lib.addIncludePath(llama_cpp_dep.path("ggml/include"));
    llama_lib.addIncludePath(llama_cpp_dep.path("ggml/src"));
    llama_lib.addIncludePath(llama_cpp_dep.path("ggml/src/ggml-cpu"));

    if (use_cuda) {
        const cuda_path = getCudaPath(b).?;
        const cuda_lib_path = switch (target.result.os.tag) {
            .windows => b.pathJoin(&.{ cuda_path, "lib", "x64" }),
            else => b.pathJoin(&.{ cuda_path, "lib64" }),
        };
        llama_lib.addLibraryPath(.{ .cwd_relative = cuda_lib_path });
        // Add CUDA stubs path for CI/build environments without a GPU driver.
        // The CUDA toolkit ships stub libraries (libcuda.so) needed at link time.
        if (target.result.os.tag != .windows) {
            const llama_cuda_stubs = b.pathJoin(&.{ cuda_path, "lib64", "stubs" });
            llama_lib.addLibraryPath(.{ .cwd_relative = llama_cuda_stubs });
        }
        // NOTE: Do NOT linkSystemLibrary for CUDA on llama_lib (static archive).
        // Same reasoning as ggml_lib — LLD warns about .so members in .a files.
        // The final exe links all CUDA libraries directly.
    }

    if (use_metal) {
        if (actual_target.result.os.tag == .macos or actual_target.result.os.tag == .ios) {
            if (b.sysroot) |sdk_root| {
                llama_lib.addFrameworkPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_root, "System", "Library", "Frameworks" }) });
                llama_lib.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ sdk_root, "usr", "include" }) });
            }
        }
    }

    // Add all .cpp files from llama.cpp/src recursively (includes model registry).
    // This mirrors the upstream CMakeLists (simpler than keeping a huge list in sync).
    const llama_src_abs = llama_cpp_dep.path("src").getPath(b);
    var llama_dir = if (std.fs.path.isAbsolute(llama_src_abs))
        std.fs.openDirAbsolute(llama_src_abs, .{ .iterate = true }) catch |err| {
            std.debug.panic("failed to open llama.cpp src dir (absolute): {s}: {any}", .{ llama_src_abs, err });
        }
    else
        std.fs.cwd().openDir(llama_src_abs, .{ .iterate = true }) catch |err| {
            std.debug.panic("failed to open llama.cpp src dir (relative): {s}: {any}", .{ llama_src_abs, err });
        };
    defer llama_dir.close();

    var walker = llama_dir.walk(b.allocator) catch @panic("oom walking llama.cpp src");
    defer walker.deinit();

    while (true) {
        const entry_opt = walker.next() catch @panic("walk failed");
        if (entry_opt == null) break;
        const entry = entry_opt.?;
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.path, ".cpp")) continue;

        const rel = b.allocator.alloc(u8, "src/".len + entry.path.len) catch @panic("oom");
        @memcpy(rel[0.."src/".len], "src/");
        @memcpy(rel["src/".len..], entry.path);
        llama_lib.addCSourceFile(.{
            .file = llama_cpp_dep.path(rel),
            .flags = cpp_flags.items,
        });
    }

    llama_lib.linkLibC();
    if (actual_target.query.abi != .msvc) {
        llama_lib.linkLibCpp();
    }
    llama_lib.linkLibrary(ggml_lib);

    // Zig 0.15 does not propagate addLibraryPath from static archives linked
    // via linkLibrary.  Duplicate the platform-specific library paths so that
    // llama can resolve ggml's transitive system library deps (e.g. Vulkan).
    if (use_vulkan) {
        switch (target.result.os.tag) {
            .linux => {
                if (b.graph.env_map.get("VULKAN_SDK")) |sdk_path| {
                    llama_lib.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_path, "lib" }) });
                } else {
                    const llama_multiarch_dir: []const u8 = switch (target.result.cpu.arch) {
                        .aarch64 => "/usr/lib/aarch64-linux-gnu",
                        else => "/usr/lib/x86_64-linux-gnu",
                    };
                    llama_lib.addLibraryPath(.{ .cwd_relative = llama_multiarch_dir });
                }
            },
            .windows => {
                if (b.graph.env_map.get("VULKAN_SDK")) |sdk_path| {
                    llama_lib.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_path, "Lib" }) });
                }
            },
            .macos => {
                if (b.graph.env_map.get("VULKAN_SDK")) |sdk_path| {
                    llama_lib.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_path, "lib" }) });
                }
            },
            else => {},
        }
    }

    const exe = b.addExecutable(.{
        .name = "MLz",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = actual_target,
            .optimize = optimize,
            .imports = &.{
                .{ .name = "MLz", .module = mod },
            },
        }),
    });

    // Link GGML
    exe.linkLibrary(ggml_lib);
    exe.addIncludePath(llama_cpp_dep.path("ggml/include"));
    exe.root_module.addIncludePath(llama_cpp_dep.path("ggml/include"));

    // Link llama.cpp C API
    exe.linkLibrary(llama_lib);
    exe.addIncludePath(llama_cpp_dep.path("include"));
    exe.root_module.addIncludePath(llama_cpp_dep.path("include"));

    exe.linkLibC();

    if (use_cuda) {
        const cuda_path = getCudaPath(b).?;
        const cuda_lib_path = switch (target.result.os.tag) {
            .windows => b.pathJoin(&.{ cuda_path, "lib", "x64" }),
            else => b.pathJoin(&.{ cuda_path, "lib64" }),
        };
        exe.addLibraryPath(.{ .cwd_relative = cuda_lib_path });
        // Add CUDA stubs path for CI/build environments without a GPU driver.
        if (target.result.os.tag != .windows) {
            const exe_cuda_stubs = b.pathJoin(&.{ cuda_path, "lib64", "stubs" });
            exe.addLibraryPath(.{ .cwd_relative = exe_cuda_stubs });
        }

        if (target.result.os.tag == .windows) {
            exe.linkSystemLibrary("cudart_static");

            // MSVC/SDK libs for linking - detect from environment
            if (getMsvcLibPath(b)) |msvc_lib| {
                exe.addLibraryPath(.{ .cwd_relative = msvc_lib });
            }
            if (getWindowsSdkLibPath(b, "ucrt")) |ucrt_lib| {
                exe.addLibraryPath(.{ .cwd_relative = ucrt_lib });
            }
            if (getWindowsSdkLibPath(b, "um")) |um_lib| {
                exe.addLibraryPath(.{ .cwd_relative = um_lib });
            }

            if (actual_target.query.abi == .msvc) {
                exe.linkSystemLibrary("libcpmt");
            }
        } else {
            exe.linkSystemLibrary("cudart");
            // On Linux, CUDA objects live in libggml-cuda.so (built by
            // compileCudaSources with g++ -shared -lstdc++).  Link the
            // .so directly to the exe so lld can resolve CUDA symbols.
            // Static archives (ggml_lib) cannot contain .so files.
            if (cuda_so_output) |cuda_so| {
                exe.addObjectFile(cuda_so);
            }
            // Set RPATH so the binary can find libggml-cuda.so at runtime
            // in the same directory as the executable.
            exe.root_module.addRPathSpecial("$ORIGIN");
        }
        exe.linkSystemLibrary("cublas");
        exe.linkSystemLibrary("cuda");
    }

    // Vulkan: link the system library on the executable only.  Do NOT put
    // linkSystemLibrary on static archives (ggml_lib/llama_lib) because LLD
    // will try to include the .so as an archive member and warn/error.
    // Zig 0.15 also does not propagate addLibraryPath from linkLibrary'd
    // static archives, so we duplicate the search paths here.
    if (use_vulkan) {
        switch (target.result.os.tag) {
            .linux => {
                if (b.graph.env_map.get("VULKAN_SDK")) |sdk_path| {
                    exe.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_path, "lib" }) });
                } else {
                    const exe_multiarch_dir: []const u8 = switch (target.result.cpu.arch) {
                        .aarch64 => "/usr/lib/aarch64-linux-gnu",
                        else => "/usr/lib/x86_64-linux-gnu",
                    };
                    exe.addLibraryPath(.{ .cwd_relative = exe_multiarch_dir });
                }
                exe.linkSystemLibrary("vulkan");
            },
            .windows => {
                if (b.graph.env_map.get("VULKAN_SDK")) |sdk_path| {
                    exe.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_path, "Lib" }) });
                }
                exe.linkSystemLibrary("vulkan-1");
            },
            .macos => {
                if (b.graph.env_map.get("VULKAN_SDK")) |sdk_path| {
                    exe.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_path, "lib" }) });
                }
                exe.linkSystemLibrary("vulkan");
            },
            else => {
                exe.linkSystemLibrary("vulkan");
            },
        }
    }

    // Link Metal frameworks to executable
    if (use_metal) {
        if (actual_target.result.os.tag == .macos or actual_target.result.os.tag == .ios) {
            // Get SDK root from sysroot
            if (b.sysroot) |sdk_root| {
                exe.addFrameworkPath(.{ .cwd_relative = b.pathJoin(&.{ sdk_root, "System", "Library", "Frameworks" }) });
            }

            exe.linkFramework("Metal");
            exe.linkFramework("Foundation");
            exe.linkFramework("MetalPerformanceShaders");
            exe.linkFramework("MetalPerformanceShadersGraph");

            // Install the Metal shader file so it can be found at runtime
            const install_metal = b.addInstallFile(llama_cpp_dep.path("ggml/src/ggml-metal/ggml-metal.metal"), "bin/ggml-metal.metal");
            b.getInstallStep().dependOn(&install_metal.step);

            // Headers required for Metal on-the-fly compilation
            const install_common_h = b.addInstallFile(llama_cpp_dep.path("ggml/src/ggml-common.h"), "bin/ggml-common.h");
            b.getInstallStep().dependOn(&install_common_h.step);

            const install_metal_common_h = b.addInstallFile(llama_cpp_dep.path("ggml/src/ggml-metal/ggml-metal-common.h"), "bin/ggml-metal-common.h");
            b.getInstallStep().dependOn(&install_metal_common_h.step);

            const install_metal_impl_h = b.addInstallFile(llama_cpp_dep.path("ggml/src/ggml-metal/ggml-metal-impl.h"), "bin/ggml-metal-impl.h");
            b.getInstallStep().dependOn(&install_metal_impl_h.step);

            const install_metal_device_h = b.addInstallFile(llama_cpp_dep.path("ggml/src/ggml-metal/ggml-metal-device.h"), "bin/ggml-metal-device.h");
            b.getInstallStep().dependOn(&install_metal_device_h.step);

            const install_metal_ops_h = b.addInstallFile(llama_cpp_dep.path("ggml/src/ggml-metal/ggml-metal-ops.h"), "bin/ggml-metal-ops.h");
            b.getInstallStep().dependOn(&install_metal_ops_h.step);
        }
    }

    b.installArtifact(exe);

    // Install CUDA shared library alongside the executable (Linux only).
    // The .so encapsulates GNU libstdc++ dependencies and is found at
    // runtime via the $ORIGIN RPATH set on the executable.
    if (cuda_so_output) |cuda_so| {
        const install_cuda_so = b.addInstallFile(cuda_so, "bin/libggml-cuda.so");
        b.getInstallStep().dependOn(&install_cuda_so.step);
    }

    const run_step = b.step("run", "Run the app");
    const run_cmd = b.addRunArtifact(exe);

    // Ensure the Metal shader is installed before running
    run_cmd.step.dependOn(b.getInstallStep());

    // Tell llama.cpp where to find the Metal shader file
    if (use_metal) {
        run_cmd.setEnvironmentVariable("GGML_METAL_PATH_RESOURCES", b.getInstallPath(.bin, ""));
    }

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    run_step.dependOn(&run_cmd.step);

    // Benchmark Step
    const bench_exe = b.addExecutable(.{
        .name = "bench_simd",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/bench_simd.zig"),
            .target = actual_target,
            .optimize = optimize,
        }),
    });
    bench_exe.linkLibrary(ggml_lib);

    const bench_run = b.addRunArtifact(bench_exe);
    if (b.args) |args| {
        bench_run.addArgs(args);
    }

    const bench_step = b.step("bench", "Run SIMD benchmarks");
    bench_step.dependOn(&bench_run.step);

    if (use_cuda) {
        if (getCudaPath(b)) |cuda_path| {
            const cuda_bin = switch (target.result.os.tag) {
                .windows => b.pathJoin(&.{ cuda_path, "bin" }),
                else => b.pathJoin(&.{ cuda_path, "bin" }),
            };
            const current_path = std.process.getEnvVarOwned(b.allocator, "PATH") catch "";
            const path_sep = if (target.result.os.tag == .windows) ";" else ":";
            run_cmd.setEnvironmentVariable("PATH", b.fmt("{s}{s}{s}", .{ cuda_bin, path_sep, current_path }));
        }
    }

    const test_step = b.step("test", "Run tests");
    const mod_tests = b.addTest(.{
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/root.zig"),
            .target = actual_target,
            .optimize = optimize,
        }),
    });
    test_step.dependOn(&b.addRunArtifact(mod_tests).step);
}

/// Get CUDA installation path from environment variable.
/// Supports CUDA_PATH (Windows default) and CUDA_HOME (Linux/macOS common).
fn getCudaPath(b: *std.Build) ?[]const u8 {
    // Try CUDA_PATH first (Windows default)
    if (b.graph.env_map.get("CUDA_PATH")) |path| {
        return path;
    }
    // Try CUDA_HOME (common on Linux/macOS)
    if (b.graph.env_map.get("CUDA_HOME")) |path| {
        return path;
    }
    // Try common default locations
    const default_paths = [_][]const u8{
        "/usr/local/cuda",
        "/opt/cuda",
    };
    for (default_paths) |path| {
        if (std.fs.accessAbsolute(path, .{})) |_| {
            return path;
        } else |_| {}
    }
    return null;
}

/// Get MSVC library path from environment or common locations.
fn getMsvcLibPath(b: *std.Build) ?[]const u8 {
    // Try VCToolsInstallDir environment variable
    if (b.graph.env_map.get("VCToolsInstallDir")) |vc_dir| {
        return b.pathJoin(&.{ vc_dir, "lib", "x64" });
    }

    // Try to detect from VSINSTALLDIR
    if (b.graph.env_map.get("VSINSTALLDIR")) |vs_dir| {
        // This is a simplified detection - in practice you'd need to find the version
        const vc_base = b.pathJoin(&.{ vs_dir, "VC", "Tools", "MSVC" });
        // For now, return null if we can't find it precisely
        _ = vc_base;
    }

    return null;
}

/// Get Windows SDK library path.
fn getWindowsSdkLibPath(b: *std.Build, lib_type: []const u8) ?[]const u8 {
    // Try WindowsSdkDir environment variable
    if (b.graph.env_map.get("WindowsSdkDir")) |sdk_dir| {
        if (b.graph.env_map.get("WindowsSDKVersion")) |sdk_ver| {
            return b.pathJoin(&.{ sdk_dir, "Lib", sdk_ver, lib_type, "x64" });
        }
    }
    return null;
}

/// Compile CUDA source files (.cu) with nvcc.
/// Supports both Windows (MSVC host compiler) and Linux (g++) targets.
///
/// On Linux, CUDA objects are linked into a shared library (libggml-cuda.so)
/// with `-lstdc++` to isolate GNU libstdc++ from Zig's LLVM libc++.  The
/// returned `LazyPath` (non-null on Linux) must be installed alongside the
/// executable so the dynamic linker can find it at runtime.
///
/// On Windows, objects are added directly to `ggml_lib` and `null` is returned.
fn compileCudaSources(
    b: *std.Build,
    ggml_lib: *std.Build.Step.Compile,
    llama_cpp_dep: *std.Build.Dependency,
    cuda_root: []const u8,
    ggml_cuda_path_abs: []const u8,
    target_os: std.Target.Os.Tag,
) ?std.Build.LazyPath {
    const nvcc_path = if (target_os == .windows)
        b.pathJoin(&.{ cuda_root, "bin", "nvcc.exe" })
    else
        b.pathJoin(&.{ cuda_root, "bin", "nvcc" });

    // MSVC host-compiler paths (Windows only)
    var msvc_base: []const u8 = "";
    var sdk_include: []const u8 = "";
    var cl_path_win: []const u8 = "";
    var cl_dir_win: []const u8 = "";
    var include_var: []const u8 = "";

    if (target_os == .windows) {
        msvc_base = b.graph.env_map.get("VCToolsInstallDir") orelse
            "C:/Program Files/Microsoft Visual Studio/2022/Community/VC/Tools/MSVC/14.44.35207";
        sdk_include = b.graph.env_map.get("WindowsSdkDir") orelse
            "C:/Program Files (x86)/Windows Kits/10/Include/10.0.26100.0";
        cl_path_win = b.pathJoin(&.{ msvc_base, "bin", "Hostx64", "x64", "cl.exe" });
        cl_dir_win = b.pathJoin(&.{ msvc_base, "bin", "Hostx64", "x64" });
        include_var = b.fmt(
            "{s}/include;{s}/ucrt;{s}/shared;{s}/um",
            .{ msvc_base, sdk_include, sdk_include, sdk_include },
        );
    }

    // Discover and compile all .cu files.
    // Use cwd().openDir() instead of openDirAbsolute() because the
    // dependency-resolved path may be relative (e.g. .zig-cache/p/…).
    // On POSIX, openat(AT_FDCWD, path) handles both absolute and relative paths.
    var cuda_dir = std.fs.cwd().openDir(ggml_cuda_path_abs, .{ .iterate = true }) catch |err| {
        std.debug.panic("failed to open ggml-cuda dir: {s}: {any}", .{ ggml_cuda_path_abs, err });
    };
    defer cuda_dir.close();

    var walker = cuda_dir.walk(b.allocator) catch @panic("oom walking ggml-cuda");
    defer walker.deinit();

    // llama.cpp include paths (computed once)
    const inc_ggml_include = llama_cpp_dep.path("ggml/include").getPath(b);
    const inc_ggml_src = llama_cpp_dep.path("ggml/src").getPath(b);

    // On Linux, collect CUDA objects to link into a shared library with
    // libstdc++.  This isolates GNU libstdc++ symbols from the main binary's
    // LLVM libc++ to avoid duplicate/undefined symbol conflicts.  On Windows,
    // CUDA objects link directly into ggml_lib (MSVC runtime has no
    // such conflict with Zig's C++ runtime).
    var linux_cuda_objs: std.ArrayList(std.Build.LazyPath) = .empty;

    while (true) {
        const entry_opt = walker.next() catch @panic("walk failed");
        if (entry_opt == null) break;
        const entry = entry_opt.?;
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.path, ".cu")) continue;

        const cc = b.addSystemCommand(&.{nvcc_path});

        if (target_os == .windows) {
            // Windows: use MSVC as host compiler
            const current_path = std.process.getEnvVarOwned(b.allocator, "PATH") catch "";
            cc.setEnvironmentVariable("PATH", b.fmt("{s};{s}", .{ cl_dir_win, current_path }));
            cc.setEnvironmentVariable("INCLUDE", include_var);
            cc.addArg("-c");
            cc.addArg("-O3");
            cc.addArg("-std=c++17");
            cc.addArg("--extended-lambda");
            cc.addArg("--use-local-env");
            cc.addArg("-ccbin");
            cc.addArg(cl_path_win);
            cc.addArg("-Xcompiler");
            cc.addArg("/bigobj");
            cc.addArg("-Xcompiler");
            cc.addArg("/std:c++17");
            cc.addArg("-Xcompiler");
            cc.addArg("/w");
        } else {
            // Linux: use g++ as host compiler (nvcc default).
            // The resulting .o files reference GNU libstdc++ symbols.
            // We handle the libc++/libstdc++ conflict by linking these
            // objects into a shared library (see below) rather than
            // adding them directly to the static ggml_lib.
            cc.addArg("-c");
            cc.addArg("-O3");
            cc.addArg("-std=c++17");
            cc.addArg("--extended-lambda");
            cc.addArg("-Xcompiler");
            cc.addArg("-fPIC");
            cc.addArg("-Xcompiler");
            cc.addArg("-w");
            // Disable glibc fortification to avoid __fprintf_chk / __*_chk
            // references that Zig's bundled libc does not expose.
            cc.addArg("-Xcompiler");
            cc.addArg("-U_FORTIFY_SOURCE");
            cc.addArg("-Xcompiler");
            cc.addArg("-D_FORTIFY_SOURCE=0");
        }

        // Common flags
        cc.addArg("-DGGML_USE_CUDA");
        cc.addArg("-D_CRT_SECURE_NO_WARNINGS");
        cc.addArg("-DGGML_VERSION=100");
        cc.addArg("-DGGML_COMMIT=unknown");

        // Include paths
        cc.addArg("-I");
        cc.addArg(b.pathJoin(&.{ cuda_root, "include" }));
        cc.addArg("-I");
        cc.addArg(ggml_cuda_path_abs);
        cc.addArg("-I");
        cc.addArg(inc_ggml_include);
        cc.addArg("-I");
        cc.addArg(inc_ggml_src);

        const source_path = b.fmt("{s}/{s}", .{ ggml_cuda_path_abs, entry.path });
        cc.addArg(source_path);

        cc.addArg("-o");
        const obj_ext = if (target_os == .windows) ".obj" else ".o";
        const obj_name = b.fmt("{s}{s}", .{ entry.path, obj_ext });
        const obj = cc.addOutputFileArg(obj_name);

        if (target_os == .windows) {
            ggml_lib.addObjectFile(obj);
        } else {
            linux_cuda_objs.append(b.allocator, obj) catch @panic("oom");
        }
    }

    // On Linux: link CUDA objects into libggml-cuda.so with g++ to isolate
    // GNU libstdc++ from Zig's LLVM libc++.  The shared library encapsulates
    // all libstdc++ dependencies; the dynamic linker keeps them in a separate
    // linking scope from the main binary's libc++.
    if (target_os != .windows and linux_cuda_objs.items.len > 0) {
        const cuda_lib_path = b.pathJoin(&.{ cuda_root, "lib64" });
        const link_so = b.addSystemCommand(&.{ "g++", "-shared", "-o" });
        const cuda_so = link_so.addOutputFileArg("libggml-cuda.so");
        for (linux_cuda_objs.items) |obj| {
            link_so.addFileArg(obj);
        }
        link_so.addArgs(&.{
            "-L",       cuda_lib_path,
            "-lcudart", "-lcublas",
            "-lstdc++",
        });
        // Return the .so path — the caller adds it directly to the exe
        // (not to ggml_lib, because static archives cannot contain .so files).
        return cuda_so;
    }
    return null;
}
